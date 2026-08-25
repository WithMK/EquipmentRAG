"""Incremental indexing pipeline for normalized local documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from app.chunkers.document_chunker import DocumentChunker
from app.config import AppConfig, ChromaConfig, DocumentConfig, load_config
from app.embedding.embedding_service import EmbeddingError, LocalEmbeddingService
from app.indexer import IndexReport
from app.models.document_models import (
    DocumentChunk,
    DocumentChunkMetadata,
    DocumentSourceFile,
    NormalizedDocument,
)
from app.ocr.tesseract import TesseractOcrProvider
from app.parsers.document_parser import DocumentParseError
from app.parsers.document_parsers import DocumentParserRegistry
from app.parsers.document_scanner import DocumentScanError, DocumentScanner
from app.vectorstore.chroma_store import (
    PersistentChromaStore,
    VectorRecord,
    VectorStoreError,
)


_STATE_VERSION = 1
_METADATA_SCHEMA_VERSION = 1


class DocumentIndexerError(RuntimeError):
    """Raised when incremental document indexing cannot complete safely."""


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class DocumentVectorStoreProvider(Protocol):
    def upsert(self, records: Sequence[VectorRecord]) -> int: ...

    def delete_by_ids(self, ids: Sequence[str]) -> int: ...

    def delete_where(self, where: Mapping[str, object]) -> int: ...


@dataclass(frozen=True)
class DocumentFileState:
    state_hash: str
    file_path: str
    chunk_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "state_hash": self.state_hash,
            "file_path": self.file_path,
            "chunk_ids": list(self.chunk_ids),
        }


@dataclass(frozen=True)
class DocumentIndexState:
    equipment: str
    collection_name: str
    settings_fingerprint: str
    files: Mapping[str, DocumentFileState]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": _STATE_VERSION,
            "equipment": self.equipment,
            "collection_name": self.collection_name,
            "settings_fingerprint": self.settings_fingerprint,
            "files": {
                path: value.to_dict() for path, value in sorted(self.files.items())
            },
        }


class IncrementalDocumentIndexer:
    def __init__(
        self,
        config: AppConfig,
        *,
        scanner: DocumentScanner | None = None,
        parsers: DocumentParserRegistry | None = None,
        chunker: DocumentChunker | None = None,
        embedding: EmbeddingProvider | None = None,
        vector_store: DocumentVectorStoreProvider | None = None,
        state_path: Path | None = None,
    ) -> None:
        document = _require_document_config(config)
        self._config = config
        self._document = document
        self._scanner = scanner or DocumentScanner(document)
        self._parsers = parsers or _build_parser_registry(config)
        self._chunker = chunker or DocumentChunker(
            document.chunk_size,
            document.chunk_overlap,
        )
        self._embedding = embedding or LocalEmbeddingService(config.embedding)
        self._vector_store = vector_store
        self._state_path = state_path or _default_state_path(config, document)

    @property
    def state_path(self) -> Path:
        return self._state_path

    def run(self, *, dry_run: bool = False, full_reindex: bool = False) -> IndexReport:
        try:
            sources = self._scanner.scan()
        except DocumentScanError as exc:
            raise DocumentIndexerError(str(exc)) from exc
        current = {source.relative_path: source for source in sources}
        if len(current) != len(sources):
            raise DocumentIndexerError("Document scanner returned duplicate paths")

        previous, state_exists = self._load_state()
        fingerprint = _settings_fingerprint(self._config, self._document)
        settings_changed = (
            state_exists and previous.settings_fingerprint != fingerprint
        )
        effective_full = full_reindex or settings_changed
        previous_paths = set(previous.files)
        current_paths = set(current)
        deleted = tuple(sorted(previous_paths - current_paths, key=str.casefold))

        if effective_full:
            new: tuple[str, ...] = ()
            changed: tuple[str, ...] = ()
            reindexed = tuple(sorted(current_paths, key=str.casefold))
            skipped: tuple[str, ...] = ()
        else:
            new = tuple(sorted(current_paths - previous_paths, key=str.casefold))
            changed = tuple(
                sorted(
                    (
                        path
                        for path in current_paths & previous_paths
                        if current[path].state_hash
                        != previous.files[path].state_hash
                    ),
                    key=str.casefold,
                )
            )
            reindexed = ()
            skipped = tuple(
                sorted(current_paths - set(new) - set(changed), key=str.casefold)
            )

        to_index = tuple((*new, *changed, *reindexed))
        prepared = self._prepare(current, to_index)
        prepared_count = sum(len(chunks) for _, _, chunks in prepared)
        reason = _reindex_reason(full_reindex, settings_changed, state_exists)
        if dry_run:
            return _report(
                True,
                effective_full,
                reason,
                sources,
                new,
                changed,
                reindexed,
                skipped,
                deleted,
                prepared_count,
                0,
                0,
            )

        records_by_file = self._embed(prepared)
        deleted_count, upserted_count = self._persist(
            prepared,
            records_by_file,
            previous,
            state_exists,
            effective_full,
            deleted,
            changed,
        )
        next_files = {path: previous.files[path] for path in skipped}
        for source, _, _ in prepared:
            records = records_by_file[source.relative_path]
            next_files[source.relative_path] = DocumentFileState(
                state_hash=source.state_hash,
                file_path=str(source.path),
                chunk_ids=tuple(record.id for record in records),
            )
        self._write_state(
            DocumentIndexState(
                equipment=self._config.equipment.name,
                collection_name=self._document.collection_name,
                settings_fingerprint=fingerprint,
                files=next_files,
            )
        )
        return _report(
            False,
            effective_full,
            reason,
            sources,
            new,
            changed,
            reindexed,
            skipped,
            deleted,
            prepared_count,
            upserted_count,
            deleted_count,
        )

    def _prepare(
        self,
        current: Mapping[str, DocumentSourceFile],
        paths: Sequence[str],
    ) -> list[tuple[DocumentSourceFile, NormalizedDocument, list[DocumentChunk]]]:
        prepared = []
        for path in paths:
            source = current[path]
            try:
                document = self._parsers.parse(source)
                document = replace(
                    document,
                    equipment=document.equipment or self._config.equipment.name,
                    project=document.project or source.root_name,
                    document_type=document.document_type
                    or document.file_extension.removeprefix("."),
                )
                chunks = self._chunker.chunk(document)
            except (DocumentParseError, ValueError) as exc:
                raise DocumentIndexerError(f"Unable to prepare document: {path}") from exc
            prepared.append((source, document, chunks))
        return prepared

    def _embed(
        self,
        prepared: Sequence[
            tuple[DocumentSourceFile, NormalizedDocument, list[DocumentChunk]]
        ],
    ) -> dict[str, list[VectorRecord]]:
        flattened = [
            (source, document, index, chunk)
            for source, document, chunks in prepared
            for index, chunk in enumerate(chunks)
        ]
        records = {source.relative_path: [] for source, _, _ in prepared}
        if not flattened:
            return records
        try:
            vectors = self._embedding.embed_documents(
                [chunk.content for _, _, _, chunk in flattened]
            )
        except EmbeddingError as exc:
            raise DocumentIndexerError("Unable to embed document chunks") from exc
        except Exception as exc:
            raise DocumentIndexerError("Document embedding provider failed") from exc
        if len(vectors) != len(flattened):
            raise DocumentIndexerError("Embedding provider returned an invalid row count")

        indexed_at = datetime.now(timezone.utc).isoformat()
        for (source, document, index, chunk), vector in zip(flattened, vectors):
            metadata = DocumentChunkMetadata(
                document_id=document.document_id,
                source_path=document.source_path,
                file_name=document.file_name,
                file_extension=document.file_extension,
                equipment=document.equipment,
                chunk_index=index,
                project=document.project,
                unit=document.unit,
                document_type=document.document_type,
                title=document.title,
                revision=document.revision,
                document_status=document.document_status,
                is_latest=document.is_latest,
                section=chunk.section,
                subsection=chunk.subsection,
                heading_path=" > ".join(chunk.heading_path),
                page=chunk.page,
                slide=chunk.slide,
                sheet=chunk.sheet,
                cell_range=chunk.cell_range,
                created_date=document.created_date,
                modified_date=document.modified_date,
                file_hash=document.file_hash,
                content_hash=chunk.content_hash,
                indexed_at=indexed_at,
            )
            records[source.relative_path].append(
                VectorRecord(
                    _chunk_id(document.equipment, document.document_id, index),
                    chunk.content,
                    vector,
                    metadata,
                )
            )
        return records

    def _persist(
        self,
        prepared: Sequence[
            tuple[DocumentSourceFile, NormalizedDocument, list[DocumentChunk]]
        ],
        records_by_file: Mapping[str, list[VectorRecord]],
        previous: DocumentIndexState,
        state_exists: bool,
        effective_full: bool,
        deleted: Sequence[str],
        changed: Sequence[str],
    ) -> tuple[int, int]:
        if not prepared and not deleted and state_exists and not effective_full:
            return 0, 0
        store = self._get_vector_store()
        deleted_count = 0
        old_ids: list[str] = []
        if effective_full and state_exists:
            old_ids = [
                chunk_id for state in previous.files.values() for chunk_id in state.chunk_ids
            ]
        elif not state_exists:
            deleted_count += store.delete_where(
                {"equipment": self._config.equipment.name}
            )
        else:
            for path in (*deleted, *changed):
                old_ids.extend(previous.files[path].chunk_ids)
        if old_ids:
            deleted_count += store.delete_by_ids(tuple(dict.fromkeys(old_ids)))

        upserted_count = 0
        for source, _, _ in prepared:
            records = records_by_file[source.relative_path]
            if records:
                upserted_count += store.upsert(records)
        return deleted_count, upserted_count

    def _get_vector_store(self) -> DocumentVectorStoreProvider:
        if self._vector_store is None:
            self._vector_store = PersistentChromaStore(
                ChromaConfig(
                    path=self._config.chromadb.path,
                    collection_name=self._document.collection_name,
                ),
                self._embedding.dimension,
                metadata_type=DocumentChunkMetadata,
            )
        return self._vector_store

    def _load_state(self) -> tuple[DocumentIndexState, bool]:
        if not self._state_path.is_file():
            return (
                DocumentIndexState(
                    self._config.equipment.name,
                    self._document.collection_name,
                    "",
                    {},
                ),
                False,
            )
        try:
            state = _parse_state(
                json.loads(self._state_path.read_text(encoding="utf-8"))
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise DocumentIndexerError(
                f"Unable to read document index state: {self._state_path}"
            ) from exc
        if state.equipment != self._config.equipment.name:
            raise DocumentIndexerError("Document state belongs to another equipment")
        if state.collection_name != self._document.collection_name:
            raise DocumentIndexerError("Document state belongs to another collection")
        return state, True

    def _write_state(self, state: DocumentIndexState) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_name(
            f"{self._state_path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, self._state_path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise DocumentIndexerError(
                f"Unable to write document state: {self._state_path}"
            ) from exc


def _require_document_config(config: AppConfig) -> DocumentConfig:
    if config.document is None:
        raise DocumentIndexerError("Document configuration is missing")
    if not config.document.enabled:
        raise DocumentIndexerError("Document indexing is disabled in configuration")
    return config.document


def _parse_state(payload: object) -> DocumentIndexState:
    if not isinstance(payload, Mapping) or payload.get("version") != _STATE_VERSION:
        raise DocumentIndexerError("Document state version is unsupported")
    header = (
        payload.get("equipment"),
        payload.get("collection_name"),
        payload.get("settings_fingerprint"),
    )
    files_payload = payload.get("files")
    if not all(isinstance(value, str) and value for value in header):
        raise DocumentIndexerError("Document state header is invalid")
    if not isinstance(files_payload, Mapping):
        raise DocumentIndexerError("Document state files are invalid")
    files: dict[str, DocumentFileState] = {}
    for path, value in files_payload.items():
        if not isinstance(path, str) or not isinstance(value, Mapping):
            raise DocumentIndexerError("Document state entry is invalid")
        state_hash = value.get("state_hash")
        file_path = value.get("file_path")
        chunk_ids = value.get("chunk_ids")
        if (
            not isinstance(state_hash, str)
            or not state_hash
            or not isinstance(file_path, str)
            or not file_path
            or not isinstance(chunk_ids, list)
            or any(not isinstance(item, str) or not item for item in chunk_ids)
        ):
            raise DocumentIndexerError("Document state metadata is invalid")
        files[path] = DocumentFileState(state_hash, file_path, tuple(chunk_ids))
    return DocumentIndexState(header[0], header[1], header[2], files)


def _default_state_path(config: AppConfig, document: DocumentConfig) -> Path:
    identity = f"document\0{config.equipment.name}\0{document.collection_name}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return config.chromadb.path / f"document-index-state-{digest}.json"


def _build_parser_registry(config: AppConfig) -> DocumentParserRegistry:
    visual = config.visual
    if visual is None or not visual.enabled:
        return DocumentParserRegistry()
    ocr = None
    if visual.tesseract_path is not None and (
        visual.pdf_ocr or visual.pptx_image_ocr
    ):
        ocr = TesseractOcrProvider(
            visual.tesseract_path,
            languages=visual.languages,
            timeout_seconds=visual.timeout_seconds,
        )
    return DocumentParserRegistry(
        ocr=ocr,
        pdf_dpi=visual.pdf_dpi,
        pdf_ocr=visual.pdf_ocr,
        pptx_image_ocr=visual.pptx_image_ocr,
        xlsx_chart_extraction=visual.xlsx_chart_extraction,
    )


def _settings_fingerprint(config: AppConfig, document: DocumentConfig) -> str:
    model_files = []
    for file_name in ("config.json", "modules.json", "model.safetensors"):
        path = config.embedding.model_path / file_name
        try:
            stat = path.stat()
        except OSError:
            continue
        model_files.append((file_name, stat.st_size, stat.st_mtime_ns))
    payload = {
        "state_version": _STATE_VERSION,
        "metadata_schema_version": _METADATA_SCHEMA_VERSION,
        "source_paths": [str(path) for path in document.source_paths],
        "extensions": sorted(document.extensions),
        "exclude_directories": sorted(document.exclude_directories),
        "chunk_size": document.chunk_size,
        "chunk_overlap": document.chunk_overlap,
        "model_path": str(config.embedding.model_path),
        "model_files": model_files,
        "normalize_embeddings": config.embedding.normalize_embeddings,
        "equipment": config.equipment.name,
        "collection_name": document.collection_name,
        "visual": (
            {
                "enabled": config.visual.enabled,
                "tesseract_path": str(config.visual.tesseract_path),
                "languages": config.visual.languages,
                "timeout_seconds": config.visual.timeout_seconds,
                "pdf_dpi": config.visual.pdf_dpi,
                "pdf_ocr": config.visual.pdf_ocr,
                "pptx_image_ocr": config.visual.pptx_image_ocr,
                "xlsx_chart_extraction": config.visual.xlsx_chart_extraction,
            }
            if config.visual is not None
            else None
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _chunk_id(equipment: str, document_id: str, chunk_index: int) -> str:
    identity = f"{equipment.casefold()}\0{document_id.casefold()}\0{chunk_index}"
    return "document-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _reindex_reason(full: bool, changed: bool, exists: bool) -> str:
    if full:
        return "requested_full_reindex"
    if changed:
        return "index_settings_changed"
    return "incremental" if exists else "initial_state"


def _report(
    dry_run: bool,
    full: bool,
    reason: str,
    sources: Sequence[DocumentSourceFile],
    new: tuple[str, ...],
    changed: tuple[str, ...],
    reindexed: tuple[str, ...],
    skipped: tuple[str, ...],
    deleted: tuple[str, ...],
    prepared: int,
    upserted: int,
    deleted_chunks: int,
) -> IndexReport:
    return IndexReport(
        dry_run=dry_run,
        full_reindex=full,
        reindex_reason=reason,
        total_files=len(sources),
        new_files=new,
        changed_files=changed,
        reindexed_files=reindexed,
        skipped_files=skipped,
        deleted_files=deleted,
        prepared_chunks=prepared,
        upserted_chunks=upserted,
        deleted_chunks=deleted_chunks,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Incrementally index local documents")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--source", action="append", help="Document root override")
    parser.add_argument("--chroma-path", help="ChromaDB path override")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.document is None:
        parser.error("Document configuration is missing")
    if args.source:
        config = replace(
            config,
            document=replace(
                config.document,
                enabled=True,
                source_paths=tuple(
                    Path(path).expanduser().resolve(strict=False)
                    for path in args.source
                ),
            ),
        )
    if args.chroma_path:
        config = replace(
            config,
            chromadb=replace(
                config.chromadb,
                path=Path(args.chroma_path).expanduser().resolve(strict=False),
            ),
        )
    try:
        report = IncrementalDocumentIndexer(config).run(
            dry_run=args.dry_run,
            full_reindex=args.full,
        )
    except (DocumentIndexerError, EmbeddingError, VectorStoreError) as exc:
        parser.error(str(exc))
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
