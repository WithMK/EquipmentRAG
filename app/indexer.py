"""Incremental C# source indexing pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

from app.chunkers.csharp_chunker import CSharpChunk, CSharpChunker
from app.config import AppConfig, load_config
from app.embedding.embedding_service import EmbeddingError, LocalEmbeddingService
from app.parsers.csharp_parser import (
    CSharpSourceError,
    CSharpSourceFile,
    CSharpSourceScanner,
)
from app.vectorstore.chroma_store import (
    ChunkMetadata,
    PersistentChromaStore,
    VectorRecord,
    VectorStoreError,
)


_STATE_VERSION = 1


class IndexerError(RuntimeError):
    """Raised when incremental indexing cannot complete safely."""


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorStoreProvider(Protocol):
    def upsert(self, records: Sequence[VectorRecord]) -> int: ...

    def delete_by_file_path(
        self, file_path: str, *, equipment: str | None = None
    ) -> int: ...

    def delete_by_equipment(self, equipment: str) -> int: ...


@dataclass(frozen=True)
class IndexedFileState:
    file_hash: str
    file_path: str
    chunk_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "file_hash": self.file_hash,
            "file_path": self.file_path,
            "chunk_ids": list(self.chunk_ids),
        }


@dataclass(frozen=True)
class IndexState:
    equipment: str
    collection_name: str
    settings_fingerprint: str
    files: Mapping[str, IndexedFileState]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": _STATE_VERSION,
            "equipment": self.equipment,
            "collection_name": self.collection_name,
            "settings_fingerprint": self.settings_fingerprint,
            "files": {
                relative_path: state.to_dict()
                for relative_path, state in sorted(self.files.items())
            },
        }


@dataclass(frozen=True)
class IndexReport:
    dry_run: bool
    full_reindex: bool
    reindex_reason: str
    total_files: int
    new_files: tuple[str, ...]
    changed_files: tuple[str, ...]
    reindexed_files: tuple[str, ...]
    skipped_files: tuple[str, ...]
    deleted_files: tuple[str, ...]
    prepared_chunks: int
    upserted_chunks: int
    deleted_chunks: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in (
            "new_files",
            "changed_files",
            "reindexed_files",
            "skipped_files",
            "deleted_files",
        ):
            payload[key] = list(payload[key])
        return payload


class IncrementalSourceIndexer:
    """Coordinate scanning, chunking, embedding, and persistent vector storage."""

    def __init__(
        self,
        config: AppConfig,
        *,
        scanner: CSharpSourceScanner | None = None,
        chunker: CSharpChunker | None = None,
        embedding: EmbeddingProvider | None = None,
        vector_store: VectorStoreProvider | None = None,
        state_path: Path | None = None,
    ) -> None:
        self._config = config
        self._scanner = scanner or CSharpSourceScanner(config.source)
        self._chunker = chunker or CSharpChunker(
            config.source.chunk_size,
            config.source.chunk_overlap,
        )
        self._embedding = embedding or LocalEmbeddingService(config.embedding)
        self._vector_store = vector_store
        self._state_path = state_path or _default_state_path(config)

    @property
    def state_path(self) -> Path:
        return self._state_path

    def run(self, *, dry_run: bool = False, full_reindex: bool = False) -> IndexReport:
        try:
            sources = self._scanner.scan()
        except CSharpSourceError as exc:
            raise IndexerError(str(exc)) from exc

        current = {source.relative_path: source for source in sources}
        if len(current) != len(sources):
            raise IndexerError("Source scanner returned duplicate relative paths")

        previous, state_exists = self._load_state()
        fingerprint = _settings_fingerprint(self._config)
        settings_changed = (
            state_exists and previous.settings_fingerprint != fingerprint
        )
        effective_full = full_reindex or settings_changed

        previous_paths = set(previous.files)
        current_paths = set(current)
        deleted_files = tuple(sorted(previous_paths - current_paths, key=str.casefold))

        if effective_full:
            new_files: tuple[str, ...] = ()
            changed_files: tuple[str, ...] = ()
            reindexed_files = tuple(sorted(current_paths, key=str.casefold))
            skipped_files: tuple[str, ...] = ()
        else:
            new_files = tuple(
                sorted(current_paths - previous_paths, key=str.casefold)
            )
            changed_files = tuple(
                sorted(
                    (
                        path
                        for path in current_paths & previous_paths
                        if current[path].file_hash != previous.files[path].file_hash
                    ),
                    key=str.casefold,
                )
            )
            reindexed_files = ()
            skipped_files = tuple(
                sorted(
                    current_paths - set(new_files) - set(changed_files),
                    key=str.casefold,
                )
            )

        to_index = tuple((*new_files, *changed_files, *reindexed_files))
        prepared = self._prepare_chunks(current, to_index)
        prepared_chunk_count = sum(len(chunks) for _, chunks in prepared)

        if full_reindex:
            reason = "requested_full_reindex"
        elif settings_changed:
            reason = "index_settings_changed"
        elif not state_exists:
            reason = "initial_state"
        else:
            reason = "incremental"

        if dry_run:
            return IndexReport(
                dry_run=True,
                full_reindex=effective_full,
                reindex_reason=reason,
                total_files=len(sources),
                new_files=new_files,
                changed_files=changed_files,
                reindexed_files=reindexed_files,
                skipped_files=skipped_files,
                deleted_files=deleted_files,
                prepared_chunks=prepared_chunk_count,
                upserted_chunks=0,
                deleted_chunks=0,
            )

        records_by_file = self._embed_prepared(prepared)
        reset_equipment = not state_exists or effective_full
        deleted_chunk_count = 0
        upserted_chunk_count = 0

        if reset_equipment or deleted_files or prepared:
            store = self._get_vector_store()
            if reset_equipment:
                deleted_chunk_count += store.delete_by_equipment(
                    self._config.equipment.name
                )
            else:
                for relative_path in deleted_files:
                    deleted_chunk_count += store.delete_by_file_path(
                        previous.files[relative_path].file_path,
                        equipment=self._config.equipment.name,
                    )
                for source, _ in prepared:
                    old_state = previous.files.get(source.relative_path)
                    file_path = (
                        old_state.file_path if old_state is not None else str(source.path)
                    )
                    deleted_chunk_count += store.delete_by_file_path(
                        file_path,
                        equipment=self._config.equipment.name,
                    )

            for source, _ in prepared:
                records = records_by_file[source.relative_path]
                if records:
                    upserted_chunk_count += store.upsert(records)

        next_files: dict[str, IndexedFileState] = {
            path: previous.files[path] for path in skipped_files
        }
        for source, _ in prepared:
            records = records_by_file[source.relative_path]
            next_files[source.relative_path] = IndexedFileState(
                file_hash=source.file_hash,
                file_path=str(source.path),
                chunk_ids=tuple(record.id for record in records),
            )

        self._write_state(
            IndexState(
                equipment=self._config.equipment.name,
                collection_name=self._config.chromadb.collection_name,
                settings_fingerprint=fingerprint,
                files=next_files,
            )
        )
        return IndexReport(
            dry_run=False,
            full_reindex=effective_full,
            reindex_reason=reason,
            total_files=len(sources),
            new_files=new_files,
            changed_files=changed_files,
            reindexed_files=reindexed_files,
            skipped_files=skipped_files,
            deleted_files=deleted_files,
            prepared_chunks=prepared_chunk_count,
            upserted_chunks=upserted_chunk_count,
            deleted_chunks=deleted_chunk_count,
        )

    def _prepare_chunks(
        self,
        current: Mapping[str, CSharpSourceFile],
        relative_paths: Sequence[str],
    ) -> list[tuple[CSharpSourceFile, list[CSharpChunk]]]:
        prepared: list[tuple[CSharpSourceFile, list[CSharpChunk]]] = []
        for relative_path in relative_paths:
            source = current[relative_path]
            try:
                chunks = self._chunker.chunk(source)
            except Exception as exc:
                raise IndexerError(f"Unable to chunk source: {relative_path}") from exc
            prepared.append((source, chunks))
        return prepared

    def _embed_prepared(
        self, prepared: Sequence[tuple[CSharpSourceFile, list[CSharpChunk]]]
    ) -> dict[str, list[VectorRecord]]:
        flattened = [
            (source, chunk_index, chunk)
            for source, chunks in prepared
            for chunk_index, chunk in enumerate(chunks)
        ]
        if not flattened:
            return {source.relative_path: [] for source, _ in prepared}

        try:
            vectors = self._embedding.embed_documents(
                [chunk.content for _, _, chunk in flattened]
            )
        except EmbeddingError as exc:
            raise IndexerError("Unable to embed C# source chunks") from exc
        except Exception as exc:
            raise IndexerError("Embedding provider failed") from exc
        if len(vectors) != len(flattened):
            raise IndexerError("Embedding provider returned an unexpected row count")

        repository = self._config.source.path.name or "local-source"
        records_by_file: dict[str, list[VectorRecord]] = {
            source.relative_path: [] for source, _ in prepared
        }
        for (source, chunk_index, chunk), vector in zip(flattened, vectors):
            records_by_file[source.relative_path].append(
                VectorRecord(
                    id=_chunk_id(
                        self._config.equipment.name,
                        source.relative_path,
                        chunk_index,
                    ),
                    document=chunk.content,
                    embedding=vector,
                    metadata=ChunkMetadata(
                        equipment=self._config.equipment.name,
                        repository=repository,
                        project=repository,
                        file_name=source.path.name,
                        file_path=str(source.path),
                        relative_path=source.relative_path,
                        class_name=chunk.class_name,
                        method_name=chunk.method_name,
                        chunk_index=chunk_index,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        file_hash=source.file_hash,
                        modified_time=source.modified_time,
                    ),
                )
            )
        return records_by_file

    def _get_vector_store(self) -> VectorStoreProvider:
        if self._vector_store is None:
            self._vector_store = PersistentChromaStore(
                self._config.chromadb,
                self._embedding.dimension,
            )
        return self._vector_store

    def _load_state(self) -> tuple[IndexState, bool]:
        if not self._state_path.is_file():
            return (
                IndexState(
                    equipment=self._config.equipment.name,
                    collection_name=self._config.chromadb.collection_name,
                    settings_fingerprint="",
                    files={},
                ),
                False,
            )
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            state = _parse_state(payload)
        except (OSError, UnicodeError, json.JSONDecodeError, IndexerError) as exc:
            raise IndexerError(f"Unable to read index state: {self._state_path}") from exc
        if state.equipment != self._config.equipment.name:
            raise IndexerError("Index state belongs to a different equipment")
        if state.collection_name != self._config.chromadb.collection_name:
            raise IndexerError("Index state belongs to a different collection")
        return state, True

    def _write_state(self, state: IndexState) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_name(
            f"{self._state_path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, self._state_path)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise IndexerError(f"Unable to write index state: {self._state_path}") from exc


def _parse_state(payload: object) -> IndexState:
    if not isinstance(payload, Mapping) or payload.get("version") != _STATE_VERSION:
        raise IndexerError("Index state version is missing or unsupported")
    equipment = payload.get("equipment")
    collection_name = payload.get("collection_name")
    fingerprint = payload.get("settings_fingerprint")
    files_payload = payload.get("files")
    if not all(
        isinstance(value, str) and value
        for value in (equipment, collection_name, fingerprint)
    ) or not isinstance(files_payload, Mapping):
        raise IndexerError("Index state header is invalid")

    files: dict[str, IndexedFileState] = {}
    for relative_path, value in files_payload.items():
        if not isinstance(relative_path, str) or not relative_path:
            raise IndexerError("Index state contains an invalid relative path")
        if not isinstance(value, Mapping):
            raise IndexerError("Index state contains an invalid file entry")
        file_hash = value.get("file_hash")
        file_path = value.get("file_path")
        chunk_ids = value.get("chunk_ids")
        if (
            not isinstance(file_hash, str)
            or not file_hash
            or not isinstance(file_path, str)
            or not file_path
            or not isinstance(chunk_ids, list)
            or any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in chunk_ids)
        ):
            raise IndexerError("Index state contains invalid file metadata")
        files[relative_path] = IndexedFileState(
            file_hash=file_hash,
            file_path=file_path,
            chunk_ids=tuple(chunk_ids),
        )
    return IndexState(
        equipment=equipment,
        collection_name=collection_name,
        settings_fingerprint=fingerprint,
        files=files,
    )


def _default_state_path(config: AppConfig) -> Path:
    identity = f"{config.equipment.name}\0{config.chromadb.collection_name}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return config.chromadb.path / f"index-state-{digest}.json"


def _settings_fingerprint(config: AppConfig) -> str:
    model_files = []
    for file_name in (
        "config.json",
        "modules.json",
        "pytorch_model.bin",
        "model.safetensors",
    ):
        path = config.embedding.model_path / file_name
        try:
            stat = path.stat()
        except OSError:
            continue
        model_files.append((file_name, stat.st_size, stat.st_mtime_ns))
    payload = {
        "state_version": _STATE_VERSION,
        "metadata_schema_version": 2,
        "source_root": str(config.source.path.resolve(strict=False)),
        "include_extensions": sorted(config.source.include_extensions),
        "exclude_directories": sorted(config.source.exclude_directories),
        "chunk_size": config.source.chunk_size,
        "chunk_overlap": config.source.chunk_overlap,
        "model_path": str(config.embedding.model_path.resolve(strict=False)),
        "model_files": model_files,
        "normalize_embeddings": config.embedding.normalize_embeddings,
        "equipment": config.equipment.name,
        "collection_name": config.chromadb.collection_name,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _chunk_id(equipment: str, relative_path: str, chunk_index: int) -> str:
    identity = f"{equipment.casefold()}\0{relative_path.casefold()}\0{chunk_index}"
    return "csharp-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Incrementally index local C# source")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--source", help="Optional C# source root override")
    parser.add_argument("--chroma-path", help="Optional ChromaDB path override")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--full", action="store_true", help="Force a full reindex")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.source:
        config = replace(
            config,
            source=replace(
                config.source,
                path=Path(args.source).expanduser().resolve(strict=False),
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
        report = IncrementalSourceIndexer(config).run(
            dry_run=args.dry_run,
            full_reindex=args.full,
        )
    except (IndexerError, EmbeddingError, VectorStoreError) as exc:
        parser.error(str(exc))
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
