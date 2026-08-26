"""Persistent SQLite FTS5 lexical index for EquipmentRAG chunks."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from app.models.document_models import DocumentChunkMetadata
from app.vectorstore.chroma_store import ChunkMetadata, MetadataCodec, VectorRecord


class LexicalStoreError(RuntimeError):
    """Raised when the persistent lexical index cannot be used safely."""


@dataclass(frozen=True)
class LexicalSearchResult:
    id: str
    document: str
    score: float
    metadata: ChunkMetadata | DocumentChunkMetadata


_FILTER_COLUMNS = {
    "equipment",
    "source_type",
    "repository",
    "file_path",
    "relative_path",
    "class_name",
    "method_name",
    "project",
    "unit",
    "document_type",
    "revision",
    "document_status",
    "is_latest",
    "document_id",
    "file_extension",
}
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[가-힣]+")
_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]*(?:[-.][A-Za-z0-9_]+)+"
)
_SCHEMA_VERSION = 1


class SQLiteFtsStore:
    """Store chunk text and metadata in a serverless, incremental FTS5 index."""

    def __init__(self, path: Path) -> None:
        self._path = path.expanduser().resolve(strict=False)
        self._connection: sqlite3.Connection | None = None
        self._lock = RLock()

    @property
    def path(self) -> Path:
        return self._path

    def open(self) -> None:
        if self._connection is not None:
            return
        with self._lock:
            if self._connection is not None:
                return
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(
                    self._path,
                    timeout=30,
                    check_same_thread=False,
                )
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        source_type UNINDEXED,
                        equipment UNINDEXED,
                        repository UNINDEXED,
                        file_path UNINDEXED,
                        relative_path UNINDEXED,
                        class_name UNINDEXED,
                        method_name UNINDEXED,
                        project UNINDEXED,
                        unit UNINDEXED,
                        document_type UNINDEXED,
                        revision UNINDEXED,
                        document_status UNINDEXED,
                        is_latest UNINDEXED,
                        document_id UNINDEXED,
                        file_extension UNINDEXED,
                        metadata_json UNINDEXED,
                        content UNINDEXED,
                        search_text,
                        tokenize='unicode61 remove_diacritics 2'
                    )
                    """
                )
                schema_version = int(
                    connection.execute("PRAGMA user_version").fetchone()[0]
                )
                if schema_version not in {0, _SCHEMA_VERSION}:
                    raise LexicalStoreError(
                        "Unsupported SQLite FTS5 lexical schema version: "
                        f"{schema_version}"
                    )
                connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
                connection.commit()
            except (sqlite3.Error, LexicalStoreError) as exc:
                try:
                    connection.close()
                except (UnboundLocalError, sqlite3.Error):
                    pass
                if isinstance(exc, LexicalStoreError):
                    raise
                raise LexicalStoreError(
                    "SQLite FTS5 is unavailable or the lexical index cannot be opened"
                ) from exc
            self._connection = connection

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def count(self) -> int:
        connection = self._get_connection()
        try:
            with self._lock:
                row = connection.execute("SELECT count(*) FROM chunks_fts").fetchone()
        except sqlite3.Error as exc:
            raise LexicalStoreError("Unable to count lexical records") from exc
        return int(row[0]) if row is not None else 0

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise LexicalStoreError("records must be a sequence")
        if not records:
            raise LexicalStoreError("records must not be empty")
        values = [self._record_row(record) for record in records]
        ids = [value[0] for value in values]
        if len(set(ids)) != len(ids):
            raise LexicalStoreError("records must not contain duplicate ids")
        connection = self._get_connection()
        placeholders = ",".join("?" for _ in ids)
        try:
            with self._lock, connection:
                connection.execute(
                    f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",
                    ids,
                )
                connection.executemany(
                    """
                    INSERT INTO chunks_fts(
                        chunk_id, source_type, equipment, repository, file_path,
                        relative_path, class_name, method_name, project, unit,
                        document_type, revision, document_status, is_latest,
                        document_id, file_extension, metadata_json, content,
                        search_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
        except sqlite3.Error as exc:
            raise LexicalStoreError("Unable to upsert lexical records") from exc
        return len(values)

    def delete_by_ids(self, ids: Sequence[str]) -> int:
        values = _validated_ids(ids)
        connection = self._get_connection()
        placeholders = ",".join("?" for _ in values)
        try:
            with self._lock, connection:
                cursor = connection.execute(
                    f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",
                    values,
                )
        except sqlite3.Error as exc:
            raise LexicalStoreError("Unable to delete lexical records") from exc
        return max(0, int(cursor.rowcount))

    def delete_by_equipment(self, equipment: str) -> int:
        return self.delete_where({"equipment": _non_empty(equipment, "equipment")})

    def delete_by_file_path(
        self, file_path: str, *, equipment: str | None = None
    ) -> int:
        path = _non_empty(file_path, "file_path").replace("\\", "/")
        if equipment is not None:
            _non_empty(equipment, "equipment")
        connection = self._get_connection()
        clauses = ["replace(file_path, '\\', '/') = ?"]
        parameters: list[Any] = [path]
        if equipment is not None:
            clauses.append("equipment = ?")
            parameters.append(equipment)
        try:
            with self._lock, connection:
                cursor = connection.execute(
                    "DELETE FROM chunks_fts WHERE " + " AND ".join(clauses),
                    parameters,
                )
        except sqlite3.Error as exc:
            raise LexicalStoreError("Unable to delete lexical file records") from exc
        return max(0, int(cursor.rowcount))

    def delete_where(self, filters: Mapping[str, Any]) -> int:
        clauses, parameters = _filter_sql(filters)
        connection = self._get_connection()
        try:
            with self._lock, connection:
                cursor = connection.execute(
                    "DELETE FROM chunks_fts WHERE " + " AND ".join(clauses),
                    parameters,
                )
        except sqlite3.Error as exc:
            raise LexicalStoreError("Unable to delete filtered lexical records") from exc
        return max(0, int(cursor.rowcount))

    def search(
        self,
        query: str,
        top_k: int,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> list[LexicalSearchResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise LexicalStoreError("top_k must be a positive integer")
        match_query = _match_query(query)
        if not match_query:
            return []
        clauses = ["chunks_fts MATCH ?"]
        parameters: list[Any] = [match_query]
        if filters:
            filter_clauses, filter_values = _filter_sql(filters)
            clauses.extend(filter_clauses)
            parameters.extend(filter_values)
        parameters.append(top_k)
        sql = (
            "SELECT chunk_id, content, metadata_json, bm25(chunks_fts) AS rank_score "
            "FROM chunks_fts WHERE "
            + " AND ".join(clauses)
            + " ORDER BY rank_score LIMIT ?"
        )
        connection = self._get_connection()
        try:
            with self._lock:
                rows = connection.execute(sql, parameters).fetchall()
        except sqlite3.Error as exc:
            raise LexicalStoreError("Unable to search SQLite FTS5 index") from exc
        return [self._search_result(row, rank) for rank, row in enumerate(rows, 1)]

    def _get_connection(self) -> sqlite3.Connection:
        self.open()
        if self._connection is None:  # pragma: no cover
            raise LexicalStoreError("SQLite FTS5 connection is unavailable")
        return self._connection

    @staticmethod
    def _record_row(record: VectorRecord) -> tuple[Any, ...]:
        if not isinstance(record, VectorRecord):
            raise LexicalStoreError("records must contain VectorRecord values")
        record_id = _non_empty(record.id, "record.id")
        content = _non_empty(record.document, "record.document")
        metadata = record.metadata.to_chroma()
        source_type = str(metadata.get("source_type", ""))
        if source_type not in {"code", "document"}:
            raise LexicalStoreError("metadata.source_type is invalid")
        values = {
            key: metadata.get(key, "") for key in _FILTER_COLUMNS
        }
        is_latest = 1 if values.get("is_latest") is True else 0
        searchable_metadata = " ".join(
            str(metadata.get(key, ""))
            for key in (
                "file_name", "relative_path", "class_name", "method_name",
                "title", "section", "subsection", "sheet", "cell_range",
                "document_type", "revision",
            )
            if metadata.get(key)
        )
        search_text = _normalized_search_text(searchable_metadata + " " + content)
        return (
            record_id,
            source_type,
            str(values.get("equipment", "")),
            str(values.get("repository", "")),
            str(values.get("file_path") or metadata.get("source_path", "")),
            str(values.get("relative_path") or metadata.get("source_path", "")),
            str(values.get("class_name", "")),
            str(values.get("method_name", "")),
            str(values.get("project", "")),
            str(values.get("unit", "")),
            str(values.get("document_type", "")),
            str(values.get("revision", "")),
            str(values.get("document_status", "")),
            is_latest,
            str(values.get("document_id", "")),
            str(values.get("file_extension", "")),
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            content,
            search_text,
        )

    @staticmethod
    def _search_result(row: sqlite3.Row, rank: int) -> LexicalSearchResult:
        try:
            metadata_payload = json.loads(row["metadata_json"])
            source_type = metadata_payload.get("source_type")
            metadata: MetadataCodec
            if source_type == "document":
                metadata = DocumentChunkMetadata.from_chroma(metadata_payload)
            else:
                metadata = ChunkMetadata.from_chroma(metadata_payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise LexicalStoreError("Lexical index contains invalid metadata") from exc
        return LexicalSearchResult(
            id=row["chunk_id"],
            document=row["content"],
            score=1.0 / rank,
            metadata=metadata,
        )


def metadata_filters(value: object, *, source_type: str) -> dict[str, Any]:
    """Convert a code/document filter dataclass into scalar SQLite filters."""

    filters = asdict(value) if value is not None else {}
    filters["source_type"] = source_type
    return {key: item for key, item in filters.items() if item is not None}


def _filter_sql(filters: Mapping[str, Any]) -> tuple[list[str], list[Any]]:
    if not isinstance(filters, Mapping) or not filters:
        raise LexicalStoreError("filters must be a non-empty mapping")
    clauses: list[str] = []
    parameters: list[Any] = []
    for key, value in filters.items():
        if key not in _FILTER_COLUMNS:
            raise LexicalStoreError(f"Unsupported lexical filter: {key}")
        if isinstance(value, bool):
            value = int(value)
        elif not isinstance(value, (str, int, float)):
            raise LexicalStoreError(f"Invalid lexical filter value: {key}")
        clauses.append(f"{key} = ?")
        parameters.append(value)
    return clauses, parameters


def _match_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise LexicalStoreError("query must be a non-empty string")
    identifiers = _IDENTIFIER_PATTERN.findall(query)
    remainder = _IDENTIFIER_PATTERN.sub(" ", query)
    tokens = [
        identifier.replace("-", "").replace(".", "").casefold()
        for identifier in identifiers
    ]
    tokens.extend(token.casefold() for token in _TOKEN_PATTERN.findall(remainder))
    tokens = list(dict.fromkeys(token for token in tokens if token))
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)


def _normalized_search_text(value: str) -> str:
    # Preserve the original text and add punctuation-free identifier aliases.
    aliases = [
        token.replace("-", "").replace(".", "").casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*", value)
    ]
    return value + (" " + " ".join(aliases) if aliases else "")


def _validated_ids(ids: Sequence[str]) -> list[str]:
    if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence) or not ids:
        raise LexicalStoreError("ids must be a non-empty sequence")
    values = [_non_empty(value, "id") for value in ids]
    if len(set(values)) != len(values):
        raise LexicalStoreError("ids must not contain duplicates")
    return values


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LexicalStoreError(f"{name} must be a non-empty string")
    return value.strip()
