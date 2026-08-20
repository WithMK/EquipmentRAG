"""Persistent ChromaDB storage for embedded C# source chunks."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from threading import RLock
from typing import Any

from app.config import ChromaConfig, load_config


class VectorStoreError(RuntimeError):
    """Raised when vector-store input is invalid or ChromaDB fails."""


@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata persisted with each C# source chunk."""

    equipment: str
    source_type: str = "code"
    repository: str = ""
    project: str = ""
    file_name: str = ""
    file_path: str = ""
    relative_path: str = ""
    class_name: str = ""
    method_name: str = ""
    chunk_index: int = 0
    start_line: int = 0
    end_line: int = 0
    file_hash: str = ""
    modified_time: str = ""
    language: str = "csharp"

    def __post_init__(self) -> None:
        string_fields = (
            "equipment",
            "source_type",
            "repository",
            "project",
            "file_name",
            "file_path",
            "relative_path",
            "class_name",
            "method_name",
            "file_hash",
            "modified_time",
            "language",
        )
        for field_name in string_fields:
            if not isinstance(getattr(self, field_name), str):
                raise VectorStoreError(f"metadata.{field_name} must be a string")
        if not self.equipment.strip():
            raise VectorStoreError("metadata.equipment must be a non-empty string")
        if self.source_type != "code":
            raise VectorStoreError("metadata.source_type must be 'code'")
        if self.language != "csharp":
            raise VectorStoreError("metadata.language must be 'csharp'")
        if (
            isinstance(self.chunk_index, bool)
            or not isinstance(self.chunk_index, int)
            or self.chunk_index < 0
        ):
            raise VectorStoreError("metadata.chunk_index must be a non-negative integer")
        for field_name in ("start_line", "end_line"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VectorStoreError(
                    f"metadata.{field_name} must be a non-negative integer"
                )
        if self.start_line and self.end_line < self.start_line:
            raise VectorStoreError("metadata.end_line must not precede start_line")

    def to_chroma(self) -> dict[str, str | int]:
        """Return the complete scalar metadata payload accepted by ChromaDB."""

        return asdict(self)

    @classmethod
    def from_chroma(cls, value: Mapping[str, Any] | None) -> "ChunkMetadata":
        if not isinstance(value, Mapping):
            raise VectorStoreError("ChromaDB returned missing or invalid metadata")
        try:
            return cls(
                equipment=value["equipment"],
                source_type=value["source_type"],
                repository=value["repository"],
                project=value["project"],
                file_name=value["file_name"],
                file_path=value["file_path"],
                relative_path=value["relative_path"],
                class_name=value["class_name"],
                method_name=value["method_name"],
                chunk_index=value["chunk_index"],
                start_line=value.get("start_line", 0),
                end_line=value.get("end_line", 0),
                file_hash=value["file_hash"],
                modified_time=value["modified_time"],
                language=value["language"],
            )
        except KeyError as exc:
            raise VectorStoreError(
                f"ChromaDB metadata is missing field: {exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class VectorRecord:
    """A source chunk and its precomputed embedding."""

    id: str
    document: str
    embedding: Sequence[float]
    metadata: ChunkMetadata


@dataclass(frozen=True)
class StoredRecord:
    id: str
    document: str
    metadata: ChunkMetadata


@dataclass(frozen=True)
class SearchResult(StoredRecord):
    distance: float


class PersistentChromaStore:
    """Store and query precomputed embeddings in a local Chroma collection."""

    def __init__(self, config: ChromaConfig, expected_dimension: int) -> None:
        if (
            isinstance(expected_dimension, bool)
            or not isinstance(expected_dimension, int)
            or expected_dimension <= 0
        ):
            raise VectorStoreError("expected_dimension must be a positive integer")
        self._config = config
        self._expected_dimension = expected_dimension
        self._client: Any | None = None
        self._collection: Any | None = None
        self._load_lock = RLock()

    @property
    def is_open(self) -> bool:
        return self._collection is not None

    @property
    def expected_dimension(self) -> int:
        return self._expected_dimension

    def open(self) -> None:
        if self.is_open:
            return

        with self._load_lock:
            if self.is_open:
                return
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError as exc:
                raise VectorStoreError(
                    "ChromaDB is not installed; install requirements.txt first"
                ) from exc

            try:
                self._config.path.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(
                    path=str(self._config.path),
                    settings=Settings(anonymized_telemetry=False),
                )
                collection = client.get_or_create_collection(
                    name=self._config.collection_name,
                    configuration={"hnsw": {"space": "cosine"}},
                    metadata={
                        "application": "EquipmentRAG",
                        "distance_space": "cosine",
                        "embedding_dimension": self._expected_dimension,
                    },
                    embedding_function=None,
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Unable to open ChromaDB collection '{self._config.collection_name}'"
                ) from exc

            collection_metadata = collection.metadata or {}
            stored_dimension = collection_metadata.get("embedding_dimension")
            if stored_dimension != self._expected_dimension:
                # Retain the client so controlled shutdown code can release its
                # Windows file handles even though the collection is rejected.
                self._client = client
                raise VectorStoreError(
                    "Collection embedding dimension does not match the configured model: "
                    f"{stored_dimension!r} != {self._expected_dimension}"
                )

            distance_space = collection_metadata.get("distance_space", "cosine")
            if distance_space != "cosine":
                self._client = client
                raise VectorStoreError(
                    "Collection distance space does not match EquipmentRAG: "
                    f"{distance_space!r} != 'cosine'"
                )

            self._client = client
            self._collection = collection

    def count(self) -> int:
        collection = self._get_collection()
        try:
            return int(collection.count())
        except Exception as exc:
            raise VectorStoreError("Unable to count ChromaDB records") from exc

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        validated = self._validate_records(records)
        collection = self._get_collection()
        try:
            collection.upsert(
                ids=[record.id for record, _ in validated],
                documents=[record.document for record, _ in validated],
                embeddings=[embedding for _, embedding in validated],
                metadatas=[record.metadata.to_chroma() for record, _ in validated],
            )
        except Exception as exc:
            raise VectorStoreError("Unable to upsert ChromaDB records") from exc
        return len(validated)

    def get_by_ids(self, ids: Sequence[str]) -> list[StoredRecord]:
        validated_ids = self._validate_ids(ids)
        collection = self._get_collection()
        try:
            payload = collection.get(
                ids=validated_ids,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise VectorStoreError("Unable to read ChromaDB records") from exc
        return self._parse_stored_records(payload)

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        *,
        where: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise VectorStoreError("top_k must be a positive integer")
        embedding = self._validate_embedding(query_embedding)
        collection = self._get_collection()
        if self.count() == 0:
            return []

        try:
            payload = collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=dict(where) if where is not None else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError("Unable to query ChromaDB records") from exc
        return self._parse_search_results(payload)

    def delete_by_ids(self, ids: Sequence[str]) -> int:
        validated_ids = self._validate_ids(ids)
        collection = self._get_collection()
        try:
            existing = collection.get(ids=validated_ids, include=[])["ids"]
            if existing:
                collection.delete(ids=list(existing))
        except Exception as exc:
            raise VectorStoreError("Unable to delete ChromaDB records") from exc
        return len(existing)

    def delete_by_file_path(
        self, file_path: str, *, equipment: str | None = None
    ) -> int:
        if not isinstance(file_path, str) or not file_path.strip():
            raise VectorStoreError("file_path must be a non-empty string")
        if equipment is not None and (
            not isinstance(equipment, str) or not equipment.strip()
        ):
            raise VectorStoreError("equipment must be a non-empty string")
        collection = self._get_collection()
        where: dict[str, Any]
        if equipment is None:
            where = {"file_path": file_path}
        else:
            where = {
                "$and": [
                    {"file_path": file_path},
                    {"equipment": equipment},
                ]
            }
        try:
            existing = collection.get(where=where, include=[])["ids"]
            if existing:
                collection.delete(where=where)
        except Exception as exc:
            raise VectorStoreError("Unable to delete records for file_path") from exc
        return len(existing)

    def delete_by_equipment(self, equipment: str) -> int:
        if not isinstance(equipment, str) or not equipment.strip():
            raise VectorStoreError("equipment must be a non-empty string")
        collection = self._get_collection()
        where = {"equipment": equipment}
        try:
            existing = collection.get(where=where, include=[])["ids"]
            if existing:
                collection.delete(where=where)
        except Exception as exc:
            raise VectorStoreError("Unable to delete records for equipment") from exc
        return len(existing)

    def _get_collection(self) -> Any:
        self.open()
        if self._collection is None:  # pragma: no cover - defensive guard
            raise VectorStoreError("ChromaDB collection did not initialize")
        return self._collection

    def _validate_records(
        self, records: Sequence[VectorRecord]
    ) -> list[tuple[VectorRecord, list[float]]]:
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise VectorStoreError("records must be a sequence")
        if not records:
            raise VectorStoreError("records must not be empty")

        validated: list[tuple[VectorRecord, list[float]]] = []
        seen_ids: set[str] = set()
        for record in records:
            if not isinstance(record, VectorRecord):
                raise VectorStoreError("records must contain VectorRecord values")
            record_id = self._validate_id(record.id)
            if record_id in seen_ids:
                raise VectorStoreError(f"duplicate record id in batch: {record_id}")
            seen_ids.add(record_id)
            if not isinstance(record.document, str) or not record.document.strip():
                raise VectorStoreError("record.document must be a non-empty string")
            if not isinstance(record.metadata, ChunkMetadata):
                raise VectorStoreError("record.metadata must be ChunkMetadata")
            validated.append((record, self._validate_embedding(record.embedding)))
        return validated

    def _validate_ids(self, ids: Sequence[str]) -> list[str]:
        if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence) or not ids:
            raise VectorStoreError("ids must be a non-empty sequence")
        validated = [self._validate_id(record_id) for record_id in ids]
        if len(set(validated)) != len(validated):
            raise VectorStoreError("ids must not contain duplicates")
        return validated

    @staticmethod
    def _validate_id(record_id: str) -> str:
        if not isinstance(record_id, str) or not record_id.strip():
            raise VectorStoreError("record id must be a non-empty string")
        if record_id != record_id.strip():
            raise VectorStoreError("record id must not contain surrounding whitespace")
        return record_id

    def _validate_embedding(self, values: Sequence[float]) -> list[float]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise VectorStoreError("embedding must be a numeric sequence")
        if len(values) != self._expected_dimension:
            raise VectorStoreError(
                "embedding dimension mismatch: "
                f"{len(values)} != {self._expected_dimension}"
            )

        embedding: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise VectorStoreError("embedding must contain only finite numbers")
            converted = float(value)
            if not math.isfinite(converted):
                raise VectorStoreError("embedding must contain only finite numbers")
            embedding.append(converted)
        return embedding

    @staticmethod
    def _parse_stored_records(payload: Mapping[str, Any]) -> list[StoredRecord]:
        ids = payload.get("ids") or []
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []
        if not (len(ids) == len(documents) == len(metadatas)):
            raise VectorStoreError("ChromaDB returned an invalid get result shape")
        return [
            StoredRecord(
                id=record_id,
                document=document,
                metadata=ChunkMetadata.from_chroma(metadata),
            )
            for record_id, document, metadata in zip(ids, documents, metadatas)
        ]

    @staticmethod
    def _parse_search_results(payload: Mapping[str, Any]) -> list[SearchResult]:
        ids_batches = payload.get("ids") or [[]]
        document_batches = payload.get("documents") or [[]]
        metadata_batches = payload.get("metadatas") or [[]]
        distance_batches = payload.get("distances") or [[]]
        if not all(
            len(batches) == 1
            for batches in (
                ids_batches,
                document_batches,
                metadata_batches,
                distance_batches,
            )
        ):
            raise VectorStoreError("ChromaDB returned an invalid query batch shape")

        ids = ids_batches[0]
        documents = document_batches[0]
        metadatas = metadata_batches[0]
        distances = distance_batches[0]
        if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
            raise VectorStoreError("ChromaDB returned an invalid query result shape")
        return [
            SearchResult(
                id=record_id,
                document=document,
                metadata=ChunkMetadata.from_chroma(metadata),
                distance=float(distance),
            )
            for record_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances
            )
        ]


def _demo_metadata(equipment: str, file_name: str, chunk_index: int) -> ChunkMetadata:
    return ChunkMetadata(
        equipment=equipment,
        repository="synthetic-demo",
        project="SampleEquipment",
        file_name=file_name,
        file_path=f"synthetic/{file_name}",
        relative_path=file_name,
        class_name=file_name.removesuffix(".cs"),
        method_name="Demo",
        chunk_index=chunk_index,
        file_hash=f"demo-{chunk_index}",
        modified_time="1970-01-01T00:00:00Z",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Store synthetic C# chunks and run local ChromaDB search"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--query", default="Z축 원점 복귀 실패")
    parser.add_argument(
        "--seed-demo",
        action="store_true",
        help="Upsert two synthetic C# chunks before searching",
    )
    args = parser.parse_args()

    from app.embedding.embedding_service import LocalEmbeddingService

    config = load_config(args.config)
    embedding = LocalEmbeddingService(config.embedding)
    embedding.load()
    store = PersistentChromaStore(config.chromadb, embedding.dimension)

    if args.seed_demo:
        documents = [
            "public void HomeZAxis() { zAxis.MoveHome(); }",
            "public void ResetPressAlarm() { press.Alarm.Reset(); }",
        ]
        vectors = embedding.embed_documents(documents)
        store.upsert(
            [
                VectorRecord(
                    id=f"synthetic-demo-{index}",
                    document=document,
                    embedding=vector,
                    metadata=_demo_metadata(
                        config.equipment.name, f"Demo{index}.cs", index
                    ),
                )
                for index, (document, vector) in enumerate(zip(documents, vectors))
            ]
        )

    results = store.search(
        embedding.embed_query(args.query),
        config.search.top_k,
        where={"equipment": config.equipment.name},
    )
    print(
        json.dumps(
            {
                "collection": config.chromadb.collection_name,
                "count": store.count(),
                "query": args.query,
                "results": [
                    {
                        "id": result.id,
                        "distance": result.distance,
                        "document": result.document,
                        "metadata": result.metadata.to_chroma(),
                    }
                    for result in results
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
