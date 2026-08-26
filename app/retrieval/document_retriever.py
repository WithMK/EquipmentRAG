"""Independent semantic retrieval service for indexed documents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from app.config import AppConfig, ChromaConfig
from app.embedding.embedding_service import EmbeddingError, LocalEmbeddingService
from app.models.document_models import DocumentChunkMetadata
from app.vectorstore.chroma_store import (
    PersistentChromaStore,
    SearchResult,
    VectorStoreError,
)


class DocumentRetrievalError(RuntimeError):
    """Raised when a document query or vector search fails."""


class EmbeddingQueryProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_query(self, text: str) -> list[float]: ...


class DocumentVectorSearchProvider(Protocol):
    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        *,
        where: Mapping[str, Any] | None = None,
        where_document: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]: ...


@dataclass(frozen=True)
class DocumentSearchFilters:
    project: str | None = None
    equipment: str | None = None
    unit: str | None = None
    document_type: str | None = None
    revision: str | None = None
    document_status: str | None = "active"
    is_latest: bool | None = True
    document_id: str | None = None
    file_extension: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "project",
            "equipment",
            "unit",
            "document_type",
            "revision",
            "document_status",
            "document_id",
            "file_extension",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise DocumentRetrievalError(
                    f"{field_name} must be a non-empty string or None"
                )
        if self.is_latest is not None and not isinstance(self.is_latest, bool):
            raise DocumentRetrievalError("is_latest must be a boolean or None")

    def to_chroma(self) -> dict[str, Any] | None:
        values: list[dict[str, Any]] = []
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if value is not None:
                values.append({field_name: value})
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return {"$and": values}


@dataclass(frozen=True)
class DocumentSearchResult:
    rank: int
    chunk_id: str
    score: float
    distance: float
    text: str
    metadata: DocumentChunkMetadata

    def to_dict(self, *, include_text: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "distance": self.distance,
            "source_path": self.metadata.source_path,
            "metadata": asdict(self.metadata),
        }
        if include_text:
            payload["text"] = self.text
        return payload


class DocumentRetriever:
    """Retrieve document chunks without requiring an LLM or UI runtime."""

    def __init__(
        self,
        config: AppConfig,
        *,
        embedding: EmbeddingQueryProvider | None = None,
        vector_store: DocumentVectorSearchProvider | None = None,
    ) -> None:
        if config.document is None:
            raise DocumentRetrievalError("Document configuration is missing")
        self._config = config
        self._document = config.document
        self._embedding = embedding or LocalEmbeddingService(config.embedding)
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        project: str | None = None,
        equipment: str | None = None,
        unit: str | None = None,
        document_type: str | None = None,
        revision: str | None = None,
        document_status: str | None = "active",
        is_latest: bool | None = True,
    ) -> list[DocumentSearchResult]:
        if revision is not None and document_status == "active" and is_latest is True:
            document_status = None
            is_latest = None
        return self.search(
            query,
            top_k=top_k,
            filters=DocumentSearchFilters(
                project=project,
                equipment=equipment or self._config.equipment.name,
                unit=unit,
                document_type=document_type,
                revision=revision,
                document_status=document_status,
                is_latest=is_latest,
            ),
        )

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: DocumentSearchFilters | None = None,
    ) -> list[DocumentSearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise DocumentRetrievalError("query must be a non-empty string")
        limit = self._config.search.top_k if top_k is None else top_k
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise DocumentRetrievalError("top_k must be a positive integer")
        effective_filters = filters or DocumentSearchFilters(
            equipment=self._config.equipment.name
        )
        try:
            query_embedding = self._embedding.embed_query(query.strip())
            matches = self._get_vector_store().search(
                query_embedding,
                limit,
                where=effective_filters.to_chroma(),
            )
        except (EmbeddingError, VectorStoreError) as exc:
            raise DocumentRetrievalError(str(exc)) from exc
        except Exception as exc:
            raise DocumentRetrievalError("Semantic document retrieval failed") from exc
        return [
            self._to_result(rank, match)
            for rank, match in enumerate(matches, start=1)
        ]

    def search_exact(
        self,
        query: str,
        terms: Sequence[str],
        *,
        top_k: int,
        filters: DocumentSearchFilters | None = None,
    ) -> list[DocumentSearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise DocumentRetrievalError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise DocumentRetrievalError("top_k must be a positive integer")
        if (
            isinstance(terms, (str, bytes))
            or not isinstance(terms, Sequence)
            or not terms
            or any(not isinstance(term, str) or not term.strip() for term in terms)
        ):
            raise DocumentRetrievalError("terms must contain non-empty strings")
        effective_filters = filters or DocumentSearchFilters(
            equipment=self._config.equipment.name
        )
        try:
            query_embedding = self._embedding.embed_query(query.strip())
            matches: dict[str, SearchResult] = {}
            for term in terms:
                for match in self._get_vector_store().search(
                    query_embedding,
                    top_k,
                    where=effective_filters.to_chroma(),
                    where_document={"$contains": term},
                ):
                    existing = matches.get(match.id)
                    if existing is None or match.distance < existing.distance:
                        matches[match.id] = match
        except (EmbeddingError, VectorStoreError) as exc:
            raise DocumentRetrievalError(str(exc)) from exc
        except Exception as exc:
            raise DocumentRetrievalError("Exact document search failed") from exc
        ordered = sorted(matches.values(), key=lambda item: item.distance)[:top_k]
        return [
            self._to_result(rank, match)
            for rank, match in enumerate(ordered, 1)
        ]

    def _get_vector_store(self) -> DocumentVectorSearchProvider:
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

    @staticmethod
    def _to_result(rank: int, match: SearchResult) -> DocumentSearchResult:
        if not isinstance(match.metadata, DocumentChunkMetadata):
            raise DocumentRetrievalError("Vector store returned non-document metadata")
        return DocumentSearchResult(
            rank=rank,
            chunk_id=match.id,
            score=1.0 - match.distance,
            distance=match.distance,
            text=match.document,
            metadata=match.metadata,
        )


def format_document_results(
    results: Sequence[DocumentSearchResult], *, include_text: bool = True
) -> str:
    if not results:
        return "No matching document chunks found."
    blocks: list[str] = []
    for result in results:
        metadata = result.metadata
        lines = [
            f"[{result.rank}]",
            f"Score: {result.score:.6f}",
            f"File: {metadata.file_name}",
            f"Type: {metadata.document_type or 'Unknown'}",
            f"Revision: {metadata.revision or 'Unknown'}",
            f"Section: {metadata.heading_path or metadata.section or 'Unknown'}",
            f"Page: {metadata.page or 'Unknown'}",
            f"Slide: {metadata.slide or 'Unknown'}",
            f"Sheet: {metadata.sheet or 'Unknown'}",
            f"Cells: {metadata.cell_range or 'Unknown'}",
            f"Path: {metadata.source_path}",
        ]
        if include_text:
            lines.extend(("Text:", result.text))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
