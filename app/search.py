"""Semantic C# code search over the local EquipmentRAG vector store."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from app.config import AppConfig, ConfigError, load_config
from app.embedding.embedding_service import EmbeddingError, LocalEmbeddingService
from app.vectorstore.chroma_store import (
    PersistentChromaStore,
    SearchResult,
    VectorStoreError,
)


class SearchError(RuntimeError):
    """Raised when a semantic code search cannot be completed."""


class EmbeddingQueryProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorSearchProvider(Protocol):
    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        *,
        where: Mapping[str, Any] | None = None,
        where_document: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]: ...


@dataclass(frozen=True)
class CodeSearchFilters:
    """Exact-match metadata filters applied by ChromaDB."""

    equipment: str
    repository: str | None = None
    relative_path: str | None = None
    class_name: str | None = None
    method_name: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if field_name == "equipment" or value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise SearchError(
                        f"search filter '{field_name}' must be a non-empty string"
                    )

    def to_chroma(self) -> dict[str, Any]:
        conditions = [
            {field_name: value}
            for field_name, value in asdict(self).items()
            if value is not None
        ]
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


@dataclass(frozen=True)
class CodeSearchResult:
    """User-facing semantic code search result."""

    rank: int
    id: str
    score: float
    distance: float
    file_name: str
    relative_path: str
    file_path: str
    class_name: str
    method_name: str
    start_line: int
    end_line: int
    chunk_index: int
    file_hash: str
    modified_time: str
    code: str

    def to_dict(self, *, include_code: bool = True) -> dict[str, object]:
        payload = asdict(self)
        if not include_code:
            payload.pop("code")
        return payload


class SemanticCodeSearch:
    """Embed one query and retrieve matching local C# source chunks."""

    def __init__(
        self,
        config: AppConfig,
        *,
        embedding: EmbeddingQueryProvider | None = None,
        vector_store: VectorSearchProvider | None = None,
    ) -> None:
        self._config = config
        self._embedding = embedding or LocalEmbeddingService(config.embedding)
        self._vector_store = vector_store

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: CodeSearchFilters | None = None,
    ) -> list[CodeSearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise SearchError("query must be a non-empty string")
        limit = self._config.search.top_k if top_k is None else top_k
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise SearchError("top_k must be a positive integer")
        effective_filters = filters or CodeSearchFilters(
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
            raise SearchError(str(exc)) from exc
        except Exception as exc:
            raise SearchError("Semantic code search failed") from exc

        return [self._to_code_result(rank, match) for rank, match in enumerate(matches, 1)]

    def search_exact(
        self,
        query: str,
        terms: Sequence[str],
        *,
        top_k: int,
        filters: CodeSearchFilters | None = None,
    ) -> list[CodeSearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise SearchError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise SearchError("top_k must be a positive integer")
        if (
            isinstance(terms, (str, bytes))
            or not isinstance(terms, Sequence)
            or not terms
            or any(not isinstance(term, str) or not term.strip() for term in terms)
        ):
            raise SearchError("terms must contain non-empty strings")
        effective_filters = filters or CodeSearchFilters(
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
            raise SearchError(str(exc)) from exc
        except Exception as exc:
            raise SearchError("Exact code search failed") from exc
        ordered = sorted(matches.values(), key=lambda item: item.distance)[:top_k]
        return [
            self._to_code_result(rank, match)
            for rank, match in enumerate(ordered, 1)
        ]

    def _get_vector_store(self) -> VectorSearchProvider:
        if self._vector_store is None:
            self._vector_store = PersistentChromaStore(
                self._config.chromadb,
                self._embedding.dimension,
            )
        return self._vector_store

    @staticmethod
    def _to_code_result(rank: int, match: SearchResult) -> CodeSearchResult:
        metadata = match.metadata
        return CodeSearchResult(
            rank=rank,
            id=match.id,
            score=1.0 - match.distance,
            distance=match.distance,
            file_name=metadata.file_name,
            relative_path=metadata.relative_path,
            file_path=metadata.file_path,
            class_name=metadata.class_name,
            method_name=metadata.method_name,
            start_line=metadata.start_line,
            end_line=metadata.end_line,
            chunk_index=metadata.chunk_index,
            file_hash=metadata.file_hash,
            modified_time=metadata.modified_time,
            code=match.document,
        )


def format_search_results(
    results: Sequence[CodeSearchResult], *, include_code: bool = True
) -> str:
    """Format ranked results for terminal review."""

    if not results:
        return "No matching code chunks found."

    blocks: list[str] = []
    for result in results:
        line_range = (
            f"{result.start_line}-{result.end_line}"
            if result.start_line > 0
            else "Unknown"
        )
        lines = [
            f"[{result.rank}]",
            f"Score: {result.score:.6f}",
            f"Distance: {result.distance:.6f}",
            f"File: {result.file_name or 'Unknown'}",
            f"Class: {result.class_name or 'Unknown'}",
            f"Method: {result.method_name or 'Unknown'}",
            f"Lines: {line_range}",
            f"Path: {result.file_path or result.relative_path or 'Unknown'}",
        ]
        if include_code:
            lines.extend(("Code:", result.code))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search indexed C# source with the local embedding model"
    )
    parser.add_argument("query", help="Natural-language code search query")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--top-k", type=int, help="Maximum number of results")
    parser.add_argument("--equipment", help="Equipment metadata filter")
    parser.add_argument("--repository", help="Repository metadata filter")
    parser.add_argument("--relative-path", help="Relative path metadata filter")
    parser.add_argument("--class-name", help="Class metadata filter")
    parser.add_argument("--method-name", help="Method metadata filter")
    parser.add_argument("--chroma-path", help="Optional ChromaDB path override")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    parser.add_argument(
        "--no-code",
        action="store_true",
        help="Omit source code from terminal or JSON output",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.chroma_path:
            config = replace(
                config,
                chromadb=replace(
                    config.chromadb,
                    path=Path(args.chroma_path).expanduser().resolve(strict=False),
                ),
            )
        filters = CodeSearchFilters(
            equipment=args.equipment or config.equipment.name,
            repository=args.repository,
            relative_path=args.relative_path,
            class_name=args.class_name,
            method_name=args.method_name,
        )
        results = SemanticCodeSearch(config).search(
            args.query,
            top_k=args.top_k,
            filters=filters,
        )
    except (ConfigError, SearchError) as exc:
        parser.error(str(exc))

    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "equipment": filters.equipment,
                    "top_k": args.top_k or config.search.top_k,
                    "result_count": len(results),
                    "results": [
                        result.to_dict(include_code=not args.no_code)
                        for result in results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(format_search_results(results, include_code=not args.no_code))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
