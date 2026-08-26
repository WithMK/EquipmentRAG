from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from app.config import (
    AppConfig,
    ChromaConfig,
    EmbeddingConfig,
    EquipmentConfig,
    LlmConfig,
    LoggingConfig,
    SearchConfig,
    SourceConfig,
)
from app.search import (
    CodeSearchFilters,
    SemanticCodeSearch,
    SearchError,
    format_search_results,
)
from app.vectorstore.chroma_store import ChunkMetadata, SearchResult


class FakeEmbedding:
    dimension = 3

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0, 0.0]


class FakeStore:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[list[float], int, dict[str, Any] | None]] = []

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        self.calls.append((query_embedding, top_k, where))
        return self.results[:top_k]


class ExactFakeStore(FakeStore):
    def __init__(self, results: list[SearchResult]) -> None:
        super().__init__(results)
        self.exact_calls: list[dict[str, Any] | None] = []

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        *,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        self.exact_calls.append(where_document)
        return self.results[:top_k]


def _config(root: Path) -> AppConfig:
    return AppConfig(
        project_root=root,
        equipment=EquipmentConfig(name="press-line-01"),
        source=SourceConfig(
            path=root / "source",
            include_extensions=(".cs",),
            exclude_directories=("bin", "obj"),
            chunk_size=1000,
            chunk_overlap=100,
        ),
        embedding=EmbeddingConfig(
            model_path=root / "model",
            batch_size=8,
            device="cpu",
            normalize_embeddings=True,
        ),
        chromadb=ChromaConfig(
            path=root / "chroma",
            collection_name="equipment_code",
        ),
        search=SearchConfig(top_k=3),
        llm=LlmConfig(
            provider="llama_cpp",
            base_url="http://127.0.0.1:8080/v1",
            model="local-model",
            request_timeout_seconds=30,
        ),
        logging=LoggingConfig(level="INFO", path=root / "search.log"),
    )


def _match() -> SearchResult:
    return SearchResult(
        id="axis-home-0",
        document="public void HomeZAxis() { zAxis.MoveHome(); }",
        distance=0.125,
        metadata=ChunkMetadata(
            equipment="press-line-01",
            repository="synthetic-source",
            project="SyntheticProject",
            file_name="AxisController.cs",
            file_path="C:/synthetic/Motion/AxisController.cs",
            relative_path="Motion/AxisController.cs",
            class_name="AxisController",
            method_name="HomeZAxis",
            chunk_index=0,
            start_line=10,
            end_line=14,
            file_hash="abc123",
            modified_time="2026-01-01T00:00:00Z",
        ),
    )


class SemanticCodeSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="equipment-rag-search-"))
        self.embedding = FakeEmbedding()
        self.store = FakeStore([_match()])
        self.search = SemanticCodeSearch(
            _config(self.root),
            embedding=self.embedding,
            vector_store=self.store,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=False)

    def test_search_uses_configured_limit_and_equipment_scope(self) -> None:
        results = self.search.search("  Z축 원점 복귀 실패  ")

        self.assertEqual(self.embedding.queries, ["Z축 원점 복귀 실패"])
        self.assertEqual(
            self.store.calls,
            [([1.0, 0.0, 0.0], 3, {"equipment": "press-line-01"})],
        )
        self.assertEqual(results[0].rank, 1)
        self.assertAlmostEqual(results[0].score, 0.875)
        self.assertEqual(results[0].method_name, "HomeZAxis")
        self.assertEqual(results[0].start_line, 10)

    def test_search_builds_combined_exact_match_filters(self) -> None:
        filters = CodeSearchFilters(
            equipment="press-line-01",
            repository="synthetic-source",
            relative_path="Motion/AxisController.cs",
            class_name="AxisController",
            method_name="HomeZAxis",
        )

        self.search.search("home", top_k=1, filters=filters)

        self.assertEqual(
            self.store.calls[0][2],
            {
                "$and": [
                    {"equipment": "press-line-01"},
                    {"repository": "synthetic-source"},
                    {"relative_path": "Motion/AxisController.cs"},
                    {"class_name": "AxisController"},
                    {"method_name": "HomeZAxis"},
                ]
            },
        )

    def test_exact_search_uses_independent_chroma_document_filters(self) -> None:
        store = ExactFakeStore([_match()])
        search = SemanticCodeSearch(
            _config(self.root),
            embedding=self.embedding,
            vector_store=store,
        )

        results = search.search_exact(
            "E-024 ServoAlarm",
            ("E-024", "ServoAlarm"),
            top_k=3,
        )

        self.assertEqual(
            store.exact_calls,
            [{"$contains": "E-024"}, {"$contains": "ServoAlarm"}],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(self.embedding.queries, ["E-024 ServoAlarm"])

    def test_rejects_invalid_query_limit_and_filter(self) -> None:
        with self.assertRaisesRegex(SearchError, "query"):
            self.search.search(" ")
        with self.assertRaisesRegex(SearchError, "top_k"):
            self.search.search("home", top_k=0)
        with self.assertRaisesRegex(SearchError, "class_name"):
            CodeSearchFilters(equipment="press-line-01", class_name=" ")

    def test_formats_human_output_and_empty_results(self) -> None:
        result = self.search.search("home")[0]

        output = format_search_results([result])

        self.assertIn("[1]", output)
        self.assertIn("Score: 0.875000", output)
        self.assertIn("File: AxisController.cs", output)
        self.assertIn("Class: AxisController", output)
        self.assertIn("Method: HomeZAxis", output)
        self.assertIn("Lines: 10-14", output)
        self.assertIn("Code:\npublic void HomeZAxis", output)
        self.assertEqual(
            format_search_results([]),
            "No matching code chunks found.",
        )

    def test_json_payload_can_omit_source_code(self) -> None:
        result = self.search.search("home")[0]

        payload = result.to_dict(include_code=False)

        self.assertNotIn("code", payload)
        self.assertEqual(payload["relative_path"], "Motion/AxisController.cs")


if __name__ == "__main__":
    unittest.main()
