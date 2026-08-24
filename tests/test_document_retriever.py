from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from app.config import (
    AppConfig,
    ChromaConfig,
    DocumentConfig,
    EmbeddingConfig,
    EquipmentConfig,
    LlmConfig,
    LoggingConfig,
    SearchConfig,
    SourceConfig,
)
from app.models.document_models import DocumentChunkMetadata
from app.retrieval.document_retriever import (
    DocumentRetrievalError,
    DocumentRetriever,
    DocumentSearchFilters,
    format_document_results,
)
from app.vectorstore.chroma_store import SearchResult


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


def _config(root: Path) -> AppConfig:
    return AppConfig(
        project_root=root,
        equipment=EquipmentConfig("Trimming"),
        source=SourceConfig(
            root / "source", (".cs",), ("bin", "obj"), 1000, 100
        ),
        embedding=EmbeddingConfig(root / "model", 8, "cpu", True),
        chromadb=ChromaConfig(root / "chroma", "equipment_code"),
        search=SearchConfig(5),
        llm=LlmConfig(
            "llama_cpp", "http://127.0.0.1:8080/v1", "local-model", 30
        ),
        logging=LoggingConfig("INFO", root / "rag.log"),
        document=DocumentConfig(
            True,
            (root / "documents",),
            (".docx", ".pdf", ".md", ".txt"),
            ("archive",),
            3000,
            300,
            "document_chunks",
        ),
    )


def _match() -> SearchResult:
    metadata = DocumentChunkMetadata(
        document_id="loader-spec",
        source_path="Specs/Loader.docx",
        file_name="Loader.docx",
        file_extension=".docx",
        equipment="Trimming",
        chunk_index=0,
        project="TrimProject",
        unit="Loader",
        document_type="Specification",
        title="Loader Specification",
        revision="Rev.3",
        section="Auto Sequence",
        subsection="Loader Interlock",
        heading_path="Auto Sequence > Loader Interlock",
        page=17,
        file_hash="file-hash",
        content_hash="content-hash",
        indexed_at="2026-08-24T00:00:00Z",
    )
    return SearchResult(
        id="doc-chunk-1",
        document="Vacuum sensor must be ON before Loader pickup.",
        metadata=metadata,
        distance=0.12,
    )


class DocumentRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.embedding = FakeEmbedding()
        self.store = FakeStore([_match()])
        self.retriever = DocumentRetriever(
            _config(self.root),
            embedding=self.embedding,
            vector_store=self.store,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_default_search_filters_active_latest_equipment(self) -> None:
        results = self.retriever.search("  Loader vacuum  ")

        self.assertEqual(self.embedding.queries, ["Loader vacuum"])
        self.assertEqual(
            self.store.calls[0],
            (
                [1.0, 0.0, 0.0],
                5,
                {
                    "$and": [
                        {"equipment": "Trimming"},
                        {"document_status": "active"},
                        {"is_latest": True},
                    ]
                },
            ),
        )
        self.assertAlmostEqual(results[0].score, 0.88)
        self.assertEqual(results[0].metadata.page, 17)

    def test_retrieve_explicit_revision_includes_obsolete_candidates(self) -> None:
        self.retriever.retrieve(
            "old specification",
            revision="Rev.1",
            unit="Loader",
        )

        self.assertEqual(
            self.store.calls[0][2],
            {
                "$and": [
                    {"equipment": "Trimming"},
                    {"unit": "Loader"},
                    {"revision": "Rev.1"},
                ]
            },
        )

    def test_filter_validation_format_and_traceable_payload(self) -> None:
        with self.assertRaisesRegex(DocumentRetrievalError, "unit"):
            DocumentSearchFilters(unit=" ")
        result = self.retriever.search(
            "vacuum",
            filters=DocumentSearchFilters(
                equipment="Trimming",
                unit="Loader",
                document_type="Specification",
            ),
        )[0]

        payload = result.to_dict(include_text=False)
        output = format_document_results([result])

        self.assertNotIn("text", payload)
        self.assertEqual(payload["source_path"], "Specs/Loader.docx")
        self.assertEqual(payload["metadata"]["revision"], "Rev.3")
        self.assertIn("Page: 17", output)
        self.assertIn("Auto Sequence > Loader Interlock", output)


if __name__ == "__main__":
    unittest.main()
