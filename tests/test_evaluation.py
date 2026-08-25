from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

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
from app.evaluation import (
    EvaluationError,
    evaluate_retrieval,
    format_evaluation_report,
    load_evaluation_dataset,
)
from app.rag_service import RagSource


def _config(root: Path) -> AppConfig:
    return AppConfig(
        project_root=root,
        equipment=EquipmentConfig("press-line-01"),
        source=SourceConfig(
            root / "source",
            (".cs",),
            ("bin", "obj"),
            1000,
            100,
        ),
        embedding=EmbeddingConfig(root / "model", 8, "cpu", True),
        chromadb=ChromaConfig(root / "chroma", "code"),
        search=SearchConfig(top_k=3),
        llm=LlmConfig("ollama", "http://127.0.0.1:11434/api", "local", 30),
        logging=LoggingConfig("INFO", root / "rag.log"),
        document=DocumentConfig(
            True,
            (root / "documents",),
            (".pdf",),
            ("archive",),
            1000,
            100,
            "documents",
        ),
    )


def _code_source(record_id: str, method: str, score: float) -> RagSource:
    return RagSource(
        source_id="C1",
        record_id=record_id,
        score=score,
        file_name="Loader.cs",
        relative_path="Controls\\Loader.cs",
        file_path="D:/source/Controls/Loader.cs",
        class_name="LoaderController",
        method_name=method,
        start_line=10,
        end_line=20,
        code="code",
    )


def _document_source(record_id: str, section: str, score: float) -> RagSource:
    return RagSource(
        source_id="D1",
        record_id=record_id,
        score=score,
        file_name="LoaderManual.pdf",
        relative_path="Manuals/LoaderManual.pdf",
        file_path="Manuals/LoaderManual.pdf",
        class_name="",
        method_name="",
        start_line=0,
        end_line=0,
        code="manual",
        source_type="document",
        document_type="Manual",
        revision="Rev.3",
        document_status="active",
        section=section,
        page=12,
    )


class _FakeService:
    def __init__(self, results: dict[str, tuple[RagSource, ...]]) -> None:
        self.results = results
        self.calls: list[tuple[str, object, object]] = []

    def retrieve(self, query: str, *, top_k: int | None, filters: object) -> tuple[RagSource, ...]:
        self.calls.append((query, top_k, filters))
        return self.results[query]


class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="equipment-rag-eval-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=False)

    def _write_dataset(self, values: list[dict[str, object]]) -> Path:
        path = self.root / "evaluation.jsonl"
        path.write_text(
            "\n".join(json.dumps(value, ensure_ascii=False) for value in values),
            encoding="utf-8",
        )
        return path

    def test_scores_hit_recall_and_mrr_across_cases(self) -> None:
        dataset = self._write_dataset(
            [
                {
                    "id": "code-hit-rank-2",
                    "query": "vacuum code",
                    "source_type": "code",
                    "top_k": 3,
                    "expected": [
                        {
                            "relative_path": "controls/loader.cs",
                            "method_name": "CheckVacuum",
                        }
                    ],
                },
                {
                    "id": "document-half-recall",
                    "query": "vacuum manual",
                    "source_type": "document",
                    "filters": {"document": {"unit": "Loader"}},
                    "expected": [
                        {"file_name": "LoaderManual.pdf", "section": "Recovery"},
                        {"file_name": "Missing.pdf"},
                    ],
                },
            ]
        )
        services = {
            "code": _FakeService(
                {
                    "vacuum code": (
                        _code_source("wrong", "Start", 0.9),
                        _code_source("right", "CheckVacuum", 0.8),
                    )
                }
            ),
            "document": _FakeService(
                {
                    "vacuum manual": (
                        _document_source("manual", "Recovery", 0.95),
                    )
                }
            ),
        }

        report = evaluate_retrieval(
            _config(self.root),
            dataset,
            service_factory=lambda _config, source_type: services[source_type],
        )

        self.assertEqual(report.case_count, 2)
        self.assertEqual(report.matched_count, 2)
        self.assertEqual(report.expected_count, 3)
        self.assertEqual(report.hit_rate, 1.0)
        self.assertAlmostEqual(report.recall_at_k, 2 / 3)
        self.assertAlmostEqual(report.mrr, 0.75)
        self.assertEqual(report.cases[0].first_relevant_rank, 2)
        self.assertEqual(report.cases[1].recall, 0.5)
        self.assertIn("Hit@K: 1.0000", format_evaluation_report(report))
        document_filters = services["document"].calls[0][2]
        self.assertEqual(document_filters.unit, "Loader")

    def test_rejects_duplicate_ids_unknown_match_fields_and_empty_dataset(self) -> None:
        duplicate = self._write_dataset(
            [
                {"id": "same", "query": "one", "expected": [{"record_id": "1"}]},
                {"id": "same", "query": "two", "expected": [{"record_id": "2"}]},
            ]
        )
        with self.assertRaisesRegex(EvaluationError, "Duplicate"):
            load_evaluation_dataset(duplicate)

        invalid = self._write_dataset(
            [
                {
                    "id": "invalid",
                    "query": "one",
                    "expected": [{"code": "secret"}],
                }
            ]
        )
        with self.assertRaisesRegex(EvaluationError, "Unknown expected"):
            load_evaluation_dataset(invalid)

        empty = self._write_dataset([])
        with self.assertRaisesRegex(EvaluationError, "no cases"):
            load_evaluation_dataset(empty)


if __name__ == "__main__":
    unittest.main()
