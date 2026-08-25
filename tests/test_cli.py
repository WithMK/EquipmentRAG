from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.cli import (
    _run_evaluate,
    _run_index,
    build_parser,
    build_status,
    format_status,
)
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
from app.evaluation import EvaluationReport


def _config(root: Path, *, document_enabled: bool = True) -> AppConfig:
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
        search=SearchConfig(top_k=5),
        llm=LlmConfig(
            provider="ollama",
            base_url="http://127.0.0.1:11434/api",
            model="local-model",
            request_timeout_seconds=30,
        ),
        logging=LoggingConfig(level="INFO", path=root / "rag.log"),
        document=DocumentConfig(
            enabled=document_enabled,
            source_paths=(root / "documents",),
            extensions=(".pdf", ".docx"),
            exclude_directories=("archive",),
            chunk_size=1000,
            chunk_overlap=100,
            collection_name="document_chunks",
        ),
    )


class FakeReport:
    def to_dict(self) -> dict[str, object]:
        return {
            "dry_run": True,
            "full_reindex": False,
            "reindex_reason": "incremental",
            "total_files": 0,
            "new_files": [],
            "changed_files": [],
            "reindexed_files": [],
            "skipped_files": [],
            "deleted_files": [],
            "prepared_chunks": 0,
            "upserted_chunks": 0,
            "deleted_chunks": 0,
        }


class FakeIndexer:
    embeddings: list[object] = []

    def __init__(self, _config: AppConfig, *, embedding: object) -> None:
        self.__class__.embeddings.append(embedding)

    def run(self, *, dry_run: bool, full_reindex: bool) -> FakeReport:
        self.dry_run = dry_run
        self.full_reindex = full_reindex
        return FakeReport()


class UnifiedCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="equipment-rag-cli-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=False)

    def test_parser_exposes_operator_commands_and_all_search(self) -> None:
        parser = build_parser()

        search = parser.parse_args(
            [
                "search",
                "Vacuum alarm",
                "--source-type",
                "all",
                "--document-type",
                "Manual",
                "--class-name",
                "AxisController",
                "--search-mode",
                "hybrid",
                "--candidate-multiplier",
                "6",
            ]
        )

        self.assertEqual(search.command, "search")
        self.assertEqual(search.source_type, "all")
        self.assertEqual(search.document_type, "Manual")
        self.assertEqual(search.class_name, "AxisController")
        self.assertEqual(search.search_mode, "hybrid")
        self.assertEqual(search.candidate_multiplier, 6)
        for command in (
            "status",
            "index",
            "ask",
            "chat",
            "serve",
            "evaluate",
        ):
            args = parser.parse_args(
                [command, "question"]
                if command == "ask"
                else [command, "--dataset", "evaluation.jsonl"]
                if command == "evaluate"
                else [command]
            )
            self.assertTrue(callable(args.handler))

    def test_status_is_read_only_and_handles_disabled_documents(self) -> None:
        config = _config(self.root, document_enabled=False)
        config.source.path.mkdir()
        config.embedding.model_path.mkdir()

        status = build_status(config)
        output = format_status(status)

        self.assertTrue(status["code_source"]["exists"])
        self.assertEqual(status["version"], "0.3.0-rc.1")
        self.assertFalse(status["documents"]["enabled"])
        self.assertIsNone(status["documents"]["index"])
        self.assertIn("Document index: disabled", output)
        self.assertIn("Embedding model: ready", output)
        self.assertFalse(config.chromadb.path.exists())

    def test_all_indexing_shares_one_embedding_service(self) -> None:
        config = _config(self.root)
        args = build_parser().parse_args(
            ["index", "--source-type", "all", "--dry-run"]
        )
        FakeIndexer.embeddings = []

        with (
            patch("app.cli._load_command_config", return_value=config),
            patch("app.cli.LocalEmbeddingService", return_value=object()),
            patch("app.cli.IncrementalSourceIndexer", FakeIndexer),
            patch("app.cli.IncrementalDocumentIndexer", FakeIndexer),
            patch("builtins.print"),
        ):
            exit_code = _run_index(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(FakeIndexer.embeddings), 2)
        self.assertIs(FakeIndexer.embeddings[0], FakeIndexer.embeddings[1])

    def test_evaluate_writes_json_and_returns_failure_for_missed_threshold(self) -> None:
        config = _config(self.root)
        output = self.root / "results" / "evaluation.json"
        args = build_parser().parse_args(
            [
                "evaluate",
                "--dataset",
                "evaluation.jsonl",
                "--min-recall",
                "0.8",
                "--output",
                str(output),
                "--json",
            ]
        )
        report = EvaluationReport(
            dataset_path="evaluation.jsonl",
            case_count=1,
            expected_count=2,
            matched_count=1,
            hit_rate=1.0,
            recall_at_k=0.5,
            mrr=1.0,
            cases=(),
        )

        with (
            patch("app.cli._load_command_config", return_value=config),
            patch("app.cli.evaluate_retrieval", return_value=report),
            patch("builtins.print"),
        ):
            exit_code = _run_evaluate(args)

        self.assertEqual(exit_code, 1)
        self.assertTrue(output.is_file())
        self.assertIn('"recall_at_k": 0.5', output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
