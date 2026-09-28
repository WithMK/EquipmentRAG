from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Sequence

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
from app.indexer import IncrementalSourceIndexer, IndexerError
from app.retrieval.sqlite_fts import SQLiteFtsStore
from app.vectorstore.chroma_store import VectorRecord


class FakeEmbedding:
    dimension = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail = False

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        self.calls.append(values)
        if self.fail:
            raise RuntimeError("synthetic embedding failure")
        return [[1.0, float(index % 2), 0.0] for index, _ in enumerate(values)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.records: dict[str, VectorRecord] = {}
        self.upsert_calls: list[list[str]] = []
        self.delete_file_calls: list[tuple[str, str | None]] = []
        self.delete_equipment_calls: list[str] = []

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        values = list(records)
        self.upsert_calls.append([record.id for record in values])
        for record in values:
            self.records[record.id] = record
        return len(values)

    def delete_by_file_path(
        self, file_path: str, *, equipment: str | None = None
    ) -> int:
        self.delete_file_calls.append((file_path, equipment))
        deleted = [
            record_id
            for record_id, record in self.records.items()
            if record.metadata.file_path == file_path
            and (equipment is None or record.metadata.equipment == equipment)
        ]
        for record_id in deleted:
            del self.records[record_id]
        return len(deleted)

    def delete_by_equipment(self, equipment: str) -> int:
        self.delete_equipment_calls.append(equipment)
        deleted = [
            record_id
            for record_id, record in self.records.items()
            if record.metadata.equipment == equipment
        ]
        for record_id in deleted:
            del self.records[record_id]
        return len(deleted)


class IncrementalSourceIndexerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_root = self.root / "source"
        self.source_root.mkdir()
        self.config = self._config()
        self.embedding = FakeEmbedding()
        self.store = FakeVectorStore()
        self.state_path = self.root / "chroma" / "index-state.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _config(self) -> AppConfig:
        return AppConfig(
            project_root=self.root,
            equipment=EquipmentConfig(name="test-equipment"),
            source=SourceConfig(
                path=self.source_root,
                include_extensions=(".cs",),
                exclude_directories=("bin", "obj", ".git"),
                chunk_size=1000,
                chunk_overlap=100,
            ),
            embedding=EmbeddingConfig(
                model_path=self.root / "model",
                batch_size=8,
                device="cpu",
                normalize_embeddings=True,
            ),
            chromadb=ChromaConfig(
                path=self.root / "chroma",
                collection_name="test_equipment_code",
            ),
            search=SearchConfig(top_k=3),
            llm=LlmConfig(
                provider="llama_cpp",
                base_url="http://127.0.0.1:8080/v1",
                model="local-model",
                request_timeout_seconds=30,
            ),
            logging=LoggingConfig(level="INFO", path=self.root / "indexer.log"),
        )

    def _indexer(self, config: AppConfig | None = None) -> IncrementalSourceIndexer:
        return IncrementalSourceIndexer(
            config or self.config,
            embedding=self.embedding,
            vector_store=self.store,
            state_path=self.state_path,
        )

    def _write(self, relative_path: str, method_name: str) -> Path:
        path = self.source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "namespace Synthetic;\n"
            "public class Controller\n"
            "{\n"
            f"    public void {method_name}()\n"
            "    {\n"
            "        machine.Run();\n"
            "    }\n"
            "}\n",
            encoding="utf-8",
        )
        return path

    def test_indexes_new_then_only_changed_and_deleted_files(self) -> None:
        first_path = self._write("Motion/Axis.cs", "Home")
        second_path = self._write("Alarm.cs", "Reset")
        empty_path = self.source_root / "Empty.cs"
        empty_path.write_text("", encoding="utf-8")

        first = self._indexer().run()

        self.assertEqual(
            first.new_files,
            ("Alarm.cs", "Empty.cs", "Motion/Axis.cs"),
        )
        self.assertEqual(first.reindex_reason, "initial_state")
        self.assertGreater(first.upserted_chunks, 0)
        self.assertEqual(self.store.delete_equipment_calls, ["test-equipment"])
        self.assertTrue(self.state_path.is_file())
        self.assertNotIn("machine.Run", self.state_path.read_text(encoding="utf-8"))
        first_embedding_calls = len(self.embedding.calls)

        unchanged = self._indexer().run()

        self.assertEqual(
            unchanged.skipped_files,
            ("Alarm.cs", "Empty.cs", "Motion/Axis.cs"),
        )
        self.assertEqual(unchanged.prepared_chunks, 0)
        self.assertEqual(len(self.embedding.calls), first_embedding_calls)

        self._write("Motion/Axis.cs", "HomeChanged")
        second_path.unlink()
        self._write("Safety.cs", "Stop")

        incremental = self._indexer().run()

        self.assertEqual(incremental.new_files, ("Safety.cs",))
        self.assertEqual(incremental.changed_files, ("Motion/Axis.cs",))
        self.assertEqual(incremental.deleted_files, ("Alarm.cs",))
        indexed_paths = {
            record.metadata.relative_path for record in self.store.records.values()
        }
        self.assertEqual(indexed_paths, {"Motion/Axis.cs", "Safety.cs"})
        self.assertTrue(
            all(
                record.metadata.file_hash
                and record.metadata.language == "csharp"
                and record.metadata.source_type == "code"
                and record.metadata.start_line > 0
                and record.metadata.end_line >= record.metadata.start_line
                for record in self.store.records.values()
            )
        )
        self.assertTrue(first_path.is_file())

    def test_dry_run_does_not_embed_mutate_store_or_write_state(self) -> None:
        self._write("Axis.cs", "Home")
        indexer = self._indexer()
        initial = indexer.run()
        state_before = self.state_path.read_bytes()
        records_before = dict(self.store.records)
        embedding_calls_before = len(self.embedding.calls)
        self._write("Alarm.cs", "Reset")

        report = self._indexer().run(dry_run=True)

        self.assertTrue(report.dry_run)
        self.assertEqual(report.new_files, ("Alarm.cs",))
        self.assertGreater(report.prepared_chunks, 0)
        self.assertEqual(report.upserted_chunks, 0)
        self.assertEqual(self.state_path.read_bytes(), state_before)
        self.assertEqual(self.store.records, records_before)
        self.assertEqual(len(self.embedding.calls), embedding_calls_before)
        self.assertFalse(initial.dry_run)

    def test_full_and_settings_change_reindex_every_current_file(self) -> None:
        self._write("Axis.cs", "Home")
        self._write("Alarm.cs", "Reset")
        self._indexer().run()

        full = self._indexer().run(full_reindex=True)

        self.assertTrue(full.full_reindex)
        self.assertEqual(full.reindex_reason, "requested_full_reindex")
        self.assertEqual(full.reindexed_files, ("Alarm.cs", "Axis.cs"))

        changed_config = replace(
            self.config,
            source=replace(self.config.source, chunk_size=800),
        )
        settings = self._indexer(changed_config).run(dry_run=True)

        self.assertTrue(settings.full_reindex)
        self.assertEqual(settings.reindex_reason, "index_settings_changed")
        self.assertEqual(settings.reindexed_files, ("Alarm.cs", "Axis.cs"))

    def test_synchronizes_incremental_changes_to_sqlite_fts(self) -> None:
        source = self._write("Alarm.cs", "ResetE024")
        lexical = SQLiteFtsStore(self.root / "lexical.sqlite3")
        config = replace(
            self.config,
            search=replace(
                self.config.search,
                lexical_backend="sqlite_fts5",
                lexical_path=lexical.path,
            ),
        )
        indexer = IncrementalSourceIndexer(
            config,
            embedding=self.embedding,
            vector_store=self.store,
            lexical_store=lexical,
            state_path=self.state_path,
        )

        indexer.run()
        self.assertEqual(lexical.search("ResetE024", 5)[0].metadata.method_name, "ResetE024")

        self._write("Alarm.cs", "ClearServoFault")
        indexer.run()
        self.assertEqual(lexical.search("ResetE024", 5), [])
        self.assertEqual(
            lexical.search("ClearServoFault", 5)[0].metadata.method_name,
            "ClearServoFault",
        )

        source.unlink()
        indexer.run()
        self.assertEqual(lexical.count(), 0)
        lexical.close()

    def test_embedding_failure_preserves_existing_store_and_manifest(self) -> None:
        self._write("Axis.cs", "Home")
        self._indexer().run()
        state_before = self.state_path.read_bytes()
        records_before = dict(self.store.records)
        self._write("Axis.cs", "HomeChanged")
        self.embedding.fail = True

        with self.assertRaisesRegex(IndexerError, "Embedding provider failed"):
            self._indexer().run()

        self.assertEqual(self.state_path.read_bytes(), state_before)
        self.assertEqual(self.store.records, records_before)


if __name__ == "__main__":
    unittest.main()
