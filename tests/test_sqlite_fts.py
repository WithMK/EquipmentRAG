from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from app.models.document_models import DocumentChunkMetadata
from app.retrieval.sqlite_fts import LexicalStoreError, SQLiteFtsStore
from app.vectorstore.chroma_store import ChunkMetadata, VectorRecord


class SQLiteFtsStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="equipment-rag-fts-"))
        self.path = self.root / "lexical.sqlite3"
        self.store = SQLiteFtsStore(self.path)

    def tearDown(self) -> None:
        self.store.close()
        shutil.rmtree(self.root, ignore_errors=False)

    @staticmethod
    def _code_record(record_id: str, document: str) -> VectorRecord:
        return VectorRecord(
            id=record_id,
            document=document,
            embedding=[1.0, 0.0],
            metadata=ChunkMetadata(
                equipment="press-line-01",
                repository="control",
                project="Equipment.Control",
                file_name="AlarmController.cs",
                file_path="C:/equipment/AlarmController.cs",
                relative_path="AlarmController.cs",
                class_name="AlarmController",
                method_name="ResetE024",
                chunk_index=0,
                start_line=10,
                end_line=20,
                file_hash="code-hash",
                modified_time="2026-08-26T00:00:00Z",
            ),
        )

    @staticmethod
    def _document_record() -> VectorRecord:
        return VectorRecord(
            id="document-1",
            document="Vacuum pressure interlock recovery procedure",
            embedding=[0.0, 1.0],
            metadata=DocumentChunkMetadata(
                document_id="manual-1",
                source_path="Manuals/Alarm.pdf",
                file_name="Alarm.pdf",
                file_extension=".pdf",
                equipment="press-line-01",
                chunk_index=0,
                document_type="maintenance",
                revision="Rev.2",
                document_status="active",
                is_latest=True,
                section="Vacuum Alarm",
            ),
        )

    def test_searches_identifiers_and_persists_across_restart(self) -> None:
        self.store.upsert(
            [
                self._code_record(
                    "code-e024",
                    'if (alarm.Code == "E-024") alarm.Reset();',
                ),
                self._code_record("code-other", "Handle generic drive alarm"),
            ]
        )

        results = self.store.search(
            "E-024",
            5,
            filters={"equipment": "press-line-01", "source_type": "code"},
        )

        self.assertEqual([result.id for result in results], ["code-e024"])
        self.store.close()
        restarted = SQLiteFtsStore(self.path)
        self.store = restarted
        self.assertEqual(restarted.count(), 2)
        self.assertEqual(restarted.search("generic drive", 1)[0].id, "code-other")

    def test_filters_code_and_document_records_independently(self) -> None:
        self.store.upsert(
            [
                self._code_record("code-1", "Vacuum interlock in controller"),
                self._document_record(),
            ]
        )

        code = self.store.search(
            "vacuum interlock",
            5,
            filters={"source_type": "code"},
        )
        documents = self.store.search(
            "vacuum interlock",
            5,
            filters={
                "source_type": "document",
                "document_status": "active",
                "is_latest": True,
            },
        )

        self.assertEqual([result.id for result in code], ["code-1"])
        self.assertEqual([result.id for result in documents], ["document-1"])
        self.assertIsInstance(documents[0].metadata, DocumentChunkMetadata)

    def test_upsert_and_delete_keep_incremental_index_consistent(self) -> None:
        self.store.upsert([self._code_record("code-1", "old alarm wording")])
        self.store.upsert([self._code_record("code-1", "new servo wording")])

        self.assertEqual(self.store.search("old", 5), [])
        self.assertEqual(self.store.search("servo", 5)[0].id, "code-1")
        self.assertEqual(
            self.store.delete_by_file_path(
                "C:/equipment/AlarmController.cs",
                equipment="press-line-01",
            ),
            1,
        )
        self.assertEqual(self.store.count(), 0)

    def test_rejects_unsupported_filter(self) -> None:
        self.store.upsert([self._code_record("code-1", "alarm")])

        with self.assertRaisesRegex(LexicalStoreError, "Unsupported lexical filter"):
            self.store.search("alarm", 1, filters={"unknown": "value"})


if __name__ == "__main__":
    unittest.main()
