from __future__ import annotations

import gc
import shutil
import tempfile
import unittest
from pathlib import Path

from app.config import ChromaConfig
from app.models.document_models import DocumentChunkMetadata
from app.vectorstore.chroma_store import (
    ChunkMetadata,
    PersistentChromaStore,
    VectorRecord,
    VectorStoreError,
)


class ChromaStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_path = Path(tempfile.mkdtemp(prefix="equipment-rag-chroma-"))
        self.config = ChromaConfig(
            path=self.temp_path / "chroma",
            collection_name="test_equipment_code",
        )
        self.stores: list[PersistentChromaStore] = []

    def tearDown(self) -> None:
        self._stop_chroma()
        shutil.rmtree(self.temp_path, ignore_errors=False)

    def _store(self, dimension: int = 3) -> PersistentChromaStore:
        store = PersistentChromaStore(self.config, dimension)
        self.stores.append(store)
        return store

    def _stop_chroma(self) -> None:
        systems = {
            store._client._system
            for store in self.stores
            if store._client is not None
        }
        for store in self.stores:
            store._collection = None
            store._client = None
        for system in systems:
            system.stop()

        if systems:
            from chromadb.api.client import SharedSystemClient

            SharedSystemClient.clear_system_cache()
        gc.collect()
        self.stores.clear()

    @staticmethod
    def _metadata(file_name: str, chunk_index: int = 0) -> ChunkMetadata:
        return ChunkMetadata(
            equipment="test-equipment",
            repository="sample-repository",
            project="SampleProject",
            file_name=file_name,
            file_path=f"C:/synthetic/{file_name}",
            relative_path=file_name,
            class_name=file_name.removesuffix(".cs"),
            method_name="Run",
            chunk_index=chunk_index,
            start_line=10 + chunk_index,
            end_line=20 + chunk_index,
            file_hash=f"hash-{file_name}",
            modified_time="2026-01-01T00:00:00Z",
        )

    def test_upsert_search_and_persist_across_client_restart(self) -> None:
        store = self._store()
        inserted = store.upsert(
            [
                VectorRecord(
                    id="z-home",
                    document="public void HomeZAxis() { zAxis.MoveHome(); }",
                    embedding=[1.0, 0.0, 0.0],
                    metadata=self._metadata("ZAxis.cs"),
                ),
                VectorRecord(
                    id="press-reset",
                    document="public void ResetPressAlarm() { press.Alarm.Reset(); }",
                    embedding=[0.0, 1.0, 0.0],
                    metadata=self._metadata("Press.cs"),
                ),
            ]
        )

        self.assertEqual(inserted, 2)
        self.assertEqual(store.count(), 2)
        results = store.search(
            [1.0, 0.0, 0.0],
            5,
            where={"equipment": "test-equipment"},
        )
        self.assertEqual(results[0].id, "z-home")
        self.assertAlmostEqual(results[0].distance, 0.0, places=6)
        self.assertEqual(results[0].metadata.language, "csharp")
        self.assertEqual(results[0].metadata.start_line, 10)
        self.assertEqual(results[0].metadata.end_line, 20)

        store.upsert(
            [
                VectorRecord(
                    id="z-home",
                    document="public void HomeZAxis() { zAxis.Home(); }",
                    embedding=[1.0, 0.0, 0.0],
                    metadata=self._metadata("ZAxis.cs"),
                )
            ]
        )
        self.assertEqual(store.count(), 2)

        self._stop_chroma()
        restarted = self._store()
        self.assertEqual(restarted.count(), 2)
        self.assertIn("zAxis.Home", restarted.get_by_ids(["z-home"])[0].document)
        self.assertTrue((self.config.path / "chroma.sqlite3").is_file())

    def test_delete_by_id_and_file_path(self) -> None:
        store = self._store()
        store.upsert(
            [
                VectorRecord(
                    id="chunk-0",
                    document="first chunk",
                    embedding=[1.0, 0.0, 0.0],
                    metadata=self._metadata("Motion.cs", 0),
                ),
                VectorRecord(
                    id="chunk-1",
                    document="second chunk",
                    embedding=[0.9, 0.1, 0.0],
                    metadata=self._metadata("Motion.cs", 1),
                ),
                VectorRecord(
                    id="chunk-2",
                    document="other file",
                    embedding=[0.0, 1.0, 0.0],
                    metadata=self._metadata("Alarm.cs", 0),
                ),
            ]
        )

        self.assertEqual(store.delete_by_file_path("C:/synthetic/Motion.cs"), 2)
        self.assertEqual(store.count(), 1)
        self.assertEqual(store.delete_by_ids(["chunk-2", "missing"]), 1)
        self.assertEqual(store.count(), 0)
        self.assertEqual(store.search([1.0, 0.0, 0.0], 3), [])

    def test_scopes_file_deletion_by_equipment(self) -> None:
        store = self._store()
        shared_path = "C:/synthetic/Shared.cs"
        first_metadata = self._metadata("Shared.cs")
        second_metadata = ChunkMetadata(
            **{
                **first_metadata.to_chroma(),
                "equipment": "other-equipment",
            }
        )
        store.upsert(
            [
                VectorRecord("first", "first", [1.0, 0.0, 0.0], first_metadata),
                VectorRecord("second", "second", [0.0, 1.0, 0.0], second_metadata),
            ]
        )

        self.assertEqual(
            store.delete_by_file_path(
                shared_path,
                equipment="test-equipment",
            ),
            1,
        )
        self.assertEqual(store.count(), 1)
        self.assertEqual(store.delete_by_equipment("other-equipment"), 1)
        self.assertEqual(store.count(), 0)

    def test_rejects_invalid_dimensions_duplicates_and_collection_reuse(self) -> None:
        store = self._store()
        with self.assertRaisesRegex(VectorStoreError, "dimension mismatch"):
            store.search([1.0, 0.0], 1)

        record = VectorRecord(
            id="duplicate",
            document="sample",
            embedding=[1.0, 0.0, 0.0],
            metadata=self._metadata("Sample.cs"),
        )
        with self.assertRaisesRegex(VectorStoreError, "duplicate record id"):
            store.upsert([record, record])

        store.open()
        self._stop_chroma()
        incompatible = self._store(dimension=2)
        with self.assertRaisesRegex(VectorStoreError, "does not match"):
            incompatible.open()

    def test_reuses_store_with_document_metadata_codec(self) -> None:
        document_config = ChromaConfig(
            path=self.temp_path / "document-chroma",
            collection_name="document_chunks",
        )
        store = PersistentChromaStore(
            document_config,
            3,
            metadata_type=DocumentChunkMetadata,
        )
        self.stores.append(store)
        metadata = DocumentChunkMetadata(
            document_id="spec-001",
            source_path="Specs/Loader.docx",
            file_name="Loader.docx",
            file_extension=".docx",
            equipment="test-equipment",
            chunk_index=0,
            title="Loader Specification",
            revision="Rev.3",
            section="Loader Interlock",
            file_hash="file-hash",
            content_hash="content-hash",
            indexed_at="2026-08-24T00:00:00Z",
        )

        store.upsert([VectorRecord("doc-1", "vacuum interlock", [1, 0, 0], metadata)])
        result = store.search(
            [1, 0, 0],
            1,
            where={"document_status": "active"},
        )[0]

        self.assertIsInstance(result.metadata, DocumentChunkMetadata)
        self.assertEqual(result.metadata.revision, "Rev.3")
        self.assertEqual(
            store.delete_where({"document_id": "spec-001"}),
            1,
        )


if __name__ == "__main__":
    unittest.main()
