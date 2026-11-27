import unittest
from pathlib import Path
from types import SimpleNamespace

from app.config import ChromaConfig
from app.vectorstore.chroma_store import PersistentChromaStore, ChunkMetadata, VectorRecord, VectorStoreError


class LimitedCollection:
    def __init__(self, limit):
        self.limit, self.rows, self.sizes = limit, {}, []
        self.fail_batch = None

    def upsert(self, ids, documents, embeddings, metadatas):
        if len(ids) > self.limit:
            raise ValueError('Batch size exceeds maximum')
        if self.fail_batch == len(self.sizes) + 1:
            raise RuntimeError('Injected write failure')
        self.sizes.append(len(ids))
        for values in zip(ids, documents, embeddings, metadatas, strict=True):
            self.rows[values[0]] = values[1:]

    def get(self, ids, include):
        assert len(ids) <= self.limit
        found = [key for key in ids if key in self.rows]
        return dict(ids=found, documents=[self.rows[k][0] for k in found], metadatas=[self.rows[k][2] for k in found])

    def delete(self, ids):
        assert len(ids) <= self.limit
        for key in ids:
            self.rows.pop(key, None)


class ChromaBatchingTests(unittest.TestCase):
    def store(self, limit):
        store = PersistentChromaStore(ChromaConfig(Path('unused'), 'batch-test'), 3)
        store._client = SimpleNamespace(get_max_batch_size=lambda:limit)
        store._collection = LimitedCollection(limit)
        return store

    def records(self, count):
        return [VectorRecord(str(i), f'text {i}', [float(i),1.,0.], ChunkMetadata(equipment='test',chunk_index=i)) for i in range(count)]

    def test_reported_19517_records_split_at_backend_limit_and_keep_alignment(self):
        store = self.store(5461)
        records = self.records(19517)
        self.assertEqual(store.upsert(records), 19517)
        self.assertEqual(store._collection.sizes, [5461,5461,5461,3134])
        for i in (0,5460,5461,16383,19516):
            text, vector, metadata = store._collection.rows[str(i)]
            self.assertEqual((text, vector[0], metadata['chunk_index']), (f'text {i}', float(i), i))
        ids = [r.id for r in records]
        self.assertEqual(len(store.get_by_ids(ids)), 19517)
        self.assertEqual(store.delete_by_ids(ids + ['missing']), 19517)
        self.assertEqual(store._collection.rows, {})

    def test_duplicate_across_batches_rejected_before_any_write(self):
        store = self.store(2)
        records = self.records(4)
        with self.assertRaises(VectorStoreError):
            store.upsert(records + [records[0]])
        self.assertEqual(store._collection.sizes, [])

    def test_later_batch_failure_raises_and_retry_converges(self):
        store = self.store(2)
        store._collection.fail_batch = 2
        records = self.records(5)
        with self.assertRaises(VectorStoreError) as caught:
            store.upsert(records)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(len(store._collection.rows), 2)
        store._collection.fail_batch = None
        self.assertEqual(store.upsert(records), 5)
        self.assertEqual(len(store._collection.rows), 5)

    def test_invalid_backend_limit_never_writes(self):
        for limit in (0,-1,True,'5461'):
            store = self.store(limit)
            with self.assertRaises(VectorStoreError):
                store.upsert(self.records(1))
            self.assertEqual(store._collection.sizes, [])


