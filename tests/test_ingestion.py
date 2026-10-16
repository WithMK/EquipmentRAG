import base64
import http.client
import json
import tempfile
import threading
import time
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from app.api import ApiApplication, ApiRequestError, EquipmentRagHttpServer
from app.document_indexer import IncrementalDocumentIndexer
from app.ingestion import IngestionError, IngestionStore, IngestionWorker, validate_upload
from app.retrieval.sqlite_fts import SQLiteFtsStore
from app.retrieval.document_retriever import DocumentSearchFilters
from test_api import _config, _FakeService
from test_document_indexer import FakeEmbedding, FakeStore


def payload(text='E-024 vacuum interlock', **kwargs):
    return dict(filename='매뉴얼.md', content_base64=base64.b64encode(text.encode()).decode(), project='alpha', equipment='press-line-01', **kwargs)


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        config = _config(self.root)
        self.config = replace(config, document=replace(config.document, extensions=('.md',), chunk_size=150, chunk_overlap=10))
        self.vector = FakeStore()
        self.embedding = FakeEmbedding()
        self.fts = SQLiteFtsStore(self.root / 'lexical.db')
        self.addCleanup(self.fts.close)
        self.indexer = IncrementalDocumentIndexer(self.config, embedding=self.embedding, vector_store=self.vector, lexical_store=self.fts)
        self.worker = IngestionWorker(self.config, self.root / 'uploads', indexer=self.indexer)
        self.store = self.worker.store
        self.addCleanup(self.worker.close)

    def test_upload_duplicate_replace_and_independent_project_search(self):
        first = self.store.submit('alpha', payload('E-024 oldtoken ' * 100))
        self.assertTrue(self.worker.process_one())
        ready = self.store.list('alpha')[0]
        self.assertEqual(ready['status'], 'ready')
        old_count = len(self.vector.records)
        self.assertGreater(old_count, 1)
        self.assertEqual(self.store.submit('alpha', payload('E-024 oldtoken ' * 100)), ready)
        self.assertFalse(self.worker.process_one())
        with self.assertRaises(IngestionError):
            self.store.submit('alpha', payload('replacement'))
        self.store.submit('beta', {**payload('otherproject token'), 'project': 'beta'})
        self.worker.process_one()
        self.store.submit('alpha', payload('E-024 newtoken', version=first['version']))
        self.worker.process_one()
        self.assertEqual(len(self.vector.records), 2)
        self.assertEqual(len(self.fts.search('newtoken', 5, filters={'project': 'alpha'})), 1)
        self.assertEqual(self.fts.search('oldtoken', 5), [])
        self.assertEqual(self.fts.search('newtoken', 5, filters={'project': 'beta'}), [])
        self.assertEqual(self.vector.delete_filters, [])

    def test_namespace_summary_tracks_indexing_lifecycle_and_authentication(self):
        first = self.store.submit('alpha', payload())
        self.store.submit('beta', payload())
        summary = self.store.summary('alpha')
        self.assertEqual(summary['counts']['queued'], 1)
        self.assertEqual(summary['needs_indexing'], 1)
        self.assertFalse(summary['folder_scan_included'])
        self.worker.process_one()
        summary = self.store.summary('alpha')
        self.assertEqual(summary['counts']['ready'], 1)
        self.assertEqual(summary['needs_indexing'], 0)
        self.store.action('alpha', first['id'], 1, 'exclude')
        self.assertEqual(self.store.summary('alpha')['needs_indexing'], 0)
        key = 'test-ingestion-key-0123456789'
        api = ApiApplication(self.config, ingestion_root=self.store.root, ingestion_key=key,
                             ingestion_indexer=self.indexer, service_factory=lambda *args: _FakeService())
        with self.assertRaises(ApiRequestError):
            api.dispatch('GET', '/v1/ingestion/alpha/summary')
        status, result = api.dispatch('GET', '/v1/ingestion/alpha/summary', authorization='Bearer '+key)
        self.assertEqual(status, 200)
        self.assertEqual(result['registered'], 1)
        self.assertEqual(api.dispatch('GET', '/v1/ingestion/empty/summary', authorization='Bearer '+key)[1]['registered'], 0)

    def test_prepare_failure_retains_old_content_and_explicit_retry(self):
        self.store.submit('alpha', payload())
        self.worker.process_one()
        old = dict(self.vector.records)
        self.store.submit('alpha', payload('changed', version=1))
        self.embedding.fail = True
        with self.assertLogs(level='ERROR'):
            self.worker.process_one()
        failed = self.store.list('alpha')[0]
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(old, self.vector.records)
        self.assertFalse(self.worker.process_one())
        with self.assertRaises(IngestionError):
            self.store.retry('beta', failed['id'], 2)
        self.embedding.fail = False
        self.store.retry('alpha', failed['id'], 2)
        self.worker.process_one()
        self.assertEqual(self.store.list('alpha')[0]['status'], 'ready')

    def test_partial_index_failure_journal_converges_on_shorter_replacement(self):
        original = self.fts.upsert
        def fail_after_write(records):
            original(records)
            raise RuntimeError('simulated disk error')
        self.fts.upsert = fail_after_write
        self.store.submit('alpha', payload('longcontent ' * 100))
        with self.assertLogs(level='ERROR'):
            self.worker.process_one()
        self.assertGreater(len(self.vector.records), 1)
        self.assertTrue(self.store.search_blocked(DocumentSearchFilters(project='alpha')))
        self.assertFalse(self.store.search_blocked(DocumentSearchFilters(project='beta')))
        app = ApiApplication(self.config, service_factory=lambda *_: _FakeService())
        app.ingestion = self.worker
        with self.assertRaises(ApiRequestError) as error:
            app.dispatch('POST', '/v1/retrieve', {'query':'longcontent', 'source_type':'document', 'filters':{'document':{'project':'alpha'}}})
        self.assertEqual(error.exception.status, 503)
        self.fts.upsert = original
        self.store.submit('alpha', payload('shortcontent', version=1))
        self.worker.process_one()
        self.assertEqual(len(self.vector.records), 1)
        self.assertEqual(self.fts.count(), 1)
        self.assertEqual(self.fts.search('longcontent', 5), [])
        self.assertFalse(self.store.search_blocked(DocumentSearchFilters(project='alpha')))

    def test_restart_recovers_interrupted_and_releases_windows_handles(self):
        self.store.submit('alpha', payload())
        with closing(self.store.connect()) as db, db:
            data = json.loads(db.execute('SELECT data FROM documents').fetchone()[0])
            data['status'] = 'indexing'
            self.store.save(db, data)
        self.worker.start()
        second = IngestionWorker(self.config, self.store.root, indexer=self.indexer)
        with self.assertRaises(IngestionError):
            second.start()
        for _ in range(100):
            if self.store.list('alpha')[0]['status'] == 'ready':
                break
            time.sleep(.01)
        self.assertEqual(self.store.list('alpha')[0]['status'], 'ready')
        self.worker.close()
        # Real unlink verifies explicit SQLite connection cleanup on Windows.
        self.store.path.unlink()
        (self.root / 'lexical.db').unlink()

    def test_validation_empty_text_and_binding_config(self):
        for value in [{**payload(), 'filename':'../secret.md'}, {**payload(), 'filename':'CON.txt'},
                      {**payload(), 'filename':'x.exe'}, {**payload(), 'content_base64':'!'},
                      {**payload(), 'project':''}, {**payload(), 'filename':'x.docx'}]:
            with self.assertRaises(IngestionError):
                validate_upload(value)
        self.store.submit('alpha', payload('   '))
        with self.assertLogs(level='ERROR'):
            self.worker.process_one()
        self.assertEqual(self.store.list('alpha')[0]['status'], 'failed')
        with self.assertRaises(IngestionError):
            IngestionWorker(replace(self.config, chromadb=replace(self.config.chromadb, path=self.root / 'other')), self.store.root)

    def test_cli_initial_index_preserves_managed_documents(self):
        self.store.submit('alpha', payload())
        self.worker.process_one()
        documents = self.root / 'documents'
        documents.mkdir()
        (documents / 'local.md').write_text('Local document', encoding='utf-8')
        self.indexer.run()
        self.assertEqual(len(self.vector.records), 2)
        self.assertEqual(self.fts.count(), 2)

    def test_code_reindex_preserves_document_fts(self):
        from app.indexer import IncrementalSourceIndexer
        from test_indexer import FakeVectorStore
        self.store.submit('alpha', payload())
        self.worker.process_one()
        source = self.root / 'source'
        source.mkdir()
        (source / 'Test.cs').write_text('class Test { public void Run() { System.Console.WriteLine("test"); } }', encoding='utf-8')
        indexer = IncrementalSourceIndexer(self.config, embedding=self.embedding, vector_store=FakeVectorStore(), lexical_store=self.fts)
        indexer.run(full_reindex=True)
        self.assertEqual(len(self.fts.search('E-024', 5, filters={'source_type':'document'})), 1)

    def test_real_http_auth_upload_and_status(self):
        key = 'test-ingestion-key-0123456789'
        app = ApiApplication(self.config, ingestion_root=self.root / 'http', ingestion_key=key, ingestion_indexer=self.indexer)
        server = EquipmentRagHttpServer(('127.0.0.1', 0), app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(app.ingestion.close)
        app.ingestion.start()
        conn = http.client.HTTPConnection(*server.server_address, timeout=5)
        self.addCleanup(conn.close)
        path = '/v1/ingestion/alpha/documents'
        conn.request('POST', path, json.dumps(payload()), {'Content-Type': 'application/json'})
        response = conn.getresponse(); response.read()
        self.assertEqual(response.status, 401)
        headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}
        conn.request('POST', path, json.dumps(payload()), headers)
        response = conn.getresponse(); response.read()
        self.assertEqual(response.status, 202)
        for _ in range(100):
            conn.request('GET', path, headers=headers)
            response = conn.getresponse(); data = json.loads(response.read())
            if data['documents'][0]['status'] == 'ready':
                break
            time.sleep(.01)
        self.assertEqual(data['documents'][0]['status'], 'ready')
        self.assertNotIn('content_base64', str(data))

    def test_exclude_restore_reindex_delete_and_original_retention(self):
        doc = self.store.submit('alpha', payload())
        self.worker.process_one()
        excluded = self.store.action('alpha', doc['id'], 1, 'exclude')
        self.assertTrue(self.store.search_blocked(DocumentSearchFilters(project='alpha')))
        self.worker.process_one()
        self.assertEqual(self.store.list('alpha')[0]['status'], 'excluded')
        self.assertEqual(self.fts.count(), 0)
        self.assertFalse(self.vector.records)
        with closing(self.store.connect()) as db:
            self.assertTrue(db.execute('SELECT content FROM documents').fetchone()[0])
        with self.assertRaises(IngestionError):
            self.store.action('alpha', doc['id'], excluded['version'], 'reindex')
        restored = self.store.action('alpha', doc['id'], excluded['version'], 'restore')
        self.worker.process_one()
        self.assertEqual(self.fts.count(), 1)
        self.assertFalse(self.store.search_blocked(DocumentSearchFilters(project='alpha')))
        call_count = len(self.embedding.calls)
        reindexed = self.store.action('alpha', doc['id'], restored['version'], 'reindex')
        self.worker.process_one()
        self.assertEqual(len(self.embedding.calls), call_count + 1)
        removed = self.store.action('alpha', doc['id'], reindexed['version'], 'delete')
        self.worker.process_one()
        self.assertEqual(self.store.list('alpha')[0]['status'], 'deleted')
        self.assertEqual(self.fts.count(), 0)
        with closing(self.store.connect()) as db:
            self.assertEqual(db.execute('SELECT content FROM documents').fetchone()[0], b'')
        with self.assertRaises(IngestionError):
            self.store.action('alpha', doc['id'], removed['version'], 'restore')
        # Manual explicit re-upload with the current version creates fresh stored content.
        self.store.submit('alpha', payload(version=removed['version']))
        self.worker.process_one()
        self.assertEqual(self.store.list('alpha')[0]['status'], 'ready')

    def test_partial_deletion_retry_and_project_ownership(self):
        doc = self.store.submit('alpha', payload())
        self.worker.process_one()
        with self.assertRaises(IngestionError):
            self.store.action('beta', doc['id'], 1, 'delete')
        delete = self.store.action('alpha', doc['id'], 1, 'delete')
        with self.assertRaises(IngestionError):
            self.store.action('alpha', doc['id'], 1, 'exclude')
        original = self.fts.delete_by_ids
        def fail(ids):
            raise RuntimeError('synthetic FTS failure')
        self.fts.delete_by_ids = fail
        with self.assertLogs(level='ERROR'):
            self.worker.process_one()
        self.assertEqual(self.store.list('alpha')[0]['status'], 'failed')
        self.assertTrue(self.store.search_blocked(DocumentSearchFilters(project='alpha')))
        with closing(self.store.connect()) as db:
            self.assertTrue(db.execute('SELECT content FROM documents').fetchone()[0])
        self.fts.delete_by_ids = original
        self.store.retry('alpha', doc['id'], delete['version'])
        self.worker.process_one()
        self.assertEqual(self.fts.count(), 0)
        self.assertFalse(self.store.search_blocked(DocumentSearchFilters(project='alpha')))

    def test_watched_identity_tombstone_and_exclusion_survive_updates(self):
        first = self.store.submit('alpha', payload(source_key='watch:folder:' + 'a'*64))
        self.worker.process_one()
        second = self.store.submit('alpha', payload(source_key='watch:folder:' + 'b'*64))
        self.worker.process_one()
        self.assertNotEqual(first['id'], second['id'])
        self.store.action('alpha', first['id'], 1, 'exclude')
        self.worker.process_one()
        calls = len(self.embedding.calls)
        self.store.submit('alpha', payload('new content', source_key='watch:folder:' + 'a'*64, version=2))
        self.worker.process_one()
        excluded = next(d for d in self.store.list('alpha') if d['id'] == first['id'])
        self.assertEqual(excluded['status'], 'excluded')
        self.assertEqual(calls, len(self.embedding.calls))
        self.store.action('alpha', first['id'], 3, 'delete')
        self.worker.process_one()
        with self.assertRaises(IngestionError):
            self.store.submit('alpha', payload('new content', source_key='watch:folder:' + 'a'*64, version=4))
        self.assertEqual(self.fts.count(), 1)

    def test_schema_v1_migration_and_interrupted_exclusion(self):
        doc = self.store.submit('alpha', payload())
        self.worker.process_one()
        with closing(self.store.connect()) as db, db:
            data = json.loads(db.execute('SELECT data FROM documents').fetchone()[0])
            for key in ('operation', 'desired_state', 'source_key'):
                data.pop(key)
            self.store.save(db, data)
            db.execute('PRAGMA user_version=1')
        migrated = IngestionStore(self.store.root)
        self.assertEqual(migrated.list('alpha')[0]['desired_state'], 'active')
        migrated.action('alpha', doc['id'], 1, 'exclude')
        with closing(migrated.connect()) as db, db:
            data = json.loads(db.execute('SELECT data FROM documents').fetchone()[0])
            data['status'] = 'indexing'
            migrated.save(db, data)
        self.worker.start()
        for _ in range(100):
            if migrated.list('alpha')[0]['status'] == 'excluded':
                break
            time.sleep(.01)
        self.assertEqual(migrated.list('alpha')[0]['status'], 'excluded')
        self.assertEqual(self.fts.count(), 0)

    def test_lifecycle_api_auth_version_and_scope(self):
        app = ApiApplication(self.config, service_factory=lambda *_: _FakeService())
        app.ingestion, app.ingestion_key = self.worker, 'key'
        doc = self.store.submit('alpha', payload())
        self.worker.process_one()
        path = '/v1/ingestion/alpha/documents/' + doc['id'] + '/exclude'
        with self.assertRaises(ApiRequestError) as error:
            app.dispatch('POST', path, {'version':1})
        self.assertEqual(error.exception.status, 401)
        status, data = app.dispatch('POST', path, {'version':1}, 'Bearer key')
        self.assertEqual(status, 202)
        with self.assertRaises(ApiRequestError) as error:
            app.dispatch('POST', '/v1/retrieve', {'query':'E-024', 'source_type':'document'})
        self.assertEqual(error.exception.status, 503)
        self.worker.process_one()
        self.assertEqual(self.store.list('alpha')[0]['status'], 'excluded')


if __name__ == '__main__':
    unittest.main()
