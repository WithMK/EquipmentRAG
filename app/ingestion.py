"""Durable, opt-in document ingestion. One worker owns one ingestion directory."""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import logging
import re
import sqlite3
import tempfile
import threading
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.document_indexer import IncrementalDocumentIndexer
from app.models.document_models import DocumentSourceFile

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_BODY_BYTES = 14 * 1024 * 1024
EXTENSIONS = {'.md', '.txt', '.pdf', '.docx', '.pptx', '.xlsx'}


class IngestionError(ValueError):
    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


def _now():
    return datetime.now(timezone.utc).isoformat()


def _text(value, name, limit=200):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or any(ord(c) < 32 for c in value):
        raise IngestionError(f'Invalid {name}')
    return value.strip()


def validate_upload(payload):
    if not isinstance(payload, dict) or set(payload) - {'filename', 'content_base64', 'project', 'equipment', 'version', 'source_key'}:
        raise IngestionError('Invalid upload fields')
    filename = _text(payload.get('filename'), 'filename', 180)
    if re.search(r'[\\/:*?"<>|]', filename) or filename.endswith(('.', ' ')) or filename.startswith('.'):
        raise IngestionError('filename must be a plain file name')
    if filename.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(10)), *(f'LPT{i}' for i in range(10))}:
        raise IngestionError('Reserved file name')
    if Path(filename).suffix.lower() not in EXTENSIONS:
        raise IngestionError('Supported formats: MD, TXT, PDF, DOCX, PPTX, XLSX')
    encoded = payload.get('content_base64')
    if not isinstance(encoded, str) or len(encoded) > MAX_BODY_BYTES:
        raise IngestionError('File exceeds 10 MiB', 413)
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise IngestionError('Invalid base64 content') from exc
    if not content or len(content) > MAX_FILE_BYTES:
        raise IngestionError('File must contain 1 byte to 10 MiB', 413)
    if Path(filename).suffix.lower() in {'.docx', '.pptx', '.xlsx'}:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                entries = archive.infolist()
                if len(entries) > 10000 or sum(x.file_size for x in entries) > 100 * 1024 * 1024:
                    raise IngestionError('Expanded Office document is too large', 413)
        except zipfile.BadZipFile as exc:
            raise IngestionError('Invalid Office document') from exc
    version = payload.get('version', 0)
    if type(version) is not int or version < 0:
        raise IngestionError('Invalid version')
    return filename, content, _text(payload.get('project'), 'project'), _text(payload.get('equipment'), 'equipment'), version


class IngestionStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'ingestion.db'
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('PRAGMA user_version').fetchone()[0] not in (0, 1, 2):
                raise IngestionError('Unsupported ingestion schema')
            db.execute('CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, namespace TEXT NOT NULL, data TEXT NOT NULL, content BLOB NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS document_namespace ON documents(namespace)')
            db.execute('CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY, document_id TEXT NOT NULL, action TEXT NOT NULL, version INTEGER NOT NULL, created_at TEXT NOT NULL)')
            for identity, raw in db.execute('SELECT id,data FROM documents').fetchall():
                data = json.loads(raw)
                data.setdefault('desired_state', 'active')
                data.setdefault('operation', 'index')
                data.setdefault('source_key', '')
                self.save(db, data)
            db.execute('PRAGMA user_version=2')

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    @staticmethod
    def namespace(value):
        if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}', value):
            raise IngestionError('Invalid namespace')
        return value

    def list(self, namespace):
        self.namespace(namespace)
        with closing(self.connect()) as db:
            return [self.public(json.loads(row[0])) for row in db.execute('SELECT data FROM documents WHERE namespace=? ORDER BY rowid DESC', (namespace,))]

    @staticmethod
    def public(data):
        return {k: v for k, v in data.items() if k not in {'chunk_ids', 'pending_ids'}}

    def summary(self, namespace):
        documents = self.list(namespace)
        counts = {status: 0 for status in ('queued', 'indexing', 'ready', 'failed', 'excluded', 'deleted')}
        for item in documents:
            counts[item['status']] = counts.get(item['status'], 0) + 1
        required = [item for item in documents if item['desired_state'] == 'active' and
                    (item['status'] in {'queued', 'indexing', 'failed'} or item['indexed_sha256'] != item['sha256'])]
        return dict(namespace=namespace, observed_at=_now(), scope='managed_documents',
                    counts=counts, registered=len(documents) - counts['deleted'],
                    needs_indexing=len(required), chunks=sum(item['chunks'] for item in documents),
                    folder_scan_included=False)

    def search_blocked(self, filters):
        """Fail closed for scopes whose two indexes may contain a partial write."""
        with closing(self.connect()) as db:
            for row in db.execute('SELECT data FROM documents'):
                data = json.loads(row[0])
                if not data['pending_ids']:
                    continue
                if any(getattr(filters, key, None) and getattr(filters, key) != data[key] for key in ('project', 'equipment')):
                    continue
                return True
        return False

    def submit(self, namespace, payload):
        self.namespace(namespace)
        filename, content, project, equipment, version = validate_upload(payload)
        source_key = payload.get('source_key', '')
        if not isinstance(source_key, str) or (source_key and not re.fullmatch(r'watch:[a-zA-Z0-9_-]{1,64}:[a-f0-9]{64}', source_key)):
            raise IngestionError('Invalid source_key')
        digest = hashlib.sha256(content).hexdigest()
        identity = hashlib.sha256((namespace + '\0' + (source_key or filename.casefold())).encode()).hexdigest()
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT data FROM documents WHERE id=?', (identity,)).fetchone()
            old = json.loads(row[0]) if row else None
            if old:
                if old['project'] != project or old['equipment'] != equipment:
                    raise IngestionError('Document search binding cannot change', 409)
                if old['status'] == 'deleted' and source_key:
                    raise IngestionError('Deleted watched files require a new source identity', 409)
                if old['sha256'] == digest and old['status'] != 'deleted':
                    return self.public(old)
                if old['status'] in {'queued', 'indexing'} or version != old['version']:
                    raise IngestionError('Document changed or is busy; refresh before replacing', 409)
            elif version:
                raise IngestionError('Document no longer exists', 409)
            desired = old['desired_state'] if old and old['desired_state'] != 'deleted' else 'active'
            if old and old['desired_state'] == 'deleted' and old['status'] != 'deleted':
                raise IngestionError('Finish or retry deletion before uploading', 409)
            data = dict(id=identity, namespace=namespace, filename=filename, project=project, equipment=equipment,
                        source_key=source_key, desired_state=desired, operation='index' if desired == 'active' else 'exclude',
                        sha256=digest, size=len(content), status='queued', error='', version=(old['version'] + 1 if old else 1),
                        chunks=0, indexed_sha256=old['indexed_sha256'] if old else '', updated_at=_now(),
                        chunk_ids=old['chunk_ids'] if old else [], pending_ids=old['pending_ids'] if old else [])
            db.execute('INSERT OR REPLACE INTO documents VALUES(?,?,?,?)', (identity, namespace, json.dumps(data), content))
            self.event(db, data, 'upload')
        return self.public(data)

    def action(self, namespace, identity, version, action):
        self.namespace(namespace)
        if action not in {'exclude', 'restore', 'delete', 'reindex'}:
            raise IngestionError('Unknown document action')
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT data FROM documents WHERE id=? AND namespace=?', (identity, namespace)).fetchone()
            if not row:
                raise IngestionError('Document not found', 404)
            data = json.loads(row[0])
            if type(version) is not int or version != data['version'] or data['status'] in {'queued', 'indexing', 'deleted'}:
                raise IngestionError('Document changed, is busy, or has been deleted', 409)
            if data['desired_state'] == 'deleted' and action != 'delete':
                raise IngestionError('Retry the pending deletion', 409)
            if action == 'restore' and data['desired_state'] != 'excluded':
                raise IngestionError('Only excluded documents can be restored', 409)
            if action == 'reindex' and data['desired_state'] != 'active':
                raise IngestionError('Restore excluded documents before reindexing', 409)
            desired = {'exclude':'excluded', 'delete':'deleted'}.get(action, 'active')
            data.update(desired_state=desired, operation=action if action in {'exclude', 'delete'} else 'index',
                        status='queued', error='', version=data['version'] + 1, updated_at=_now())
            if desired != 'active':
                # Block the affected API search scope immediately, until removal completes.
                data['pending_ids'] = sorted(set(data['pending_ids']) | set(data['chunk_ids']))
            self.save(db, data)
            self.event(db, data, action)
        return self.public(data)

    def retry(self, namespace, identity, version):
        self.namespace(namespace)
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT data FROM documents WHERE id=? AND namespace=?', (identity, namespace)).fetchone()
            if not row:
                raise IngestionError('Document not found', 404)
            data = json.loads(row[0])
            if type(version) is not int or data['version'] != version or data['status'] != 'failed':
                raise IngestionError('Only the current failed version can be retried', 409)
            data.update(status='queued', error='', updated_at=_now())
            self.save(db, data)
            self.event(db, data, 'retry')
        return self.public(data)

    @staticmethod
    def event(db, data, action):
        db.execute('INSERT INTO events(document_id,action,version,created_at) VALUES(?,?,?,?)',
                   (data['id'], action, data['version'], _now()))

    @staticmethod
    def save(db, data):
        db.execute('UPDATE documents SET data=? WHERE id=?', (json.dumps(data), data['id']))


class IngestionWorker:
    def __init__(self, config, root, *, indexer=None, request_lock=None):
        self.store = IngestionStore(root)
        self.config = config
        # An existing journal must never report ready against another index/model.
        from dataclasses import asdict
        binding = json.dumps({'chroma': asdict(config.chromadb), 'document': asdict(config.document),
                              'embedding': asdict(config.embedding), 'equipment': config.equipment.name,
                              'lexical_backend': config.search.lexical_backend, 'lexical_path': config.search.lexical_path,
                              'visual': asdict(config.visual) if config.visual else None},
                             default=str, sort_keys=True)
        with closing(self.store.connect()) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS binding (value TEXT NOT NULL)')
            old = db.execute('SELECT value FROM binding').fetchone()
            if old and old[0] != binding:
                raise IngestionError('Ingestion configuration changed; restore the original index/model configuration')
            if not old:
                db.execute('INSERT INTO binding VALUES(?)', (binding,))
        self.indexer = indexer
        self.request_lock = request_lock or threading.Lock()
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread = None
        self._lock_file = None

    def start(self):
        # OS locks release on crashes, unlike an exclusive-create sentinel.
        self._lock_file = (self.store.root / 'worker.lock').open('a+b')
        try:
            if __import__('os').name == 'nt':
                import msvcrt
                self._lock_file.write(b'0'); self._lock_file.flush(); self._lock_file.seek(0)
                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_file.close(); self._lock_file = None
            raise IngestionError('An ingestion worker already owns this directory', 409)
        with closing(self.store.connect()) as db, db:
            for row in db.execute('SELECT data FROM documents').fetchall():
                data = json.loads(row[0])
                if data['status'] == 'indexing':
                    data.update(status='queued', updated_at=_now())
                    self.store.save(db, data)
        self.thread = threading.Thread(target=self._run, name='document-ingestion', daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set(); self.wake.set()
        if self.thread:
            self.thread.join()
        if self.indexer is not None:
            self.indexer.close()
        if self._lock_file:
            self._lock_file.close(); self._lock_file = None

    def _run(self):
        while not self.stop.is_set():
            try:
                if self.process_one():
                    continue
            except Exception:
                logging.exception('Unable to poll ingestion queue')
            self.wake.wait(1); self.wake.clear()

    def process_one(self):
        with closing(self.store.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT data,content FROM documents WHERE json_extract(data,'$.status')='queued' ORDER BY rowid LIMIT 1").fetchone()
            if not row:
                return False
            data, content = json.loads(row[0]), row[1]
            data.update(status='indexing', updated_at=_now())
            self.store.save(db, data)
        try:
            if self.indexer is None:
                self.indexer = IncrementalDocumentIndexer(self.config)
            if data['operation'] in {'exclude', 'delete'}:
                self._remove(data)
                return True
            with tempfile.TemporaryDirectory(dir=self.store.root) as directory:
                path = Path(directory) / data['filename']
                path.write_bytes(content)
                source = DocumentSourceFile(path, f"uploads/{data['namespace']}/{data['filename']}", data['project'],
                    data['sha256'], data['sha256'], data['updated_at'],
                    {'document_id': 'upload-' + data['id'], 'project': data['project'], 'equipment': data['equipment']})
                records = self.indexer.prepare_records(source)
            if not records:
                raise IngestionError('No extractable text; check the file or OCR configuration')
            # Journal every possibly written ID before either index is touched.
            data['pending_ids'] = sorted(set(data['pending_ids']) | {r.id for r in records})
            with closing(self.store.connect()) as db, db:
                self.store.save(db, data)
            with self.request_lock:
                self.indexer.replace_records(records, set(data['chunk_ids']) | set(data['pending_ids']))
            data.update(status='ready', error='', chunks=len(records), chunk_ids=[r.id for r in records],
                        pending_ids=[], indexed_sha256=data['sha256'], updated_at=_now())
        except Exception as exc:
            logging.exception('Document ingestion failed: %s', data['id'])
            data.update(status='failed', error=str(exc) if isinstance(exc, IngestionError) else 'Indexing failed; check the RAG server log and retry', updated_at=_now())
        with closing(self.store.connect()) as db, db:
            self.store.save(db, data)
        return True

    def _remove(self, data):
        data['pending_ids'] = sorted(set(data['chunk_ids']) | set(data['pending_ids']))
        with closing(self.store.connect()) as db, db:
            self.store.save(db, data)
        with self.request_lock:
            self.indexer.delete_records(data['pending_ids'])
            data.update(status='deleted' if data['operation'] == 'delete' else 'excluded', error='',
                        chunks=0, chunk_ids=[], pending_ids=[], indexed_sha256='', updated_at=_now())
            with closing(self.store.connect()) as db, db:
                if data['status'] == 'deleted':
                    db.execute("UPDATE documents SET content=X'' WHERE id=?", (data['id'],))
                self.store.save(db, data)
                self.store.event(db, data, data['status'])
