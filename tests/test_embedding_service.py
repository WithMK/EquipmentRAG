from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import EmbeddingConfig
from app.embedding.embedding_service import EmbeddingError, LocalEmbeddingService


class FakeArray:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        return self._rows


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], dict[str, object]]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts: list[str], **kwargs: object) -> FakeArray:
        self.calls.append(("encode", texts, kwargs))
        return FakeArray([[1, 2, 3] for _ in texts])

    def encode_document(self, texts: list[str], **kwargs: object) -> FakeArray:
        self.calls.append(("document", texts, kwargs))
        return FakeArray([[4, 5, 6] for _ in texts])

    def encode_query(self, texts: list[str], **kwargs: object) -> FakeArray:
        self.calls.append(("query", texts, kwargs))
        return FakeArray([[7, 8, 9] for _ in texts])


class FakeSentenceTransformerFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.model = FakeModel()

    def __call__(self, model_path: str, **kwargs: object) -> FakeModel:
        self.calls.append((model_path, kwargs))
        return self.model


class RecordingTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], dict[str, str], float]] = []

    def __call__(
        self, url: str, body: bytes, headers: dict[str, str], timeout: float
    ) -> bytes:
        payload = json.loads(body.decode("utf-8"))
        self.calls.append((url, payload, headers, timeout))
        data = [
            {"index": index, "embedding": [index + 1, 2, 3, 4]}
            for index, _ in enumerate(payload["input"])
        ]
        return json.dumps({"object": "list", "data": data}).encode("utf-8")


class EmbeddingServiceTests(unittest.TestCase):
    def _config(self, model_path: Path) -> EmbeddingConfig:
        return EmbeddingConfig(
            model_path=model_path,
            batch_size=2,
            device="cpu",
            normalize_embeddings=True,
        )

    def _gguf_config(self, model_path: Path) -> EmbeddingConfig:
        return EmbeddingConfig(
            model_path=model_path,
            batch_size=2,
            device=None,
            normalize_embeddings=True,
            backend="llama_cpp",
            base_url="http://127.0.0.1:8081/v1/",
            model="bge-m3-q4",
            dimension=4,
            request_timeout_seconds=45,
        )

    def test_loads_local_model_once_and_uses_task_specific_encoders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "bge-m3"
            model_path.mkdir()
            factory = FakeSentenceTransformerFactory()
            fake_module = types.SimpleNamespace(SentenceTransformer=factory)

            with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
                service = LocalEmbeddingService(self._config(model_path))
                documents = service.embed_documents(["class A", "class B"])
                query = service.embed_query("A 클래스")
                generic = service.embed_texts(["plain text"])

            self.assertTrue(service.is_loaded)
            self.assertEqual(service.dimension, 3)
            self.assertEqual(documents, [[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
            self.assertEqual(query, [7.0, 8.0, 9.0])
            self.assertEqual(generic, [[1.0, 2.0, 3.0]])
            self.assertEqual(
                factory.calls,
                [(str(model_path.resolve()), {"local_files_only": True, "device": "cpu"})],
            )
            self.assertEqual(
                [call[0] for call in factory.model.calls],
                ["document", "query", "encode"],
            )
            for _, _, options in factory.model.calls:
                self.assertEqual(options["batch_size"], 2)
                self.assertFalse(options["show_progress_bar"])
                self.assertTrue(options["convert_to_numpy"])
                self.assertTrue(options["normalize_embeddings"])

    def test_llama_cpp_backend_batches_openai_embedding_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "bge-m3-Q4_0.gguf"
            model_path.write_bytes(b"GGUF-test-placeholder")
            transport = RecordingTransport()
            service = LocalEmbeddingService(
                self._gguf_config(model_path), transport=transport
            )

            documents = service.embed_documents(["a", "b", "c"])
            query = service.embed_query("query")

            self.assertTrue(service.is_loaded)
            self.assertEqual(service.dimension, 4)
            self.assertEqual(len(documents), 3)
            self.assertEqual(query, [1.0, 2.0, 3.0, 4.0])
            self.assertEqual(len(transport.calls), 3)
            first_url, first_payload, first_headers, first_timeout = transport.calls[0]
            self.assertEqual(first_url, "http://127.0.0.1:8081/v1/embeddings")
            self.assertEqual(
                first_payload,
                {"model": "bge-m3-q4", "input": ["a", "b"]},
            )
            self.assertEqual(first_headers["Content-Type"], "application/json")
            self.assertEqual(first_timeout, 45.0)

    def test_llama_cpp_backend_rejects_wrong_vector_dimension(self) -> None:
        def wrong_dimension(
            url: str, body: bytes, headers: dict[str, str], timeout: float
        ) -> bytes:
            return json.dumps(
                {"data": [{"index": 0, "embedding": [1, 2, 3]}]}
            ).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "bge-m3-Q4_0.gguf"
            model_path.write_bytes(b"GGUF-test-placeholder")
            service = LocalEmbeddingService(
                self._gguf_config(model_path), transport=wrong_dimension
            )

            with self.assertRaisesRegex(EmbeddingError, "vector shape"):
                service.embed_query("query")

    def test_rejects_missing_model_directory_without_importing_runtime(self) -> None:
        missing = Path("missing-model-directory")
        service = LocalEmbeddingService(self._config(missing))

        with self.assertRaisesRegex(EmbeddingError, "directory not found"):
            service.load()

        self.assertFalse(service.is_loaded)

    def test_rejects_missing_gguf_model_file(self) -> None:
        service = LocalEmbeddingService(self._gguf_config(Path("missing-model.gguf")))

        with self.assertRaisesRegex(EmbeddingError, "GGUF.*file not found"):
            service.load()

        self.assertFalse(service.is_loaded)

    def test_rejects_empty_or_blank_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = LocalEmbeddingService(self._config(Path(temp_dir)))

            with self.assertRaisesRegex(EmbeddingError, "non-empty strings"):
                service.embed_texts([])
            with self.assertRaisesRegex(EmbeddingError, "non-empty strings"):
                service.embed_query("   ")


if __name__ == "__main__":
    unittest.main()
