from __future__ import annotations

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


class EmbeddingServiceTests(unittest.TestCase):
    def _config(self, model_path: Path) -> EmbeddingConfig:
        return EmbeddingConfig(
            model_path=model_path,
            batch_size=2,
            device="cpu",
            normalize_embeddings=True,
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

    def test_rejects_missing_model_directory_without_importing_runtime(self) -> None:
        missing = Path("missing-model-directory")
        service = LocalEmbeddingService(self._config(missing))

        with self.assertRaisesRegex(EmbeddingError, "directory not found"):
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
