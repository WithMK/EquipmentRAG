from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.config import ConfigError, load_config


VALID_CONFIG = """
equipment:
  name: test-equipment
source:
  path: ./data/source
  include_extensions: [.cs]
  exclude_directories: [bin, obj]
  chunk_size: 4000
  chunk_overlap: 400
embedding:
  model_path: ./models/bge-m3
  batch_size: 8
  device: cpu
  normalize_embeddings: true
chromadb:
  path: ./data/chroma
  collection_name: equipment_code
search:
  top_k: 3
llm:
  provider: llama_cpp
  base_url: http://127.0.0.1:8080/v1/
  model: local-model
  request_timeout_seconds: 30
logging:
  level: info
  path: ./logs/test.log
""".strip()

DOCUMENT_SECTION = """
document:
  enabled: true
  source_paths: [./data/documents, ./data/reviews]
  extensions: [.docx, pdf, .md, .txt]
  exclude_directories: [.git, archive]
  chunk_size: 3000
  chunk_overlap: 300
  collection_name: document_chunks
""".strip()


class ConfigTests(unittest.TestCase):
    def _write(self, root: Path, content: str) -> Path:
        config_dir = root / "config"
        config_dir.mkdir()
        config_path = config_dir / "config.yaml"
        config_path.write_text(content, encoding="utf-8")
        return config_path

    def test_existing_configuration_defaults_to_sentence_transformers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_config(self._write(root, VALID_CONFIG))
        self.assertEqual(config.embedding.backend, "sentence_transformers")
        self.assertEqual(config.embedding.dimension, 1024)
        self.assertEqual(config.embedding.base_url, "http://127.0.0.1:8081/v1")

    def test_gguf_example_remains_a_valid_configuration(self) -> None:
        example = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "config.gguf.example.yaml"
        )
        config = load_config(example)
        self.assertEqual(config.equipment.name, "replace-with-equipment-name")
        self.assertEqual(config.embedding.backend, "llama_cpp")
        self.assertEqual(config.embedding.model_path, Path(
            "D:/OfflineAssets/models/embedding/bge-m3-Q4_0.gguf"
        ))
        self.assertEqual(config.embedding.dimension, 1024)
        self.assertEqual(config.chromadb.path, Path(
            "D:/OfflineRuntime/EquipmentRAG/chroma-gguf-q4"
        ))

    def test_offline_example_remains_a_valid_configuration(self) -> None:
        example = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "config.offline.example.yaml"
        )
        config = load_config(example)
        self.assertEqual(config.equipment.name, "replace-with-equipment-name")
        self.assertEqual(config.embedding.backend, "sentence_transformers")
        self.assertEqual(
            config.embedding.model_path,
            Path("D:/OfflineAssets/models/embedding/bge-m3"),
        )
        self.assertEqual(
            config.chromadb.path,
            Path("D:/OfflineRuntime/EquipmentRAG/chroma"),
        )

    def test_loads_and_resolves_relative_paths_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_config(self._write(root, VALID_CONFIG))
        self.assertEqual(config.equipment.name, "test-equipment")
        self.assertEqual(config.source.path, (root / "data/source").resolve())
        self.assertEqual(config.source.chunk_size, 4000)
        self.assertEqual(config.source.chunk_overlap, 400)
        self.assertEqual(
            config.embedding.model_path, (root / "models/bge-m3").resolve()
        )
        self.assertEqual(config.embedding.device, "cpu")
        self.assertTrue(config.embedding.normalize_embeddings)
        self.assertEqual(config.llm.provider, "llama_cpp")
        self.assertEqual(config.llm.base_url, "http://127.0.0.1:8080/v1")
        self.assertEqual(config.logging.level, "INFO")
        self.assertIsNone(config.document)

    def test_loads_optional_document_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_config(
                self._write(root, VALID_CONFIG + "\n" + DOCUMENT_SECTION)
            )
        self.assertIsNotNone(config.document)
        assert config.document is not None
        self.assertTrue(config.document.enabled)
        self.assertEqual(
            config.document.source_paths,
            (
                (root / "data/documents").resolve(),
                (root / "data/reviews").resolve(),
            ),
        )
        self.assertEqual(config.document.extensions[1], ".pdf")
        self.assertEqual(config.document.collection_name, "document_chunks")

    def test_loads_llama_cpp_embedding_configuration(self) -> None:
        gguf = VALID_CONFIG.replace(
            "embedding:\n  model_path: ./models/bge-m3",
            """embedding:
  backend: llama_cpp
  model_path: ./models/bge-m3-Q4_0.gguf
  base_url: http://127.0.0.1:8081/v1/
  model: bge-m3-q4
  dimension: 1024
  request_timeout_seconds: 45""",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_config(self._write(root, gguf))
        self.assertEqual(config.embedding.backend, "llama_cpp")
        self.assertEqual(config.embedding.model_path.name, "bge-m3-Q4_0.gguf")
        self.assertEqual(config.embedding.base_url, "http://127.0.0.1:8081/v1")
        self.assertEqual(config.embedding.request_timeout_seconds, 45)

    def test_rejects_unknown_embedding_backend(self) -> None:
        invalid = VALID_CONFIG.replace(
            "embedding:\n", "embedding:\n  backend: unsupported\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ConfigError, "embedding.backend"):
                load_config(self._write(root, invalid))

    def test_rejects_unsupported_llm_provider(self) -> None:
        invalid = VALID_CONFIG.replace("provider: llama_cpp", "provider: cloud")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ConfigError, "llama_cpp.*ollama"):
                load_config(self._write(root, invalid))

    def test_rejects_missing_configuration_file(self) -> None:
        with self.assertRaisesRegex(ConfigError, "not found"):
            load_config("missing-config.yaml")

    def test_rejects_chunk_overlap_not_smaller_than_chunk_size(self) -> None:
        invalid = VALID_CONFIG.replace("chunk_overlap: 400", "chunk_overlap: 4000")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ConfigError, "chunk_overlap.*smaller"):
                load_config(self._write(root, invalid))


if __name__ == "__main__":
    unittest.main()
