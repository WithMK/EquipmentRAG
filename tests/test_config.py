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
    def test_offline_example_remains_a_valid_configuration(self) -> None:
        example = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "config.offline.example.yaml"
        )

        config = load_config(example)

        self.assertEqual(config.equipment.name, "replace-with-equipment-name")
        self.assertEqual(config.llm.provider, "llama_cpp")
        self.assertEqual(config.source.path, Path("D:/EquipmentData/source"))
        assert config.document is not None
        self.assertEqual(
            config.document.source_paths,
            (Path("D:/EquipmentData/documents"),),
        )
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
            config_dir = root / "config"
            config_dir.mkdir()
            config_path = config_dir / "config.yaml"
            config_path.write_text(VALID_CONFIG, encoding="utf-8")

            config = load_config(config_path)

            self.assertEqual(config.equipment.name, "test-equipment")
            self.assertEqual(config.source.path, (root / "data/source").resolve())
            self.assertEqual(config.source.chunk_size, 4000)
            self.assertEqual(config.source.chunk_overlap, 400)
            self.assertEqual(config.embedding.model_path, (root / "models/bge-m3").resolve())
            self.assertEqual(config.embedding.device, "cpu")
            self.assertTrue(config.embedding.normalize_embeddings)
            self.assertEqual(config.llm.provider, "llama_cpp")
            self.assertEqual(config.llm.base_url, "http://127.0.0.1:8080/v1")
            self.assertEqual(config.logging.level, "INFO")
            self.assertIsNone(config.document)

    def test_loads_optional_document_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            config_path = config_dir / "config.yaml"
            config_path.write_text(
                VALID_CONFIG + "\n" + DOCUMENT_SECTION,
                encoding="utf-8",
            )

            config = load_config(config_path)

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

    def test_rejects_unsupported_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            config_path = config_dir / "config.yaml"
            config_path.write_text(
                VALID_CONFIG.replace("provider: llama_cpp", "provider: cloud"),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigError, "llama_cpp.*ollama"):
                load_config(config_path)

    def test_rejects_missing_configuration_file(self) -> None:
        with self.assertRaisesRegex(ConfigError, "not found"):
            load_config("missing-config.yaml")

    def test_rejects_chunk_overlap_not_smaller_than_chunk_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            config_path = config_dir / "config.yaml"
            config_path.write_text(
                VALID_CONFIG.replace("chunk_overlap: 400", "chunk_overlap: 4000"),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigError, "chunk_overlap.*smaller"):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
