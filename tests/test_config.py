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

VISUAL_SECTION = """
visual:
  enabled: true
  tesseract_path: ./tools/tesseract/tesseract.exe
  languages: kor+eng
  timeout_seconds: 45
  pdf_dpi: 240
  pdf_ocr: true
  pptx_image_ocr: true
  xlsx_chart_extraction: true
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
        assert config.visual is not None
        self.assertFalse(config.visual.enabled)

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
            self.assertIsNone(config.visual)
            self.assertEqual(config.search.mode, "semantic")
            self.assertEqual(config.search.candidate_multiplier, 4)
            self.assertIsNone(config.search.reranker_model_path)

    def test_loads_hybrid_search_and_local_reranker_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            config_path = config_dir / "config.yaml"
            config_path.write_text(
                VALID_CONFIG.replace(
                    "search:\n  top_k: 3",
                    """search:
  top_k: 3
  mode: hybrid
  candidate_multiplier: 6
  semantic_weight: 0.4
  lexical_weight: 0.6
  reranker_model_path: ./models/reranker
  reranker_weight: 0.7
  reranker_device: cpu""",
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.search.mode, "hybrid")
            self.assertEqual(config.search.candidate_multiplier, 6)
            self.assertEqual(config.search.semantic_weight, 0.4)
            self.assertEqual(config.search.lexical_weight, 0.6)
            self.assertEqual(
                config.search.reranker_model_path,
                (root / "models/reranker").resolve(),
            )
            self.assertEqual(config.search.reranker_device, "cpu")

    def test_rejects_invalid_hybrid_search_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            config_path = config_dir / "config.yaml"
            config_path.write_text(
                VALID_CONFIG.replace(
                    "search:\n  top_k: 3",
                    """search:
  top_k: 3
  mode: hybrid
  semantic_weight: 0
  lexical_weight: 0""",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigError, "cannot both be zero"):
                load_config(config_path)

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

    def test_loads_optional_visual_document_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            config_path = config_dir / "config.yaml"
            config_path.write_text(
                VALID_CONFIG + "\n" + VISUAL_SECTION,
                encoding="utf-8",
            )

            config = load_config(config_path)

            assert config.visual is not None
            self.assertTrue(config.visual.enabled)
            self.assertEqual(
                config.visual.tesseract_path,
                (root / "tools/tesseract/tesseract.exe").resolve(),
            )
            self.assertEqual(config.visual.languages, "kor+eng")
            self.assertEqual(config.visual.pdf_dpi, 240)

    def test_rejects_enabled_ocr_without_tesseract_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            config_path = config_dir / "config.yaml"
            config_path.write_text(
                VALID_CONFIG
                + "\nvisual:\n"
                "  enabled: true\n"
                "  tesseract_path: null\n"
                "  languages: kor+eng\n"
                "  timeout_seconds: 60\n"
                "  pdf_dpi: 200\n"
                "  pdf_ocr: true\n"
                "  pptx_image_ocr: true\n"
                "  xlsx_chart_extraction: true\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigError, "tesseract_path"):
                load_config(config_path)

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
