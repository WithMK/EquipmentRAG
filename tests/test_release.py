from __future__ import annotations

import unittest
from pathlib import Path

from app import __version__


class ReleaseMetadataTests(unittest.TestCase):
    def test_version_file_matches_application_version(self) -> None:
        version_file = Path(__file__).resolve().parents[1] / "VERSION"

        self.assertEqual(version_file.read_text(encoding="utf-8").strip(), __version__)

    def test_gitignore_excludes_runtime_and_offline_assets(self) -> None:
        ignore_file = Path(__file__).resolve().parents[1] / ".gitignore"
        patterns = set(ignore_file.read_text(encoding="utf-8").splitlines())

        for required in (
            "/models/",
            "/wheels/",
            "/data/source/",
            "/data/documents/",
            "/data/chroma/",
            "*.gguf",
            ".env",
            "config/config.local.yaml",
        ):
            self.assertIn(required, patterns)


if __name__ == "__main__":
    unittest.main()
