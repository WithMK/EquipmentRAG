from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from deploy.prepare_offline_assets import AssetBundleError, prepare_assets
from deploy.reassemble_offline_assets import ReassemblyError, reassemble


class OfflineAssetBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="equipment-rag-assets-"))
        self.project = self.root / "project"
        (self.project / "models" / "embedding").mkdir(parents=True)
        (self.project / "wheels").mkdir()
        (self.project / "models" / "embedding" / "model.bin").write_bytes(
            bytes(range(256)) * 3000
        )
        (self.project / "wheels" / "package-1.0-py3-none-any.whl").write_bytes(
            b"wheel-data"
        )
        (self.project / "VERSION").write_text("0.3.0-rc.1\n", encoding="utf-8")
        (self.project / "requirements.txt").write_text("package==1.0\n", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=False)

    def test_prepares_split_bundle_then_verifies_and_extracts(self) -> None:
        bundle = self.root / "bundle"

        info = prepare_assets(
            self.project,
            bundle,
            part_size_bytes=256 * 1024,
        )

        self.assertGreater(len(info["parts"]), 1)
        self.assertTrue((bundle / "ARCHIVE_INFO.json").is_file())
        self.assertTrue((bundle / "REASSEMBLE.py.txt").is_file())
        self.assertFalse((bundle / info["archive_name"]).exists())
        self.assertTrue(
            all((bundle / entry["name"]).stat().st_size <= 256 * 1024 for entry in info["parts"])
        )

        extracted = self.root / "extracted"
        archive = reassemble(bundle, extract_to=extracted)

        self.assertTrue(archive.is_file())
        self.assertEqual(
            (extracted / "models" / "embedding" / "model.bin").read_bytes(),
            (self.project / "models" / "embedding" / "model.bin").read_bytes(),
        )
        self.assertTrue((extracted / "ASSET_CONTENTS.json").is_file())

    def test_rejects_repository_output_and_missing_wheels(self) -> None:
        with self.assertRaisesRegex(AssetBundleError, "outside"):
            prepare_assets(
                self.project,
                self.project / "bundle",
                part_size_bytes=1024,
            )

        (self.project / "wheels" / "package-1.0-py3-none-any.whl").unlink()
        with self.assertRaisesRegex(AssetBundleError, "Wheel"):
            prepare_assets(
                self.project,
                self.root / "missing-wheels",
                part_size_bytes=1024,
            )

    def test_detects_tampered_part(self) -> None:
        bundle = self.root / "tampered"
        info = prepare_assets(
            self.project,
            bundle,
            part_size_bytes=256 * 1024,
        )
        first_part = bundle / info["parts"][0]["name"]
        with first_part.open("r+b") as target:
            target.seek(0)
            target.write(b"tampered")

        with self.assertRaisesRegex(ReassemblyError, "SHA-256 mismatch"):
            reassemble(bundle)


if __name__ == "__main__":
    unittest.main()
