from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from deploy.download_offline_wheels import WheelDownloadError, download_wheels


class WheelDownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="equipment-rag-wheels-"))
        self.project = self.root / "project"
        self.project.mkdir()
        (self.project / "requirements-vision.txt").write_text(
            "package==1.0\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=False)

    def test_downloads_binary_wheels_and_writes_hash_manifest(self) -> None:
        output = self.root / "wheels"

        def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
            self.assertFalse(check)
            destination = Path(command[command.index("--dest") + 1])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "package-1.0-py3-none-any.whl").write_bytes(b"wheel")
            self.assertIn("--only-binary=:all:", command)
            return SimpleNamespace(returncode=0)

        with (
            patch("deploy.download_offline_wheels.platform.system", return_value="Windows"),
            patch(
                "deploy.download_offline_wheels.platform.platform",
                return_value="Windows-11-x86_64",
            ),
            patch("deploy.download_offline_wheels.sys.version_info", (3, 12)),
            patch("deploy.download_offline_wheels.subprocess.run", side_effect=fake_run),
        ):
            manifest = download_wheels(
                self.project,
                output,
                python_executable=Path("python.exe"),
            )

        self.assertEqual(manifest["wheel_count"], 1)
        saved = json.loads(
            (output / "WHEEL_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(saved["wheels"][0]["sha256"]), 64)

    def test_rejects_wrong_platform_and_failed_download(self) -> None:
        with patch(
            "deploy.download_offline_wheels.platform.system",
            return_value="Linux",
        ):
            with self.assertRaisesRegex(WheelDownloadError, "Windows"):
                download_wheels(self.project, self.root / "linux")

        with (
            patch("deploy.download_offline_wheels.platform.system", return_value="Windows"),
            patch("deploy.download_offline_wheels.sys.version_info", (3, 12)),
            patch(
                "deploy.download_offline_wheels.subprocess.run",
                return_value=SimpleNamespace(returncode=1),
            ),
        ):
            with self.assertRaisesRegex(WheelDownloadError, "could not download"):
                download_wheels(self.project, self.root / "failed")


if __name__ == "__main__":
    unittest.main()
