from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.ocr.tesseract import OcrError, TesseractOcrProvider


class TesseractOcrProviderTests(unittest.TestCase):
    def test_runs_explicit_executable_and_reads_utf8_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            executable = Path(temp_dir) / "tesseract.exe"
            executable.touch()
            provider = TesseractOcrProvider(executable, languages="kor+eng")

            def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
                Path(command[2]).with_suffix(".txt").write_text(
                    "진공 센서 점검",
                    encoding="utf-8",
                )
                self.assertNotIn("shell", kwargs)
                self.assertEqual(command[3:5], ["-l", "kor+eng"])
                return SimpleNamespace(returncode=0, stderr="")

            with patch("app.ocr.tesseract.subprocess.run", side_effect=fake_run):
                result = provider.recognize(b"image", extension=".png")

            self.assertEqual(result, "진공 센서 점검")

    def test_rejects_missing_executable(self) -> None:
        provider = TesseractOcrProvider(Path("missing-tesseract.exe"))

        with self.assertRaisesRegex(OcrError, "not found"):
            provider.recognize(b"image")

    def test_converts_timeout_to_ocr_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            executable = Path(temp_dir) / "tesseract.exe"
            executable.touch()
            provider = TesseractOcrProvider(executable)
            with patch(
                "app.ocr.tesseract.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["tesseract"], 1),
            ):
                with self.assertRaisesRegex(OcrError, "Unable to run"):
                    provider.recognize(b"image")


if __name__ == "__main__":
    unittest.main()
