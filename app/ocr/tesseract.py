"""Safe local Tesseract OCR adapter."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol


class OcrError(RuntimeError):
    """Raised when local OCR cannot be completed."""


class OcrProvider(Protocol):
    def recognize(self, image_bytes: bytes, *, extension: str = ".png") -> str: ...


class TesseractOcrProvider:
    """Run an explicitly configured local Tesseract executable without a shell."""

    def __init__(
        self,
        executable: Path,
        *,
        languages: str = "kor+eng",
        timeout_seconds: int = 60,
    ) -> None:
        if not isinstance(languages, str) or not re.fullmatch(
            r"[A-Za-z0-9_+\-]+",
            languages,
        ):
            raise OcrError("OCR languages value is invalid")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or timeout_seconds <= 0
        ):
            raise OcrError("OCR timeout must be a positive integer")
        self._executable = executable
        self._languages = languages
        self._timeout_seconds = timeout_seconds

    def recognize(self, image_bytes: bytes, *, extension: str = ".png") -> str:
        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise OcrError("OCR image must contain bytes")
        suffix = extension.casefold()
        if not re.fullmatch(r"\.[a-z0-9]{2,5}", suffix):
            raise OcrError("OCR image extension is invalid")
        executable = self._executable.resolve(strict=False)
        if not executable.is_file():
            raise OcrError(f"Tesseract executable not found: {executable}")

        with tempfile.TemporaryDirectory(prefix="equipment-rag-ocr-") as temp_dir:
            root = Path(temp_dir)
            input_path = root / f"input{suffix}"
            output_base = root / "result"
            input_path.write_bytes(image_bytes)
            command = [
                str(executable),
                str(input_path),
                str(output_base),
                "-l",
                self._languages,
                "--psm",
                "6",
            ]
            creation_flags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0
            )
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_seconds,
                    creationflags=creation_flags,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise OcrError("Unable to run local Tesseract OCR") from exc
            if completed.returncode != 0:
                detail = completed.stderr.strip()
                raise OcrError(
                    "Tesseract OCR failed" + (f": {detail}" if detail else "")
                )
            output_path = output_base.with_suffix(".txt")
            try:
                return output_path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError) as exc:
                raise OcrError("Unable to read Tesseract OCR output") from exc
