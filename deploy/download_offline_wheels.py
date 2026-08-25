"""Download a complete binary-only wheelhouse and write its SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


class WheelDownloadError(RuntimeError):
    """Raised when a reproducible offline wheelhouse cannot be collected."""


def download_wheels(
    project_root: Path,
    output_root: Path,
    *,
    python_executable: Path = Path(sys.executable),
    allow_non_windows: bool = False,
) -> dict[str, object]:
    project = project_root.expanduser().resolve(strict=True)
    output = output_root.expanduser().resolve(strict=False)
    if platform.system() != "Windows" and not allow_non_windows:
        raise WheelDownloadError(
            "Run this command on the target Windows architecture"
        )
    if sys.version_info[:2] != (3, 12):
        raise WheelDownloadError("Python 3.12 is required for this release wheelhouse")
    requirements = project / "requirements-vision.txt"
    if not requirements.is_file():
        raise WheelDownloadError(f"Requirements file not found: {requirements}")
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_executable),
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "--dest",
        str(output),
        "--requirement",
        str(requirements),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise WheelDownloadError("pip could not download the offline wheelhouse")
    wheels = tuple(sorted(output.glob("*.whl"), key=lambda path: path.name.casefold()))
    if not wheels:
        raise WheelDownloadError("No wheel files were downloaded")
    unexpected = tuple(path for path in output.iterdir() if path.is_file() and path.suffix.casefold() not in {".whl", ".json"})
    if unexpected:
        raise WheelDownloadError(
            "Non-wheel files were downloaded: "
            + ", ".join(path.name for path in unexpected)
        )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "wheel_count": len(wheels),
        "wheels": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in wheels
        ],
    }
    (output / "WHEEL_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Windows Python 3.12 EquipmentRAG wheels"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.project_root / "wheels"
    try:
        manifest = download_wheels(args.project_root, output)
    except (WheelDownloadError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
