"""Create a hash-verified, split offline asset bundle outside the Git repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


_BUFFER_SIZE = 8 * 1024 * 1024
_DEFAULT_PART_SIZE_MB = 1900


class AssetBundleError(RuntimeError):
    """Raised when an offline asset bundle cannot be produced safely."""


def prepare_assets(
    project_root: Path,
    output_root: Path,
    *,
    part_size_bytes: int = _DEFAULT_PART_SIZE_MB * 1024 * 1024,
    allow_missing_wheels: bool = False,
) -> dict[str, object]:
    project = project_root.expanduser().resolve(strict=True)
    output = output_root.expanduser().resolve(strict=False)
    if output == project or output.is_relative_to(project):
        raise AssetBundleError("Output directory must be outside the Git repository")
    if output.exists() and any(output.iterdir()):
        raise AssetBundleError(f"Output directory is not empty: {output}")
    if isinstance(part_size_bytes, bool) or part_size_bytes <= 0:
        raise AssetBundleError("Part size must be a positive integer")

    model_root = project / "models"
    wheels_root = project / "wheels"
    if not model_root.is_dir() or not any(model_root.rglob("*")):
        raise AssetBundleError(f"Model directory is missing or empty: {model_root}")
    if not wheels_root.is_dir() or not any(wheels_root.glob("*.whl")):
        if not allow_missing_wheels:
            raise AssetBundleError(
                "Wheel directory is missing or empty; download offline wheels first"
            )

    version = _read_version(project)
    output.mkdir(parents=True, exist_ok=True)
    archive_name = f"EquipmentRAG-offline-assets-{version}.zip"
    archive_path = output / archive_name
    files = _asset_files(project, allow_missing_wheels=allow_missing_wheels)
    contents = {
        "bundle_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [
            {
                "path": path.relative_to(project).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
    }
    contents_text = json.dumps(contents, ensure_ascii=False, indent=2) + "\n"
    (output / "ASSET_CONTENTS.json").write_text(contents_text, encoding="utf-8")

    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        archive.writestr("ASSET_CONTENTS.json", contents_text)
        for path in files:
            archive.write(path, path.relative_to(project).as_posix())

    archive_sha256 = _sha256(archive_path)
    parts = _split_archive(archive_path, part_size_bytes)
    if len(parts) > 1:
        archive_path.unlink()
    part_entries = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in parts
    ]
    model_count = sum(
        1 for path in files if path.is_relative_to(project / "models")
    )
    wheel_count = sum(1 for path in files if path.suffix.casefold() == ".whl")
    info = {
        "bundle_version": version,
        "archive_name": archive_name,
        "archive_bytes": sum(entry["bytes"] for entry in part_entries),
        "archive_sha256": archive_sha256,
        "part_size_bytes": part_size_bytes,
        "model_file_count": model_count,
        "wheel_file_count": wheel_count,
        "parts": part_entries,
    }
    (output / "ARCHIVE_INFO.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    helper = Path(__file__).with_name("reassemble_offline_assets.py")
    shutil.copyfile(helper, output / "REASSEMBLE.py.txt")
    (output / "README_TRANSFER.txt").write_text(
        _transfer_readme(info),
        encoding="utf-8",
    )
    return info


def _asset_files(project: Path, *, allow_missing_wheels: bool) -> tuple[Path, ...]:
    files: list[Path] = []
    for directory in (project / "models", project / "wheels"):
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_symlink():
                    raise AssetBundleError(f"Symbolic links are not allowed: {path}")
                if path.is_file():
                    files.append(path)
    for name in (
        "VERSION",
        "requirements.txt",
        "requirements-offline.txt",
        "requirements-vision.txt",
    ):
        path = project / name
        if path.is_file():
            files.append(path)
    if not allow_missing_wheels and not any(
        path.parent == project / "wheels" and path.suffix.casefold() == ".whl"
        for path in files
    ):
        raise AssetBundleError("No wheel files were selected")
    return tuple(sorted(files, key=lambda value: value.relative_to(project).as_posix()))


def _split_archive(archive: Path, part_size_bytes: int) -> tuple[Path, ...]:
    if archive.stat().st_size <= part_size_bytes:
        return (archive,)
    parts: list[Path] = []
    with archive.open("rb") as source:
        part_number = 1
        while source.tell() < archive.stat().st_size:
            part = archive.with_name(f"{archive.name}.bin.{part_number:03d}")
            remaining = part_size_bytes
            with part.open("xb") as destination:
                while remaining > 0:
                    block = source.read(min(_BUFFER_SIZE, remaining))
                    if not block:
                        break
                    destination.write(block)
                    remaining -= len(block)
            parts.append(part)
            part_number += 1
    return tuple(parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(_BUFFER_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _read_version(project: Path) -> str:
    try:
        version = (project / "VERSION").read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise AssetBundleError("VERSION file is missing") from exc
    if not version or any(character not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.-" for character in version):
        raise AssetBundleError("VERSION contains unsupported characters")
    return version


def _transfer_readme(info: dict[str, object]) -> str:
    return (
        "EquipmentRAG offline asset transfer bundle\n"
        "==========================================\n\n"
        "This directory contains models and Windows wheels only. Source code is transferred through GitHub.\n\n"
        "1. Transfer every file in this directory.\n"
        "2. Keep all .bin.NNN parts in the same directory.\n"
        "3. Run without renaming the helper:\n\n"
        "   python REASSEMBLE.py.txt --extract-to D:\\OfflineAssets\\EquipmentRAG\n\n"
        "The helper verifies every part and the complete archive before extraction.\n"
        f"Archive: {info['archive_name']}\n"
        f"SHA-256: {info['archive_sha256']}\n"
        f"Model files: {info['model_file_count']}\n"
        f"Wheel files: {info['wheel_file_count']}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare split EquipmentRAG model and wheel assets"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part-size-mb", type=int, default=_DEFAULT_PART_SIZE_MB)
    parser.add_argument("--allow-missing-wheels", action="store_true")
    args = parser.parse_args()
    try:
        info = prepare_assets(
            args.project_root,
            args.output,
            part_size_bytes=args.part_size_mb * 1024 * 1024,
            allow_missing_wheels=args.allow_missing_wheels,
        )
    except (AssetBundleError, OSError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
