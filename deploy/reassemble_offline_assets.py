"""Verify, reassemble, and optionally extract an EquipmentRAG asset bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


_BUFFER_SIZE = 8 * 1024 * 1024


class ReassemblyError(RuntimeError):
    """Raised when transferred parts fail validation or safe extraction."""


def reassemble(bundle_root: Path, *, extract_to: Path | None = None) -> Path:
    root = bundle_root.expanduser().resolve(strict=True)
    info = _load_info(root / "ARCHIVE_INFO.json")
    archive = root / info["archive_name"]
    parts = info["parts"]
    if archive.exists() and len(parts) > 1:
        raise ReassemblyError(f"Refusing to overwrite existing archive: {archive}")
    if len(parts) == 1 and parts[0]["name"] == archive.name:
        _verify_file(archive, parts[0]["sha256"])
    else:
        for entry in parts:
            _verify_file(root / entry["name"], entry["sha256"])
        with archive.open("xb") as destination:
            for entry in parts:
                part = root / entry["name"]
                with part.open("rb") as source:
                    while block := source.read(_BUFFER_SIZE):
                        destination.write(block)
    _verify_file(archive, info["archive_sha256"])
    if extract_to is not None:
        _safe_extract(archive, extract_to.expanduser().resolve(strict=False))
    return archive


def _load_info(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReassemblyError(f"Unable to read archive information: {path}") from exc
    if not isinstance(value, dict):
        raise ReassemblyError("ARCHIVE_INFO.json must contain an object")
    archive_name = value.get("archive_name")
    archive_hash = value.get("archive_sha256")
    parts = value.get("parts")
    if (
        not isinstance(archive_name, str)
        or Path(archive_name).name != archive_name
        or not _is_sha256(archive_hash)
        or not isinstance(parts, list)
        or not parts
    ):
        raise ReassemblyError("ARCHIVE_INFO.json is invalid")
    for entry in parts:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("name"), str)
            or Path(entry["name"]).name != entry["name"]
            or not _is_sha256(entry.get("sha256"))
        ):
            raise ReassemblyError("Archive part metadata is invalid")
    return value


def _verify_file(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ReassemblyError(f"Required file is missing: {path.name}")
    actual = _sha256(path)
    if actual.casefold() != expected.casefold():
        raise ReassemblyError(f"SHA-256 mismatch: {path.name}")


def _safe_extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            relative = PurePosixPath(member.filename)
            unix_mode = member.external_attr >> 16
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or (relative.parts and ":" in relative.parts[0])
                or unix_mode & 0o170000 == 0o120000
            ):
                raise ReassemblyError(f"Unsafe archive path: {member.filename}")
            destination = (target / Path(*relative.parts)).resolve(strict=False)
            if not destination.is_relative_to(target):
                raise ReassemblyError(f"Unsafe archive target: {member.filename}")
        source.extractall(target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(_BUFFER_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and reassemble EquipmentRAG offline assets"
    )
    parser.add_argument("--bundle-root", type=Path, default=Path.cwd())
    parser.add_argument("--extract-to", type=Path)
    args = parser.parse_args()
    try:
        archive = reassemble(args.bundle_root, extract_to=args.extract_to)
    except (ReassemblyError, OSError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(f"Verified archive: {archive}")
    if args.extract_to is not None:
        print(f"Extracted to: {args.extract_to.resolve(strict=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
