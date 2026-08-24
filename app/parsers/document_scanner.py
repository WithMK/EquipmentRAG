"""Offline document discovery, hashing, and sidecar metadata loading."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.config import DocumentConfig
from app.models.document_models import DocumentSourceFile


class DocumentScanError(RuntimeError):
    """Raised when configured document roots or metadata are invalid."""


class DocumentScanner:
    def __init__(self, config: DocumentConfig) -> None:
        self._config = config
        self._extensions = {value.casefold() for value in config.extensions}
        self._excluded = {value.casefold() for value in config.exclude_directories}

    def scan(self) -> list[DocumentSourceFile]:
        discovered: list[DocumentSourceFile] = []
        multiple_roots = len(self._config.source_paths) > 1
        for root in self._config.source_paths:
            resolved = root.resolve(strict=False)
            if not resolved.is_dir():
                raise DocumentScanError(f"Document directory not found: {resolved}")
            discovered.extend(self._scan_root(resolved, multiple_roots))
        discovered.sort(key=lambda item: item.relative_path.casefold())
        return discovered

    def _scan_root(
        self, root: Path, multiple_roots: bool
    ) -> list[DocumentSourceFile]:
        values: list[DocumentSourceFile] = []
        for current_root, directories, file_names in os.walk(root, topdown=True):
            current_path = Path(current_root)
            directories[:] = sorted(
                (
                    name
                    for name in directories
                    if name.casefold() not in self._excluded
                    and not (current_path / name).is_symlink()
                ),
                key=str.casefold,
            )
            for file_name in sorted(file_names, key=str.casefold):
                path = current_path / file_name
                if path.is_symlink() or path.suffix.casefold() not in self._extensions:
                    continue
                relative = path.relative_to(root).as_posix()
                if multiple_roots:
                    relative = f"{root.name}/{relative}"
                values.append(self._read_source(path, relative, root.name))
        return values

    def _read_source(
        self, path: Path, relative_path: str, root_name: str
    ) -> DocumentSourceFile:
        try:
            raw = path.read_bytes()
            modified = datetime.fromtimestamp(
                path.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat()
        except OSError as exc:
            raise DocumentScanError(f"Unable to read document: {path}") from exc
        return DocumentSourceFile(
            path=path.resolve(strict=False),
            relative_path=relative_path,
            root_name=root_name,
            file_hash=hashlib.sha256(raw).hexdigest(),
            modified_time=modified,
            metadata=self._load_sidecar(path),
        )

    @staticmethod
    def _load_sidecar(path: Path) -> Mapping[str, Any]:
        candidates = (
            path.with_name(path.name + ".metadata.yaml"),
            path.with_suffix(".metadata.yaml"),
        )
        sidecar = next((value for value in candidates if value.is_file()), None)
        if sidecar is None:
            return {}
        try:
            payload = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise DocumentScanError(
                f"Unable to read document metadata sidecar: {sidecar}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise DocumentScanError(
                f"Document metadata sidecar must be a mapping: {sidecar}"
            )
        return dict(payload)

