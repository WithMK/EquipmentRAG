"""Shared parser contract and normalized-document construction helpers."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any, Protocol

from app.models.document_models import (
    DocumentBlock,
    DocumentSourceFile,
    NormalizedDocument,
)


class DocumentParseError(RuntimeError):
    """Raised when a supported document cannot be parsed safely."""


class DocumentParser(Protocol):
    def parse(self, source: DocumentSourceFile) -> NormalizedDocument: ...


def build_normalized_document(
    source: DocumentSourceFile,
    blocks: list[DocumentBlock],
    *,
    detected_title: str = "",
    detected_created_date: str = "",
) -> NormalizedDocument:
    """Combine parsed blocks with optional sidecar metadata."""

    metadata = source.metadata
    title = _string(metadata.get("title")) or detected_title or source.path.stem
    document_id = _string(metadata.get("document_id")) or _document_id(
        source.relative_path
    )
    equipment = _string(metadata.get("equipment"))
    return NormalizedDocument(
        document_id=document_id,
        source_path=source.relative_path,
        file_name=source.path.name,
        file_extension=source.extension,
        title=title,
        blocks=tuple(blocks),
        project=_string(metadata.get("project")),
        equipment=equipment,
        unit=_string(metadata.get("unit")),
        document_type=_string(metadata.get("document_type")),
        revision=_string(metadata.get("revision")),
        document_status=_string(metadata.get("document_status")) or "active",
        is_latest=_boolean(metadata.get("is_latest"), default=True),
        created_date=_date_string(metadata.get("created_date"))
        or detected_created_date,
        modified_date=_date_string(metadata.get("modified_date"))
        or source.modified_time,
        file_hash=source.file_hash,
    )


def read_text_document(source: DocumentSourceFile) -> str:
    try:
        raw = source.path.read_bytes()
    except OSError as exc:
        raise DocumentParseError(f"Unable to read document: {source.path}") from exc
    for encoding in ("utf-8-sig", "utf-16", "cp949"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentParseError(f"Unable to decode text document: {source.path}")


def _document_id(relative_path: str) -> str:
    digest = hashlib.sha256(relative_path.casefold().encode("utf-8")).hexdigest()
    return "document-" + digest


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value).strip()
    raise DocumentParseError("Document sidecar scalar value is invalid")


def _boolean(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise DocumentParseError("Document sidecar is_latest must be a boolean")
    return value


def _date_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return _string(value)
