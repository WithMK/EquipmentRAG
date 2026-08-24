"""Normalized document and document-vector metadata models."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any


class DocumentModelError(ValueError):
    """Raised when normalized document data is inconsistent."""


@dataclass(frozen=True)
class DocumentSourceFile:
    path: Path
    relative_path: str
    root_name: str
    file_hash: str
    state_hash: str
    modified_time: str
    metadata: Mapping[str, Any]

    @property
    def extension(self) -> str:
        return self.path.suffix.casefold()


@dataclass(frozen=True)
class DocumentBlock:
    type: str
    text: str
    level: int = 0
    page: int = 0
    slide: int = 0
    sheet: str = ""
    cell_range: str = ""
    rows: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        valid_types = {"heading", "paragraph", "table", "list", "code"}
        if self.type not in valid_types:
            raise DocumentModelError(f"Unsupported document block type: {self.type}")
        if not isinstance(self.text, str) or not self.text.strip():
            raise DocumentModelError("Document block text must be non-empty")
        for field_name in ("level", "page", "slide"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DocumentModelError(
                    f"Document block {field_name} must be a non-negative integer"
                )
        for field_name in ("sheet", "cell_range"):
            if not isinstance(getattr(self, field_name), str):
                raise DocumentModelError(
                    f"Document block {field_name} must be a string"
                )


@dataclass(frozen=True)
class NormalizedDocument:
    document_id: str
    source_path: str
    file_name: str
    file_extension: str
    title: str
    blocks: tuple[DocumentBlock, ...]
    project: str = ""
    equipment: str = ""
    unit: str = ""
    document_type: str = ""
    revision: str = ""
    document_status: str = "active"
    is_latest: bool = True
    created_date: str = ""
    modified_date: str = ""
    file_hash: str = ""

    def __post_init__(self) -> None:
        required = ("document_id", "source_path", "file_name", "file_extension")
        for field_name in required:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise DocumentModelError(f"{field_name} must be a non-empty string")
        if not isinstance(self.blocks, tuple):
            raise DocumentModelError("blocks must be a tuple")
        if not isinstance(self.is_latest, bool):
            raise DocumentModelError("is_latest must be a boolean")


@dataclass(frozen=True)
class DocumentChunk:
    content: str
    heading_path: tuple[str, ...]
    section: str
    subsection: str
    page: int
    slide: int
    sheet: str
    cell_range: str
    content_hash: str

    @classmethod
    def create(
        cls,
        content: str,
        *,
        heading_path: tuple[str, ...] = (),
        page: int = 0,
        slide: int = 0,
        sheet: str = "",
        cell_range: str = "",
    ) -> "DocumentChunk":
        normalized = content.strip()
        if not normalized:
            raise DocumentModelError("Document chunk content must be non-empty")
        section = heading_path[0] if heading_path else ""
        subsection = heading_path[-1] if len(heading_path) > 1 else ""
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return cls(
            content=normalized,
            heading_path=heading_path,
            section=section,
            subsection=subsection,
            page=page,
            slide=slide,
            sheet=sheet,
            cell_range=cell_range,
            content_hash=digest,
        )


@dataclass(frozen=True)
class DocumentChunkMetadata:
    """Complete scalar metadata stored with one document chunk."""

    document_id: str
    source_path: str
    file_name: str
    file_extension: str
    equipment: str
    chunk_index: int
    project: str = ""
    unit: str = ""
    document_type: str = ""
    title: str = ""
    revision: str = ""
    document_status: str = "active"
    is_latest: bool = True
    section: str = ""
    subsection: str = ""
    heading_path: str = ""
    page: int = 0
    slide: int = 0
    sheet: str = ""
    cell_range: str = ""
    created_date: str = ""
    modified_date: str = ""
    file_hash: str = ""
    content_hash: str = ""
    indexed_at: str = ""
    source_type: str = "document"

    def __post_init__(self) -> None:
        string_fields = tuple(
            field_name
            for field_name in self.__dataclass_fields__
            if field_name not in {"chunk_index", "page", "slide", "is_latest"}
        )
        for field_name in string_fields:
            if not isinstance(getattr(self, field_name), str):
                raise DocumentModelError(f"metadata.{field_name} must be a string")
        for field_name in ("document_id", "source_path", "file_name", "equipment"):
            if not getattr(self, field_name).strip():
                raise DocumentModelError(
                    f"metadata.{field_name} must be a non-empty string"
                )
        if self.source_type != "document":
            raise DocumentModelError("metadata.source_type must be 'document'")
        if not isinstance(self.is_latest, bool):
            raise DocumentModelError("metadata.is_latest must be a boolean")
        for field_name in ("chunk_index", "page", "slide"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DocumentModelError(
                    f"metadata.{field_name} must be a non-negative integer"
                )

    def to_chroma(self) -> dict[str, str | int | bool]:
        return asdict(self)

    @classmethod
    def from_chroma(
        cls, value: Mapping[str, Any] | None
    ) -> "DocumentChunkMetadata":
        if not isinstance(value, Mapping):
            raise DocumentModelError("ChromaDB returned invalid document metadata")
        parsed: dict[str, Any] = {}
        try:
            for model_field in fields(cls):
                if model_field.name in value:
                    parsed[model_field.name] = value[model_field.name]
                elif model_field.default is not MISSING:
                    parsed[model_field.name] = model_field.default
                else:
                    raise KeyError(model_field.name)
            return cls(**parsed)
        except KeyError as exc:
            raise DocumentModelError(
                f"Document metadata is missing field: {exc.args[0]}"
            ) from exc
