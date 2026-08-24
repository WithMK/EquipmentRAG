"""DOCX parser that preserves paragraph, heading, list, and table order."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from app.models.document_models import DocumentBlock, DocumentSourceFile, NormalizedDocument
from app.parsers.document_parser import DocumentParseError, build_normalized_document


_HEADING_LEVEL = re.compile(r"heading\s*(\d+)", re.IGNORECASE)


class DocxDocumentParser:
    def parse(self, source: DocumentSourceFile) -> NormalizedDocument:
        try:
            from docx import Document
            from docx.oxml.table import CT_Tbl
            from docx.oxml.text.paragraph import CT_P
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:
            raise DocumentParseError(
                "python-docx is required to parse DOCX documents"
            ) from exc

        try:
            document = Document(str(source.path))
        except Exception as exc:
            raise DocumentParseError(f"Unable to parse DOCX: {source.path}") from exc

        blocks: list[DocumentBlock] = []
        detected_title = _string(document.core_properties.title)
        for child in document.element.body.iterchildren():
            if isinstance(child, CT_P):
                paragraph = Paragraph(child, document)
                text = paragraph.text.strip()
                if not text:
                    continue
                style_name = _string(getattr(paragraph.style, "name", ""))
                level = _heading_level(style_name)
                if style_name.casefold() == "title" and not detected_title:
                    detected_title = text
                    blocks.append(DocumentBlock("heading", text, level=1))
                elif level:
                    blocks.append(DocumentBlock("heading", text, level=level))
                    if not detected_title and level == 1:
                        detected_title = text
                elif _is_list(paragraph, style_name):
                    blocks.append(DocumentBlock("list", text))
                else:
                    blocks.append(DocumentBlock("paragraph", text))
            elif isinstance(child, CT_Tbl):
                table = Table(child, document)
                rows = tuple(
                    tuple(cell.text.strip() for cell in row.cells)
                    for row in table.rows
                )
                non_empty = tuple(row for row in rows if any(row))
                if non_empty:
                    text = "\n".join(" | ".join(row) for row in non_empty)
                    blocks.append(DocumentBlock("table", text, rows=non_empty))

        created = _date_string(document.core_properties.created)
        return build_normalized_document(
            source,
            blocks,
            detected_title=detected_title,
            detected_created_date=created,
        )


def _heading_level(style_name: str) -> int:
    match = _HEADING_LEVEL.search(style_name)
    return int(match.group(1)) if match else 0


def _is_list(paragraph: Any, style_name: str) -> bool:
    if "list" in style_name.casefold():
        return True
    properties = getattr(paragraph._p, "pPr", None)
    return properties is not None and properties.numPr is not None


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _date_string(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return ""

