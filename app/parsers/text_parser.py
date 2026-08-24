"""Paragraph and section-aware plain-text parser."""

from __future__ import annotations

import re

from app.models.document_models import DocumentBlock, DocumentSourceFile, NormalizedDocument
from app.parsers.document_parser import build_normalized_document, read_text_document


_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+\S+")


class TextDocumentParser:
    def parse(self, source: DocumentSourceFile) -> NormalizedDocument:
        text = read_text_document(source)
        blocks: list[DocumentBlock] = []
        detected_title = ""
        for paragraph in re.split(r"\n\s*\n", text):
            normalized = "\n".join(
                line.rstrip() for line in paragraph.splitlines() if line.strip()
            ).strip()
            if not normalized:
                continue
            lines = normalized.splitlines()
            first = lines[0].strip()
            if _is_heading(first):
                level = max(1, first.count(".") + 1)
                heading = first.rstrip(":").strip()
                blocks.append(DocumentBlock("heading", heading, level=level))
                if not detected_title:
                    detected_title = heading
                remainder = "\n".join(lines[1:]).strip()
                if remainder:
                    blocks.append(DocumentBlock("paragraph", remainder))
            else:
                blocks.append(DocumentBlock("paragraph", normalized))
        return build_normalized_document(
            source,
            blocks,
            detected_title=detected_title,
        )


def _is_heading(value: str) -> bool:
    if len(value) > 120:
        return False
    return bool(
        value.endswith(":")
        or _NUMBERED_HEADING.match(value)
        or (value.isupper() and any(character.isalpha() for character in value))
    )
