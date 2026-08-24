"""Text-PDF parser with page traceability and repeated margin removal."""

from __future__ import annotations

import math
import re
from collections import Counter

from app.models.document_models import DocumentBlock, DocumentSourceFile, NormalizedDocument
from app.parsers.document_parser import DocumentParseError, build_normalized_document


_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+\S+")


class PdfDocumentParser:
    def parse(self, source: DocumentSourceFile) -> NormalizedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentParseError("pypdf is required to parse PDF documents") from exc
        try:
            reader = PdfReader(str(source.path))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:
            raise DocumentParseError(f"Unable to parse PDF: {source.path}") from exc

        cleaned_pages = _remove_repeated_margins(pages)
        blocks: list[DocumentBlock] = []
        detected_title = ""
        for page_number, page_text in enumerate(cleaned_pages, 1):
            for paragraph in re.split(r"\n\s*\n", page_text):
                lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
                if not lines:
                    continue
                body: list[str] = []
                for line in lines:
                    if _is_heading(line):
                        if body:
                            blocks.append(
                                DocumentBlock(
                                    "paragraph",
                                    "\n".join(body),
                                    page=page_number,
                                )
                            )
                            body.clear()
                        level = max(1, line.split(maxsplit=1)[0].count(".") + 1)
                        heading = line.rstrip(":")
                        blocks.append(
                            DocumentBlock(
                                "heading",
                                heading,
                                level=level,
                                page=page_number,
                            )
                        )
                        if not detected_title:
                            detected_title = heading
                    else:
                        body.append(line)
                if body:
                    blocks.append(
                        DocumentBlock(
                            "paragraph",
                            "\n".join(body),
                            page=page_number,
                        )
                    )
        return build_normalized_document(
            source,
            blocks,
            detected_title=detected_title,
        )


def _remove_repeated_margins(pages: list[str]) -> list[str]:
    split_pages = [[line.strip() for line in page.splitlines()] for page in pages]
    non_empty = [[line for line in lines if line] for lines in split_pages]
    if len(non_empty) < 2:
        return pages
    threshold = max(2, math.ceil(len(non_empty) * 0.6))
    headers = Counter(lines[0] for lines in non_empty if lines)
    footers = Counter(lines[-1] for lines in non_empty if lines)
    repeated_headers = {value for value, count in headers.items() if count >= threshold}
    repeated_footers = {value for value, count in footers.items() if count >= threshold}
    cleaned: list[str] = []
    for lines in split_pages:
        values = list(lines)
        first = next((index for index, value in enumerate(values) if value), None)
        last = next(
            (index for index in range(len(values) - 1, -1, -1) if values[index]),
            None,
        )
        if first is not None and values[first] in repeated_headers:
            values[first] = ""
        if last is not None and values[last] in repeated_footers:
            values[last] = ""
        cleaned.append("\n".join(values))
    return cleaned


def _is_heading(value: str) -> bool:
    if len(value) > 100:
        return False
    return bool(
        value.endswith(":")
        or _NUMBERED_HEADING.match(value)
        or (value.isupper() and any(character.isalpha() for character in value))
    )

