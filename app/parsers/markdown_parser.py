"""Structure-preserving Markdown parser."""

from __future__ import annotations

import re

from app.models.document_models import DocumentBlock, DocumentSourceFile, NormalizedDocument
from app.parsers.document_parser import build_normalized_document, read_text_document


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_LIST = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")


class MarkdownDocumentParser:
    def parse(self, source: DocumentSourceFile) -> NormalizedDocument:
        text = read_text_document(source)
        lines = text.splitlines()
        blocks: list[DocumentBlock] = []
        paragraph: list[str] = []
        list_lines: list[str] = []
        code_lines: list[str] = []
        in_code = False
        detected_title = ""

        def flush_paragraph() -> None:
            if paragraph:
                blocks.append(DocumentBlock("paragraph", "\n".join(paragraph)))
                paragraph.clear()

        def flush_list() -> None:
            if list_lines:
                blocks.append(DocumentBlock("list", "\n".join(list_lines)))
                list_lines.clear()

        for line in lines:
            if line.lstrip().startswith("```"):
                flush_paragraph()
                flush_list()
                if in_code:
                    blocks.append(DocumentBlock("code", "\n".join(code_lines)))
                    code_lines.clear()
                    in_code = False
                else:
                    in_code = True
                continue
            if in_code:
                code_lines.append(line)
                continue
            match = _HEADING.match(line)
            if match:
                flush_paragraph()
                flush_list()
                title = match.group(2).strip()
                level = len(match.group(1))
                blocks.append(DocumentBlock("heading", title, level=level))
                if not detected_title and level == 1:
                    detected_title = title
                continue
            if _LIST.match(line):
                flush_paragraph()
                list_lines.append(line.strip())
                continue
            if not line.strip():
                flush_paragraph()
                flush_list()
                continue
            flush_list()
            paragraph.append(line.rstrip())

        flush_paragraph()
        flush_list()
        if code_lines:
            blocks.append(DocumentBlock("code", "\n".join(code_lines)))
        return build_normalized_document(
            source,
            blocks,
            detected_title=detected_title,
        )
