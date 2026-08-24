"""Document parser registry for the Phase 1 file formats."""

from __future__ import annotations

from app.models.document_models import DocumentSourceFile, NormalizedDocument
from app.parsers.document_parser import DocumentParseError, DocumentParser
from app.parsers.docx_parser import DocxDocumentParser
from app.parsers.markdown_parser import MarkdownDocumentParser
from app.parsers.pdf_parser import PdfDocumentParser
from app.parsers.text_parser import TextDocumentParser


class DocumentParserRegistry:
    def __init__(self, parsers: dict[str, DocumentParser] | None = None) -> None:
        self._parsers = parsers or {
            ".docx": DocxDocumentParser(),
            ".pdf": PdfDocumentParser(),
            ".md": MarkdownDocumentParser(),
            ".txt": TextDocumentParser(),
        }

    def parse(self, source: DocumentSourceFile) -> NormalizedDocument:
        parser = self._parsers.get(source.extension)
        if parser is None:
            raise DocumentParseError(
                f"No document parser registered for: {source.extension}"
            )
        return parser.parse(source)
