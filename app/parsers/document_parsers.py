"""Document parser registry for supported offline document formats."""

from __future__ import annotations

from app.models.document_models import DocumentSourceFile, NormalizedDocument
from app.ocr.tesseract import OcrProvider
from app.parsers.document_parser import DocumentParseError, DocumentParser
from app.parsers.docx_parser import DocxDocumentParser
from app.parsers.markdown_parser import MarkdownDocumentParser
from app.parsers.pdf_parser import PdfDocumentParser
from app.parsers.pptx_parser import PptxDocumentParser
from app.parsers.text_parser import TextDocumentParser
from app.parsers.xlsx_parser import XlsxDocumentParser


class DocumentParserRegistry:
    def __init__(
        self,
        parsers: dict[str, DocumentParser] | None = None,
        *,
        ocr: OcrProvider | None = None,
        pdf_dpi: int = 200,
        pdf_ocr: bool = False,
        pptx_image_ocr: bool = False,
        xlsx_chart_extraction: bool = False,
    ) -> None:
        self._parsers = parsers or {
            ".docx": DocxDocumentParser(),
            ".pdf": PdfDocumentParser(
                ocr=ocr if pdf_ocr else None,
                ocr_dpi=pdf_dpi,
            ),
            ".pptx": PptxDocumentParser(
                ocr=ocr if pptx_image_ocr else None,
            ),
            ".xlsx": XlsxDocumentParser(
                extract_charts=xlsx_chart_extraction,
            ),
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
