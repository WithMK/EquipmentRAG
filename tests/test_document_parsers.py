from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.config import DocumentConfig
from app.parsers.document_parsers import DocumentParserRegistry
from app.parsers.document_scanner import DocumentScanner


class DocumentParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = DocumentConfig(
            enabled=True,
            source_paths=(self.root,),
            extensions=(".docx", ".pdf", ".pptx", ".xlsx", ".md", ".txt"),
            exclude_directories=("archive",),
            chunk_size=500,
            chunk_overlap=50,
            collection_name="document_chunks",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_scanner_hashes_files_and_loads_revision_sidecar(self) -> None:
        path = self.root / "Loader.md"
        path.write_text("# Loader\n\nVacuum interlock", encoding="utf-8")
        path.with_name("Loader.md.metadata.yaml").write_text(
            "document_id: loader-spec\n"
            "revision: Rev.3\n"
            "document_status: active\n"
            "is_latest: true\n",
            encoding="utf-8",
        )
        archive = self.root / "archive"
        archive.mkdir()
        (archive / "old.txt").write_text("obsolete", encoding="utf-8")

        sources = DocumentScanner(self.config).scan()

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].relative_path, "Loader.md")
        self.assertEqual(len(sources[0].file_hash), 64)
        self.assertEqual(sources[0].metadata["revision"], "Rev.3")

    def test_markdown_preserves_headings_lists_and_code_blocks(self) -> None:
        path = self.root / "Manual.md"
        path.write_text(
            "# Trimming\n\n## Loader\n\n- Vacuum sensor ON\n"
            "- Door closed\n\n```csharp\nVacuum.On();\n```\n",
            encoding="utf-8",
        )
        source = DocumentScanner(self.config).scan()[0]

        document = DocumentParserRegistry().parse(source)

        self.assertEqual(document.title, "Trimming")
        self.assertEqual(
            [block.type for block in document.blocks],
            ["heading", "heading", "list", "code"],
        )
        self.assertIn("Vacuum.On", document.blocks[-1].text)

    def test_txt_uses_section_patterns_before_paragraph_fallback(self) -> None:
        path = self.root / "Trouble.txt"
        path.write_text(
            "1. Loader Alarm\nVacuum sensor was not detected.\n\n"
            "ACTION:\nCheck the vacuum sensor.",
            encoding="utf-8",
        )
        source = DocumentScanner(self.config).scan()[0]

        document = DocumentParserRegistry().parse(source)

        self.assertEqual(document.blocks[0].type, "heading")
        self.assertEqual(document.blocks[1].type, "paragraph")
        self.assertEqual(document.blocks[2].text, "ACTION")

    def test_docx_preserves_heading_paragraph_list_and_table(self) -> None:
        from docx import Document

        path = self.root / "Specification.docx"
        document = Document()
        document.core_properties.title = "Loader Specification"
        document.add_heading("Loader Interlock", level=1)
        document.add_paragraph("Vacuum sensor must be ON.")
        document.add_paragraph("Door must be closed.", style="List Bullet")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Signal"
        table.cell(0, 1).text = "Condition"
        table.cell(1, 0).text = "Vacuum"
        table.cell(1, 1).text = "ON"
        document.save(path)
        source = DocumentScanner(self.config).scan()[0]

        normalized = DocumentParserRegistry().parse(source)

        self.assertEqual(normalized.title, "Loader Specification")
        self.assertEqual(
            [block.type for block in normalized.blocks],
            ["heading", "paragraph", "list", "table"],
        )
        self.assertEqual(normalized.blocks[-1].rows[1], ("Vacuum", "ON"))

    def test_text_pdf_preserves_page_and_removes_repeated_margins(self) -> None:
        path = self.root / "Alarm.pdf"
        _write_pdf(
            path,
            [
                ["EQUIPMENT MANUAL", "1 Loader", "Vacuum timeout", "CONFIDENTIAL"],
                ["EQUIPMENT MANUAL", "2 Recovery", "Check sensor", "CONFIDENTIAL"],
            ],
        )
        source = DocumentScanner(self.config).scan()[0]

        document = DocumentParserRegistry().parse(source)

        text = "\n".join(block.text for block in document.blocks)
        self.assertNotIn("EQUIPMENT MANUAL", text)
        self.assertNotIn("CONFIDENTIAL", text)
        self.assertEqual({block.page for block in document.blocks}, {1, 2})
        self.assertIn("Vacuum timeout", text)

    def test_pptx_preserves_slides_tables_and_speaker_notes(self) -> None:
        from pptx import Presentation
        from pptx.util import Inches

        path = self.root / "DesignReview.pptx"
        presentation = Presentation()
        presentation.core_properties.title = "Loader Design Review"
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = "Loader Safety"
        body = slide.placeholders[1].text_frame
        body.text = "Safety door must be closed."
        bullet = body.add_paragraph()
        bullet.text = "Vacuum sensor must be ON."
        bullet.level = 1
        table = slide.shapes.add_table(
            2,
            2,
            Inches(1),
            Inches(4),
            Inches(6),
            Inches(1),
        ).table
        table.cell(0, 0).text = "Signal"
        table.cell(0, 1).text = "Condition"
        table.cell(1, 0).text = "Vacuum"
        table.cell(1, 1).text = "ON"
        slide.notes_slide.notes_text_frame.text = "Verify PLC input X100."
        presentation.save(path)
        source = DocumentScanner(self.config).scan()[0]

        document = DocumentParserRegistry().parse(source)

        self.assertEqual(document.title, "Loader Design Review")
        self.assertTrue(all(block.slide == 1 for block in document.blocks))
        self.assertEqual(document.blocks[0].text, "Loader Safety")
        self.assertTrue(any(block.type == "table" for block in document.blocks))
        self.assertTrue(
            any("Verify PLC input X100" in block.text for block in document.blocks)
        )

    def test_xlsx_preserves_sheets_regions_formulas_and_cell_ranges(self) -> None:
        from openpyxl import Workbook

        path = self.root / "Signals.xlsx"
        workbook = Workbook()
        workbook.properties.title = "Loader Signal List"
        sheet = workbook.active
        sheet.title = "Loader IO"
        sheet.append(("Signal", "Description"))
        sheet.append(("X100", "Vacuum sensor"))
        sheet.append(("Result", "=COUNTA(A2:A2)"))
        sheet.append((None, None))
        sheet.append(("Alarm", "Recovery"))
        sheet.append(("AL101", "Check vacuum"))
        workbook.save(path)
        source = DocumentScanner(self.config).scan()[0]

        document = DocumentParserRegistry().parse(source)

        self.assertEqual(document.title, "Loader Signal List")
        tables = [block for block in document.blocks if block.type == "table"]
        self.assertEqual([block.cell_range for block in tables], ["A1:B3", "A5:B6"])
        self.assertTrue(all(block.sheet == "Loader IO" for block in tables))
        self.assertIn("Formula: =COUNTA(A2:A2)", tables[0].text)
        self.assertIn("AL101", tables[1].text)


def _write_pdf(path: Path, page_lines: list[list[str]]) -> None:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    for lines in page_lines:
        page = writer.add_blank_page(width=612, height=792)
        resources = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font_ref}
                )
            }
        )
        page[NameObject("/Resources")] = resources
        commands = ["BT", "/F1 12 Tf", "72 740 Td", "14 TL"]
        for index, line in enumerate(lines):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            if index:
                commands.append("T*")
            commands.append(f"({escaped}) Tj")
        commands.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(commands).encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)


if __name__ == "__main__":
    unittest.main()
