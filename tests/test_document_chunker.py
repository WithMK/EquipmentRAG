from __future__ import annotations

import unittest

from app.chunkers.document_chunker import DocumentChunker
from app.models.document_models import DocumentBlock, NormalizedDocument


def _document(blocks: tuple[DocumentBlock, ...]) -> NormalizedDocument:
    return NormalizedDocument(
        document_id="loader-spec",
        source_path="Specs/Loader.docx",
        file_name="Loader.docx",
        file_extension=".docx",
        title="Loader Specification",
        blocks=blocks,
        equipment="Trimming",
        revision="Rev.3",
        file_hash="abc",
    )


class DocumentChunkerTests(unittest.TestCase):
    def test_preserves_heading_path_table_and_page_traceability(self) -> None:
        document = _document(
            (
                DocumentBlock("heading", "Auto Sequence", level=1, page=3),
                DocumentBlock("heading", "Loader", level=2, page=3),
                DocumentBlock("paragraph", "Vacuum sensor must be ON.", page=3),
                DocumentBlock(
                    "table",
                    "Signal | Condition\nVacuum | ON",
                    page=3,
                    rows=(("Signal", "Condition"), ("Vacuum", "ON")),
                ),
            )
        )

        chunks = DocumentChunker(500, 50).chunk(document)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].heading_path, ("Auto Sequence", "Loader"))
        self.assertEqual(chunks[0].section, "Auto Sequence")
        self.assertEqual(chunks[0].subsection, "Loader")
        self.assertEqual(chunks[0].page, 3)
        self.assertIn("Section: Auto Sequence > Loader", chunks[0].content)
        self.assertIn("Table:", chunks[0].content)

    def test_splits_only_long_sections_and_keeps_heading_context(self) -> None:
        document = _document(
            (
                DocumentBlock("heading", "Troubleshooting", level=1),
                DocumentBlock("paragraph", "Vacuum " * 100),
            )
        )

        chunks = DocumentChunker(180, 30).chunk(document)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.content) <= 180 for chunk in chunks))
        self.assertTrue(
            all("Section: Troubleshooting" in chunk.content for chunk in chunks)
        )
        self.assertTrue(all(chunk.content_hash for chunk in chunks))


if __name__ == "__main__":
    unittest.main()

