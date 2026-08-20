from __future__ import annotations

import unittest
from pathlib import Path

from app.chunkers.csharp_chunker import CSharpChunkError, CSharpChunker
from app.parsers.csharp_parser import (
    CSharpSourceFile,
    CSharpSymbol,
    ParsedCSharpFile,
)


def _source(text: str) -> CSharpSourceFile:
    return CSharpSourceFile(
        path=Path("Machine.cs"),
        relative_path="Machine.cs",
        text=text,
        encoding="utf-8",
        file_hash="hash",
        modified_time="2026-01-01T00:00:00Z",
    )


class FakeStructureProvider:
    def __init__(self, parsed: ParsedCSharpFile) -> None:
        self.parsed = parsed
        self.calls: list[CSharpSourceFile] = []

    def parse(self, source: CSharpSourceFile) -> ParsedCSharpFile:
        self.calls.append(source)
        return self.parsed


class CSharpChunkerTests(unittest.TestCase):
    def test_chunks_methods_with_original_source_and_structure_metadata(self) -> None:
        text = """namespace Factory;
public class Machine
{
    /// <summary>Home axis</summary>
    public void Home()
    {
        axis.Prepare();
        axis.MoveHome();
        axis.Wait();
    }

    public void Reset()
    {
        alarm.Reset();
    }
}
"""
        source = _source(text)
        chunks = CSharpChunker(max_chars=90, overlap_chars=20).chunk(source)

        self.assertGreaterEqual(len(chunks), 4)
        self.assertTrue(all(chunk.content in text for chunk in chunks))
        self.assertTrue(all(len(chunk.content) <= 90 for chunk in chunks))
        home_chunks = [chunk for chunk in chunks if chunk.method_name == "Home"]
        reset_chunks = [chunk for chunk in chunks if chunk.method_name == "Reset"]
        self.assertTrue(home_chunks)
        self.assertTrue(reset_chunks)
        self.assertTrue(all(chunk.namespace == "Factory" for chunk in chunks))
        self.assertTrue(all(chunk.class_name == "Machine" for chunk in home_chunks))
        self.assertIn("/// <summary>", "".join(chunk.content for chunk in home_chunks))

    def test_uses_file_fallback_and_splits_a_long_line_safely(self) -> None:
        text = "class Settings { string Data = \"" + ("x" * 150) + "\"; }"
        chunks = CSharpChunker(max_chars=50, overlap_chars=10).chunk(_source(text))

        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(chunk.kind == "file" for chunk in chunks))
        self.assertTrue(all(chunk.start_line == chunk.end_line == 1 for chunk in chunks))
        self.assertTrue(all(len(chunk.content) <= 50 for chunk in chunks))
        self.assertTrue(all(chunk.content in text for chunk in chunks))

    def test_accepts_interchangeable_structure_provider_for_future_roslyn_data(self) -> None:
        source = _source("class Machine { void Run() { } }")
        parsed = ParsedCSharpFile(
            source=source,
            namespace="Factory",
            symbols=(
                CSharpSymbol(
                    kind="method",
                    name="Run",
                    start_line=1,
                    end_line=1,
                    namespace="Factory",
                    class_name="Machine",
                    method_name="Run",
                ),
            ),
        )
        provider = FakeStructureProvider(parsed)

        chunks = CSharpChunker(100, 10, provider).chunk(source)

        self.assertEqual(provider.calls, [source])
        self.assertEqual(chunks[0].method_name, "Run")
        self.assertEqual(chunks[0].class_name, "Machine")

    def test_rejects_invalid_overlap(self) -> None:
        with self.assertRaisesRegex(CSharpChunkError, "smaller"):
            CSharpChunker(max_chars=100, overlap_chars=100)


if __name__ == "__main__":
    unittest.main()
