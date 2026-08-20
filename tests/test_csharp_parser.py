from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from app.config import SourceConfig
from app.parsers.csharp_parser import (
    CSharpSourceFile,
    CSharpSourceScanner,
    LightweightCSharpParser,
)


SAMPLE_SOURCE = """namespace Factory.Motion;

public class AxisController
{
    #region Homing
    /// <summary>
    /// Z축 원점 복귀를 수행한다.
    /// </summary>
    [System.Obsolete]
    public async Task HomeAsync()
    {
        var ignored = \"brace } and // comment\";
        await zAxis.MoveHomeAsync();
    }
    #endregion

    public AxisController()
    {
        var character = '}';
    }
}
"""


class CSharpParserTests(unittest.TestCase):
    def _config(self, root: Path) -> SourceConfig:
        return SourceConfig(
            path=root,
            include_extensions=(".cs",),
            exclude_directories=("bin", "obj", ".git"),
            chunk_size=4000,
            chunk_overlap=400,
        )

    def test_scans_recursively_filters_directories_and_decodes_common_encodings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            utf8_path = root / "Motion" / "AxisController.cs"
            utf8_path.parent.mkdir()
            utf8_bytes = SAMPLE_SOURCE.encode("utf-8-sig")
            utf8_path.write_bytes(utf8_bytes)

            cp949_path = root / "Legacy.CS"
            cp949_path.write_bytes("class 레거시 { }".encode("cp949"))

            excluded_path = root / "BIN" / "Generated.cs"
            excluded_path.parent.mkdir()
            excluded_path.write_text("class Generated { }", encoding="utf-8")
            (root / "notes.txt").write_text("not C#", encoding="utf-8")

            sources = CSharpSourceScanner(self._config(root)).scan()

            self.assertEqual(
                [source.relative_path for source in sources],
                ["Legacy.CS", "Motion/AxisController.cs"],
            )
            by_name = {source.path.name: source for source in sources}
            self.assertEqual(by_name["AxisController.cs"].encoding, "utf-8-sig")
            self.assertEqual(
                by_name["AxisController.cs"].file_hash,
                hashlib.sha256(utf8_bytes).hexdigest(),
            )
            self.assertIn("레거시", by_name["Legacy.CS"].text)
            self.assertEqual(by_name["Legacy.CS"].encoding, "cp949")

    def test_extracts_namespace_type_methods_regions_and_leading_comments(self) -> None:
        source = CSharpSourceFile(
            path=Path("AxisController.cs"),
            relative_path="AxisController.cs",
            text=SAMPLE_SOURCE,
            encoding="utf-8",
            file_hash="hash",
            modified_time="2026-01-01T00:00:00Z",
        )

        parsed = LightweightCSharpParser().parse(source)

        self.assertEqual(parsed.namespace, "Factory.Motion")
        type_symbol = next(symbol for symbol in parsed.symbols if symbol.kind == "type")
        self.assertEqual(type_symbol.class_name, "AxisController")
        self.assertEqual(type_symbol.end_line, 21)

        methods = [symbol for symbol in parsed.symbols if symbol.kind == "method"]
        self.assertEqual(
            [symbol.method_name for symbol in methods],
            ["HomeAsync", "AxisController"],
        )
        self.assertEqual(methods[0].start_line, 6)
        self.assertEqual(methods[0].end_line, 14)
        self.assertTrue(all(symbol.class_name == "AxisController" for symbol in methods))

        region = next(symbol for symbol in parsed.symbols if symbol.kind == "region")
        self.assertEqual(region.name, "Homing")
        self.assertEqual((region.start_line, region.end_line), (5, 15))


if __name__ == "__main__":
    unittest.main()
