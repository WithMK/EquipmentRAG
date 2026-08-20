"""C# source discovery and lightweight structural parsing without Roslyn."""

from __future__ import annotations

import codecs
import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from app.config import SourceConfig


class CSharpSourceError(RuntimeError):
    """Raised when C# source discovery, decoding, or parsing fails."""


@dataclass(frozen=True)
class CSharpSourceFile:
    path: Path
    relative_path: str
    text: str
    encoding: str
    file_hash: str
    modified_time: str


@dataclass(frozen=True)
class CSharpSymbol:
    kind: str
    name: str
    start_line: int
    end_line: int
    namespace: str = ""
    class_name: str = ""
    method_name: str = ""


@dataclass(frozen=True)
class ParsedCSharpFile:
    source: CSharpSourceFile
    namespace: str
    symbols: tuple[CSharpSymbol, ...]


class CSharpStructureProvider(Protocol):
    """Interface that a future Roslyn-backed parser can implement."""

    def parse(self, source: CSharpSourceFile) -> ParsedCSharpFile: ...


class CSharpSourceScanner:
    """Recursively discover and decode C# files under a configured root."""

    def __init__(self, config: SourceConfig) -> None:
        self._config = config
        self._extensions = {
            extension.casefold()
            if extension.startswith(".")
            else f".{extension.casefold()}"
            for extension in config.include_extensions
        }
        self._excluded = {name.casefold() for name in config.exclude_directories}

    def scan(self) -> list[CSharpSourceFile]:
        root = self._config.path.resolve(strict=False)
        if not root.is_dir():
            raise CSharpSourceError(f"C# source directory not found: {root}")

        discovered: list[Path] = []
        for current_root, directories, file_names in os.walk(root, topdown=True):
            current_path = Path(current_root)
            directories[:] = sorted(
                (
                    name
                    for name in directories
                    if name.casefold() not in self._excluded
                    and not (current_path / name).is_symlink()
                ),
                key=str.casefold,
            )
            for file_name in sorted(file_names, key=str.casefold):
                path = current_path / file_name
                if path.is_symlink() or path.suffix.casefold() not in self._extensions:
                    continue
                discovered.append(path)

        return [self._read_source(path, root) for path in discovered]

    @staticmethod
    def _read_source(path: Path, root: Path) -> CSharpSourceFile:
        try:
            raw = path.read_bytes()
            stat = path.stat()
        except OSError as exc:
            raise CSharpSourceError(f"Unable to read C# source file: {path}") from exc

        text, encoding = _decode_csharp(raw, path)
        modified_time = datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat().replace("+00:00", "Z")
        return CSharpSourceFile(
            path=path.resolve(strict=False),
            relative_path=path.relative_to(root).as_posix(),
            text=text,
            encoding=encoding,
            file_hash=hashlib.sha256(raw).hexdigest(),
            modified_time=modified_time,
        )


def _decode_csharp(raw: bytes, path: Path) -> tuple[str, str]:
    if raw.startswith(codecs.BOM_UTF8):
        candidates = ("utf-8-sig",)
    elif raw.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        candidates = ("utf-32",)
    elif raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        candidates = ("utf-16",)
    else:
        candidates = ("utf-8", "cp949")

    for encoding in candidates:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise CSharpSourceError(
        f"Unsupported C# source encoding for {path}; expected UTF-8, UTF-16, or CP949"
    )


_NAMESPACE_PATTERN = re.compile(
    r"\bnamespace\s+(?P<name>@?[A-Za-z_]\w*(?:\.@?[A-Za-z_]\w*)*)\s*(?:;|\{)"
)
_TYPE_PATTERN = re.compile(
    r"(?m)^[ \t]*(?:(?:public|private|protected|internal|static|abstract|sealed|"
    r"partial|readonly|ref|unsafe|new)\s+)*(?:class|struct|interface|enum|"
    r"record(?:\s+(?:class|struct))?)\s+(?P<name>@?[A-Za-z_]\w*)[^;{]*\{"
)
_CALLABLE_PATTERN = re.compile(
    r"(?m)^[ \t]*(?:\[[^\]\r\n]+\][ \t]*(?:\r?\n[ \t]*)?)*"
    r"(?:(?:public|private|protected|internal|static|virtual|override|abstract|"
    r"sealed|async|extern|unsafe|new|partial|readonly)\s+)*"
    r"(?:[A-Za-z_@][\w@.<>,?\[\] \t]*[ \t]+)?"
    r"(?P<name>@?[A-Za-z_]\w*)[ \t]*(?:<[^>{}\r\n]+>)?[ \t]*"
    r"\([^;{}]*\)\s*(?:where[^{}\r\n]+)?\s*\{"
)
_CONTROL_KEYWORDS = {
    "catch",
    "checked",
    "do",
    "else",
    "fixed",
    "for",
    "foreach",
    "if",
    "lock",
    "switch",
    "unchecked",
    "using",
    "while",
}
_REGION_START = re.compile(r"^\s*#region(?:\s+(?P<name>.*?))?\s*$")
_REGION_END = re.compile(r"^\s*#endregion(?:\s+.*?)?\s*$")


class LightweightCSharpParser:
    """Extract namespace, type, method, and region line spans conservatively."""

    def parse(self, source: CSharpSourceFile) -> ParsedCSharpFile:
        masked = _mask_non_code(source.text)
        namespace_match = _NAMESPACE_PATTERN.search(masked)
        namespace = namespace_match.group("name") if namespace_match else ""
        line_starts = _line_starts(source.text)

        type_symbols = self._parse_types(masked, line_starts, namespace)
        callable_symbols = self._parse_callables(
            source.text, masked, line_starts, namespace, type_symbols
        )
        region_symbols = self._parse_regions(masked, namespace, type_symbols)
        symbols = sorted(
            (*type_symbols, *callable_symbols, *region_symbols),
            key=lambda symbol: (symbol.start_line, symbol.end_line, symbol.kind),
        )
        return ParsedCSharpFile(
            source=source,
            namespace=namespace,
            symbols=tuple(symbols),
        )

    @staticmethod
    def _parse_types(
        masked: str, line_starts: list[int], namespace: str
    ) -> list[CSharpSymbol]:
        symbols: list[CSharpSymbol] = []
        for match in _TYPE_PATTERN.finditer(masked):
            opening_brace = masked.rfind("{", match.start(), match.end())
            closing_brace = _matching_brace(masked, opening_brace)
            if closing_brace is None:
                continue
            symbols.append(
                CSharpSymbol(
                    kind="type",
                    name=match.group("name"),
                    start_line=_offset_to_line(line_starts, match.start()),
                    end_line=_offset_to_line(line_starts, closing_brace),
                    namespace=namespace,
                    class_name=match.group("name"),
                )
            )
        return symbols

    @staticmethod
    def _parse_callables(
        text: str,
        masked: str,
        line_starts: list[int],
        namespace: str,
        type_symbols: list[CSharpSymbol],
    ) -> list[CSharpSymbol]:
        symbols: list[CSharpSymbol] = []
        text_lines = text.splitlines()
        for match in _CALLABLE_PATTERN.finditer(masked):
            method_name = match.group("name")
            if method_name.lstrip("@").casefold() in _CONTROL_KEYWORDS:
                continue
            opening_brace = masked.rfind("{", match.start(), match.end())
            closing_brace = _matching_brace(masked, opening_brace)
            if closing_brace is None:
                continue

            declaration_line = _offset_to_line(line_starts, match.start())
            start_line = _include_leading_context(text_lines, declaration_line)
            end_line = _offset_to_line(line_starts, closing_brace)
            containing_type = _innermost_type(type_symbols, declaration_line, end_line)
            class_name = containing_type.class_name if containing_type else ""
            symbols.append(
                CSharpSymbol(
                    kind="method",
                    name=method_name,
                    start_line=start_line,
                    end_line=end_line,
                    namespace=namespace,
                    class_name=class_name,
                    method_name=method_name,
                )
            )
        return _outermost_non_overlapping(symbols)

    @staticmethod
    def _parse_regions(
        text: str, namespace: str, type_symbols: list[CSharpSymbol]
    ) -> list[CSharpSymbol]:
        regions: list[CSharpSymbol] = []
        stack: list[tuple[str, int]] = []
        lines = text.splitlines()
        for line_number, line in enumerate(lines, start=1):
            start_match = _REGION_START.match(line)
            if start_match:
                stack.append(((start_match.group("name") or "region").strip(), line_number))
                continue
            if _REGION_END.match(line) and stack:
                name, start_line = stack.pop()
                containing_type = _innermost_type(
                    type_symbols, start_line, line_number
                )
                regions.append(
                    CSharpSymbol(
                        kind="region",
                        name=name,
                        start_line=start_line,
                        end_line=line_number,
                        namespace=namespace,
                        class_name=(
                            containing_type.class_name if containing_type else ""
                        ),
                    )
                )
        for name, start_line in stack:
            containing_type = _innermost_type(type_symbols, start_line, len(lines))
            regions.append(
                CSharpSymbol(
                    kind="region",
                    name=name,
                    start_line=start_line,
                    end_line=len(lines),
                    namespace=namespace,
                    class_name=containing_type.class_name if containing_type else "",
                )
            )
        return regions


def _line_starts(text: str) -> list[int]:
    starts = [0]
    starts.extend(match.end() for match in re.finditer(r"\n", text))
    return starts


def _offset_to_line(line_starts: list[int], offset: int) -> int:
    import bisect

    return bisect.bisect_right(line_starts, offset)


def _matching_brace(masked: str, opening_brace: int) -> int | None:
    if opening_brace < 0 or opening_brace >= len(masked) or masked[opening_brace] != "{":
        return None
    depth = 0
    for offset in range(opening_brace, len(masked)):
        character = masked[offset]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return offset
    return None


def _include_leading_context(lines: list[str], declaration_line: int) -> int:
    start = declaration_line
    cursor = declaration_line - 2
    inside_block_comment = False
    while cursor >= 0:
        stripped = lines[cursor].strip()
        if not stripped:
            break
        if stripped.endswith("*/"):
            inside_block_comment = True
        is_context = (
            inside_block_comment
            or stripped.startswith(("///", "//", "/*", "*", "["))
        )
        if not is_context:
            break
        start = cursor + 1
        if stripped.startswith("/*"):
            inside_block_comment = False
        cursor -= 1
    return start


def _innermost_type(
    type_symbols: list[CSharpSymbol], start_line: int, end_line: int
) -> CSharpSymbol | None:
    candidates = [
        symbol
        for symbol in type_symbols
        if symbol.start_line <= start_line and symbol.end_line >= end_line
    ]
    return min(
        candidates,
        key=lambda symbol: symbol.end_line - symbol.start_line,
        default=None,
    )


def _outermost_non_overlapping(symbols: list[CSharpSymbol]) -> list[CSharpSymbol]:
    selected: list[CSharpSymbol] = []
    for symbol in sorted(
        symbols, key=lambda item: (item.start_line, -item.end_line, item.name)
    ):
        if any(
            existing.start_line <= symbol.start_line
            and existing.end_line >= symbol.end_line
            for existing in selected
        ):
            continue
        selected.append(symbol)
    return selected


def _mask_non_code(text: str) -> str:
    """Replace comments and literals with spaces while preserving offsets/newlines."""

    masked = list(text)
    length = len(text)
    index = 0

    def blank(start: int, end: int) -> None:
        for position in range(start, end):
            if masked[position] not in ("\r", "\n"):
                masked[position] = " "

    while index < length:
        if text.startswith("//", index):
            end = text.find("\n", index + 2)
            end = length if end < 0 else end
            blank(index, end)
            index = end
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = length if end < 0 else end + 2
            blank(index, end)
            index = end
            continue

        raw_match = re.match(r'\$?"{3,}', text[index:])
        if raw_match:
            delimiter = '"' * (len(raw_match.group(0)) - raw_match.group(0).count("$"))
            end = text.find(delimiter, index + len(raw_match.group(0)))
            end = length if end < 0 else end + len(delimiter)
            blank(index, end)
            index = end
            continue

        verbatim_prefix_length = 0
        for prefix in ('$@"', '@$"', '@"'):
            if text.startswith(prefix, index):
                verbatim_prefix_length = len(prefix)
                break
        if verbatim_prefix_length:
            cursor = index + verbatim_prefix_length
            while cursor < length:
                if text.startswith('""', cursor):
                    cursor += 2
                    continue
                if text[cursor] == '"':
                    cursor += 1
                    break
                cursor += 1
            blank(index, cursor)
            index = cursor
            continue

        regular_prefix_length = 2 if text.startswith('$"', index) else 1
        if text.startswith('"', index) or text.startswith('$"', index):
            cursor = index + regular_prefix_length
            while cursor < length:
                if text[cursor] == "\\":
                    cursor += 2
                    continue
                if cursor < length and text[cursor] == '"':
                    cursor += 1
                    break
                cursor += 1
            blank(index, min(cursor, length))
            index = min(cursor, length)
            continue

        if text[index] == "'":
            cursor = index + 1
            while cursor < length:
                if text[cursor] == "\\":
                    cursor += 2
                    continue
                if cursor < length and text[cursor] == "'":
                    cursor += 1
                    break
                cursor += 1
            blank(index, min(cursor, length))
            index = min(cursor, length)
            continue
        index += 1
    return "".join(masked)
