"""Structure-aware C# chunking with a safe line-based fallback."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from app.config import load_config
from app.parsers.csharp_parser import (
    CSharpSourceError,
    CSharpSourceFile,
    CSharpSourceScanner,
    CSharpStructureProvider,
    CSharpSymbol,
    LightweightCSharpParser,
    ParsedCSharpFile,
)


class CSharpChunkError(ValueError):
    """Raised when chunking configuration or source input is invalid."""


@dataclass(frozen=True)
class CSharpChunk:
    content: str
    start_line: int
    end_line: int
    namespace: str = ""
    class_name: str = ""
    method_name: str = ""
    kind: str = "context"


@dataclass(frozen=True)
class _Segment:
    start_line: int
    end_line: int
    namespace: str
    class_name: str
    method_name: str
    kind: str


class CSharpChunker:
    """Split parsed C# source while preserving original source substrings."""

    def __init__(
        self,
        max_chars: int,
        overlap_chars: int,
        structure_provider: CSharpStructureProvider | None = None,
    ) -> None:
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
            raise CSharpChunkError("max_chars must be a positive integer")
        if (
            isinstance(overlap_chars, bool)
            or not isinstance(overlap_chars, int)
            or overlap_chars < 0
        ):
            raise CSharpChunkError("overlap_chars must be a non-negative integer")
        if overlap_chars >= max_chars:
            raise CSharpChunkError("overlap_chars must be smaller than max_chars")
        self._max_chars = max_chars
        self._overlap_chars = overlap_chars
        self._structure_provider = structure_provider or LightweightCSharpParser()

    def chunk(self, source: CSharpSourceFile) -> list[CSharpChunk]:
        if not isinstance(source, CSharpSourceFile):
            raise CSharpChunkError("source must be a CSharpSourceFile")
        return self.chunk_parsed(self._structure_provider.parse(source))

    def chunk_parsed(self, parsed: ParsedCSharpFile) -> list[CSharpChunk]:
        """Chunk an interchangeable parsed structure, including future Roslyn data."""

        if not isinstance(parsed, ParsedCSharpFile):
            raise CSharpChunkError("parsed must be a ParsedCSharpFile")
        if not parsed.source.text:
            return []

        lines = parsed.source.text.splitlines(keepends=True)
        if not lines:
            return []
        segments = self._build_segments(parsed, len(lines))
        chunks: list[CSharpChunk] = []
        for segment in segments:
            chunks.extend(self._split_segment(lines, segment))
        return [chunk for chunk in chunks if chunk.content.strip()]

    @staticmethod
    def _build_segments(parsed: ParsedCSharpFile, line_count: int) -> list[_Segment]:
        method_symbols = [
            symbol for symbol in parsed.symbols if symbol.kind == "method"
        ]
        structural = method_symbols or _non_overlapping_regions(parsed.symbols)
        if not structural:
            return [
                _Segment(
                    start_line=1,
                    end_line=line_count,
                    namespace=parsed.namespace,
                    class_name="",
                    method_name="",
                    kind="file",
                )
            ]

        type_symbols = [symbol for symbol in parsed.symbols if symbol.kind == "type"]
        segments: list[_Segment] = []
        cursor = 1
        for symbol in sorted(structural, key=lambda item: (item.start_line, item.end_line)):
            start_line = max(cursor, symbol.start_line)
            if start_line > symbol.end_line:
                continue
            if cursor < start_line:
                segments.append(
                    _context_segment(
                        cursor,
                        start_line - 1,
                        parsed.namespace,
                        type_symbols,
                    )
                )
            segments.append(
                _Segment(
                    start_line=start_line,
                    end_line=min(symbol.end_line, line_count),
                    namespace=symbol.namespace or parsed.namespace,
                    class_name=symbol.class_name,
                    method_name=symbol.method_name,
                    kind=symbol.kind,
                )
            )
            cursor = min(symbol.end_line, line_count) + 1
        if cursor <= line_count:
            segments.append(
                _context_segment(cursor, line_count, parsed.namespace, type_symbols)
            )
        return segments

    def _split_segment(
        self, lines: list[str], segment: _Segment
    ) -> list[CSharpChunk]:
        start_index = segment.start_line - 1
        end_index = segment.end_line
        chunks: list[CSharpChunk] = []
        cursor = start_index

        while cursor < end_index:
            if len(lines[cursor]) > self._max_chars:
                chunks.extend(self._split_long_line(lines[cursor], cursor + 1, segment))
                cursor += 1
                continue

            next_index = cursor
            size = 0
            while next_index < end_index:
                line_size = len(lines[next_index])
                if next_index > cursor and size + line_size > self._max_chars:
                    break
                size += line_size
                next_index += 1
                if size >= self._max_chars:
                    break

            content = "".join(lines[cursor:next_index])
            chunks.append(
                CSharpChunk(
                    content=content,
                    start_line=cursor + 1,
                    end_line=next_index,
                    namespace=segment.namespace,
                    class_name=segment.class_name,
                    method_name=segment.method_name,
                    kind=segment.kind,
                )
            )
            if next_index >= end_index:
                break
            cursor = self._overlap_start(lines, cursor, next_index)
        return chunks

    def _overlap_start(
        self, lines: list[str], current_start: int, next_index: int
    ) -> int:
        if self._overlap_chars == 0:
            return next_index
        overlap_size = 0
        overlap_start = next_index
        while overlap_start > current_start + 1:
            candidate_size = len(lines[overlap_start - 1])
            if overlap_size + candidate_size > self._overlap_chars:
                break
            overlap_start -= 1
            overlap_size += candidate_size
        return overlap_start if overlap_start < next_index else next_index

    def _split_long_line(
        self, line: str, line_number: int, segment: _Segment
    ) -> list[CSharpChunk]:
        chunks: list[CSharpChunk] = []
        step = self._max_chars - self._overlap_chars
        start = 0
        while start < len(line):
            content = line[start : start + self._max_chars]
            chunks.append(
                CSharpChunk(
                    content=content,
                    start_line=line_number,
                    end_line=line_number,
                    namespace=segment.namespace,
                    class_name=segment.class_name,
                    method_name=segment.method_name,
                    kind=segment.kind,
                )
            )
            if start + self._max_chars >= len(line):
                break
            start += step
        return chunks


def _non_overlapping_regions(symbols: tuple[CSharpSymbol, ...]) -> list[CSharpSymbol]:
    selected: list[CSharpSymbol] = []
    for symbol in sorted(
        (item for item in symbols if item.kind == "region"),
        key=lambda item: (item.start_line, -item.end_line),
    ):
        if any(
            existing.start_line <= symbol.start_line
            and existing.end_line >= symbol.end_line
            for existing in selected
        ):
            continue
        selected.append(symbol)
    return selected


def _context_segment(
    start_line: int,
    end_line: int,
    namespace: str,
    type_symbols: list[CSharpSymbol],
) -> _Segment:
    containing_types = [
        symbol
        for symbol in type_symbols
        if symbol.start_line <= start_line and symbol.end_line >= end_line
    ]
    containing_type = min(
        containing_types,
        key=lambda symbol: symbol.end_line - symbol.start_line,
        default=None,
    )
    return _Segment(
        start_line=start_line,
        end_line=end_line,
        namespace=namespace,
        class_name=containing_type.class_name if containing_type else "",
        method_name="",
        kind="context",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan and chunk local C# source files")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--source",
        help="Optional source root override; useful for synthetic test data",
    )
    parser.add_argument(
        "--show-content",
        action="store_true",
        help="Include source content in output; disabled by default for safety",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    source_config = config.source
    if args.source:
        source_config = replace(
            source_config,
            path=Path(args.source).expanduser().resolve(strict=False),
        )

    try:
        sources = CSharpSourceScanner(source_config).scan()
        chunker = CSharpChunker(
            source_config.chunk_size,
            source_config.chunk_overlap,
        )
        files = []
        total_chunks = 0
        for source in sources:
            chunks = chunker.chunk(source)
            total_chunks += len(chunks)
            chunk_payloads = []
            for index, chunk in enumerate(chunks):
                payload = {
                    "chunk_index": index,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "namespace": chunk.namespace,
                    "class_name": chunk.class_name,
                    "method_name": chunk.method_name,
                    "kind": chunk.kind,
                    "character_count": len(chunk.content),
                }
                if args.show_content:
                    payload["content"] = chunk.content
                chunk_payloads.append(payload)
            files.append(
                {
                    "relative_path": source.relative_path,
                    "encoding": source.encoding,
                    "file_hash": source.file_hash,
                    "modified_time": source.modified_time,
                    "chunks": chunk_payloads,
                }
            )
    except (CSharpSourceError, CSharpChunkError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "source_root": str(source_config.path),
                "file_count": len(sources),
                "chunk_count": total_chunks,
                "files": files,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
