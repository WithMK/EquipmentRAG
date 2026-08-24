"""Heading-aware structural chunking for normalized documents."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.document_models import DocumentBlock, DocumentChunk, NormalizedDocument


class DocumentChunkError(ValueError):
    """Raised when document chunking settings or input are invalid."""


@dataclass(frozen=True)
class _Group:
    heading_path: tuple[str, ...]
    blocks: tuple[DocumentBlock, ...]
    page: int
    slide: int
    sheet: str


class DocumentChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise DocumentChunkError("chunk_size must be a positive integer")
        if chunk_size <= 0:
            raise DocumentChunkError("chunk_size must be a positive integer")
        if (
            isinstance(chunk_overlap, bool)
            or not isinstance(chunk_overlap, int)
            or chunk_overlap < 0
        ):
            raise DocumentChunkError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise DocumentChunkError("chunk_overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: NormalizedDocument) -> list[DocumentChunk]:
        groups = self._group_blocks(document.blocks)
        chunks: list[DocumentChunk] = []
        for group in groups:
            prefix = self._context_prefix(document.title, group.heading_path)
            available = max(1, self._chunk_size - len(prefix) - 2)
            bodies = self._split_blocks(group.blocks, available)
            for body in bodies:
                content = f"{prefix}\n\n{body}" if prefix else body
                chunks.append(
                    DocumentChunk.create(
                        content,
                        heading_path=group.heading_path,
                        page=group.page,
                        slide=group.slide,
                        sheet=group.sheet,
                    )
                )
        return chunks

    @staticmethod
    def _group_blocks(blocks: tuple[DocumentBlock, ...]) -> list[_Group]:
        heading_levels: dict[int, str] = {}
        current: list[DocumentBlock] = []
        groups: list[_Group] = []
        current_location = (0, 0, "")

        def flush() -> None:
            if not current:
                return
            path = tuple(heading_levels[level] for level in sorted(heading_levels))
            groups.append(
                _Group(path, tuple(current), *current_location)
            )
            current.clear()

        for block in blocks:
            if block.type == "heading":
                flush()
                level = max(1, block.level)
                for old_level in tuple(heading_levels):
                    if old_level >= level:
                        del heading_levels[old_level]
                heading_levels[level] = block.text.strip()
                current_location = (block.page, block.slide, block.sheet)
                continue
            location = (block.page, block.slide, block.sheet)
            if current and location != current_location and any(location):
                flush()
            if not current:
                current_location = location
            current.append(block)
        flush()
        return groups

    @staticmethod
    def _context_prefix(title: str, heading_path: tuple[str, ...]) -> str:
        lines = [f"Document: {title}"] if title else []
        if heading_path:
            lines.append("Section: " + " > ".join(heading_path))
        return "\n".join(lines)

    def _split_blocks(
        self, blocks: tuple[DocumentBlock, ...], limit: int
    ) -> list[str]:
        units = [self._format_block(block) for block in blocks]
        output: list[str] = []
        current = ""
        for unit in units:
            if len(unit) > limit:
                if current:
                    output.append(current)
                    current = ""
                output.extend(self._split_long_text(unit, limit))
                continue
            candidate = f"{current}\n\n{unit}" if current else unit
            if len(candidate) <= limit:
                current = candidate
            else:
                output.append(current)
                overlap = current[-self._chunk_overlap :] if self._chunk_overlap else ""
                candidate = f"{overlap}\n\n{unit}" if overlap else unit
                current = candidate if len(candidate) <= limit else unit
        if current:
            output.append(current)
        return output

    def _split_long_text(self, text: str, limit: int) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + limit)
            if end < len(text):
                boundary = text.rfind("\n", start, end)
                if boundary <= start:
                    boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(end - self._chunk_overlap, start + 1)
        return [value for value in chunks if value]

    @staticmethod
    def _format_block(block: DocumentBlock) -> str:
        if block.type == "table":
            return "Table:\n" + block.text
        if block.type == "list":
            return "List:\n" + block.text
        if block.type == "code":
            return "Code:\n```\n" + block.text + "\n```"
        return block.text

