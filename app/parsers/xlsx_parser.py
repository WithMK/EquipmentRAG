"""XLSX parser with sheet, table-region, and cell-range traceability."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

from app.models.document_models import (
    DocumentBlock,
    DocumentSourceFile,
    NormalizedDocument,
)
from app.parsers.document_parser import (
    DocumentParseError,
    build_normalized_document,
)


_MAX_SCANNED_CELLS = 1_000_000


@dataclass(frozen=True)
class _Region:
    row_start: int
    row_end: int
    column_start: int
    column_end: int
    rows: tuple[tuple[str, ...], ...]


class XlsxDocumentParser:
    def parse(self, source: DocumentSourceFile) -> NormalizedDocument:
        try:
            from openpyxl import load_workbook
            from openpyxl.utils import get_column_letter
        except ImportError as exc:
            raise DocumentParseError(
                "openpyxl is required to parse XLSX documents"
            ) from exc

        formula_book = None
        value_book = None
        try:
            formula_book = load_workbook(
                source.path,
                read_only=True,
                data_only=False,
            )
            value_book = load_workbook(
                source.path,
                read_only=True,
                data_only=True,
            )
            blocks = self._blocks(formula_book, value_book, get_column_letter)
            properties = formula_book.properties
            title = _string(properties.title)
            created = _date_string(properties.created)
        except DocumentParseError:
            raise
        except Exception as exc:
            raise DocumentParseError(
                f"Unable to parse XLSX: {source.path}"
            ) from exc
        finally:
            if formula_book is not None:
                formula_book.close()
            if value_book is not None:
                value_book.close()

        return build_normalized_document(
            source,
            blocks,
            detected_title=title,
            detected_created_date=created,
        )

    def _blocks(
        self,
        formula_book: Any,
        value_book: Any,
        get_column_letter: Any,
    ) -> list[DocumentBlock]:
        blocks: list[DocumentBlock] = []
        for formula_sheet in formula_book.worksheets:
            value_sheet = value_book[formula_sheet.title]
            regions = _sheet_regions(formula_sheet, value_sheet)
            if not regions:
                continue
            blocks.append(
                DocumentBlock(
                    "heading",
                    formula_sheet.title,
                    level=1,
                    sheet=formula_sheet.title,
                )
            )
            for region in regions:
                column_start = region.column_start
                column_end = region.column_end
                cell_range = (
                    f"{get_column_letter(column_start)}{region.row_start}:"
                    f"{get_column_letter(column_end)}{region.row_end}"
                )
                blocks.append(
                    DocumentBlock(
                        "table",
                        _format_rows(region.rows),
                        sheet=formula_sheet.title,
                        cell_range=cell_range,
                        rows=region.rows,
                    )
                )
        return blocks


def _sheet_regions(
    formula_sheet: Any,
    value_sheet: Any,
) -> list[_Region]:
    cell_count = formula_sheet.max_row * formula_sheet.max_column
    if cell_count > _MAX_SCANNED_CELLS:
        raise DocumentParseError(
            f"XLSX sheet is too large to scan safely: {formula_sheet.title}"
        )

    populated: list[tuple[int, dict[int, str]]] = []
    formula_rows = formula_sheet.iter_rows()
    value_rows = value_sheet.iter_rows()
    for row_index, (formula_row, value_row) in enumerate(
        zip(formula_rows, value_rows),
        1,
    ):
        values = {
            column_index: text
            for column_index, (formula_cell, value_cell) in enumerate(
                zip(formula_row, value_row),
                1,
            )
            if (text := _cell_text(formula_cell, value_cell))
        }
        if values:
            populated.append((row_index, values))

    grouped: list[tuple[int, int, set[int], list[dict[int, str]]]] = []
    for row_index, values in populated:
        columns = set(values)
        if grouped and row_index == grouped[-1][1] + 1:
            start, _, existing_columns, rows = grouped[-1]
            grouped[-1] = (
                start,
                row_index,
                existing_columns | columns,
                [*rows, values],
            )
        else:
            grouped.append((row_index, row_index, columns, [values]))

    regions: list[_Region] = []
    for row_start, row_end, columns, values_by_row in grouped:
        column_start = min(columns)
        column_end = max(columns)
        rows = tuple(
            tuple(
                values.get(column_index, "")
                for column_index in range(column_start, column_end + 1)
            )
            for values in values_by_row
        )
        regions.append(
            _Region(
                row_start,
                row_end,
                column_start,
                column_end,
                rows,
            )
        )
    return regions


def _cell_text(formula_cell: Any, value_cell: Any) -> str:
    value = formula_cell.value
    if formula_cell.data_type == "f" or (
        isinstance(value, str) and value.startswith("=")
    ):
        formula = value if str(value).startswith("=") else f"={value}"
        cached = _scalar_text(value_cell.value)
        if cached:
            return f"Formula: {formula}; Cached value: {cached}"
        return f"Formula: {formula}"
    return _scalar_text(value)


def _scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    return str(value).strip()


def _format_rows(rows: tuple[tuple[str, ...], ...]) -> str:
    return "\n".join(" | ".join(row) for row in rows)


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _date_string(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return ""
