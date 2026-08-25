"""Offline retrieval quality evaluation for traceable EquipmentRAG sources."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from app.api import ApiRequestError, build_json_filters
from app.config import AppConfig
from app.rag_service import RagError, RagService, RagSource


_MAX_DATASET_BYTES = 10 * 1024 * 1024
_MAX_CASES = 10_000
_MATCH_FIELDS = {
    "record_id",
    "source_type",
    "file_name",
    "relative_path",
    "class_name",
    "method_name",
    "document_type",
    "revision",
    "document_status",
    "section",
    "subsection",
    "page",
    "slide",
    "sheet",
    "cell_range",
}
_PATH_FIELDS = {"relative_path"}


class EvaluationError(RuntimeError):
    """Raised when an evaluation dataset or run is invalid."""


class RetrievalProvider(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: object = None,
    ) -> tuple[RagSource, ...]: ...


ServiceFactory = Callable[[AppConfig, str], RetrievalProvider]


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    query: str
    source_type: str
    top_k: int | None
    filters: Mapping[str, Any] | None
    expected: tuple[Mapping[str, str | int], ...]


@dataclass(frozen=True)
class EvaluationCaseResult:
    case_id: str
    query: str
    source_type: str
    top_k: int
    expected_count: int
    matched_count: int
    hit: bool
    recall: float
    reciprocal_rank: float
    first_relevant_rank: int | None
    matched_expected: tuple[int, ...]
    retrieved_sources: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationReport:
    dataset_path: str
    case_count: int
    expected_count: int
    matched_count: int
    hit_rate: float
    recall_at_k: float
    mrr: float
    cases: tuple[EvaluationCaseResult, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_evaluation_dataset(path: Path) -> tuple[EvaluationCase, ...]:
    source = path.expanduser().resolve(strict=False)
    try:
        stat = source.stat()
    except OSError as exc:
        raise EvaluationError(f"Evaluation dataset not found: {source}") from exc
    if not source.is_file():
        raise EvaluationError(f"Evaluation dataset is not a file: {source}")
    if stat.st_size > _MAX_DATASET_BYTES:
        raise EvaluationError("Evaluation dataset exceeds the 10 MiB safety limit")
    try:
        lines = source.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvaluationError(f"Unable to read evaluation dataset: {source}") from exc

    cases: list[EvaluationCase] = []
    identifiers: set[str] = set()
    for line_number, raw_line in enumerate(lines, 1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(
                f"Invalid JSON on evaluation line {line_number}"
            ) from exc
        case = _parse_case(value, line_number)
        if case.case_id in identifiers:
            raise EvaluationError(f"Duplicate evaluation id: {case.case_id}")
        identifiers.add(case.case_id)
        cases.append(case)
        if len(cases) > _MAX_CASES:
            raise EvaluationError("Evaluation dataset exceeds 10000 cases")
    if not cases:
        raise EvaluationError("Evaluation dataset contains no cases")
    return tuple(cases)


def evaluate_retrieval(
    config: AppConfig,
    dataset_path: Path,
    *,
    default_top_k: int | None = None,
    service_factory: ServiceFactory | None = None,
) -> EvaluationReport:
    if default_top_k is not None and (
        isinstance(default_top_k, bool)
        or not isinstance(default_top_k, int)
        or default_top_k <= 0
        or default_top_k > 100
    ):
        raise EvaluationError("default_top_k must be between 1 and 100 or None")
    cases = load_evaluation_dataset(dataset_path)
    factory = service_factory or (
        lambda value, source_type: RagService(value, source_type=source_type)
    )
    services: dict[str, RetrievalProvider] = {}
    results: list[EvaluationCaseResult] = []
    for case in cases:
        service = services.get(case.source_type)
        if service is None:
            try:
                service = factory(config, case.source_type)
            except Exception as exc:
                raise EvaluationError(
                    f"Unable to initialize retrieval for case '{case.case_id}'"
                ) from exc
            services[case.source_type] = service
        limit = case.top_k or default_top_k or config.search.top_k
        try:
            filters = build_json_filters(case.filters, case.source_type, config)
            sources = service.retrieve(case.query, top_k=limit, filters=filters)
        except (ApiRequestError, RagError) as exc:
            raise EvaluationError(
                f"Evaluation case '{case.case_id}' failed: {exc}"
            ) from exc
        except Exception as exc:
            raise EvaluationError(
                f"Evaluation case '{case.case_id}' retrieval failed"
            ) from exc
        results.append(_score_case(case, sources, limit))

    expected_count = sum(result.expected_count for result in results)
    matched_count = sum(result.matched_count for result in results)
    case_count = len(results)
    return EvaluationReport(
        dataset_path=str(dataset_path.expanduser().resolve(strict=False)),
        case_count=case_count,
        expected_count=expected_count,
        matched_count=matched_count,
        hit_rate=sum(1 for result in results if result.hit) / case_count,
        recall_at_k=matched_count / expected_count,
        mrr=sum(result.reciprocal_rank for result in results) / case_count,
        cases=tuple(results),
    )


def format_evaluation_report(report: EvaluationReport) -> str:
    lines = [
        f"Dataset: {report.dataset_path}",
        f"Cases: {report.case_count}",
        f"Hit@K: {report.hit_rate:.4f}",
        f"Recall@K: {report.recall_at_k:.4f}",
        f"MRR: {report.mrr:.4f}",
        f"Expected sources matched: {report.matched_count}/{report.expected_count}",
        "",
        "Per-case results:",
    ]
    for result in report.cases:
        rank = result.first_relevant_rank if result.first_relevant_rank is not None else "-"
        lines.append(
            f"  {result.case_id}: hit={'yes' if result.hit else 'no'}, "
            f"recall={result.recall:.4f}, first_rank={rank}, top_k={result.top_k}"
        )
    return "\n".join(lines)


def _parse_case(value: object, line_number: int) -> EvaluationCase:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"Evaluation line {line_number} must be a JSON object")
    allowed = {"id", "query", "source_type", "top_k", "filters", "expected"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise EvaluationError(
            f"Unknown field(s) on evaluation line {line_number}: {', '.join(unknown)}"
        )
    case_id = _required_string(value.get("id"), "id", line_number)
    query = _required_string(value.get("query"), "query", line_number)
    source_type = value.get("source_type", "all")
    if not isinstance(source_type, str) or source_type not in {
        "code",
        "document",
        "all",
    }:
        raise EvaluationError(
            f"source_type on evaluation line {line_number} is invalid"
        )
    top_k = value.get("top_k")
    if top_k is not None and (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k <= 0
        or top_k > 100
    ):
        raise EvaluationError(
            f"top_k on evaluation line {line_number} must be between 1 and 100"
        )
    filters = value.get("filters")
    if filters is not None and not isinstance(filters, Mapping):
        raise EvaluationError(
            f"filters on evaluation line {line_number} must be an object"
        )
    raw_expected = value.get("expected")
    if not isinstance(raw_expected, list) or not raw_expected:
        raise EvaluationError(
            f"expected on evaluation line {line_number} must be a non-empty array"
        )
    expected = tuple(
        _parse_expected(item, line_number, index)
        for index, item in enumerate(raw_expected, 1)
    )
    return EvaluationCase(case_id, query, source_type, top_k, filters, expected)


def _parse_expected(
    value: object,
    line_number: int,
    index: int,
) -> Mapping[str, str | int]:
    if not isinstance(value, Mapping) or not value:
        raise EvaluationError(
            f"expected[{index}] on evaluation line {line_number} must be an object"
        )
    unknown = sorted(set(value) - _MATCH_FIELDS)
    if unknown:
        raise EvaluationError(
            f"Unknown expected field(s) on line {line_number}: {', '.join(unknown)}"
        )
    result: dict[str, str | int] = {}
    for field_name, field_value in value.items():
        if field_name in {"page", "slide"}:
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, int)
                or field_value < 0
            ):
                raise EvaluationError(
                    f"expected {field_name} on line {line_number} must be non-negative"
                )
        elif not isinstance(field_value, str) or not field_value.strip():
            raise EvaluationError(
                f"expected {field_name} on line {line_number} must be non-empty"
            )
        result[field_name] = field_value
    return result


def _required_string(value: object, name: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(
            f"{name} on evaluation line {line_number} must be a non-empty string"
        )
    return value.strip()


def _score_case(
    case: EvaluationCase,
    sources: Sequence[RagSource],
    top_k: int,
) -> EvaluationCaseResult:
    limited = tuple(sources[:top_k])
    matched = tuple(
        index
        for index, expected in enumerate(case.expected, 1)
        if any(_matches(source, expected) for source in limited)
    )
    first_rank = next(
        (
            rank
            for rank, source in enumerate(limited, 1)
            if any(_matches(source, expected) for expected in case.expected)
        ),
        None,
    )
    return EvaluationCaseResult(
        case_id=case.case_id,
        query=case.query,
        source_type=case.source_type,
        top_k=top_k,
        expected_count=len(case.expected),
        matched_count=len(matched),
        hit=bool(matched),
        recall=len(matched) / len(case.expected),
        reciprocal_rank=0.0 if first_rank is None else 1 / first_rank,
        first_relevant_rank=first_rank,
        matched_expected=matched,
        retrieved_sources=tuple(
            source.to_dict(include_code=False) for source in limited
        ),
    )


def _matches(source: RagSource, expected: Mapping[str, str | int]) -> bool:
    for field_name, expected_value in expected.items():
        actual = getattr(source, field_name)
        if isinstance(expected_value, str):
            actual_value = str(actual).strip()
            expected_text = expected_value.strip()
            if field_name in _PATH_FIELDS:
                actual_value = actual_value.replace("\\", "/")
                expected_text = expected_text.replace("\\", "/")
            if actual_value.casefold() != expected_text.casefold():
                return False
        elif actual != expected_value:
            return False
    return True
