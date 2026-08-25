"""Optional offline Cross-Encoder reranking."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Sequence


class RerankerError(RuntimeError):
    """Raised when a local reranker cannot be loaded or evaluated."""


class LocalCrossEncoderReranker:
    """Lazy local-only Sentence Transformers Cross-Encoder wrapper."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise RerankerError("reranker batch_size must be a positive integer")
        self._model_path = model_path
        self._device = device
        self._batch_size = batch_size
        self._model: Any | None = None
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise RerankerError("reranker query must be a non-empty string")
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise RerankerError("reranker documents must be a sequence")
        values = list(documents)
        if not values:
            return []
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise RerankerError("reranker documents must contain non-empty strings")
        model = self._load()
        try:
            scores = model.predict(
                [(query.strip(), document) for document in values],
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise RerankerError("Local reranker prediction failed") from exc
        converted = scores.tolist() if hasattr(scores, "tolist") else scores
        if not isinstance(converted, list) or len(converted) != len(values):
            raise RerankerError("Local reranker returned an unexpected score count")
        normalized: list[float] = []
        for value in converted:
            if isinstance(value, list):
                if len(value) != 1:
                    raise RerankerError("Local reranker must return one score per pair")
                value = value[0]
            try:
                normalized.append(float(value))
            except (TypeError, ValueError) as exc:
                raise RerankerError("Local reranker returned a non-numeric score") from exc
        return normalized

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            model_path = self._model_path.resolve(strict=False)
            if not model_path.is_dir():
                raise RerankerError(
                    f"Local reranker model directory not found: {model_path}"
                )
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RerankerError(
                    "sentence-transformers is required for local reranking"
                ) from exc
            options: dict[str, Any] = {
                "local_files_only": True,
                "trust_remote_code": False,
            }
            if self._device is not None:
                options["device"] = self._device
            try:
                self._model = CrossEncoder(str(model_path), **options)
            except Exception as exc:
                raise RerankerError(
                    f"Unable to load local reranker model: {model_path}"
                ) from exc
            return self._model
