"""Offline-only Sentence Transformers embedding service."""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from app.config import EmbeddingConfig, load_config


class EmbeddingError(RuntimeError):
    """Raised when a local embedding model cannot be loaded or used."""


@dataclass(frozen=True)
class EmbeddingModelInfo:
    model_path: str
    dimension: int
    device: str | None
    normalize_embeddings: bool


class LocalEmbeddingService:
    """Load one local Sentence Transformers model and reuse it for encoding.

    The service rejects non-existent model directories before importing the ML
    runtime and always passes ``local_files_only=True`` to prevent Hub access.
    Model creation is lazy and protected by a lock so repeated or concurrent
    calls reuse a single in-memory model instance.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: Any | None = None
        self._dimension: int | None = None
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def dimension(self) -> int:
        self.load()
        if self._dimension is None:  # pragma: no cover - defensive invariant
            raise EmbeddingError("Embedding dimension is unavailable")
        return self._dimension

    @property
    def info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            model_path=str(self._config.model_path),
            dimension=self.dimension,
            device=self._config.device,
            normalize_embeddings=self._config.normalize_embeddings,
        )

    def load(self) -> None:
        """Load the configured local model exactly once."""

        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            model_path = self._config.model_path.resolve(strict=False)
            if not model_path.is_dir():
                raise EmbeddingError(
                    f"Local embedding model directory not found: {model_path}"
                )

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed; install requirements.txt"
                ) from exc

            load_options: dict[str, Any] = {"local_files_only": True}
            if self._config.device is not None:
                load_options["device"] = self._config.device

            try:
                model = SentenceTransformer(str(model_path), **load_options)
                dimension = self._read_dimension(model)
            except Exception as exc:
                raise EmbeddingError(
                    f"Unable to load local embedding model: {model_path}"
                ) from exc

            self._model = model
            self._dimension = dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed generic text with the model's standard encoding method."""

        return self._embed(texts, preferred_method="encode")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed source chunks using document-aware encoding when available."""

        return self._embed(texts, preferred_method="encode_document")

    def embed_query(self, text: str) -> list[float]:
        """Embed one search query using query-aware encoding when available."""

        rows = self._embed([text], preferred_method="encode_query")
        return rows[0]

    def _embed(self, texts: Sequence[str], preferred_method: str) -> list[list[float]]:
        validated = self._validate_texts(texts)
        self.load()
        if self._model is None:  # pragma: no cover - defensive invariant
            raise EmbeddingError("Embedding model is unavailable")

        encoder: Callable[..., Any] | None = getattr(self._model, preferred_method, None)
        if not callable(encoder):
            encoder = getattr(self._model, "encode", None)
        if not callable(encoder):
            raise EmbeddingError("Embedding model does not provide an encode method")

        try:
            result = encoder(
                validated,
                batch_size=self._config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self._config.normalize_embeddings,
            )
        except Exception as exc:
            raise EmbeddingError("Embedding generation failed") from exc

        return self._to_float_rows(result, expected_rows=len(validated))

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> list[str]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise EmbeddingError("texts must be a sequence of non-empty strings")
        validated = list(texts)
        if not validated or any(
            not isinstance(text, str) or not text.strip() for text in validated
        ):
            raise EmbeddingError("texts must contain non-empty strings")
        return validated

    @staticmethod
    def _read_dimension(model: Any) -> int:
        getter = getattr(model, "get_embedding_dimension", None)
        if not callable(getter):
            getter = getattr(model, "get_sentence_embedding_dimension", None)
        if not callable(getter):
            raise EmbeddingError("Embedding model does not expose its output dimension")

        dimension = getter()
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise EmbeddingError("Embedding model returned an invalid output dimension")
        return dimension

    def _to_float_rows(self, value: Any, expected_rows: int) -> list[list[float]]:
        converted = value.tolist() if hasattr(value, "tolist") else value
        if not isinstance(converted, list) or len(converted) != expected_rows:
            raise EmbeddingError("Embedding model returned an unexpected batch shape")

        rows: list[list[float]] = []
        for row in converted:
            if not isinstance(row, (list, tuple)) or len(row) != self.dimension:
                raise EmbeddingError("Embedding model returned an unexpected vector shape")
            try:
                rows.append([float(item) for item in row])
            except (TypeError, ValueError) as exc:
                raise EmbeddingError("Embedding vector contains a non-numeric value") from exc
        return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a local embedding model")
    parser.add_argument("--config", default="config/config.yaml", help="YAML configuration path")
    parser.add_argument("--text", default="로컬 임베딩 모델 확인", help="Sample text")
    parser.add_argument(
        "--task",
        choices=("query", "document"),
        default="query",
        help="Encoding task used for the sample",
    )
    args = parser.parse_args()

    service = LocalEmbeddingService(load_config(args.config).embedding)
    try:
        if args.task == "query":
            vector = service.embed_query(args.text)
        else:
            vector = service.embed_documents([args.text])[0]
    except EmbeddingError as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "model": asdict(service.info),
                "task": args.task,
                "vector_length": len(vector),
                "vector_preview": vector[:8],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
