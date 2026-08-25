"""Offline embedding service with Sentence Transformers and llama.cpp backends."""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import EmbeddingConfig, load_config


class EmbeddingError(RuntimeError):
    """Raised when a local embedding model cannot be loaded or used."""


class HttpPostTransport(Protocol):
    def __call__(
        self,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout: float,
    ) -> bytes: ...


@dataclass(frozen=True)
class EmbeddingModelInfo:
    backend: str
    model_path: str
    dimension: int
    device: str | None
    normalize_embeddings: bool
    endpoint: str | None = None


class LocalEmbeddingService:
    """Reuse either a local Sentence Transformers model or llama.cpp server.

    ``sentence_transformers`` remains the default and loads an offline model
    directory in-process. ``llama_cpp`` validates a local GGUF file and sends
    OpenAI-compatible embedding requests to a separately started llama-server.
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        transport: HttpPostTransport | None = None,
    ) -> None:
        self._config = config
        self._model: Any | None = None
        self._dimension: int | None = None
        self._load_lock = threading.Lock()
        self._transport = transport or _post_json

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def endpoint(self) -> str | None:
        if self._config.backend != "llama_cpp":
            return None
        return f"{self._config.base_url.rstrip('/')}/embeddings"

    @property
    def dimension(self) -> int:
        self.load()
        if self._dimension is None:  # pragma: no cover
            raise EmbeddingError("Embedding dimension is unavailable")
        return self._dimension

    @property
    def info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            backend=self._config.backend,
            model_path=str(self._config.model_path),
            dimension=self.dimension,
            device=self._config.device,
            normalize_embeddings=self._config.normalize_embeddings,
            endpoint=self.endpoint,
        )

    def load(self) -> None:
        """Validate and initialize the configured backend exactly once."""

        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            model_path = self._config.model_path.resolve(strict=False)
            if self._config.backend == "llama_cpp":
                if not model_path.is_file():
                    raise EmbeddingError(
                        f"Local GGUF embedding model file not found: {model_path}"
                    )
                self._model = True
                self._dimension = self._config.dimension
                return
            if self._config.backend != "sentence_transformers":
                raise EmbeddingError(
                    f"Unsupported embedding backend: {self._config.backend}"
                )
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
        return self._embed(texts, preferred_method="encode")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, preferred_method="encode_document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], preferred_method="encode_query")[0]

    def _embed(self, texts: Sequence[str], preferred_method: str) -> list[list[float]]:
        validated = self._validate_texts(texts)
        self.load()
        if self._config.backend == "llama_cpp":
            return self._embed_with_llama_cpp(validated)
        if self._model is None:  # pragma: no cover
            raise EmbeddingError("Embedding model is unavailable")
        encoder: Callable[..., Any] | None = getattr(
            self._model, preferred_method, None
        )
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

    def _embed_with_llama_cpp(self, texts: list[str]) -> list[list[float]]:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self._config.batch_size):
            batch = texts[start : start + self._config.batch_size]
            payload = {"model": self._config.model, "input": batch}
            try:
                response = self._transport(
                    self.endpoint or "",
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    {"Content-Type": "application/json", "Accept": "application/json"},
                    float(self._config.request_timeout_seconds),
                )
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError("llama.cpp embedding API request failed") from exc
            rows.extend(self._parse_llama_cpp_response(response, len(batch)))
        return rows

    def _parse_llama_cpp_response(
        self, body: bytes, expected_rows: int
    ) -> list[list[float]]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise EmbeddingError("llama.cpp embedding API returned invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise EmbeddingError("llama.cpp embedding API returned an invalid object")
        if payload.get("error") is not None:
            error = payload["error"]
            detail = error.get("message") if isinstance(error, Mapping) else error
            raise EmbeddingError(f"llama.cpp embedding API error: {_safe_detail(detail)}")
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != expected_rows:
            raise EmbeddingError("llama.cpp embedding API returned an unexpected batch")
        indexed: dict[int, Any] = {}
        for position, item in enumerate(data):
            if not isinstance(item, Mapping):
                raise EmbeddingError("llama.cpp embedding API returned an invalid item")
            index = item.get("index", position)
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= expected_rows
                or index in indexed
            ):
                raise EmbeddingError("llama.cpp embedding API returned an invalid index")
            indexed[index] = item.get("embedding")
        ordered = [indexed[index] for index in range(expected_rows)]
        return self._to_float_rows(ordered, expected_rows=expected_rows)

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


def _post_json(
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    timeout: float,
) -> bytes:
    request = Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except OSError:
            detail = ""
        suffix = f": {_safe_detail(detail)}" if detail else ""
        raise EmbeddingError(
            f"llama.cpp embedding API returned HTTP {exc.code}{suffix}"
        ) from exc
    except URLError as exc:
        raise EmbeddingError(
            f"Unable to connect to llama.cpp embedding API: {_safe_detail(exc.reason)}"
        ) from exc
    except TimeoutError as exc:
        raise EmbeddingError("llama.cpp embedding API request timed out") from exc
    except OSError as exc:
        raise EmbeddingError("llama.cpp embedding API request failed") from exc


def _safe_detail(value: object, limit: int = 300) -> str:
    if value is None:
        return "unknown error"
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text[:limit] if text else "unknown error"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a local embedding model")
    parser.add_argument("--config", default="config/config.yaml", help="YAML configuration path")
    parser.add_argument("--text", default="로컬 임베딩 모델 확인", help="Sample text")
    parser.add_argument(
        "--task", choices=("query", "document"), default="query"
    )
    args = parser.parse_args()
    service = LocalEmbeddingService(load_config(args.config).embedding)
    try:
        vector = (
            service.embed_query(args.text)
            if args.task == "query"
            else service.embed_documents([args.text])[0]
        )
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
