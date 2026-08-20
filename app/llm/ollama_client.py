"""Offline client for Ollama's native local chat API."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import ConfigError, LlmConfig, load_config
from app.llm.base import ChatMessage, LlmError, LlmResponse, LlmUsage


class OllamaError(LlmError):
    """Raised when the local Ollama API cannot complete a request."""


class HttpPostTransport(Protocol):
    def __call__(
        self,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout: float,
    ) -> bytes: ...


class OllamaClient:
    """Call a local Ollama server through ``/api/chat`` without streaming."""

    def __init__(
        self,
        config: LlmConfig,
        *,
        transport: HttpPostTransport | None = None,
    ) -> None:
        if config.provider != "ollama":
            raise OllamaError("Ollama client requires llm.provider=ollama")
        self._config = config
        self._transport = transport or _post_json

    @property
    def endpoint(self) -> str:
        return f"{self._config.base_url.rstrip('/')}/chat"

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> LlmResponse:
        validated_messages = _validate_messages(messages)
        _validate_generation_options(temperature, max_tokens)
        options: dict[str, object] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload: dict[str, object] = {
            "model": self._config.model,
            "messages": [message.to_dict() for message in validated_messages],
            "stream": False,
            "think": False,
            "options": options,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        try:
            response_body = self._transport(
                self.endpoint,
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers,
                float(self._config.request_timeout_seconds),
            )
        except OllamaError:
            raise
        except Exception as exc:
            raise OllamaError("Ollama API request failed") from exc
        return _parse_response(response_body, fallback_model=self._config.model)


def _validate_messages(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise OllamaError("messages must be a sequence of ChatMessage values")
    validated = list(messages)
    if not validated or any(not isinstance(message, ChatMessage) for message in validated):
        raise OllamaError("messages must contain ChatMessage values")
    return validated


def _validate_generation_options(
    temperature: float, max_tokens: int | None
) -> None:
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0 <= temperature <= 2
    ):
        raise OllamaError("temperature must be a number between 0 and 2")
    if max_tokens is not None and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise OllamaError("max_tokens must be a positive integer or None")


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
        detail = _read_http_error(exc)
        suffix = f": {detail}" if detail else ""
        raise OllamaError(f"Ollama API returned HTTP {exc.code}{suffix}") from exc
    except URLError as exc:
        raise OllamaError(
            f"Unable to connect to Ollama API: {_safe_detail(exc.reason)}"
        ) from exc
    except TimeoutError as exc:
        raise OllamaError("Ollama API request timed out") from exc
    except OSError as exc:
        raise OllamaError("Ollama API request failed") from exc


def _read_http_error(error: HTTPError) -> str:
    try:
        raw = error.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""
    if isinstance(payload, Mapping):
        return _safe_detail(payload.get("error"))
    return ""


def _safe_detail(value: object, limit: int = 300) -> str:
    if value is None:
        return "unknown error"
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text[:limit] if text else "unknown error"


def _parse_response(body: bytes, *, fallback_model: str) -> LlmResponse:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OllamaError("Ollama API returned invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise OllamaError("Ollama API returned an invalid response object")
    if payload.get("error") is not None:
        raise OllamaError(f"Ollama API error: {_safe_detail(payload.get('error'))}")
    if payload.get("done") is False:
        raise OllamaError("Ollama API returned an incomplete non-streaming response")

    message = payload.get("message")
    if not isinstance(message, Mapping):
        raise OllamaError("Ollama API response has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise OllamaError("Ollama API returned empty message content")
    model = payload.get("model", fallback_model)
    finish_reason = payload.get("done_reason", "")
    if not isinstance(model, str) or not isinstance(finish_reason, str):
        raise OllamaError("Ollama API returned invalid response metadata")
    return LlmResponse(
        content=content,
        model=model,
        finish_reason=finish_reason,
        usage=_parse_usage(payload),
    )


def _parse_usage(payload: Mapping[str, Any]) -> LlmUsage | None:
    prompt_tokens = payload.get("prompt_eval_count")
    completion_tokens = payload.get("eval_count")
    if prompt_tokens is None and completion_tokens is None:
        return None
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens < 0
        or isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens < 0
    ):
        raise OllamaError("Ollama API returned invalid token usage")
    return LlmUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send one prompt to a local Ollama chat API"
    )
    parser.add_argument("prompt", help="User prompt sent to the local model")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--system", help="Optional system message")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--json", action="store_true", help="Emit response metadata")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        messages = []
        if args.system:
            messages.append(ChatMessage("system", args.system))
        messages.append(ChatMessage("user", args.prompt))
        response = OllamaClient(config.llm).chat(
            messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except (ConfigError, LlmError) as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(response.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
