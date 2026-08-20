"""Offline llama.cpp client using its OpenAI-compatible chat API."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import ConfigError, LlmConfig, load_config
from app.llm.base import ChatMessage, LlmError, LlmResponse, LlmUsage


class LlamaCppError(LlmError):
    """Raised when the local llama.cpp API cannot complete a request."""


class HttpPostTransport(Protocol):
    def __call__(
        self,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout: float,
    ) -> bytes: ...


class LlamaCppClient:
    """Call a local llama.cpp server through ``/v1/chat/completions``."""

    def __init__(
        self,
        config: LlmConfig,
        *,
        api_key: str | None = None,
        transport: HttpPostTransport | None = None,
    ) -> None:
        if config.provider != "llama_cpp":
            raise LlamaCppError("llama.cpp client requires llm.provider=llama_cpp")
        if api_key is not None and (
            not isinstance(api_key, str) or not api_key.strip()
        ):
            raise LlamaCppError("api_key must be a non-empty string or None")
        self._config = config
        self._api_key = api_key.strip() if api_key is not None else None
        self._transport = transport or _post_json

    @property
    def endpoint(self) -> str:
        return f"{self._config.base_url.rstrip('/')}/chat/completions"

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> LlmResponse:
        validated_messages = _validate_messages(messages)
        _validate_generation_options(temperature, max_tokens)
        payload: dict[str, object] = {
            "model": self._config.model,
            "messages": [message.to_dict() for message in validated_messages],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            response_body = self._transport(
                self.endpoint,
                body,
                headers,
                float(self._config.request_timeout_seconds),
            )
        except LlamaCppError:
            raise
        except Exception as exc:
            raise LlamaCppError("llama.cpp API request failed") from exc
        return _parse_response(response_body, fallback_model=self._config.model)


def _validate_messages(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise LlamaCppError("messages must be a sequence of ChatMessage values")
    validated = list(messages)
    if not validated or any(not isinstance(message, ChatMessage) for message in validated):
        raise LlamaCppError("messages must contain ChatMessage values")
    return validated


def _validate_generation_options(
    temperature: float, max_tokens: int | None
) -> None:
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0 <= temperature <= 2
    ):
        raise LlamaCppError("temperature must be a number between 0 and 2")
    if max_tokens is not None and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise LlamaCppError("max_tokens must be a positive integer or None")


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
        raise LlamaCppError(
            f"llama.cpp API returned HTTP {exc.code}{suffix}"
        ) from exc
    except URLError as exc:
        raise LlamaCppError(
            f"Unable to connect to llama.cpp API: {_safe_detail(exc.reason)}"
        ) from exc
    except TimeoutError as exc:
        raise LlamaCppError("llama.cpp API request timed out") from exc
    except OSError as exc:
        raise LlamaCppError("llama.cpp API request failed") from exc


def _read_http_error(error: HTTPError) -> str:
    try:
        raw = error.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""
    if isinstance(payload, Mapping):
        error_payload = payload.get("error")
        if isinstance(error_payload, Mapping):
            return _safe_detail(error_payload.get("message"))
        return _safe_detail(error_payload)
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
        raise LlamaCppError("llama.cpp API returned invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise LlamaCppError("llama.cpp API returned an invalid response object")

    error_payload = payload.get("error")
    if error_payload is not None:
        if isinstance(error_payload, Mapping):
            detail = _safe_detail(error_payload.get("message"))
        else:
            detail = _safe_detail(error_payload)
        raise LlamaCppError(f"llama.cpp API error: {detail}")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlamaCppError("llama.cpp API response has no choices")
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise LlamaCppError("llama.cpp API returned an invalid choice")
    message = first_choice.get("message")
    if not isinstance(message, Mapping):
        raise LlamaCppError("llama.cpp API choice has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LlamaCppError("llama.cpp API returned empty message content")

    model = payload.get("model", fallback_model)
    finish_reason = first_choice.get("finish_reason", "")
    if not isinstance(model, str) or not isinstance(finish_reason, str):
        raise LlamaCppError("llama.cpp API returned invalid response metadata")
    return LlmResponse(
        content=content,
        model=model,
        finish_reason=finish_reason,
        usage=_parse_usage(payload.get("usage")),
    )


def _parse_usage(value: Any) -> LlmUsage | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise LlamaCppError("llama.cpp API returned invalid token usage")
    try:
        return LlmUsage(
            prompt_tokens=value["prompt_tokens"],
            completion_tokens=value["completion_tokens"],
            total_tokens=value["total_tokens"],
        )
    except (KeyError, LlmError) as exc:
        raise LlamaCppError("llama.cpp API returned invalid token usage") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send one prompt to a local llama.cpp chat completions API"
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
        response = LlamaCppClient(
            config.llm,
            api_key=os.environ.get("LLAMA_CPP_API_KEY"),
        ).chat(
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
