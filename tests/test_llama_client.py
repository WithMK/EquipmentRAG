from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from app.config import LlmConfig
from app.llm.base import ChatMessage, LlmError
from app.llm.llama_client import LlamaCppClient, LlamaCppError


def _config(base_url: str = "http://127.0.0.1:8080/v1") -> LlmConfig:
    return LlmConfig(
        provider="llama_cpp",
        base_url=base_url,
        model="local-model",
        request_timeout_seconds=12,
    )


def _response_body() -> bytes:
    return json.dumps(
        {
            "model": "local-model-q4",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "로컬 응답"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 3,
                "total_tokens": 11,
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")


class RecordingTransport:
    def __init__(self, response: bytes | None = None) -> None:
        self.response = response or _response_body()
        self.calls: list[tuple[str, bytes, dict[str, str], float]] = []

    def __call__(
        self,
        url: str,
        body: bytes,
        headers: Any,
        timeout: float,
    ) -> bytes:
        self.calls.append((url, body, dict(headers), timeout))
        return self.response


class LlamaCppClientTests(unittest.TestCase):
    def test_sends_openai_compatible_payload_and_normalizes_response(self) -> None:
        transport = RecordingTransport()
        client = LlamaCppClient(_config(), transport=transport)

        response = client.chat(
            [
                ChatMessage("system", "코드 근거만 사용하세요."),
                ChatMessage("user", "Z축 원점 복귀 실패 원인은?"),
            ]
        )

        url, body, headers, timeout = transport.calls[0]
        payload = json.loads(body)
        self.assertEqual(url, "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(payload["model"], "local-model")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["temperature"], 0.1)
        self.assertFalse(payload["stream"])
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertNotIn("Authorization", headers)
        self.assertEqual(timeout, 12.0)
        self.assertEqual(response.content, "로컬 응답")
        self.assertEqual(response.model, "local-model-q4")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage.total_tokens, 11)

    def test_supports_optional_local_api_key_and_generation_limit(self) -> None:
        transport = RecordingTransport()
        client = LlamaCppClient(
            _config(),
            api_key="local-only",
            transport=transport,
        )

        client.chat(
            [ChatMessage("user", "상태를 설명하세요.")],
            temperature=0,
            max_tokens=256,
        )

        _, body, headers, _ = transport.calls[0]
        self.assertEqual(headers["Authorization"], "Bearer local-only")
        self.assertEqual(json.loads(body)["max_tokens"], 256)

    def test_rejects_invalid_provider_messages_and_generation_options(self) -> None:
        with self.assertRaisesRegex(LlamaCppError, "provider"):
            LlamaCppClient(
                LlmConfig("ollama", "http://127.0.0.1:11434", "model", 10)
            )
        with self.assertRaisesRegex(LlmError, "role"):
            ChatMessage("tool", "invalid")

        client = LlamaCppClient(_config(), transport=RecordingTransport())
        with self.assertRaisesRegex(LlamaCppError, "messages"):
            client.chat([])
        with self.assertRaisesRegex(LlamaCppError, "temperature"):
            client.chat([ChatMessage("user", "hello")], temperature=2.1)
        with self.assertRaisesRegex(LlamaCppError, "max_tokens"):
            client.chat([ChatMessage("user", "hello")], max_tokens=0)

    def test_reports_invalid_json_api_errors_and_invalid_usage(self) -> None:
        cases = (
            (b"not-json", "invalid JSON"),
            (b'{"error":{"message":"model is not loaded"}}', "not loaded"),
            (b'{"choices":[]}', "no choices"),
            (
                b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}',
                "invalid token usage",
            ),
        )
        for body, expected in cases:
            with self.subTest(expected=expected):
                client = LlamaCppClient(
                    _config(),
                    transport=RecordingTransport(body),
                )
                with self.assertRaisesRegex(LlamaCppError, expected):
                    client.chat([ChatMessage("user", "hello")])

    def test_uses_configured_model_when_response_omits_model_and_usage(self) -> None:
        body = b'{"choices":[{"message":{"content":"ok"}}]}'
        client = LlamaCppClient(_config(), transport=RecordingTransport(body))

        response = client.chat([ChatMessage("user", "hello")])

        self.assertEqual(response.model, "local-model")
        self.assertEqual(response.finish_reason, "")
        self.assertIsNone(response.usage)

    def test_reports_http_error_detail_and_connection_failure(self) -> None:
        http_error = HTTPError(
            "http://127.0.0.1:8080/v1/chat/completions",
            503,
            "Unavailable",
            {},
            BytesIO(b'{"error":{"message":"model is loading"}}'),
        )
        client = LlamaCppClient(_config())
        with patch("app.llm.llama_client.urlopen", side_effect=http_error):
            with self.assertRaisesRegex(LlamaCppError, "503: model is loading"):
                client.chat([ChatMessage("user", "hello")])

        with patch(
            "app.llm.llama_client.urlopen",
            side_effect=URLError("connection refused"),
        ):
            with self.assertRaisesRegex(LlamaCppError, "connection refused"):
                client.chat([ChatMessage("user", "hello")])


class LlamaCppLocalHttpTests(unittest.TestCase):
    def test_posts_to_a_local_openai_compatible_stub(self) -> None:
        received: dict[str, object] = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
                length = int(self.headers["Content-Length"])
                received["path"] = self.path
                received["payload"] = json.loads(self.rfile.read(length))
                response = _response_body()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}/v1"
            response = LlamaCppClient(_config(base_url)).chat(
                [ChatMessage("user", "local smoke test")],
                max_tokens=32,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(received["path"], "/v1/chat/completions")
        self.assertEqual(received["payload"]["max_tokens"], 32)
        self.assertEqual(response.content, "로컬 응답")


if __name__ == "__main__":
    unittest.main()
