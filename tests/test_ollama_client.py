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
from app.llm.base import ChatMessage
from app.llm.ollama_client import OllamaClient, OllamaError


def _config(base_url: str = "http://127.0.0.1:11434/api") -> LlmConfig:
    return LlmConfig(
        provider="ollama",
        base_url=base_url,
        model="qwen-local:latest",
        request_timeout_seconds=15,
    )


def _response_body() -> bytes:
    return json.dumps(
        {
            "model": "qwen-local:latest",
            "message": {"role": "assistant", "content": "Ollama 로컬 응답"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 4,
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


class OllamaClientTests(unittest.TestCase):
    def test_sends_native_chat_payload_and_normalizes_response(self) -> None:
        transport = RecordingTransport()
        client = OllamaClient(_config(), transport=transport)

        response = client.chat(
            [
                ChatMessage("system", "코드 근거만 사용하세요."),
                ChatMessage("user", "Z축 Home 실패 원인은?"),
            ],
            temperature=0.2,
            max_tokens=300,
        )

        url, body, headers, timeout = transport.calls[0]
        payload = json.loads(body)
        self.assertEqual(url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(payload["model"], "qwen-local:latest")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["temperature"], 0.2)
        self.assertEqual(payload["options"]["num_predict"], 300)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertNotIn("Authorization", headers)
        self.assertEqual(timeout, 15.0)
        self.assertEqual(response.content, "Ollama 로컬 응답")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage.prompt_tokens, 10)
        self.assertEqual(response.usage.completion_tokens, 4)
        self.assertEqual(response.usage.total_tokens, 14)

    def test_omits_num_predict_without_generation_limit(self) -> None:
        transport = RecordingTransport()

        OllamaClient(_config(), transport=transport).chat(
            [ChatMessage("user", "hello")]
        )

        self.assertNotIn("num_predict", json.loads(transport.calls[0][1])["options"])

    def test_rejects_invalid_provider_messages_and_generation_options(self) -> None:
        with self.assertRaisesRegex(OllamaError, "provider"):
            OllamaClient(
                LlmConfig("llama_cpp", "http://127.0.0.1:8080/v1", "model", 10)
            )

        client = OllamaClient(_config(), transport=RecordingTransport())
        with self.assertRaisesRegex(OllamaError, "messages"):
            client.chat([])
        with self.assertRaisesRegex(OllamaError, "temperature"):
            client.chat([ChatMessage("user", "hello")], temperature=3)
        with self.assertRaisesRegex(OllamaError, "max_tokens"):
            client.chat([ChatMessage("user", "hello")], max_tokens=False)

    def test_reports_invalid_json_api_errors_incomplete_and_usage(self) -> None:
        cases = (
            (b"not-json", "invalid JSON"),
            (b'{"error":"model not found"}', "model not found"),
            (
                b'{"message":{"content":"partial"},"done":false}',
                "incomplete",
            ),
            (
                b'{"message":{"content":"ok"},"prompt_eval_count":"bad"}',
                "invalid token usage",
            ),
        )
        for body, expected in cases:
            with self.subTest(expected=expected):
                client = OllamaClient(
                    _config(),
                    transport=RecordingTransport(body),
                )
                with self.assertRaisesRegex(OllamaError, expected):
                    client.chat([ChatMessage("user", "hello")])

    def test_supports_response_without_optional_usage_or_model(self) -> None:
        body = b'{"message":{"content":"ok"},"done":true}'
        client = OllamaClient(_config(), transport=RecordingTransport(body))

        response = client.chat([ChatMessage("user", "hello")])

        self.assertEqual(response.model, "qwen-local:latest")
        self.assertEqual(response.finish_reason, "")
        self.assertIsNone(response.usage)

    def test_reports_http_error_detail_and_connection_failure(self) -> None:
        http_error = HTTPError(
            "http://127.0.0.1:11434/api/chat",
            404,
            "Not Found",
            {},
            BytesIO(b'{"error":"model not found"}'),
        )
        client = OllamaClient(_config())
        with patch("app.llm.ollama_client.urlopen", side_effect=http_error):
            with self.assertRaisesRegex(OllamaError, "404: model not found"):
                client.chat([ChatMessage("user", "hello")])

        with patch(
            "app.llm.ollama_client.urlopen",
            side_effect=URLError("connection refused"),
        ):
            with self.assertRaisesRegex(OllamaError, "connection refused"):
                client.chat([ChatMessage("user", "hello")])


class OllamaLocalHttpTests(unittest.TestCase):
    def test_posts_to_a_local_ollama_compatible_stub(self) -> None:
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
            base_url = f"http://127.0.0.1:{server.server_port}/api"
            response = OllamaClient(_config(base_url)).chat(
                [ChatMessage("user", "local smoke test")],
                max_tokens=32,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(received["path"], "/api/chat")
        self.assertEqual(received["payload"]["options"]["num_predict"], 32)
        self.assertEqual(response.content, "Ollama 로컬 응답")


if __name__ == "__main__":
    unittest.main()
