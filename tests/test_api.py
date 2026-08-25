from __future__ import annotations

import http.client
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from app.api import (
    ApiApplication,
    ApiRequestError,
    EquipmentRagHttpServer,
    load_static_asset,
    validate_bind_address,
)
from app.config import (
    AppConfig,
    ChromaConfig,
    DocumentConfig,
    EmbeddingConfig,
    EquipmentConfig,
    LlmConfig,
    LoggingConfig,
    SearchConfig,
    SourceConfig,
)
from app.rag_service import RagAnswer, RagSource, UnifiedSearchFilters


def _config(root: Path) -> AppConfig:
    return AppConfig(
        project_root=root,
        equipment=EquipmentConfig(name="press-line-01"),
        source=SourceConfig(
            path=root / "source",
            include_extensions=(".cs",),
            exclude_directories=("bin", "obj"),
            chunk_size=1000,
            chunk_overlap=100,
        ),
        embedding=EmbeddingConfig(
            model_path=root / "model",
            batch_size=8,
            device="cpu",
            normalize_embeddings=True,
        ),
        chromadb=ChromaConfig(path=root / "chroma", collection_name="code"),
        search=SearchConfig(top_k=5),
        llm=LlmConfig(
            provider="ollama",
            base_url="http://127.0.0.1:11434/api",
            model="local-model",
            request_timeout_seconds=30,
        ),
        logging=LoggingConfig(level="INFO", path=root / "rag.log"),
        document=DocumentConfig(
            enabled=True,
            source_paths=(root / "documents",),
            extensions=(".pdf",),
            exclude_directories=("archive",),
            chunk_size=1000,
            chunk_overlap=100,
            collection_name="documents",
        ),
    )


def _source() -> RagSource:
    return RagSource(
        source_id="C1",
        record_id="record-1",
        score=0.9,
        file_name="Loader.cs",
        relative_path="Loader.cs",
        file_path="D:/source/Loader.cs",
        class_name="Loader",
        method_name="CheckVacuum",
        start_line=10,
        end_line=20,
        code="if (!vacuum) Alarm();",
    )


class _FakeService:
    def __init__(self) -> None:
        self.retrieve_calls: list[tuple[object, ...]] = []
        self.ask_calls: list[tuple[object, ...]] = []

    def retrieve(self, query: str, **kwargs: object) -> tuple[RagSource, ...]:
        self.retrieve_calls.append((query, kwargs))
        return (_source(),)

    def ask(self, question: str, **kwargs: object) -> RagAnswer:
        self.ask_calls.append((question, kwargs))
        return RagAnswer(
            question=question,
            answer="진공 신호를 확인하세요. [C1]",
            sources=(_source(),),
            model="local-model",
            finish_reason="stop",
        )


class ApiApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="equipment-rag-api-"))
        self.services: dict[str, _FakeService] = {}

        def factory(_config: AppConfig, source_type: str) -> _FakeService:
            service = _FakeService()
            self.services[source_type] = service
            return service

        self.application = ApiApplication(_config(self.root), service_factory=factory)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=False)

    def test_health_does_not_initialize_models(self) -> None:
        status, payload = self.application.dispatch("GET", "/health")

        self.assertEqual(status, 200)
        self.assertEqual(payload["equipment"], "press-line-01")
        self.assertEqual(self.services, {})

    def test_local_ui_assets_are_self_contained(self) -> None:
        html_asset = load_static_asset("/")
        css_asset = load_static_asset("/assets/styles.css")
        script_asset = load_static_asset("/assets/app.js")

        self.assertIsNotNone(html_asset)
        self.assertIsNotNone(css_asset)
        self.assertIsNotNone(script_asset)
        assert html_asset is not None
        html = html_asset[1].decode("utf-8")
        script = script_asset[1].decode("utf-8")
        self.assertIn("EquipmentRAG", html)
        self.assertIn('id="query-form"', html)
        self.assertNotIn("https://", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("innerHTML", script)
        self.assertNotIn("https://", script)
        self.assertIsNone(load_static_asset("/assets/unknown.js"))

    def test_retrieve_returns_traceable_sources_and_default_filters(self) -> None:
        status, payload = self.application.dispatch(
            "POST",
            "/v1/retrieve",
            {
                "query": "Vacuum alarm",
                "source_type": "all",
                "top_k": 4,
                "include_content": True,
                "filters": {"document": {"unit": "Loader"}},
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["result_count"], 1)
        self.assertIn("code", payload["sources"][0])
        _, kwargs = self.services["all"].retrieve_calls[0]
        filters = kwargs["filters"]
        self.assertIsInstance(filters, UnifiedSearchFilters)
        self.assertEqual(filters.document.unit, "Loader")
        self.assertEqual(filters.document.document_status, "active")
        self.assertTrue(filters.document.is_latest)

    def test_answer_passes_generation_options_and_omits_content_by_default(self) -> None:
        status, payload = self.application.dispatch(
            "POST",
            "/v1/answer",
            {
                "question": "복구 방법은?",
                "source_type": "code",
                "temperature": 0.2,
                "max_tokens": 256,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["model"], "local-model")
        self.assertNotIn("code", payload["sources"][0])
        _, kwargs = self.services["code"].ask_calls[0]
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertEqual(kwargs["max_tokens"], 256)

    def test_invalid_requests_do_not_initialize_models(self) -> None:
        with self.assertRaisesRegex(ApiRequestError, "unknown request"):
            self.application.dispatch(
                "POST",
                "/v1/retrieve",
                {"query": "test", "unexpected": True},
            )
        with self.assertRaisesRegex(ApiRequestError, "endpoint not found"):
            self.application.dispatch("GET", "/missing")

        self.assertEqual(self.services, {})

    def test_rejects_generation_values_outside_safe_range(self) -> None:
        with self.assertRaisesRegex(ApiRequestError, "temperature"):
            self.application.dispatch(
                "POST",
                "/v1/answer",
                {"question": "test", "temperature": 3},
            )

        self.assertEqual(self.services, {})

    def test_rejects_excessive_result_limit_before_model_initialization(self) -> None:
        with self.assertRaisesRegex(ApiRequestError, "must not exceed"):
            self.application.dispatch(
                "POST",
                "/v1/retrieve",
                {"query": "test", "top_k": 101},
            )

        self.assertEqual(self.services, {})

    def test_non_loopback_bind_requires_explicit_permission(self) -> None:
        validate_bind_address("127.0.0.1", allow_remote=False)
        validate_bind_address("localhost", allow_remote=False)
        with self.assertRaisesRegex(ApiRequestError, "--allow-remote"):
            validate_bind_address("0.0.0.0", allow_remote=False)

    def test_http_server_returns_utf8_json_and_security_headers(self) -> None:
        server = EquipmentRagHttpServer(("127.0.0.1", 0), self.application)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection(
                "127.0.0.1",
                server.server_address[1],
                timeout=3,
            )
            body = json.dumps({"query": "진공", "source_type": "code"})
            connection.request(
                "POST",
                "/v1/retrieve",
                body=body.encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()

            self.assertEqual(response.status, 200)
            self.assertEqual(payload["query"], "진공")
            self.assertEqual(response.getheader("Cache-Control"), "no-store")
            self.assertEqual(
                response.getheader("X-Content-Type-Options"),
                "nosniff",
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_http_server_serves_ui_with_strict_browser_policy(self) -> None:
        server = EquipmentRagHttpServer(("127.0.0.1", 0), self.application)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection(
                "127.0.0.1",
                server.server_address[1],
                timeout=3,
            )
            connection.request("GET", "/")
            response = connection.getresponse()
            body = response.read().decode("utf-8")
            connection.close()

            self.assertEqual(response.status, 200)
            self.assertTrue(response.getheader("Content-Type").startswith("text/html"))
            self.assertIn("default-src 'self'", response.getheader("Content-Security-Policy"))
            self.assertEqual(response.getheader("X-Frame-Options"), "DENY")
            self.assertIn("Local Knowledge Console", body)
            self.assertEqual(self.services, {})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
