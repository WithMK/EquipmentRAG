"""Loopback-first JSON API for local EquipmentRAG integrations."""

from __future__ import annotations

import argparse
import ipaddress
import json
import threading
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from app import __version__
from app.config import AppConfig, ConfigError, load_config
from app.rag_service import RagAnswer, RagError, RagService, RagSource, UnifiedSearchFilters
from app.retrieval.document_retriever import DocumentSearchFilters
from app.search import CodeSearchFilters


_MAX_REQUEST_BYTES = 1_048_576
_SOURCE_TYPES = {"code", "document", "all"}
_WEB_ROOT = Path(__file__).with_name("web")
_STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/ui": ("index.html", "text/html; charset=utf-8"),
    "/ui/": ("index.html", "text/html; charset=utf-8"),
    "/assets/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
_CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "connect-src 'self'; "
    "img-src 'self' data:; "
    "style-src 'self'; "
    "script-src 'self'; "
    "base-uri 'none'; "
    "frame-ancestors 'none'; "
    "form-action 'self'"
)


class ApiRequestError(ValueError):
    """A safe client-facing API request error."""

    def __init__(self, message: str, *, status: int = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = int(status)


class RetrievalService(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: object = None,
    ) -> tuple[RagSource, ...]: ...

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filters: object = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> RagAnswer: ...


ServiceFactory = Callable[[AppConfig, str], RetrievalService]


class ApiApplication:
    """Validate JSON requests and invoke cached source-specific RAG services."""

    def __init__(
        self,
        config: AppConfig,
        *,
        service_factory: ServiceFactory | None = None,
    ) -> None:
        self._config = config
        self._service_factory = service_factory or (
            lambda value, source_type: RagService(value, source_type=source_type)
        )
        self._services: dict[str, RetrievalService] = {}
        self._service_lock = threading.Lock()
        self._request_lock = threading.Lock()

    def dispatch(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        route = urlsplit(path).path.rstrip("/") or "/"
        if route == "/health":
            if method != "GET":
                raise ApiRequestError(
                    "method not allowed",
                    status=HTTPStatus.METHOD_NOT_ALLOWED,
                )
            return HTTPStatus.OK, {
                "status": "ok",
                "service": "EquipmentRAG",
                "api_version": "v1",
                "release_version": __version__,
                "equipment": self._config.equipment.name,
            }
        if route not in {"/v1/retrieve", "/v1/answer"}:
            raise ApiRequestError("endpoint not found", status=HTTPStatus.NOT_FOUND)
        if method != "POST":
            raise ApiRequestError(
                "method not allowed",
                status=HTTPStatus.METHOD_NOT_ALLOWED,
            )
        if not isinstance(payload, Mapping):
            raise ApiRequestError("request body must be a JSON object")

        source_type = _source_type(payload.get("source_type", "all"))
        top_k = _optional_top_k(payload.get("top_k"))
        include_content = _boolean(payload.get("include_content", False), "include_content")
        filters = build_json_filters(payload.get("filters"), source_type, self._config)

        if route == "/v1/retrieve":
            _reject_unknown_keys(
                payload,
                {"query", "source_type", "top_k", "filters", "include_content"},
            )
            query = _required_string(payload.get("query"), "query")
            service = self._service(source_type)
            with self._request_lock:
                sources = service.retrieve(query, top_k=top_k, filters=filters)
            return HTTPStatus.OK, {
                "query": query,
                "source_type": source_type,
                "result_count": len(sources),
                "sources": [
                    source.to_dict(include_code=include_content)
                    for source in sources
                ],
            }

        _reject_unknown_keys(
            payload,
            {
                "question",
                "source_type",
                "top_k",
                "filters",
                "include_content",
                "temperature",
                "max_tokens",
            },
        )
        question = _required_string(payload.get("question"), "question")
        temperature = _number(payload.get("temperature", 0.1), "temperature")
        if not 0 <= temperature <= 2:
            raise ApiRequestError("temperature must be between 0 and 2")
        max_tokens = _optional_positive_int(payload.get("max_tokens"), "max_tokens")
        service = self._service(source_type)
        with self._request_lock:
            answer = service.ask(
                question,
                top_k=top_k,
                filters=filters,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return HTTPStatus.OK, answer.to_dict(
            include_source_code=include_content
        )

    def _service(self, source_type: str) -> RetrievalService:
        with self._service_lock:
            service = self._services.get(source_type)
            if service is None:
                service = self._service_factory(self._config, source_type)
                self._services[source_type] = service
            return service


class EquipmentRagHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], application: ApiApplication) -> None:
        self.application = application
        super().__init__(address, EquipmentRagRequestHandler)


class EquipmentRagRequestHandler(BaseHTTPRequestHandler):
    server: EquipmentRagHttpServer

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_PUT(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()

    def log_message(self, format: str, *args: object) -> None:
        return

    def _handle(self) -> None:
        try:
            if self.command == "GET":
                asset = load_static_asset(self.path)
                if asset is not None:
                    content_type, body = asset
                    self._write_static(HTTPStatus.OK, content_type, body)
                    return
            payload = self._read_payload() if self.command == "POST" else None
            status, response = self.server.application.dispatch(
                self.command,
                self.path,
                payload,
            )
        except ApiRequestError as exc:
            status, response = exc.status, {"error": str(exc)}
        except RagError as exc:
            status, response = HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)}
        except Exception:
            status, response = HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": "internal server error"
            }
        self._write_json(int(status), response)

    def _read_payload(self) -> Mapping[str, Any]:
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            raise ApiRequestError(
                "Content-Type must be application/json",
                status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ApiRequestError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ApiRequestError("Content-Length is invalid") from exc
        if length <= 0:
            raise ApiRequestError("request body is empty")
        if length > _MAX_REQUEST_BYTES:
            raise ApiRequestError(
                "request body is too large",
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiRequestError("request body is not valid UTF-8 JSON") from exc
        if not isinstance(payload, Mapping):
            raise ApiRequestError("request body must be a JSON object")
        return payload

    def _write_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", _CONTENT_SECURITY_POLICY)
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _write_static(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", _CONTENT_SECURITY_POLICY)
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)


def load_static_asset(path: str) -> tuple[str, bytes] | None:
    route = urlsplit(path).path
    definition = _STATIC_ROUTES.get(route)
    if definition is None:
        return None
    file_name, content_type = definition
    try:
        body = (_WEB_ROOT / file_name).read_bytes()
    except OSError as exc:
        raise RuntimeError("EquipmentRAG UI asset is unavailable") from exc
    return content_type, body


def build_json_filters(
    value: object,
    source_type: str,
    config: AppConfig,
) -> CodeSearchFilters | DocumentSearchFilters | UnifiedSearchFilters:
    if value is None:
        data: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        data = value
    else:
        raise ApiRequestError("filters must be a JSON object")
    _reject_unknown_keys(data, {"code", "document"}, prefix="filters")
    code_data = _mapping(data.get("code"), "filters.code")
    document_data = _mapping(data.get("document"), "filters.document")

    code = None
    if source_type in {"code", "all"}:
        _reject_unknown_keys(
            code_data,
            {"equipment", "repository", "relative_path", "class_name", "method_name"},
            prefix="filters.code",
        )
        code = CodeSearchFilters(
            equipment=_optional_string(code_data.get("equipment"), "filters.code.equipment")
            or config.equipment.name,
            repository=_optional_string(code_data.get("repository"), "filters.code.repository"),
            relative_path=_optional_string(code_data.get("relative_path"), "filters.code.relative_path"),
            class_name=_optional_string(code_data.get("class_name"), "filters.code.class_name"),
            method_name=_optional_string(code_data.get("method_name"), "filters.code.method_name"),
        )

    document = None
    if source_type in {"document", "all"}:
        _reject_unknown_keys(
            document_data,
            {
                "project",
                "equipment",
                "unit",
                "document_type",
                "revision",
                "document_status",
                "is_latest",
                "document_id",
                "file_extension",
            },
            prefix="filters.document",
        )
        revision = _optional_string(
            document_data.get("revision"),
            "filters.document.revision",
        )
        status = (
            _optional_string(
                document_data.get("document_status"),
                "filters.document.document_status",
            )
            if "document_status" in document_data
            else (None if revision else "active")
        )
        latest = (
            _optional_boolean(
                document_data.get("is_latest"),
                "filters.document.is_latest",
            )
            if "is_latest" in document_data
            else (None if revision else True)
        )
        document = DocumentSearchFilters(
            project=_optional_string(document_data.get("project"), "filters.document.project"),
            equipment=_optional_string(document_data.get("equipment"), "filters.document.equipment")
            or config.equipment.name,
            unit=_optional_string(document_data.get("unit"), "filters.document.unit"),
            document_type=_optional_string(document_data.get("document_type"), "filters.document.document_type"),
            revision=revision,
            document_status=status,
            is_latest=latest,
            document_id=_optional_string(document_data.get("document_id"), "filters.document.document_id"),
            file_extension=_optional_string(document_data.get("file_extension"), "filters.document.file_extension"),
        )

    if source_type == "all":
        return UnifiedSearchFilters(code=code, document=document)
    if source_type == "document":
        assert document is not None
        return document
    assert code is not None
    return code


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ApiRequestError(f"{name} must be a JSON object")
    return value


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    *,
    prefix: str = "request",
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ApiRequestError(f"unknown {prefix} field(s): {', '.join(unknown)}")


def _source_type(value: object) -> str:
    if not isinstance(value, str) or value not in _SOURCE_TYPES:
        raise ApiRequestError("source_type must be 'code', 'document', or 'all'")
    return value


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApiRequestError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, name)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ApiRequestError(f"{name} must be a boolean")
    return value


def _optional_boolean(value: object, name: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, name)


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ApiRequestError(f"{name} must be a positive integer or null")
    return value


def _optional_top_k(value: object) -> int | None:
    result = _optional_positive_int(value, "top_k")
    if result is not None and result > 100:
        raise ApiRequestError("top_k must not exceed 100")
    return result


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiRequestError(f"{name} must be a number")
    return float(value)


def validate_bind_address(host: str, *, allow_remote: bool) -> None:
    if allow_remote or host.casefold() == "localhost":
        return
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise ApiRequestError(
            "non-loopback binding requires --allow-remote"
        )


def serve(
    config: AppConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    allow_remote: bool = False,
) -> None:
    validate_bind_address(host, allow_remote=allow_remote)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ApiRequestError("port must be between 1 and 65535")
    server = EquipmentRagHttpServer((host, port), ApiApplication(config))
    try:
        print(f"EquipmentRAG API listening on http://{host}:{port}")
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local EquipmentRAG JSON API")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-remote", action="store_true")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        serve(
            config,
            host=args.host,
            port=args.port,
            allow_remote=args.allow_remote,
        )
    except (ConfigError, ApiRequestError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
