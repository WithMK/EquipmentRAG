"""Typed configuration loading for EquipmentRAG."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

import yaml


class ConfigError(ValueError):
    """Raised when the YAML configuration is missing or invalid."""


@dataclass(frozen=True)
class EquipmentConfig:
    name: str


@dataclass(frozen=True)
class SourceConfig:
    path: Path
    include_extensions: tuple[str, ...]
    exclude_directories: tuple[str, ...]
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class DocumentConfig:
    enabled: bool
    source_paths: tuple[Path, ...]
    extensions: tuple[str, ...]
    exclude_directories: tuple[str, ...]
    chunk_size: int
    chunk_overlap: int
    collection_name: str


@dataclass(frozen=True)
class EmbeddingConfig:
    model_path: Path
    batch_size: int
    device: str | None
    normalize_embeddings: bool


@dataclass(frozen=True)
class ChromaConfig:
    path: Path
    collection_name: str


@dataclass(frozen=True)
class SearchConfig:
    top_k: int


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    base_url: str
    model: str
    request_timeout_seconds: int


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    path: Path


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    equipment: EquipmentConfig
    source: SourceConfig
    embedding: EmbeddingConfig
    chromadb: ChromaConfig
    search: SearchConfig
    llm: LlmConfig
    logging: LoggingConfig
    document: DocumentConfig | None = None

    def safe_summary(self) -> dict[str, Any]:
        """Return a JSON-compatible summary without credential fields."""

        summary = asdict(self)
        return _paths_to_strings(summary)


def _paths_to_strings(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _paths_to_strings(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_paths_to_strings(item) for item in value]
    return value


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Missing or invalid '{name}' section")
    return value


def _required_string(data: Mapping[str, Any], key: str, section_name: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{section_name}.{key}' must be a non-empty string")
    return value.strip()


def _positive_int(data: Mapping[str, Any], key: str, section_name: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"'{section_name}.{key}' must be a positive integer")
    return value


def _non_negative_int(data: Mapping[str, Any], key: str, section_name: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"'{section_name}.{key}' must be a non-negative integer")
    return value


def _optional_string(
    data: Mapping[str, Any], key: str, section_name: str
) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{section_name}.{key}' must be a non-empty string or null")
    return value.strip()


def _boolean(data: Mapping[str, Any], key: str, section_name: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"'{section_name}.{key}' must be true or false")
    return value


def _string_tuple(data: Mapping[str, Any], key: str, section_name: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"'{section_name}.{key}' must be a list of strings")
    items = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if len(items) != len(value) or not items:
        raise ConfigError(f"'{section_name}.{key}' must contain non-empty strings")
    return items


def _path_tuple(
    data: Mapping[str, Any],
    key: str,
    section_name: str,
    project_root: Path,
) -> tuple[Path, ...]:
    values = _string_tuple(data, key, section_name)
    return tuple(_resolve_path(value, project_root) for value in values)


def _resolve_path(raw_path: str, project_root: Path) -> Path:
    expanded_text = os.path.expandvars(os.path.expanduser(raw_path))
    expanded = Path(expanded_text)
    if not expanded.is_absolute() and PureWindowsPath(expanded_text).is_absolute():
        return expanded
    if not expanded.is_absolute():
        expanded = project_root / expanded
    return expanded.resolve(strict=False)


def load_config(config_path: str | Path = "config/config.yaml") -> AppConfig:
    """Load and validate a YAML configuration file.

    Relative values in the YAML file are resolved from the project root, which
    is assumed to be the parent directory of the configuration directory.
    """

    path = Path(config_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Unable to read configuration: {path}") from exc

    if not isinstance(parsed, Mapping):
        raise ConfigError("Configuration root must be a mapping")

    project_root = path.parent.parent.resolve(strict=False)
    equipment = _section(parsed, "equipment")
    source = _section(parsed, "source")
    embedding = _section(parsed, "embedding")
    chromadb = _section(parsed, "chromadb")
    search = _section(parsed, "search")
    llm = _section(parsed, "llm")
    logging_config = _section(parsed, "logging")
    document_data = parsed.get("document")
    if document_data is not None and not isinstance(document_data, Mapping):
        raise ConfigError("'document' section must be a mapping")

    chunk_size = _positive_int(source, "chunk_size", "source")
    chunk_overlap = _non_negative_int(source, "chunk_overlap", "source")
    if chunk_overlap >= chunk_size:
        raise ConfigError("'source.chunk_overlap' must be smaller than 'source.chunk_size'")

    document_config: DocumentConfig | None = None
    if isinstance(document_data, Mapping):
        document_chunk_size = _positive_int(
            document_data, "chunk_size", "document"
        )
        document_chunk_overlap = _non_negative_int(
            document_data, "chunk_overlap", "document"
        )
        if document_chunk_overlap >= document_chunk_size:
            raise ConfigError(
                "'document.chunk_overlap' must be smaller than 'document.chunk_size'"
            )
        extensions = tuple(
            value if value.startswith(".") else f".{value}"
            for value in _string_tuple(document_data, "extensions", "document")
        )
        document_config = DocumentConfig(
            enabled=_boolean(document_data, "enabled", "document"),
            source_paths=_path_tuple(
                document_data,
                "source_paths",
                "document",
                project_root,
            ),
            extensions=tuple(value.casefold() for value in extensions),
            exclude_directories=_string_tuple(
                document_data,
                "exclude_directories",
                "document",
            ),
            chunk_size=document_chunk_size,
            chunk_overlap=document_chunk_overlap,
            collection_name=_required_string(
                document_data,
                "collection_name",
                "document",
            ),
        )

    provider = _required_string(llm, "provider", "llm").lower()
    if provider not in {"llama_cpp", "ollama"}:
        raise ConfigError("'llm.provider' must be 'llama_cpp' or 'ollama'")

    level = _required_string(logging_config, "level", "logging").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError("'logging.level' is invalid")

    return AppConfig(
        project_root=project_root,
        equipment=EquipmentConfig(name=_required_string(equipment, "name", "equipment")),
        source=SourceConfig(
            path=_resolve_path(_required_string(source, "path", "source"), project_root),
            include_extensions=_string_tuple(source, "include_extensions", "source"),
            exclude_directories=_string_tuple(source, "exclude_directories", "source"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
        embedding=EmbeddingConfig(
            model_path=_resolve_path(
                _required_string(embedding, "model_path", "embedding"), project_root
            ),
            batch_size=_positive_int(embedding, "batch_size", "embedding"),
            device=_optional_string(embedding, "device", "embedding"),
            normalize_embeddings=_boolean(
                embedding, "normalize_embeddings", "embedding"
            ),
        ),
        chromadb=ChromaConfig(
            path=_resolve_path(_required_string(chromadb, "path", "chromadb"), project_root),
            collection_name=_required_string(chromadb, "collection_name", "chromadb"),
        ),
        search=SearchConfig(top_k=_positive_int(search, "top_k", "search")),
        llm=LlmConfig(
            provider=provider,
            base_url=_required_string(llm, "base_url", "llm").rstrip("/"),
            model=_required_string(llm, "model", "llm"),
            request_timeout_seconds=_positive_int(
                llm, "request_timeout_seconds", "llm"
            ),
        ),
        logging=LoggingConfig(
            level=level,
            path=_resolve_path(
                _required_string(logging_config, "path", "logging"), project_root
            ),
        ),
        document=document_config,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate EquipmentRAG configuration")
    parser.add_argument("--config", default="config/config.yaml", help="YAML configuration path")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        parser.error(str(exc))

    print(json.dumps(config.safe_summary(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
