"""Unified operator CLI for local EquipmentRAG workflows."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dataclasses import replace

from app.api import ApiRequestError, serve as serve_api
from app import __version__
from app.chat import ConversationSession, build_search_filters, run_chat
from app.config import AppConfig, ConfigError, load_config
from app.document_indexer import (
    DocumentIndexerError,
    IncrementalDocumentIndexer,
)
from app.embedding.embedding_service import LocalEmbeddingService
from app.evaluation import (
    EvaluationError,
    evaluate_retrieval,
    format_evaluation_report,
)
from app.indexer import IncrementalSourceIndexer, IndexerError
from app.rag_service import RagError, RagService, format_rag_answer, format_rag_sources
from app.retrieval.document_retriever import DocumentRetrievalError
from app.search import SearchError


def build_status(config: AppConfig) -> dict[str, object]:
    """Build a read-only local readiness and index-state summary."""

    code_indexer = IncrementalSourceIndexer(config)
    document_state: dict[str, object] | None = None
    document_paths: list[dict[str, object]] = []
    if config.document is not None:
        document_paths = [
            {"path": str(path), "exists": path.is_dir()}
            for path in config.document.source_paths
        ]
        if config.document.enabled:
            document_indexer = IncrementalDocumentIndexer(config)
            document_state = _read_state_summary(document_indexer.state_path)

    return {
        "version": __version__,
        "equipment": config.equipment.name,
        "project_root": str(config.project_root),
        "code_source": {
            "path": str(config.source.path),
            "exists": config.source.path.is_dir(),
            "index": _read_state_summary(code_indexer.state_path),
        },
        "documents": {
            "configured": config.document is not None,
            "enabled": bool(config.document and config.document.enabled),
            "source_paths": document_paths,
            "index": document_state,
        },
        "embedding_model": {
            "path": str(config.embedding.model_path),
            "exists": config.embedding.model_path.is_dir(),
        },
        "chromadb": {
            "path": str(config.chromadb.path),
            "exists": config.chromadb.path.is_dir(),
            "code_collection": config.chromadb.collection_name,
            "document_collection": (
                config.document.collection_name if config.document else None
            ),
        },
        "llm": {
            "provider": config.llm.provider,
            "base_url": config.llm.base_url,
            "model": config.llm.model,
        },
        "search": {
            "mode": config.search.mode,
            "top_k": config.search.top_k,
            "candidate_multiplier": config.search.candidate_multiplier,
            "semantic_weight": config.search.semantic_weight,
            "lexical_weight": config.search.lexical_weight,
            "reranker_model_path": (
                str(config.search.reranker_model_path)
                if config.search.reranker_model_path
                else None
            ),
            "reranker_ready": bool(
                config.search.reranker_model_path
                and config.search.reranker_model_path.is_dir()
            ),
            "reranker_weight": config.search.reranker_weight,
        },
        "visual_documents": {
            "configured": config.visual is not None,
            "enabled": bool(config.visual and config.visual.enabled),
            "tesseract_path": (
                str(config.visual.tesseract_path)
                if config.visual and config.visual.tesseract_path
                else None
            ),
            "tesseract_ready": bool(
                config.visual
                and config.visual.tesseract_path
                and config.visual.tesseract_path.is_file()
            ),
            "pdf_ocr": bool(config.visual and config.visual.pdf_ocr),
            "pptx_image_ocr": bool(
                config.visual and config.visual.pptx_image_ocr
            ),
            "xlsx_chart_extraction": bool(
                config.visual and config.visual.xlsx_chart_extraction
            ),
        },
    }


def _read_state_summary(path: Path) -> dict[str, object]:
    summary: dict[str, object] = {
        "path": str(path),
        "exists": path.is_file(),
        "valid": False,
        "file_count": 0,
        "chunk_count": 0,
    }
    if not path.is_file():
        return summary
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        files = payload.get("files") if isinstance(payload, dict) else None
        if not isinstance(files, dict):
            raise ValueError("state files value is invalid")
        chunk_count = 0
        for value in files.values():
            if not isinstance(value, dict) or not isinstance(
                value.get("chunk_ids"), list
            ):
                raise ValueError("state file entry is invalid")
            chunk_count += len(value["chunk_ids"])
        summary.update(
            {
                "valid": True,
                "file_count": len(files),
                "chunk_count": chunk_count,
                "modified_at": datetime.fromtimestamp(
                    path.stat().st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
            }
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        summary["error"] = str(exc)
    return summary


def format_status(status: dict[str, object]) -> str:
    code = status["code_source"]
    documents = status["documents"]
    embedding = status["embedding_model"]
    chromadb = status["chromadb"]
    llm = status["llm"]
    search = status["search"]
    assert isinstance(code, dict)
    assert isinstance(documents, dict)
    assert isinstance(embedding, dict)
    assert isinstance(chromadb, dict)
    assert isinstance(llm, dict)
    assert isinstance(search, dict)
    code_index = code["index"]
    assert isinstance(code_index, dict)
    document_index = documents.get("index")
    lines = [
        f"Version: {status['version']}",
        f"Equipment: {status['equipment']}",
        f"Code source: {_presence(code['exists'])} ({code['path']})",
        (
            "Code index: "
            f"{_state_label(code_index)} "
            f"(files={code_index['file_count']}, chunks={code_index['chunk_count']})"
        ),
        (
            "Documents: "
            f"{'enabled' if documents['enabled'] else 'disabled'} "
            f"({len(documents['source_paths'])} source path(s))"
        ),
    ]
    if isinstance(document_index, dict):
        lines.append(
            "Document index: "
            f"{_state_label(document_index)} "
            f"(files={document_index['file_count']}, "
            f"chunks={document_index['chunk_count']})"
        )
    else:
        lines.append(
            "Document index: disabled"
            if documents["configured"]
            else "Document index: not configured"
        )
    lines.extend(
        (
            f"Embedding model: {_presence(embedding['exists'])} ({embedding['path']})",
            f"ChromaDB: {_presence(chromadb['exists'])} ({chromadb['path']})",
            f"LLM: {llm['provider']} / {llm['model']} ({llm['base_url']})",
            (
                f"Search: {search['mode']} "
                f"(top_k={search['top_k']}, candidates=x{search['candidate_multiplier']})"
            ),
            (
                "Reranker: "
                + (
                    "ready"
                    if search["reranker_ready"]
                    else "not configured"
                    if search["reranker_model_path"] is None
                    else "missing"
                )
            ),
        )
    )
    return "\n".join(lines)


def _presence(value: object) -> str:
    return "ready" if value is True else "missing"


def _state_label(state: dict[str, object]) -> str:
    if state["valid"] is True:
        return "ready"
    return "invalid" if state["exists"] is True else "not created"


def _load_command_config(args: argparse.Namespace) -> AppConfig:
    config = load_config(args.config)
    if getattr(args, "chroma_path", None):
        config = replace(
            config,
            chromadb=replace(
                config.chromadb,
                path=Path(args.chroma_path).expanduser().resolve(strict=False),
            ),
        )
    search_updates: dict[str, object] = {}
    for argument, field_name in (
        ("search_mode", "mode"),
        ("candidate_multiplier", "candidate_multiplier"),
        ("semantic_weight", "semantic_weight"),
        ("lexical_weight", "lexical_weight"),
        ("reranker_weight", "reranker_weight"),
        ("reranker_device", "reranker_device"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            search_updates[field_name] = value
    reranker_path = getattr(args, "reranker_model_path", None)
    if reranker_path:
        search_updates["reranker_model_path"] = (
            Path(reranker_path).expanduser().resolve(strict=False)
        )
    if search_updates:
        config = replace(
            config,
            search=replace(config.search, **search_updates),
        )
    _validate_runtime_search_config(config)
    return config


def _validate_runtime_search_config(config: AppConfig) -> None:
    search = config.search
    if (
        isinstance(search.candidate_multiplier, bool)
        or not isinstance(search.candidate_multiplier, int)
        or search.candidate_multiplier <= 0
    ):
        raise ConfigError("candidate multiplier must be a positive integer")
    for name, value in (
        ("semantic weight", search.semantic_weight),
        ("lexical weight", search.lexical_weight),
        ("reranker weight", search.reranker_weight),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ConfigError(f"{name} must be between 0 and 1")
    if search.semantic_weight + search.lexical_weight <= 0:
        raise ConfigError("semantic and lexical weights cannot both be zero")


def _run_status(args: argparse.Namespace) -> int:
    status = build_status(_load_command_config(args))
    print(
        json.dumps(status, ensure_ascii=False, indent=2)
        if args.json
        else format_status(status)
    )
    return 0


def _run_index(args: argparse.Namespace) -> int:
    config = _load_command_config(args)
    if args.code_source:
        config = replace(
            config,
            source=replace(
                config.source,
                path=Path(args.code_source).expanduser().resolve(strict=False),
            ),
        )
    if args.document_source:
        if config.document is None:
            raise DocumentIndexerError("Document configuration is missing")
        config = replace(
            config,
            document=replace(
                config.document,
                enabled=True,
                source_paths=tuple(
                    Path(path).expanduser().resolve(strict=False)
                    for path in args.document_source
                ),
            ),
        )

    shared_embedding = LocalEmbeddingService(config.embedding)
    reports: dict[str, object] = {}
    if args.source_type in {"code", "all"}:
        reports["code"] = IncrementalSourceIndexer(
            config,
            embedding=shared_embedding,
        ).run(dry_run=args.dry_run, full_reindex=args.full).to_dict()
    if args.source_type in {"document", "all"}:
        if config.document is None:
            if args.source_type == "document":
                raise DocumentIndexerError("Document configuration is missing")
            reports["document"] = {
                "skipped": True,
                "reason": "document_not_configured",
            }
        elif args.source_type == "all" and not config.document.enabled:
            reports["document"] = {
                "skipped": True,
                "reason": "document_disabled",
            }
        else:
            reports["document"] = IncrementalDocumentIndexer(
                config,
                embedding=shared_embedding,
            ).run(dry_run=args.dry_run, full_reindex=args.full).to_dict()

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
    else:
        print(_format_index_reports(reports))
    return 0


def _format_index_reports(reports: dict[str, object]) -> str:
    blocks: list[str] = []
    for source_type, value in reports.items():
        if not isinstance(value, dict):
            continue
        if value.get("skipped"):
            blocks.append(f"{source_type.title()}: skipped ({value['reason']})")
            continue
        blocks.append(
            "\n".join(
                (
                    f"{source_type.title()} index:",
                    f"  files: {value['total_files']}",
                    f"  new: {len(value['new_files'])}",
                    f"  changed: {len(value['changed_files'])}",
                    f"  deleted: {len(value['deleted_files'])}",
                    f"  prepared chunks: {value['prepared_chunks']}",
                    f"  upserted chunks: {value['upserted_chunks']}",
                    f"  dry run: {value['dry_run']}",
                )
            )
        )
    return "\n\n".join(blocks)


def _run_search(args: argparse.Namespace) -> int:
    config = _load_command_config(args)
    filters = build_search_filters(args, config)
    sources = RagService(config, source_type=args.source_type).retrieve(
        args.query,
        top_k=args.top_k,
        filters=filters,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "source_type": args.source_type,
                    "result_count": len(sources),
                    "sources": [
                        source.to_dict(include_code=args.include_content)
                        for source in sources
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(
            format_rag_sources(
                sources,
                include_source_code=args.include_content,
            )
        )
    return 0


def _run_ask(args: argparse.Namespace) -> int:
    config = _load_command_config(args)
    filters = build_search_filters(args, config)
    result = RagService(config, source_type=args.source_type).ask(
        args.question,
        top_k=args.top_k,
        filters=filters,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    if args.json:
        print(
            json.dumps(
                result.to_dict(include_source_code=args.include_content),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(
            format_rag_answer(
                result,
                include_source_code=args.include_content,
            )
        )
    return 0


def _run_chat(args: argparse.Namespace) -> int:
    config = _load_command_config(args)
    filters = build_search_filters(args, config)
    session = ConversationSession(
        RagService(config, source_type=args.source_type),
        filters=filters,
        top_k=args.top_k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_history_turns=args.max_history_turns,
    )
    return run_chat(
        session,
        show_sources=args.show_sources,
        include_source_content=args.include_content,
    )


def _run_serve(args: argparse.Namespace) -> int:
    serve_api(
        _load_command_config(args),
        host=args.host,
        port=args.port,
        allow_remote=args.allow_remote,
    )
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    config = _load_command_config(args)
    for name in ("min_hit_rate", "min_recall", "min_mrr"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise EvaluationError(
                f"{name.replace('_', '-')} must be between 0 and 1"
            )
    report = evaluate_retrieval(
        config,
        Path(args.dataset),
        default_top_k=args.top_k,
    )
    payload = report.to_dict()
    if args.output:
        output_path = Path(args.output).expanduser().resolve(strict=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(payload, ensure_ascii=False, indent=2)
        if args.json
        else format_evaluation_report(report)
    )
    passed = (
        report.hit_rate >= args.min_hit_rate
        and report.recall_at_k >= args.min_recall
        and report.mrr >= args.min_mrr
    )
    if not passed:
        print("Evaluation quality threshold was not met.", file=sys.stderr)
        return 1
    return 0


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--chroma-path")


def _add_source_type(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-type",
        choices=("code", "document", "all"),
        default="all",
    )


def _add_filter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--equipment")
    parser.add_argument("--repository")
    parser.add_argument("--relative-path")
    parser.add_argument("--class-name")
    parser.add_argument("--method-name")
    parser.add_argument("--project")
    parser.add_argument("--unit")
    parser.add_argument("--document-type")
    parser.add_argument("--revision")
    parser.add_argument("--document-status")
    parser.add_argument("--include-obsolete", action="store_true")
    parser.add_argument("--all-revisions", action="store_true")


def _add_retrieval_arguments(parser: argparse.ArgumentParser) -> None:
    _add_config_arguments(parser)
    _add_source_type(parser)
    _add_filter_arguments(parser)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--search-mode", choices=("semantic", "hybrid"))
    parser.add_argument("--candidate-multiplier", type=int)
    parser.add_argument("--semantic-weight", type=float)
    parser.add_argument("--lexical-weight", type=float)
    parser.add_argument("--reranker-model-path")
    parser.add_argument("--reranker-weight", type=float)
    parser.add_argument("--reranker-device")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="equipment-rag",
        description="Unified local EquipmentRAG operator command",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="Show local readiness and index state")
    _add_config_arguments(status)
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler=_run_status)

    index = commands.add_parser("index", help="Index code and documents")
    _add_config_arguments(index)
    _add_source_type(index)
    index.add_argument("--code-source")
    index.add_argument("--document-source", action="append")
    index.add_argument("--dry-run", action="store_true")
    index.add_argument("--full", action="store_true")
    index.add_argument("--json", action="store_true")
    index.set_defaults(handler=_run_index)

    search = commands.add_parser("search", help="Retrieve sources without an LLM")
    search.add_argument("query")
    _add_retrieval_arguments(search)
    search.add_argument("--include-content", action="store_true")
    search.add_argument("--json", action="store_true")
    search.set_defaults(handler=_run_search)

    ask = commands.add_parser("ask", help="Generate one grounded answer")
    ask.add_argument("question")
    _add_retrieval_arguments(ask)
    ask.add_argument("--temperature", type=float, default=0.1)
    ask.add_argument("--max-tokens", type=int)
    ask.add_argument("--include-content", action="store_true")
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(handler=_run_ask)

    chat = commands.add_parser("chat", help="Start an interactive grounded chat")
    _add_retrieval_arguments(chat)
    chat.add_argument("--temperature", type=float, default=0.1)
    chat.add_argument("--max-tokens", type=int)
    chat.add_argument("--max-history-turns", type=int, default=4)
    chat.add_argument("--show-sources", action="store_true")
    chat.add_argument("--include-content", action="store_true")
    chat.set_defaults(handler=_run_chat)

    serve = commands.add_parser(
        "serve",
        help="Run the local retrieval and answer JSON API",
    )
    _add_config_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--allow-remote", action="store_true")
    serve.set_defaults(handler=_run_serve)

    evaluate = commands.add_parser(
        "evaluate",
        help="Measure offline retrieval quality from a JSONL dataset",
    )
    _add_config_arguments(evaluate)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--top-k", type=int)
    evaluate.add_argument("--min-hit-rate", type=float, default=0.0)
    evaluate.add_argument("--min-recall", type=float, default=0.0)
    evaluate.add_argument("--min-mrr", type=float, default=0.0)
    evaluate.add_argument("--output")
    evaluate.add_argument("--json", action="store_true")
    evaluate.set_defaults(handler=_run_evaluate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.handler(args)
    except (
        ConfigError,
        IndexerError,
        DocumentIndexerError,
        SearchError,
        DocumentRetrievalError,
        RagError,
        ApiRequestError,
        EvaluationError,
    ) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
