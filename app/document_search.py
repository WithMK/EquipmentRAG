"""CLI for library-first semantic document retrieval."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from app.config import ConfigError, load_config
from app.retrieval.document_retriever import (
    DocumentRetrievalError,
    DocumentRetriever,
    DocumentSearchFilters,
    format_document_results,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search indexed local documents")
    parser.add_argument("query", help="Natural-language document query")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--project")
    parser.add_argument("--equipment")
    parser.add_argument("--unit")
    parser.add_argument("--document-type")
    parser.add_argument("--revision")
    parser.add_argument("--document-status")
    parser.add_argument("--document-id")
    parser.add_argument("--file-extension")
    parser.add_argument("--include-obsolete", action="store_true")
    parser.add_argument("--all-revisions", action="store_true")
    parser.add_argument("--chroma-path")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-text", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.chroma_path:
            config = replace(
                config,
                chromadb=replace(
                    config.chromadb,
                    path=Path(args.chroma_path).expanduser().resolve(strict=False),
                ),
            )
        explicit_revision = args.revision is not None
        status = args.document_status
        if status is None and not args.include_obsolete and not explicit_revision:
            status = "active"
        latest = None if args.all_revisions or explicit_revision else True
        if args.include_obsolete:
            status = None
            latest = None
        filters = DocumentSearchFilters(
            project=args.project,
            equipment=args.equipment or config.equipment.name,
            unit=args.unit,
            document_type=args.document_type,
            revision=args.revision,
            document_status=status,
            is_latest=latest,
            document_id=args.document_id,
            file_extension=args.file_extension,
        )
        results = DocumentRetriever(config).search(
            args.query,
            top_k=args.top_k,
            filters=filters,
        )
    except (ConfigError, DocumentRetrievalError) as exc:
        parser.error(str(exc))

    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "result_count": len(results),
                    "results": [
                        result.to_dict(include_text=not args.no_text)
                        for result in results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(format_document_results(results, include_text=not args.no_text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
