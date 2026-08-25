"""Interactive multi-turn chat for local EquipmentRAG."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from app.config import AppConfig, ConfigError, load_config
from app.llm.base import ChatMessage
from app.rag_service import (
    RagAnswer,
    RagError,
    RagService,
    UnifiedSearchFilters,
    format_rag_sources,
)
from app.retrieval.document_retriever import (
    DocumentRetrievalError,
    DocumentSearchFilters,
)
from app.search import CodeSearchFilters, SearchError


SearchFilters = (
    CodeSearchFilters
    | DocumentSearchFilters
    | UnifiedSearchFilters
    | None
)


class ConversationRagProvider(Protocol):
    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filters: SearchFilters = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        conversation: Sequence[ChatMessage] = (),
        retrieval_query: str | None = None,
    ) -> RagAnswer: ...


@dataclass(frozen=True)
class ConversationTurn:
    """One completed user question and grounded answer."""

    question: str
    retrieval_query: str
    answer: RagAnswer


class ConversationSession:
    """Keep bounded chat history while retrieving fresh evidence every turn."""

    def __init__(
        self,
        service: ConversationRagProvider,
        *,
        filters: SearchFilters = None,
        top_k: int | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        max_history_turns: int = 4,
    ) -> None:
        if (
            isinstance(max_history_turns, bool)
            or not isinstance(max_history_turns, int)
            or max_history_turns <= 0
        ):
            raise RagError("max_history_turns must be a positive integer")
        self._service = service
        self._filters = filters
        self._top_k = top_k
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_history_turns = max_history_turns
        self._turns: list[ConversationTurn] = []

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        return tuple(self._turns)

    @property
    def last_answer(self) -> RagAnswer | None:
        return self._turns[-1].answer if self._turns else None

    def ask(self, question: str) -> RagAnswer:
        normalized = question.strip() if isinstance(question, str) else question
        if not isinstance(normalized, str) or not normalized:
            raise RagError("question must be a non-empty string")
        history_turns = self._turns[-self._max_history_turns :]
        conversation = tuple(
            message
            for turn in history_turns
            for message in (
                ChatMessage("user", turn.question),
                ChatMessage("assistant", turn.answer.answer),
            )
        )
        retrieval_query = _build_follow_up_query(history_turns, normalized)
        answer = self._service.ask(
            normalized,
            top_k=self._top_k,
            filters=self._filters,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            conversation=conversation,
            retrieval_query=retrieval_query,
        )
        self._turns.append(
            ConversationTurn(
                question=normalized,
                retrieval_query=retrieval_query,
                answer=answer,
            )
        )
        return answer

    def clear(self) -> None:
        self._turns.clear()


def _build_follow_up_query(
    history: Sequence[ConversationTurn],
    question: str,
) -> str:
    if not history:
        return question
    previous_questions = "\n".join(f"- {turn.question}" for turn in history)
    return (
        "이전 대화의 질문:\n"
        f"{previous_questions}\n"
        "현재 후속 질문:\n"
        f"{question}"
    )


HELP_TEXT = """Commands:
  /sources  Show sources used for the last answer
  /clear    Clear conversation history
  /help     Show this command list
  /exit     End the chat"""


def run_chat(
    session: ConversationSession,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    show_sources: bool = False,
    include_source_content: bool = False,
) -> int:
    """Run an interactive prompt using injectable terminal I/O."""

    output_fn("EquipmentRAG interactive chat. Type /help for commands.")
    while True:
        try:
            raw = input_fn("You> ")
        except (EOFError, KeyboardInterrupt):
            output_fn("Chat ended.")
            return 0
        question = raw.strip()
        if not question:
            continue
        command = question.casefold()
        if command in {"/exit", "/quit"}:
            output_fn("Chat ended.")
            return 0
        if command == "/help":
            output_fn(HELP_TEXT)
            continue
        if command == "/clear":
            session.clear()
            output_fn("Conversation history cleared.")
            continue
        if command == "/sources":
            answer = session.last_answer
            output_fn(
                "No previous answer sources."
                if answer is None
                else format_rag_sources(
                    answer.sources,
                    include_source_code=include_source_content,
                )
            )
            continue
        if command.startswith("/"):
            output_fn("Unknown command. Type /help for commands.")
            continue

        try:
            answer = session.ask(question)
        except RagError as exc:
            output_fn(f"Error: {exc}")
            continue
        output_fn(f"Assistant> {answer.answer}")
        if show_sources:
            output_fn(
                format_rag_sources(
                    answer.sources,
                    include_source_code=include_source_content,
                )
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive grounded chat over indexed code and documents"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--source-type",
        choices=("code", "document", "all"),
        default="all",
    )
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-history-turns", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int)
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
    parser.add_argument("--chroma-path")
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument("--include-source-content", action="store_true")
    return parser


def build_search_filters(
    args: argparse.Namespace,
    config: AppConfig,
) -> SearchFilters:
    """Build source-specific metadata filters from shared CLI arguments."""
    code_filters = None
    if args.source_type in {"code", "all"}:
        code_filters = CodeSearchFilters(
            equipment=args.equipment or config.equipment.name,
            repository=args.repository,
            relative_path=args.relative_path,
            class_name=args.class_name,
            method_name=args.method_name,
        )

    document_filters = None
    if args.source_type in {"document", "all"}:
        explicit_revision = args.revision is not None
        status = args.document_status
        if status is None and not args.include_obsolete and not explicit_revision:
            status = "active"
        latest = None if args.all_revisions or explicit_revision else True
        if args.include_obsolete:
            status = None
            latest = None
        document_filters = DocumentSearchFilters(
            project=args.project,
            equipment=args.equipment or config.equipment.name,
            unit=args.unit,
            document_type=args.document_type,
            revision=args.revision,
            document_status=status,
            is_latest=latest,
        )

    if args.source_type == "all":
        return UnifiedSearchFilters(
            code=code_filters,
            document=document_filters,
        )
    if args.source_type == "document":
        return document_filters
    return code_filters


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
        filters = build_search_filters(args, config)
        service = RagService(config, source_type=args.source_type)
        session = ConversationSession(
            service,
            filters=filters,
            top_k=args.top_k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_history_turns=args.max_history_turns,
        )
    except (
        ConfigError,
        SearchError,
        DocumentRetrievalError,
        RagError,
    ) as exc:
        parser.error(str(exc))

    return run_chat(
        session,
        show_sources=args.show_sources,
        include_source_content=args.include_source_content,
    )


if __name__ == "__main__":
    raise SystemExit(main())
