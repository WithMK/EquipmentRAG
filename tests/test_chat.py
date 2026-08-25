from __future__ import annotations

import unittest
from collections.abc import Sequence
from typing import Any

from app.chat import ConversationSession, run_chat
from app.llm.base import ChatMessage
from app.rag_service import RagAnswer, RagError, RagSource


def _source() -> RagSource:
    return RagSource(
        source_id="C1",
        record_id="code-1",
        score=0.91,
        file_name="AxisController.cs",
        relative_path="Motion/AxisController.cs",
        file_path="C:/synthetic/Motion/AxisController.cs",
        class_name="AxisController",
        method_name="HomeZAxis",
        start_line=10,
        end_line=20,
        code="public void HomeZAxis() {}",
    )


class FakeConversationRag:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filters: Any = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        conversation: Sequence[ChatMessage] = (),
        retrieval_query: str | None = None,
    ) -> RagAnswer:
        self.calls.append(
            {
                "question": question,
                "top_k": top_k,
                "filters": filters,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "conversation": tuple(conversation),
                "retrieval_query": retrieval_query,
            }
        )
        return RagAnswer(
            question=question,
            answer=f"answer-{len(self.calls)}",
            sources=(_source(),),
            model="test-model",
            finish_reason="stop",
        )


class ConversationSessionTests(unittest.TestCase):
    def test_follow_up_query_and_history_are_bounded(self) -> None:
        service = FakeConversationRag()
        session = ConversationSession(
            service,
            filters="filters",
            top_k=6,
            temperature=0.2,
            max_tokens=512,
            max_history_turns=1,
        )

        first = session.ask("Vacuum alarm 원인은?")
        second = session.ask("복구 절차는?")
        session.ask("관련 메서드는?")

        self.assertEqual(first.answer, "answer-1")
        self.assertEqual(second.answer, "answer-2")
        self.assertEqual(service.calls[0]["conversation"], ())
        self.assertEqual(
            service.calls[0]["retrieval_query"],
            "Vacuum alarm 원인은?",
        )
        self.assertEqual(
            service.calls[1]["conversation"],
            (
                ChatMessage("user", "Vacuum alarm 원인은?"),
                ChatMessage("assistant", "answer-1"),
            ),
        )
        self.assertIn("Vacuum alarm 원인은?", service.calls[1]["retrieval_query"])
        self.assertIn("복구 절차는?", service.calls[1]["retrieval_query"])
        self.assertNotIn("Vacuum alarm 원인은?", service.calls[2]["retrieval_query"])
        self.assertIn("복구 절차는?", service.calls[2]["retrieval_query"])
        self.assertEqual(service.calls[2]["top_k"], 6)
        self.assertEqual(service.calls[2]["filters"], "filters")
        self.assertEqual(service.calls[2]["temperature"], 0.2)
        self.assertEqual(service.calls[2]["max_tokens"], 512)

    def test_clear_removes_history_and_last_answer(self) -> None:
        service = FakeConversationRag()
        session = ConversationSession(service)
        session.ask("first")

        session.clear()

        self.assertEqual(session.turns, ())
        self.assertIsNone(session.last_answer)
        session.ask("new topic")
        self.assertEqual(service.calls[-1]["conversation"], ())
        self.assertEqual(service.calls[-1]["retrieval_query"], "new topic")

    def test_rejects_invalid_history_limit_and_blank_question(self) -> None:
        with self.assertRaisesRegex(RagError, "max_history_turns"):
            ConversationSession(FakeConversationRag(), max_history_turns=0)
        session = ConversationSession(FakeConversationRag())
        with self.assertRaisesRegex(RagError, "question"):
            session.ask("  ")

    def test_interactive_commands_show_sources_and_clear_state(self) -> None:
        service = FakeConversationRag()
        session = ConversationSession(service)
        inputs = iter(
            ("Vacuum alarm?", "/sources", "/clear", "/sources", "/exit")
        )
        outputs: list[str] = []

        exit_code = run_chat(
            session,
            input_fn=lambda _prompt: next(inputs),
            output_fn=outputs.append,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(service.calls), 1)
        self.assertTrue(any("Assistant> answer-1" in item for item in outputs))
        self.assertTrue(any("[C1]" in item for item in outputs))
        self.assertIn("Conversation history cleared.", outputs)
        self.assertIn("No previous answer sources.", outputs)
        self.assertEqual(outputs[-1], "Chat ended.")


if __name__ == "__main__":
    unittest.main()
