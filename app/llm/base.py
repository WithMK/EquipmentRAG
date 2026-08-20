"""Provider-neutral contracts for local LLM clients."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, Sequence


class LlmError(RuntimeError):
    """Raised when an LLM request or response is invalid."""


@dataclass(frozen=True)
class ChatMessage:
    """One text message sent to an OpenAI-compatible chat endpoint."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise LlmError("message role must be system, user, or assistant")
        if not isinstance(self.content, str) or not self.content.strip():
            raise LlmError("message content must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class LlmUsage:
    """Token usage reported by a local model server when available."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LlmError(f"usage.{field_name} must be a non-negative integer")


@dataclass(frozen=True)
class LlmResponse:
    """Normalized response shared by all local LLM providers."""

    content: str
    model: str
    finish_reason: str
    usage: LlmUsage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise LlmError("LLM response content must be a non-empty string")
        for field_name in ("model", "finish_reason"):
            if not isinstance(getattr(self, field_name), str):
                raise LlmError(f"LLM response {field_name} must be a string")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LlmClient(Protocol):
    """Interface implemented by llama.cpp and future Ollama clients."""

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> LlmResponse: ...
