"""Create local LLM clients without coupling callers to a provider."""

from __future__ import annotations

import os

from app.config import LlmConfig
from app.llm.base import LlmClient, LlmError
from app.llm.llama_client import LlamaCppClient


def create_llm_client(config: LlmConfig) -> LlmClient:
    """Create the configured local provider client."""

    if config.provider == "llama_cpp":
        return LlamaCppClient(
            config,
            api_key=os.environ.get("LLAMA_CPP_API_KEY"),
        )
    raise LlmError(
        f"LLM provider '{config.provider}' is not implemented; use llama_cpp for Phase 8"
    )
