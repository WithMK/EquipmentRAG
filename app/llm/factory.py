"""Create local LLM clients without coupling callers to a provider."""

from __future__ import annotations

import os

from app.config import LlmConfig
from app.llm.base import LlmClient, LlmError
from app.llm.llama_client import LlamaCppClient
from app.llm.ollama_client import OllamaClient


def create_llm_client(config: LlmConfig) -> LlmClient:
    """Create the configured local provider client."""

    if config.provider == "llama_cpp":
        return LlamaCppClient(
            config,
            api_key=os.environ.get("LLAMA_CPP_API_KEY"),
        )
    if config.provider == "ollama":
        return OllamaClient(config)
    raise LlmError(f"Unsupported LLM provider: {config.provider}")
