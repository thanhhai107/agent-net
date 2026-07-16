"""LLM compatibility used only by local training extensions.

NIKA's original ``agent.llm.model_factory`` remains untouched. This adapter
accepts the older ``llm_backend`` name and maps ``CUSTOM_API_URL`` to an
OpenAI-compatible endpoint, including the Netmind gateway.
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from agent.extensions.config import default_llm_provider, default_model


def _custom_api_url() -> str | None:
    value = os.getenv("CUSTOM_API_URL") or os.getenv("CUSTOM_API_BASE")
    return value.rstrip("/") if value else None


def load_extension_model(
    llm_provider: str | None = None,
    model: str | None = None,
    *,
    llm_backend: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> BaseChatModel:
    provider = (llm_provider or llm_backend or default_llm_provider()).lower()
    selected_model = model or default_model()

    if provider == "ollama":
        return ChatOllama(
            model=selected_model,
            temperature=0,
            validate_model_on_init=True,
            base_url=os.getenv("OLLAMA_API_URL"),
        )
    if provider == "openai":
        return ChatOpenAI(
            model=selected_model,
            temperature=0,
            timeout=timeout,
            max_retries=max_retries,
        )
    if provider == "deepseek":
        return ChatDeepSeek(
            model=selected_model,
            base_url="https://api.deepseek.com",
            temperature=0,
            timeout=timeout,
            max_retries=max_retries,
        )
    if provider == "custom":
        base_url = _custom_api_url()
        if not base_url:
            raise ValueError(
                "CUSTOM_API_URL (or upstream-compatible CUSTOM_API_BASE) is required for the custom provider."
            )
        return ChatOpenAI(
            model=selected_model,
            base_url=base_url,
            api_key=os.getenv("CUSTOM_API_KEY") or "dummy",
            temperature=0,
            timeout=timeout,
            max_retries=max_retries,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")
