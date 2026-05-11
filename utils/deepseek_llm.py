"""DeepSeek LLM wrapper for LangChain and LangGraph usage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from utils.config import AppSettings, get_settings


def create_deepseek_chat_model(settings: AppSettings | None = None) -> ChatOpenAI:
    settings = settings or get_settings()
    settings.validate_for_llm()

    return ChatOpenAI(
        model=settings.llm.model_id,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        timeout=settings.llm.timeout,
        temperature=settings.llm.temperature,
    )


def build_messages(prompt: str, system_prompt: str | None = None) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))
    return messages


def invoke_deepseek(
    prompt: str | Sequence[BaseMessage],
    *,
    system_prompt: str | None = None,
    settings: AppSettings | None = None,
    **kwargs: Any,
) -> str:
    model = create_deepseek_chat_model(settings)
    messages = build_messages(prompt, system_prompt) if isinstance(prompt, str) else list(prompt)
    response = model.invoke(messages, **kwargs)
    return str(response.content)
