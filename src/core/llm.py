"""LLM client factory — Groq primary, Google Gemini fallback.

Usage::

    from src.core.llm import get_llm

    llm = get_llm()           # returns Groq client
    llm = get_llm(force_gemini=True)  # returns Gemini client
"""

from __future__ import annotations

import logging
from enum import Enum

from langchain_core.language_models import BaseChatModel

from src.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    GROQ = "groq"
    GEMINI = "gemini"


def get_llm(
    *,
    force_gemini: bool = False,
    temperature: float = 0.1,
    streaming: bool = False,
) -> BaseChatModel:
    """Return the appropriate LangChain chat model.

    Args:
        force_gemini:  Skip Groq and use Gemini directly.
        temperature:   Sampling temperature (low = more deterministic).
        streaming:     Enable streaming output (Groq supports it natively).

    Returns:
        A LangChain-compatible BaseChatModel.
    """
    settings = get_settings()

    if not force_gemini and settings.groq_api_key:
        return _build_groq(settings, temperature=temperature, streaming=streaming)

    if settings.google_api_key:
        logger.info("Groq unavailable or forced — falling back to Gemini")
        return _build_gemini(settings, temperature=temperature)

    raise RuntimeError(
        "No LLM API key configured. Set GROQ_API_KEY or GOOGLE_API_KEY in .env"
    )


def get_groq_llm(temperature: float = 0.1, streaming: bool = False) -> BaseChatModel:
    """Convenience: always return the Groq model."""
    return get_llm(force_gemini=False, temperature=temperature, streaming=streaming)


def get_gemini_llm(temperature: float = 0.1) -> BaseChatModel:
    """Convenience: always return the Gemini fallback model."""
    return get_llm(force_gemini=True, temperature=temperature)


# ---------------------------------------------------------------------------
# Private builders
# ---------------------------------------------------------------------------


def _build_groq(settings, *, temperature: float, streaming: bool) -> BaseChatModel:
    from langchain_groq import ChatGroq  # lazy import to avoid startup error if not installed

    logger.debug("Initialising Groq LLM: %s", settings.groq_model)
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=temperature,
        streaming=streaming,
    )


def _build_gemini(settings, *, temperature: float) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI  # lazy import

    logger.debug("Initialising Gemini LLM: %s", settings.gemini_model)
    return ChatGoogleGenerativeAI(
        api_key=settings.google_api_key,
        model=settings.gemini_model,
        temperature=temperature,
    )
