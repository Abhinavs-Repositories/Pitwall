"""Router Agent — classifies the user query and extracts driver references.

Uses the Groq LLM (fast, low-cost) for classification.
Returns a partial AgentState update with query_type and target_drivers.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.agents.state import AgentState
from src.core.llm import get_llm

logger = logging.getLogger(__name__)

# Canonical query types
QUERY_TYPES = frozenset(
    ["race_status", "tire_analysis", "weather", "strategy", "comparison", "historical"]
)

# Well-known driver number → name mappings for mention extraction
_DRIVER_NAMES: dict[str, int] = {
    "verstappen": 1, "max": 1, "ver": 1,
    "hamilton": 44, "lewis": 44, "ham": 44,
    "leclerc": 16, "charles": 16, "lec": 16,
    "norris": 4, "lando": 4, "nor": 4,
    "sainz": 55, "carlos": 55, "sai": 55,
    "russell": 63, "george": 63, "rus": 63,
    "piastri": 81, "oscar": 81, "pia": 81,
    "perez": 11, "checo": 11, "per": 11,
    "alonso": 14, "fernando": 14, "alo": 14,
    "stroll": 18, "lance": 18, "str": 18,
    "albon": 23, "alex": 23, "alb": 23,
    "ocon": 31, "esteban": 31, "oco": 31,
    "gasly": 10, "pierre": 10, "gas": 10,
    "tsunoda": 22, "yuki": 22, "tsu": 22,
    "bottas": 77, "valtteri": 77, "bot": 77,
    "zhou": 24, "guanyu": 24,
    "magnussen": 20, "kevin": 20, "mag": 20,
    "hulkenberg": 27, "nico": 27, "hul": 27,
    "sargeant": 2, "logan": 2, "sar": 2,
}

_CLASSIFICATION_PROMPT = """You are classifying an F1 strategy question into exactly one category.

Categories:
- race_status: current positions, gaps, standings, lap count
- tire_analysis: tire age, degradation, compound performance, cliff prediction
- weather: weather conditions, rain threat, temperature
- strategy: pit stop recommendations, when to pit, undercut/overcut, compound choice
- comparison: comparing two or more drivers (pace, tires, strategy)
- historical: past race results, historical strategies, track records

User message: "{message}"

Respond with ONLY the category name, nothing else."""


def router_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: classify query and extract driver mentions."""
    message = state.user_message.strip()

    # Extract driver numbers from the message text
    target_drivers = _extract_drivers(message)

    # LLM classification
    query_type = _classify(message)

    logger.info(
        "Router classified query",
        extra={
            "query_type": query_type,
            "target_drivers": target_drivers,
            "message": message[:80],
        },
    )

    return {
        "query_type": query_type,
        "target_drivers": target_drivers,
        "agents_used": ["router"],
    }


def _classify(message: str) -> str:
    """Use LLM to classify the query. Falls back to rule-based on LLM failure."""
    try:
        llm = get_llm()
        prompt = _CLASSIFICATION_PROMPT.format(message=message)
        response = llm.invoke(prompt)
        result = str(response.content).strip().lower()
        if result in QUERY_TYPES:
            return result
    except Exception as exc:
        logger.warning("LLM classification failed, using rule-based fallback: %s", exc)

    return _rule_based_classify(message)


def _rule_based_classify(message: str) -> str:
    """Fast rule-based fallback classifier."""
    lower = message.lower()

    if any(kw in lower for kw in ("pit", "stop", "undercut", "overcut", "compound", "strategy", "recommend")):
        return "strategy"
    if any(kw in lower for kw in ("tire", "tyre", "degrad", "cliff", "wear", "compound", "stint")):
        return "tire_analysis"
    if any(kw in lower for kw in ("weather", "rain", "temperature", "wet", "dry", "humidity")):
        return "weather"
    if any(kw in lower for kw in ("compare", "vs", "versus", "difference between", "better than")):
        return "comparison"
    if any(kw in lower for kw in ("last year", "2023", "2024", "historical", "previous", "won", "winner")):
        return "historical"

    return "race_status"


def _extract_drivers(message: str) -> list[int]:
    """Extract referenced driver numbers from the user's message."""
    found: set[int] = set()
    lower = message.lower()

    for name, number in _DRIVER_NAMES.items():
        # Match whole word
        if re.search(rf"\b{re.escape(name)}\b", lower):
            found.add(number)

    # Also match explicit driver numbers like "#1" or "driver 1"
    for match in re.finditer(r"\b(?:driver\s+)?(\d{1,2})\b", lower):
        num = int(match.group(1))
        if 1 <= num <= 99:
            found.add(num)

    return sorted(found)
