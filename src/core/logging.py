"""Structured logging setup for Pitwall-AI."""

import logging
import sys
from typing import Any

from src.core.config import get_settings


# ---------------------------------------------------------------------------
# Patch Logger.makeRecord at import time so it never crashes on reserved keys.
# LangChain/LangGraph thread executors inject context (including "message") into
# log records; the default Python implementation raises KeyError on those keys.
# ---------------------------------------------------------------------------

def _safe_make_record(self, name, level, fn, lno, msg, args, exc_info,
                      func=None, extra=None, sinfo=None):
    rv = logging.LogRecord(name, level, fn, lno, msg, args, exc_info, func, sinfo)
    if extra:
        reserved = {"message", "asctime"} | set(rv.__dict__.keys())
        for key, value in extra.items():
            if key not in reserved:
                rv.__dict__[key] = value
    return rv

logging.Logger.makeRecord = _safe_make_record  # applied at import time


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for the given module name."""
    return logging.getLogger(name)


def configure_logging() -> None:
    """Configure root logging once at application startup."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = _StructuredFormatter()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid adding duplicate handlers when called multiple times (e.g. in tests)
    if not root.handlers:
        root.addHandler(handler)
    else:
        root.handlers.clear()
        root.addHandler(handler)


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class _StructuredFormatter(logging.Formatter):
    """JSON-ish single-line structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"[{self.formatTime(record, '%Y-%m-%dT%H:%M:%S')}] "
            f"{record.levelname:8s} "
            f"{record.name} — "
            f"{record.getMessage()}"
        )
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        extra = _extract_extra(record)
        if extra:
            kv = " | ".join(f"{k}={v}" for k, v in extra.items())
            base += f" | {kv}"
        return base


_STDLIB_ATTRS: frozenset[str] = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "message",
        "taskName",
    }
)


def _extract_extra(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in record.__dict__.items() if k not in _STDLIB_ATTRS}
