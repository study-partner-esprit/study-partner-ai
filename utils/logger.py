"""
Structured logging utility for the Study Partner AI service.

Replaces all print() calls with structured JSON logs compatible with
Datadog, Loki, CloudWatch, or any JSON-aware log aggregator.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("goal_decomposed", extra={"task_count": 5, "user_id": "u1"})
"""

import logging
import json
import sys
from datetime import datetime, timezone

_RESERVED_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }
)


class JSONFormatter(logging.Formatter):
    """
    Emits one JSON object per log line.
    Standard fields: ts, level, logger, msg
    Any key passed via extra={} is merged at the top level.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Include file location for warnings and above
        if record.levelno >= logging.WARNING:
            log_obj["loc"] = f"{record.module}:{record.lineno}"

        # Merge extra fields (e.g. trace_id, user_id, etc.)
        for key, val in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                log_obj[key] = val

        # Include exception info if present
        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with the JSON formatter attached.
    Idempotent — safe to call multiple times with the same name.

    Args:
        name: Logger name (use __name__ as convention).
        level: Minimum log level (default: INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False  # prevent double-logging via root logger

    return logger
