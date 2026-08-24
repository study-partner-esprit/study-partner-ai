"""Failure classification for AI jobs (AI-COM-06).

Python mirror of `classifyFailure` in
`study-partner-api/shared/ai-messaging/topology.js` — the pattern lists must
match so both sides route failures identically.
"""

from __future__ import annotations

from enum import Enum


class FailureClass(str, Enum):
    RETRYABLE = "retryable"
    TERMINAL = "terminal"


class RetryableError(Exception):
    """Raised by handlers to force a retry regardless of message content."""


class TerminalError(Exception):
    """Raised by handlers to skip retries and dead-letter immediately."""


_TERMINAL_PATTERNS = [
    "validation",
    "invalid",
    "schema",
    "unauthorized",
    "forbidden",
    "notfound",
    "not found",
    "rejected",  # output rejected by validation pipeline
    "parseerror",
    "payload",
]

_RETRYABLE_PATTERNS = [
    "timeout",
    "timed out",
    "etimedout",
    "econnrefused",
    "econnreset",
    "econnaborted",
    "enotfound",
    "ehostunreach",
    "enetunreach",
    "socket hang up",
    "rate limit",
    "quota",
    "429",
    "502",
    "503",
    "504",
    "temporarily unavailable",
    "connection closed",
]


def classify_failure(err: BaseException | str) -> FailureClass:
    """Classify a failure as retryable or terminal.

    Unknown failures default to retryable: transient blips recover, and
    persistent unknowns exhaust retries and land in the DLQ anyway.
    """
    name = getattr(err, "name", "") or ""
    code = getattr(err, "code", "") or ""
    message = err if isinstance(err, str) else (getattr(err, "message", "") or str(err))
    combined = f"{name} {code} {message}".lower()

    if any(p in combined for p in _TERMINAL_PATTERNS):
        return FailureClass.TERMINAL
    if any(p in combined for p in _RETRYABLE_PATTERNS):
        return FailureClass.RETRYABLE
    return FailureClass.RETRYABLE


def sanitized_error(err: BaseException | str) -> str:
    """One-line error safe to expose to clients / result events (audit §7.5)."""
    message = err if isinstance(err, str) else (getattr(err, "message", "") or str(err))
    flat = " ".join(str(message).split())
    return flat[:512]
