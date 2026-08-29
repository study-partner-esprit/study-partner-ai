"""Coach context preprocessing (F03 / COACH-04).

Only recent, relevant context reaches the LLM — cheaper, faster, and less
exposed to stale or sensitive content:

- `window_history` — last N coach actions within the last M minutes
- `downsample_signals` — decimate a timestamped signal series to ≤ K evenly
  spaced points inside a time window (the series consumer lands with COACH-13)
- `redact_pii` — emails and person names excluded before prompting

Every window is configurable via env vars so behaviour can be tuned without
redeploys:

    COACH_HISTORY_LIMIT          (default 5)
    COACH_HISTORY_WINDOW_MINUTES (default 120)
    COACH_SIGNAL_MAX_POINTS      (default 6)
    COACH_SIGNAL_WINDOW_MINUTES  (default 30)
    COACH_REDACT_EMAILS          (default true)
    COACH_REDACT_NAMES           (default true)
    COACH_REDACTED_NAMES         (comma-separated explicit blocklist)
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone

DEFAULT_HISTORY_LIMIT = 5
DEFAULT_HISTORY_WINDOW_MINUTES = 120
DEFAULT_SIGNAL_MAX_POINTS = 6
DEFAULT_SIGNAL_WINDOW_MINUTES = 30

_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# High-precision person-name pattern: academic/professional title followed by
# capitalized words (Prof. Doe, Dr. Marie Curie, Mr. John Smith).
_TITLE_NAME_PATTERN = re.compile(
    r"\b(?:Prof\.|Professor|Dr\.|Docteur|M\.|Mme|Madame|Monsieur|"
    r"Mr\.|Mrs\.|Ms\.)\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*"
)

_EMAIL_PLACEHOLDER = "[EMAIL_REDACTED]"
_NAME_PLACEHOLDER = "[NAME_REDACTED]"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_name_list(name: str) -> list:
    return [n.strip() for n in os.getenv(name, "").split(",") if n.strip()]


def coach_context_config() -> dict:
    """Resolve all COACH-04 window/PII settings from the environment."""
    return {
        "history_limit": _env_int("COACH_HISTORY_LIMIT", DEFAULT_HISTORY_LIMIT),
        "history_window_minutes": _env_int(
            "COACH_HISTORY_WINDOW_MINUTES", DEFAULT_HISTORY_WINDOW_MINUTES
        ),
        "signal_max_points": _env_int("COACH_SIGNAL_MAX_POINTS", DEFAULT_SIGNAL_MAX_POINTS),
        "signal_window_minutes": _env_int(
            "COACH_SIGNAL_WINDOW_MINUTES", DEFAULT_SIGNAL_WINDOW_MINUTES
        ),
        "redact_emails": _env_bool("COACH_REDACT_EMAILS", True),
        "redact_names": _env_bool("COACH_REDACT_NAMES", True),
        "redacted_names": _env_name_list("COACH_REDACTED_NAMES"),
    }


# ---------------------------------------------------------------- datetime #

def _to_aware_dt(value):
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ----------------------------------------------------------------- history #

def window_history(
    history,
    limit=None,
    window_minutes=None,
    now=None,
) -> list:
    """Keep only the most recent coach actions, capped by count and age.

    Args:
        history:        List of dicts with a `ts` (datetime or ISO string).
        limit:          Max items to keep (default from config).
        window_minutes: Keep only items within this many minutes of `now`
                        (default from config).
        now:            Reference "now" for tests; defaults to wall clock.

    Returns:
        Items sorted newest-first, truncated to `limit`.
    """
    cfg = coach_context_config()
    limit = int(limit) if limit is not None else cfg["history_limit"]
    window_minutes = (
        int(window_minutes)
        if window_minutes is not None
        else cfg["history_window_minutes"]
    )
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(minutes=window_minutes)

    with_ts = []
    for item in history or []:
        ts = _to_aware_dt(item.get("ts"))
        if ts is not None and ts >= cutoff:
            with_ts.append((ts, item))
    with_ts.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in with_ts[:limit]]


# ---------------------------------------------------------------- signals #

def downsample_signals(
    signals,
    max_points=None,
    window_minutes=None,
) -> list:
    """Decimate a timestamped signal series to ≤ `max_points` evenly spaced
    points inside the window ending at the latest signal.

    Args:
        signals:        List of dicts with a `timestamp` (datetime or ISO
                        string) — mirrors CoachRequest.signals / SignalSnapshot.
        max_points:     Max points to keep (default from config).
        window_minutes: Only signals within this many minutes of the latest
                        point (default from config).

    Returns:
        Up to `max_points` signals, sorted oldest → newest. The series
        consumer (COACH-13) will feed the result into the coach context.
    """
    cfg = coach_context_config()
    max_points = (
        int(max_points)
        if max_points is not None
        else cfg["signal_max_points"]
    )
    window_minutes = (
        int(window_minutes)
        if window_minutes is not None
        else cfg["signal_window_minutes"]
    )
    if not signals:
        return []

    with_ts = []
    for s in signals:
        ts = _to_aware_dt(
            s.get("timestamp") if s.get("timestamp") is not None else s.get("ts")
        )
        if ts is not None:
            with_ts.append((ts, s))
    if not with_ts:
        return list(signals[: max(0, max_points)])

    with_ts.sort(key=lambda pair: pair[0])
    latest = with_ts[-1][0]
    cutoff = latest - timedelta(minutes=window_minutes)
    windowed = [s for ts, s in with_ts if ts >= cutoff]

    if len(windowed) <= max_points:
        return windowed
    if max_points <= 1:
        return windowed[-max_points:] if max_points > 0 else []

    step = (len(windowed) - 1) / max(1, max_points - 1)
    indices = sorted({round(i * step) for i in range(max_points)})
    return [windowed[i] for i in indices]


# ------------------------------------------------------------------- PII #

def redact_pii(text, redact_emails=None, redact_names=None) -> str:
    """Exclude emails and person names from untrusted text.

    Args:
        text:            Raw user-generated string.
        redact_emails:   Override for the env toggle (default from config).
        redact_names:    Override for the env toggle (default from config).

    Returns:
        The text with sensitive values replaced by placeholders.
    """
    cfg = coach_context_config()
    if redact_emails is None:
        redact_emails = cfg["redact_emails"]
    if redact_names is None:
        redact_names = cfg["redact_names"]
    if not text:
        return "" if text == "" else str(text)

    out = str(text)
    if redact_emails:
        out = _EMAIL_PATTERN.sub(_EMAIL_PLACEHOLDER, out)
    if redact_names:
        for name in cfg["redacted_names"]:
            out = re.sub(
                rf"\b{re.escape(name)}\b", _NAME_PLACEHOLDER, out, flags=re.IGNORECASE
            )
        out = _TITLE_NAME_PATTERN.sub(_NAME_PLACEHOLDER, out)
    return out