"""
EMA (Exponential Moving Average) smoothing for real-time focus/fatigue signals.

Maintains a lightweight in-process state (dict) per user so that noisy
per-frame ML predictions are smoothed before being stored or acted upon.

The EMA formula:
    S_t = α · x_t + (1 − α) · S_{t-1}

where α controls responsiveness (higher = more reactive).

Usage:
    from services.signal_processing_service.smoothing import get_ema_state
    ema = get_ema_state()
    result = ema.update("user_42", focus_score=0.72, fatigue_score=0.31)
    print(result.smooth_focus, result.smooth_fatigue)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional

# --- Tunable constants ------------------------------------------------------ #

FOCUS_ALPHA = 0.3  # Smoothing factor for focus   (lower = smoother)
FATIGUE_ALPHA = 0.2  # Smoothing factor for fatigue (lower = smoother)


# ---------------------------------------------------------------------------- #
# Public dataclass returned by update()                                         #
# ---------------------------------------------------------------------------- #


@dataclass
class EMAResult:
    user_id: str
    smooth_focus: float
    smooth_fatigue: float
    raw_focus: float
    raw_fatigue: float


# ---------------------------------------------------------------------------- #
# Internal per-user state                                                       #
# ---------------------------------------------------------------------------- #


@dataclass
class _UserEMA:
    smooth_focus: float = 0.5  # Initialise to neutral
    smooth_fatigue: float = 0.2  # Initialise to low fatigue
    n_updates: int = 0  # Number of updates (useful for warm-up logic)


class EMAState:
    """
    Thread-safe in-process EMA state store.

    Process-level singleton (use get_ema_state()).
    """

    def __init__(
        self,
        focus_alpha: float = FOCUS_ALPHA,
        fatigue_alpha: float = FATIGUE_ALPHA,
    ) -> None:
        self._focus_alpha = focus_alpha
        self._fatigue_alpha = fatigue_alpha
        self._states: Dict[str, _UserEMA] = {}
        self._lock = threading.Lock()

    def update(
        self,
        user_id: str,
        focus_score: float,
        fatigue_score: float,
    ) -> EMAResult:
        """
        Apply one EMA step for *user_id* and return the smoothed result.

        Args:
            user_id:      Unique user key.
            focus_score:  Raw focus score in [0, 1].
            fatigue_score: Raw fatigue score in [0, 1].

        Returns:
            EMAResult with both raw and smoothed values.
        """
        with self._lock:
            state = self._states.get(user_id)
            if state is None:
                # Cold-start: initialise with the first observation
                state = _UserEMA(
                    smooth_focus=focus_score,
                    smooth_fatigue=fatigue_score,
                    n_updates=0,
                )
                self._states[user_id] = state

            state.smooth_focus = (
                self._focus_alpha * focus_score
                + (1.0 - self._focus_alpha) * state.smooth_focus
            )
            state.smooth_fatigue = (
                self._fatigue_alpha * fatigue_score
                + (1.0 - self._fatigue_alpha) * state.smooth_fatigue
            )
            state.n_updates += 1

        return EMAResult(
            user_id=user_id,
            smooth_focus=state.smooth_focus,
            smooth_fatigue=state.smooth_fatigue,
            raw_focus=focus_score,
            raw_fatigue=fatigue_score,
        )

    def get(self, user_id: str) -> Optional[_UserEMA]:
        """Return the current EMA state for *user_id*, or None."""
        with self._lock:
            return self._states.get(user_id)

    def reset(self, user_id: str) -> None:
        """Clear EMA state for *user_id* (e.g. after a long break)."""
        with self._lock:
            self._states.pop(user_id, None)


@lru_cache(maxsize=1)
def get_ema_state() -> EMAState:
    """Process-level singleton."""
    return EMAState()
