"""
Unit tests for EMAState (focus/fatigue smoothing).

Run with:
    pytest services/signal_processing_service/tests/test_smoothing.py -v
"""

import pytest
from services.signal_processing_service.smoothing import (
    EMAState,
    FOCUS_ALPHA,
    FATIGUE_ALPHA,
)


class TestEMAState:

    def _make_ema(self, fa=FOCUS_ALPHA, fta=FATIGUE_ALPHA):
        return EMAState(focus_alpha=fa, fatigue_alpha=fta)

    def test_first_update_initialises_with_observed_value(self):
        ema = self._make_ema()
        result = ema.update("u1", focus_score=0.8, fatigue_score=0.2)
        # Cold start: smooth == raw
        assert result.smooth_focus == pytest.approx(0.8)
        assert result.smooth_fatigue == pytest.approx(0.2)

    def test_subsequent_update_smooths(self):
        ema = self._make_ema(fa=0.5, fta=0.5)
        ema.update("u1", 0.8, 0.2)
        result = ema.update("u1", 0.4, 0.6)
        # S = 0.5*(0.4) + 0.5*(0.8) = 0.6
        assert result.smooth_focus == pytest.approx(0.6)
        assert result.smooth_fatigue == pytest.approx(0.4)

    def test_raw_values_preserved(self):
        ema = self._make_ema()
        result = ema.update("u1", 0.7, 0.3)
        assert result.raw_focus == pytest.approx(0.7)
        assert result.raw_fatigue == pytest.approx(0.3)

    def test_independent_users(self):
        ema = self._make_ema()
        ema.update("u1", 0.9, 0.1)
        r2 = ema.update("u2", 0.1, 0.9)
        # u2 should have its own cold-start
        assert r2.smooth_focus == pytest.approx(0.1)
        assert r2.smooth_fatigue == pytest.approx(0.9)

    def test_get_returns_state(self):
        ema = self._make_ema()
        ema.update("u1", 0.6, 0.4)
        state = ema.get("u1")
        assert state is not None
        assert state.smooth_focus == pytest.approx(0.6)

    def test_get_missing_user_returns_none(self):
        ema = self._make_ema()
        assert ema.get("noone") is None

    def test_reset_clears_state(self):
        ema = self._make_ema()
        ema.update("u1", 0.8, 0.2)
        ema.reset("u1")
        assert ema.get("u1") is None

    def test_reset_nonexistent_user_is_safe(self):
        ema = self._make_ema()
        ema.reset("nobody")  # should not raise

    def test_n_updates_increments(self):
        ema = self._make_ema()
        for i in range(5):
            ema.update("u1", 0.5, 0.5)
        assert ema.get("u1").n_updates == 5
