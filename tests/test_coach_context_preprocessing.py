"""COACH-04: context filtering — windowing, signal downsampling, PII redaction.

Covers `agents.coach.context.preprocess` and its wiring into the coach prompt:
- window_history: recent actions only (count + age caps)
- downsample_signals: decimated window, evenly spaced, oldest→newest
- redact_pii: emails / names excluded (placeholder), env-configurable
- prompt pipeline never sees stale history or raw PII
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from agents.coach.context.preprocess import (
    coach_context_config,
    downsample_signals,
    redact_pii,
    window_history,
)
from agents.coach.decision.llm_decider import decide_with_llm
from agents.coach.models.schemas import CoachAction, CoachInput, FatigueState, FocusState, ScheduledTask

NOW = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)


def _history(ts_list):
    return [
        {"ts": ts, "action_type": "nudge", "message": f"msg{i}"}
        for i, ts in enumerate(ts_list)
    ]


def _signals_at(minutes_ago, count):
    return [
        {
            "timestamp": (NOW - timedelta(minutes=m)).isoformat(),
            "state": "focused",
            "score": 0.9,
        }
        for m in range(minutes_ago, minutes_ago + count)
    ]


def _sig_ts(sig):
    return sig["timestamp"]


# ---------------------------------------------------------------- history #

def test_window_history_drops_stale_actions():
    recent = NOW - timedelta(minutes=5)
    stale = NOW - timedelta(minutes=300)
    items = _history([stale, recent])
    kept = window_history(items, limit=5, window_minutes=60, now=NOW)
    assert [k["ts"] for k in kept] == [recent]


def test_window_history_caps_at_limit_keeping_newest():
    items = _history([NOW - timedelta(minutes=i * 10) for i in range(10)])
    kept = window_history(items, limit=3, window_minutes=600, now=NOW)
    assert len(kept) == 3
    assert kept == items[:3]


def test_window_history_sorts_newest_first_regardless_of_order():
    items = _history([NOW - timedelta(minutes=20), NOW - timedelta(minutes=5)])
    kept = window_history(items, limit=5, window_minutes=60, now=NOW)
    assert [k["ts"] for k in kept] == [
        NOW - timedelta(minutes=5),
        NOW - timedelta(minutes=20),
    ]


def test_window_history_handles_iso_and_naive_ts():
    naive = NOW.replace(tzinfo=None)
    items = [
        {"ts": NOW.isoformat(), "action_type": "nudge", "message": "a"},
        {"ts": naive, "action_type": "encourage", "message": "b"},
    ]
    assert len(window_history(items, limit=5, window_minutes=60, now=NOW)) == 2


def test_window_history_empty_and_bad_ts():
    assert window_history([], limit=5, window_minutes=60, now=NOW) == []
    assert window_history(None, limit=5, window_minutes=60, now=NOW) == []
    kept = window_history(
        [{"ts": "not-a-date", "action_type": "nudge"}], limit=5, window_minutes=60, now=NOW
    )
    assert kept == []


# --------------------------------------------------------------- signals #

def test_downsample_signals_windows_by_latest():
    old = _signals_at(240, 3)
    fresh = _signals_at(5, 3)
    out = downsample_signals(old + fresh, max_points=10, window_minutes=60)
    assert sorted(out, key=_sig_ts) == out  # chronological
    assert set(_sig_ts(o) for o in out) == set(_sig_ts(o) for o in fresh)


def test_downsample_signals_keeps_all_when_under_cap():
    sigs = _signals_at(0, 4)
    assert len(downsample_signals(sigs, max_points=10, window_minutes=60)) == 4


def test_downsample_signals_decimates_evenly_spaced():
    sigs = _signals_at(0, 12)
    out = downsample_signals(sigs, max_points=6, window_minutes=600)
    assert len(out) == 6
    ts = [_sig_ts(o) for o in out]
    assert ts == sorted(ts)  # oldest → newest
    assert ts[0] == sigs[-1]["timestamp"]  # earliest point kept
    assert ts[-1] == sigs[0]["timestamp"]  # latest point kept


def test_downsample_signals_progressive_spacing_of_first_last_kept():
    sigs = _signals_at(0, 7)
    out = downsample_signals(sigs, max_points=3, window_minutes=600)
    assert len(out) == 3
    assert _sig_ts(out[0]) == sigs[-1]["timestamp"]  # oldest kept
    assert _sig_ts(out[-1]) == sigs[0]["timestamp"]  # newest kept


def test_downsample_signals_empty_and_single():
    assert downsample_signals([], max_points=6, window_minutes=60) == []
    assert downsample_signals(None, max_points=6, window_minutes=60) == []
    assert len(downsample_signals(_signals_at(0, 1), max_points=6, window_minutes=60)) == 1


# ------------------------------------------------------------------- PII #

def test_redact_pii_redacts_emails():
    out = redact_pii("Contact Prof. Smith at john.doe@example.com soon.")
    assert "@example.com" not in out
    assert "john.doe@" not in out
    assert "[EMAIL_REDACTED]" in out


def test_redact_pii_redacts_titled_names():
    out = redact_pii("Talk to Dr. Marie Curie about the lab.")
    assert "Marie Curie" not in out
    assert "[NAME_REDACTED]" in out
    assert "Dr." not in redact_pii("Mr. John Smith is my tutor.")


def test_redact_pii_redacts_configured_blocklist(monkeypatch):
    monkeypatch.setenv("COACH_REDACTED_NAMES", "Alice Dupont")
    out = redact_pii("My teacher Alice Dupont sent me a note.")
    assert "Alice Dupont" not in out
    assert "[NAME_REDACTED]" in out


def test_redact_pii_disabled_keeps_content():
    text = "Email sarah@example.com, Prof. X."
    assert redact_pii(text, redact_emails=False, redact_names=False) == text


def test_redact_pii_leaves_ordinary_text_intact():
    text = "I am reviewing integrals and derivatives for the exam."
    assert redact_pii(text) == text


def test_redact_pii_empty_and_none():
    assert redact_pii("") == ""
    assert redact_pii(None) is not None


# ---------------------------------------------------------------- config #

def test_config_defaults(monkeypatch):
    for var in [
        "COACH_HISTORY_LIMIT",
        "COACH_HISTORY_WINDOW_MINUTES",
        "COACH_SIGNAL_MAX_POINTS",
        "COACH_SIGNAL_WINDOW_MINUTES",
        "COACH_REDACT_EMAILS",
        "COACH_REDACT_NAMES",
        "COACH_REDACTED_NAMES",
    ]:
        monkeypatch.delenv(var, raising=False)
    cfg = coach_context_config()
    assert cfg["history_limit"] == 5
    assert cfg["history_window_minutes"] == 120
    assert cfg["signal_max_points"] == 6
    assert cfg["signal_window_minutes"] == 30
    assert cfg["redact_emails"] is True
    assert cfg["redact_names"] is True
    assert cfg["redacted_names"] == []


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("COACH_HISTORY_LIMIT", "3")
    monkeypatch.setenv("COACH_HISTORY_WINDOW_MINUTES", "45")
    monkeypatch.setenv("COACH_SIGNAL_MAX_POINTS", "8")
    monkeypatch.setenv("COACH_SIGNAL_WINDOW_MINUTES", "15")
    monkeypatch.setenv("COACH_REDACT_EMAILS", "false")
    monkeypatch.setenv("COACH_REDACT_NAMES", "no")
    monkeypatch.setenv("COACH_REDACTED_NAMES", "Alice Dupont,Bob Lee")
    cfg = coach_context_config()
    assert cfg["history_limit"] == 3
    assert cfg["history_window_minutes"] == 45
    assert cfg["signal_max_points"] == 8
    assert cfg["signal_window_minutes"] == 15
    assert cfg["redact_emails"] is False
    assert cfg["redact_names"] is False
    assert cfg["redacted_names"] == ["Alice Dupont", "Bob Lee"]


# ------------------------------------------------------- prompt wiring #

def make_coach_input() -> CoachInput:
    return CoachInput(
        scheduled_tasks=[
            ScheduledTask(
                task_id="t1",
                title="Solve exercises",
                start_time=NOW,
                end_time=NOW,
                priority=1,
            )
        ],
        current_time=NOW,
        focus_state=FocusState(state="Drifting", score=0.4),
        fatigue_state=FatigueState(state="Alert", score=0.2),
        affective_state="engaged",
        ignored_count=0,
        do_not_disturb=False,
        is_late=False,
    )


@patch("agents.coach.decision.llm_decider.call_gemini")
def test_decide_with_llm_excludes_stale_history(mock_call):
    mock_call.return_value = (
        '{"action_type": "encourage", "message": "ok", '
        '"reasoning": "test", "target_task_id": "t1"}'
    )
    now = datetime.now(timezone.utc)
    stale = now - timedelta(minutes=300)
    recent = now - timedelta(minutes=5)
    decide_with_llm(
        make_coach_input(),
        recent_history=_history([stale, recent]),
        trace_id="tr-1",
    )
    _, user = mock_call.call_args.args[:2]
    assert "msg0" not in user  # stale action's message gone
    assert "msg1" in user


@patch("agents.coach.decision.llm_decider.call_gemini")
def test_decide_with_llm_redacts_pii_from_prompt(mock_call):
    mock_call.return_value = (
        '{"action_type": "encourage", "message": "ok", '
        '"reasoning": "test", "target_task_id": "t1"}'
    )
    cfg = make_coach_input()
    cfg.scheduled_tasks[0].title = "Email Prof. Ada Lovelace at ada@example.com"
    cfg.current_task_key_concepts = ["Dr. Grace Hopper"]
    now = datetime.now(timezone.utc)
    decide_with_llm(cfg, recent_history=_history([now - timedelta(minutes=5)]))
    _, user = mock_call.call_args.args[:2]
    assert "ada@example.com" not in user
    assert "Ada Lovelace" not in user
    assert "Grace Hopper" not in user
    assert "[EMAIL_REDACTED]" in user
    assert "[NAME_REDACTED]" in user
    action = CoachAction(action_type="encourage", message="ok", reasoning="test", target_task_id="t1")
    assert isinstance(action, CoachAction)