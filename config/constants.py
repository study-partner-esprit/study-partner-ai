"""Application constants."""

# Agent types
AGENT_PLANNER = "planner"
AGENT_COACH = "coach"
AGENT_EVALUATOR = "evaluator"
AGENT_META = "meta"

# Session states
SESSION_STATE_ACTIVE = "active"
SESSION_STATE_PAUSED = "paused"
SESSION_STATE_COMPLETED = "completed"

# Task priorities
PRIORITY_LOW = "low"
PRIORITY_MEDIUM = "medium"
PRIORITY_HIGH = "high"

# Decision types
DECISION_PLAN = "plan"
DECISION_COACH = "coach"
DECISION_EVALUATE = "evaluate"

# Signal types
SIGNAL_START_SESSION = "start_session"
SIGNAL_PAUSE_SESSION = "pause_session"
SIGNAL_COMPLETE_TASK = "complete_task"
SIGNAL_REQUEST_HELP = "request_help"
SIGNAL_PROGRESS_UPDATE = "progress_update"

# Thresholds
DIFFICULTY_THRESHOLD = 0.7
MOTIVATION_THRESHOLD = 0.5
PROGRESS_THRESHOLD = 0.8
