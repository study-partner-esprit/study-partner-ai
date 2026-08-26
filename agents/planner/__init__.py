"""Planner agent package.

Kept intentionally light: `PlannerAgent` pulls in embedding models and other
heavy optional dependencies. Import `agents.planner.agent` explicitly (or use
workers' lazy construction) so lightweight consumers like
`agents.planner.models.task_graph` stay dependency-free.
"""
