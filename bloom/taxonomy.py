"""
Canonical Bloom's Taxonomy constants (BLOOM-01 — Shared Taxonomy Constants).

Node mirror: `study-partner-api/shared/bloom/taxonomy.js`. Values MUST stay
identical on both sides — parity is covered by contract tests against
docs/contracts/bloom-fixture.json.
"""

from typing import Optional

BLOOM_LEVELS = (
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
)

KNOWLEDGE_TYPES = (
    "factual",
    "conceptual",
    "procedural",
    "metacognitive",
)

VERB_MAP = {
    "remember": ("Define", "List"),
    "understand": ("Explain", "Summarize"),
    "apply": ("Solve", "Implement"),
    "analyze": ("Compare", "Diagnose"),
    "evaluate": ("Justify", "Critique"),
    "create": ("Design", "Compose"),
}

UNLOCK_THRESHOLD = 0.7


def next_level(level: str) -> Optional[str]:
    """Return the next Bloom level in progression order, or None if `level`
    is already the highest level ('create') or not recognized."""
    try:
        idx = BLOOM_LEVELS.index(level)
    except ValueError:
        return None
    if idx == len(BLOOM_LEVELS) - 1:
        return None
    return BLOOM_LEVELS[idx + 1]


def is_level_unlocked(scores, level: str, unlock_threshold: float = UNLOCK_THRESHOLD) -> bool:
    """Progression gate (BLOOM-10): level N is unlocked only when level N-1
    is >= `unlock_threshold` (0.7).

    `scores` maps a bloom level -> 0..1 score. 'remember' is always unlocked.
    A missing predecessor score is treated as 0 (not unlocked).
    """
    if level not in BLOOM_LEVELS:
        return False
    idx = BLOOM_LEVELS.index(level)
    if idx == 0:
        return True
    prev_level = BLOOM_LEVELS[idx - 1]
    prev_score = scores.get(prev_level)
    return prev_score is not None and prev_score >= unlock_threshold


def unlocked_levels(scores, unlock_threshold: float = UNLOCK_THRESHOLD) -> tuple:
    """Return every level currently unlocked for a given competency score map."""
    return tuple(lvl for lvl in BLOOM_LEVELS if is_level_unlocked(scores, lvl, unlock_threshold))
