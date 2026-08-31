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
