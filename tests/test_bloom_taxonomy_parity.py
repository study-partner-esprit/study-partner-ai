"""
Bloom taxonomy parity tests (BLOOM-01): the Python taxonomy module must
match docs/contracts/bloom-fixture.json — the same fixture the Node side
(study-partner-api/shared/bloom/taxonomy.js) validates against. Drift fails
CI.

Note: bloom/taxonomy.py intentionally uses tuples (immutable, mirroring the
Object.freeze()'d arrays on the Node side), while the JSON fixture parses
into lists. Comparisons below convert to lists so the check is about
content, not container type.
"""

import json
from pathlib import Path

from bloom.taxonomy import (
    BLOOM_LEVELS,
    KNOWLEDGE_TYPES,
    UNLOCK_THRESHOLD,
    VERB_MAP,
    next_level,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "contracts" / "bloom-fixture.json"
)


def _load_fixture():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_bloom_levels_match_fixture():
    fixture = _load_fixture()
    assert list(BLOOM_LEVELS) == fixture["bloomLevels"]


def test_knowledge_types_match_fixture():
    fixture = _load_fixture()
    assert list(KNOWLEDGE_TYPES) == fixture["knowledgeTypes"]


def test_verb_map_matches_fixture():
    fixture = _load_fixture()
    verb_map_as_lists = {level: list(verbs) for level, verbs in VERB_MAP.items()}
    assert verb_map_as_lists == fixture["verbMap"]


def test_unlock_threshold_matches_fixture():
    fixture = _load_fixture()
    assert UNLOCK_THRESHOLD == fixture["unlockThreshold"]


def test_next_level_progression():
    fixture = _load_fixture()
    levels = fixture["bloomLevels"]
    for i in range(len(levels) - 1):
        assert next_level(levels[i]) == levels[i + 1]
    assert next_level(levels[-1]) is None
    assert next_level("not-a-level") is None
