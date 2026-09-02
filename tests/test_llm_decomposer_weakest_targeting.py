"""BLOOM-10 weakest-first decomposer targeting tests.

These exercise the deterministic parts of the LLM decomposer (prompt
construction and objective/target mapping) without requiring a provider key.
The `_completion` path is covered elsewhere and falls back gracefully.
"""

from agents.planner.decomposition.llm_decomposer_real import LLMDecomposerReal
from agents.planner.models.task_graph import AtomicTask, WeakCompetency


def _task(title, desc="", task_id=None):
    return AtomicTask(
        id=task_id or title.lower().replace(" ", "-"),
        title=title,
        description=desc,
        estimated_minutes=30,
        difficulty=0.5,
        prerequisites=[],
    )


def _wc(**kw):
    base = {
        "topic_id": "sorting",
        "topic_title": "Sorting Algorithms",
        "scores": {"remember": 0.9, "understand": 0.8, "apply": 0.4},
        "current_level": "apply",
        "unlocked_levels": ("remember", "understand", "apply"),
    }
    base.update(kw)
    return WeakCompetency(**base)


def test_no_weak_competencies_leaves_tasks_untagged():
    tasks = [_task("Sorting Algorithms Practice", "implement quicksort")]
    LLMDecomposerReal()._attach_weakest_targets(tasks, [])
    assert tasks[0].objective_id is None
    assert tasks[0].target_bloom_level is None


def test_weak_topic_task_gets_objective_and_target():
    tasks = [_task("Sorting Algorithms Practice", "implement quicksort")]
    LLMDecomposerReal()._attach_weakest_targets(tasks, [_wc()])
    assert tasks[0].objective_id == "sorting"
    # apply is the highest unlocked level below threshold (0.4 < 0.7)
    assert tasks[0].target_bloom_level == "apply"


def test_unrelated_task_not_tagged():
    tasks = [_task("Arrays review", "basic arrays")]
    LLMDecomposerReal()._attach_weakest_targets(tasks, [_wc()])
    assert tasks[0].objective_id is None
    assert tasks[0].target_bloom_level is None


def test_fully_mastered_unlocked_levels_no_target():
    wc = _wc(
        scores={"remember": 0.9, "understand": 0.85, "apply": 0.8},
        unlocked_levels=("remember", "understand", "apply"),
    )
    tasks = [_task("Sorting Algorithms deep", "advanced")]
    LLMDecomposerReal()._attach_weakest_targets(tasks, [wc])
    assert tasks[0].objective_id is None
    assert tasks[0].target_bloom_level is None


def test_locked_weak_level_does_not_override_gate():
    # analyze is weak (0.2) but locked (apply is 0.4 < 0.7) -> target stays apply
    wc = _wc(scores={"remember": 0.9, "understand": 0.8, "apply": 0.4, "analyze": 0.2})
    tasks = [_task("Sorting Algorithms x", "x")]
    LLMDecomposerReal()._attach_weakest_targets(tasks, [wc])
    assert tasks[0].target_bloom_level == "apply"


def test_prompt_includes_weak_competency_context_only_when_present():
    d = LLMDecomposerReal()
    sys_no, user_no = d._build_prompt("goal", ["concept"], 120, [])
    assert "weakest-first" not in user_no
    assert "WEAK_COMPETENCIES" not in user_no

    sys_yes, user_yes = d._build_prompt("goal", ["concept"], 120, [_wc()])
    assert "weakest-first" in user_yes
    assert "progression gate" in user_yes
    assert "Sorting Algorithms" in user_yes
