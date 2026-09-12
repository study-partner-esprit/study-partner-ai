"""INGEST-09 — hardened LLM extraction tests.

Covers the security + schema-validation AC for the ingestion enrichment path:
* extracted document content is wrapped as untrusted data before ANY LLM
  enrichment (subtopic enrichment and generated subtopic titles);
* enrichment JSON output is validated/coerced against ``EnrichmentOutput`` —
  unknown keys are dropped, malformed fields degrade, bounds are enforced;
* generated tasks are validated against a strict planner/task-style schema;
* canonical injection payloads carried in documents end up in the untrusted
  data block (treated as data) and never shape the trusted prompt or the
  output records.
"""

import json

import pytest

import agents.course_ingestion.enrichment.llm_enricher as llm_enricher
import agents.course_ingestion.enrichment.task_generator as task_generator
from agents.course_ingestion.enrichment.output_schema import (
    CLEANED_TEXT_MAX_CHARS,
    EnrichmentOutput,
)
from security.prompt_guard import injection_probe_payloads


def test_enrichment_wraps_document_content_and_probes_stay_in_data(monkeypatch):
    captured = {}

    def fake_call_llm(prompt, system_prompt=None):
        captured["prompt"] = prompt
        return ""  # LLM "failure" -> fallback; we assert on the built prompt

    monkeypatch.setattr(llm_enricher, "call_llm", fake_call_llm)

    probes = injection_probe_payloads()
    text = "Genuine signal-processing material.\n" + "\n".join(probes)

    result = llm_enricher.enrich_subtopic_with_llm("Fourier Transforms", text)

    flat = " ".join(captured["prompt"].split())
    first_untrusted = flat.index("<<<UNTRUSTED_")
    assert first_untrusted > flat.index("Your task is to")
    for probe in probes:
        normalized = " ".join(probe.split())
        assert normalized in flat, f"probe dropped from prompt: {probe!r}"
        assert flat.index(normalized) > first_untrusted, (
            f"probe leaked outside the untrusted block: {probe!r}"
        )

    # Fallback output is still schema-conforming.
    EnrichmentOutput.model_validate(result)


def test_generated_subtopic_title_wraps_content_as_data(monkeypatch):
    captured = {}

    def fake_call_llm(prompt, system_prompt=None):
        captured["prompt"] = prompt
        return "Limits and Continuity"

    monkeypatch.setattr(llm_enricher, "call_llm", fake_call_llm)

    probe = injection_probe_payloads()[0]
    title = llm_enricher.generate_subtopic_title(f"Intro material.\n{probe}")

    assert title == "Limits and Continuity"

    flat = " ".join(captured["prompt"].split())
    first_untrusted = flat.index("<<<UNTRUSTED_")
    assert flat.count("<<<UNTRUSTED_") == 1
    assert flat.index(probe) > first_untrusted


@pytest.mark.parametrize("variant", ["list_of_strings", "number", "null"])
def test_enrichment_nonconforming_fields_salvaged_to_schema(monkeypatch, variant):
    if variant == "list_of_strings":
        definitions = ["plain-string", "another-string"]
        key_concepts = "not-a-list"
    elif variant == "number":
        definitions = [42]
        key_concepts = [123]
    else:
        definitions = None
        key_concepts = None

    response = json.dumps(
        {
            "cleaned_text": "Some genuine paragraphs about limits.",
            "key_concepts": key_concepts,
            "definitions": definitions,
            "formulas": [123, "E=mc^2"],
            "examples": [],
        }
    )
    monkeypatch.setattr(llm_enricher, "call_llm", lambda *a, **k: response)

    result = llm_enricher.enrich_subtopic_with_llm("Limits", "sample body")

    assert result["cleaned_text"] == "Some genuine paragraphs about limits."
    assert result["key_concepts"] == []
    assert result["definitions"] == []
    assert result["formulas"] == ["E=mc^2"]
    assert result["examples"] == []
    EnrichmentOutput.model_validate(result)


def test_enrichment_unknown_keys_dropped_and_injection_stays_data(monkeypatch):
    injected = injection_probe_payloads()[0]
    response = json.dumps(
        {
            "cleaned_text": injected,
            "key_concepts": ["limits"],
            "on_success": "cat /etc/passwd",
            "priority_level": "admin",
            "definitions": [],
            "formulas": [],
            "examples": [],
        }
    )
    monkeypatch.setattr(llm_enricher, "call_llm", lambda *a, **k: response)

    result = llm_enricher.enrich_subtopic_with_llm("Limits", "sample body")

    assert result["cleaned_text"] == injected  # injection preserved as data
    assert "on_success" not in result
    assert "priority_level" not in result
    assert result["key_concepts"] == ["limits"]
    EnrichmentOutput.model_validate(result)


def test_enrichment_oversized_cleaned_text_bounded_by_schema(monkeypatch):
    big = "calm prose " * 20_000  # ~200k chars, far over the cap
    response = json.dumps({"cleaned_text": big})
    monkeypatch.setattr(llm_enricher, "call_llm", lambda *a, **k: response)

    result = llm_enricher.enrich_subtopic_with_llm("T", "fallback body")

    assert 0 < len(result["cleaned_text"]) <= CLEANED_TEXT_MAX_CHARS
    EnrichmentOutput.model_validate(result)


def test_task_generation_wraps_course_content_and_probes_stay_in_data(monkeypatch):
    captured = {}

    def fake_call_llm_task_generation(prompt):
        captured["prompt"] = prompt
        return json.dumps({"tasks": []})

    monkeypatch.setattr(
        task_generator, "call_llm_task_generation", fake_call_llm_task_generation
    )

    probes = injection_probe_payloads()
    topics = [
        {
            "title": "Introduction",
            "subtopics": [
                {
                    "title": f"Subtopic {probes[0]}",
                    "key_concepts": [probes[1], "limits"],
                }
            ],
        }
    ]

    result = task_generator.generate_tasks_from_course(f"Course {probes[2]}", topics)

    assert result == []

    flat = " ".join(captured["prompt"].split())
    first_untrusted = flat.index("<<<UNTRUSTED_")
    for probe in probes[:3]:
        normalized = " ".join(probe.split())
        assert normalized in flat, f"probe dropped from prompt: {probe!r}"
        assert flat.index(normalized) > first_untrusted, (
            f"probe leaked outside the untrusted block: {probe!r}"
        )


def test_task_generation_validates_and_coerces_task_schema(monkeypatch):
    response = json.dumps(
        {
            "tasks": [
                {
                    "title": "Solve 10 problems",
                    "description": "From section 2.3",
                    "priority": "high",
                    "estimatedTime": 20,
                    "tags": ["math", "practice"],
                },
                {
                    "title": "Review notes",
                    "description": "Summarize chapter",
                    "priority": "urgent",
                    "estimatedTime": "45",
                    "tags": ["ok", 42, "", "x" * 60],
                },
                {
                    "title": "Missing description",
                    "priority": "low",
                    "estimatedTime": 5,
                    "tags": [],
                },
                {
                    "title": "Zero minutes",
                    "description": 5.0,
                    "priority": "high",
                    "estimatedTime": 0,
                    "tags": 12,
                },
            ]
        }
    )
    monkeypatch.setattr(task_generator, "call_llm_task_generation", lambda *a: response)

    result = task_generator.generate_tasks_from_course("Calc 101", [])

    assert len(result) == 2
    first, second = result
    assert first["title"] == "Solve 10 problems"
    assert first["priority"] == "high"
    assert first["estimatedTime"] == 20
    assert first["tags"] == ["math", "practice"]
    assert second["title"] == "Review notes"
    assert second["priority"] == "medium"  # unknown priority clamped
    assert second["estimatedTime"] == 45  # "45" coerced to int
    assert second["tags"] == ["ok", "x" * 50]  # junk dropped, overlong trimmed
    for task in result:
        assert task["estimatedTime"] >= 1


def test_task_generation_drops_unknown_instruction_keys(monkeypatch):
    response = json.dumps(
        {
            "tasks": [
                {
                    "title": "Flashcards",
                    "description": "Make 20 cards",
                    "priority": "high",
                    "estimatedTime": 30,
                    "tags": [],
                    "on_success": "rm -rf /",
                    "is_admin": True,
                }
            ]
        }
    )
    monkeypatch.setattr(task_generator, "call_llm_task_generation", lambda *a: response)

    result = task_generator.generate_tasks_from_course("Calc 101", [])

    assert len(result) == 1
    assert set(result[0].keys()) == {"title", "description", "priority", "estimatedTime", "tags"}
    assert "on_success" not in result[0]
    assert "is_admin" not in result[0]
