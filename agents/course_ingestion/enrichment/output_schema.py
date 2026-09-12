"""INGEST-09 — strict output schema for LLM course-material enrichment.

Every field the enrichment stage hands back to the pipeline is defined here
and validated through ``EnrichmentOutput``. ``normalize_enrichment`` salvages
best-effort fields from a raw LLM JSON payload — coercing types, dropping
malformed entries, truncating to safe limits — so the pipeline always
receives a schema-conforming record and never an exception. Metadata cleanup
stays in ``llm_enricher``; this module only guards shape and safety bounds.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List

from pydantic import BaseModel, Field, StringConstraints

LIST_MAX_ITEMS = 40
CLEANED_TEXT_MAX_CHARS = 50_000
CONCEPT_MAX_CHARS = 300
TERM_MAX_CHARS = 200
DEFINITION_MAX_CHARS = 2_000
FORMULA_MAX_CHARS = 500
EXAMPLE_MAX_CHARS = 2_000


def _scalar(value: Any) -> str:
    """Best-effort string coercion; non-strings become plain text."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def _string_list(value: Any, max_chars: int, max_items: int = LIST_MAX_ITEMS) -> List[str]:
    """Keep only real strings from a payload field, trimmed and bounded.

    Non-string entries are treated as junk and dropped (LLM-invented
    structures are data noise, not enrichment).
    """
    if not isinstance(value, list):
        return []
    cleaned: List[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        cleaned.append(text[:max_chars])
        if len(cleaned) >= max_items:
            break
    return cleaned


ConceptStr = Annotated[str, StringConstraints(max_length=CONCEPT_MAX_CHARS, strip_whitespace=True)]
FormulaStr = Annotated[str, StringConstraints(max_length=FORMULA_MAX_CHARS, strip_whitespace=True)]
ExampleStr = Annotated[str, StringConstraints(max_length=EXAMPLE_MAX_CHARS, strip_whitespace=True)]


class Definition(BaseModel):
    """A cleaned term/definition pair produced by enrichment."""

    term: Annotated[str, StringConstraints(max_length=TERM_MAX_CHARS, strip_whitespace=True)]
    definition: Annotated[str, StringConstraints(max_length=DEFINITION_MAX_CHARS, strip_whitespace=True)]


class EnrichmentOutput(BaseModel):
    """Schema-validated enrichment result for one course document."""

    cleaned_text: Annotated[
        str, StringConstraints(max_length=CLEANED_TEXT_MAX_CHARS, strip_whitespace=True)
    ]
    key_concepts: List[ConceptStr] = Field(default_factory=list, max_length=LIST_MAX_ITEMS)
    definitions: List[Definition] = Field(default_factory=list, max_length=LIST_MAX_ITEMS)
    formulas: List[FormulaStr] = Field(default_factory=list, max_length=LIST_MAX_ITEMS)
    examples: List[ExampleStr] = Field(default_factory=list, max_length=LIST_MAX_ITEMS)


def normalize_enrichment(data: Any, fallback_text: str) -> EnrichmentOutput:
    """Build a schema-valid ``EnrichmentOutput`` from raw LLM JSON payload.

    Unknown keys are ignored; malformed fields degrade to their defaults.
    """
    if not isinstance(data, dict):
        data = {}

    cleaned_text = _scalar(data.get("cleaned_text") or fallback_text)[:CLEANED_TEXT_MAX_CHARS]

    definitions: List[Dict[str, str]] = []
    for entry in (data.get("definitions") or []):
        if not isinstance(entry, dict):
            continue
        definitions.append(
            {
                "term": _scalar(entry.get("term"))[:TERM_MAX_CHARS],
                "definition": _scalar(entry.get("definition"))[:DEFINITION_MAX_CHARS],
            }
        )
        if len(definitions) >= LIST_MAX_ITEMS:
            break

    return EnrichmentOutput(
        cleaned_text=cleaned_text,
        key_concepts=_string_list(data.get("key_concepts"), CONCEPT_MAX_CHARS),
        definitions=definitions,
        formulas=_string_list(data.get("formulas"), FORMULA_MAX_CHARS),
        examples=_string_list(data.get("examples"), EXAMPLE_MAX_CHARS),
    )
