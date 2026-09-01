"""KnowledgeExtractRequest schema tests (BLOOM-03).

Mirrors `tests/shared/payload-schemas.test.js` (`validateKnowledgeExtractPayload`
block) on the Node side. Limits must stay identical on both sides — the
orchestrator rejects pre-publish exactly what the worker rejects post-delivery.
"""

import pytest
from pydantic import ValidationError

from workers.schemas import (
    CONTENT_REF_MAX_CHARS,
    COURSE_ID_MAX_CHARS,
    DOCUMENT_ID_MAX_CHARS,
    KnowledgeExtractRequest,
)


def test_accepts_a_valid_payload():
    r = KnowledgeExtractRequest(
        documentId="doc-1", courseId="c-1", contentRef="s3://bucket/object"
    )
    assert r.documentId == "doc-1"
    assert r.courseId == "c-1"
    assert r.contentRef == "s3://bucket/object"


def test_rejects_missing_or_blank_references():
    with pytest.raises(ValidationError):
        KnowledgeExtractRequest()
    with pytest.raises(ValidationError):
        KnowledgeExtractRequest(documentId="d", courseId="c")
    with pytest.raises(ValidationError):
        KnowledgeExtractRequest(documentId="d", courseId="c", contentRef="")
    with pytest.raises(ValidationError):
        KnowledgeExtractRequest(documentId="  ", courseId="c", contentRef="r")


def test_rejects_over_length_references():
    with pytest.raises(ValidationError):
        KnowledgeExtractRequest(
            documentId="x" * (DOCUMENT_ID_MAX_CHARS + 1), courseId="c", contentRef="r"
        )
    with pytest.raises(ValidationError):
        KnowledgeExtractRequest(
            documentId="d", courseId="x" * (COURSE_ID_MAX_CHARS + 1), contentRef="r"
        )
    with pytest.raises(ValidationError):
        KnowledgeExtractRequest(
            documentId="d", courseId="c", contentRef="x" * (CONTENT_REF_MAX_CHARS + 1)
        )


def test_unknown_fields_forbidden():
    with pytest.raises(ValidationError):
        KnowledgeExtractRequest(
            documentId="d",
            courseId="c",
            contentRef="r",
            raw_content="must never be inline in the envelope",
        )