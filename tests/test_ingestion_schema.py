"""INGEST-05 — `study.ingest.course` payload mirror parity.

Limit names mirror `payloadSchemas.js` INGEST_* constants; drift breaks here so
the Node side and the worker stay typed to the same contract.
"""

import pytest
from pydantic import ValidationError

from workers.schemas import (
    INGEST_FILENAME_MAX_CHARS,
    INGEST_MAX_FILES,
    INGEST_MIME_MAX_CHARS,
    IngestionRequest,
)

VALID = {
    "courseId": "course-1",
    "fileRef": "uploads/courses/course-1",
    "files": [
        {
            "filename": "abc.pdf",
            "originalName": "notes.pdf",
            "mimetype": "application/pdf",
            "size": 2048,
            "path": "courses/course-1/abc.pdf",
        }
    ],
}


def test_accepts_valid_payload():
    req = IngestionRequest.model_validate(VALID)
    assert req.courseId == "course-1"
    assert len(req.files) == 1


def test_requires_course_id_and_file_ref():
    with pytest.raises(ValidationError):
        IngestionRequest.model_validate({})
    with pytest.raises(ValidationError):
        IngestionRequest.model_validate({"courseId": "c"})  # fileRef missing
    with pytest.raises(ValidationError):
        IngestionRequest.model_validate({"courseId": "c", "fileRef": " "})


def test_rejects_file_metadata_violations():
    bad = dict(VALID)
    bad["files"] = [{**VALID["files"][0], "size": -1}]
    with pytest.raises(ValidationError):
        IngestionRequest.model_validate(bad)

    bad2 = dict(VALID)
    bad2["files"] = [{**VALID["files"][0], "mimetype": "x" * (INGEST_MIME_MAX_CHARS + 1)}]
    assert len(bad2["files"][0]["mimetype"]) > INGEST_MIME_MAX_CHARS
    with pytest.raises(ValidationError):
        IngestionRequest.model_validate(bad2)


def test_rejects_too_many_files():
    many = [dict(VALID["files"][0], filename=f"f{i}.pdf") for i in range(INGEST_MAX_FILES + 1)]
    with pytest.raises(ValidationError):
        IngestionRequest.model_validate({**VALID, "files": many})


def test_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        IngestionRequest.model_validate({**VALID, "rawContent": "x"})


def test_filename_limit_matches_js():
    assert INGEST_FILENAME_MAX_CHARS == 256