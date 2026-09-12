"""
INGEST-04 — sandboxed PDF parsing.

`extract_text_from_pdf_sandboxed` runs PyMuPDF in a forked, resource-limited
child. These tests exercise the supervisor contract without needing PyMuPDF
installed: the extractor is monkeypatched (fork inherits the patch), so we can
verify happy path, error propagation, timeout enforcement and limit aborts.
"""

import time

import pytest

from agents.course_ingestion.extraction import pdf_loader


@pytest.fixture
def fake_pdf(tmp_path):
    return str(tmp_path / "doc.pdf")


def test_returns_extracted_text(monkeypatch, fake_pdf):
    monkeypatch.setattr(pdf_loader, "extract_text_from_pdf", lambda _pdf: "hello course")
    assert pdf_loader.extract_text_from_pdf_sandboxed(fake_pdf) == "hello course"


def test_propagates_encrypted_pdf_error(monkeypatch, fake_pdf):
    def _encrypted(_pdf):
        raise pdf_loader.EncryptedPdfError("PDF is encrypted/password-protected")

    monkeypatch.setattr(pdf_loader, "extract_text_from_pdf", _encrypted)
    with pytest.raises(pdf_loader.EncryptedPdfError, match="encrypted"):
        pdf_loader.extract_text_from_pdf_sandboxed(fake_pdf)


def test_wraps_other_parse_failures_in_pdf_validation_error(monkeypatch, fake_pdf):
    def _boom(_pdf):
        raise RuntimeError("fitz exploded")

    monkeypatch.setattr(pdf_loader, "extract_text_from_pdf", _boom)
    with pytest.raises(pdf_loader.PdfValidationError, match="fitz exploded"):
        pdf_loader.extract_text_from_pdf_sandboxed(fake_pdf)


def test_enforces_wall_clock_timeout(monkeypatch, fake_pdf):
    monkeypatch.setattr(pdf_loader, "_apply_resource_limits", lambda _mem, _cpu: None)
    monkeypatch.setattr(
        pdf_loader, "extract_text_from_pdf", lambda _pdf: time.sleep(5.0)
    )
    with pytest.raises(pdf_loader.PdfParseTimeoutError, match="sandbox timeout"):
        pdf_loader.extract_text_from_pdf_sandboxed(fake_pdf, timeout_s=0)


def test_limit_abort_when_child_never_reports(monkeypatch, fake_pdf):
    # Child exits without sending — as happens on SIGXCPU / OOM / SIGSEGV.
    monkeypatch.setattr(pdf_loader, "_run_sandboxed_worker", lambda *args: None)
    with pytest.raises(pdf_loader.PdfParseLimitError, match="limits"):
        pdf_loader.extract_text_from_pdf_sandboxed(fake_pdf)


def test_rejects_missing_file_with_clear_error(fake_pdf):
    with pytest.raises(pdf_loader.PdfValidationError):
        pdf_loader.extract_text_from_pdf_sandboxed(fake_pdf)