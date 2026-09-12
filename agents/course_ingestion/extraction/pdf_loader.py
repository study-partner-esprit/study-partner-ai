"""
PDF text extraction (INGEST-04).

Uploaded PDFs are untrusted input: a crafted document can make PyMuPDF spin,
allocate unbounded memory, or hit parser edge cases. `extract_text_from_pdf`
therefore only ever runs inside an OS-level sandbox spawned by
`extract_text_from_pdf_sandboxed`:

  * a fresh forked process (crashes/hangs stay contained),
  * RLIMIT_AS + RLIMIT_CPU applied *before* any parsing code runs,
  * a strict wall-clock timeout enforced by the supervisor (kills on breach).

Encrypted/password-protected documents are rejected with a clear error
(INGEST-04 acceptance criterion) instead of being passed to a parser we do not
trust with attacker-chosen passwords.
"""

import multiprocessing as mp
import os

# ---------------------------------------------------------------------------
# Error contract
# ---------------------------------------------------------------------------


class PdfValidationError(Exception):
    """Base error for PDF extraction failures in the ingestion pipeline."""


class EncryptedPdfError(PdfValidationError):
    """Raised when a PDF is encrypted/password-protected and cannot be parsed."""


class PdfParseTimeoutError(PdfValidationError):
    """Raised when parsing exceeds the strict wall-clock sandbox timeout."""


class PdfParseLimitError(PdfValidationError):
    """Raised when parsing exceeds the sandbox CPU/memory limits."""


# ---------------------------------------------------------------------------
# Extraction — runs only inside the sandboxed child
# ---------------------------------------------------------------------------


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a digital PDF using PyMuPDF.

    PyMuPDF is imported lazily so importing this module never requires it (and
    the repository stays testable / fast-to-start without fitz installed).
    Called by the sandbox child only.
    """
    import fitz  # noqa: PLC0415 — lazy: startup-friendly, tests stay PyMuPDF-free

    doc = fitz.open(pdf_path)
    try:
        if doc.needs_pass:
            raise EncryptedPdfError(
                f"PDF is encrypted/password-protected: {os.path.basename(pdf_path)}"
            )
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    finally:
        doc.close()


def _apply_resource_limits(max_memory_mb: int, max_cpu_s: int) -> None:
    """Constrain the child before any parsing code runs (POSIX only)."""
    try:
        import resource  # noqa: PLC0415
    except ImportError:  # non-POSIX: the wall-clock timeout still applies
        return
    mem_bytes = max_memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (max_cpu_s, max_cpu_s))


def _run_sandboxed_worker(pdf_path: str, conn, max_memory_mb: int, max_cpu_s: int) -> None:
    """Child-process entrypoint: bound the process, parse, then report back."""
    try:
        _apply_resource_limits(max_memory_mb, max_cpu_s)
        text = extract_text_from_pdf(pdf_path)
        conn.send(("ok", text))
    except BaseException as exc:  # noqa: BLE001 — report every outcome to parent
        try:
            conn.send(("error", type(exc).__name__, str(exc)))
        except Exception:  # noqa: BLE001 — parent may already have timed out
            pass
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Sandbox supervisor — used by the ingestion agent
# ---------------------------------------------------------------------------


def extract_text_from_pdf_sandboxed(
    pdf_path: str,
    timeout_s: int = 45,
    max_memory_mb: int = 512,
) -> str:
    """
    Extract PDF text inside a bounded sandbox (INGEST-04).

    Raises:
        EncryptedPdfError: the document is encrypted/password-protected.
        PdfParseTimeoutError: parsing exceeded `timeout_s`.
        PdfParseLimitError: the child was killed by CPU/memory limits.
        PdfValidationError: any other (reported) parse failure.
    """
    ctx = mp.get_context("fork")  # POSIX sandboxing; Windows is not a target
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=_run_sandboxed_worker,
        args=(pdf_path, child_conn, max_memory_mb, timeout_s),
    )
    proc.start()
    child_conn.close()  # parent never uses the child end

    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(2)
        if proc.is_alive():
            proc.kill()
        parent_conn.close()
        raise PdfParseTimeoutError(
            f"PDF parsing exceeded the {timeout_s}s sandbox timeout: "
            f"{os.path.basename(pdf_path)}"
        )

    if not parent_conn.poll(0):
        # Child ended without reporting (SIGXCPU / OOM / segfault).
        parent_conn.close()
        raise PdfParseLimitError(
            f"PDF parsing exceeded sandbox CPU/memory limits: "
            f"{os.path.basename(pdf_path)}"
        )

    try:
        status = parent_conn.recv()
    except (EOFError, OSError):
        raise PdfParseLimitError(
            f"PDF parsing aborted inside sandbox (CPU/memory limits): "
            f"{os.path.basename(pdf_path)}"
        ) from None
    finally:
        parent_conn.close()

    if status[0] == "ok":
        return status[1]

    _, kind, detail = status
    if kind == "EncryptedPdfError":
        raise EncryptedPdfError(detail)
    raise PdfValidationError(f"PDF parsing failed: {detail}")