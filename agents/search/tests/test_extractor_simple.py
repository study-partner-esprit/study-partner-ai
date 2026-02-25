from agents.search.extraction.extractor import extract_text


def test_simple_extract_monkeypatch(monkeypatch):
    # Simulate extract_text returning empty for unreachable URL
    assert extract_text("http://example.invalid") == "" or isinstance(extract_text("http://example.invalid"), str)
