from agents.search.extraction.extractor import extract_text


def test_extract_text_invalid_url():
    # Should handle invalid URL gracefully
    assert extract_text("http://nonexistent.invalid/") == "" or isinstance(
        extract_text("http://nonexistent.invalid/"), str
    )
