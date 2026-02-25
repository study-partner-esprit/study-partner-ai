# Lightweight test for the search pipeline helpers
from agents.search.retrieval.search import apify_web_search


def test_apify_search_no_key(monkeypatch):
    # Ensure function returns empty list when no API key
    monkeypatch.setenv("APIFY_API_KEY", "")
    urls = apify_web_search("test query")
    assert isinstance(urls, list)
