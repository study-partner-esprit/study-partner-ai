from agents.search.retrieval.search import apify_web_search


def test_apify_no_key(monkeypatch):
    monkeypatch.setenv("APIFY_API_KEY", "")
    assert apify_web_search("anything") == []
