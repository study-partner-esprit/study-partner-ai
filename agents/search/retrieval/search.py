import os
import requests
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env from package root and search agent directory
load_dotenv()
_SEARCH_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if _SEARCH_ENV_PATH.exists():
    load_dotenv(dotenv_path=_SEARCH_ENV_PATH, override=False)


def _get_apify_api_key():
    return (os.getenv("APIFY_API_KEY") or os.getenv("HF_API_KEY") or "").strip()


def _duckduckgo_web_search(query, max_results=5):
    search_url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    try:
        response = requests.post(
            search_url,
            data={"q": query},
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(response.text, "html.parser")
        urls = []
        for tag in soup.select("a.result__a"):
            href = tag.get("href")
            if href and href.startswith("http") and href not in urls:
                urls.append(href)
            if len(urls) >= max_results:
                break
        return urls
    except Exception:
        return []


def apify_web_search(query, max_results=5, wait_timeout=12):
    apify_api_key = _get_apify_api_key()

    if not apify_api_key:
        print("❌ APIFY_API_KEY not found")
        return []

    url = f"https://api.apify.com/v2/acts/apify~google-search-scraper/runs?token={apify_api_key}"
    payload = {
        "queries": query,
        "maxPagesPerQuery": 1,
        "resultsPerPage": max_results,
    }

    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code not in [200, 201]:
            print(f"❌ Error response: {response.text}")
            return []

        run_data = response.json()
        run_id = None
        if "data" in run_data:
            run_id = run_data["data"].get("id")

        if not run_id:
            return []

        run_status_url = (
            f"https://api.apify.com/v2/actor-runs/{run_id}?token={apify_api_key}"
        )
        start_time = time.time()

        while time.time() - start_time < wait_timeout:
            status_response = requests.get(run_status_url, timeout=5)
            status_response.raise_for_status()
            status_data = status_response.json()
            status = status_data.get("data", {}).get("status")
            if status == "SUCCEEDED":
                dataset_id = status_data.get("data", {}).get("defaultDatasetId")
                if not dataset_id:
                    return []
                dataset_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?token={apify_api_key}"
                dataset_response = requests.get(dataset_url, timeout=5)
                dataset_response.raise_for_status()
                items = dataset_response.json()
                urls = []
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            organic_results = item.get("organicResults", [])
                            if isinstance(organic_results, list) and organic_results:
                                for result in organic_results:
                                    url = result.get("url") or result.get("link")
                                    if (
                                        url
                                        and isinstance(url, str)
                                        and url.startswith("http")
                                    ):
                                        if url not in urls:
                                            urls.append(url)
                            if not urls:
                                url = (
                                    item.get("url")
                                    or item.get("link")
                                    or item.get("href")
                                )
                                if (
                                    url
                                    and isinstance(url, str)
                                    and url.startswith("http")
                                    and "google.com/search" not in url
                                ):
                                    urls.append(url)
                elif isinstance(items, dict):
                    for key in ["results", "organicResults", "items", "data"]:
                        if key in items and isinstance(items[key], list):
                            for item in items[key]:
                                if isinstance(item, dict):
                                    url = item.get("url") or item.get("link")
                                    if url and url.startswith("http"):
                                        urls.append(url)

                return urls[:max_results]

            elif status in ["FAILED", "ABORTED", "TIMED-OUT"]:
                return []

            time.sleep(1)

        return []

    except requests.exceptions.RequestException as e:
        print(f"❌ Apify API error: {e}")
        return []
    except Exception as e:
        print(f"❌ Error in apify_web_search: {e}")
        return []


def web_search(query, max_results=5, wait_timeout=None):
    if wait_timeout is None:
        wait_timeout = int(os.getenv("SEARCH_RETRIEVAL_TIMEOUT_SECONDS", "12"))
    urls = apify_web_search(query, max_results=max_results, wait_timeout=wait_timeout)
    if urls:
        return urls
    return _duckduckgo_web_search(query, max_results=max_results)
