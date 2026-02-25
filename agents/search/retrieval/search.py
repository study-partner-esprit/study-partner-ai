import os
import requests
import time
import json
from dotenv import load_dotenv

# Load .env from package root
load_dotenv()

APIFY_API_KEY = os.getenv("APIFY_API_KEY") or ""


def apify_web_search(query, max_results=5, wait_timeout=30):
    if not APIFY_API_KEY:
        print("❌ APIFY_API_KEY not found")
        return []

    url = f"https://api.apify.com/v2/acts/apify~google-search-scraper/runs?token={APIFY_API_KEY}"
    payload = {
        "queries": query,
        "maxPagesPerQuery": 1,
        "resultsPerPage": max_results,
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code not in [200, 201]:
            print(f"❌ Error response: {response.text}")
            return []

        run_data = response.json()
        run_id = None
        if "data" in run_data:
            run_id = run_data["data"].get("id")

        if not run_id:
            return []

        run_status_url = f"https://api.apify.com/v2/actor-runs/{run_id}?token={APIFY_API_KEY}"
        start_time = time.time()

        while time.time() - start_time < wait_timeout:
            status_response = requests.get(run_status_url, timeout=10)
            status_response.raise_for_status()
            status_data = status_response.json()
            status = status_data.get("data", {}).get("status")
            if status == "SUCCEEDED":
                dataset_id = status_data.get("data", {}).get("defaultDatasetId")
                if not dataset_id:
                    return []
                dataset_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?token={APIFY_API_KEY}"
                dataset_response = requests.get(dataset_url, timeout=10)
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
                                    if url and isinstance(url, str) and url.startswith("http"):
                                        if url not in urls:
                                            urls.append(url)
                            if not urls:
                                url = item.get("url") or item.get("link") or item.get("href")
                                if url and isinstance(url, str) and url.startswith("http") and "google.com/search" not in url:
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

            time.sleep(2)

        return []

    except requests.exceptions.RequestException as e:
        print(f"❌ Apify API error: {e}")
        return []
    except Exception as e:
        print(f"❌ Error in apify_web_search: {e}")
        return []
