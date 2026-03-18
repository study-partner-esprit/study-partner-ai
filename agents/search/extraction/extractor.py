import requests
from bs4 import BeautifulSoup


def extract_text(url, max_chars=12000, max_bytes=400000, request_timeout=(3, 7)):
    try:
        response = requests.get(
            url,
            timeout=request_timeout,
            stream=True,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        response.raise_for_status()

        content_type = (response.headers.get("Content-Type") or "").lower()
        if (
            "text/html" not in content_type
            and "application/xhtml+xml" not in content_type
        ):
            return ""

        chunks = []
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
            if not chunk:
                continue
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", errors="ignore")
            downloaded += len(chunk.encode("utf-8", errors="ignore"))
            chunks.append(chunk)
            if downloaded >= max_bytes:
                break

        html = "".join(chunks)
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.decompose()

        paragraphs = soup.find_all("p")
        clean_paragraphs = []

        for p in paragraphs:
            text = p.get_text(strip=True)
            if 50 < len(text) < 500:
                clean_paragraphs.append(text)

        text = " ".join(clean_paragraphs)
        text = " ".join(text.split())

        return text[:max_chars]

    except Exception as e:
        print(f"Erreur extraction {url}: {e}")
        return ""
