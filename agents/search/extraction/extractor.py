import requests
from bs4 import BeautifulSoup

def extract_text(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")

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

        return text

    except Exception as e:
        print(f"Erreur extraction {url}: {e}")
        return ""
