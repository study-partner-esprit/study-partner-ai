import ipaddress
import socket
from urllib.parse import urlsplit, urljoin

import requests
from bs4 import BeautifulSoup

# Blocks private / loopback / link-local / reserved networks so extraction
# can never be redirected into the internal network (SSRF).
_BLOCKED_NETWORKS = (
    # IPv4
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.0.0.0/24",
    "192.0.2.0/24",
    "192.168.0.0/16",
    "198.18.0.0/15",
    "198.51.100.0/24",
    "203.0.113.0/24",
    "224.0.0.0/4",
    "240.0.0.0/4",
    # IPv6
    "::/128",
    "::1/128",
    "fc00::/7",
    "fe80::/10",
    "ff00::/8",
)
_BLOCKED_NETS = [ipaddress.ip_network(net) for net in _BLOCKED_NETWORKS]

_MAX_REDIRECTS = 5


def _is_blocked_ip(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparseable → refuse
    for net in _BLOCKED_NETS:
        if addr in net:
            return True
    return not addr.is_global


def _validate_url(url: str):
    """Return (ok: bool, reason: str). Blocks schemes, credentials, non-global IPs."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False, "invalid-url"
    if parts.scheme not in ("http", "https"):
        return False, "scheme"
    if parts.username or parts.password:
        return False, "credentials-in-url"
    hostname = parts.hostname
    if not hostname:
        return False, "no-hostname"

    port = parts.port
    if port is not None and not (1 <= port <= 65535):
        return False, "invalid-port"

    try:
        infos = socket.getaddrinfo(hostname, port or (443 if parts.scheme == "https" else 80))
    except (socket.gaierror, OSError):
        return False, "dns-resolution-failed"

    for info in infos:
        ip = info[4][0]
        if _is_blocked_ip(ip):
            return False, "blocked-ip"

    return True, ""


def _fetch(url, session, request_timeout, headers):
    """Fetch *url* following at most _MAX_REDIRECTS redirects, validating every hop."""
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        ok, reason = _validate_url(current)
        if not ok:
            print(f"SSRF guard blocked {current} ({reason})")
            return None

        response = session.get(
            current,
            timeout=request_timeout,
            stream=True,
            allow_redirects=False,
            headers=headers,
        )
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location")
            response.close()
            if not location:
                return None
            current = urljoin(current, location)
            continue
        return response

    return None


def extract_text(url, max_chars=12000, max_bytes=400000, request_timeout=(3, 7)):
    try:
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return ""

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml",
        }
        with requests.Session() as session:
            response = _fetch(url, session, request_timeout, headers)

        if response is None:
            return ""

        response.raise_for_status()

        content_type = (response.headers.get("Content-Type") or "").lower()
        if (
            "text/html" not in content_type
            and "application/xhtml+xml" not in content_type
        ):
            response.close()
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
        response.close()

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
