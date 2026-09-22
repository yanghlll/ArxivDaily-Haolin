from urllib.request import Request, urlopen

DEFAULT_URL = "https://raw.githubusercontent.com/zezhishao/MTS_Daily_ArXiv/main/README.md"


def fetch_markdown(url: str = DEFAULT_URL, *, timeout: float = 15.0) -> str:
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    request = Request(url, headers={"User-Agent": "mts-daily-arxiv/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(f"could not fetch paper digest: {exc}") from exc
