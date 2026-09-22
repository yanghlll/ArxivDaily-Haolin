"""Small, dependency-free Markdown parser for the repository's paper tables."""

import re
from datetime import date
from .models import Paper

_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _clean(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[*_`]+", "", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_papers(markdown: str, *, section: str = "Unified", on: str | None = None,
                 limit: int = 15) -> list[Paper]:
    """Extract papers from a Markdown table below *section*.

    Rows are read in source order, duplicates (by arXiv URL) are removed, and
    ``on`` filters the ISO date. Invalid rows are ignored deliberately so a
    partially edited daily page remains readable.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    if on is not None:
        try:
            date.fromisoformat(on)
        except ValueError as exc:
            raise ValueError("on must be an ISO date (YYYY-MM-DD)") from exc
    heading = re.compile(rf"^\s*(?:#+\s*)?{re.escape(section)}\s*$", re.I)
    in_section = False
    seen: set[str] = set()
    result: list[Paper] = []
    for raw in markdown.splitlines():
        if re.match(r"^\s*#{1,6}\s+", raw):
            in_section = bool(heading.match(re.sub(r"^\s*#+\s*", "", raw)))
            continue
        if not in_section or "|" not in raw:
            continue
        cells = [c.strip() for c in raw.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0].lower() in {"title", "---"}:
            continue
        match = _LINK.search(cells[0])
        if not match or not _DATE.fullmatch(cells[1]):
            continue
        url = match.group(2)
        if url in seen:
            continue
        if on is not None and cells[1] != on:
            continue
        seen.add(url)
        result.append(Paper(_clean(match.group(1)), url, cells[1], _clean(cells[2])))
        if len(result) == limit:
            break
    return result
