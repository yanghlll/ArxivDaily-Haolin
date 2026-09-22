import argparse
from .fetch import DEFAULT_URL, fetch_markdown
from .parser import parse_papers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print daily MTS arXiv papers")
    parser.add_argument("--date", dest="on", help="ISO date filter")
    parser.add_argument("--section", default="Unified")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args(argv)
    try:
        papers = parse_papers(fetch_markdown(args.url), section=args.section,
                              on=args.on, limit=args.limit)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    for paper in papers:
        suffix = f" — {paper.comment}" if paper.comment else ""
        print(f"{paper.date} | {paper.title} | {paper.url}{suffix}")
    return 0
