import pytest
from mts_daily_arxiv import parse_papers

DOC = """# Unified\n| Title | Date | Comment |\n|---|---|---|\n| [One](https://arxiv.org/abs/1) | 2026-09-21 | <details>nine pages</details> |\n| [Two](https://arxiv.org/abs/2) | 2026-09-22 |  |\n| [Duplicate](https://arxiv.org/abs/1) | 2026-09-21 | no |\n\n# Other\n| [Three](https://arxiv.org/abs/3) | 2026-09-21 | no |\n"""


def test_parses_and_cleans_table_rows():
    papers = parse_papers(DOC)
    assert papers[0].title == "One"
    assert papers[0].comment == "nine pages"
    assert papers[0].arxiv_id == "1"
    assert len(papers) == 2


def test_date_filter_and_limit():
    assert [p.title for p in parse_papers(DOC, on="2026-09-22")] == ["Two"]
    assert len(parse_papers(DOC, limit=1)) == 1


def test_ignores_malformed_rows_and_missing_section():
    assert parse_papers("# Unified\n| bad | nope | x |\n") == []
    assert parse_papers(DOC, section="Missing") == []


def test_rejects_invalid_options():
    with pytest.raises(ValueError):
        parse_papers(DOC, on="tomorrow")
    with pytest.raises(ValueError):
        parse_papers(DOC, limit=0)
