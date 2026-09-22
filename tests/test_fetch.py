import pytest
from mts_daily_arxiv.fetch import fetch_markdown


def test_fetch_rejects_nonpositive_timeout():
    with pytest.raises(ValueError):
        fetch_markdown(timeout=0)
