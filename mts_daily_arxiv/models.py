from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Paper:
    title: str
    url: str
    date: str
    comment: str = ""

    @property
    def arxiv_id(self) -> str:
        marker = "/abs/"
        if marker not in self.url:
            return ""
        return self.url.split(marker, 1)[1].split("?", 1)[0]
