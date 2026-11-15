from pathlib import Path

import pytest

from app.detectors.text.extract import extract
from app.evidence import document_bibliography as rules

PROSE = (
    "The researchers stored the raw observations at the field office. They checked the instruments every morning "
    "and repeated the same measurement at different times of day. The final analysis combines the records "
    "from all four visits and describes which measurements were missing."
)
REFERENCES = [
    "de Silva, AB. (2018). The first published measurements. Journal of Methods. doi:10.1234/alpha.",
    "Smith R. (2017). Recording site conditions. Journal of Methods. 10(4):22-31.",
    "Rivera, A. (2009). Repeat observations. Journal of Research, 4(1), 12–20.",
    "Ito, T. (2019). Documentation between visits. Field Review. doi:10.1234/delta.",
]


def source(tmp_path: Path, entries):
    path = tmp_path / "article.txt"
    raw = "\r\n\r\n".join([PROSE, "Works Cited", *entries])
    path.write_bytes(raw.encode("utf-8"))
    return raw, extract(path, ".txt")


def test_terminal_references_are_removed_at_original_source_boundaries(tmp_path):
    raw, article = source(tmp_path, REFERENCES)
    scoped = rules.body_view(article)
    record = scoped.meta[rules.KEY]
    assert scoped.text == PROSE and scoped.source_map == article.source_map[:1]
    assert article.text.endswith(REFERENCES[-1]) and rules.body_view(scoped) is scoped
    assert record["entry_count"] == 4 and record["section_start"] == len(PROSE) + 2
    assert record["first_location"]["start"] == raw.index("Works Cited")
    assert record["last_location"]["end"] == len(raw)
    evidence = {"source": {"chars": len(PROSE), "method": "plain"}, "coverage": {rules.KEY: record}}
    rules.verify_source(evidence, scoped.meta)
    fake = {**evidence, "coverage": {rules.KEY: {**record, "entry_count": 5}}}
    with pytest.raises(ValueError, match="recorded bibliography differs"):
        rules.verify_source(fake, scoped.meta)


@pytest.mark.parametrize("ending", [REFERENCES[:3], [*REFERENCES, PROSE], [*REFERENCES[:2], PROSE, *REFERENCES[2:]]])
def test_incomplete_or_interleaved_lists_keep_all_prose(tmp_path, ending):
    _, article = source(tmp_path, ending)
    assert rules.body_view(article) is article
