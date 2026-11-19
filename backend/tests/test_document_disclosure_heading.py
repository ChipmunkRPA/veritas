from copy import deepcopy

import pymupdf
import pytest

from app.core.canonical import canonical_hash
from app.detectors.text.extract import extract
from app.evidence import document_disclosure_heading as heading


def _pdf(path, *, kind="normal", title="AI Usage Disclosure:"):
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((80, 130), "The next item is quoted from another document.")
        value = {
            "quoted": f'"{title}"',
            "inline": f"Look for the phrase {title} in the appendix.",
            "split": "AI Usage\nDisclosure:",
        }.get(kind, title)
        page.insert_text((80, 200), value, fontname="hebo", fontsize=12,
                         color=(1, 1, 1) if kind == "white" else (0, 0, 0),
                         render_mode=3 if kind == "invisible" else 0)
        if kind == "duplicate":
            page.insert_text((80, 240), title, fontname="hebo", fontsize=12)
        if kind == "covered":
            page.draw_rect(pymupdf.Rect(74, 178, 280, 214), color=None, fill=(1, 1, 1))
        page.insert_text((80, 300), "This example text declares that someone used a writing assistant.")
        if kind == "rotated":
            page.set_rotation(90)
        if kind == "too_many_pages":
            for _ in range(heading.MAX_PAGES):
                pdf.new_page().insert_text((80, 300), "This page contains additional selectable words.")
        pdf.save(path)
    return extract(path, ".pdf", allow_ocr=False)


def test_replays_only_the_printed_heading_with_native_word_boxes(tmp_path):
    path = tmp_path / "source.pdf"
    doc = _pdf(path)
    record = heading.capture(path, doc)
    assert record is not None
    assert record["schema"] == heading.SCHEMA
    assert doc.text[record["start"]:record["end"]] == record["text"] == "AI Usage Disclosure:"
    assert record["page"] == 1 and record["block"] == 1
    assert record["affects_overall_confidence"] is False
    assert [word["text"] for word in record["word_boxes"]] == ["AI", "Usage", "Disclosure:"]
    assert "quotation or example" in heading.LIMITATION
    with pymupdf.open(path) as pdf:
        raw = [word for word in pdf[0].get_text("words") if word[5] == record["block"]]
    assert [[round(side, 6) for side in word[:4]] for word in raw] == [item["box"] for item in record["word_boxes"]]
    heading.verify_original(record, path, doc)


@pytest.mark.parametrize("kind", ["quoted", "inline", "split", "duplicate", "white", "invisible", "covered", "rotated",
                                "too_many_pages"])
def test_withholds_unsupported_or_unverified_heading_locations(tmp_path, kind):
    path = tmp_path / "source.pdf"
    assert heading.capture(path, _pdf(path, kind=kind)) is None


def test_validates_the_public_record_and_original_independently(tmp_path):
    path = tmp_path / "source.pdf"
    doc = _pdf(path, title="Generative AI use disclosure")
    record = heading.capture(path, doc)
    assert record is not None
    evidence = {"source": {"method": "pymupdf", "chars": len(doc.text), heading.COMMITMENT: canonical_hash(record)},
                heading.KEY: record}
    heading.validate_evidence({"source": {"method": "pymupdf"}})
    heading.validate_evidence(evidence)
    with pytest.raises(ValueError, match="bound to its PDF source"):
        heading.validate_evidence({"source": evidence["source"]})
    with pytest.raises(ValueError, match="bound to its PDF source"):
        heading.validate_evidence({**evidence, "source": {**evidence["source"], "method": "plain"}})
    with pytest.raises(ValueError, match="source commitment"):
        heading.validate_evidence({**evidence, heading.KEY: {**record, "block": 2}})
    with pytest.raises(ValueError, match="original PDF"):
        heading.verify_original({**record, "block": 2}, path, doc)

    moved = deepcopy(record)
    moved["word_boxes"][0]["box"][0] += 1
    with pytest.raises(ValueError, match="original PDF"):
        heading.verify_original(moved, path, doc)

    copy = tmp_path / "other.pdf"
    _pdf(copy, title="AI Use Disclosure")
    assert heading.capture(copy, doc) is None


@pytest.mark.parametrize("update", [
    {"affects_overall_confidence": 0}, {"page": True}, {"end": 50000}, {"box": [0.0, 0.0, 1.0000001, 1.0]},
    {"word_boxes": []}, {"claimed_scope": "manuscript"},
])
def test_rejects_invalid_coordinates_and_added_claims(tmp_path, update):
    path = tmp_path / "source.pdf"
    doc = _pdf(path)
    record = heading.capture(path, doc)
    assert record is not None
    with pytest.raises(ValueError):
        heading.validate(record | update, source_chars=len(doc.text), pages=1)
