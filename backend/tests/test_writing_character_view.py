import pytest
from app.core.canonical import canonical_hash
from app.detectors.text.character_view import derive
from app.detectors.text.extract import extract
from app.detectors.text.segment import segment
from app.evidence import document_character_review as review


def test_writing_character_view_preserves_source_positions_for_recovery():
    source = "The compu\u200bter checked its journаl. The relea\u00adse met the policy."
    alternate, edits = derive(source)

    assert alternate == "The computer checked its journal. The release met the policy."
    assert [(source[index], replacement) for index, replacement in edits] == [
        ("\u200b", ""), ("а", "a"), ("\u00ad", ""),
    ]
    replacements = dict(edits)
    assert "".join(replacements.get(index, ch) for index, ch in enumerate(source)) == alternate
    assert derive(alternate) == (alternate, [])


@pytest.mark.parametrize("source", [
    "Greek άνθρωπος, Cyrillic договор, Arabic مثال\u200cكتابي.",
    "A combining mark keeps mixed word journа́l intact.",
    "https://exаmple.org and <chаrles@example.org>.",
    "Family 👨\u200d👩\u200d👧, noon\u00a0today, and visible long-term work.",
    "Keep directional controls \u2067בחינה\u2069 and breaks \u200b between visible words.",
])
def test_writing_character_view_keeps_legitimate_text(source):
    assert derive(source) == (source, [])


def test_character_review_rejects_a_rehashed_alternate_input(tmp_path):
    file = tmp_path / "body.txt"
    file.write_text("The original is on file. The compu\u200bter checked its approved journal. All entries are complete.")
    source = extract(file, ".txt")
    spans = [{"start": unit.start, "end": unit.end} for unit in segment(source)]
    row, adjusted = review.geometry(source.text, spans, source.source_map, source.meta)
    assert "The computer checked" in adjusted
    earlier = {"response_exact_hex": (0.4).hex(), "tokens": 28}
    later = {"response_exact_hex": (0.6).hex(), "tokens": 26}
    row.update(as_submitted=earlier, adjusted_primary=later, adjusted_second=later, state="changed")
    reading = {"schema": review.SCHEMA, "input_view": review.INPUT_VIEW, "ranges": [row], "authorship_confidence_contribution": 0}
    evidence = {"source": {"chars": len(source.text), "method": "plain", review.COMMITMENT: canonical_hash(reading)},
                "coverage": {review.KEY: reading}}
    review.verify_source(evidence, source.text, source.source_map, source.meta, sentences=spans)
    row["adjusted_input_sha256"] = "a" * 64
    evidence["source"][review.COMMITMENT] = canonical_hash(reading)
    with pytest.raises(ValueError, match="differ from the source text"):
        review.verify_source(evidence, source.text, source.source_map, source.meta, sentences=spans)
