from __future__ import annotations

from copy import deepcopy

import docx
import pytest
from docx.oxml import OxmlElement

from app.core.canonical import canonical_hash
from app.detectors.text.extract import extract
from app.evidence import document_doi_reference as binding

CITED = "10.1111/j.1365-3040.2005.01490.x"
PRINTED = "10.3389/fpsyg.2013.00513"
REFERENCE = (
    "[1] Schurr U, Walter A, Rascher U. Functional dynamics of plant growth and "
    "photosynthesis–from steady-state to dynamics–from homogeneity to heterogeneity. "
    "Plant Cell Environ. 2006;29: 340–352. doi: "
)


def test_snapshot_disagreement_keeps_both_original_unicode_ranges(tmp_path):
    path = tmp_path / "source.txt"
    literal = PRINTED.upper()
    reference = REFERENCE + literal
    body = "Biology field notes include figures for two seasons.\r\n\r\nReferences\r\n\r\n"
    raw = body + reference + "\r\n\r\n[2] This extra entry has no registered identifier."
    path.write_bytes(raw.encode("utf-16"))
    original = extract(path, ".txt")
    result = binding.capture(original)
    assert result["snapshot"] == binding.SNAPSHOT and result["affects_overall_confidence"] is False
    assert (result["references"], result["consistent"], result["different_work"], result["unassessed"]) == (2, 0, 1, 1)
    [location] = result["locations"]
    assert original.text[location["start"]:location["end"]] == location["text"] == literal
    for field, text in (("location", literal), ("title_location", location["cited_title"])):
        assert "".join(raw[start:end] for start, end in location[field]["ranges"]) == text
    assert (location["printed"]["doi"], location["cited"]["doi"]) == (PRINTED, CITED)
    evidence = {binding.KEY: result, "source": {**result["source"], binding.COMMITMENT: canonical_hash(result)}}
    binding.verify_original(evidence, original)

    changed = deepcopy(evidence)
    region = changed[binding.KEY]["locations"][0]["title_location"]["ranges"][0]
    region[:] = [offset + 1 for offset in region]
    changed["source"][binding.COMMITMENT] = canonical_hash(changed[binding.KEY])
    with pytest.raises(ValueError, match="differs from the frozen extraction"):
        binding.verify_original(changed, original)


def test_hidden_word_reference_is_unassessed(tmp_path):
    path = tmp_path / "hidden.docx"
    word = docx.Document()
    word.add_paragraph("References")
    paragraph = word.add_paragraph(REFERENCE)
    hidden = paragraph.add_run(PRINTED)
    hidden._element.get_or_add_rPr().append(OxmlElement("w:vanish"))
    word.add_paragraph(REFERENCE + CITED)
    word.save(path)
    fitted = binding.capture(extract(path, ".docx"))
    assert (fitted["references"], fitted["consistent"], fitted["different_work"], fitted["unassessed"]) == (2, 1, 0, 1)
    assert fitted["locations"] == []
