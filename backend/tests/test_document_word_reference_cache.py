# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import hashlib

import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from test_document_word_saved_fields import add_field, word_bytes

from app.core.canonical import canonical_hash
from app.evidence import document_word_reference_cache as reader
from app.evidence import document_word_reference_cache_binding as binding

SOURCE_PREFIX = "😀  The approved version is "
FIELD_PREFIX = "The saved Word display says "


def _element(parent, tag, **attrs):
    node = OxmlElement(f"w:{tag}")
    for name, value in attrs.items():
        node.set(qn(f"w:{name}"), value)
    parent.append(node)
    return node


def _word(*, shape, instruction, text, hidden=False, locked=False, ambiguous=False):
    word = Document()
    target = word.add_paragraph(SOURCE_PREFIX)
    _element(target._p, "bookmarkStart", id="4", name="SourceLabel")
    target.add_run("Café ")
    if hidden:
        target.runs[-1].font.hidden = True
    target.add_run("Phase C")
    _element(target._p, "bookmarkEnd", id="4")
    target.add_run(" in the approved schedule.")
    field = word.add_paragraph(FIELD_PREFIX)
    if shape == "simple":
        marker = add_field(field, instruction, text)
    else:
        marker = _element(field.add_run()._r, "fldChar", fldCharType="begin")
        _element(field.add_run()._r, "instrText").text = instruction
        _element(field.add_run()._r, "fldChar", fldCharType="separate")
        field.add_run(text)
        _element(field.add_run()._r, "fldChar", fldCharType="end")
    if locked:
        marker.set(qn("w:fldLock"), "true")
    field.add_run(" in the summary.")
    if ambiguous:
        other = word.add_paragraph()
        _element(other._p, "bookmarkStart", id="5", name="SourceLabel")
        other.add_run("Café Phase B")
        _element(other._p, "bookmarkEnd", id="5")
    return word


def _save(tmp_path, *, strict=False, shape="simple", instruction=r"REF SourceLabel \h \* MERGEFORMAT",
          text="Café Phase A", **options):
    raw = word_bytes(_word(shape=shape, instruction=instruction, text=text, **options), strict=strict)
    path = tmp_path / "reference.docx"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("shape", ["simple", "complex"])
def test_saved_ref_mismatch_retains_both_exact_word_addresses(strict, shape, tmp_path):
    original, sha = _save(tmp_path, strict=strict, shape=shape)
    record = reader.inspect(original, source_sha256=sha)
    assert record == {
        "schema": binding.SCHEMA, "checked": 1, "mismatches": 1, "authorship_confidence_contribution": 0,
        "findings": [{
            "field": {"body_index": 1, "child_index": 1,
                      "range": [len(FIELD_PREFIX), len(FIELD_PREFIX + "Café Phase A")], "text": "Café Phase A"},
            "target": {"body_index": 0, "child_index": 1,
                       "range": [len(SOURCE_PREFIX), len(SOURCE_PREFIX + "Café Phase C")], "text": "Café Phase C"},
            "record_shape": shape,
        }],
    }
    source = {"method": "python-docx", binding.COMMITMENT: canonical_hash(record)}
    document = {"source": source, binding.KEY: record}
    binding.verify_record(document, reader.inspect(original, source_sha256=sha), commitment=canonical_hash(record))
    with pytest.raises(ValueError, match="differs from the cross-reference source"):
        reader.inspect(original, source_sha256="0" * 64)

    changed = copy.deepcopy(document)
    changed[binding.KEY]["findings"][0]["target"]["range"] = [index + 1 for index in
                                                              record["findings"][0]["target"]["range"]]
    changed["source"][binding.COMMITMENT] = canonical_hash(changed[binding.KEY])
    binding.validate(changed)
    with pytest.raises(ValueError, match="differs from the original inspection"):
        binding.verify_record(changed, record, commitment=canonical_hash(record))
    with pytest.raises(ValueError, match="source commitment"):
        binding.validate({"source": source})
    binding.verify_record({"source": {"method": "python-docx"}}, None, commitment=None)


@pytest.mark.parametrize("options", [
    {"text": "Café  Phase C"},
    {"locked": True},
    {"shape": "complex", "locked": True},
    {"hidden": True},
    {"ambiguous": True},
    {"instruction": "REF MissingBookmark"},
    {"instruction": r"REF SourceLabel \* UPPER"},
    {"instruction": r"REF SourceLabel \p"},
    {"instruction": "PAGEREF SourceLabel"},
])
def test_ordinary_or_ambiguous_saved_ref_emits_no_claim(options, tmp_path):
    original, sha = _save(tmp_path, **options)
    assert reader.inspect(original, source_sha256=sha) is None
