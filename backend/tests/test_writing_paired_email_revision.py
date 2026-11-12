from __future__ import annotations

from email.message import EmailMessage

from app.detectors.text.extract import extract, text_hash
from scripts.writing_paired_email_revision import assess, differences, digest, pair, tokens


def test_changed_extents_preserve_unicode_coordinates_and_exact_changed_tokens():
    before = "🚀 The local draft lists small blue tiles."
    after = "🚀 The local draft lists large red tiles."
    found, = differences(tokens(before, [(0, len(before))]), tokens(after, [(0, len(after))]))
    a, b = found["earlier"]
    c, d = found["later"]
    assert before[a:b] == "small blue"
    assert after[c:d] == "large red"
    assert [after[x:y] for x, y in found["later_changed_tokens"]] == ["large", "red"]


def _fixture(tmp_path, name, body):
    path = tmp_path / f"{name}.eml"
    message = EmailMessage()
    message.set_content(body)
    message.add_alternative("<p>Independent unselected input.</p>", subtype="html")
    path.write_bytes(message.as_bytes())
    doc = extract(path, ".eml")
    return {"file": path.name, "sha256": digest(path), "text_sha256": text_hash(doc.text),
            "roles": [[r["quote_role"], r["start"], r["end"]] for r in doc.source_map]}


def test_quote_change_gets_exact_offsets_without_triggering_body_comparison(tmp_path):
    body = "Our team will send the signed records at noon."
    early = _fixture(tmp_path, "early", body + "\n\n> Please wait.")
    late = _fixture(tmp_path, "late", body + "\n\n> Please proceed.")
    record, inputs, text = pair(tmp_path, {"case": "quote", "group": "g", "earlier": early, "later": late})
    assert not record["body"]
    change, = record["quoted"]
    a, b = change["later"]
    assert text[a:b] == "proceed"
    assert any("Please proceed" in value for value in inputs.values())


def test_two_directions_are_required_for_paired_body_evidence(tmp_path):
    early = _fixture(tmp_path, "early", "The work was done. We received the paper form. It arrived today.")
    late = _fixture(tmp_path, "late", "The work was done. We received the digital form. It arrived today.")
    row = {"case": "body", "group": "g", "earlier": early, "later": late, "body_truth": [], "quote_truth": []}
    record, inputs, text = pair(tmp_path, row)
    diff, = record["body"]
    comparison = diff["comparison"]
    old, new = (comparison[key]["sha256"] for key in ("earlier", "later"))
    assert inputs[old] == "The work was done. We received the paper form. It arrived today."
    assert inputs[new] == "The work was done. We received the digital form. It arrived today."
    first = {key: {"score": .04} for key in inputs}
    first[new] = {"score": .16}
    second = {old: {"score": .25}, new: {"score": .28}}
    assert not assess(row, record, text, first, second)["paired_highlight"]
    second[new] = {"score": .5}
    assert assess(row, record, text, first, second)["paired_highlight"]
