from __future__ import annotations

from copy import deepcopy

from test_document_packets import _add_ratio
from test_document_packets import packet_factory as packet_factory

from app.core.canonical import canonical_hash
from app.evidence.document_packets import build_packet_set, packet_hash
from app.reports import engine, multimedia_evidence
from app.reports.specimens import specimen


def test_pdf_keeps_ratio_reduction_distinct_from_full_forward_context(packet_factory, monkeypatch):
    text = ("A longer opening sentence provides source context. Yes we can. "
            "A final sentence remains available as forward-only context.")
    db, store, analysis, result, decoded, _ = packet_factory(text=text, scores=(0.1, 0.9, 0.1))
    ratio_signal = _add_ratio(result, decoded)
    measured = ratio_signal["details"]["text_evidence"]["windows"][1]
    # The short focus borrows its preceding sentence for the ratio reduction.
    # The final sentence remains in the actual forward input, but its losses
    # must not be described as part of this particular passage's ratio.
    reduction_end = decoded["sentences"][1]["end"]
    measured.update(scored_spans=[{"start": 0, "end": reduction_end}],
                    scored_token_indices=list(range(reduction_end)), tokens_scored=reduction_end,
                    neighbor_expansion=True,
                    token_coverage={"tokens": reduction_end, "covered_tokens": reduction_end, "token_windows": 1})
    analysis.result_hash = canonical_hash(result)
    evidence = build_packet_set(db, analysis, store, result)
    assert len(evidence["packets"]) == 1
    full_end = len(decoded["text"])
    assert reduction_end < full_end

    snapshot = specimen("document", "standard")
    report_result = deepcopy(snapshot.result)
    report_result["input"]["sha256"] = analysis.sha256
    report_result["manifest"]["sampling"] = deepcopy(result["manifest"]["sampling"])
    report_result["signals"] = deepcopy(result["signals"])
    evidence.update(analysis_id=snapshot.analysis.id, revision=snapshot.analysis.revision,
                    original_result_hash=canonical_hash(report_result))
    evidence["sha256"] = packet_hash(evidence)
    snapshot = snapshot.model_copy(update={"result": report_result, "evidence": evidence,
                                           "media": snapshot.media.model_copy(update={"sha256": analysis.sha256})})
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
    original = deepcopy(evidence)
    viewed = multimedia_evidence.view(snapshot, store=store)
    ratio = viewed["packets"][0]["windows"][1]
    assert ratio["scored_spans"] == [{"start": 0, "end": reduction_end}]
    assert ratio["tokens_scored"] == reduction_end and ratio["neighbor_expansion"] is True
    assert ratio["contexts"][0]["source_token_range"] == [0, full_end]
    assert ratio["contexts"][0]["scored_token_range"] == [0, full_end]
    assert ratio["supports_ai"] is False  # Raw ratio 1.4 is above the lower-is-AI boundary.
    assert "threshold" not in ratio and "ai_when" not in ratio
    html = " ".join(engine.render_html(snapshot, embed_fonts=False).split())
    assert f"This passage's ratio uses {reduction_end} selected token(s), intersecting source characters [0, {reduction_end})." in html
    assert "The reduction includes neighbouring sentences because the focus sentence was short." in html
    assert f"Actual token context [0, {full_end}); predicted tokens in that forward pass [0, {full_end})" in html
    assert "not necessarily all the tokens used for this passage's ratio" in html
    assert "No AI-like flag for this input." in html
    assert "It carries no passage-level authorship confidence." in html
    assert "The recorded checks disagree. One response was AI-like and another did not flag its input." in html
    assert "Inspection boundary:" not in html
    assert "Each check uses its own recorded operating threshold" not in html
    pdf = " ".join(engine.pdf_text(engine.render_pdf(snapshot).pdf).split())
    assert "No AI-like flag for this input." in pdf
    assert "Inspection boundary:" not in pdf
    assert "Each check uses its own recorded operating threshold" not in pdf
    assert snapshot.evidence == original  # Rendering does not rewrite the frozen evidence.
