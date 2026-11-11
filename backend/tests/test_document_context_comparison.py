from copy import deepcopy

import pytest
from test_document_packets import packet_factory as packet_factory

from app.core.canonical import canonical_hash
from app.evidence.document_context import build, validate
from app.evidence.document_packets import _source_anchors, build_packet_set, packet_hash, validate_packet_set
from app.reports.multimedia_evidence import verify_assets
from app.storage.objects import ObjectIntegrityError


def test_actual_source_contexts_keep_extremes_without_an_authorship_probability(packet_factory):
    db, store, analysis, result, decoded, _ = packet_factory()
    before = deepcopy(result)
    doc = build_packet_set(db, analysis, store, result)
    packet = doc["packets"][0]
    block = packet["context_comparison"]
    assert doc["version"] == "document-evidence-5"
    assert block["status"] == "recorded" and block["authorship_confidence_pct"] is None
    assert block["affects_overall_confidence"] is False
    check = block["checks"][0]
    assert check["distinct_inputs"] == 3 and check["omitted_inputs"] == 1
    assert (check["lowest_response"], check["highest_response"], check["response_spread"]) == (0.1, 0.9, 0.8)
    assert [w["score"] for w in check["inputs"]] == [0.1, 0.9]
    validate(block, focus=packet["focus"], chars=len(decoded["text"]), text=decoded["text"])
    verify_assets(doc, store, result=result)
    assert result == before
    assert "PRIVATE MODEL" not in str(block) and "txt.desklib" not in str(block)


def test_equal_responses_are_recorded_without_establishing_originality(packet_factory):
    _, _, _, result, decoded, _ = packet_factory(scores=(0.9, 0.9, 0.9))
    focus = decoded["sentences"][1]
    block = build(result, decoded["text"], focus)
    check = block["checks"][0]
    assert block["status"] == "recorded" and check["response_spread"] == 0
    assert len({w["input_sha256"] for w in check["inputs"]}) == 2
    assert block["authorship_confidence_pct"] is None


@pytest.mark.parametrize("problem", ["single", "skipped", "ratio"])
def test_missing_compatible_contexts_remain_unassessed(packet_factory, problem):
    _, _, _, result, decoded, _ = packet_factory()
    if problem == "single":
        result["signals"][0]["details"]["text_evidence"]["windows"] = result["signals"][0]["details"][
            "text_evidence"
        ]["windows"][1:2]
    elif problem == "skipped":
        result["signals"][0]["status"] = "skipped"
    else:
        for w in result["signals"][0]["details"]["text_evidence"]["windows"]:
            w["score_kind"] = "perplexity_ratio"
            w["score"] = 1.3
    block = build(result, decoded["text"], decoded["sentences"][1])
    assert block["status"] == "not_assessed" and not block["checks"]
    assert block["reason"] and block["authorship_confidence_pct"] is None


def test_duplicate_input_is_not_counted_as_independent_context(packet_factory):
    _, _, _, result, decoded, _ = packet_factory()
    windows = result["signals"][0]["details"]["text_evidence"]["windows"]
    windows.append(deepcopy(windows[1]))
    block = build(result, decoded["text"], decoded["sentences"][1])
    assert block["checks"][0]["distinct_inputs"] == 3
    windows[-1]["score"] = 0.7
    with pytest.raises(ValueError, match="inconsistent responses"):
        build(result, decoded["text"], decoded["sentences"][1])


@pytest.mark.parametrize("change", ["confidence", "overall", "source", "input", "range", "extra"])
def test_fabricated_measurements_or_scope_are_refused(packet_factory, change):
    db, store, analysis, result, decoded, _ = packet_factory()
    doc = build_packet_set(db, analysis, store, result)
    packet = doc["packets"][0]
    block = packet["context_comparison"]
    if change == "confidence":
        block["authorship_confidence_pct"] = 90
    elif change == "overall":
        block["affects_overall_confidence"] = True
    elif change == "source":
        block["checks"][0]["inputs"][0]["spans"][0]["start"] += 1
    elif change == "input":
        block["checks"][0]["inputs"][0]["text"] += "x"
    elif change == "range":
        block["checks"][0]["response_spread"] = 0.1
    else:
        block["confidence_pct"] = 99
    with pytest.raises(ValueError):
        validate(block, focus=packet["focus"], chars=len(decoded["text"]), text=decoded["text"])


def test_rehashed_consistent_but_fabricated_score_still_fails_saved_result_binding(packet_factory):
    db, store, analysis, result, _, _ = packet_factory()
    doc = build_packet_set(db, analysis, store, result)
    changed = deepcopy(doc)
    check = changed["packets"][0]["context_comparison"]["checks"][0]
    check["highest_response"] = check["inputs"][1]["score"] = 0.95
    check["response_spread"] = 0.85
    changed["sha256"] = packet_hash(changed)
    with pytest.raises(ObjectIntegrityError, match="actual published classifier inputs"):
        verify_assets(changed, store, result=result)


@pytest.mark.parametrize("version", ["document-evidence-2", "document-evidence-3", "document-evidence-4"])
def test_historical_packet_keeps_its_supported_format_without_new_fields(packet_factory, version):
    db, store, analysis, result, decoded, _ = packet_factory()
    doc = build_packet_set(db, analysis, store, result)
    doc["version"] = version
    del doc["repeated_passages"]
    for packet in doc["packets"]:
        if version != "document-evidence-4":
            del packet["inspection_scope"]
            focus = packet["focus"]
            packet["anchors"] = _source_anchors(decoded["source_map"], focus["start"], focus["end"], len(decoded["text"]))
        if version == "document-evidence-2":
            del packet["context_comparison"]
    doc["sha256"] = packet_hash(doc)
    validate_packet_set(
        doc,
        analysis_id=analysis.id,
        revision=analysis.revision,
        original_result_hash=canonical_hash(result),
        source_sha256=analysis.sha256,
    )
    verify_assets(doc, store, result=result)
