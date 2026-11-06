import hashlib
import json

import pytest

from scripts.publish_audio_identity_results_v9 import (
    BASELINE_FIELDS,
    PROBE_FIELDS,
    REFERENCE_SHA,
    comparable,
    guard_only_ceiling,
    jsonl_digest,
    metadata,
    pick,
    privacy_check,
    public_summary,
    review,
)


def test_original_jsonl_hash_includes_header_order_unicode_and_full_float_precision():
    header = {"planned": 2, "schema": "fixture"}
    rows = [{"id": "a", "raw": 0.80000001}, {"id": "b", "raw": -0.0, "text": "é"}]
    original = "".join(json.dumps(x, allow_nan=False) + "\n" for x in (header, *rows)).encode()
    assert jsonl_digest(header, rows) == hashlib.sha256(original).hexdigest()
    rows[0]["raw"] = 0.8
    assert jsonl_digest(header, rows) != hashlib.sha256(original).hexdigest()


def test_public_source_metadata_uses_hashed_source_and_background_identity():
    original = {
        "base_id_sha256": "source-digest",
        "background_sha256": "background-digest",
        "id": "input-digest",
    }
    actual = metadata(original)
    assert actual["base_id"] == "source-digest" and actual["background_id"] == "background-digest"
    assert "base_id_sha256" not in actual and "base_id_sha256" in original


def test_explanation_and_probe_allowlists_omit_hidden_paths_transcripts_and_exception_text():
    original = {
        "probability_ai": 0.5,
        "status": "computed",
        "raw": 0.5,
        "source_path": "/private/input",
        "transcript": "private transcript",
        "error": "private error",
    }
    actual = {"baseline": pick(original, BASELINE_FIELDS), "probe": pick(original, PROBE_FIELDS)}
    text = json.dumps(actual)
    assert "/private/" not in text and "private transcript" not in text and "private error" not in text


def test_summary_hashes_provider_ids_without_changing_counts_or_original():
    original = {
        "schema": "wrapper",
        "groups": {"all": {"false_ai": 2}},
        "paired_differences": {"all": {"brier_difference": -0.02, "interpretation": "old scope"}},
        "strata": {"background_id": {"provider/raw-file": {"count": 9}, "None": {"count": 1}}},
    }
    rows = [
        {"background_id": "provider/raw-file", "background_sha256": "digest"},
        {"background_id": None, "background_sha256": None},
    ]
    actual = public_summary(original, rows)
    assert set(actual["strata"]["background_id"]) == {"digest", "None"}
    assert actual["groups"] == original["groups"]
    assert comparable(actual)["paired_differences"]["all"] == {"brier_difference": -0.02}
    assert "provider/raw-file" not in json.dumps(actual)
    assert "provider/raw-file" in original["strata"]["background_id"]


def test_claimed_reference_hashes_cannot_be_replaced_with_a_different_source_plan():
    body = {
        "schema": "chipmunk.audio-identity-public-development/9",
        "deployment_eligible": False,
        "references": {**REFERENCE_SHA, "audio-v9-identity-plan.json.gz": "0" * 64},
    }
    with pytest.raises(ValueError, match="reference commitments"):
        review(body, {}, {}, {})


def test_unknown_feature_fields_and_modified_receipts_are_refused_before_refitting():
    header = {
        k: None
        for k in (
            "schema",
            "lock_sha256",
            "planned",
            "weights",
            "runtime",
            "cache_sha256",
            "head_weight",
            "head_bias",
        )
    }
    body = {
        "schema": "chipmunk.audio-identity-public-development/9",
        "deployment_eligible": False,
        "references": REFERENCE_SHA,
        "representation_header": header,
        "features": [{"id": "fixture", "private_path": "/private/hidden"}],
        "original_receipt_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="Local paths|Unknown feature fields"):
        review(body, {}, {}, {})
    body["features"] = [{"id": "fixture"}]
    with pytest.raises(ValueError, match="receipt cannot be reconstructed"):
        review(body, {}, {}, {})


def test_nested_private_source_fields_and_local_paths_are_refused():
    with pytest.raises(ValueError, match="Private source fields"):
        privacy_check({"head": {"transcript": "do not publish"}})
    with pytest.raises(ValueError, match="Local paths"):
        privacy_check({"records": [{"notes": "/Users/owner/private-input.wav"}]})
    privacy_check({"id": "a" * 64, "embedding": [0.1, 0.2], "source_sha256": "b" * 64})


def test_guard_only_bound_cannot_recover_responses_below_the_fixed_point():
    rows = [
        {
            "label": 1,
            "generator": "fixture",
            "original_decision": "ai",
            "candidate_decision": "inconclusive",
            "candidate_p": p,
        }
        for p in (0.9, 0.79, 0.7, 0.6, None)
    ]
    result = guard_only_ceiling(rows)
    assert result["guard_only_cannot_pass"]
    assert result["families"][0]["fixed_probability_ai_ceiling_without_any_guard_or_stability_veto"] == 1
    assert result["families"][0]["frozen_minimum_to_retain"] == 3


def test_empty_ai_panel_cannot_vacuously_pass_retention_bound():
    with pytest.raises(ValueError, match="Generated-source"):
        guard_only_ceiling([])
