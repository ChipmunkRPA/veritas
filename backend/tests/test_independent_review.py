from __future__ import annotations

from copy import deepcopy

import pytest

from app.bench import independent as review


def fixture():
    cases = []
    for modality in sorted(review.MEDIA):
        for split in ("development", "confirmation"):
            for label in (0, 1):
                group = f"{modality}/{split}/{label}"
                source = review.digest({"source": group})
                for condition in ("clean", "processed"):
                    ident = f"{group}/{condition}"
                    cases.append({"id": ident, "group": group, "modality": modality, "split": split,
                                  "label": label, "family": "synthetic-test-fixture", "condition": condition,
                                  "input_sha256": review.digest({"input": ident}), "source_sha256": source,
                                  "lineage": [group], "license": {"id": "test fixture",
                                      "evaluation_permitted": True, "redistribution_permitted": False}})
    manifest = {"schema": review.SCHEMA, "cases": cases}

    def receipt(split, system="baseline"):
        return {"manifest_sha256": review.digest(manifest), "split": split, "system": system,
                "implementation_sha256": review.digest({"implementation": system}),
                "weight_sha256": [review.digest({"weights": system})], "score_kind": "raw",
                "response_units": "synthetic test response, not confidence",
                "predictions": [{"id": r["id"], "input_sha256": r["input_sha256"], "status": "ok",
                                 "response": .8 if r["label"] else .2}
                                for r in cases if r["split"] == split]}
    return manifest, receipt


def test_four_media_fixture_audits_and_summarizes_without_loading_models():
    manifest, receipt = fixture()
    original = deepcopy(manifest)
    assert review.audit_manifest(manifest)["cases"] == 32
    lock = review.freeze(manifest, receipt("development"), thresholds=dict.fromkeys(review.MEDIA, .5))
    summary = review.summarize(manifest, receipt("confirmation"), lock)
    assert len(summary["strata"]) == 8
    for row in summary["strata"]:
        assert (row["planned"], row["scored"], row["false_alerts"], row["ai_alerts"]) == (2, 2, 0, 1)
        assert row["assessable_auc"] == 1
        assert row["assessable_brier"] is None  # Raw responses are not calibrated probabilities.
    assert manifest == original
    # Summary deliberately excludes source paths, source URLs, individual IDs and media bytes.
    assert "predictions" not in summary and "cases" not in summary


@pytest.mark.parametrize("field", ["group", "source_sha256", "input_sha256", "lineage"])
def test_transformed_source_or_known_lineage_cannot_cross_splits(field):
    manifest, _ = fixture()
    before = manifest["cases"][0]
    after = next(r for r in manifest["cases"] if r["split"] == "confirmation")
    after[field] = before[field]
    with pytest.raises(ValueError, match="leakage"):
        review.audit_manifest(manifest)


@pytest.mark.parametrize("field,value", [("label", True), ("label", .5), ("modality", "unknown"),
                                         ("split", "test"), ("source_sha256", "unknown")])
def test_manifest_rejects_ambiguous_labels_scopes_or_source_identity(field, value):
    manifest, _ = fixture()
    manifest["cases"][0][field] = value
    with pytest.raises(ValueError):
        review.audit_manifest(manifest)


def test_rights_are_explicit_and_missing_license_permission_is_rejected():
    manifest, _ = fixture()
    manifest["cases"][0]["license"]["evaluation_permitted"] = False
    with pytest.raises(ValueError, match="permission"):
        review.audit_manifest(manifest)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong-input"])
def test_predictions_cannot_cherry_pick_cases_or_change_inputs(mutation):
    manifest, receipt = fixture()
    rows = receipt("confirmation")
    if mutation == "missing":
        rows["predictions"].pop()
    elif mutation == "duplicate":
        rows["predictions"][-1] = rows["predictions"][0]
    else:
        rows["predictions"][0]["input_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        review.audit_receipt(manifest, rows, "confirmation")


@pytest.mark.parametrize("response", [float("nan"), float("inf"), True, None])
def test_scored_response_must_be_finite_and_bounded(response):
    manifest, receipt = fixture()
    rows = receipt("confirmation")
    rows["predictions"][0]["response"] = response
    with pytest.raises(ValueError, match="finite"):
        review.audit_receipt(manifest, rows, "confirmation")


def test_missing_scores_stay_in_denominator_and_do_not_become_original_verdicts():
    manifest, receipt = fixture()
    lock = review.freeze(manifest, receipt("development"), thresholds=dict.fromkeys(review.MEDIA, .5))
    result = receipt("confirmation")
    for r in result["predictions"]:
        r.update(status="inconclusive", response=None, reason="Unsupported conditions")
    summary = review.summarize(manifest, result, lock)
    for row in summary["strata"]:
        assert row["planned"] == row["inconclusive"] == row["unassessed"] == 2
        assert row["scored"] == row["ai_alerts"] == row["false_alerts"] == 0
        assert row["ai_without_alert"] == 1
        assert row["assessable_auc"] is None


def test_failure_cannot_contain_a_negative_authenticity_response():
    manifest, receipt = fixture()
    rows = receipt("confirmation")
    rows["predictions"][0].update(status="failed", response=0., reason="Decode failure")
    with pytest.raises(ValueError, match="cannot carry"):
        review.audit_receipt(manifest, rows, "confirmation")


@pytest.mark.parametrize("key,value", [("implementation_sha256", "0"*64), ("weight_sha256", ["0"*64]),
                                       ("system", "changed"), ("score_kind", "calibrated")])
def test_confirmation_cannot_change_frozen_system(key, value):
    manifest, receipt = fixture()
    lock = review.freeze(manifest, receipt("development"), thresholds=dict.fromkeys(review.MEDIA, .5))
    rows = receipt("confirmation")
    rows[key] = value
    with pytest.raises(ValueError):
        review.summarize(manifest, rows, lock)


def test_calibration_digest_is_required_and_cannot_change_at_confirmation():
    manifest, receipt = fixture()
    dev = receipt("development")
    dev["score_kind"] = "calibrated"
    with pytest.raises(ValueError, match="calibration"):
        review.freeze(manifest, dev, thresholds=dict.fromkeys(review.MEDIA, .5))
    dev["calibration_sha256"] = "a"*64
    lock = review.freeze(manifest, dev, thresholds=dict.fromkeys(review.MEDIA, .5))
    test = receipt("confirmation")
    test.update(score_kind="calibrated", calibration_sha256="a"*64)
    assert review.summarize(manifest, test, lock)["strata"][0]["assessable_brier"] == pytest.approx(.04)
    test["calibration_sha256"] = "b"*64
    with pytest.raises(ValueError, match="calibration_sha256"):
        review.summarize(manifest, test, lock)


def test_fixed_comparison_preserves_all_cases_and_does_not_inflate_source_groups():
    manifest, receipt = fixture()
    points = dict.fromkeys(review.MEDIA, .5)
    old_lock = review.freeze(manifest, receipt("development"), thresholds=points)
    new_lock = review.freeze(manifest, receipt("development", "candidate"), thresholds=points)
    old, new = receipt("confirmation"), receipt("confirmation", "candidate")
    for row in new["predictions"]:
        if row["response"] == .8:
            row.update(status="not_applicable", response=None, reason="No usable input")
    comparison = review.compare(manifest, old, old_lock, new, new_lock, iterations=100)
    for row in comparison["paired_differences"]:
        assert row["source_groups"] == 1  # Not two independent sources because of variants.
        assert row["group_bootstrap_95_percent"] is None
        expected = -1 if row["metric"] == "ai_alert_rate" else 0
        assert row["candidate_minus_baseline"] == expected


def test_raw_response_auc_handles_ties_and_omits_no_cases_from_counts():
    manifest, receipt = fixture()
    dev, test = receipt("development"), receipt("confirmation")
    for row in test["predictions"]:
        row["response"] = .5
    lock = review.freeze(manifest, dev, thresholds=dict.fromkeys(review.MEDIA, .5))
    assert all(r["assessable_auc"] == .5 for r in review.summarize(manifest, test, lock)["strata"])


def test_freeze_detaches_mutable_system_and_threshold_fields():
    manifest, receipt = fixture()
    dev = receipt("development")
    points = dict.fromkeys(review.MEDIA, .5)
    lock = review.freeze(manifest, dev, thresholds=points)
    original = deepcopy(lock)
    points["audio"] = .1
    dev["weight_sha256"].append("b" * 64)
    assert lock == original


def test_rank_auc_matches_explicit_pairs_with_ties():
    rows = [{"label": label, "response": response}
            for label, response in [(0, .1), (1, .1), (0, .3), (1, .6), (1, .6), (0, .9)]]
    pairs = [(a, b) for a in rows if a["label"] for b in rows if not b["label"]]
    expected = sum((a["response"] > b["response"]) + .5 * (a["response"] == b["response"])
                   for a, b in pairs) / len(pairs)
    assert review._auc(rows) == expected


def test_signed_raw_responses_keep_their_units_and_are_not_clipped_into_probabilities():
    manifest, receipt = fixture()
    dev, test = receipt("development"), receipt("confirmation")
    for response in (dev, test):
        response["response_units"] = "signed inspection response"
        for row in response["predictions"]:
            row["response"] = -1. if row["response"] == .8 else -3.
    lock = review.freeze(manifest, dev, thresholds=dict.fromkeys(review.MEDIA, -2.))
    values = review.summarize(manifest, test, lock)["strata"]
    assert all(r["ai_alerts"] == 1 and r["false_alerts"] == 0 and r["assessable_auc"] == 1 for r in values)
    test["response_units"] = "different measurement"
    with pytest.raises(ValueError, match="response_units"):
        review.summarize(manifest, test, lock)


@pytest.mark.parametrize("value", [-1., 1.01])
def test_calibrated_responses_cannot_use_raw_unbounded_units(value):
    manifest, receipt = fixture()
    test = receipt("confirmation")
    test.update(score_kind="calibrated", calibration_sha256="a"*64)
    test["predictions"][0]["response"] = value
    with pytest.raises(ValueError, match="calibrated estimates"):
        review.audit_receipt(manifest, test, "confirmation")
