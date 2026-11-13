from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from app.bench import independent as review
from app.bench import independent_inputs as files
from app.bench import independent_locations as locations
from app.bench import independent_replication as replication


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


def test_coordinate_locations_count_shared_area_once_and_remove_private_positions():
    manifest, receipt = fixture()
    for case in manifest["cases"]:
        case["localization_truth"] = {"unit": review.LOCALIZATION_UNITS[case["modality"]], "source_units": 100,
                                      "target_units": 30 if case["label"] else 0}
    response = receipt("confirmation")
    by_id = {row["id"]: row for row in response["predictions"]}
    truth = {"schema": locations.TRUTH_SCHEMA, "split": "confirmation", "manifest_sha256": review.digest(manifest),
             "cases": []}
    for case in manifest["cases"]:
        if case["split"] != "confirmation":
            continue
        image = case["modality"] == "image"
        target = [[0, 0, 5, 5], [1, 0, 6, 5]] if image else [[5, 25], [15, 35]]
        selected = [[3, 2, 8, 7], [4, 2, 9, 7]] if image else [[20, 40], [30, 50]]
        truth["cases"].append({"id": case["id"], "input_sha256": case["input_sha256"],
                               "extent": [10, 10] if image else [100], "regions": target if case["label"] else []})
        by_id[case["id"]]["localization"] = {"status": "ok", "regions": selected}
    original = deepcopy(response)
    output = locations.count_locations(manifest, truth, response)
    assert original == response
    cases = {case["id"]: case for case in manifest["cases"]}
    for row in output["predictions"]:
        case = cases[row["id"]]
        overlap = (9 if case["modality"] == "image" else 15) if case["label"] else 0
        assert row["localization"] == {"status": "ok", "reported_units": 30, "overlap_units": overlap}

    with pytest.raises(ValueError, match="Every planned"):
        locations.count_locations(manifest, {**truth, "cases": truth["cases"][1:]}, response)
    with pytest.raises(ValueError, match="exact manifest"):
        locations.count_locations(manifest, {**truth, "manifest_sha256": "0" * 64}, response)
    image_id = next(case["id"] for case in manifest["cases"] if case["modality"] == "image" and case["split"] == "confirmation")
    by_id[image_id]["localization"]["regions"] = [[0, 0, 11, 10]]
    with pytest.raises(ValueError, match="within the source"):
        locations.count_locations(manifest, truth, response)
    by_id[image_id]["localization"] = {"status": "not_applicable", "reason": "No valid source map"}
    output = locations.count_locations(manifest, truth, response)
    assert next(row["localization"] for row in output["predictions"] if row["id"] == image_id) == by_id[image_id]["localization"]


def test_exact_image_masks_and_rectangles_use_the_same_native_geometry(tmp_path):
    manifest, receipt = fixture()
    case = next(row for row in manifest["cases"] if row["modality"] == "image"
                and row["split"] == "confirmation" and row["label"] == 1)
    case["localization_truth"] = {"unit": "pixels", "source_units": 15, "target_units": 4}
    response = receipt("confirmation")
    prediction = next(row for row in response["predictions"] if row["id"] == case["id"])

    def make_mask(name, pixels):
        body = b"P4\n5 3\n" + bytes(pixels)
        (tmp_path / name).write_bytes(body)
        return {"file": name, "sha256": hashlib.sha256(body).hexdigest()}

    target = make_mask("target.pbm", [0b10100_000, 0b01000_000, 0b00001_000])
    measured = make_mask("reported.pbm", [0b10001_000, 0b00000_000, 0b00001_000])
    source = {"id": case["id"], "input_sha256": case["input_sha256"], "extent": [5, 3], "mask": target}
    truth = {"schema": locations.TRUTH_SCHEMA, "manifest_sha256": review.digest(manifest),
             "split": "confirmation", "cases": [source]}

    def run(reported_units, overlap_units):
        counted = locations.count_locations(manifest, truth, response, mask_directory=tmp_path)
        measured_row = next(row for row in counted["predictions"] if row["id"] == case["id"])
        assert measured_row["localization"] == {"status": "ok", "reported_units": reported_units,
                                                "overlap_units": overlap_units}

    prediction["localization"] = {"status": "ok", "mask": measured}
    run(3, 2)
    source.pop("mask")
    source["regions"] = [[0, 0, 2, 2]]
    run(3, 1)
    source.pop("regions")
    source["mask"] = target
    prediction["localization"] = {"status": "ok", "regions": [[0, 0, 3, 2], [2, 0, 5, 1]]}
    run(8, 3)
    with pytest.raises(ValueError, match="supplied mask directory"):
        locations.count_locations(manifest, truth, response)
    invalid = make_mask("padding.pbm", [0b10100_001, 0b01000_000, 0b00001_000])
    for mask, error in (({**target, "file": "../target.pbm"}, "basename"),
                        ({**target, "sha256": "0" * 64}, "SHA-256 differ"),
                        (invalid, "padding bits"),
                        (measured, "target extent")):
        source["mask"] = mask
        with pytest.raises(ValueError, match=error):
            locations.count_locations(manifest, truth, response, mask_directory=tmp_path)
    source["mask"] = target
    source["extent"] = [3, 5]
    with pytest.raises(ValueError, match="dimensions"):
        locations.count_locations(manifest, truth, response, mask_directory=tmp_path)


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


def test_verify_input_and_parent_bytes_returns_counts_only(tmp_path):
    manifest, _ = fixture()
    root = tmp_path / "private"
    root.mkdir()
    rows = []
    for i, case in enumerate(manifest["cases"]):
        source = f"source-{i // 2}.dat"
        delivery = f"input-{i}.dat"
        for role, name in (("source", source), ("input", delivery)):
            body = name.encode()
            (root / name).write_bytes(body)
            case[f"{role}_sha256"] = hashlib.sha256(body).hexdigest()
        rows.append({"id": case["id"], "source": source, "input": delivery})
    index = {"schema": files.INDEX_SCHEMA, "manifest_sha256": review.digest(manifest), "cases": rows}
    assert files.verify_inputs(manifest, index, root) == {
        "schema": files.RECEIPT_SCHEMA, "manifest_sha256": review.digest(manifest),
        "verified_cases": len(rows), "verified_file_bindings": len(rows) * 2,
    }
    with pytest.raises(ValueError, match="Every planned"):
        files.verify_inputs(manifest, {**index, "cases": rows[1:]}, root)
    with pytest.raises(ValueError, match="exact evaluation manifest"):
        files.verify_inputs(manifest, {**index, "manifest_sha256": "0" * 64}, root)
    with pytest.raises(ValueError, match="duplicate"):
        files.verify_inputs(manifest, {**index, "cases": [*rows, rows[0]]}, root)
    outside = tmp_path / "outside.dat"
    outside.write_bytes((root / rows[0]["source"]).read_bytes())
    (root / "linked.dat").symlink_to(outside)
    for source in ("../outside.dat", str(outside), "linked.dat"):
        with pytest.raises(ValueError, match="supplied root"):
            files.verify_inputs(manifest, {**index, "cases": [{**rows[0], "source": source}, *rows[1:]]}, root)
    (root / rows[0]["source"]).write_bytes(b"changed parent")
    with pytest.raises(ValueError, match="differ from the manifest"):
        files.verify_inputs(manifest, index, root)
    (root / rows[0]["input"]).unlink()
    with pytest.raises(ValueError, match="Could not read"):
        files.verify_inputs(manifest, index, root)


def test_same_freeze_replication_reports_numerical_and_coverage_differences():
    manifest, receipt = fixture()
    for row in manifest["cases"]:
        row["localization_truth"] = {"unit": review.LOCALIZATION_UNITS[row["modality"]],
                                     "source_units": 200, "target_units": 20 if row["label"] else 0}
    dev = receipt("development")
    baseline = receipt("confirmation")
    for row in dev["predictions"] + baseline["predictions"]:
        row["localization"] = {"status": "ok", "reported_units": 0, "overlap_units": 0}
    lock = review.freeze(manifest, dev, thresholds=dict.fromkeys(review.MEDIA, .5))
    baseline["freeze_sha256"] = review.digest(lock)
    repeated = deepcopy(baseline)
    changed = next(r for r in repeated["predictions"] if r["id"] == "image/confirmation/1/clean")
    changed.update(response=.49, localization={"status": "ok", "reported_units": 10, "overlap_units": 4})
    not_assessed = next(r for r in repeated["predictions"] if r["id"] == "audio/confirmation/1/clean")
    not_assessed.update(response=None, status="inconclusive", reason="Scope", localization={"status": "not_applicable", "reason": "No clock"})
    result = replication.replicate(manifest, baseline, repeated, lock)
    lookup = {(r["modality"], r["condition"], r["label"]): r for r in result["strata"]}
    image = lookup["image", "clean", 1]
    assert (image["changed_score"], image["lost_flags"], image["changed_overlap_counts"], image["groups_with_differences"]) == (1, 1, 1, 1)
    audio = lookup["audio", "clean", 1]
    assert (audio["changed_status"], audio["changed_score"], audio["lost_flags"], audio["changed_location_status"]) == (1, 0, 1, 1)
    assert not any(r["groups_with_differences"] for r in replication.replicate(manifest, baseline, baseline, lock)["strata"])
    assert "image/confirmation" not in str(result) and "No clock" not in str(result)
    with pytest.raises(ValueError, match="exact freeze"):
        replication.replicate(manifest, baseline, {**repeated, "freeze_sha256": "0" * 64}, lock)


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


def test_model_free_receipt_keeps_its_declared_kind_after_freeze():
    manifest, receipt = fixture()
    development = receipt("development")
    confirmation = receipt("confirmation")
    for record in (development, confirmation):
        record.update(method_kind="model_free", weight_sha256=[])
    lock = review.freeze(manifest, development, thresholds=dict.fromkeys(review.MEDIA, .5))
    assert lock["method_kind"] == "model_free"
    assert len(review.summarize(manifest, confirmation, lock)["strata"]) == 8
    confirmation.pop("method_kind")
    with pytest.raises(ValueError, match="Pinned model weight"):
        review.summarize(manifest, confirmation, lock)


def test_confirmation_binds_the_exact_operating_points_when_supplied():
    manifest, receipt = fixture()
    lock = review.freeze(manifest, receipt("development"), thresholds=dict.fromkeys(review.MEDIA, .5))
    confirmation = receipt("confirmation")
    with pytest.raises(ValueError, match="exact freeze"):
        review.summarize(manifest, confirmation, lock, require_freeze_commitment=True)
    confirmation["freeze_sha256"] = review.digest(lock)
    assert review.summarize(manifest, confirmation, lock, require_freeze_commitment=True)["lock_sha256"] == review.digest(lock)
    modified = deepcopy(lock)
    modified["thresholds"]["image"] = .9
    with pytest.raises(ValueError, match="exact freeze"):
        review.compare(manifest, confirmation, lock, confirmation, modified)


def test_localization_includes_misses_and_control_regions_at_all_four_scales():
    manifest, receipt = fixture()
    for case in manifest["cases"]:
        case["localization_truth"] = {"unit": review.LOCALIZATION_UNITS[case["modality"]], "source_units": 100,
                                      "target_units": 20 if case["label"] else 0}
    development, confirmation = receipt("development"), receipt("confirmation")
    cases = {case["id"]: case for case in manifest["cases"]}
    for record in (development, confirmation):
        for outcome in record["predictions"]:
            case = cases[outcome["id"]]
            outcome["localization"] = {"status": "ok", "reported_units": 16, "overlap_units": 12 if case["label"] else 0}
            if case["label"] and case["condition"] == "processed":
                outcome["localization"] = {"status": "not_applicable", "reason": "No valid source mapping"}
    lock = review.freeze(manifest, development, thresholds=dict.fromkeys(review.MEDIA, .5))
    summary = review.summarize(manifest, confirmation, lock)
    for row in summary["strata"]:
        regions = row["localization"]
        assert regions["unit"] == review.LOCALIZATION_UNITS[row["modality"]]
        assert regions["control_inputs"] == regions["control_regions"] == regions["positive_inputs"] == 1
        expected = (0.5, 2 / 3, 0.6) if row["condition"] == "clean" else (0.0, 0.0, 0.0)
        assert (regions["mean_positive_iou"], regions["mean_positive_f1"], regions["mean_positive_recall"]) == expected
        assert row["ai_alerts"] == 1  # a ranked response and an unassessed location can coexist
        assert regions["status_counts"] == ({"ok": 2} if row["condition"] == "clean" else {"ok": 1, "not_applicable": 1})


@pytest.mark.parametrize("outcome", [None, {"status": "ok", "reported_units": 95, "overlap_units": 0},
                                     {"status": "ok", "reported_units": 12, "overlap_units": 13},
                                     {"status": "failed", "reported_units": 0, "reason": "No source map"}])
def test_impossible_or_missing_localization_is_rejected(outcome):
    manifest, receipt = fixture()
    target = next(case for case in manifest["cases"] if case["split"] == "confirmation" and case["label"] == 1)
    target["localization_truth"] = {"unit": review.LOCALIZATION_UNITS[target["modality"]], "source_units": 100,
                                    "target_units": 20}
    record = receipt("confirmation")
    row = next(row for row in record["predictions"] if row["id"] == target["id"])
    row["localization"] = outcome
    with pytest.raises(ValueError, match="localization"):
        review.audit_receipt(manifest, record, "confirmation")


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
    assert "paired_localization_differences" not in comparison
    for row in comparison["paired_differences"]:
        assert row["source_groups"] == 1  # Not two independent sources because of variants.
        assert row["group_bootstrap_95_percent"] is None
        expected = -1 if row["metric"] == "ai_alert_rate" else 0
        assert row["candidate_minus_baseline"] == expected


def test_paired_localization_averages_variants_and_counts_assessment_losses():
    manifest, receipt = fixture()
    edited = next(row for row in manifest["cases"] if (row["modality"], row["split"], row["condition"], row["label"])
                  == ("image", "confirmation", "clean", 1))
    edited["localization_truth"] = {"unit": "pixels", "source_units": 100, "target_units": 20}
    for suffix in ("jpeg", "resize"):
        variant = deepcopy(edited)
        variant.update(id=f"variant-{suffix}", input_sha256=review.digest({"variant": suffix}))
        manifest["cases"].append(variant)
    other = deepcopy(edited)
    other.update(id="other", group="other", lineage=["other"], source_sha256="a" * 64, input_sha256="b" * 64)
    manifest["cases"].append(other)
    planned = {edited["id"], "variant-jpeg", "variant-resize", "other"}

    points = dict.fromkeys(review.MEDIA, .5)
    old_lock = review.freeze(manifest, receipt("development"), thresholds=points)
    new_lock = review.freeze(manifest, receipt("development", "candidate"), thresholds=points)
    old, new = receipt("confirmation"), receipt("confirmation", "candidate")
    for result in (old, new):
        for row in result["predictions"]:
            if row["id"] in planned:
                row["localization"] = {"status": "ok", "reported_units": 0, "overlap_units": 0}
    for row in old["predictions"]:
        if row["id"] == "other":
            row["localization"].update(reported_units=20, overlap_units=20)
    for row in new["predictions"]:
        if row["id"] in {edited["id"], "variant-jpeg"}:
            row["localization"].update(reported_units=10, overlap_units=10)
        if row["id"] == "other":
            row["localization"] = {"status": "failed", "reason": "Lost source mapping"}
    result = review.compare(manifest, old, old_lock, new, new_lock, iterations=100)
    comparisons = {row["metric"]: row for row in result["paired_localization_differences"]}
    iou = comparisons["positive_iou"]
    assert (iou["planned"], iou["source_groups"]) == (4, 2)
    assert iou["candidate_minus_baseline"] == pytest.approx(-1 / 3)
    assert iou["case_transitions"] == {"increased": 2, "decreased": 1, "tied": 1}
    assert iou["group_bootstrap_95_percent"] == [-1, pytest.approx(1 / 3)]
    coverage = comparisons["localization_coverage"]
    assert coverage["candidate_minus_baseline"] == -.5
    assert coverage["case_transitions"] == {"increased": 0, "decreased": 1, "tied": 3}


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
