import ast
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from scripts import review_image_jpeg_aggregate_v207 as review


def metric(total, truth, selected, overlap):
    union = truth + selected - overlap
    return {
        "pixels": total,
        "reference_positive_pixels": truth,
        "predicted_positive_pixels": selected,
        "intersection": overlap,
        "false_positive_pixels": selected - overlap,
        "false_negative_pixels": truth - overlap,
        "iou": overlap / union if union else 1.0,
        "pixel_f1": 2 * overlap / (truth + selected) if truth + selected else 1.0,
        "precision": overlap / selected if selected else None,
        "recall": overlap / truth if truth else None,
        "flagged_area_fraction": selected / total,
    }


def row(n, component, kind, chosen, footprint=0):
    return {
        "n": n,
        "component": component,
        "quality": 75,
        "kind": kind,
        "changed_pixels": footprint,
        "methods": {
            name: {
                "ai": metric(20, footprint, selected, overlap)
                if kind == "ai"
                else metric(20, 0, selected, 0),
                "ordinary": metric(20, footprint, selected, overlap)
                if kind in review.KINDS[2:] and footprint
                else None,
                "mask_sha256": "a" * 64,
            }
            for name, (selected, overlap) in zip(review.METHODS, chosen, strict=True)
        },
    }


def test_exact_public_freeze_and_no_scoring_imports():
    freeze = review.verify_public()
    assert len(freeze["code_sha256"]) == 23
    code = ast.parse(Path(review.__file__).read_bytes())
    modules = {node.module for node in ast.walk(code) if isinstance(node, ast.ImportFrom)}
    modules.update(
        alias.name for node in ast.walk(code) if isinstance(node, ast.Import) for alias in node.names
    )
    assert all(not name.startswith(("scripts", "app", "torch", "PIL", "cv2")) for name in modules)


def test_rational_metrics_check_threshold_ties_and_ordinary_false_area():
    tie = review.counts(metric(20, 6, 7, 3), 20, 6)
    assert tie.iou == Fraction(3, 10) and tie.recall == Fraction(1, 2) and tie.useful
    assert not review.counts(metric(20, 6, 8, 3), 20, 6).useful
    case = row(0, "a", "copy_move", ((7, 2), (8, 4), (4, 2)), 6)
    ai, ordinary = review.derive_row(case)
    assert ai["current"].area == Fraction(7, 20) and ai["current"].truth == 0
    assert ordinary["current"].iou == Fraction(2, 11)
    case["methods"]["current"]["ai"]["flagged_area_fraction"] = 0.5
    with pytest.raises(ValueError, match="flagged_area_fraction"):
        review.derive_row(case)


def test_impossible_shared_overlap_is_rejected_in_truth_and_ordinary_strata():
    for kind in ("ai", "tone"):
        impossible = row(0, "a", kind, ((4, 3), (4, 3), (1, 1)), 4)
        with pytest.raises(ValueError, match="truth stratum"):
            review.derive_row(impossible)
    empty = row(0, "a", "blur", ((0, 0), (0, 0), (0, 0)))
    assert review.derive_row(empty)[1] == {}
    empty["methods"]["shared"]["ordinary"] = metric(20, 0, 0, 0)
    with pytest.raises(ValueError, match="changed ordinary controls"):
        review.derive_row(empty)


def test_unequal_group_sizes_and_independent_bootstrap_formula():
    groups = ["a", "a", "b"]
    rows = []
    for n, group in enumerate(groups):
        for kind in review.KINDS:
            zero = (0, 0)
            if kind == "ai":
                new = (4, 4) if n in (0, 2) else zero
                rows.append(row(n, group, kind, (zero, new, zero), 4))
            elif kind == "copy_move":
                new = (5, 2) if n == 0 else zero
                rows.append(row(n, group, kind, (zero, new, zero), 2 if n < 2 else 0))
            else:
                rows.append(row(n, group, kind, (zero, zero, zero)))
    weights = review.bootstrap_weights(2)
    assert weights.shape == (10_000, 2) and np.all(weights.sum(axis=1) == 2)
    summary = review.recompute_quality(rows, groups, 75, weights)["methods"]["sensitive"]
    assert summary["edited_photos_usefully_localized"] == 2
    assert summary["group_mean_useful_fraction"] == Fraction(3, 4)
    assert summary["paired_against_current"]["edited_iou"] == {
        "difference": Fraction(3, 4),
        "pointwise_95pct_interval": [0.5, 1.0],
    }
    ordinary = summary["controls"]["copy_move"]
    assert ordinary["flagged_groups"] == ordinary["ordinary_footprints_overlapped"] == 1
    assert ordinary["ordinary_footprints_with_changed_pixels"] == 2
    assert ordinary["mean_iou_with_ordinary_edit_if_changed"] == Fraction(1, 5)
    assert ordinary["group_mean_highlighted_area_fraction"] == Fraction(1, 16)
    with pytest.raises(ValueError, match="incomplete"):
        review.recompute_quality(rows[:-1], groups, 75, weights)


def test_multiplicity_bootstrap_matches_gathered_means_on_48_authored_groups():
    current = [Fraction(n % 19, 23) for n in range(48)]
    candidate = [Fraction((n * 13 + 1) % 43, 47) for n in range(48)]
    weights = review.bootstrap_weights(48)
    independent = review.paired_interval(candidate, current, weights)
    generator = np.random.default_rng(20261108)
    draws = generator.integers(0, 48, size=(10_000, 48))
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(current, dtype=np.float64)
    direct = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
    assert independent["pointwise_95pct_interval"] == pytest.approx(direct.tolist(), abs=2e-12, rel=0)


def write_json(path, value):
    path.write_text(json.dumps(value))


def test_exact_phase_profile_and_row_commitments(tmp_path):
    trial = tmp_path / "trial"
    name = "readout-q90-11"
    output, profile = trial / name, trial / f"{name}-profile"
    output.mkdir(parents=True)
    profile.mkdir()
    resources = profile / "resources.jsonl"
    resources.write_text(
        json.dumps(
            {
                "elapsed_s": 0.0,
                "tree_rss_bytes": 0,
                "ram_available_bytes": review.LIMITS["start_ram"],
                "disk_free_bytes": review.LIMITS["start_disk"],
                "raw_disk_free_bytes": review.LIMITS["start_disk"],
                "violations": [],
            }
        )
        + "\n"
    )
    monitor = "d" * 64
    write_json(
        profile / "profile.json",
        {
            "schema": "chipmunk.image-jpeg-diagnostic-profile/189",
            "worker": review.MODULE,
            "command": "readout",
            "worker_arguments": ["readout", str(trial), review.FREEZE_SHA, "90", "11"],
            "monitor_sha256": monitor,
            "limits": review.LIMITS,
            "exit_code": 0,
            "failure": None,
            "duration_seconds": 1.0,
            "resource_sha256": review.sha(resources),
            "sampled_tree_rss_lower_bound_bytes": 0,
        },
    )
    journal = output / "rows.jsonl"
    support = review.phase_support(11)
    journal.write_text("".join(json.dumps({"n": n, "kind": kind}) + "\n" for n, kind in support))
    write_json(
        output / "completed.json",
        {
            "freeze_sha256": review.FREEZE_SHA,
            "phase": 11,
            "quality": 90,
            "views": 20,
            "neural_calls": 0,
            "rows_sha256": review.sha(journal),
        },
    )
    profiles = {name: review.sha(profile / "profile.json")}
    found, rows, receipt = review.phase(trial, 90, 11, "readout", profiles, monitor)
    assert found == name and len(rows) == 20 and receipt["row_journal_sha256"] == review.sha(journal)
    journal.write_text(journal.read_text().replace('"kind": "ai"', '"kind": "original"', 1))
    with pytest.raises(ValueError, match="Journal digest"):
        review.phase(trial, 90, 11, "readout", profiles, monitor)


def test_compare_requires_all_fields_and_rejects_changes():
    review.compare({"measure": [0.3333333333333], "count": 1}, {"measure": [Fraction(1, 3)], "count": 1})
    with pytest.raises(ValueError, match="arithmetic differs"):
        review.compare({"measure": [0.32], "count": 1}, {"measure": [Fraction(1, 3)], "count": 1})
    with pytest.raises(ValueError, match="fields differ"):
        review.compare({"measure": [0.32]}, {"measure": [Fraction(1, 3)], "count": 1})
