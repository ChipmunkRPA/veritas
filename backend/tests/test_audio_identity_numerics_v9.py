import copy
import json

import numpy as np
import pytest

from scripts import audio_multisource_v8 as method
from scripts import review_audio_identity_numerics_v9 as numerical
from scripts.review_audio_identity_numerics_v9 import decisions, guards


def accepted_guards():
    return {"0": [{"maximum": 0.2}] * 9, "1": [{"maximum": 0.2}] * 9}


@pytest.mark.parametrize(
    "p,band,raw,band_raw",
    [
        (0.1, 0.11, 0.1, 0.11),
        (0.1, None, 0.1, None),
        (0.1, 0.11, 0.1, 0.8),
        (0.1, 0.8, 0.1, 0.11),
        (0.9, None, 0.9, None),
        (None, None, None, None),
    ],
)
def test_separate_dual_stability_reconstruction_matches_frozen_rules(p, band, raw, band_raw):
    assert decisions(p, band, raw, band_raw, accepted_guards()) == method.final(
        p, band, raw, band_raw, accepted_guards()
    )


def test_missing_probe_cannot_establish_an_original_verdict():
    result = decisions(0.1, None, 0.1, None, accepted_guards())
    assert result["decision"] == "inconclusive" and result["original_stability_veto"]


def test_group_and_class_guard_ranks_preserve_unassessed_and_paired_sources():
    head = {
        "transform": {"unit_head": [1, 0], "center": [0, 0], "directions": [[0, 1]], "scale": [1]},
        "coefficients": [1, 0, 0],
    }
    rows = [
        {
            "role": "cal",
            "group": str(i),
            "label": c,
            "raw": 0.1 if c == 0 else 0.9,
            "embedding": np.zeros(2).tolist(),
        }
        for c in (0, 1)
        for i in range(9)
    ]
    rows.append({**rows[0], "raw": None, "embedding": None})
    result = guards(rows, head)
    assert result["0"][0]["maximum"] is None and result["0"][0]["rows"] == 2
    assert result["1"][0]["maximum"] is not None
    assert len(result["0"]) == len(result["1"]) == 9
    assert result == method.calibrate(rows, head)
    rows[0]["role"] = "development"
    with pytest.raises(ValueError, match="separate calibration"):
        guards(rows, head)


def test_complete_summary_keeps_v9_wrapper_and_checks_all_statistical_fields(tmp_path, monkeypatch):
    lock, receipt, head_path, responses, summary_path = [
        tmp_path / name
        for name in ("lock.json", "features.jsonl", "head.json", "responses.jsonl", "summary.json")
    ]
    lock.write_text("{}")
    receipt.write_text("retained features\n")
    fitted = {"fixture": True}
    fit_record = {
        "lock_sha256": numerical.digest(lock),
        "receipt_sha256": numerical.digest(receipt),
        "fitted": fitted,
    }
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(json.dumps(fit_record))
    head = {**fit_record, "fit_sha256": numerical.digest(fit_path)}
    head_path.write_text(json.dumps(head))
    response_header = {
        "lock_sha256": numerical.digest(lock),
        "receipt_sha256": numerical.digest(receipt),
        "head_sha256": numerical.digest(head_path),
    }
    responses.write_text(json.dumps(response_header) + "\n")
    rebuilt = {
        "schema": "chipmunk.audio-multisource-comparison/8",
        "groups": {"all": {"count": 1449}},
        "gates": {"complete_cases": True},
        "advance_to_fresh_confirmation": False,
    }
    summary = {
        **copy.deepcopy(rebuilt),
        "schema": "chipmunk.audio-identity-development-comparison/9",
        **response_header,
        "responses_sha256": numerical.digest(responses),
    }
    summary_path.write_text(json.dumps(summary))
    monkeypatch.setattr(numerical.execution, "check", lambda _: ({"serving_bundle": {}}, {}))
    monkeypatch.setattr(numerical.execution, "feature_rows", lambda *_: ({}, {}, []))
    monkeypatch.setattr(numerical, "reconstruct", lambda *a, **kw: (rebuilt, {"passed": True}))
    output = tmp_path / "review.json"
    numerical.review(lock, receipt, head_path, responses, summary_path, output, False)
    assert json.loads(output.read_text())["passed"]
    summary["groups"]["all"]["count"] = 1448
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="statistical summary differs"):
        numerical.review(lock, receipt, head_path, responses, summary_path, tmp_path / "altered.json", False)
