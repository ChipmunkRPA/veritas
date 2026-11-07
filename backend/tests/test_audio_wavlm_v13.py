"""Research boundary tests: source clocks, missingness and independent roles."""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest
import torch

from scripts import audio_wavlm_features_v13 as features
from scripts import audio_wavlm_response_v13 as method
from scripts import run_audio_wavlm_v13 as run


def states():
    values = [torch.zeros((1, 3, 768)) for _ in range(13)]
    for layer, numbers in ((4, (1, 3, 100)), (8, (10, 14, 900)), (12, (0, 0, 123))):
        values[layer] = torch.tensor(numbers, dtype=torch.float32)[None, :, None].expand(1, 3, 768)
    return values


def test_population_pooling_preserves_layer_order_and_excludes_padding():
    result = features.pool(states(), torch.tensor([[1, 1, 0]]))
    expected = np.repeat(np.array([2, 1, 12, 2, 0, 0], np.float32), 768)
    assert result["wavlm"] == expected.tolist()
    assert result["wavlm_sha256"] == hashlib.sha256(expected.astype("<f4").tobytes()).hexdigest()
    assert result["feature_frames"] == 3 and result["valid_feature_frames"] == 2


@pytest.mark.parametrize("mask", [torch.zeros((1, 3)), torch.tensor([[1, 2, 0]]), torch.ones(3), torch.ones((2, 3))])
def test_invalid_masks_cannot_supply_evidence(mask):
    with pytest.raises(ValueError):
        features.pool(states(), mask)


@pytest.mark.parametrize("fault", ["missing_state", "wrong_clock", "wrong_width", "nonfinite"])
def test_selected_state_integrity(fault):
    values = states()
    if fault == "missing_state":
        values.pop()
    elif fault == "wrong_clock":
        values[8] = torch.zeros((1, 2, 768))
    elif fault == "wrong_width":
        values[8] = torch.zeros((1, 3, 767))
    else:
        values[8] = torch.full((1, 3, 768), float("nan"))
    with pytest.raises(ValueError):
        features.pool(values, torch.ones((1, 3)))


@pytest.mark.parametrize("pcm", [np.zeros(63999, np.float32), np.zeros(64000, np.float64),
                                 np.zeros((2, 32000), np.float32), np.full(64000, np.nan, np.float32)])
def test_bad_source_never_reaches_model(pcm):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid PCM must not run neural inference")
    with pytest.raises(ValueError):
        features.extract(pcm, forbidden)


def test_source_clock_and_payload_are_bound():
    values = np.zeros(features.DIMENSIONS, np.float32)
    record = {"status": "ok", "wavlm": values.tolist(), "wavlm_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
              "feature_frames": 199, "valid_feature_frames": 199, "input_pcm_sha256": "a" * 64}
    source = {"status": "ok", "pcm_sha256": "a" * 64}
    features.validate(record, source)
    for key, value in (("input_pcm_sha256", "b" * 64), ("feature_frames", 200),
                       ("valid_feature_frames", 198), ("feature_frames", 199.), ("wavlm_sha256", "f" * 64)):
        with pytest.raises(ValueError):
            features.validate({**record, key: value}, source)


def test_unassessed_cannot_gain_features_or_become_original():
    source = {"status": "not_applicable"}
    record = {"status": "not_applicable", **{k: None for k in
              ("wavlm", "wavlm_sha256", "feature_frames", "valid_feature_frames", "input_pcm_sha256")}}
    features.validate(record, source)
    with pytest.raises(ValueError):
        features.validate({**record, "status": "ok"}, source)
    with pytest.raises(ValueError):
        features.validate({**record, "wavlm": [0.]}, source)


def fitting_rows():
    return [{"role": "fit", "raw": .2 if label == 0 else .8, "wavlm": [label * 2 - 1., offset],
             "label": label, "generator": "human" if label == 0 else "generator",
             "group": f"{label}-{offset}", "base_id": f"source-{label}-{offset}"}
            for label in (0, 1) for offset in (0., 1., 2.)]


@pytest.mark.parametrize("arm", method.ARMS)
def test_fitting_is_role_isolated_and_learns_from_features(monkeypatch, arm):
    monkeypatch.setattr(method, "DIMENSIONS", 2)
    rows = fitting_rows()
    fitted = method.fit(rows, arm)
    assert method.predict(rows[0], fitted) < method.predict(rows[-1], fitted)
    assert fitted["fit_groups_by_class"] == [3, 3]
    assert fitted["transform"]["center"] == pytest.approx([0., 1.])
    if arm == "standalone":
        assert method.predict({**rows[0], "raw": .99}, fitted) == method.predict(rows[0], fitted)
    else:
        assert fitted["coefficients"][0] >= 0
    for role in ("cal", "development", "confirmation"):
        with pytest.raises(ValueError):
            method.fit([{**r, "role": role} for r in rows], arm)
    with pytest.raises(ValueError):
        method.calibrate(rows, fitted)
    assert method.predict({"raw": None, "wavlm": None}, fitted) is None


@pytest.mark.parametrize("raw", [-.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_raw_is_not_silent_missingness(raw):
    with pytest.raises(ValueError):
        method.feature({"raw": raw, "wavlm": None})


def test_calibration_keeps_unassessed_group_worst_case(monkeypatch):
    monkeypatch.setattr(method, "DIMENSIONS", 2)
    fitted = method.fit(fitting_rows(), "standalone")
    rows = [{"role": "cal", "label": label, "group": f"{label}-{group}", "raw": .5,
             "wavlm": [float(label), 1.]} for label in (0, 1) for group in range(9)]
    rows.append({"role": "cal", "label": 0, "group": "0-0", "raw": None, "wavlm": None})
    guard = method.calibrate(rows, fitted)
    assert guard["0"][0] == {"group": "0-0", "maximum": None, "rows": 2}
    with pytest.raises(ValueError):
        method.calibrate([{**r, "role": "fit"} for r in rows], fitted)


def test_failed_gate_cannot_select_an_arm():
    summaries = {arm: {"advance_to_fresh_confirmation": False} for arm in method.ARMS}
    assert method.select(summaries) is None
    summaries["standalone"]["advance_to_fresh_confirmation"] = True
    assert method.select(summaries) == "standalone"


def test_prospective_lock_rejects_dropped_roles_or_unsupported_claims(monkeypatch):
    monkeypatch.setattr(run, "implementation", lambda: {"method": "fixed"})
    monkeypatch.setattr(run, "runtime", lambda: {"threads": 4})
    monkeypatch.setattr(run, "installed_implementation", lambda: {"wavlm": "fixed"})
    monkeypatch.setattr(run.serving, "load", lambda: {"fixed": True})
    lock = {"schema": run.SCHEMA, "method": method.METHOD, "implementation_sha256": run.implementation(),
            "runtime": run.runtime(), "role_counts": run.COUNTS, "planned": 4248, "expected_unassessed": 62,
            "installed_implementation_sha256": run.installed_implementation(),
            "source_sha256": run.prior.prior.original.SOURCE_SHA,
            "variants_sha256": run.prior.prior.original.VARIANTS_SHA,
            "serving_bundle": run.serving.load(), "serving_calibration_sha256": run.serving.SHA256,
            "original_receipt_sha256": run.prior.prior.reuse.RECEIPT_SHA,
            "original_responses_sha256": run.prior.prior.reuse.RESPONSES_SHA,
            "reserved_confirmation_sha256": run.prior.prior.original.CONFIRMATION_SHA,
            "project_forwards_started": False, "fitting_started": False, "deployment_eligible": False}
    run.validate_lock(lock)
    for key, value in (("planned", 4247), ("role_counts", {"fit": 4248}), ("expected_unassessed", 0),
                       ("implementation_sha256", {}), ("project_forwards_started", True),
                       ("fitting_started", True), ("deployment_eligible", True)):
        bad = copy.deepcopy(lock)
        bad[key] = value
        with pytest.raises(ValueError):
            run.validate_lock(bad)
