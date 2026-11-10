"""Refit the frozen video scalar experiment in a fresh directory using public data.

Same fitter/runtime reproduction, not new inference or an outside review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts import video_joint_scalar_v106 as study

FIELDS = (
    "source_files",
    "evaluated_files",
    "known_evaluation_source_groups",
    "results",
    "common_genuine_calibration_groups",
    "new_points",
    "fitting_support",
)


def compare(expected, actual):
    if type(expected) is float:
        study.require(
            type(actual) is float
            and math.isfinite(actual)
            and math.isclose(expected, actual, abs_tol=1e-9, rel_tol=1e-12),
            "Reproduced numerical result differs",
        )
    elif type(expected) is dict:
        study.require(type(actual) is dict and set(expected) == set(actual), "Reproduced result keys differ")
        for key in expected:
            compare(expected[key], actual[key])
    elif type(expected) is list:
        study.require(type(actual) is list and len(expected) == len(actual), "Reproduced sequence differs")
        for first, second in zip(expected, actual, strict=True):
            compare(first, second)
    else:
        study.require(
            type(expected) is type(actual) and expected == actual, "Reproduced count/status differs"
        )


def reproduce(folder):
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    original_panel, original_trial, original_public = study.PANEL, study.TRIAL, study.PUBLIC
    expected_path = original_panel / "video-joint-scalar-result-v106.json"
    expected = study.read(expected_path)
    inputs = folder / "inputs"
    inputs.mkdir()
    trial = folder / "trial"
    trial.mkdir()
    for name in study.FILES:
        with (inputs / name).open("xb") as stream:
            stream.write((original_panel / name).read_bytes())
    frozen = original_public.read_bytes()
    with (trial / "freeze.json").open("xb") as stream:
        stream.write(frozen)
    public = folder / "published-freeze.json"
    with public.open("xb") as stream:
        stream.write(frozen)
    try:
        study.PANEL, study.TRIAL, study.PUBLIC = inputs, trial, public
        study.run()
    finally:
        study.PANEL, study.TRIAL, study.PUBLIC = original_panel, original_trial, original_public
    actual = study.read(inputs / "video-joint-scalar-result-v106.json")
    compare({key: expected[key] for key in FIELDS}, {key: actual[key] for key in FIELDS})
    result = {
        "schema": "chipmunk.video-joint-scalar-reproduction/106",
        "status": "passed",
        "evaluated_files": actual["evaluated_files"],
        "compared_fields": list(FIELDS),
        "expected_result_sha256": study.sha(expected_path),
        "reproduced_result_sha256": study.sha(inputs / "video-joint-scalar-result-v106.json"),
        "exact_fitted_artifact_hash": expected["fitted_model_sha256"] == actual["fitted_model_sha256"],
        "exact_prediction_journal_hash": expected["prediction_journal_sha256"]
        == actual["prediction_journal_sha256"],
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "new_neural_calls": 0,
        "scope": "Fresh-directory refit/calibration/evaluation using public scalar inputs and the same frozen fitter/runtime. Not fresh-source confirmation, native/model re-execution, independent fitting software or outside human review.",
    }
    study.write(folder / "reproduction.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    reproduce(parser.parse_args().output)
