"""Separate non-neural replay of complete recorded speech features and head scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def recorded_vector(row, name, width):
    values = np.asarray(row[name], np.float64)
    require(values.shape == (width,) and np.isfinite(values).all(), "Complete finite recorded representation required")
    require(values.tolist() == values.astype(np.float32).astype(np.float64).tolist(), "Recorded representation is not lossless float32")
    actual = hashlib.sha256(np.asarray(values, "<f4").tobytes()).hexdigest()
    require(actual == row[name + "_sha256"], "Recorded representation hash differs")
    return values


def reconstruct(row, reference, old, newer, calibration):
    embedding = recorded_vector(row, "embedding", 1024)
    values = recorded_vector(row, "wavlm", 4608)
    raw = row["raw"]
    require(isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw) and 0 <= raw <= 1,
            "Actual finite raw model response required")
    clipped = float(np.clip(raw, 1e-6, 1 - 1e-6))
    scalar_logit = math.log(clipped) - math.log1p(-clipped)
    array_logit = float(np.log(clipped) - np.log1p(-clipped))
    expected = {"original_head": float(expit(embedding @ reference["head_weight"] + reference["head_bias"])),
                "serving_calibration": float(expit(calibration["slope"] * scalar_logit + calibration["intercept"]))}
    transform = old["transform"]
    direction = np.asarray(transform["unit_head"])
    orthogonal = embedding - (embedding @ direction) * direction
    residual = ((orthogonal - transform["center"]) @ np.asarray(transform["directions"]).T) / transform["scale"]
    expected["v7"] = float(expit(np.r_[array_logit, residual, 1.0] @ old["coefficients"]))
    for arm in ("standalone", "complement"):
        model = newer[arm]
        tr = model["transform"]
        normalized = (values - tr["center"]) / tr["scale"] * tr["variance_factor"]
        design = np.r_[scalar_logit, normalized, 1.0] if arm == "complement" else np.r_[normalized, 1.0]
        expected[arm] = float(expit(design @ model["coefficients"]))
    observed = {"original_head": raw, "serving_calibration": row["original_p"], "v7": row["candidate_p"],
                **{arm: row["candidates"][arm]["p"] for arm in ("standalone", "complement")}}
    require(all(isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p) and 0 <= p <= 1
                for p in observed.values()), "Actual finite recorded probabilities required")
    errors = {name: abs(expected[name] - observed[name]) for name in expected}
    require(errors["original_head"] < 1e-6 and max(value for name, value in errors.items() if name != "original_head") < 1e-10,
            "Recorded head probability differs from separate numerical replay")
    return errors


def replay(path):
    original_path = BASE / "audio-v7/development/trained/head.json"
    newer_path = BASE / "audio-v13-wavlm/trained/head.json"
    reference_path = BASE / "audio-v9/development/representations.jsonl"
    calibration_path = ROOT / "backend/app/detectors/audio/calibrations/local-speech-v3.json"
    frozen = json.loads((BASE / "audio-combined-confidence-v84/freeze.json").read_text())
    require(sha(original_path) == frozen["heads_sha256"]["v7"]
            and sha(newer_path) == frozen["heads_sha256"]["v13"]
            and sha(calibration_path) == frozen["protected_audio_sha256"], "Exact frozen head/calibration artifacts required")
    prior = json.loads((BASE / "audio-shared-confirmation-v62/freeze.json").read_text())
    require(sha(reference_path) == prior["artifacts_sha256"][str(reference_path.relative_to(ROOT))],
            "Exact original saved raw-head reference required")
    with reference_path.open() as stream:
        reference = json.loads(next(stream))
    old = json.loads(original_path.read_text())["fitted"]
    newer = json.loads(newer_path.read_text())["fitted"]
    calibration = json.loads(calibration_path.read_text())["parameters"]
    maximum = dict.fromkeys(("original_head", "serving_calibration", "v7", "standalone", "complement"), 0.0)
    total = assessed = unassessed = 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            require(row["status"] in {"ok", "not_applicable"}, "A failed recorded extraction cannot pass replay")
            if row["status"] == "ok":
                require(row["input_pcm_sha256"] == row["pcm_sha256"] and row["feature_frames"] == row["valid_feature_frames"] == 199
                        and row["windows"] == [{"start_sample": 0, "end_sample": 64000}],
                        "Exact recorded four-second source clock and input samples required")
                errors = reconstruct(row, reference, old, newer, calibration)
                for name, value in errors.items():
                    maximum[name] = max(maximum[name], value)
                assessed += 1
            else:
                require(all(row[name] is None for name in ("raw", "embedding", "wavlm", "original_p", "candidate_p"))
                        and all(row["candidates"][name]["p"] is None for name in ("standalone", "complement"))
                        and set(row["decisions"].values()) == {"unassessed"}, "A skipped check cannot supply features or an original verdict")
                unassessed += 1
            total += 1
    return {"schema": "chipmunk.audio-head-numerical-replay/85", "status": "complete", "cases": total,
            "assessed_cases_replayed": assessed, "unassessed_cases_preserved": unassessed,
            "maximum_probability_reconstruction_errors": maximum, "input_journal_sha256": sha(path),
            "reviewer_sha256": sha(Path(__file__)), "new_neural_calls": 0,
            "scope": "Separate recorded-feature/scalar-head arithmetic, hashes and clock review. Not fresh model execution, truth adjudication, stability-decision reanalysis, empirical accuracy, outside human review or confidence reliability."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = replay(args.journal)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result), flush=True)
