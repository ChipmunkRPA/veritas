"""Frozen, original-waveform quality calibration development; never serving code."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.special import expit

from scripts.review_audio_noise_v4 import CONDITIONS, logit, metrics, paired, valid, weights

METHOD = {
    "version": "audio-original-quality-development/5.0",
    "sample_rate": 16000,
    "frame_samples": 512,
    "hop_samples": 256,
    "fft": "noncentered real FFT; periodic Hann",
    "spectral_band_hz": [80, 7600],
    "flatness": "median geometric/arithmetic frame power, relative 1e-12 floor",
    "flatness_coordinate": "clip((log10(flatness) + 5) / 5, 0, 1)",
    "floor_coordinate": "clip((log10(frame_energy_p10 / frame_energy_p90) + 4) / 4, 0, 1)",
    "normalization": "divide waveform by peak for scale invariance; never alter classifier input",
    "candidate": "raw logit + two quality coordinates + both raw/quality interactions + intercept",
    "constraint": "nonnegative raw-logit slope at all four quality-box corners",
    "ridge": 0.001,
    "reference_prior": 0.5,
    "selection": "one prespecified candidate; quality-only fit is a nuisance diagnostic",
    "deployment_eligible": False,
}
ARMS = ("original", "quality", "quality_only")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def features(pcm):
    """Two bounded waveform descriptors, not measured SNR or authenticity evidence."""
    x = np.asarray(pcm)
    if x.dtype != np.float32 or x.ndim != 1 or not 40000 <= len(x) <= 64000 or not np.isfinite(x).all():
        raise ValueError("Quality development requires finite float32 mono 2.5–4 second PCM")
    peak = float(np.max(np.abs(x)))
    if peak == 0:
        raise ValueError("Silent audio has no supported quality calibration input")
    x = x.astype(np.float64) / peak
    frames = np.lib.stride_tricks.sliding_window_view(x, 512)[::256]
    window = np.hanning(513)[:-1]
    power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(512, 1 / 16000)
    power = power[:, (frequencies >= 80) & (frequencies <= 7600)]
    average = power.mean(axis=1)
    floor = np.maximum(average * 1e-12, np.finfo(np.float64).tiny)
    stabilized = np.maximum(power, floor[:, None])
    flatness = float(np.median(np.exp(np.log(stabilized).mean(axis=1)) / stabilized.mean(axis=1)))
    energy = (frames**2).mean(axis=1)
    low, high = np.quantile(energy, [0.1, 0.9])
    if high <= np.finfo(np.float64).tiny:
        raise ValueError("The audio has no supported frame energy")
    ratio = float(low / high)
    return {
        "flatness": float(np.clip((np.log10(max(flatness, 1e-12)) + 5) / 5, 0, 1)),
        "floor": float(np.clip((np.log10(max(ratio, 1e-12)) + 4) / 4, 0, 1)),
    }


def design(row, arm):
    if not valid(row.get("raw")):
        return None
    x = float(logit(row["raw"]))
    if arm == "original":
        return np.array([x, 1.0])
    q = row.get("quality")
    if not isinstance(q, dict) or set(q) != {"flatness", "floor"} or not all(valid(v) for v in q.values()):
        return None
    a, b = q["flatness"], q["floor"]
    if arm == "quality_only":
        return np.array([a, b, 1.0])
    if arm != "quality":
        raise ValueError("Unknown development arm")
    return np.array([x, a, b, x * a, x * b, 1.0])


def fit(rows, arm):
    selected = [r for r in rows if design(r, arm) is not None]
    w = weights(selected) if selected else None
    if w is None:
        raise ValueError("Both source classes are required for fitting")
    x = np.array([design(r, arm) for r in selected])
    y = np.array([r["label"] for r in selected])
    ns = [len({r["base_id"] for r in selected if r["label"] == c}) for c in (0, 1)]
    target = np.where(y == 1, (ns[1] + 1) / (ns[1] + 2), 1 / (ns[0] + 2))

    def objective(p):
        z = x @ p
        penalty = METHOD["ridge"] * p.copy()
        penalty[-1] = 0
        return (
            float(w @ (np.logaddexp(0, z) - target * z) + 0.5 * np.dot(p, penalty)),
            x.T @ (w * (expit(z) - target)) + penalty,
        )

    constraints = []
    if arm == "quality":
        corners = np.zeros((4, x.shape[1]))
        for i, (a, b) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            corners[i, [0, 3, 4]] = [1, a, b]
        constraints = [LinearConstraint(corners, 0, np.inf)]
    initial = np.zeros(x.shape[1])
    if arm != "quality_only":
        initial[0] = 1
    result = minimize(
        objective,
        initial,
        jac=True,
        method="SLSQP",
        constraints=constraints,
        bounds=[(0, None), (None, None)] if arm == "original" else None,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("Quality development fit did not converge: " + result.message)
    if arm == "quality" and np.min(corners @ result.x) < -1e-8:
        raise ValueError("The fitted mapping reverses the original response direction")
    return {"arm": arm, "coefficients": result.x.tolist(), "fit_originals_by_class": ns}


def predict(row, parameters):
    x = design(row, parameters["arm"])
    return None if x is None else float(expit(x @ parameters["coefficients"]))


def cv(rows):
    out, fits = [dict(r) for r in rows], []
    for fold in sorted({r["fold"] for r in rows}):
        train = [r for r in rows if r["fold"] != fold]
        held = [r for r in out if r["fold"] == fold]
        if {r["group"] for r in train} & {r["group"] for r in held}:
            raise ValueError("A connected source group crossed development folds")
        for arm in ARMS:
            parameters = fit(train, arm)
            fits.append({"fold": fold, **parameters})
            for row in held:
                row[arm + "_p"] = predict(row, parameters)
    return out, fits


def freeze(source, output):
    prior = json.loads(source.read_text())
    if prior["role"] != "consumed v3 development; not independent confirmation":
        raise ValueError("Only consumed development inputs are permitted")
    root = Path(__file__).resolve().parents[2]
    pins = [
        "backend/scripts/audio_quality_v5.py",
        "backend/scripts/review_audio_noise_v4.py",
        "backend/app/detectors/audio/decode.py",
        "docs/audio-quality-v5-development.md",
    ]
    body = {
        "schema": "chipmunk.audio-quality-development/5",
        "method": METHOD,
        "prior_manifest_sha256": digest(source),
        "prior_manifest": str(source.resolve()),
        "source_root": prior["source_root"],
        "variants": prior["variants"],
        "groups": prior["groups"],
        "fold_class_counts": prior["fold_class_counts"],
        "implementation_sha256": {n: digest(root / n) for n in pins},
        "limitations": prior["limitations"]
        + [
            "Waveform quality features can encode source/channel differences; inspect the joint VCTK panel and quality-only nuisance diagnostic.",
            "This experiment uses no new checkpoint and does not alter the existing voice calibration.",
        ],
        "deployment_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"inputs": len(body["variants"]), "manifest_sha256": digest(output)}))


def verify(manifest):
    root = Path(__file__).resolve().parents[2]
    for name, expected in manifest["implementation_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("Frozen implementation changed: " + name)
    if digest(manifest["prior_manifest"]) != manifest["prior_manifest_sha256"]:
        raise ValueError("Consumed source manifest changed")
    prior = json.loads(Path(manifest["prior_manifest"]).read_text())
    source = Path(manifest["source_root"])
    for name, key in (
        ("manifest.json", "source_manifest_file_sha256"),
        ("predictions.json", "reference_predictions_file_sha256"),
    ):
        if digest(source / name) != prior[key]:
            raise ValueError("Consumed raw score/source records changed")


def extract(manifest_path, output):
    from app.detectors.audio.decode import decode_audio

    m = json.loads(manifest_path.read_text())
    verify(m)
    header = {
        "schema": "chipmunk.audio-quality-features/5",
        "manifest_sha256": digest(manifest_path),
        "method": METHOD,
    }
    completed = []
    if output.exists():
        saved, *completed = [json.loads(s) for s in output.read_text().splitlines()]
        if saved != header or [r["id"] for r in completed] != [
            r["id"] for r in m["variants"][: len(completed)]
        ]:
            raise ValueError("Cannot resume different or reordered inputs")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(json.dumps(header, allow_nan=False) + "\n")
    with output.open("a") as stream:
        for n, row in enumerate(m["variants"][len(completed) :], len(completed) + 1):
            out = {
                "id": row["id"],
                "source_sha256": row["sha256"],
                "quality": None,
                "status": "not_applicable",
            }
            start = time.monotonic()
            try:
                decoded = decode_audio(Path(m["source_root"]) / row["path"], source_sha256=row["sha256"])
                if decoded.mapping["pcm_sha256"] != row["pcm_sha256"] or len(decoded.pcm) != row["samples"]:
                    raise ValueError("Production-decoded source PCM changed")
                if row["raw"] is not None:
                    out.update(quality=features(decoded.pcm), status="ok")
            except Exception as error:
                out.update(status="failed", reason=f"{type(error).__name__}: {error}")
            out["runtime_s"] = time.monotonic() - start
            stream.write(json.dumps(out, allow_nan=False) + "\n")
            stream.flush()
            if n % 60 == 0 or out["status"] == "failed":
                print(
                    json.dumps({"completed": n, "planned": len(m["variants"]), "status": out["status"]}),
                    flush=True,
                )


def review(manifest_path, scores_path, output):
    m = json.loads(manifest_path.read_text())
    verify(m)
    header, *scores = [json.loads(s) for s in scores_path.read_text().splitlines()]
    if header["manifest_sha256"] != digest(manifest_path) or [r["id"] for r in scores] != [
        r["id"] for r in m["variants"]
    ]:
        raise ValueError("Every frozen input requires one ordered terminal outcome")
    rows = []
    for source, actual in zip(m["variants"], scores, strict=True):
        if source["sha256"] != actual["source_sha256"]:
            raise ValueError("Feature/source hash mismatch")
        rows.append({**source, "quality": actual["quality"], "execution_status": actual["status"]})
    predictions, fits = cv(rows)
    panels = {c: [r for r in predictions if c == "all" or r["condition"] == c] for c in CONDITIONS}
    panels["joint_vctk"] = [r for r in predictions if r["dataset"] == "isjwdu/DFADD"]
    groups = {c: {a: metrics(values, a + "_p") for a in ARMS} for c, values in panels.items()}
    differences = {c: paired(values, "quality") for c, values in panels.items()}
    baseline, candidate, delta = groups["all"]["original"], groups["all"]["quality"], differences["all"]
    gates = {
        "overall_brier_gain": delta["brier_difference"] <= -0.005,
        "overall_log_loss_gain": delta["log_loss_difference"] <= -0.01,
        "brier_interval_below_zero": delta["brier_difference_ci95"][1] < 0,
        "strong_condition_proper_scores": all(
            differences[c]["brier_difference"] < 0 for c in ("noise_0", "music_5")
        ),
        "joint_voice_panel_brier_improves": differences["joint_vctk"]["brier_difference"] < 0,
        "no_material_false_call_increase": all(
            groups[c]["quality"][k] <= groups[c]["original"][k] + 1
            for c in CONDITIONS[1:]
            for k in ("false_ai", "false_original")
        ),
        "clean_correct_call_retention": all(
            groups["clean"]["quality"][k] >= groups["clean"]["original"][k] - 2
            for k in ("correct_ai", "correct_original")
        ),
        "strong_condition_usefulness": all(
            sum(
                groups[c]["quality"][k] - groups[c]["original"][k] for k in ("correct_ai", "correct_original")
            )
            >= 5
            for c in ("noise_0", "music_5")
        ),
        "no_new_execution_failure": all(
            r["execution_status"] in ("ok", "not_applicable") for r in predictions
        ),
        "same_assessed_inputs": baseline["assessed_inputs"] == candidate["assessed_inputs"],
    }
    partitions = {}
    for key in ("generator", "dataset", "background_id"):
        partitions[key] = {
            str(value): {a: metrics([r for r in predictions if r[key] == value], a + "_p") for a in ARMS}
            for value in sorted({r[key] for r in predictions}, key=str)
        }
    report = {
        "schema": "chipmunk.audio-quality-development-summary/5",
        "method": METHOD,
        "manifest_sha256": digest(manifest_path),
        "feature_receipt_sha256": digest(scores_path),
        "primary_for_fresh_confirmation": "quality" if all(gates.values()) else None,
        "development_gates": gates,
        "deployment_eligible": False,
        "execution_status": dict(Counter(r["execution_status"] for r in predictions)),
        "groups": groups,
        "paired": differences,
        "partitions": partitions,
        "cv_parameters": fits,
        "operational_reference": {
            c: metrics(values, "serving_p", decision_key="serving_decision") for c, values in panels.items()
        },
        "limitations": m["limitations"]
        + ["No outside independent review; CV probabilities are not service calibrations."],
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with output.with_suffix(".predictions.json").open("x") as stream:
        json.dump(predictions, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "primary": report["primary_for_fresh_confirmation"],
                "gates": gates,
                "all": groups["all"],
                "paired": differences["all"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    p = commands.add_parser("freeze")
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p = commands.add_parser("extract")
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    p = commands.add_parser("review")
    p.add_argument("manifest", type=Path)
    p.add_argument("scores", type=Path)
    p.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.source, args.output)
    elif args.stage == "extract":
        extract(args.manifest, args.output)
    else:
        review(args.manifest, args.scores, args.output)
