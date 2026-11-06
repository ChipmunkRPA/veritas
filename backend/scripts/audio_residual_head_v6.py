"""Consumed-data frozen-representation speech-head development, never serving code."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from scripts.audio_quality_v5 import digest, verify
from scripts.review_audio_noise_v4 import CONDITIONS, logit, metrics, paired, valid, weights

METHOD = {
    "version": "audio-residual-head-development/6.0",
    "backbone": "existing pinned commercial-permission Pella v2; all layers frozen",
    "feature": "unchanged predictor head input, arithmetic mean only across original windows",
    "projection": "subtract component parallel to existing head; source-weighted training-only centered PCA",
    "components": 16,
    "scaling": "training PCA standard deviations, minimum 1e-6",
    "predictors": "original clipped logit + 16 residual components + intercept",
    "original_slope": "nonnegative",
    "original_ridge": 0.001,
    "residual_ridge": 0.05,
    "reference_prior": 0.5,
    "new_pretrained_checkpoint": False,
    "new_learned_head": True,
    "development_only": True,
    "deployment_eligible": False,
}


def projection(rows, head):
    selected = [r for r in rows if valid(r.get("raw")) and r.get("embedding") is not None]
    if not selected:
        raise ValueError("No assessed representation inputs")
    e = np.array([r["embedding"] for r in selected], dtype=np.float64)
    h = np.asarray(head, dtype=np.float64)
    if e.ndim != 2 or h.shape != (e.shape[1],) or not np.isfinite(e).all() or not np.isfinite(h).all():
        raise ValueError("Invalid representation or original head")
    norm = float(np.linalg.norm(h))
    if norm <= 0:
        raise ValueError("The original head has no direction")
    h = h / norm
    e = e - np.outer(e @ h, h)
    count = Counter(r["base_id"] for r in selected)
    w = np.array([1 / count[r["base_id"]] for r in selected])
    w /= w.sum()
    center = w @ e
    _, singular, directions = np.linalg.svd((e - center) * np.sqrt(w[:, None]), full_matrices=False)
    k = METHOD["components"]
    if min(directions.shape) < k or singular[k - 1] <= 1e-8:
        raise ValueError("Insufficient training representation rank for the fixed projection")
    return {
        "unit_head": h.tolist(),
        "center": center.tolist(),
        "directions": directions[:k].tolist(),
        "scale": np.maximum(singular[:k], 1e-6).tolist(),
        "fit_sources": len(count),
    }


def design(row, transform):
    if not valid(row.get("raw")):
        return None
    x = float(logit(row["raw"]))
    if transform is None:
        return np.array([x, 1.0])
    embedding = row.get("embedding")
    if embedding is None:
        return None
    e, h = np.asarray(embedding, np.float64), np.asarray(transform["unit_head"])
    if e.shape != h.shape or not np.isfinite(e).all():
        raise ValueError("Invalid held-out representation")
    e = e - np.dot(e, h) * h
    residual = ((e - transform["center"]) @ np.array(transform["directions"]).T) / transform["scale"]
    return np.r_[x, residual, 1.0]


def fit(rows, head=None):
    transform = projection(rows, head) if head is not None else None
    selected = [r for r in rows if design(r, transform) is not None]
    w = weights(selected) if selected else None
    if w is None:
        raise ValueError("Both source classes are required for fitting")
    x = np.array([design(r, transform) for r in selected])
    y = np.array([r["label"] for r in selected])
    ns = [len({r["base_id"] for r in selected if r["label"] == c}) for c in (0, 1)]
    target = np.where(y == 1, (ns[1] + 1) / (ns[1] + 2), 1 / (ns[0] + 2))
    ridge = np.full(x.shape[1], METHOD["residual_ridge"])
    ridge[0], ridge[-1] = METHOD["original_ridge"], 0

    def objective(p):
        z = x @ p
        return (
            float(w @ (np.logaddexp(0, z) - target * z) + 0.5 * np.dot(ridge, p**2)),
            x.T @ (w * (expit(z) - target)) + ridge * p,
        )

    initial = np.zeros(x.shape[1])
    initial[0] = 1
    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0, None)] + [(None, None)] * (len(initial) - 1),
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("Residual head did not converge: " + result.message)
    return {"transform": transform, "coefficients": result.x.tolist(), "fit_originals_by_class": ns}


def predict(row, pars):
    x = design(row, pars["transform"])
    return None if x is None else float(expit(x @ pars["coefficients"]))


def cv(rows, head):
    out, fits = [dict(r) for r in rows], []
    for fold in sorted({r["fold"] for r in rows}):
        train, held = [r for r in rows if r["fold"] != fold], [r for r in out if r["fold"] == fold]
        if {r["group"] for r in train} & {r["group"] for r in held}:
            raise ValueError("A connected source group crossed development folds")
        for arm in ("original", "residual"):
            p = fit(train, head if arm == "residual" else None)
            fits.append({"fold": fold, "arm": arm, **p})
            for row in held:
                row[arm + "_p"] = predict(row, p)
    return out, fits


def freeze(source, output):
    prior = json.loads(source.read_text())
    verify(prior)
    root = Path(__file__).resolve().parents[2]
    pins = [
        "backend/scripts/audio_residual_head_v6.py",
        "backend/scripts/audio_quality_v5.py",
        "backend/scripts/review_audio_noise_v4.py",
        "backend/app/detectors/audio/upgraded.py",
        "backend/app/detectors/audio/decode.py",
        "backend/app/third_party/kaldi/fbank.py",
        "backend/app/models.yaml",
        "docs/audio-residual-v6-development.md",
    ]
    body = {
        **prior,
        "schema": "chipmunk.audio-residual-development/6",
        "method": METHOD,
        "prior_quality_manifest": str(source.resolve()),
        "prior_quality_manifest_sha256": digest(source),
        "implementation_sha256": {n: digest(root / n) for n in pins},
    }
    if any(
        r["raw"] is not None
        and (
            len(r["original_windows"]) != 1
            or r["original_windows"][0]["start_sample"] != 0
            or r["original_windows"][0]["end_sample"] != r["samples"]
        )
        for r in body["variants"]
    ):
        raise ValueError("This development panel must retain one complete original window per assessed input")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"inputs": len(body["variants"]), "manifest_sha256": digest(output)}))


def extract(manifest_path, output):
    import torch

    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.upgraded import load_voice
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    m = json.loads(manifest_path.read_text())
    verify(m)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model_spec = get_registry().manifest.model("pella.voice_v2")
    if model_spec.licence.class_ != "commercial":
        raise ValueError("Missing commercial checkpoint permission record")
    files = ensure_verified(model_spec, force=True)
    model = load_voice("pella.voice_v2", "cpu", "float32").obj
    header = {
        "schema": "chipmunk.audio-representation-receipt/6",
        "method": METHOD,
        "manifest_sha256": digest(manifest_path),
        "torch_version": torch.__version__,
        "weights": [{"name": p.name, "sha256": digest(p)} for p in files],
        "head_weight": model.head.weight.detach().cpu().numpy()[0].tolist(),
        "head_bias": float(model.head.bias.detach().cpu()[0]),
    }
    completed = []
    if output.exists():
        saved, *completed = [json.loads(s) for s in output.read_text().splitlines()]
        if saved != header or [r["id"] for r in completed] != [
            r["id"] for r in m["variants"][: len(completed)]
        ]:
            raise ValueError("Cannot resume different inputs, weights, method or runtime")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(json.dumps(header, allow_nan=False) + "\n")
    captured = []
    hook = model.head.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach().cpu().numpy().copy())
    )
    try:
        with output.open("a") as stream:
            for n, row in enumerate(m["variants"][len(completed) :], len(completed) + 1):
                out = {
                    "id": row["id"],
                    "source_sha256": row["sha256"],
                    "status": "not_applicable",
                    "embedding": None,
                }
                started = time.monotonic()
                try:
                    decoded = decode_audio(Path(m["source_root"]) / row["path"], source_sha256=row["sha256"])
                    if (
                        decoded.mapping["pcm_sha256"] != row["pcm_sha256"]
                        or len(decoded.pcm) != row["samples"]
                    ):
                        raise ValueError("Production decoded PCM changed")
                    if row["raw"] is not None:
                        captured.clear()
                        chunks = [
                            decoded.pcm[w["start_sample"] : w["end_sample"]] for w in row["original_windows"]
                        ]
                        values = model.predict(chunks)
                        if (
                            len(values) != len(chunks)
                            or len(captured) != len(chunks)
                            or abs(float(np.mean(values)) - row["raw"]) > 1e-7
                        ):
                            raise ValueError("Unchanged predictor no longer matches stored raw responses")
                        embedding = np.mean(np.concatenate(captured, axis=0), axis=0).astype(np.float32)
                        if (
                            embedding.shape != (len(header["head_weight"]),)
                            or not np.isfinite(embedding).all()
                        ):
                            raise ValueError("Invalid pooled representation")
                        reconstructed = float(
                            expit(
                                np.dot(embedding.astype(np.float64), header["head_weight"])
                                + header["head_bias"]
                            )
                        )
                        if abs(reconstructed - row["raw"]) > 1e-6:
                            raise ValueError("Captured representation does not reconstruct the old response")
                        out.update(
                            status="ok",
                            embedding=embedding.tolist(),
                            raw_replay=float(np.mean(values)),
                            head_reconstructed_raw=reconstructed,
                            embedding_sha256=hashlib.sha256(embedding.astype("<f4").tobytes()).hexdigest(),
                        )
                except Exception as error:
                    out.update(status="failed", reason=f"{type(error).__name__}: {error}")
                out["runtime_s"] = time.monotonic() - started
                stream.write(json.dumps(out, allow_nan=False) + "\n")
                stream.flush()
                if n % 30 == 0 or out["status"] == "failed":
                    print(
                        json.dumps({"completed": n, "planned": len(m["variants"]), "status": out["status"]}),
                        flush=True,
                    )
    finally:
        hook.remove()


def review(manifest_path, receipt_path, output):
    m = json.loads(manifest_path.read_text())
    verify(m)
    header, *actual = [json.loads(s) for s in receipt_path.read_text().splitlines()]
    if header["manifest_sha256"] != digest(manifest_path) or [r["id"] for r in actual] != [
        r["id"] for r in m["variants"]
    ]:
        raise ValueError("All frozen inputs require an ordered terminal representation receipt")
    rows = []
    for expected, scored in zip(m["variants"], actual, strict=True):
        if expected["sha256"] != scored["source_sha256"]:
            raise ValueError("Representation/source hash mismatch")
        if scored["status"] == "ok" and digest_embedding(scored["embedding"]) != scored["embedding_sha256"]:
            raise ValueError("Saved representation changed")
        rows.append({**expected, "embedding": scored["embedding"], "execution_status": scored["status"]})
    predictions, fits = cv(rows, header["head_weight"])
    panels = {c: [r for r in predictions if c == "all" or r["condition"] == c] for c in CONDITIONS}
    panels["joint_vctk"] = [r for r in predictions if r["dataset"] == "isjwdu/DFADD"]
    groups = {c: {a: metrics(v, a + "_p") for a in ("original", "residual")} for c, v in panels.items()}
    differences = {c: paired(v, "residual") for c, v in panels.items()}
    d = differences["all"]
    gates = {
        "overall_brier_gain": d["brier_difference"] <= -0.005,
        "overall_log_loss_gain": d["log_loss_difference"] <= -0.01,
        "brier_interval_below_zero": d["brier_difference_ci95"][1] < 0,
        "strong_condition_proper_scores": all(
            differences[c]["brier_difference"] < 0 for c in ("noise_0", "music_5")
        ),
        "joint_voice_panel_brier_improves": differences["joint_vctk"]["brier_difference"] < 0,
        "no_material_false_call_increase": all(
            groups[c]["residual"][k] <= groups[c]["original"][k] + 1
            for c in CONDITIONS[1:]
            for k in ("false_ai", "false_original")
        ),
        "clean_correct_call_retention": all(
            groups["clean"]["residual"][k] >= groups["clean"]["original"][k] - 2
            for k in ("correct_ai", "correct_original")
        ),
        "strong_condition_usefulness": all(
            sum(
                groups[c]["residual"][k] - groups[c]["original"][k]
                for k in ("correct_ai", "correct_original")
            )
            >= 5
            for c in ("noise_0", "music_5")
        ),
        "same_assessed_inputs": groups["all"]["original"]["assessed_inputs"]
        == groups["all"]["residual"]["assessed_inputs"],
        "no_new_execution_failure": all(
            r["execution_status"] in ("ok", "not_applicable") for r in predictions
        ),
    }
    partitions = {
        key: {
            str(value): {
                a: metrics([r for r in predictions if r[key] == value], a + "_p")
                for a in ("original", "residual")
            }
            for value in sorted({r[key] for r in predictions}, key=str)
        }
        for key in ("generator", "dataset", "background_id")
    }
    report = {
        "schema": "chipmunk.audio-residual-development-summary/6",
        "method": METHOD,
        "manifest_sha256": digest(manifest_path),
        "representation_receipt_sha256": digest(receipt_path),
        "primary_for_fresh_confirmation": "residual" if all(gates.values()) else None,
        "development_gates": gates,
        "deployment_eligible": False,
        "execution_status": dict(Counter(r["execution_status"] for r in predictions)),
        "groups": groups,
        "paired": differences,
        "partitions": partitions,
        "operational_reference": {
            c: metrics(v, "serving_p", decision_key="serving_decision") for c, v in panels.items()
        },
        "limitations": m["limitations"]
        + [
            "Representations and source channels may encode confounds. No outside independent review.",
            "New head mappings are development fits, not approved service calibrations.",
        ],
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with output.with_suffix(".fits.json").open("x") as stream:
        json.dump(fits, stream, allow_nan=False)
        stream.write("\n")
    for row in predictions:
        row.pop("embedding", None)
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


def digest_embedding(value):
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes()).hexdigest()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "extract", "review"])
    p.add_argument("manifest", type=Path)
    p.add_argument("receipt", type=Path)
    p.add_argument("output", type=Path, nargs="?")
    a = p.parse_args()
    if a.stage == "freeze":
        freeze(a.manifest, a.receipt)
    elif a.stage == "extract":
        extract(a.manifest, a.receipt)
    else:
        if a.output is None:
            p.error("review requires an output path")
        review(a.manifest, a.receipt, a.output)
