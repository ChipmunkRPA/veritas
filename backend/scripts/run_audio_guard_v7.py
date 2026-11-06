"""Extract, fit, apply independent guards and compare actual service outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.special import expit

from scripts import audio_guard_v7 as method
from scripts import audio_residual_head_v6 as learner
from scripts.prepare_audio_guard_v7 import check, write
from scripts.review_audio_noise_v4 import CONDITIONS, metrics, paired


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_rows(path):
    header, *rows = [json.loads(line) for line in path.read_text().splitlines()]
    return header, rows


def setup():
    import torch

    from app.detectors.audio.upgraded import load_voice
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    spec = get_registry().manifest.model("pella.voice_v2")
    if spec.licence.class_ != "commercial":
        raise ValueError("Verified commercial checkpoint declaration required")
    files = ensure_verified(spec, force=True)
    model = load_voice("pella.voice_v2", "cpu", "float32").obj
    return model, [{"name": p.name, "sha256": digest(p)} for p in files]


def capture(model, pcm, captured):
    captured.clear()
    values = model.predict([pcm])
    if len(values) != 1 or len(captured) != 1:
        raise ValueError("Exactly one original complete-window representation required")
    embedding = np.mean(np.concatenate(captured, axis=0), axis=0).astype(np.float32)
    if embedding.shape != (1024,) or not np.isfinite(embedding).all():
        raise ValueError("Invalid frozen speech representation")
    return float(values[0]), embedding


def source_pcm(m, row):
    from app.detectors.audio.decode import decode_audio

    path = Path(m["source_root"]) / row["path"]
    if digest(path) != row["sha256"]:
        raise ValueError("Frozen input bytes changed")
    audio = decode_audio(path, source_sha256=row["sha256"])
    if audio.mapping["pcm_sha256"] != row["pcm_sha256"] or len(audio.pcm) != row["samples"]:
        raise ValueError("Production decoded PCM differs from frozen source")
    return audio.pcm


def extract(manifest, output):
    import torch

    m = check(manifest)
    old_header, old_rows = load_rows(Path(m["prior_representation_receipt"]))
    prior = {r["id"]: r for r in old_rows}
    model, weights = setup()
    direction = model.head.weight.detach().cpu().numpy()[0]
    bias = float(model.head.bias.detach().cpu()[0])
    if (
        weights != old_header["weights"]
        or not np.array_equal(direction, np.asarray(old_header["head_weight"], np.float32))
        or bias != old_header["head_bias"]
    ):
        raise ValueError("Pretrained weights/head differ from retained representations")
    header = {
        "schema": "chipmunk.audio-guard-representations/7",
        "manifest_sha256": digest(manifest),
        "weights": weights,
        "head_weight": direction.tolist(),
        "head_bias": bias,
        "torch_version": torch.__version__,
        "planned": 1440,
        "device": "cpu",
    }
    complete = []
    if output.exists():
        old, complete = load_rows(output)
        if old != header or [r["id"] for r in complete] != [r["id"] for r in m["variants"][: len(complete)]]:
            raise ValueError("Cannot resume changed input/model experiment")
    else:
        with output.open("x") as stream:
            stream.write(json.dumps(header) + "\n")
    captured = []
    hook = model.head.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach().cpu().numpy().copy())
    )
    try:
        with output.open("a") as stream:
            for n, row in enumerate(m["variants"][len(complete) :], len(complete) + 1):
                started = time.monotonic()
                out = {
                    "id": row["id"],
                    "role": row["role"],
                    "source_sha256": row["sha256"],
                    "pcm_sha256": row["pcm_sha256"],
                    "status": "not_applicable",
                    "embedding": None,
                    "reused_prior_representation": row["role"] == "development",
                }
                try:
                    pcm = source_pcm(m, row)
                    if row["raw"] is not None:
                        if row["role"] == "development":
                            p = prior[row["id"]]
                            if (
                                p["status"] != "ok"
                                or p["source_sha256"] != row["sha256"]
                                or learner.digest_embedding(p["embedding"]) != p["embedding_sha256"]
                                or abs(p["raw_replay"] - row["raw"]) > 1e-7
                            ):
                                raise ValueError("Immutable development representation/source differs")
                            raw, embedding = p["raw_replay"], np.asarray(p["embedding"], np.float32)
                        else:
                            raw, embedding = capture(model, pcm, captured)
                        reconstructed = float(expit(np.dot(embedding.astype(np.float64), direction) + bias))
                        if abs(raw - row["raw"]) > 1e-7 or abs(reconstructed - row["raw"]) > 1e-6:
                            raise ValueError("Original predictor/linear-head response parity failed")
                        out.update(
                            status="ok",
                            embedding=embedding.tolist(),
                            raw_replay=raw,
                            head_reconstructed_raw=reconstructed,
                            embedding_sha256=learner.digest_embedding(embedding),
                        )
                    elif row["role"] != "development" or prior[row["id"]]["status"] != "not_applicable":
                        raise ValueError("Unexpected unassessed source")
                except Exception as error:
                    out.update(status="failed", reason=f"{type(error).__name__}: {error}")
                out["runtime_s"] = time.monotonic() - started
                stream.write(json.dumps(out, allow_nan=False) + "\n")
                stream.flush()
                if n % 30 == 0 or out["status"] == "failed":
                    print(
                        json.dumps(
                            {"completed": n, "planned": 1440, "role": row["role"], "status": out["status"]}
                        ),
                        flush=True,
                    )
    finally:
        hook.remove()


def feature_rows(manifest, receipt):
    m = check(manifest)
    header, actual = load_rows(receipt)
    if header["manifest_sha256"] != digest(manifest) or [r["id"] for r in actual] != [
        r["id"] for r in m["variants"]
    ]:
        raise ValueError("Every original source needs a terminal representation record")
    rows = []
    for row, record in zip(m["variants"], actual, strict=True):
        if (
            record["source_sha256"] != row["sha256"]
            or record["pcm_sha256"] != row["pcm_sha256"]
            or record["role"] != row["role"]
        ):
            raise ValueError("Changed source/PCM/role binding")
        if record["status"] != ("ok" if row["raw"] is not None else "not_applicable"):
            raise ValueError("A required representation failed or an unassessed row changed")
        if (
            record["status"] == "ok"
            and learner.digest_embedding(record["embedding"]) != record["embedding_sha256"]
        ):
            raise ValueError("Saved representation bytes changed")
        rows.append({**row, "embedding": record["embedding"]})
    return m, header, rows


def train(manifest, receipt, output):
    _, header, rows = feature_rows(manifest, receipt)
    fit = [r for r in rows if r["role"] == "fit"]
    cal = [r for r in rows if r["role"] == "cal"]
    fitted = method.fit(fit, header["head_weight"])
    # Commit the FIT ONLY parameters before any calibration scoring.
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "fit.json",
        {
            "manifest_sha256": digest(manifest),
            "receipt_sha256": digest(receipt),
            "fit_groups": sorted({r["group"] for r in fit}),
            "fitted": fitted,
            "deployment_eligible": False,
        },
    )
    guards = method.calibrate(cal, fitted)
    write(
        output / "head.json",
        {
            "schema": "chipmunk.audio-source-guard-head/7",
            "manifest_sha256": digest(manifest),
            "receipt_sha256": digest(receipt),
            "fit_sha256": digest(output / "fit.json"),
            "fitted": fitted,
            "guards": guards,
            "fit_groups": sorted({r["group"] for r in fit}),
            "cal_groups": sorted({r["group"] for r in cal}),
            "deployment_eligible": False,
        },
    )
    print(
        json.dumps(
            {
                "fit_sources": 40,
                "calibration_sources": 80,
                "guard_groups": {k: len(v) for k, v in guards.items()},
                "fit_sha256": digest(output / "fit.json"),
                "head_sha256": digest(output / "head.json"),
            }
        ),
        flush=True,
    )


def evaluate(manifest, receipt, head_path, output):
    m, _, rows = feature_rows(manifest, receipt)
    head = json.loads(head_path.read_text())
    if (
        head["manifest_sha256"] != digest(manifest)
        or head["receipt_sha256"] != digest(receipt)
        or head["fit_sha256"] != digest(head_path.parent / "fit.json")
    ):
        raise ValueError("Wrong frozen fitting/calibration artifacts")
    model, weights = setup()
    header = {
        "schema": "chipmunk.audio-source-guard-development/7",
        "manifest_sha256": digest(manifest),
        "receipt_sha256": digest(receipt),
        "head_sha256": digest(head_path),
        "planned": 720,
        "weights": weights,
    }
    expected = [r for r in rows if r["role"] == "development"]
    complete = []
    if output.exists():
        old, complete = load_rows(output)
        if old != header or [r["id"] for r in complete] != [r["id"] for r in expected[: len(complete)]]:
            raise ValueError("Cannot resume changed guarded comparison")
    else:
        with output.open("x") as stream:
            stream.write(json.dumps(header) + "\n")
    captured = []
    hook = model.head.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach().cpu().numpy().copy())
    )
    try:
        with output.open("a") as stream:
            for n, row in enumerate(expected[len(complete) :], len(complete) + 1):
                started = time.monotonic()
                p = learner.predict(row, head["fitted"])
                proposed = method.initial(p, head["guards"])
                probe = {"status": "not_required", "raw": None, "p": None, "embedding": None}
                if proposed == "original":
                    try:
                        pcm = source_pcm(m, row)
                        filtered = sosfiltfilt(
                            butter(4, [100.0, 3800.0], btype="bandpass", fs=16000, output="sos"),
                            np.asarray(pcm, np.float32),
                        ).astype(np.float32)
                        if len(filtered) != len(pcm) or not np.isfinite(filtered).all():
                            raise ValueError("Stability view changed the sample clock or is nonfinite")
                        raw, embedding = capture(model, filtered, captured)
                        q = learner.predict({**row, "raw": raw, "embedding": embedding}, head["fitted"])
                        probe.update(
                            status="computed",
                            raw=raw,
                            p=q,
                            embedding=embedding.tolist(),
                            embedding_sha256=learner.digest_embedding(embedding),
                            filtered_pcm_sha256=hashlib.sha256(filtered.astype("<f4").tobytes()).hexdigest(),
                            samples=len(filtered),
                        )
                    except Exception as error:
                        probe.update(status="failed", reason=f"{type(error).__name__}: {error}")
                decision = method.final(p, probe["p"], head["guards"])
                result = {k: v for k, v in row.items() if k != "embedding"}
                result.update(
                    candidate_p=p,
                    candidate_decision=decision["decision"],
                    candidate_detail=decision,
                    probe=probe,
                    original_p=row["serving_p"],
                    original_decision=row["serving_decision"],
                    runtime_s=time.monotonic() - started,
                )
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                if n % 30 == 0 or probe["status"] == "failed":
                    print(json.dumps({"completed": n, "planned": 720, "probe": probe["status"]}), flush=True)
    finally:
        hook.remove()


def review(manifest, receipt, head_path, scores_path, output):
    m, _, sources = feature_rows(manifest, receipt)
    header, actual = load_rows(scores_path)
    expected = [r for r in sources if r["role"] == "development"]
    head = json.loads(head_path.read_text())
    if (
        header["manifest_sha256"] != digest(manifest)
        or header["receipt_sha256"] != digest(receipt)
        or header["head_sha256"] != digest(head_path)
        or [r["id"] for r in actual] != [r["id"] for r in expected]
    ):
        raise ValueError("Every exact frozen development outcome is required")
    for row, source in zip(actual, expected, strict=True):
        if any(
            row[k] != source[k]
            for k in ("group", "label", "condition", "sha256", "pcm_sha256", "serving_p", "serving_decision")
        ):
            raise ValueError("Outcome bound to another source/reference")
        p = learner.predict(source, head["fitted"])
        if row["candidate_p"] != p or row["candidate_detail"] != method.final(
            p, row["probe"]["p"], head["guards"]
        ):
            raise ValueError("Candidate arithmetic differs from frozen head/guard")
        if row["candidate_decision"] != row["candidate_detail"]["decision"]:
            raise ValueError("Candidate conclusion differs from guarded decision")
        if row["probe"]["status"] == "computed":
            probe = row["probe"]
            if (
                learner.digest_embedding(probe["embedding"]) != probe["embedding_sha256"]
                or learner.predict(
                    {**source, "raw": probe["raw"], "embedding": probe["embedding"]}, head["fitted"]
                )
                != probe["p"]
            ):
                raise ValueError("Probe representation/margin differs")
    panels = {c: [r for r in actual if c == "all" or r["condition"] == c] for c in CONDITIONS}
    panels["joint_vctk"] = [r for r in actual if r["dataset"] == "isjwdu/DFADD"]
    groups = {
        c: {
            "service": metrics(rows, "original_p", "original_decision"),
            "candidate": metrics(rows, "candidate_p", "candidate_decision"),
        }
        for c, rows in panels.items()
    }
    differences = {c: paired(rows, "candidate") for c, rows in panels.items()}
    d = differences["all"]

    def errors(c, k):
        return groups[c]["candidate"][k] <= groups[c]["service"][k] + 1

    gates = {
        "brier_gain": d["brier_difference"] <= -0.005,
        "log_loss_gain": d["log_loss_difference"] <= -0.01,
        "brier_interval_below_zero": d["brier_difference_ci95"][1] < 0,
        "strong_condition_brier": all(differences[c]["brier_difference"] < 0 for c in ("noise_0", "music_5")),
        "joint_voice_brier": differences["joint_vctk"]["brier_difference"] < 0,
        "total_false_calls": all(errors("all", k) for k in ("false_ai", "false_original")),
        "condition_false_calls": all(
            errors(c, k) for c in CONDITIONS[1:] for k in ("false_ai", "false_original")
        ),
        "clean_retention": all(
            groups["clean"]["candidate"][k] >= groups["clean"]["service"][k] - 2
            for k in ("correct_ai", "correct_original")
        ),
        "strong_condition_usefulness": all(
            sum(
                groups[c]["candidate"][k] - groups[c]["service"][k]
                for k in ("correct_ai", "correct_original")
            )
            >= 5
            for c in ("noise_0", "music_5")
        ),
        "same_assessed": groups["all"]["service"]["assessed_inputs"]
        == groups["all"]["candidate"]["assessed_inputs"],
        "complete_cases": len(actual) == 720 and len(sources) == 1440,
        "no_probe_failures": all(r["probe"]["status"] != "failed" for r in actual),
    }
    strata = {}
    for field in ("generator", "dataset", "background_id"):
        strata[field] = {
            str(value): {
                a: metrics([r for r in actual if r[field] == value], key, decision)
                for a, key, decision in (
                    ("service", "original_p", "original_decision"),
                    ("candidate", "candidate_p", "candidate_decision"),
                )
            }
            for value in sorted({r[field] for r in actual}, key=str)
        }
    body = {
        "schema": "chipmunk.audio-source-guard-summary/7",
        "manifest_sha256": digest(manifest),
        "receipt_sha256": digest(receipt),
        "head_sha256": digest(head_path),
        "scores_sha256": digest(scores_path),
        "groups": groups,
        "differences": differences,
        "gates": gates,
        "passes": all(gates.values()),
        "strata": strata,
        "probe_counts": {
            status: sum(r["probe"]["status"] == status for r in actual)
            for status in ("computed", "not_required", "failed")
        },
        "deployment_eligible": False,
        "limitations": m.get("limitations", []),
    }
    write(output, body)
    print(json.dumps({k: v for k, v in body.items() if k not in ("strata", "limitations")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["extract", "train", "evaluate", "review"])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--scores", type=Path)
    args = parser.parse_args()
    if args.stage == "extract":
        extract(args.manifest, args.output)
    elif args.stage == "train" and args.receipt:
        train(args.manifest, args.receipt, args.output)
    elif args.stage == "evaluate" and args.receipt and args.head:
        evaluate(args.manifest, args.receipt, args.head, args.output)
    elif args.stage == "review" and args.receipt and args.head and args.scores:
        review(args.manifest, args.receipt, args.head, args.scores, args.output)
    else:
        parser.error(
            "train/evaluate/review need --receipt; evaluate/review need --head; review needs --scores"
        )
