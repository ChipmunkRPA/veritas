"""Apply fixed four-second paired nuisance correction to distinct native AMI room cases."""

from __future__ import annotations

import argparse
import hashlib
import math
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from scripts import audio_paired_nuisance_transfer as prior
from scripts.prepare_audio_specialists_v15 import ASSETS

SCHEMA = "chipmunk.audio-paired-nuisance-native/1"
SAMPLES = 48000
PANEL = prior.PANEL / "audio-spectra-recovery-comparison-2026-11-11.json"
CONDITIONS = ("clean", "mostly_single_interferer", "overlapping_interferers",
              "isolated_mostly_single_interferer", "isolated_overlapping_interferers")


def pcm(source, row):
    wave = (source / "pcm" / f"{row['id']:03d}.pcm").read_bytes()
    prior.require(len(wave) == SAMPLES * 2 and hashlib.sha256(wave).hexdigest() == row["pcm_sha256"],
                  "Archived native sample clock or digest differs")
    return np.frombuffer(wave, "<i2").astype(np.float32) / 32768


def bind(study, source):
    frozen, files, _, _ = prior.checked(study)
    fitted = prior.read(study / "fitted-private.json")
    points = prior.read(study / "points.json")
    prior.require(fitted["freeze_sha256"] == prior.digest(study / "freeze.json")
                  and points["fitted_sha256"] == prior.digest(study / "fitted-private.json"),
                  "The exact frozen four-second fit and points are required")
    published = prior.read(PANEL)
    source_digests = {name: prior.digest(source / name) for name in ("freeze.json", "baseline.json", "roster.json")}
    source_lock = prior.read(source / "freeze.json")
    prior.require(source_digests["freeze.json"] == published["freeze_sha256"]
                  and source_digests["baseline.json"] == published["baseline_sha256"]
                  and source_digests["roster.json"] == source_lock["roster_sha256"],
                  "The prior independently qualified room source and baseline commitments differ")
    clips = [row for row in prior.read(source / "roster.json") if row["role"] == "prospective"]
    scores = {row["id"]: row for row in prior.read(source / "baseline.json")["rows"]}
    all_v84 = prior.read(files["variants"])["variants"]
    v84_ids = {row[key] for row in all_v84 for key in ("source_sha256", "group", "pcm_sha256")}
    v84_backgrounds = {row["background"]["raw_sha256"] for row in all_v84 if row["background"]}
    for row in clips:
        overlap = {row.get(k) for k in ("source_sha256", "source_pcm_sha256", "pcm_sha256", "group")} & v84_ids
        prior.require(not overlap and row.get("source_sha256") not in v84_backgrounds,
                      "Recorded native source or group overlaps a four-second source")
        value = scores.pop(row["id"])
        prior.require(value["pcm_sha256"] == row["pcm_sha256"] and value["decision"] in
                      ("ai", "original", "inconclusive", "unassessed") and (value["raw"] is None) ==
                      (value["decision"] == "unassessed"), "Baseline response does not bind native PCM or admission")
        row["baseline"] = value
    prior.require(len(clips) == 77 and not scores and {row["condition"] for row in clips} == set(CONDITIONS),
                  "The exact native transfer roster is required")
    lock = {"schema": SCHEMA, "runner_sha256": prior.digest(Path(__file__)), "study_freeze_sha256": fitted["freeze_sha256"],
            "study_fitted_sha256": points["fitted_sha256"], "study_points_sha256": prior.digest(study / "points.json"),
            "prior_public_receipt_sha256": prior.digest(PANEL), "source_sha256": source_digests,
            "checkpoint_sha256": ASSETS, "same_clip_raw_error_bound": 1e-6, "cases": len(clips),
            "known_v84_aliases": 0, "frozen_decision": frozen["method"]["policy"],
            "point_source": "Existing four-second calibration points with no native refit",
            "background_recordings": 2, "foreground_source_groups": len({row["group"] for row in clips
                                                                         if row["family"] != "background"})}
    return lock, clips, fitted, points["points"]


def report(rows, points):
    result = {}
    for condition in CONDITIONS:
        present = [row for row in rows if row["condition"] == condition]
        for arm in ("raw_point", "paired_correction"):
            tally = Counter()
            labels, responses = [], []
            for row in present:
                family, old, value = row["family"], row["old"], row[arm]
                kind = family if family in ("genuine", "background") else "synthetic"
                current = prior.decision(old, value, points[arm]["point"])
                tally.update((f"{kind}_current_{old}", f"{kind}_candidate_{current}"))
                if old != "ai" and current == "ai":
                    tally[f"{kind}_added"] += 1
                if old == "ai" and current != "ai":
                    tally[f"{kind}_lost"] += 1
                if kind != "background" and value is not None:
                    labels.append(int(kind == "synthetic"))
                    responses.append(value)
            result.setdefault(condition, {})[arm] = {"counts": dict(tally), "auc": prior.auc(labels, responses)}
    return result


def score(study, source, output, weights):
    from scripts.run_audio_specialists_v15 import load_voice

    lock, rows, fitted, points = bind(study, source)
    prior.require(prior.read(output / "freeze.json") == lock, "Frozen native source or method changed")

    def offline(event, _):
        if event == "socket.connect":
            raise RuntimeError("Native transfer forbids network connections")

    sys.addaudithook(offline)
    started = time.monotonic()
    model = load_voice({"assets": ASSETS, "weights": str(weights)})
    _, _, weight, bias = prior.checked(study)
    prior.require(model.head.weight.detach().cpu().numpy()[0].tolist() == weight.tolist()
                  and model.head.bias.detach().cpu().item() == bias, "Installed head differs from the exact fit head")
    seen = []

    def capture(_, args):
        seen.append(args[0][0].detach().cpu().numpy().astype(np.float64, copy=True))

    hook = model.head.register_forward_pre_hook(capture)
    correction = np.asarray(fitted["correction"])
    center = np.asarray(fitted["center"])
    values, max_discrepancy = [], 0.0
    forwards = 0
    try:
        for row in rows:
            wave = pcm(source, row)
            old = row["baseline"]["decision"]
            projected = raw = None
            if old != "unassessed":
                response, = model.predict([wave])
                forwards += 1
                embedding, = seen
                seen.clear()
                discrepancy = max(abs(response - row["baseline"]["raw"]),
                                  abs(response - float(expit(embedding @ weight + bias))))
                prior.require(discrepancy < lock["same_clip_raw_error_bound"], "Native head does not reproduce baseline")
                max_discrepancy = max(discrepancy, max_discrepancy)
                raw = float(logit(np.clip(row["baseline"]["raw"], 1e-6, 1 - 1e-6)))
                projected = raw - float((embedding - center) @ correction)
                prior.require(math.isfinite(projected), "Native correction is not finite")
            values.append({"id": row["id"], "pcm_sha256": row["pcm_sha256"], "family": row["family"],
                           "condition": row["condition"], "old": old, "raw_point": raw, "paired_correction": projected})
            if len(values) % 10 == 0:
                print(f"{len(values)}/{len(rows)} native views, {forwards} voice forwards", flush=True)
    finally:
        hook.remove()
    ledger = output / "scalar-ledger-private.json"
    prior.write(ledger, values)
    prior.write(output / "aggregate.json", {"schema": SCHEMA, "native_freeze_sha256": prior.digest(output / "freeze.json"),
                "ledger_sha256": prior.digest(ledger), "cases": len(rows), "forwards": forwards,
                "raw_replay_maximum_error": max_discrepancy, "elapsed_s": time.monotonic() - started,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "conditions": report(values, points)})
    print(f"{len(values)} native cases, {forwards} exact-head replay forwards", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "score"))
    for field in ("study", "source", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--weights", type=Path, default=prior.ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    if args.stage == "freeze":
        lock, rows, _, _ = bind(args.study, args.source)
        for row in rows:
            pcm(args.source, row)
        args.output.mkdir(mode=0o700)
        prior.write(args.output / "freeze.json", lock)
    else:
        score(args.study, args.source, args.output, args.weights)


if __name__ == "__main__":
    main()
