"""Test the frozen paired-speech score on native competing-voice deliveries."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from scripts import audio_paired_loud_background as prior

PARENT = prior.PANELS / "audio-spectra-recovery-comparison-2026-11-11.json"
ARCHIVE = prior.ROOT / "backend/data/independent-evaluation/spectra-recovery-2026-11-11/comparison"
SCHEMA = "chipmunk.audio-paired-competing-speech/1"
ORIGINAL_POINT = 0.4355485730182533
METHOD = {
    "inputs": "All 77 previously qualified clean, matched −5 dB and isolated-background deliveries from the AMI recovery panel.",
    "rescue": "For an admitted primary inconclusive result, add an AI flag only if the paired score strictly exceeds its fixed AI point.",
    "corroboration": "Evaluate separately whether requiring the paired score to stay at or above its own fixed original point removes primary false flags. A withheld result becomes inconclusive.",
    "scope": "Run the paired representation only on exactly three seconds of admitted original speech. Preserve other directions and report source-group reuse.",
    "advance": "Rescue requires at least two added synthetic source groups across two declared generator families, zero added clean or isolated flags, and at most one added noisy genuine group. Corroboration requires fewer genuine noisy primary flags across both scenes, no more than one lost synthetic noisy delivery and no clean losses.",
}
NOISY = ("mostly_single_interferer", "overlapping_interferers")
FAMILIES = {"clone", "expressive", "stock"}


def identity():
    return {"schema": SCHEMA, "method": METHOD, "runner_sha256": prior.digest(Path(__file__)),
            "parent_sha256": prior.digest(PARENT), "environmental_runner_sha256": prior.digest(Path(prior.__file__)),
            "ai_point": prior.PAIRED_POINT, "original_point": ORIGINAL_POINT}


def rows_from_archive(archive):
    sources = prior.read(PARENT)
    freeze, roster, baseline = (archive / key for key in ("freeze.json", "roster.json", "baseline.json"))
    prior.check(prior.digest(freeze) == sources["freeze_sha256"] and prior.digest(baseline) == sources["baseline_sha256"]
                and prior.digest(roster) == sources["freeze"]["roster_sha256"], "The source bank or incumbent ledger differs")
    admitted = {row["id"]: row for row in prior.read(baseline)["rows"]}
    rows = []
    for row in prior.read(roster):
        if row["role"] != "prospective":
            continue
        current = admitted.pop(row["id"])
        prior.check(current["pcm_sha256"] == row["pcm_sha256"] and current["decision"] in
                    {"ai", "original", "inconclusive", "unassessed"}, "The source clock or baseline direction differs")
        rows.append((row, current))
    prior.check(not admitted and len(rows) == 77 and len({row["pcm_sha256"] for row, _ in rows}) == 77,
                "Incomplete source replay")
    return rows


def score(run, archive):
    import numpy as np
    import torch

    from scripts import audio_paired_transfer_v230 as transfer
    from scripts import audio_wavlm_three_second_v226 as native
    from scripts.audio_strong_traffic_view import network_free
    from scripts.audio_wavlm_features_v13 import load

    network_free()
    head = prior.settings()
    calibration = prior.ROOT / "backend/data/independent-evaluation/audio-paired-consistency-v223/calibration.json"
    prior.check(prior.read(calibration)["points"][prior.METHOD]["original_below"] == ORIGINAL_POINT,
                "The original diagnostic point differs")
    prior.check(native.checkpoint() == prior.read(native.PUBLIC)["checkpoint"], "The encoder or runtime differs")
    inputs = rows_from_archive(archive)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = load(native.WEIGHTS)
    rows, groups = [], {}
    for (row, current) in inputs:
        raw = (archive / "pcm" / f"{row['id']:03}.pcm").read_bytes()
        prior.check(len(raw) == 96_000 and hashlib.sha256(raw).hexdigest() == row["pcm_sha256"],
                    "Incorrect source samples or clock")
        direction = current["decision"]
        value = None
        if direction in ("ai", "inconclusive"):
            pcm = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
            packet = native.extract(pcm, model)
            prior.check(packet["input_pcm_sha256"] == native.float_sha(pcm), "The paired input samples changed")
            value = transfer.paired_score(current["raw"], packet, head)
        group = groups.setdefault(row["group"], len(groups))
        rows.append({"id": row["id"], "condition": row["condition"], "family": row["family"],
                     "group": group, "primary": direction, "paired": value})
        if len(rows) % 20 == 0:
            print(f"Replayed competing speech on {len(rows)} files", flush=True)
    receipt = {"freeze": identity(), "rows": rows, "summary": summarize(rows)}
    prior.write(run / "receipt.json", receipt)
    print(json.dumps(receipt["summary"], indent=2))


def summarize(rows):
    counts, rescued, families = {}, {"genuine": set(), "synthetic": set()}, set()
    for row in rows:
        condition, family, group = (row[key] for key in ("condition", "family", "group"))
        primary, paired = row["primary"], row["paired"]
        prior.check((paired is not None) == (primary in ("ai", "inconclusive"))
                    and (paired is None or math.isfinite(paired) and 0 <= paired <= 1), "Incorrect paired coverage")
        label = "synthetic" if family in FAMILIES else family
        count = counts.setdefault(f"{condition}/{label}", Counter())
        count["files"] += 1
        count[f"primary_{primary}"] += 1
        count["paired_assessed"] += paired is not None
        gained = primary == "inconclusive" and paired > prior.PAIRED_POINT
        withheld = primary == "ai" and paired < ORIGINAL_POINT
        count["rescued"] += gained
        count["withheld"] += withheld
        if gained and condition in NOISY:
            rescued[label].add(group)
            if label == "synthetic":
                families.add(family)
    prior.check(len(rows) == 77 and len({row["id"] for row in rows}) == 77
                and set(counts) == {f"{scene}/{label}" for scene in ("clean", *NOISY) for label in ("genuine", "synthetic")}
                | {f"isolated_{scene}/background" for scene in NOISY}, "Unexpected test cells")
    for scene in ("clean", *NOISY):
        prior.check(counts[f"{scene}/synthetic"]["files"] == 9 and counts[f"{scene}/genuine"]["files"] == 16,
                    "Each matched scene needs 9 synthetic and 16 genuine deliveries")
    isolated = sum(count["rescued"] for key, count in counts.items() if key.endswith("/background"))
    rescue_ok = (len(rescued["synthetic"]) >= 2 and len(families) >= 2 and len(rescued["genuine"]) <= 1
                 and counts["clean/genuine"]["rescued"] == 0 and isolated == 0)
    losses = sum(counts[f"{scene}/synthetic"]["withheld"] for scene in NOISY)
    corroboration_ok = (all(counts[f"{scene}/genuine"]["withheld"] > 0 for scene in NOISY)
                        and losses <= 1 and counts["clean/synthetic"]["withheld"] == 0)
    return {"counts": counts, "rescued_groups": {key: len(value) for key, value in rescued.items()},
            "rescued_families": sorted(families), "rescue_advance": rescue_ok, "corroboration_advance": corroboration_ok}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    if args.phase == "freeze":
        rows_from_archive(args.archive)
        args.path.mkdir(mode=0o700)
        prior.write(args.path / "freeze.json", identity())
    elif args.phase == "score":
        prior.check(prior.read(args.path / "freeze.json") == identity(), "Prospective decision rule differs")
        score(args.path, args.archive)
    else:
        receipt = prior.read(args.path)
        summary = summarize(receipt["rows"])
        prior.check(receipt["freeze"] == identity() and receipt["summary"] == summary, "Published counts differ")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
