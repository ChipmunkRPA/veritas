"""Replay short whole-track bandwidth interventions and untouched controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

if __package__:
    from .review_audio_formant_local_transfer import canonical, require, useful
else:
    from review_audio_formant_local_transfer import canonical, require, useful

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "chipmunk.audio-bandwidth-return-transfer/1"
METHOD = {
    "scope": "Single submitted mixed track, six seconds, native PCM16 at 16 kHz. Report temporary upper-band attenuation as processing context and no AI weight.",
    "point": "Fixed 12 dB drop in both upper bands, bounded change in a middle band, and similar measurable flanks. No data-fitted point. Both flanks contain five 80 ms source cells. Ranges from 7 to 15 cells in two-cell steps.",
    "positives": "Zero-phase Butterworth whole-mix low-pass at 3400 or 3800 Hz, with 25 ms raised-cosine edit edges. Edit start and duration determined by exact source digest before inference. Recompute at least half edit coverage and 0.30 time IoU.",
    "controls": "Matched untouched and whole-file low-pass variants. Add brief matched level reduction, ordinary edits, AI-only voice replacements, natural background excerpts, and global EQ controls. The known voice replacements do not filter the whole mix and are negative for this physical measurement.",
    "gate": "At least 80 percent useful locations on each cutoff, at most one matched untouched or global-filter false flag, at most one flag among unfiltered voice replacements/ordinary controls/background, and a gain of at least ten useful intervals over the prior sound-transition leads.",
}


def report(rows):
    output = {"method": {}, "gate_pass": False}
    positives, controls, gain = {}, 0, 0
    for row in rows:
        count = output["method"].setdefault(row["variant"], Counter())
        count["planned"] += 1
        count["unassessed"] += row["best"] is None
        flagged = row["best"] is not None and row["best"]["drop_db"] > 12
        count["flags"] += flagged
        truth = row["truth"]
        if truth is None:
            controls += flagged
            continue
        hit = flagged and useful(row["best"]["range"], truth)
        baseline = (any(abs(tick - truth[0]) <= 4000 for tick in row["existing_transitions"])
                    and any(abs(tick - truth[1]) <= 4000 for tick in row["existing_transitions"]))
        count["useful"] += hit
        count["baseline_two_edges"] += baseline
        gain += hit - baseline
        positives[row["variant"]] = 5 * count["useful"] >= 4 * count["planned"]
    matched = sum(output["method"][arm]["flags"] for arm in ("untouched", "global_lowpass"))
    output["gain_in_useful_locations"] = gain
    output["gate_pass"] = all(positives.values()) and matched <= 1 and controls - matched <= 1 and gain >= 10
    return output


def review(receipt):
    rows = receipt["rows"]
    require(receipt["freeze"]["schema"] == SCHEMA and receipt["freeze"]["method"] == METHOD, "Unexpected experiment")
    require(len(rows) == 170 and len({row["id"] for row in rows}) == len(rows), "Incorrect delivery extent")
    for row in rows:
        require(len(row["sha256"]) == 64 and len(row["parent_sha256"]) == 64, "Invalid source hash")
        require(0 <= row["candidates"] <= 305 and bool(row["candidates"]) == bool(row["best"]), "Invalid measured coverage")
        for tick in row["existing_transitions"]:
            require(tick % 4000 == 0 and 8000 <= tick <= 88000, "Prior transition outside native clock")
        if row["best"] is not None:
            a, b = row["best"]["range"]
            require(a % 1280 == 0 and a >= 6400 and b <= 89600 and (b - a) in (8960, 11520, 14080, 16640, 19200),
                    "New range differs from the 80 ms sample grid")
            left, interior, right = row["best"]["relative_band_db"]
            drop = [min(x, z) - y for x, y, z in zip(left, interior, right, strict=True)]
            require(abs(drop[0]) <= 5.000002 and abs(min(drop[1:]) - row["best"]["drop_db"]) <= .000002,
                    "Recorded attenuation differs from its acoustic support")
    summary = report(rows)
    require(summary == receipt["summary"], "Summary does not match recorded spans")
    return summary


def _roster(args):
    from scripts.audio_local_context_transfer import bindings, pcm

    bound, paths, _ = bindings(args)
    held = [row for row in bound if row["role"] == "held"]
    require(len(held) == 52, "Incorrect three-generator native source extent")
    for item in held:
        raw = pcm(paths[item["id"]])
        parent = item["pcm_sha256"]
        require(hashlib.sha256(raw).hexdigest() == parent, "Native source changed")
        if item["arm"] in {"untouched", "background_only"}:
            for variant in ("untouched", "global_lowpass", "local_lowpass_3400", "local_lowpass_3800", "local_gain"):
                yield item["id"], parent, raw, variant
        else:
            yield item["id"], parent, raw, "unfiltered_" + item["arm"]
    # Add a second bank with independent native readers and other background roles.
    transfer = json.loads(args.transfer.read_bytes())
    from scripts.audio_overlap_pitch_transfer import key, wav_pcm

    for item in transfer["assessment"]:
        if item["kind"] not in {"untouched", "global_eq"}:
            continue
        raw = wav_pcm(args.transfer_wavs / "{}_{}_{}_submitted.wav".format(*key(item)))
        parent = item["submitted_sha256"]
        require(hashlib.sha256(raw).hexdigest() == parent, "Transfer source changed")
        for variant in (("untouched", "global_lowpass", "local_lowpass_3400", "local_lowpass_3800", "local_gain")
                        if item["kind"] == "untouched" else ("prior_global_eq",)):
            yield "transfer-{}-{}-{}".format(*key(item)), parent, raw, variant


def _measure(roster):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    from app.evidence.audio_bandwidth_return import locate
    from app.evidence.audio_continuity import candidates, select

    rows = []
    for name, parent, raw, variant in roster:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768
        first = 16000 + int(parent[:4], 16) % 56000
        duration = 12800 + int(parent[4:8], 16) % 3201
        end = first + duration
        truth = [first, end] if variant.startswith("local_lowpass") else None
        if variant in {"global_lowpass", "local_gain"} or truth:
            cutoff = int(variant.rsplit("_", 1)[-1]) if truth else 3400
            filtered = (samples * .38 if variant == "local_gain" else sosfiltfilt(butter(10, cutoff, fs=16000, output="sos"), samples))
            if variant == "global_lowpass":
                samples = filtered
            else:
                rise = .5 - .5 * np.cos(np.linspace(0, np.pi, 400))
                blend = np.r_[rise, np.ones(duration - 800), rise[::-1]]
                samples[first:end] += (filtered[first:end] - samples[first:end]) * blend
        encoded = np.clip(np.rint(samples * 32768), -32768, 32767).astype("<i2")
        samples = encoded.astype(np.float64) / 32768
        count, result = locate(samples)
        rows.append({"id": name + ":" + variant, "parent_sha256": parent, "sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
                     "variant": variant, "truth": truth, "candidates": count, "best": result,
                     "existing_transitions": [row["boundary_sample"] for row in select(candidates(samples))]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "replay", "review"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--primary-wavs", type=Path)
    parser.add_argument("--transfer-wavs", type=Path)
    parser.add_argument("--held", type=Path, default=ROOT / "backend/data/independent-evaluation/audio-remote-neighbor-replication-2026-11-12")
    panel = ROOT / "docs/evaluation-panels"
    parser.add_argument("--intake", type=Path, default=panel / "audio-phoneme-short-edit-intake-2026-11-13.json")
    parser.add_argument("--prior", type=Path, default=panel / "audio-remote-neighbor-replication-2026-11-12.json")
    parser.add_argument("--transfer", type=Path, default=panel / "audio-clean-reference-transfer-2026-11-13.json")
    args = parser.parse_args()
    if args.phase == "review":
        print(json.dumps(review(json.loads(args.run.read_bytes())), sort_keys=True))
        return
    args.development_wavs = args.primary_wavs
    roster = list(_roster(args))
    freeze = {"schema": SCHEMA, "method": METHOD, "roster_sha256": hashlib.sha256(canonical([(name, sha, variant) for name, sha, _, variant in roster])).hexdigest(),
              "source_pins": [hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.prior, args.transfer, Path(__file__),
                              ROOT / "backend/app/evidence/audio_bandwidth_return.py", ROOT / "backend/app/evidence/audio_continuity.py")]}
    frozen = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        frozen.write_bytes(canonical(freeze))
        print(f"Froze {len(roster)} deliveries before inference")
        return
    require(frozen.read_bytes() == canonical(freeze), "Frozen source or method changed")
    rows = _measure(roster)
    receipt = {"freeze": freeze, "rows": rows, "summary": report(rows)}
    review(receipt)
    public = args.run / "public.json"
    if args.phase == "score":
        with public.open("xb") as stream:
            stream.write(canonical(receipt))
    else:
        require(public.read_bytes() == canonical(receipt), "Native replay differs")
    print(json.dumps(receipt["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
