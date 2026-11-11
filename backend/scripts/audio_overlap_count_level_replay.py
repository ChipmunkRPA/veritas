"""Replay fixed level normalization for the independent overlap-count model."""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_independent_overlap_count as independent
from scripts import audio_measured_speaker_routing as prior

LEVELS = {"minus24_dbfs": -24, "minus18_dbfs": -18}
PEAK_LIMIT = .95
FIXED = {
    "inputs": "Consumed 182 exact recordings from the independent overlap-count comparison",
    "level_targets_dbfs_rms": LEVELS,
    "peak_limit": PEAK_LIMIT,
    "method": "Multiply the entire original signal by one scalar before centered silence padding. No clipping or waveform dynamics. Compare an explicit three-or-more prediction and a variant that also withholds an implausible zero count.",
}


def level(audio, target_dbfs):
    import numpy as np

    if audio.shape != (prior.SAMPLES,) or not np.isfinite(audio).all() or target_dbfs not in LEVELS.values():
        raise ValueError("Expected exact finite mono source and frozen level target")
    rms = math.sqrt(float(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    if rms == 0:
        return audio, 1., False
    target = 10 ** (target_dbfs / 20) / rms
    limited = PEAK_LIMIT / peak
    gain = min(target, limited)
    return audio * gain, gain, limited < target


def summarize(rows, recorded, estimates):
    panels = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    for row, measurement, scored in zip(rows, recorded, estimates, strict=True):
        if row["id"] != measurement["id"] or row["id"] != scored["id"] or row["pcm_sha256"] != scored["pcm_sha256"]:
            raise ValueError("Expected exact paired routing and count outputs")
        if set(scored["arms"]) != {"original_center", *LEVELS}:
            raise ValueError("Incomplete fixed level views")
        routed = prior.routes(measurement["measurements"])["corroborated"] == "single_interferer"
        for name, result in scored["arms"].items():
            estimate = result["count"]
            if type(estimate) is not int or not 0 <= estimate <= 10:
                raise ValueError("Invalid independently measured count")
            current = panels[row["panel"]][row["condition"]][name]
            current[f"count_{estimate}"] += 1
            current["peak_limited"] += int(result["peak_limited"])
            current["diarization_two_voices"] += int(routed)
            for mode, allowed in (("overlap_only", estimate < 3), ("zero_or_overlap", 0 < estimate < 3)):
                if not (routed and allowed):
                    continue
                current[f"{mode}:routed"] += 1
                if row["decision"] == "inconclusive" and row["crossed_room_point"].get("relation_unpaired"):
                    current[f"{mode}:added_{'synthetic' if row['label'] else 'genuine'}"] += 1
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    rows, old, bound = independent.load(folder)
    if prior.read(folder / "freeze.json") != bound:
        raise ValueError("Original independent overlap study binding changed")
    raw_path = folder / "counts-private.json"
    plan = {"schema": "chipmunk.overlap-level-replay/1", "method": FIXED, "implementation_sha256": prior.sha(Path(__file__)),
            "original_freeze_sha256": prior.sha(folder / "freeze.json"), "original_responses_sha256": prior.sha(raw_path),
            "new_count_forwards": len(rows) * len(LEVELS)}
    freeze_path = folder / "level-freeze.json"
    if args.action == "freeze":
        prior.write(freeze_path, plan)
        return
    if prior.read(freeze_path) != plan:
        raise ValueError("Use the originally frozen level comparison")
    recorded = old + independent.fresh_measured(folder, rows[len(old):])
    raw = prior.read(raw_path)
    if raw["freeze_sha256"] != plan["original_freeze_sha256"]:
        raise ValueError("Invalid frozen baseline binding")
    model = independent.CountNet(folder)
    model.torch.set_num_threads(2)
    scored = []
    for row, raw_row in zip(rows, raw["rows"], strict=True):
        if row["id"] != raw_row["id"] or row["pcm_sha256"] != raw_row["pcm_sha256"]:
            raise ValueError("Independent count baseline source binding changed")
        audio = independent.decode(prior.pcm(folder, row))
        arms = {"original_center": {"count": independent.count(raw_row["probabilities"]["center"]),
                                     "gain": 1., "peak_limited": False}}
        for arm, target in LEVELS.items():
            scaled, gain, peak_limited = level(audio, target)
            result = model.score(independent.padded(scaled, independent.POSITIONS["center"]))
            arms[arm] = {"count": independent.count(result), "gain": gain, "peak_limited": peak_limited}
        scored.append({"id": row["id"], "pcm_sha256": row["pcm_sha256"], "arms": arms})
    response = folder / "level-counts-private.json"
    prior.write(response, scored)
    prior.write(folder / "level-aggregate.json", {"counts": summarize(rows, recorded, scored), "plan": plan,
                "freeze_sha256": prior.sha(freeze_path), "response_sha256": prior.sha(response)})


if __name__ == "__main__":
    main()
