# SPDX-License-Identifier: Apache-2.0
"""Compare lag-peak separation on deliberately displaced single-track video."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts import review_video_lipsync_signed_transitions as rule
from scripts.video_short_soundtrack_transfer import sources as public_sources

SCHEMA = "chipmunk.video-lip-lag-prominence/1"
POINT = .25
MARGIN = .10
OFFSETS = (68, 80)
CASES = ("original", "gain", "lowpass", "noise", "global_early", "local_early", "global_late", "local_late",
         "matched_picture_early", "matched_picture_late")
ARMS = ("current", "fixed_bracket", "distinct_peak_bracket")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def candidate(visual, speech, grid):
    import numpy as np

    from scripts.video_lipsync_signed_transitions import _matrix, _profiles

    rows = _profiles(visual, speech, grid)
    best = dict.fromkeys(ARMS[1:])
    coverage = 0
    for width in rule.WIDTHS:
        for start in range(grid.ticks):
            profiles = [rows.get(span, {}).get("aperture_level") for span in rule.brackets(start, width)]
            if any(profile is None for profile in profiles):
                continue
            coverage += 1
            matrix = _matrix(*profiles)
            lags = np.arange(-rule.MAX_LAG, rule.MAX_LAG + 1)
            nearby = abs(lags[None, :] - lags[:, None]) < 3
            margins = [profile - np.max(np.where(nearby, -np.inf, profile), axis=1) for profile in profiles]
            admitted = (margins[0][:, None] >= MARGIN) & (margins[2][:, None] >= MARGIN) & (margins[1][None, :] >= MARGIN)
            for arm, scores in zip(ARMS[1:], (matrix, np.where(admitted, matrix, 0)), strict=True):
                i, j = np.unravel_index(np.argmax(scores), scores.shape)
                score = float(scores[i, j])
                if score <= (best[arm] or {}).get("score", 0):
                    continue
                best[arm] = {"samples": [start * rule.HOP, (start + width) * rule.HOP],
                             "shift_ticks": int(lags[j] - lags[i]), "score": score,
                             "correlation_margins": [round(float(margins[k][p]), 6) for k, p in enumerate((i, j, i))]}
    return coverage, best


def useful(samples, shift, truth, lag):
    a, b = samples
    x, y = truth
    return 5 * max(0, min(b, y) - max(a, x)) >= 2 * (max(b, y) - min(a, x)) and shift is not None and abs(shift-lag) <= 2


def review(doc):
    if doc["schema"] != SCHEMA or doc["controls"] != list(CASES) or doc["points"] != [POINT, MARGIN]:
        raise ValueError("Unexpected comparison design")
    if [row["source"] for row in doc["rows"]] != doc["roster"]:
        raise ValueError("Source roster differs")
    totals = {}
    for group in doc["rows"]:
        cohort = group["source"]["bank"]
        counts = totals.setdefault(cohort, {arm: Counter() for arm in ARMS})
        cases = group.get("cases", [])
        if cases and [case["role"] for case in cases] != list(CASES):
            raise ValueError("Missing matched cases")
        for arm, count in counts.items():
            count.update({"sources": 1, "edits": 2, "controls": len(CASES) - 2})
            for case in cases:
                truth, shift = case["truth"], case["truth_lag"]
                if (truth is not None) != case["role"].startswith("local_"):
                    raise ValueError("A case differs from its declared edit role")
                if arm == "current":
                    data = case[arm]
                    assessed = bool(data["starts"])
                    flags = [([a * (rule.HOP // 2), b * (rule.HOP // 2)], lag) for _, a, b, lag in data["ranges"]]
                else:
                    proposal = case[arm]
                    assessed = case["eligible_brackets"] > 0
                    if arm == "distinct_peak_bracket" and proposal and min(proposal["correlation_margins"]) < MARGIN:
                        raise ValueError("Distinct lag lacks recorded separation")
                    flags = [(proposal["samples"], proposal["shift_ticks"])] if proposal and proposal["score"] >= POINT else []
                count["edits_measured" if truth else "controls_measured"] += assessed
                if truth:
                    qualified = any(useful(span, lag, truth, shift) for span, lag in flags)
                    count["useful"] += qualified
                    count["wrong_location"] += bool(flags) and not qualified
                elif flags:
                    count["control_flags"] += 1
                    count[f"flag_{case['role']}"] += 1
    return {key: {arm: dict(counter) for arm, counter in counts.items()} for key, counts in totals.items()}


def run(actors, conversations, fresh, output):
    from dataclasses import replace
    from tempfile import TemporaryDirectory

    import cv2
    import numpy as np

    from app.detectors.image.faces import load_yunet
    from app.detectors.video import lipsync, xcorr
    from scripts import video_lipsync_signed_transitions as prior
    from scripts import video_lipsync_window_schedule as source

    rows = [row for row in source.freeze(actors, conversations)["sources"] if row["split"] == "held"]
    transfer = [{"bank": "transfer", "group": number, "offset_s": offset, "file": item["file"], "sha256": item["sha256"]}
                for number, item in enumerate(public_sources(), 1) for offset in OFFSETS]
    if any(digest(fresh / row["file"]) != row["sha256"] for row in transfer) or digest(source.FACE_FILE) != source.FACE_SHA:
        raise ValueError("Source or face geometry differs from its recorded bytes")
    roster = rows + transfer
    doc = {"schema": SCHEMA, "points": [POINT, MARGIN], "controls": list(CASES), "roster": roster, "rows": []}
    cv2.setNumThreads(1)
    detector = load_yunet(source.FACE_FILE)
    with TemporaryDirectory(prefix="video-lag-prominence-") as temp:
        for plan in roster:
            row = {"source": plan}
            doc["rows"].append(row)
            fresh_case = plan["bank"] == "transfer"
            request = plan | {"bank": "conversation"} if fresh_case else plan
            try:
                grid, table, origin, pcm, _, faces, patches, _ = source._observe(
                    request, actors, fresh if fresh_case else conversations, detector, Path(temp))
            except ValueError as error:
                row["unavailable"] = str(error)
                continue
            if grid.ticks < 55:
                row["unavailable"] = "short_input"
                continue
            row["cases"] = []
            row["source_clock"] = {"time_base": table.time_base, "audio_origin_s": str(origin), "decoded_samples": len(pcm),
                                   "native_pictures": len(table)}
            for role, wave, truth, lag, indices in prior.variants(pcm, grid, plan["group"]):
                picture = replace(grid, frames=tuple(grid.frames[i] for i in indices))
                crops = {i: patches[index] for i, index in enumerate(indices) if index in patches}
                aperture = {i: xcorr.aperture(patch) for i, patch in crops.items()}
                vad = lipsync.vad(wave.astype(np.float64) / 32768)
                eligible, proposed = candidate(np.array([aperture.get(i, np.nan) for i in range(grid.ticks)]), vad, picture)
                row["cases"].append({"role": role, "truth": truth, "truth_lag": lag, "eligible_brackets": eligible,
                                     "current": prior.current(faces, crops, aperture, vad, picture),
                                     "pcm_sha256": hashlib.sha256(wave.tobytes()).hexdigest(), **proposed})
            print(plan["bank"], plan["group"], plan["offset_s"], flush=True)
    doc = json.loads(json.dumps(doc, allow_nan=False))
    doc["results"] = review(doc)
    output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps(doc["results"], sort_keys=True))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("run", "review"))
    cli.add_argument("receipt", type=Path)
    for key in ("actors", "conversations", "fresh"):
        cli.add_argument(f"--{key}", type=Path)
    args = cli.parse_args()
    if args.mode == "run":
        if not (args.actors and args.conversations and args.fresh):
            cli.error("Native comparison requires all three source folders")
        run(args.actors, args.conversations, args.fresh, args.receipt)
    else:
        doc = json.loads(args.receipt.read_bytes())
        counts = review(doc)
        if counts != doc["results"]:
            raise ValueError("Recomputed counts differ from the saved receipt")
        print(json.dumps(counts, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
