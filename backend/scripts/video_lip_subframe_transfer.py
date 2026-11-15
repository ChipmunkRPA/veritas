# SPDX-License-Identifier: MIT
"""Compare source-clock interpolation and single-event sensitivity for lip timing."""

import argparse
import bisect
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.evidence import video_viseme_timing as timing
from scripts import video_lip_viseme_transfer as previous

phone, parent = previous.phone, previous.parent
PANELS = previous.prior.PANEL
PRIMARY = PANELS / "video-lip-viseme-transfer-2026-11-15.json.gz"
TRANSFER = PANELS / "video-lip-viseme-carrier-2026-11-15.json.gz"
SCHEMA = "chipmunk.video-lip-subframe-transfer/1"
ARMS = ("nearest", "interpolated", "event_stability", "interpolated_stability")
METHOD = {
    "interpolation": "Replace nearest-frame aperture with linear measurement between consecutive native landmark observations "
                     "at most 50 ms apart. Require observations at both ends of an interval and keep the nearest original image as its anchor.",
    "stability": "Keep the selected local hypothesis only if it still beats the strongest constant offset after removing "
                 "each interior acoustic event once. Keep the same candidate interval and the same two offsets. Do not search replacements.",
    "split": "Refit aperture reference medians for interpolation only on the original fit speakers. Recompute both points per "
             "arm on the same four calibration speakers before held scoring. Use the 16-speaker primary set, then transfer "
             "the frozen settings to the separate eight-speaker PCM and AAC set. Preserve original truth and score requirements.",
}


def read_inputs():
    original = previous.load(PRIMARY)
    transfer = previous.load(TRANSFER)
    bases = previous.load(previous.prior.SOURCE)["rows"]
    capture = original["acoustic"]
    phone.require(capture["parent_sha256"] == phone.sha(previous.prior.SOURCE)
                  and transfer["primary_sha256"] == phone.sha(PRIMARY), "Source or carrier receipt differs")
    return bases, original, transfer


def reference_profiles(base, captured):
    return {"nearest": previous.fit(base, captured), "interpolated": previous.fit(base, captured, sample_at=interpolate)}


def interpolate(points, sample):
    """Estimate aperture between adjacent native observations on the source clock."""
    times, gaps, indices = points
    right_index = bisect.bisect_left(times, sample)
    if right_index < len(times) and times[right_index] == sample:
        return gaps[right_index], indices[right_index]
    if not 0 < right_index < len(times):
        return None
    left_index = right_index - 1
    width = times[right_index] - times[left_index]
    if indices[right_index] != indices[left_index] + 1 or width > 2 * timing.MAX_NEAREST:
        return None
    left_distance = sample - times[left_index]
    right_distance = times[right_index] - sample
    aperture = (gaps[left_index] * right_distance + gaps[right_index] * left_distance) / width
    anchor = left_index if left_distance <= right_distance else right_index
    return aperture, indices[anchor]


def stable_local_gain(events, candidate):
    """Check that the proposed change beats a constant lag without any one inside event."""
    start, stop = candidate["span"]
    outer = timing.LAGS.index(candidate["baseline_lag"])
    inner = timing.LAGS.index(candidate["baseline_lag"] + candidate["change"])
    constant = timing.totals(events)
    local_events = [event for event in events if start <= event[0] < stop]
    local_total = constant[outer] + sum(event[2][inner] - event[2][outer] for event in local_events)
    return len(local_events) >= 2 and all(
        local_total - event[2][inner] > max(total - value for total, value in zip(constant, event[2], strict=True))
        for event in local_events
    )


def candidates(row, events, references):
    result = {}
    for label, sample in (("nearest", timing.nearest), ("interpolated", interpolate)):
        measured = timing.compatible(row, events, references[label], sample_at=sample)
        first = timing.hypotheses(measured, row["samples"])
        stable = first["status"] != "measured" or stable_local_gain(measured, first)
        alternate = "event_stability" if label == "nearest" else "interpolated_stability"
        result[label] = first
        result[alternate] = first if stable else {"status": "one_event_support", "eligible_events": first["eligible_events"]}
    return result


def primary_rows(base, captured, reference, *, calibration):
    for row, measured in zip(base, captured, strict=True):
        if previous.is_fit(row) or previous.is_calibration(row) != calibration:
            continue
        phone.require((row["source"], row["cohort"]) == (measured["source"], measured["cohort"]), "Primary source order differs")
        for role, case in measured["cases"].items():
            phone.require(case["pcm_sha256"] == row["cases"][role]["pcm_sha256"]
                          and previous.valid_events(case["phones"], row["samples"]), "Primary speech witness differs")
            yield {key: row[key] for key in ("source", "speaker")} | {
                "role": role, "truth": row["edit"] if role in parent.EDITS else None,
                "arms": candidates(row, case["phones"], reference),
            }


def carrier_rows(rows, codec, reference):
    for row in rows:
        for role, case in row["cases"][codec].items():
            phone.require(case["original_sha256"] == row["cases"]["pcm"][role]["sha256"]
                          and previous.valid_events(case["phones"], row["samples"]), "Carrier speech witness differs")
            yield {key: row[key] for key in ("source", "speaker")} | {
                "role": role, "truth": row["edit"] if role in parent.EDITS else None,
                "arms": candidates(row, case["phones"], reference),
            }


def operating_points(calibration):
    result = {}
    for name in ARMS:
        maxima = defaultdict(float)
        for row in calibration:
            if row["truth"] is None:
                maxima[row["speaker"]] = max(maxima[row["speaker"]], row["arms"][name].get("score", 0.))
        values = sorted(maxima.values(), reverse=True)
        phone.require(len(maxima) == 4, "A calibration speaker is missing")
        result[name] = {"tight": values[0], "wider": values[1]}
    return result


def flags(row, name, point):
    item = row["arms"][name]
    return item["status"] == "measured" and item["score"] > point


def compare(rows, points):
    counts = defaultdict(Counter)
    for row in rows:
        shift = parent.EDITS.get(row["role"])
        kind = "local" if shift else "constant" if row["role"].startswith("global_") else "ordinary"
        for point, baseline_point in points["nearest"].items():
            incumbent = row["arms"]["nearest"]
            before = flags(row, "nearest", baseline_point)
            old_useful = bool(shift and before and parent.useful(incumbent["span"], incumbent["change"], row["truth"], shift))
            for name in ARMS:
                item = row["arms"][name]
                hit = flags(row, name, points[name][point])
                useful = bool(shift and hit and parent.useful(item["span"], item["change"], row["truth"], shift))
                counts[f"{name}/{point}/{kind}"].update(files=1, flagged=hit, useful=useful, added_useful=useful and not old_useful,
                                                        lost_useful=old_useful and not useful, added_flags=hit and not before,
                                                        removed_flags=before and not hit, insufficient=item["status"] == "insufficient_events")
    return {key: dict(values) for key, values in counts.items()}


def evaluate(points_path=None):
    base, original, transfer = read_inputs()
    captured = original["acoustic"]["rows"]
    references = reference_profiles(base, captured)
    phone.require(references["nearest"] == original["result"]["references"], "The existing reference fit changed")
    fit = list(primary_rows(base, captured, references, calibration=True))
    points = operating_points(fit)
    phone.require(points["nearest"] == original["result"]["points"]["all_visible"], "The existing points changed")
    frozen = {"schema": SCHEMA, "method": METHOD, "sources": {path.name: phone.sha(path) for path in (PRIMARY, TRANSFER, previous.prior.SOURCE)},
              "reference_profiles": references, "points": points}
    if points_path is not None:
        phone.write(points_path, frozen)
    measured = {"primary": list(primary_rows(base, captured, references, calibration=False))}
    measured.update({codec: list(carrier_rows(transfer["rows"], codec, references)) for codec in previous.CODECS})
    outcomes = {panel: compare(rows, points) for panel, rows in measured.items()}
    for panel, values in outcomes.items():
        prior = original["result"]["outcomes"] if panel == "primary" else transfer["outcomes"][panel]["candidate"]
        for key, stats in values.items():
            if key.startswith("nearest/"):
                expected = prior[key.replace("nearest/", "all_visible/")]
                phone.require((stats["files"], stats["flagged"], stats["useful"]) == tuple(expected[name] for name in ("cases", "flagged", "useful")),
                              "The existing check failed its published comparison")
    return {"freeze": frozen, "calibration": fit, "held": measured, "outcomes": outcomes}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("measure", "review"))
    cli.add_argument("path", type=Path)
    args = cli.parse_args()
    receipt = evaluate(args.path.with_suffix(".points.json") if args.phase == "measure" else None)
    if args.phase == "measure":
        with args.path.open("xb") as stream:
            stream.write(gzip.compress((json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode(), mtime=0))
    else:
        phone.require(previous.load(args.path) == receipt, "Saved timing findings do not reproduce")
    print(json.dumps(receipt["outcomes"], sort_keys=True, indent=2))
