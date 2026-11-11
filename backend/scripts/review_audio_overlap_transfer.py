"""Recompute the hard-overlap readout from private local receipts without models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

TICKS = 300
SAMPLES_PER_TICK = 160
ARMS = ("diarization", "raw_overlap_veto", "normalized_overlap_veto", "normalized_zero_or_overlap_veto")


def load(path: Path):
    return json.loads(path.read_bytes())


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def grid(segments: list[dict]) -> dict:
    def tick(value):
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError("Expected finite diarization timestamps")
        index = round(value * 100)
        if not 0 <= index <= 301 or abs(value * 100 - index) > 0.000001:
            raise ValueError("Timestamp outside the declared 10 ms sample grid")
        return min(index, TICKS)

    cells = [set() for _ in range(TICKS)]
    for segment in segments:
        start, end, track = tick(segment["start"]), tick(segment["end"]), segment["speaker"]
        if type(track) is not int or not 1 <= track <= 8 or start > end:
            raise ValueError("Invalid diarization source interval")
        for cell in cells[start:end]:
            cell.add(track)
    duration = Counter(track for cell in cells for track in cell)
    tracks = sum(ticks >= 16 for ticks in duration.values())
    two = sum(len(cell) == 2 for cell in cells)
    three = sum(len(cell) >= 3 for cell in cells)
    longest = max((sum(1 for _ in run) for active, run in groupby(len(cell) == 2 for cell in cells) if active), default=0)
    route = "unassessed"
    if three >= 16:
        route = "overlap_detected"
    elif tracks == 2 and two >= 24 and longest >= 18:
        route = "single_interferer"
    return {"route": route, "substantive_tracks": tracks, "two_samples": two * SAMPLES_PER_TICK,
            "three_or_more_samples": three * SAMPLES_PER_TICK, "longest_two_samples": longest * SAMPLES_PER_TICK}


def estimated_count(probabilities: list[float]) -> int:
    if (len(probabilities) != 11 or any(type(p) not in {float, int} or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
            or abs(sum(probabilities) - 1) > 1e-6):
        raise ValueError("Invalid eleven-class speaker-count distribution")
    return probabilities.index(max(probabilities))


def verify(root: Path, published: Path | None = None) -> dict:
    manifest_path = root / "manifest.json"
    manifest = load(manifest_path)
    measured = load(root / "measured.json")
    counts = load(root / "counts-private.json")
    aggregate = load(root / "aggregate.json")
    expected = {"manifest": sha256(manifest_path), "measurements": sha256(root / "measured.json"),
                "counts": sha256(root / "counts-private.json"), "diarization_plan": sha256(root / "freeze.json"),
                "count_plan": sha256(root / "overlap-freeze.json")}
    if (measured["freeze_sha256"] != expected["diarization_plan"] or counts["freeze_sha256"] != expected["count_plan"]
            or aggregate["freeze_sha256"] != expected["count_plan"] or aggregate["diarization_receipt_sha256"] != expected["measurements"]
            or aggregate["count_receipt_sha256"] != expected["counts"] or aggregate["plan"] != load(root / "overlap-freeze.json")
            or any(load(root / name)["manifest_sha256"] != expected["manifest"] for name in ("freeze.json", "overlap-freeze.json"))):
        raise ValueError("Saved measurements, aggregate and source freeze do not match")
    stats = defaultdict(Counter)
    false_groups = {arm: set() for arm in ARMS}
    true_groups = {arm: set() for arm in ARMS}
    records = len(manifest["rows"])
    if aggregate["count_forwards"] != 2 * records or measured["calls"] != 2 * records or aggregate["diarization_forwards"] != 2 * records:
        raise ValueError("The receipts do not account for every frozen waveform")
    for index, (row, score, observation) in enumerate(zip(manifest["rows"], counts["rows"], measured["rows"], strict=True)):
        source = root / "pcm" / f"{index:03}.pcm"
        if (index != row["id"] or index != score["id"] or index != observation["id"] or source.stat().st_size != 96000
                or row["pcm_sha256"] != score["pcm_sha256"] or sha256(source) != row["pcm_sha256"]):
            raise ValueError("Source sample or row binding differs from the frozen manifest")
        routes = []
        for point in ("nominal", "sensitive"):
            path = root / point / f"{index:03}.json"
            intervals = load(path)
            if Path(intervals["file"]).name != f"{index:03}.wav":
                raise ValueError("Saved diarization points to another waveform")
            calculation = grid(intervals["segments"])
            routes.append(calculation["route"])
            if {**calculation, "segments_sha256": sha256(path)} != observation["measurements"][point]:
                raise ValueError("Diarization interval arithmetic differs from the saved readout")
        baseline = "overlap_detected" if "overlap_detected" in routes else routes[0] if routes[0] == routes[1] else "unassessed"
        original, normalized = (estimated_count(score["probabilities"][key]) for key in ("original", "minus24"))
        allowed = (True, original <= 2, normalized <= 2, normalized in {1, 2})
        three = row["expected_voices"] == 3
        for arm, allow in zip(ARMS, allowed, strict=True):
            routed = baseline == "single_interferer" and allow
            if routed:
                (false_groups if three else true_groups)[arm].add(row["group"])
            for condition in ("all", f"{row['delivery']}:{row['condition']}", f"language:{row['language']}"):
                stats[f"{arm}/{condition}"].update({"planned_three" if three else "planned_two": 1,
                    "incorrect_three_routes" if three else "correct_two_routes": int(routed),
                    "recorded_diarization_overlap": int(baseline == "overlap_detected"),
                    "diarization_unassessed": int(baseline == "unassessed"), "count_vetoes": int(baseline == "single_interferer" and not allow)})
    if (stats != aggregate["counts"] or {arm: len(group) for arm, group in true_groups.items()} != aggregate["groups_with_correct_two_route"]
            or {arm: len(group) for arm, group in false_groups.items()} != aggregate["groups_with_incorrect_three_route"]):
        raise ValueError("Published condition or group arithmetic differs from the receipts")
    candidate = ARMS[-1]
    current, revised = stats[f"{ARMS[0]}/all"], stats[f"{candidate}/all"]
    advances = (len(false_groups[ARMS[0]]) >= 5 and len(false_groups[candidate]) <= 1
                and revised["incorrect_three_routes"] * 2 <= current["incorrect_three_routes"]
                and current["correct_two_routes"] > 0 and revised["correct_two_routes"] * 5 >= current["correct_two_routes"] * 4)
    if aggregate["advance"] is not advances:
        raise ValueError("The advancement outcome differs from the frozen criteria")
    if published is not None:
        report = load(published)
        if any(report.get(key) != value for key, value in aggregate.items()):
            raise ValueError("Public report differs from the verified local aggregate")
        if report.get("portable_verifier_sha256") != sha256(Path(__file__)):
            raise ValueError("Public report identifies a different verifier")
    return {"verified_pcm": records, "verified_interval_grids": 2 * records, "verified_count_distributions": 2 * records,
            "verified_condition_rollups": len(stats), "verified_source_group_totals": 2 * len(ARMS),
            "verified_advancement_outcome": advances, "sha256": expected}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run", type=Path)
    cli.add_argument("--published", type=Path)
    options = cli.parse_args()
    print(json.dumps(verify(options.run, options.published), indent=2, sort_keys=True))
