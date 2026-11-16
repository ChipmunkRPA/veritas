# SPDX-License-Identifier: MIT
"""Compare short speech-edit locations across overlapping six-second excerpts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.evidence import audio_focused_edit_binding as focused
from app.evidence import audio_long_review_binding as long_review

SCHEMA = "chipmunk.audio-focused-context-transfer/1"
RATE = focused.RATE
MINIMUM, MAXIMUM = 9 * RATE, 14 * RATE
STRIDE = 2 * RATE
ARMS = ("current", "current_plus_short", "current_plus_focused", "current_plus_repeated")
KINDS = ("edited", "ordinary", "background")
METHOD = {
    "sources": "Study continuous native 16 kHz signed mono 16-bit recordings 9 to 14 seconds long. "
               "Fix the source and delivery digests and exact labeled sample intervals before scoring.",
    "comparison": "Replay today's longer-file review. Compare extending the two-second raw scan to six-second excerpts; "
                  "adding the strongest short interval in each excerpt at its frozen point; and retaining short "
                  "intervals supported by overlapping measurements from different excerpt starts.",
    "clock": "Use six-second excerpts every two seconds and at the exact end of the file. "
             "All outputs use the original recording's sample indices.",
    "repeat": "Two positive ranges from different excerpts agree when they overlap by at least half of both ranges. "
              "Keep each agreeing measured range with its original timing. Do not multiply confidence across excerpts.",
    "outcomes": "Count one-to-one edit locations with at least 50% of the labeled duration and 30% intersection over union. "
                "Count ordinary and isolated-background submissions with any flag, and highlighted genuine seconds including "
                "inside edited files. Report the unassessed counts, lost locations and new source groups.",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(file: Path):
    data = file.read_bytes()
    return json.loads(gzip.decompress(data) if file.suffix == ".gz" else data)


def save(file: Path, data) -> None:
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with file.open("xb") as stream:
        stream.write(gzip.compress(raw, mtime=0) if file.suffix == ".gz" else raw + b"\n")


def excerpts(samples: int) -> list[int]:
    if not MINIMUM <= samples <= MAXIMUM:
        raise ValueError("Native recording is outside this comparison")
    final = samples - focused.SAMPLES
    return sorted(set(range(0, final + 1, STRIDE)) | {final})


def overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def merge(ranges):
    spans = []
    for start, end in sorted(set(ranges)):
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    return spans


def same_location(left, right) -> bool:
    return 2 * overlap(left, right) >= max(left[1] - left[0], right[1] - right[0])


def proposed(row) -> dict[str, list[tuple[int, int]]]:
    current = [tuple(interval) for interval in row["current"]]
    short = []
    candidate = []
    for reading in row["excerpts"]:
        origin = reading["start_sample"]
        if reading["short"] is not None:
            short.append(tuple(origin + sample for sample in reading["short"]))
        record = reading["focused"]
        if record is not None and record["score_units"] > focused.REVIEW_POINT:
            candidate.append((origin, (origin + record["start_sample"], origin + record["end_sample"]), record["score_units"]))
    measured = {interval for _, interval, _ in candidate}
    repeated = {interval for origin, interval, _ in candidate
                if any(other != origin and same_location(interval, match) for other, match, _ in candidate)}
    # Summarize the same detection once across shifted excerpts. Keep the
    # strongest supported measured interval and every non-overlapping lead.
    best = {}
    for _, interval, score in candidate:
        best[interval] = max(score, best.get(interval, score))

    def deduplicate(intervals):
        selected = []
        for interval in sorted(intervals, key=lambda span: (-best[span], span)):
            if not any(same_location(interval, other) for other in selected):
                selected.append(interval)
        return sorted(selected)

    return {"current": current, "current_plus_short": sorted(set(current + short)),
            "current_plus_focused": sorted(set(current + deduplicate(measured))),
            "current_plus_repeated": sorted(set(current + deduplicate(repeated)))}


def matches(ranges, truth) -> set[int]:
    edges = []
    for known in truth:
        candidates = []
        for i, reported in enumerate(ranges):
            common = overlap(reported, known)
            union = reported[1] - reported[0] + known[1] - known[0] - common
            if 2 * common >= known[1] - known[0] and 10 * common >= 3 * union:
                candidates.append((common / union, i))
        edges.append([i for _, i in sorted(candidates, key=lambda value: (-value[0], value[1]))])
    assigned = {}

    def place(label, visited):
        for i in edges[label]:
            if i in visited:
                continue
            visited.add(i)
            if i not in assigned or place(assigned[i], visited):
                assigned[i] = label
                return True
        return False

    for label in range(len(truth)):
        place(label, set())
    return set(assigned.values())


def summarize(rows):
    tables = defaultdict(lambda: {arm: Counter() for arm in ARMS})
    groups = defaultdict(set)
    families = defaultdict(set)
    for row in rows:
        treatment, kind = row["treatment"], row["kind"]
        ranges = proposed(row)
        truth = row["truth"]
        base = matches(ranges["current"], truth)
        measured = {
            "current": row["current_assessed"],
            "current_plus_short": row["current_assessed"] or any(r["short_assessed"] for r in row["excerpts"]),
            "current_plus_focused": row["current_assessed"] or any(r["checked_windows"] for r in row["excerpts"]),
            "current_plus_repeated": row["current_assessed"] or sum(r["checked_windows"] > 0 for r in row["excerpts"]) >= 2,
        }
        for arm, suggested in ranges.items():
            out = tables[treatment][arm]
            out["recordings"] += 1
            out[f"{kind}_recordings"] += 1
            out[f"{kind}_unassessed"] += int(not measured[arm])
            out[f"{kind}_flagged"] += int(bool(suggested))
            match = matches(suggested, truth)
            out["truth_ranges"] += len(truth)
            out["located"] += len(match)
            out["new_locations"] += len(match - base)
            out["lost_locations"] += len(base - match)
            if match - base:
                groups[treatment, arm].add(row["group"])
                families[treatment, arm].add(row["family"])
            if kind != "background":
                out["genuine_samples"] += row["samples"] - sum(b - a for a, b in truth)
                for span in merge(suggested):
                    out["flagged_genuine_samples"] += span[1] - span[0] - sum(overlap(span, t) for t in truth)
    return {treatment: {arm: dict(counter, added_groups=len(groups[treatment, arm]),
                                added_families=sorted(families[treatment, arm]))
                        for arm, counter in result.items()} for treatment, result in sorted(tables.items())}


def validate(payload):
    frozen, rows = payload["freeze"], payload["rows"]
    focused_only = payload.get("stage") == "focused-only"
    assert payload.get("stage") in (None, "focused-only")
    assert frozen["schema"] == SCHEMA and frozen["method"] == METHOD and frozen["point_units"] == focused.REVIEW_POINT
    assert len(rows) == len(frozen["cases"]) and len({r["id"] for r in frozen["cases"]}) == len(rows)
    for row, case in zip(rows, frozen["cases"], strict=True):
        assert all(row[key] == value for key, value in case.items())
        samples = row["samples"]
        assert row["kind"] in KINDS and bool(row["truth"]) == (row["kind"] == "edited")
        assert row["truth"] == [list(span) for span in merge(map(tuple, row["truth"]))]
        assert all(0 <= a < b <= samples for a, b in row["truth"] + row["current"])
        assert [r["start_sample"] for r in row["excerpts"]] == excerpts(samples)
        assert not row["current"] or row["current_assessed"]
        if focused_only:
            assert not row["current"] and not row["current_assessed"]
        for r in row["excerpts"]:
            assert type(r["checked_windows"]) is int and 0 <= r["checked_windows"] <= focused.WINDOWS
            assert (r["checked_windows"] > 0) == (r["focused"] is not None)
            if result := r["focused"]:
                assert type(result["score_units"]) is int and focused.on_grid(result["start_sample"], result["end_sample"])
            if r["short"] is not None:
                a, b = r["short"]
                assert r["short_assessed"] and b - a == 2 * RATE and 0 <= a < b <= focused.SAMPLES
            if focused_only:
                assert not r["short_assessed"] and r["short"] is None
    totals = summarize(rows)
    assert totals == payload["summary"]
    return totals


def score(args, *, focused_only=False):
    import numpy as np
    import torch

    from app.detectors.audio import focused_edit_review, short_voice_review
    from app.engine.boundary import Deadline
    from scripts import audio_joint_generator_transfer as previous
    from scripts.audio_radar_transfer import adapter_result
    from scripts.run_audio_specialists_v15 import load_voice

    frozen = read(args.media / "freeze.json")
    assert frozen["schema"] == SCHEMA and frozen["method"] == METHOD and frozen["point_units"] == focused.REVIEW_POINT
    weights = args.weights
    activity = previous.prior.runtime(weights / "silero_vad.jit")
    model = focused_edit_review.load_model(focused_edit_review.MODEL_ID, "cpu", "float32")
    assert model.revision == frozen["focused_head_sha256"]
    voice = None
    if focused_only:
        # Match the thread counts configured by load_voice in the paired comparison.
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    else:
        assets = previous.prior.read(previous.panels.PRIMARY)["incumbent"]["assets"]
        voice = load_voice({"weights": weights, "assets": assets})
    rows = []
    for index, case in enumerate(frozen["cases"], 1):
        raw = (args.media / "pcm" / f"{case['pcm_sha256']}.pcm").read_bytes()
        samples = case["samples"]
        assert len(raw) == 2 * samples and digest(raw) == case["pcm_sha256"]
        pcm = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.
        dense = bool(long_review.focused_intervals(samples))
        windows = []
        schedule = long_review.focused_intervals(samples) if dense else long_review.intervals(samples)
        if voice is not None:
            for a, b in schedule:
                result = adapter_result(pcm[a:b], voice, activity)
                windows.append({"start_sample": a, "end_sample": b, "direction": result["decision"]})
        if dense and windows:
            current = [[region["start_sample"], region["end_sample"]] for region in long_review.focused_regions(samples, windows)]
        else:
            current = [[w["start_sample"], w["end_sample"]] for w in windows if w["direction"] == "ai"
                       and long_review.status(windows) == "corroborated"]
        scans = []
        for start in excerpts(samples):
            chunk = pcm[start:start + focused.SAMPLES]
            speech = activity.spans(chunk)
            reading = focused_edit_review.scan(chunk, speech, model.obj, Deadline(None))
            short = (short_voice_review.scan(chunk, speech, voice, Deadline(None), codec="pcm_s16le")
                     if voice is not None else {"selected": None, "responses": []})
            spans = short["selected"]
            scans.append({"start_sample": start, "pcm_sha256": digest(chunk.astype("<f4").tobytes()),
                          "short_assessed": bool(short["responses"]),
                          "short": [spans["start_sample"], spans["end_sample"]] if spans else None,
                          "checked_windows": reading["checked_windows"], "focused": reading["best"]})
        rows.append(case | {"excerpts": scans, "current": current,
                            "current_assessed": any(w["direction"] != "unassessed" for w in windows)})
        if index % 12 == 0 or index == len(frozen["cases"]):
            print(f"Compared {index}/{len(frozen['cases'])} recordings.", flush=True)
    receipt = {"freeze": frozen, "rows": rows, "summary": summarize(rows)}
    if focused_only:
        receipt["stage"] = "focused-only"
    validate(receipt)
    save(args.output, receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("score", "score-focused"):
        run = actions.add_parser(action)
        for name in ("media", "weights", "output"):
            run.add_argument(name, type=Path)
    actions.add_parser("review").add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.action in ("score", "score-focused"):
        score(args, focused_only=args.action == "score-focused")
    else:
        print(json.dumps(validate(read(args.receipt)), indent=2, sort_keys=True))
