# SPDX-License-Identifier: MIT
"""Replay the frozen lip-timing check on alternate picture clocks."""

import argparse
import bisect
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from app.evidence import video_viseme_binding as binding
from app.evidence import video_viseme_timing as timing
from scripts import video_lip_viseme_transfer as study

SOURCE = study.prior.PANEL / "video-lip-viseme-carrier-2026-11-15.json.gz"
CLOCKS = {"native": None, "30/hold": (1000, "hold"), "30/nearest": (1000, "nearest"),
          "29.97/hold": (1001, "hold"), "29.97/nearest": (1001, "nearest")}
UNRESOLVED = {"insufficient_events", "ambiguous_change", "incomplete_context"}


def picture_clock(row, clock):
    frames = row["frames"]
    base = Fraction(row["time_base"])
    if clock is not None:
        period, schedule = clock
        times = [frame["pts"] * base for frame in frames]
        duration = times[-1] + (times[-1] - times[-2])
        base = Fraction(1, 30_000)
        converted = []
        for pts in range(0, -(-duration // base), period):
            at = pts * base
            left = bisect.bisect_right(times, at) - 1
            right = min(left + 1, len(times) - 1)
            if schedule == "nearest" and right != left and times[right] - at < at - times[left]:
                left = right
            converted.append({**frames[left], "index": len(converted), "pts": pts})
        frames = converted
    raw = [{key: value for key, value in frame.items() if key not in {"sample", "gap"}}
           | {"gap_units": None if frame["gap"] is None else round(frame["gap"] * timing.SCORE_UNITS)}
           for frame in frames]
    return raw, str(base)


def read_cases(document):
    study.phone.require(document["points"]["all_visible"]["tight"] == binding.POINT
                        and {key: {"median": value["median"]} for key, value in document["references"].items()}
                        == binding.REFERENCE, "Study and application timing points differ")
    for row in document["rows"]:
        original = row["frames"]
        study.phone.require(original[0]["pts"] == 0, "Expected zero-based picture presentation")
        pictures = {name: picture_clock(row, clock) for name, clock in CLOCKS.items()}
        for carrier, cases in row["cases"].items():
            for role, case in cases.items():
                samples = row["samples"]
                study.phone.require(case["original_sha256"] == row["cases"]["pcm"][role]["sha256"]
                                    and study.valid_events(case["phones"], samples), "Soundtrack receipt differs")
                # The original carrier study already froze the acoustic events on the analysis clock.
                # Use that clock directly to vary picture timestamps only, with no codec claim.
                mapping = {"samples": samples, "source_sample_rate": timing.RATE, "source_channels": 1,
                           "timeline_time_base": "1/16000", "pcm_sha256": case["sha256"],
                           "timeline": [{"start_sample": 0, "end_sample": samples, "source_pts": 0}]}
                offset = study.parent.EDITS.get(role)
                kind = "local" if offset else "constant" if role.startswith("global_") else "ordinary"
                results = {}
                for name, (frames, base) in pictures.items():
                    reading = binding.measured({"frames": frames, "phones": case["phones"]},
                                               timestamps=[frame["pts"] for frame in frames], time_base=base,
                                               source={"width": 1280, "height": 720}, mapping=mapping)
                    hit = reading["witness"]
                    useful = bool(offset and hit and study.parent.useful(
                        hit["audio_source_pts"], hit["change_samples"], row["edit"], offset))
                    results[name] = {"status": reading["status"], "useful": useful, "witness": hit}
                yield {"source": row["source"], "speaker": row["speaker"], "role": role, "kind": kind,
                       "carrier": carrier, "truth": row["edit"] if offset else None, "results": results}


def evaluate():
    document = study.load(SOURCE)
    rows = list(read_cases(document))
    counts = defaultdict(Counter)
    for row in rows:
        baseline = row["results"]["native"]
        for name, result in row["results"].items():
            flag = result["witness"] is not None
            counts[f"{row['carrier']}/{name}/{row['kind']}"].update(
                files=1, flagged=flag, useful=result["useful"], unresolved=result["status"] in UNRESOLVED,
                new_flags=flag and baseline["witness"] is None,
                added_useful=result["useful"] and not baseline["useful"],
                lost_useful=baseline["useful"] and not result["useful"])
    return {"schema": "chipmunk.video-lip-frame-clock-transfer/1", "source_sha256": study.phone.sha(SOURCE),
            "point": binding.POINT, "reference": binding.REFERENCE, "counts": dict(counts), "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    path = parser.add_mutually_exclusive_group()
    path.add_argument("--write", type=Path)
    path.add_argument("--verify", type=Path)
    args = parser.parse_args()
    receipt = evaluate()
    if args.write:
        args.write.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    if args.verify:
        study.phone.require(study.load(args.verify) == receipt, "The clock comparisons do not reproduce")
    for label, counts in sorted(receipt["counts"].items()):
        print(label, dict(counts))
