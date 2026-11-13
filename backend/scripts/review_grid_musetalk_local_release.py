# SPDX-License-Identifier: Apache-2.0
"""Replay the sanitized short-event qualification separately from detector testing."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.grid-musetalk-local-release/1"
RATE, TICK = 16000, 640


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def qualifies(events, sample):
    local = [i for i in events if 29 <= i <= 41]
    if len(local) == 1 and sample is not None and abs(local[0] * TICK + TICK // 2 - sample) <= 1280:
        return local[0]
    return None


def disposition(row):
    events = row["visual_last_closed"]
    genuine, shift = (row[key].get("sample") for key in ("acoustic", "shifted_acoustic"))
    original = qualifies(events["original"], genuine)
    same = {
        role: qualifies(events[role], genuine)
        for role in (
            "resample_musetalk",
            "brightness_musetalk",
            "musetalk_source_time",
            "musetalk_earlier_320ms",
        )
    }
    matches = {
        role: original is not None and i is not None and abs(i - original) <= 1 for role, i in same.items()
    }
    ordinary = matches["resample_musetalk"] and matches["brightness_musetalk"]
    picture = (
        original is not None
        and any(abs(i - original - 8) <= 1 for i in events["genuine_frame_retime"])
        and qualifies(events["genuine_frame_retime"], genuine) is None
    )
    audio = (
        original is not None
        and shift is not None
        and abs(shift - genuine - 8 * TICK) <= 240
        and qualifies(events["audio_delay_320ms"], shift) is None
    )
    return {
        "genuine_audio_visual_event": original is not None,
        "ordinary_preserved_event": ordinary,
        "picture_retime_control": picture,
        "soundtrack_delay_control": audio,
        "redraw_retained_event": matches["musetalk_source_time"],
        "earlier_neural_also_matches": matches["musetalk_earlier_320ms"],
        "qualified": ordinary
        and picture
        and audio
        and matches["musetalk_source_time"]
        and not matches["musetalk_earlier_320ms"],
    }


def summary(rows):
    if len(rows) != 16 or len({r["group"] for r in rows}) != 8:
        raise ValueError("Missing frozen pilot sources")
    counters = Counter()
    for row in rows:
        decisions = disposition(row)
        counters.update(decisions)
        counters[row["acoustic"]["status"]] += 1
        if not all(row[k] == v for k, v in decisions.items()):
            raise ValueError("Measured short-event eligibility differs")
    return counters


def assemble(folder, color_freeze):
    locked, held = (folder / (k + ".json") for k in ("freeze", "held"))
    data = read(held)
    if data["freeze_sha256"] != sha(locked):
        raise ValueError("Short-event result does not bind its freeze")
    groups = {}
    development = {
        r["group"]
        for r in read(color_freeze)["rows"]
        if r["split"] == "development" and r["family"] == "grid"
    }
    rows = []
    for i, row in enumerate(data["rows"], 1):
        group = groups.setdefault(row["group"], len(groups) + 1)
        compact = {k: v for k, v in row.items() if k not in {"source_commitments", "native_clip", "group"}}
        rows.append(
            {"clip": i, "group": group, "overlaps_chroma_development": row["group"] in development, **compact}
        )
    return {
        "schema": SCHEMA,
        "freeze_sha256": sha(locked),
        "private_result_sha256": sha(held),
        "chroma_freeze_sha256": sha(color_freeze),
        "protocol": read(locked)["protocol"],
        "rows": rows,
        "summary": data["summary"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--assemble", nargs=2, type=Path, metavar=("PRIVATE_RUN", "COLOR_FREEZE"))
    args = parser.parse_args()
    if args.assemble:
        with args.receipt.open("x") as stream:
            json.dump(assemble(*args.assemble), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    data = read(args.receipt)
    if data["schema"] != SCHEMA or summary(data["rows"]) != data["summary"]:
        raise ValueError("Short-event public totals do not replay")
    print(json.dumps(data["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
