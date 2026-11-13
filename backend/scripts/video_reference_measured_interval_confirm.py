# SPDX-License-Identifier: Apache-2.0
"""Confirm strong paired-picture support on source-disjoint authored clips."""

import argparse
import json
from itertools import chain
from pathlib import Path

import cv2
import numpy as np

from app.evidence import video_reference_measured_interval as measured
from scripts import video_reference_boundary_transfer as development
from scripts.video_common_sources_v89 import read, require, sha, write

SCHEMA = "chipmunk.video-reference-measured-interval-confirmation/1"
DEVELOPMENT_FREEZE = "870da5722982b9e820b7e158433654bd43cad8388e5fe621ba6ae65e23f4ac87"
PROTOCOL = {
    "measure": "Confirm the existing local finding with a contiguous run above the flanks at the original tile. Both edges must step by at least half of the median core signal above the background. The first trial is development for this stricter gate.",
    "held": "Take the next eight eligible archived CREMA-D groups after the first trial. Use the same native decode and clocks but fresh edit positions and durations. Exact source intervals are known before encoding. The same redraws are diagnostic with prior exposure.",
    "gate": "The earlier gate applies to newly authored clips with at least twelve exact refinements and six gains across at least four distinct groups. Require at least two retained neural redraws with reduced endpoint error across at least two distinct source groups, including one independently timing-qualified. Preserve no neutral findings, wrong-source rejection, median neural error, and spatial support.",
}


def freeze(args):
    first = args.development / "freeze.json"
    require(sha(first) == DEVELOPMENT_FREEZE, "Unexpected development freeze")
    original = read(first)
    candidates = [json.loads(line) for path in sorted(development.media.DIR.glob("extract-*/output/cases.jsonl"))
                  for line in path.read_text().splitlines()]
    groups = {r["asset_index"]: r["source_group"] for r in candidates}
    excluded = {r["source_group"] for r in original["candidate_sources"]}
    for path in args.exclusion:
        roster = read(path)
        excluded.update(row.get("source_group", row.get("group", groups.get(row.get("asset_index"))))
                        for row in roster.get("sources", roster.get("roster", [])))
    require(len(excluded) == 32 and [sha(path) for path in args.exclusion] == original["exclusions_sha256"],
            "Prior group exclusions differ")
    eligible = [row for row in candidates if row["condition"] == "original" and row["status"] == "ready"
                and row["source_group"] not in excluded and len(row["expected"]["rgb_sha256"]) >= 50]
    sources = [{key: row[key] for key in development.SOURCE_FIELDS}
               for row in sorted(eligible, key=lambda item: item["asset_index"])[:8]]
    require(len({row["source_group"] for row in sources}) == 8, "Held actor groups overlap")
    return original | {"schema": SCHEMA, "protocol": PROTOCOL, "development_freeze_sha256": DEVELOPMENT_FREEZE,
                       "candidate_sources": sources, "confirmation_sha256": sha(__file__), "interval_sha256": sha(measured.__file__)}


def edit(frames, index, role):
    start = 8 + (5 * index) % 13 + (3 if role == "texture" else 0)
    length = (15, 7, 19, 11, 21, 9, 17, 13)[index]
    stop = start + length
    x, y = 129 + (index * 59) % 360, 86 + 42 * (index % 5)
    w, h = (120, 88) if role == "texture" else (144, 88)
    altered = frames.copy()
    original = frames[start:stop, y:y+h, x:x+w]
    if role == "texture":
        changed = frames[start:stop, y:y+h, x-22:x+w-22]
    else:
        changed = np.clip(1.18 * original.astype(np.float32) - 52, 0, 255)
        if role == "fade":
            opacity = np.minimum(np.arange(1, length + 1), np.arange(length, 0, -1)) / ((length + 1) // 2)
            opacity = opacity[:, None, None, None]
            changed = opacity * changed + (1 - opacity) * original
    altered[start:stop, y:y+h, x:x+w] = changed.astype(np.uint8)
    mask = np.any(altered != frames, axis=3)
    require(np.flatnonzero(mask.any(axis=(1, 2))).tolist() == list(range(start, stop)), "Authored bounds differ")
    return altered, [start, stop], mask


def check(rows):
    summary = development.summarize(rows)
    groups, improved, timing = set(), 0, 0
    for row in rows:
        old, new = row["arms"]["serving"], row["arms"]["refined"]
        if row["role"] != "redraw" or old is None or new is None:
            continue
        old_error, new_error = (sum(abs(a - b) for a, b in zip(row["truth"], arm["frames"], strict=True)) for arm in (old, new))
        if new_error < old_error:
            improved += 1
            timing += row["aligned"]
            groups.add((row["bank"], row["group"]))
    return summary | {"redraw_endpoint_gains": improved, "redraw_endpoint_gain_groups": len(groups),
                      "timing_qualified_redraw_endpoint_gains": timing, "passes": summary["passes"] and len(groups) >= 2 and timing >= 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run"))
    parser.add_argument("development", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("exclusion", type=Path, nargs="+")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    seal = args.run / "freeze.json"
    lock = freeze(args)
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(seal, lock)
        return
    require(read(seal) == lock, "Frozen boundary confirmation changed")
    require(sha(development.__file__) == lock["code"][Path(development.__file__).name], "Trial runner changed")
    development.boundaries = measured
    development.edit = edit
    with (args.run / "rows.jsonl").open("x") as out:
        held = development.held_sources(lock["candidate_sources"], args.run)
        redraw = development.retained(lock["neural"], lock["retained"])
        for row in chain(held, redraw):
            out.write(json.dumps(row, allow_nan=False) + "\n")
            out.flush()
    rows = [json.loads(line) for line in (args.run / "rows.jsonl").read_text().splitlines()]
    public = {key: value for key, value in lock.items() if key not in {"candidate_sources", "neural", "retained"}}
    public.update(freeze_sha256=sha(seal), source_sha256=[r["file_sha256"] for r in lock["candidate_sources"]])
    summary = check(rows)
    write(args.run / "result.json", {"freeze": public, "rows": rows, "summary": summary})
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
