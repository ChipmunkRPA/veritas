# SPDX-License-Identifier: Apache-2.0
"""Compare exact paired-picture interval boundaries using verified native pictures."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video.decode import probe_frames
from app.evidence import video_reference as reference
from app.evidence import video_reference_boundaries as boundaries
from app.evidence import video_reference_local as local
from app.evidence import video_reference_photometric as luma
from scripts import video_reference_chroma_transfer as paired
from scripts import video_reference_extent_controls as media
from scripts.video_common_sources_v89 import read, require, sha, write

SCHEMA = "chipmunk.video-reference-boundary-transfer/1"
ROLES = ("grade", "texture", "fade", "delivery", "global", "wrong")
SOURCE_FIELDS = ("asset_index", "source_group", "file", "file_sha256", "expected")
PROTOCOL = {
    "measure": "Use the fixed serving local luma selector and rectangle. Refine only its time range from the original seed tile, requiring consecutive above-point measurements and abrupt boundaries relative to both flanks. The output abstains unless both endpoints qualify. The existing point and region stay fixed.",
    "held": "Use the first eight archived CREMA-D originals by asset index after excluding the actor groups of the previous paired luma and extent studies. Test two newly authored abrupt local changes and one gradual local transition per group at prespecified odd lengths, plus neutral H.264 delivery, global grade and wrong-source cases. These cameras had prior exposure in unrelated project experiments.",
    "diagnostic": "Use 17 previously exposed native LivePortrait and MuseTalk redraws, without fitting or modifying the candidate, and report the independently timing-qualified subset. This is not a fresh neural holdout.",
    "gate": "For fresh abrupt edits require at least 12 of 16 exact intervals, an increase of at least six across four groups, zero worsening by more than one frame at either edge, and at most one newly asserted abrupt fade. For native redraws, combined endpoint error may not worsen on either generator or timing-qualified median, and may worsen by more than two frames per endpoint on at most one clip. Neutral, wrong-source, and existing finding status must stay unchanged. No previously supported abrupt or neural crop may fall below 0.10 pixel IoU at the selected middle picture.",
}


def freeze(args):
    previous = args.previous
    require(sha(previous / "held.json") == paired.PRIOR, "Earlier mouth comparison differs")
    candidates = [json.loads(line) for path in sorted(media.DIR.glob("extract-*/output/cases.jsonl"))
                  for line in path.read_text().splitlines()]
    groups = {r["asset_index"]: r["source_group"] for r in candidates}
    excluded = set()
    for path in args.exclusion:
        roster = read(path)
        for row in roster.get("sources", roster.get("roster", [])):
            group = row.get("source_group", row.get("group", groups.get(row.get("asset_index"))))
            if isinstance(group, str) and group.startswith("CREMA-D:"):
                excluded.add(group)
    require(len(excluded) >= 24, "Prior comparison group exclusions are incomplete")
    eligible = [row for row in candidates if row["condition"] == "original" and row["status"] == "ready"
                and row["source_group"] not in excluded and len(row["expected"]["rgb_sha256"]) >= 50]
    sources = [{key: row[key] for key in SOURCE_FIELDS}
               for row in sorted(eligible, key=lambda item: item["asset_index"])[:8]]
    require(len({row["source_group"] for row in sources}) == 8, "Held actor groups overlap")
    roster = read(previous / "freeze.json")["roster"]
    neural = [r for r in read(previous / "held.json")["rows"] if r["role"] in {"source_time_neural", "same_time_neural"}]
    require(len(neural) == 17, "Redraw roster differs")
    code = (__file__, boundaries.__file__, luma.__file__, local.__file__, reference.__file__, media.__file__, paired.__file__)
    return {"schema": SCHEMA, "protocol": PROTOCOL, "candidate_sources": sources,
            "exclusions_sha256": [sha(path) for path in args.exclusion],
            "neural_roster_sha256": sha(previous / "freeze.json"), "neural_receipt_sha256": paired.PRIOR,
            "neural": neural, "retained": {r["carrier_sha256"]: r for r in roster},
            "code": {Path(name).name: sha(name) for name in code}, "numpy": np.__version__, "opencv": cv2.__version__,
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True, timeout=5).splitlines()[0]}


def edit(frames, index, role):
    start = 10 + (7 * index) % 13
    length = (7, 9, 11, 13, 15, 17, 19, 21)[index]
    if role == "texture":
        start -= 3
        length = 7 + 2 * ((index + 3) % 8)
    stop = start + length
    x, y = 85 + (index * 43) % 360, 113 + 37 * (index % 4)
    w, h = (120, 80) if role == "texture" else (152, 88)
    changed = frames.copy()
    region = frames[start:stop, y:y+h, x:x+w]
    if role == "texture":
        region = frames[start:stop, y:y+h, x+17:x+w+17]
    else:
        region = np.clip(region.astype(np.float32) * .60 + 96, 0, 255)
        if role == "fade":
            alpha = np.minimum(np.arange(1, length + 1), np.arange(length, 0, -1)) / ((length + 1) // 2)
            alpha = alpha[:, None, None, None]
            region = region * alpha + (1 - alpha) * frames[start:stop, y:y+h, x:x+w]
    changed[start:stop, y:y+h, x:x+w] = region.astype(np.uint8)
    mask = np.any(changed != frames, axis=3)
    require(np.flatnonzero(mask.any(axis=(1, 2))).tolist() == list(range(start, stop)), "Authored edit bounds differ")
    return changed, [start, stop], mask


def analyze(paths, mask):
    tables = tuple(probe_frames(path, cap=105, timeout_s=15) for path in paths)
    clocks = tuple(reference._elapsed(table) for table in tables)
    require(clocks[0] == clocks[1] and luma.eligible(paths, tables, clocks[0]), "Missing matched luma picture scope")
    rects = local.rectangles(tables[0].height, tables[0].width)
    signals, errors, grid, hashes = [], [], [], []
    for a, b in reference._frame_pairs(paths, tables):
        signal, error, cells = luma.measurements(a.rgb, b.rgb, rects)
        signals.append(signal)
        errors.append(error)
        grid.append(cells)
        hashes.append([hashlib.sha256(frame.rgb).hexdigest() for frame in (a, b)])
    require(len(hashes) == len(tables[0]), "Incomplete paired native decode")
    status, found = luma.assess(signals, errors, rects)
    result = {"status": status, "pictures": len(hashes), "first_pts": [t.pts[0] for t in tables],
              "steps": [t.pts[1] - t.pts[0] for t in tables], "time_bases": [t.time_base for t in tables],
              "files_sha256": [sha(path) for path in paths], "arms": {"serving": None, "refined": None}}
    if found is None:
        return result
    tile = found["tile"]
    xs, ys, _, _ = rects
    seed = int(np.flatnonzero((xs == tile["x"]) & (ys == tile["y"]))[0])
    result["seed"] = tile
    values = np.asarray(signals)[:, seed]
    result["series"] = [float(value) if np.isfinite(value) else None for value in values]
    proposed = boundaries.refine(values, found["frames"])
    found["tile"] = luma.expanded_region(np.asarray(grid), found, constrained=False)
    paired.evidence(paths, tables, found, hashes, mask)
    result["arms"]["serving"] = found
    if proposed is not None:
        start, stop = proposed
        precise = {"frames": proposed, "tile": found["tile"], "score": found["score"],
                   "samples": {"before": start - 1, "during": (start + stop) // 2, "after": stop}}
        paired.evidence(paths, tables, precise, hashes, mask)
        result["arms"]["refined"] = precise
    return result


def held_sources(sources, run):
    for index, source in enumerate(sources):
        frames = media.source(source)
        earlier = run / f"held-{index:02d}-earlier.mp4"
        media.encode(earlier, frames, 23)
        for role in ROLES:
            submitted = run / f"held-{index:02d}-{role}.mp4"
            altered, truth, mask = frames, None, None
            if role in {"grade", "texture", "fade"}:
                altered, truth, mask = edit(frames, index, role)
            elif role == "global":
                altered = np.clip(frames.astype(np.float32) * .88 + 17, 0, 255).astype(np.uint8)
            elif role == "wrong":
                altered = media.source(sources[(index + 1) % len(sources)])
            media.encode(submitted, altered, 28 if truth else 34)
            yield {"bank": "crema", "group": index, "role": role, "truth": truth,
                   "aligned": False} | analyze((earlier, submitted), None if mask is None else mask.__getitem__)
        print("held source", index + 1, flush=True)


def retained(neural, roster):
    for item in neural:
        first, second = (roster[item[key]] for key in ("earlier_sha256", "submitted_sha256"))
        paths = tuple(Path(row["carrier"]) for row in (first, second))
        require(all(sha(path) == row["carrier_sha256"] for path, row in zip(paths, (first, second), strict=True)),
                "Retained redraw carrier hash differs")
        truth = second.get("edit")
        if truth is None:
            table = probe_frames(paths[1], cap=105)
            ticks = [t * table.tb for t in table.pts]
            truth = [ticks.index(Fraction(t)) for t in second["native_truth"]]
        mask = paired.native_truth([Path(r["native"]) for r in (first, second)], (first, second))
        yield {"bank": item["family"], "group": item["group"], "role": "redraw", "truth": truth,
               "aligned": item["aligned"], "case": item["case"]} | analyze(paths, mask)
        print("retained", item["family"], item["case"], flush=True)


def summarize(rows):
    counts, gains, old_errors, new_errors = Counter(), set(), {}, {}
    for row in rows:
        role, truth = row["role"], row["truth"]
        category = "abrupt" if role in {"grade", "texture"} else role
        old, new = (row["arms"][arm] for arm in ("serving", "refined"))
        counts[category] += 1
        counts[category + "_old"] += old is not None
        counts[category + "_refined"] += new is not None
        if not truth or not old:
            continue
        final = new or old
        old_delta, new_delta = ([abs(a - b) for a, b in zip(choice["frames"], truth, strict=True)]
                                for choice in (old, final))
        counts[category + "_old_exact"] += not any(old_delta)
        counts[category + "_final_exact"] += not any(new_delta)
        counts[category + "_worse"] += max(b - a for a, b in zip(old_delta, new_delta, strict=True)) > (1 if category == "abrupt" else 2)
        if category == "abrupt" and any(old_delta) and not any(new_delta):
            gains.add(row["group"])
        if category == "redraw":
            for name in [row["bank"], *("timing_qualified",) * bool(row["aligned"])]:
                old_errors.setdefault(name, []).append(sum(old_delta))
                new_errors.setdefault(name, []).append(sum(new_delta))
        old_pixel, new_pixel = (item["pixel_support"] for item in (old, final))
        area = old["tile"]["w"] * old["tile"]["h"]
        old_covered, new_covered = (10 * p["covered"] >= p["changed"] + area - p["covered"]
                                    for p in (old_pixel, new_pixel))
        counts[category + "_pixel_losses"] += old_covered and not new_covered
    require([counts[key] for key in ("abrupt", "fade", "delivery", "global", "wrong", "redraw")] == [16, 8, 8, 8, 8, 17],
            "Boundary panel is incomplete")
    medians = {name: [float(np.median(values)), float(np.median(new_errors[name]))] for name, values in old_errors.items()}
    gate = (counts["abrupt_final_exact"] >= 12 and counts["abrupt_final_exact"] - counts["abrupt_old_exact"] >= 6
            and len(gains) >= 4 and counts["abrupt_worse"] == 0 and counts["fade_refined"] <= 1
            and all(new <= old for old, new in medians.values()) and counts["redraw_worse"] <= 1
            and all(counts[key + "_refined"] == 0 for key in ("delivery", "global", "wrong"))
            and counts["abrupt_pixel_losses"] == counts["redraw_pixel_losses"] == 0)
    return dict(counts) | {"source_groups_with_new_exact_range": len(gains), "redraw_median_total_endpoint_error": medians,
                           "passes": gate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run"))
    parser.add_argument("previous", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("exclusion", type=Path, nargs="+")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    lock = freeze(args)
    seal = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(seal, lock)
        return
    require(read(seal) == lock, "The boundary protocol changed")
    with (args.run / "rows.jsonl").open("x") as out:
        for cohort in (held_sources(lock["candidate_sources"], args.run), retained(lock["neural"], lock["retained"])):
            for row in cohort:
                out.write(json.dumps(row, allow_nan=False) + "\n")
                out.flush()
    rows = [json.loads(line) for line in (args.run / "rows.jsonl").read_text().splitlines()]
    public = {key: lock[key] for key in ("schema", "protocol", "exclusions_sha256", "neural_roster_sha256",
                                        "neural_receipt_sha256", "code", "numpy", "opencv", "ffmpeg")}
    public.update(freeze_sha256=sha(seal), source_sha256=[r["file_sha256"] for r in lock["candidate_sources"]])
    summary = summarize(rows)
    write(args.run / "result.json", {"freeze": public, "rows": rows, "summary": summary})
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
