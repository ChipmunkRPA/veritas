# SPDX-License-Identifier: Apache-2.0
"""Check connected earlier-copy video regions on retained and source-disjoint pairs."""

import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import video_reference as reference
from app.evidence import video_reference_local as local
from scripts import video_paired_native_mouth_transfer as prior
from scripts import video_reference_extent_method as extent
from scripts.video_common_sources_v89 import read, require, sha, write

LOCKS = ("1f815104bf7d855a4f79f42d42474780612422ad310141308b907746b6edcb10",
         "6f5bac4581103053f8f57031143cab0e1dfb99f218b9ea4728aca217994698ff")
NATIVE_ROOT = Path("backend/data/independent-evaluation/native-mouth-clips-v120")
CONFIRM_ROLES = {"original", "resample", "ai_aligned", "ai_earlier_320ms"}
PROTOCOL = {
    "calibration": "Keep the previous luminance point and window selection. Compare the same selected tile with the connected extent.",
    "region": "Use the center half of the selected window and both five-picture flanks, 8-pixel cells, an absolute floor 1.75 and a relative floor 7% of the brightest seed cell. Close one-cell holes, select the connected component at the seed peak and bound the rectangle area.",
    "development": "All 17 prior authored neural positives are a consumed diagnostic. Include the same independently qualified second-family case for exact extent.",
    "transfer": "All ten distinct CREMA-D confirmation sources from the earlier 30 fps paired study. Use only source picture indices, no audio. All files receive a shared new 24 fps 720 by 480 carrier, nearest-neighbor spatial resize also applied to held truth. No new synthesis or independent timing evidence is claimed.",
    "controls": "For each transfer source keep identical, global tone at CRF32, recompression at CRF32 and another source with the same clock. Include the actual ordinary local resampling condition as a real difference.",
    "acceptance": "Original 0.5 time IoU and 0.1 pixel IoU. The diagnostic must recover the exactly timed second-family edit. On transfer retain or add localization on at least 15/20 authored edits over at least seven source groups, add at least three locations over the tile and lose at most one. At most one flag on 30 neutral cases and zero wrong-source admissions.",
}


def freeze(previous):
    files = [previous / name for name in ("freeze.json", "held.json")]
    require(tuple(sha(path) for path in files) == LOCKS, "Previous frozen diagnostics differ")
    records = []
    for path in sorted(NATIVE_ROOT.glob("extract-*/output/cases.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["study_role"] == "confirmation" and row["condition"] in CONFIRM_ROLES:
                require(row["status"] == "ready", "A previously frozen native transfer is unavailable")
                records.append({key: row[key] for key in ("asset_index", "condition", "file", "file_sha256",
                                                           "edit_frames_half_open", "expected")})
    require(len(records) == 40 and len({row["asset_index"] for row in records}) == 10,
            "Expected four exact native roles per transfer source")
    bindings = {Path(path).name: sha(Path(path)) for path in (__file__, extent.__file__, local.__file__)}
    return {"protocol": PROTOCOL, "code_sha256": bindings, "prior_sha256": list(LOCKS), "roster": records,
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True, timeout=5).splitlines()[0]}


def encode(source, dest, count, quality, *, tone=False):
    vf = "scale=720:480:flags=neighbor,setsar=1,settb=1/24,setpts=N"
    if tone:
        vf += ",eq=brightness=0.045:contrast=0.94"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(source), "-map", "0:v:0", "-an",
                    "-vf", vf, "-fps_mode", "passthrough", "-enc_time_base", "1/24",
                    "-video_track_timescale", "12288", "-use_editlist", "0", "-frames:v", str(count),
                    "-c:v", "libx264", "-crf", str(quality), "-pix_fmt", "yuv420p", "-bf", "0", "-threads", "1",
                    str(dest)], check=True, timeout=30)


def compare(paths):
    tables = tuple(prior.checked(path) for path in paths)
    clock = reference._elapsed(tables[0])
    require(clock == reference._elapsed(tables[1]) and extent.eligible(paths, tables, clock),
            "Both sources need the same native clock and qualified carrier")
    signals, backgrounds, commitments = [], [], []
    rects = local.rectangles(tables[0].height, tables[0].width)
    for a, b in reference._frame_pairs(paths, tables):
        signal, background = local.measure(a.rgb, b.rgb, rects)
        signals.append(signal)
        backgrounds.append(background)
        commitments.append([hashlib.sha256(item.rgb).hexdigest() for item in (a, b)])
    require(len(commitments) == len(clock), "Carrier decoding ended early")
    status, choice = extent.assess(signals, backgrounds, rects)
    if choice is None:
        return {"status": status, "pictures": len(clock)}
    def bound(index, a, b):
        return [hashlib.sha256(item).hexdigest() for item in (a, b)] == commitments[index]
    box = extent.region(paths, tables, choice, bound)
    return {"status": status, "pictures": len(clock), "time_base": tables[0].time_base,
            "carrier_hashes": [sha(p) for p in paths], "score": choice["score"], "frames": choice["frames"],
            "tile": choice["tile"], "extent": box,
            "anchors": {role: {"index": i, "time": str(clock[i]), "rgb_sha256": commitments[i]}
                        for role, i in choice["samples"].items()}}


def overlap(row, native, truth, expected=None):
    row["truth"] = truth
    if row["status"] != "local_difference":
        return
    i = row["anchors"]["during"]["index"]
    sources = []
    for side, path in enumerate(native):
        table = probe_frames(path, cap=100, timeout_s=10)
        [frame] = list(decode_frames(path, table, [i], timeout_s=10))
        require(frame.pts_verified and frame.index == i, "Unverified native truth picture")
        if expected:
            require(hashlib.sha256(frame.rgb).hexdigest() == expected[side]["rgb_sha256"][i]
                    and frame.pts * table.tb == Fraction(expected[side]["picture_pts_ms"][i], 1000),
                    "Lossless source picture differs from its frozen record")
        sources.append(frame.rgb)
    mask = np.any(sources[0] != sources[1], axis=2).astype(np.uint8)
    if mask.shape != (480, 720):
        mask = cv2.resize(mask, (720, 480), interpolation=cv2.INTER_NEAREST)
    row["overlap"] = {"changed": int(mask.sum())}
    for arm in ("tile", "extent"):
        x, y, w, h = (row[arm][key] for key in ("x", "y", "w", "h"))
        row["overlap"][arm] = int(mask[y:y + h, x:x + w].sum())


def diagnostic(previous):
    roster = read(previous / "freeze.json")["roster"]
    records = read(previous / "held.json")["rows"]
    for record in records:
        if record["truth"] is None:
            continue
        bank = record["family"]
        earlier, altered = next(
            (base, row) for base in roster for row in roster
            if base["role"] == "original" and base["carrier_sha256"] == record["earlier_sha256"]
            and row["role"] == record["role"] and row["carrier_sha256"] == record["submitted_sha256"]
        )
        paths = [Path(r["carrier"]) for r in (earlier, altered)]
        result = compare(paths)
        selected = record["evidence"].get("luminance")
        require((result["status"] == "local_difference") == bool(selected), "Time selector changed the prior decision")
        if selected:
            require(result["frames"] == record["measurements"]["luminance"]["frames"], "Time selector moved the interval")
        clock = reference._elapsed(prior.checked(paths[0]))
        truth = [clock.index(Fraction(value)) for value in record["truth"]]
        overlap(result, [Path(r["native"]) for r in (earlier, altered)], truth)
        print("diagnostic", record["case"], result["status"], flush=True)
        yield result | {"source": record["case"], "bank": bank, "aligned": record["aligned"], "role": "neural"}


def transfer(run, lock):
    roster = lock["roster"]
    groups = sorted({r["asset_index"] for r in roster})
    for index, group in enumerate(groups):
        cases = {r["condition"]: r for r in roster if r["asset_index"] == group}
        earlier = cases["original"]
        path = Path(earlier["file"])
        native_count = len(earlier["expected"]["picture_pts_ms"])
        other = next((r for r in roster if r["condition"] == "original" and r["asset_index"] != group
                      and len(r["expected"]["picture_pts_ms"]) >= native_count), None)
        for r in cases.values():
            require(sha(r["file"]) == r["file_sha256"], "Archived transfer source differs")
        if other:
            require(sha(other["file"]) == other["file_sha256"], "Archived wrong-source control differs")
        earlier_path = run / f"{index:02d}-earlier.mp4"
        encode(path, earlier_path, native_count, 23)
        roles = (*sorted(CONFIRM_ROLES), "reencode", "global_tone")
        if other:
            roles += ("wrong_source",)
        for role in roles:
            r = cases.get(role)
            source = Path((r or (other if role == "wrong_source" else earlier))["file"])
            later = earlier_path
            if role != "original":
                later = run / f"{index:02d}-{role}.mp4"
                encode(source, later, native_count, 28 if r else 32, tone=role == "global_tone")
            row = compare((earlier_path, later))
            truth = r["edit_frames_half_open"] if r else None
            if truth:
                overlap(row, (path, source), truth, (earlier["expected"], r["expected"]))
            print("transfer", index, role, row["status"], flush=True)
            yield row | {"source": index, "bank": "crema", "role": role, "truth": truth}


def good(row, arm):
    if not row["truth"] or "overlap" not in row:
        return False
    a, b = row["frames"]
    left, right = row["truth"]
    if 2 * (min(b, right) - max(a, left)) < max(b, right) - min(a, left):
        return False
    tile = row[arm]
    counts = row["overlap"]
    return 10 * counts[arm] >= counts["changed"] + tile["w"] * tile["h"] - counts[arm]


def report(diagnostics, confirmation):
    neural = [r for r in confirmation if r["role"].startswith("ai_")]
    neutrals = [r for r in confirmation if r["role"] in {"original", "reencode", "global_tone"}]
    def counts(rows):
        return {arm: sum(good(r, arm) for r in rows) for arm in ("tile", "extent")}
    additions = sum(good(r, "extent") and not good(r, "tile") for r in neural)
    losses = sum(good(r, "tile") and not good(r, "extent") for r in neural)
    qualified = [r for r in diagnostics if r["bank"] == "musetalk" and r["aligned"]]
    return {"diagnostic": counts(diagnostics), "qualified_second_family": counts(qualified),
            "transfer": counts(neural), "extent_additions": additions, "extent_losses": losses,
            "extent_groups": len({r["source"] for r in neural if good(r, "extent")}),
            "neutral_flags": sum(r["status"] == "local_difference" for r in neutrals),
            "wrong_source_admissions": sum(r["status"] not in {"unverified_photometric_support", "insufficient_shared_picture_content"}
                                           for r in confirmation if r["role"] == "wrong_source")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "diagnostic", "transfer", "report"))
    parser.add_argument("previous", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    lock = freeze(args.previous)
    sealed = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(sealed, lock)
        return
    require(read(sealed) == lock, "Source or method freeze differs")
    if args.phase == "report":
        diagnostics, confirmation = [read(args.run / f"{name}.json") for name in ("diagnostic", "transfer")]
        results = report(diagnostics, confirmation)
        write(args.run / "result.json", {"schema": "chipmunk.video-reference-extent/1", "protocol": PROTOCOL,
                                       "freeze_sha256": sha(sealed), "diagnostic": diagnostics, "transfer": confirmation,
                                       "results": results})
        print(results)
        return
    rows = diagnostic(args.previous) if args.phase == "diagnostic" else transfer(args.run, lock)
    write(args.run / f"{args.phase}.json", list(rows))


if __name__ == "__main__":
    main()
