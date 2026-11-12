"""Independently verify held late repeat localization and preserved opening results."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from scripts.review_video_lossy_sequence_transfer import agreement, approximate, ffmpeg, read, sha_file

SCHEMA = "chipmunk.video-motion-late-transfer/1"
CODECS = ("h264", "vp9")
ROLES = ("late_cross", "late_both", "opening", "opening_after_gap", "ordinary", "cut", "hold", "reverse")
POSITIVE = ROLES[:4]
RGB_BYTES = 640 * 360 * 3
OPENING, EXTENDED = 225, 450


def metrics(rows):
    totals, admissible = {}, True
    for codec in CODECS:
        subset = [row for row in rows if row["codec"] == codec]
        positives = [row for row in subset if row["role"] in POSITIVE]
        controls = [row for row in subset if row["role"] not in POSITIVE]
        opening_useful = Counter()
        extension_useful = Counter()
        for row in positives:
            before = row["exact"] or row["opening"]
            after = before or row["extension"]
            scope = f"{row['cohort']}_{row['role']}"
            opening_useful[scope] += int(agreement(before, row["truth"]))
            extension_useful[scope] += int(agreement(after, row["truth"]))
        new = Counter(row["role"] for row in controls if row["extension"] is not None)
        old = Counter(row["role"] for row in controls if row["exact"] or row["opening"])
        lost = sum(agreement(row["exact"] or row["opening"], row["truth"])
                   and not agreement(row["exact"] or row["opening"] or row["extension"], row["truth"]) for row in positives)
        supported = sum(row["full_clock"] for row in subset)
        totals[codec] = {"opening_useful": dict(opening_useful), "extension_useful": dict(extension_useful),
                         "controls": len(controls), "opening_flags": dict(old), "added_flags": dict(new),
                         "lost_existing": lost, "continuous_full_clock": supported, "cases": len(subset)}
        publisher = [extension_useful[f"publisher_{key}"] for key in ("late_cross", "late_both")]
        camera = sum(extension_useful[f"camera_{key}"] for key in ("late_cross", "late_both"))
        admissible &= (min(publisher) >= 3 and sum(publisher) >= 8 and camera >= 3 and new.total() <= 1
                       and new["hold"] == 0 and lost == 0)
    return totals, admissible


def verify_public(receipt):
    rows = receipt["rows"]
    known = {row["name"]: row["cohort"] for row in receipt["sources"]}
    expected = {(name, codec, role) for name in known for codec in CODECS for role in ROLES}
    actual = {(row["source"], row["codec"], row["role"]) for row in rows}
    if (receipt["schema"] != SCHEMA or len(rows) != len(expected) or actual != expected
            or Counter(known.values()) != {"camera": 3, "publisher": 6}):
        raise ValueError("Incomplete held delivery panel")
    for row in rows:
        if (row["cohort"] != known[row["source"]] or row["full_clock"] == (row["role"] == "opening_after_gap")
                or (row["truth"] is None) == (row["role"] in POSITIVE)):
            raise ValueError("Public source or timing truth differs from its frozen role")
        for name in ("exact", "opening", "extension", "truth"):
            if (pair := row[name]) is None:
                continue
            a, b, c, d = pair
            limit = OPENING if name == "opening" else EXTENDED
            if not 0 <= a < b < c < d < limit or b - a != d - c:
                raise ValueError("Invalid matched picture positions")
        if row["extension"] is not None and (row["exact"] or row["opening"] or not row["full_clock"]
                                             or row["extension"][-1] < OPENING):
            raise ValueError("The longer comparison displaced an existing result or omitted its clock")
    if metrics(rows) != (receipt["metrics"], receipt["admitted"]):
        raise ValueError("Public arithmetic differs from the saved outcomes")


def descriptor_stream(path, hashes):
    import numpy as np

    args = ["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-threads", "1", "-copyts",
            "-i", f"file:{path}", "-map", "0:v:0", "-an", "-filter_threads", "1", "-vf", "format=rgb24",
            "-fps_mode", "passthrough", "-c:v", "rawvideo", "-f", "rawvideo", "pipe:1"]
    values = []
    with subprocess.Popen(args, stdout=subprocess.PIPE) as process:
        try:
            for digest in hashes:
                frame = process.stdout.read(RGB_BYTES)
                if len(frame) != RGB_BYTES or hashlib.sha256(frame).hexdigest() != digest:
                    raise ValueError("Independent rawvideo stream differs from its verified framehash")
                rgb = np.frombuffer(frame, dtype="u1").reshape(30, 12, 40, 16, 3)
                sums = rgb.sum(axis=(1, 3)).astype(np.float64)
                luma = (77 * sums[..., 0] + 150 * sums[..., 1] + 29 * sums[..., 2]) / (192 * 256)
                values.append(luma.reshape(-1))
            if process.stdout.read(1) or process.wait(timeout=5):
                raise ValueError("Native replay emitted additional pictures or failed")
        finally:
            if process.poll() is None:
                process.kill()
    return np.asarray(values)


def verify_native(path, row):
    from scripts.video_exact_sequence_transfer import oracle

    if sha_file(path) != row["sha256"]:
        raise ValueError("Committed delivery bytes changed")
    result = ffmpeg(path, ["-enc_time_base", row["time_base"], "-c:v", "rawvideo", "-f", "framehash", "-hash", "sha256"],
                    text=True).splitlines()
    values = [line.split(",") for line in result if not line.startswith("#")]
    ticks = [int(line[2]) for line in values]
    hashes = [line[5].strip() for line in values]
    if (f"#tb 0: {row['time_base']}" not in result or len(values) != EXTENDED or ticks != row["pts"]
            or hashes != row["rgb_sha256"] or any(int(line[4]) != RGB_BYTES for line in values)):
        raise ValueError("Independent decode returned different native pictures or clock")
    base = Fraction(row["time_base"])
    millis = [tick * base * 1000 for tick in ticks]
    expected = [40 * i + 8 * int(row["role"] == "opening_after_gap" and i >= 310) for i in range(EXTENDED)]
    if millis != expected or all(b - a == 40 for a, b in pairwise(millis)) != row["full_clock"]:
        raise ValueError("Decoded clock differs from the frozen delivery plan")
    exact = oracle(ticks, hashes, base)
    opening = extension = None
    if exact is None:
        luminance = descriptor_stream(path, hashes)
        opening = approximate(expected[:OPENING], luminance[:OPENING])["motion"]
        if opening is None and row["full_clock"]:
            extension = approximate(expected, luminance)["motion"]
    if (exact, opening, extension) != tuple(row[key] for key in ("exact", "opening", "extension")):
        raise ValueError(f"Independent repeat search disagrees on {path.name}")


def build(run, workers):
    freeze_path, record = run / "freeze.json", read(run / "native.json")
    freeze = read(freeze_path)
    if sha_file(freeze_path) != record["freeze_sha256"]:
        raise ValueError("Frozen source and threshold commitment differs")
    for group in freeze["sources"]:
        for source in group["files"]:
            if sha_file(Path(source["file"])) != source["sha256"]:
                raise ValueError("Source commitment differs")
    paths = [run / "media" / row["file"] for row in record["rows"]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(verify_native, paths, record["rows"]), 1):
            if i % 16 == 0:
                print(f"Independently replayed {i} deliveries", flush=True)
    sources = [{**group, "files": [{key: value for key, value in part.items() if key != "file"} for part in group["files"]]}
               for group in freeze["sources"]]
    rows = [{key: row[key] for key in ("source", "cohort", "role", "codec", "truth", "exact", "opening", "extension", "full_clock")}
            for row in record["rows"]]
    result, admitted = metrics(rows)
    verification = {"deliveries": len(rows), "native_pictures": len(rows) * EXTENDED, "source_files": 12,
                    "max_decode_and_describe_s": max(row["decode_and_describe_s"] for row in record["rows"]),
                    "max_motion_comparison_s": max(row["motion_comparison_s"] for row in record["rows"])}
    return {"schema": SCHEMA, "freeze_sha256": record["freeze_sha256"], "parameters": freeze["parameters"],
            "sources": sources, "rows": rows, "metrics": result, "admitted": admitted, "verification": verification}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    receipt = build(args.run, args.workers) if args.run else read(args.receipt)
    verify_public(receipt)
    if args.run:
        args.receipt.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"metrics": receipt["metrics"], "admitted": receipt["admitted"], "verification": receipt["verification"]}))
