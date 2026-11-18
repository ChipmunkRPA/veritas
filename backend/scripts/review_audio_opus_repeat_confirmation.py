# SPDX-License-Identifier: Apache-2.0
"""Independently recount stereo Opus repeat geometry from frozen measurements."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.dcase-opus-repeat-confirmation/1"
POSITIVES = {"repeat_10db", "repeat_18db"}
ROLES = POSITIVES | {"recorded", "ordinary_panned", "different_background", "moving_voice",
                     "repeated_foreground", "same_channel_twice"}


def read(path):
    body = path.read_bytes()
    return json.loads(gzip.decompress(body) if path.suffix == ".gz" else body)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def useful(truth, candidates):
    return bool(truth and any(all(
        2 * max(0, min(end, b) - max(start, a)) >= max(end, b) - min(start, a)
        for (start, end), (a, b) in zip(pair, truth, strict=True)
    ) for pair in candidates))


def review(plan, rows):
    by_group = {row["id"]: row for row in plan["groups"]}
    require(plan["schema"] == SCHEMA and len(by_group) == 24 and set(plan["roles"]) == ROLES
            and len({row["source_sha256"] for row in by_group.values()}) == 24
            and len({(row["voice_room"], row["voice_start"]) for row in by_group.values()}) == 24,
            "Source roster is incomplete or repeats a claimed independent excerpt")
    require(Counter(row["samples"] for row in by_group.values()) == {144000: 8, 168000: 8, 192000: 8},
            "Incomplete presentation lengths")
    expected = {(group, role) for group in by_group for role in ROLES}
    counts, gains, ordinary = defaultdict(Counter), [], []
    for row in rows:
        identity = row["group"], row["role"]
        group = by_group[row["group"]]
        require(identity in expected and row["scene"] == group["scene"], "Wrong held input identity")
        expected.remove(identity)
        n = group["samples"]
        clock = row["clock"]
        require((clock["channels"], clock["sample_rate"], clock["presentation_start_pts"],
                 clock["presentation_end_exclusive_pts"], clock["pre_skip"]) == (2, 48000, 0, n, 312)
                and clock["end_granule"] == n + 312 and clock["end_trim"] == clock["packets"] * 960 - n - 312,
                "Opus presented samples and packet clock disagree")
        require(all(len(row[key]) == 64 for key in ("parent_sha256", "opus_sha256", "pcm_sha256")),
                "Missing source or media commitment")
        truth = row["truth"]
        positive = row["role"] in POSITIVES
        require(bool(truth) == positive, "Unexpected edit label")
        if truth:
            length = (11520, 19200, 30720)[group["variant"]]
            a, c = 19341 + group["variant"] * 599, 106137 + group["variant"] * 2339
            require(truth == [[a, a + length], [c, c + length]], "Altered edit geometry")
        base, fallback = row["baseline"], row["sparse"]
        require(fallback is not None if base["status"] is None and not base["spans"] else fallback is None,
                "Fallback must run only after an assessed baseline without a reportable pair")
        additional = fallback["spans"] if fallback else []
        if fallback:
            available = fallback["surviving_lags"]
            require(fallback["tested_lags"] == (available if available <= 12 else 0)
                    and (fallback["tested_lags"] != 0 or not additional), "Fallback exceeds its frozen search budget")
        final = base["spans"] + additional
        for ((a, b), (c, d)) in final:
            require(0 <= a < b <= c < d <= n and b - a == d - c, "Reported intervals are off the native clock")
        before, after = useful(truth, base["spans"]), useful(truth, final)
        role = row["role"] if positive else "ordinary"
        for scope in ("all", str(n), row["scene"]):
            counts[scope, role].update(
                total=1, assessed=base["status"] is None, baseline_flagged=bool(base["spans"]), flagged=bool(final),
                baseline_located=before, located=after, baseline_misplaced=positive and bool(base["spans"]) and not before,
                misplaced=positive and bool(final) and not after,
            )
        if after and not before:
            gains.append({"group": row["group"], "scene": group["scene"], "role": role, "native_samples": n,
                          "truth": truth, "pairs": additional})
        if not positive and additional:
            ordinary.append({"group": row["group"], "role": row["role"], "pairs": additional})
    require(not expected, "Missing frozen deliverables")
    return {"counts": {"/".join(key): dict(value) for key, value in counts.items()}, "new_locations": gains,
            "additional_ordinary_flags": ordinary}


def pack(folder):
    path = folder / "freeze.json"
    plan, held = read(path), read(folder / "held.json.gz")
    freeze_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    require(freeze_sha == held["freeze_sha256"], "Run does not correspond to this exact source freeze")
    return {"plan": plan, "freeze_sha256": freeze_sha, "rows": held["rows"], "summary": review(plan, held["rows"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("directory", type=Path)
    build.add_argument("output", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("receipt", type=Path)
    verify.add_argument("--native", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        report = pack(args.directory)
        contents = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        with args.output.open("xb") as target:
            target.write(gzip.compress(contents, mtime=0))
    else:
        report = read(args.receipt)
        require(review(report["plan"], report["rows"]) == report["summary"], "Independent numbers differ from the ledger")
        if args.native:
            require(pack(args.native) == report, "Native input/output replay differs from the public ledger")
    print(json.dumps(report["summary"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
