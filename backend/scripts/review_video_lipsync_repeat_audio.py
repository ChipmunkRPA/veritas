# SPDX-License-Identifier: Apache-2.0
"""Independently tally the paired soundtrack/mouth study and check saved native deliveries."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def overlap(a, b):
    return Fraction(max(0, min(a[1], b[1]) - max(a[0], b[0])), max(a[1], b[1]) - min(a[0], b[0]))


def tally(rows, spec):
    results = {}
    for split, groups in rows.items():
        stage = {}
        for carrier in ("pcm", "aac"):
            carrier_cases = [(r["group"], c) for r in groups for c in r["cases"] if c["carrier"] == carrier]
            require(
                len(carrier_cases) == len(groups) * (len(spec["positive"]) + len(spec["control"])),
                "Incomplete carrier panel",
            )
            carrier_result = {}
            for arm in ("moving_mouth", "native_mouth_guard"):
                measured, useful, wrong, controls, affected, current, current_measured, similar, exact = (
                    Counter() for _ in range(9)
                )
                correct_groups = set()
                for group, case in carrier_cases:
                    role = case["role"]
                    candidate = case["candidate"]
                    measured[role] += candidate["assessed"]
                    current[role] += bool(case["current"]["ranges"])
                    current_measured[role] += bool(case["current"]["measured_windows"])
                    similar[role] += bool(candidate["similar_audio"])
                    exact[role] += bool(candidate["exact_audio"])
                    match = candidate["matches"][arm]
                    require(
                        (case["truth"] is not None) == (role in spec["positive"]),
                        "Inconsistent authored role",
                    )
                    if role not in spec["positive"]:
                        controls[role] += bool(match)
                        continue
                    affected[role] += 1
                    correct = any(
                        all(
                            overlap(a, b) >= Fraction(spec["gate"]["minimum_iou"])
                            for a, b in zip(item["samples"], case["truth"], strict=True)
                        )
                        for item in match
                    )
                    useful[role] += correct
                    wrong[role] += bool(match) and not correct
                    if correct:
                        correct_groups.add(group)
                stage_gate = spec["gate"]
                checks = {
                    "locations": sum(useful.values()) >= stage_gate["minimum_added"],
                    "groups": len(correct_groups) >= stage_gate["minimum_groups"],
                    "ordinary": sum(controls.values()) <= stage_gate["maximum_ordinary"],
                    "synchronized": sum(controls[k] for k in ("matched_av", "matched_mouth_sound", "global"))
                    <= stage_gate["maximum_synchronized"],
                    "coverage": Fraction(sum(measured.values()), len(carrier_cases))
                    >= Fraction(str(stage_gate["minimum_coverage"])),
                }
                carrier_result[arm] = {
                    "cases": len(carrier_cases),
                    "measured": sum(measured.values()),
                    "correct_groups": len(correct_groups),
                    "useful": dict(useful),
                    "wrong": dict(wrong),
                    "truth": dict(affected),
                    "ordinary_flags": dict(controls),
                    "current_ranges": dict(current),
                    "current_measured": dict(current_measured),
                    "similar_audio_pairs": dict(similar),
                    "exact_audio_pairs": dict(exact),
                    "gate": checks,
                    "advance": all(checks.values()),
                }
            stage[carrier] = carrier_result
        results[split] = stage
    return results


def public_rows(run):
    groups = [read(f) for f in sorted(run.glob("group-*/results.json"))]
    return {
        split: [
            {
                "group": g["group"],
                "source_sha256": g["source_sha256"],
                "cases": [
                    {
                        **{
                            k: c[k]
                            for k in (
                                "carrier",
                                "role",
                                "truth",
                                "digest",
                                "samples",
                                "pcm_sha256",
                                "pts_sha256",
                                "frame_time_base",
                                "current",
                            )
                        },
                        "candidate": {
                            **c["candidate"],
                            "matches": {
                                arm: [
                                    {
                                        "samples": m["samples"],
                                        "frame_pairs": m["frame_pairs"],
                                        "pcm_sha256": m["pcm_sha256"],
                                        "source_rgb_sha256": [r["rgb_sha256"] for r in m["pictures"]],
                                    }
                                    for m in matches
                                ]
                                for arm, matches in c["candidate"]["matches"].items()
                            },
                        },
                    }
                    for c in g["cases"]
                ],
            }
            for g in groups
            if (g["group"] <= 4) == (split == "development")
        ]
        for split in ("development", "held")
    }


def cmd(*args):
    return subprocess.check_output(args, timeout=15, stdin=subprocess.DEVNULL)


def native(run):
    counts = Counter()
    for group_path in sorted(run.glob("group-*")):
        group = read(group_path / "results.json")
        for case in group["cases"]:
            suffix = "mov" if case["carrier"] == "pcm" else "mp4"
            file = group_path / f"{case['role']}-{case['carrier']}.{suffix}"
            if not file.exists():
                continue
            with file.open("rb") as stream:
                require(
                    hashlib.file_digest(stream, "sha256").hexdigest() == case["digest"],
                    "Delivery bytes differ",
                )
            table = json.loads(
                cmd(
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "frame=best_effort_timestamp:stream=time_base,width,height",
                    "-of",
                    "json",
                    str(file),
                )
            )
            [source] = table["streams"]
            pts = tuple(f["best_effort_timestamp"] for f in table["frames"])
            require(
                source["time_base"] == case["frame_time_base"]
                and len(pts) == 63
                and hashlib.sha256(str(pts).encode()).hexdigest() == case["pts_sha256"],
                "Independent native picture clock differs",
            )
            raw = cmd(
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(file),
                "-map",
                "0:a:0",
                "-af",
                "aformat=sample_fmts=s16:channel_layouts=mono,aresample=16000",
                "-f",
                "s16le",
                "-",
            )
            require(
                len(raw) == 2 * case["samples"] and hashlib.sha256(raw).hexdigest() == case["pcm_sha256"],
                "Independent native PCM differs",
            )
            counts.update(deliveries=1, picture_pts=len(pts), pcm_samples=len(raw) // 2)
            frames = {}
            for match in case["candidate"]["matches"]["native_mouth_guard"]:
                for item in match["pictures"]:
                    frames[item["index"]] = item["rgb_sha256"]
                for (a, b), digest in zip(match["samples"], match["pcm_sha256"], strict=True):
                    require(
                        hashlib.sha256(raw[2 * a : 2 * b]).hexdigest() == digest,
                        "Reported audio excerpt differs",
                    )
                    counts["reported_pcm_excerpts"] += 1
            if not frames:
                continue
            ids = sorted(frames)
            selection = "+".join(f"eq(n\\,{i})" for i in ids)
            rgb = cmd(
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-threads",
                "1",
                "-i",
                str(file),
                "-an",
                "-vf",
                f"select={selection}",
                "-fps_mode",
                "passthrough",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "-",
            )
            width = source["width"] * source["height"] * 3
            require(len(rgb) == width * len(ids), "Reported picture count differs")
            for j, i in enumerate(ids):
                require(
                    hashlib.sha256(rgb[j * width : (j + 1) * width]).hexdigest() == frames[i],
                    "Reported source pixels differ",
                )
                counts["reported_rgb_frames"] += 1
    return dict(counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("--assemble", type=Path)
    p.add_argument("--native-reconstruction", type=Path)
    a = p.parse_args()
    if a.assemble:
        lock = read(a.assemble / "freeze.json")
        qual = {split: read(a.assemble / f"{split}-qualification.json") for split in ("development", "held")}
        spec = {k: lock[k] for k in ("gate", "method", "positive", "control", "bindings", "sources_sha256")}
        rows = public_rows(a.assemble)
        for split in rows:
            require(
                [r["group"] for r in rows[split]] == qual[split]["qualified_groups"],
                "Qualified roster differs",
            )
        summary = tally(rows, spec)
        native_totals = Counter(native(a.assemble))
        if a.native_reconstruction:
            for folder in a.native_reconstruction.glob("group-*"):
                expected = (a.assemble / folder.name / "results.json").read_bytes()
                require(
                    (folder / "results.json").read_bytes() == expected, "Reconstructed held results differ"
                )
            native_totals.update(native(a.native_reconstruction))
        document = {
            "schema": "chipmunk.video-lipsync-repeat-audio-public/1",
            "spec": spec,
            "qualification": qual,
            "rows": rows,
            "summary": summary,
            "saved_native_replay": dict(native_totals),
            "freeze_sha256": hashlib.sha256((a.assemble / "freeze.json").read_bytes()).hexdigest(),
        }
        a.receipt.write_text(json.dumps(document, indent=2, allow_nan=False, sort_keys=True) + "\n")
    receipt = read(a.receipt)
    require(
        receipt["schema"] == "chipmunk.video-lipsync-repeat-audio-public/1"
        and tally(receipt["rows"], receipt["spec"]) == receipt["summary"],
        "Published result does not recompute",
    )
    for split, carriers in receipt["summary"].items():
        for carrier, arms in carriers.items():
            for arm, r in arms.items():
                print(
                    split,
                    carrier,
                    arm,
                    r["cases"],
                    r["measured"],
                    sum(r["useful"].values()),
                    sum(r["ordinary_flags"].values()),
                    r["gate"],
                )
    print("saved native replay", receipt["saved_native_replay"])


if __name__ == "__main__":
    main()
