# SPDX-License-Identifier: Apache-2.0
"""Replay the Lombard cohort arithmetic and exact source clock, pixel and PCM evidence."""

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


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def command(*args):
    return subprocess.check_output(args, timeout=15, stdin=subprocess.DEVNULL)


def tally(rows):
    counts = Counter()
    chosen = []
    keys = set()
    for row in rows:
        counts.update(("planned", row["split"], row["audio_status"], row["visual_status"], row["status"]))
        if row["status"] != "qualified_original_event":
            continue
        audio, visual, edit = (
            [Fraction(s) for s in row[f"{kind}_range_s"]] for kind in ("audio", "visual", "native_edit")
        )
        delta = (sum(visual) - sum(audio)) / 2
        require(
            delta == Fraction(row["signed_midpoint_delta_s"])
            and abs(delta) <= Fraction(2, 25)
            and edit[0] <= min(audio[0], visual[0])
            and max(audio[1], visual[1]) < edit[1],
            "Source event arithmetic differs",
        )
        require(row["native_edit_frames"][1] - row["native_edit_frames"][0] == 16, "Native span changed")
        key = row["group"], row["condition"]
        if key not in keys:
            keys.add(key)
            chosen.append(row["ordinal"])
    return {"counts": dict(counts), "scheduled": chosen, "source_groups": len({group for group, _ in keys})}


def audit(run, sources, sound, visual, qualified):
    counts = Counter()
    for row, audio, picture, result in zip(sources, sound, visual, qualified, strict=True):
        require(
            row["ordinal"] == audio["ordinal"] == picture["ordinal"] == result["ordinal"],
            "Source roster differs",
        )
        front, wave = (run / "sources" / row["inputs"][k]["file"] for k in ("front", "audio"))
        require(
            sha(front) == picture["video_sha256"] == row["inputs"]["front"]["sha256"]
            and sha(wave) == audio["audio_sha256"] == row["inputs"]["audio"]["sha256"],
            "Publisher member bytes differ",
        )
        table = json.loads(
            command(
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=best_effort_timestamp:stream=codec_name,time_base,width,height",
                "-of",
                "json",
                str(front),
            )
        )
        [track] = table["streams"]
        pts = [f["best_effort_timestamp"] for f in table["frames"]]
        require(
            track["codec_name"] == "h264"
            and track["time_base"] == picture["time_base"]
            and pts == picture["pts"],
            "Native picture clock differs",
        )
        data = command(
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(wave),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "-",
        )
        require(
            len(data) == 4 * audio["samples"] and digest(data) == audio["native_pcm_sha256"],
            "Native float PCM differs",
        )
        counts.update(front_files=1, audio_files=1, picture_pts=len(pts), audio_samples=len(data) // 4)
        if "range_samples" in audio:
            a, b = audio["range_samples"]
            require(
                b - a == 240 and digest(data[4 * a : 4 * b]) == audio["burst_native_pcm_sha256"],
                "Acoustic slice differs",
            )
            counts["burst_excerpts"] += 1
        if "anchors" in picture:
            anchors = picture["anchors"]
            indices = [x["frame"] for x in anchors]
            select = "+".join(f"eq(n\\,{i})" for i in indices)
            rgb = command(
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-threads",
                "1",
                "-i",
                str(front),
                "-an",
                "-vf",
                f"select={select}",
                "-fps_mode",
                "passthrough",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "-",
            )
            width, height = track["width"], track["height"]
            stride = width * height * 3
            require(len(rgb) == stride * len(indices), "Visual bracket readback differs")
            for pos, item in enumerate(anchors):
                frame = rgb[pos * stride : (pos + 1) * stride]
                x0, y0, x1, y1 = item["box_xyxy"]
                crop = b"".join(frame[(y * width + x0) * 3 : (y * width + x1) * 3] for y in range(y0, y1))
                require(
                    pts[item["frame"]] == item["pts"]
                    and digest(frame) == item["rgb_sha256"]
                    and digest(crop) == item["crop_sha256"],
                    "Visible release source/crop differs",
                )
                counts["observed_native_crops"] += 1
    return dict(counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("--assemble", type=Path)
    args = p.parse_args()
    if args.assemble:
        run = args.assemble
        freeze = read(run / "event-freeze.json")
        qualification = read(run / "qualification.json")
        audio, visual = (read(run / f"{k}-labels.json") for k in ("audio", "visual"))
        intake = read(run / "source-receipt.json")
        pin = sha(run / "event-freeze.json")
        require(
            intake["freeze_sha256"] == sha(run / "freeze.json")
            and freeze["source_receipt_sha256"] == sha(run / "source-receipt.json")
            and all(doc["freeze_sha256"] == pin for doc in (qualification, audio, visual)),
            "Frozen cohort ledgers differ",
        )
        private = qualification["cases"]
        rows = [{k: v for k, v in row.items() if k != "visual_anchors"} for row in private]
        results = tally(rows)
        require(
            results["scheduled"] == qualification["scheduled_render_ids"]
            and results["source_groups"] == qualification["qualified_groups"],
            "Sealed roster differs",
        )
        result = {
            "schema": "chipmunk.lombard-original-mouth-event-public/1",
            "license": "CC BY 4.0",
            "publisher_record": 3736465,
            "originals_only": True,
            "freeze_sha256": pin,
            "spec": freeze["spec"],
            "results": results,
            "rows": rows,
            "source_audit": audit(run, intake["rows"], audio["rows"], visual["rows"], private),
        }
        args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    receipt = read(args.receipt)
    require(
        receipt["originals_only"] is True
        and receipt["schema"] == "chipmunk.lombard-original-mouth-event-public/1"
        and tally(receipt["rows"]) == receipt["results"],
        "Published original-source cohort arithmetic differs",
    )
    print(json.dumps({"results": receipt["results"], "source_audit": receipt["source_audit"]}, indent=2))


if __name__ == "__main__":
    main()
