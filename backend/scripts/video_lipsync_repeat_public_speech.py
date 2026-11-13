# SPDX-License-Identifier: Apache-2.0
"""Verify frozen repeat-speech checks on independently published conversational recordings."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from scripts import review_video_lipsync_repeat_public_speech as review
from scripts import video_lipsync_repeat_fresh_actors as actors
from scripts import video_lipsync_repeat_motion as base
from scripts import video_lipsync_repeat_picture_guard as picture
from scripts.video_common_sources_v89 import read, require, sha, write

SOURCES = actors.ROOT / "docs/evaluation-panels/video-lipsync-repeat-public-speech-sources-2026-11-13.json"


def contract(run):
    public = read(SOURCES)
    require(len(public["sources"]) == 4 and all(sha(run / f'{row["id"]}.webm') == row["sha256"] for row in public["sources"]),
            "Public source file differs from its digest")
    parent = read(actors.PANEL)
    require(parent["point"] == .65 and parent["gate"] == base.GATE, "The published operating point differs")
    return {"schema": review.SCHEMA, "source_plan_sha256": sha(SOURCES), "parent_panel_sha256": sha(actors.PANEL),
            "point": parent["point"], "gate": parent["gate"],
            "bindings": base.prior.binding() | {Path(p).name: sha(p) for p in (__file__, base.__file__, picture.__file__, review.__file__)},
            "runtime": {"opencv": cv2.__version__, "numpy": np.__version__,
                        "ffmpeg": base.prior.delivery.command("ffmpeg", "-version").decode().splitlines()[0]}}


def prepare(source, target, offset):
    width, height = 480, 360
    pictures = base.prior.N
    codec = "pcm_s16le"
    subprocess.run(("ffmpeg", "-nostdin", "-v", "error", "-ss", str(offset), "-i", str(source),
                    "-map", "0:v:0", "-map", "0:a:0", "-t", "2.52", "-vf",
                    f"fps=25,scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "1",
                    "-video_track_timescale", "16000", "-c:a", codec, "-ar", "16000", "-ac", "1", str(target)),
                   check=True, timeout=20, capture_output=True)
    prepared = base.prior.original(target, sha(target))
    if prepared is None or len(prepared[0]) != pictures:
        return None
    return prepared


def qualify(run, public, detector):
    results, selected = [], []
    for i, row in enumerate(public["sources"], 301):
        outcomes = []
        state = {"group": i, "source_sha256": row["sha256"], "attempts": outcomes}
        results.append(state)
        for offset in row["offsets"]:
            if "selected" in state:
                outcomes.append({"offset": offset, "outcome": "after_selection"})
                continue
            with TemporaryDirectory(prefix="source-", dir=run) as temp:
                source = run / f'{row["id"]}.webm'
                target = Path(temp) / "excerpt.mov"
                prepared = prepare(source, target, offset)
                disposition = "outside_shared_duration"
                if prepared is not None:
                    frames, pcm, _, _, _ = prepared
                    first = 14 + i % 3
                    points = [base.detect_faces(cv2.cvtColor(frames[k], cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
                              for k in (first, first + 5, first + 10)]
                    speech = base.lipsync.vad(pcm.astype(np.float32) / 32768).speech
                    disposition = "outside_source_geometry" if any(len(found) != 1 for found in points) else "insufficient_speech"
                    if disposition == "insufficient_speech" and speech.mean() >= .45:
                        digest = sha(target)
                        state["selected"] = offset
                        state["prepared_sha256"] = digest
                        info = {"group": i, "ordinal": i, "bank": "public-conversation", "sha256": digest}
                        selected.append((info, prepared))
                        disposition = "selected"
                outcomes.append({"offset": offset, "outcome": disposition})
    return results, selected


def score(run, lock):
    public = read(SOURCES)
    cv2.setNumThreads(1)
    detector = base.load_yunet(base.prior.WEIGHT)
    dispositions, selected = qualify(run, public, detector)
    require(len(selected) >= 2, "Insufficient separately published speech sources qualified")
    measured = []
    for i, (row, source) in enumerate(selected):
        donor, donor_source = selected[(i + 1) % len(selected)]
        outcomes = base.assess(run, row, source, donor_source, detector, picture.measurement)
        require(outcomes is not None, "Qualified clip geometry changed")
        measured.append(outcomes | {"donor_group": donor["group"]})
        print("measured conversational recording", row["group"], flush=True)
    receipt = {"freeze": lock, "freeze_sha256": sha(run / "freeze.json"), "qualification": dispositions, "rows": measured}
    receipt["summary"] = review.summarize(receipt)
    write(run / "receipt.json", receipt)
    print(receipt["summary"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score"))
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    lock = contract(args.run)
    if args.phase == "freeze":
        write(args.run / "freeze.json", lock)
    else:
        require(lock == read(args.run / "freeze.json"), "Frozen public speech plan differs")
        score(args.run, lock)


if __name__ == "__main__":
    main()
