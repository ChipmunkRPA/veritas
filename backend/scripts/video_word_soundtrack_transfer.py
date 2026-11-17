# SPDX-License-Identifier: Apache-2.0
"""Freeze and replay alternative soundtrack word proposals on source-clocked clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch

from app.bench import video_word_soundtrack as review
from scripts import video_word_audio_windows as windows
from scripts import video_word_content_model as readers
from scripts import video_word_content_transfer as shared


def observer(model):
    def measure(pcm, patches, valid, samples, voice, visible, digits, frame_reading):
        reading = dict.fromkeys(
            ("mouth", "mouth_unfiltered", "conditional", "word_crop", "scan_speech", "scan_mouth")
        )
        if voice and voice["span"][1] <= len(pcm):
            mouth = shared.word_window(
                patches, valid, samples, voice["span"], voice["digit"], visible, digits
            )
            reading.update(
                mouth_unfiltered=mouth,
                word_crop=windows.read_crop(model, pcm, voice["span"], voice["digit"]),
                conditional={
                    "score": voice["posterior"] / voice["measure"],
                    "margin": voice["margin"] / voice["measure"],
                },
            )
            if review.eligible_primary(voice):
                reading["mouth"] = mouth
                return reading
        candidate = windows.scan_soundtrack(model, pcm, frame_reading)
        if candidate:
            reading.update(
                scan_speech=candidate,
                scan_mouth=shared.word_window(
                    patches, valid, samples, candidate["span"], candidate["digit"], visible, digits
                ),
            )
        return reading

    return measure


def commitments(previous_path, window_weights):
    prior = readers.json_file(previous_path)
    stable = {
        "previous": readers.pin(previous_path.read_bytes()),
        "visual_points": prior["points"],
        "configuration": prior["configuration"],
        "weight": readers.pin(window_weights.read_bytes()),
        "method_pins": {
            Path(module.__file__).name: readers.pin(Path(module.__file__).read_bytes())
            for module in (review, windows, readers, shared)
        },
    }
    return prior, stable


def seal(path, rows, stable, inputs, *, phase, freeze_path=None):
    if freeze_path is None:
        if phase != "calibration":
            raise ValueError("Freeze the acoustic cutoffs on separate development clips first")
        points, frozen = review.select_points(rows), None
    else:
        freeze = readers.json_file(freeze_path)
        if any(freeze[key] != value for key, value in stable.items()):
            raise ValueError("The held procedure or weights differ from the development freeze")
        if {row["speaker"] for row in rows} & {row["speaker"] for row in freeze["rows"]}:
            raise ValueError("A held speaker contributed to acoustic selection")
        points, frozen = freeze["points"], readers.pin(freeze_path.read_bytes())
    results = review.counts(rows, points, stable["visual_points"])
    value = {
        "schema": review.SCHEMA,
        "phase": phase,
        "rows": rows,
        "points": points,
        "freeze": frozen,
        **stable,
        **inputs,
        "results": results,
    }
    shared.write(path, value)
    print(
        json.dumps(
            {
                "phase": phase,
                "sources": len(rows),
                "points": points,
                "edit_results": {
                    method: {k: v for k, v in arms["visual"].items() if k.endswith("/different_words")}
                    for method, arms in results.items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main(args):
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    previous, stable = commitments(args.previous, args.window_weights)
    if args.freeze:
        freeze = readers.json_file(args.freeze)
        if any(freeze[key] != value for key, value in stable.items()):
            raise ValueError("The held procedure differs from the acoustic freeze")
    rows, configuration = shared.collect(
        args.old_dir,
        args.source,
        args.plan,
        args.crops,
        args.baseline,
        previous,
        args.phase,
        observe=observer(windows.load(args.window_weights)),
    )
    if configuration != stable["configuration"]:
        raise ValueError("The assessed model differs from the original model freeze")
    paths = {"source": args.plan, "pictures": args.crops / "capture.json.gz", "baseline": args.baseline}
    inputs = {key: readers.pin(path.read_bytes()) for key, path in paths.items()}
    seal(args.output, rows, stable, inputs, phase=args.phase, freeze_path=args.freeze)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old_dir", "previous", "window_weights", "source", "plan", "crops", "baseline", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--phase", default="calibration", choices=("calibration", "held"))
    parser.add_argument("--freeze", type=Path)
    main(parser.parse_args())
