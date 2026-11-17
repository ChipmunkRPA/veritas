# SPDX-License-Identifier: Apache-2.0
"""Replay the frozen local speech comparison on the separate GRID camera corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import torch

from app.bench import video_av_local_agreement as bench
from app.detectors.video import av_local_agreement as model
from app.evidence.video_reference_timing import MODEL_SHA256, _landmarker
from scripts import video_av_local_agreement as trial
from scripts.video_lip_delivery_transfer import qualified_48k
from scripts.video_word_content_model import pin
from scripts.video_word_soundtrack_grid import source_data


def transfer(grid, baseline, fit, vision, assets, frozen, output):
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    sys.path.insert(0, str((assets / "runtime").resolve()))
    landmarker = assets / "face_landmarker.task"
    if bench.digest(landmarker) != MODEL_SHA256:
        raise ValueError("The published facial landmark asset differs from the reference")
    eyes, models, prior_fit = trial.read_models(fit, vision)
    freeze = bench.read(frozen)
    prior = {item["source_sha256"]: item for item in bench.read(baseline)["rows"]}
    if (
        prior_fit["implementation"] != freeze["implementation"]
        or (prior_fit["weights"] | {"vision": prior_fit["vision"]}) != freeze["models"]
    ):
        raise ValueError("The alternate camera test must use the frozen detector")
    source_plan = grid / "with-audio.json"
    inputs = json.loads(source_plan.read_bytes())
    rows = []
    with _landmarker(landmarker) as (mp, task), TemporaryDirectory(prefix="chipmunk-av-grid-") as hold:
        audio = Path(hold) / "track.m4a"
        for item in inputs:
            previous = prior.pop(item["video_sha256"])
            span, tracks, patches, valid, times, _, table, _ = source_data(grid, item, mp, task)
            if [*span] != previous["truth"] or previous["source_clock"] != {
                "pts": list(table.pts),
                "time_base": table.time_base,
            }:
                raise ValueError("The independent source timings differ")
            pictures = model.picture_windows(eyes, patches, valid)
            deliveries = {}
            for carrier in bench.CARRIERS:
                cases = {}
                for role in bench.ROLES:
                    pcm = tracks[role]
                    encoded = None
                    if carrier == "aac48":
                        pcm, _, details = qualified_48k(pcm, audio)
                        encoded = details["file_sha256"]
                    old = previous["cases"][carrier][role]
                    if (pin(pcm.tobytes()), encoded) != (old["sha256"], old["encoded_sha256"]):
                        raise ValueError(
                            "The transfer input differs from the independently measured comparator"
                        )
                    sound = model.sound(pcm)
                    outcomes = {}
                    for arm, network in models.items():
                        status, candidate = model.localize(
                            times, model.similarities(network, pictures, sound, times), len(pcm)
                        )
                        outcomes[arm] = {"status": status, "candidate": candidate}
                    cases[role] = {
                        "proposals": outcomes,
                        "baseline": old["baseline"],
                        "baseline_status": old["baseline_status"],
                        "audio_sha256": old["sha256"],
                        "carrier_sha256": encoded,
                    }
                deliveries[carrier] = cases
            rows.append(
                {
                    "speaker": f"grid/{item['speaker']}",
                    "source_sha256": item["video_sha256"],
                    "truth": list(span),
                    "condition": "grid",
                    "samples": len(tracks["original"]),
                    "cases": deliveries,
                    "native_times": times.tolist(),
                    "frame_pts": list(table.pts),
                    "time_base": table.time_base,
                }
            )
            print(f"Verified GRID source {len(rows)}/{len(inputs)}", flush=True)
    if prior:
        raise ValueError("The transfer omitted a source video")
    report = {key: freeze[key] for key in ("schema", "implementation", "models", "options", "choices")}
    outcomes = bench.results(rows, freeze["options"], freeze["choices"])
    trial.save(
        output,
        report
        | {
            "rows": rows,
            "freeze": bench.digest(frozen),
            "source": bench.digest(source_plan),
            "baseline": bench.digest(baseline),
            "outcomes": outcomes,
        },
    )
    bench.review(output, frozen)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("grid", "baseline", "fit", "vision", "assets", "frozen", "output"):
        cli.add_argument(name, type=Path)
    transfer(**vars(cli.parse_args()))
