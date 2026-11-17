# SPDX-License-Identifier: Apache-2.0
"""Rebuild and check the exact delivered clips and supporting native frames."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

from app.bench import independent
from app.bench import video_av_local_agreement as review
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.detectors.video.decode import decode_frames, probe_frames
from scripts.export_video_lip_correspondence import frame_range
from scripts.export_video_lip_sequence import _readbacks
from scripts.video_av_local_agreement import save


def witnesses(video, row, proposal):
    table = probe_frames(video, cap=120, timeout_s=15, max_pixels=1280 * 720)
    at = proposal["bracket_frames"]
    pictures = list(decode_frames(video, table, at, timeout_s=15, max_pixels=1280 * 720))
    if [image.index for image in pictures] != at or any(
        not image.pts_verified or image.pts * table.tb != row["pts"][image.index] * Fraction(row["time_base"])
        for image in pictures
    ):
        raise ValueError("Native evidence images are missing or changed")
    return [
        {"frame": image.index, "pts": image.pts, "rgb_sha256": sha256(image.rgb.tobytes()).hexdigest()}
        for image in pictures
    ]


def export(plan, frozen, result, sources, output):
    review.review(result, frozen)
    measured = review.read(result)
    if len(measured["inputs"]) != 1 or measured["inputs"][0][0] != review.digest(plan):
        raise ValueError("The public source plan differs from the assessed split")
    clips = review.read(plan)["rows"]
    if {row["target"]["front_sha256"] for row in clips} != {row["source_sha256"] for row in measured["rows"]}:
        raise ValueError("Native rendering requires the complete held case set")
    cases = {}
    for row in measured["rows"]:
        source = row["source_sha256"]
        for carrier, values in row["cases"].items():
            for role, value in values.items():
                cases[source, carrier, role] = value | {
                    "speaker": row["speaker"],
                    "picture_sha256": source,
                    "carrier": carrier,
                    "role": role,
                    "pts": row["frame_pts"],
                    "time_base": row["time_base"],
                    "truth": row["truth"],
                    "source_pcm_sha256": value["audio_sha256"],
                }
    output.mkdir(mode=0o700)
    manifests, indexes, results = [], [], []
    totals = Counter()
    for path, original, times, row in _readbacks(clips, cases, sources, output):
        carrier, role = row["carrier"], row["role"]
        choice = measured["choices"][carrier]
        candidate = row["proposals"][choice["arm"]]["candidate"]
        befores = {
            "id": path.stem,
            "input_sha256": review.digest(path),
            "source_sha256": row["picture_sha256"],
        }
        target = frame_range(row["truth"], times) if role == "different_words" else None
        manifests.append(
            befores
            | {
                "modality": "video",
                "split": "confirmation",
                "group": row["speaker"],
                "label": int(target is not None),
                "condition": f"{carrier}/{role}",
                "family": "Lombard Grid",
                "license": {
                    "id": "CC BY 4.0",
                    "evaluation_permitted": True,
                    "redistribution_permitted": True,
                },
                "localization_truth": {
                    "unit": "frames",
                    "source_units": len(times),
                    "target_units": target[1] - target[0] if target is not None else 0,
                },
            }
        )
        indexes.append(
            {
                "id": path.stem,
                "input": path.relative_to(output).as_posix(),
                "source": original.relative_to(output).as_posix(),
            }
        )
        witness = {}
        if not row["baseline"] and candidate and candidate["score"] > choice["point"]:
            witness = {"comparison": candidate, "source_frames": witnesses(path, row, candidate)}
            if target:
                selected = frame_range(candidate["span"], times)
                totals[carrier] += review.useful(selected, target)
        results.append(
            befores
            | {
                "carrier": carrier,
                "role": role,
                "source_pcm_sha256": row["source_pcm_sha256"],
                "truth_frames": target,
            }
            | witness
        )
    manifest = {"schema": independent.SCHEMA, "cases": manifests}
    checked = independent.audit_manifest(manifest)
    index = {"schema": INDEX_SCHEMA, "manifest_sha256": checked["manifest_sha256"], "cases": indexes}
    inputs = verify_inputs(manifest, index, output)
    save(
        output / "review.json.gz",
        {
            "manifest": manifest,
            "file_index": index,
            "verified": inputs,
            "deliveries": results,
            "added_correct_at_frame_resolution": dict(totals),
            "held_sha256": review.digest(result),
        },
    )
    print(inputs, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "frozen", "result", "sources", "output"):
        cli.add_argument(name, type=Path)
    export(**vars(cli.parse_args()))
