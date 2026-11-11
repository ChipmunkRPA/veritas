"""Replay the frozen phase cue on the qualified BG-REAL comparison views."""

from __future__ import annotations

import json
from collections import Counter
from statistics import mean

import numpy as np

from scripts import image_bayer_phase_v237 as phase
from scripts import image_bg_real_jpeg_v229 as prior

TRIAL = prior.ROOT / "backend/data/independent-evaluation/image-bg-real-jpeg-v229"
PUBLIC_RESULT = prior.ROOT / "docs/evaluation-panels/image-bg-real-jpeg-complete-v229.json"
SOURCE_FREEZE = "1d4d0161226ebfc5a1236e88a194b06dac6dbd58139a0b0ee917c1c662ba226d"
ENCODINGS = ("source_png", "jpeg75", "jpeg90")


def footprint_concordance(scores, mask):
    rows, cols = scores.shape
    tile = phase.TILE_SIZE
    covered = mask[: rows * tile, : cols * tile].reshape(rows, tile, cols, tile).any(axis=(1, 3))
    usable = np.isfinite(scores)
    inside, outside = scores[covered & usable], scores[~covered & usable]
    if not len(inside) or not len(outside):
        return None
    diffs = inside[:, None] - outside[None, :]
    return float(np.mean(diffs > 0) + 0.5 * np.mean(diffs == 0))


def run():
    sealed = prior.read(PUBLIC_RESULT)
    prior.require(sealed == prior.read(TRIAL / "summary.json"), "Exact sealed v229 result required")
    freeze_hash = sealed["freeze_sha256"]
    prior.require(freeze_hash == SOURCE_FREEZE, "Exact prepared v229 input freeze required")
    inputs, cases, index = prior.lease(TRIAL, freeze_hash)
    counts = {(fmt, kind): Counter() for fmt in ENCODINGS for kind in prior.KINDS}
    concordance = {key: [] for key in counts}
    for position, entry in enumerate(inputs["sources"]):
        images, footprints = prior.material(cases, index, position)
        prior.require(
            entry["changed_pixels"] == {kind: int(mask.sum()) for kind, mask in footprints.items()},
            "Footprint differs from the prepared study",
        )
        for kind in prior.KINDS:
            views = {"source_png": images[kind]}
            for quality in prior.QUALITIES:
                payload, decoded = prior.original.jpeg_view(images[kind], quality)
                prior.require(
                    prior.original.view_binding(payload, decoded) == entry["inputs"][str(quality)][kind],
                    "Reconstructed JPEG differs from the prepared study",
                )
                views[f"jpeg{quality}"] = decoded
            for fmt, view in views.items():
                key = fmt, kind
                result = phase.analyze_bayer_phase(np.asarray(view, dtype=np.uint8))
                counts[key][result.status] += 1
                if result.scores is not None:
                    value = footprint_concordance(result.scores, footprints[kind])
                    if value is not None:
                        concordance[key].append(value)
    prior.require(all(sum(row.values()) == len(cases) for row in counts.values()), "Incomplete coverage")
    return {
        "schema": "chipmunk.image-bayer-phase-bg-real-replay/238",
        "source_groups": len(cases),
        "v229_public_result_sha256": prior.sha(PUBLIC_RESULT),
        "v229_input_freeze_sha256": freeze_hash,
        "cue_implementation_sha256": prior.sha(prior.ROOT / "backend/scripts/image_bayer_phase_v237.py"),
        "replay_implementation_sha256": prior.sha(
            prior.ROOT / "backend/scripts/image_bayer_phase_bg_v238.py"
        ),
        "total_views": sum(sum(row.values()) for row in counts.values()),
        "encodings": {
            fmt: {
                kind: {
                    "status_counts": dict(counts[fmt, kind]),
                    "footprint_concordance_groups": len(concordance[fmt, kind]),
                    "mean_footprint_concordance": mean(concordance[fmt, kind])
                    if concordance[fmt, kind]
                    else None,
                }
                for kind in prior.KINDS
            }
            for fmt in ENCODINGS
        },
        "neural_calls": 0,
        "tuned_on_this_panel": False,
        "serving_admitted": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2, allow_nan=False))
