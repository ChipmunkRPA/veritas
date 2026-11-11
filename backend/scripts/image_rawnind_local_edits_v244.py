"""Measure the fixed phase cue on known conventional edits of native RAW renders."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

from scripts import image_bayer_phase_bg_v238 as metrics
from scripts import image_bayer_phase_v237 as phase
from scripts import image_rawnind_phase_v239 as prior

REPOSITORY = Path(__file__).resolve().parents[2]
PUBLISHED = REPOSITORY / "docs/evaluation-panels/image-rawnind-phase-complete-v239.json"
VIEWS = ("even_copy", "odd_copy", "tone", "blur")


def interventions(pixels):
    tile = phase.TILE_SIZE
    height, width = pixels.shape[:2]
    prior.require(min(height, width) >= 24 * tile, "Native dimensions must support both distinct patches")
    extent = 3 * tile
    donor_y, donor_x = (height // (3 * tile)) * tile, (width // (3 * tile)) * tile
    target_y, target_x = 2 * donor_y, 2 * donor_x
    target = np.s_[target_y : target_y + extent, target_x : target_x + extent]
    original = pixels[target]
    for name, patch in (
        ("even_copy", pixels[donor_y : donor_y + extent, donor_x : donor_x + extent]),
        ("odd_copy", pixels[donor_y : donor_y + extent, donor_x + 1 : donor_x + extent + 1]),
        (
            "tone",
            np.clip(np.rint((original.astype(np.float32) - 127.5) * 1.12 + 130.5), 0, 255).astype(np.uint8),
        ),
        ("blur", cv2.GaussianBlur(original, (5, 5), 1.25)),
    ):
        modified = pixels.copy()
        modified[target] = patch
        changed = np.zeros((height, width), dtype=bool)
        changed[target] = np.any(patch != original, axis=2)
        prior.require(changed.any(), "The native control has no exact changed pixels")
        yield name, modified, changed, (target_y // tile, target_x // tile)


def measure(pixels):
    baseline = phase.analyze_bayer_phase(pixels)
    rows = {}
    for name, view, changed, (y, x) in interventions(pixels):
        observed = phase.analyze_bayer_phase(view)
        row = {"status": observed.status, "rank": None, "target_median_score_change": None}
        if observed.scores is not None:
            row["rank"] = metrics.footprint_concordance(observed.scores, changed)
            if baseline.scores is not None:
                delta = observed.scores[y : y + 3, x : x + 3] - baseline.scores[y : y + 3, x : x + 3]
                finite = delta[np.isfinite(delta)]
                if finite.size:
                    row["target_median_score_change"] = float(np.median(finite))
        rows[name] = row
    return rows


def run(root):
    captures = prior.frozen_roster(root)
    published = json.loads(PUBLISHED.read_bytes())
    prior.require(
        published["captures"] == len(captures) == 4 and published["roster_sha256"] == prior.ROSTER_SHA,
        "Exact prior native camera replay required",
    )
    wheel = prior.verified_wheel(root / prior.WHEEL)
    prior.require("rawpy" not in sys.modules, "Fresh decoder process required")
    rows = {name: [] for name in VIEWS}
    cv2.setNumThreads(1)
    with tempfile.TemporaryDirectory(prefix="rawnind-local-edits-") as directory:
        prior.unpack_decoder(wheel, directory)
        sys.path.insert(0, directory)
        try:
            rawpy = importlib.import_module("rawpy")
            prior.require(
                Path(rawpy.__file__).resolve().is_relative_to(Path(directory)),
                "Exact isolated decoder required",
            )
            for ordinal, source in enumerate(captures):
                path = prior.raw_path(root, ordinal, source)
                prior.verify_raw(prior.read_private(path, source["bytes"]), source)
                for name, observation in measure(prior.decode(rawpy, path)).items():
                    rows[name].append(observation)
        finally:
            sys.path.pop(0)
    aggregate = {}
    for name, group in rows.items():
        ranks = [row["rank"] for row in group if row["rank"] is not None]
        changes = [
            row["target_median_score_change"]
            for row in group
            if row["target_median_score_change"] is not None
        ]
        aggregate[name] = {
            "status_counts": dict(Counter(row["status"] for row in group)),
            "ranked_captures": len(ranks),
            "mean_footprint_concordance": mean(ranks) if ranks else None,
            "matched_captures": len(changes),
            "mean_target_median_score_change": mean(changes) if changes else None,
        }
    paired = [
        (even["rank"], odd["rank"])
        for even, odd in zip(rows["even_copy"], rows["odd_copy"], strict=True)
        if even["rank"] is not None and odd["rank"] is not None
    ]
    return {
        "schema": "chipmunk.image-rawnind-local-edits-result/244",
        "captures": len(captures),
        "ordinary_views": sum(len(group) for group in rows.values()),
        "v239_public_result_sha256": prior.sha256(PUBLISHED.read_bytes()),
        "implementation_sha256": prior.sha256(Path(__file__).read_bytes()),
        "views": aggregate,
        "paired_copy_ranked_captures": len(paired),
        "paired_copy_odd_rank_higher": sum(odd > even for even, odd in paired),
        "new_neural_calls": 0,
        "all_edits_conventional": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(run(parser.parse_args().root), indent=2, sort_keys=True, allow_nan=False))
