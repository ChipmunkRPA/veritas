"""Public authored crop/point arithmetic; no photographs or neural dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from scripts import image_iml_zoom_v16 as primary
from scripts import review_image_iml_zoom_v16 as independent

ROOT = Path(__file__).resolve().parents[2]
SCIENCE = "d1165ffe510eabb38355d0511e52388bea0097a1"
FREEZE_SHA256 = "202c8a51784d86519322499f9941dafa68b9579ab3207b7ad08a57146e2be522"


def review(output):
    primary.require(not output.exists(), "Retain earlier review receipts")
    frozen = (ROOT / "docs/evaluation-panels/image-iml-zoom-v16-freeze.json").read_bytes()
    primary.require(hashlib.sha256(frozen).hexdigest() == FREEZE_SHA256, "Original prospective freeze changed")
    freeze = json.loads(frozen)
    for name in ("backend/scripts/image_iml_zoom_v16.py", "backend/scripts/review_image_iml_zoom_v16.py"):
        primary.require(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == freeze["code_sha256"][name],
            "Original prospective geometry implementation changed",
        )
    rng, maximum, geometries = np.random.default_rng(161616), 0.0, 0
    for height, width in ((67, 103), (131, 97), (312, 513)):
        grids = rng.integers(0, 256, (5, 256, 256), dtype=np.uint8)
        a, b = primary.reconstruct(grids, width, height), independent.reconstruct(grids, width, height)
        maximum = max(maximum, *[float(np.max(np.abs(a[k] - b[k]))) for k in a])
        constant = primary.reconstruct(np.full((5, 256, 256), 51, np.uint8), width, height)
        primary.require(
            np.max(np.abs(constant["zoom"] - 0.2)) < 1e-12, "Complete source edge/interior coverage differs"
        )
        geometries += 1
    raw = rng.uniform(size=(517, 683)).astype(np.float32)
    maximum = max(
        maximum, float(np.max(np.abs(primary.resize(raw, 256, 256) - independent.resize(raw, 256, 256))))
    )
    controls = [
        {
            "role": "point",
            "kind": kind,
            "format": "native",
            "label": 0,
            "status": "ok",
            "source_id": str(i),
            "responses": {
                arm: {"active_fraction": 0.5, "ungated": {"predicted_positive_pixels": 1}}
                for arm in ("whole", "zoom")
            },
        }
        for i in range(24)
        for kind in ("original", "ordinary")
    ]
    points = primary.point(controls)
    primary.require(
        all(
            p["groups_flagged"] == 0 and p["minimum_active_fraction"] > 0.5 and not p["uses_ai_or_diagnostic"]
            for p in points.values()
        ),
        "Tied source-group point permits a false positive claim",
    )
    controls[0]["label"] = 1
    refused = False
    try:
        primary.point(controls)
    except ValueError:
        refused = True
    primary.require(
        refused and maximum <= 1e-12 and not any(n in sys.modules for n in ("torch", "timm", "transformers")),
        "Source-role, arithmetic or no-model import contract failed",
    )
    record = {
        "schema": "chipmunk.image-iml-zoom-public-math-review/16",
        "science_commit": SCIENCE,
        "authored_nonsquare_source_geometries": geometries,
        "recorded_crop_grids_checked": geometries * 5,
        "genuine_point_groups": 24,
        "tied_points_and_ai_role_refusal": True,
        "maximum_scalar_grid_difference": maximum,
        "absolute_limit": 1e-12,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "source_photographs": 0,
        "neural_forwards": 0,
        "outside_human_reviewers": 0,
        "scope": "Authored grid/interpolation/point arithmetic only; no native image-model or detection-quality claim.",
    }
    with output.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    review(parser.parse_args().output)
