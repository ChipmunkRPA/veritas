"""Reproduce the earlier-copy opacity check with a pinned public photo roster."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from app.detectors.image import reference_png_opacity
from app.evidence.image_reference import inspect
from scripts.image_jpeg_copy_public_replay import _source, roster


def run(folder: Path) -> dict:
    counts = {}
    sources = roster()
    with tempfile.TemporaryDirectory() as temporary:
        earlier, later = (Path(temporary) / name for name in ("earlier.png", "submitted.png"))
        for index, entry in enumerate(sources):
            path = folder / entry["file"]
            if not path.is_file():
                path = folder / f"source-{entry['slot']:02d}.jpg"
            _, image = _source(path, entry)
            original = np.array(image.convert("RGBA"))
            x, y = 114 + index * 9, 103 + index * 5
            box = np.s_[y:y+29, x:x+33]
            for role in ("local_cutout", "partial_opacity", "local_reveal", "opaque_reencode", "hidden_rgb",
                         "global_opacity", "rotated"):
                first = original.copy()
                if role in {"local_reveal", "hidden_rgb", "rotated"}:
                    first[box][:, :, 3] = 0
                second = first.copy()
                if role in {"local_cutout", "partial_opacity", "local_reveal"}:
                    second[box][:, :, 3] = {"local_cutout": 0, "partial_opacity": 112, "local_reveal": 255}[role]
                if role == "hidden_rgb":
                    second[box][:, :, :3] = 255 - second[box][:, :, :3]
                if role == "global_opacity":
                    second[:, :, 3] = 112
                if role == "rotated":
                    second = second[::-1, ::-1].copy()
                Image.fromarray(first[:, :, :3] if role == "opaque_reencode" else first).save(earlier, compress_level=1)
                Image.fromarray(second).save(later, compress_level=1)
                old = inspect(earlier, later, allow_proportional_resize=True, allow_oriented_crop=True,
                              allow_oriented_resize=True, allow_embedded_photo=True)
                new = inspect(earlier, later, allow_png_opacity=True)
                tally = counts.setdefault(role, dict(total=0, baseline_comparable=0, baseline_flags=0,
                                                     comparable=0, flags=0, exact_locations=0))
                tally["total"] += 1
                tally["baseline_comparable"] += old["status"] == "comparable"
                tally["baseline_flags"] += bool(old["regions"])
                tally["comparable"] += new["status"] == "comparable"
                tally["flags"] += bool(new["regions"])
                selected = new["regions"]
                tally["exact_locations"] += (
                    role in {"local_cutout", "partial_opacity", "local_reveal"}
                    and new["schema"] == reference_png_opacity.SCHEMA
                    and selected == [{"x": x, "y": y, "w": 33, "h": 29, "selected_pixels": 33 * 29}]
                    and new["png_opacity"]["opacity_changed_pixels"] == 33 * 29)
    return {"schema": "chipmunk.image-png-opacity-transfer/1", "source_photos": len(sources), "rows": counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photos", type=Path, help="Public replay acquisitions or the original pinned source folder")
    print(json.dumps(run(parser.parse_args().photos), indent=2))
