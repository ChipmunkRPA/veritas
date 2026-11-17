"""Replay a perspective comparison from the original image files."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from app.detectors.image import reference_heif, reference_perspective
from app.evidence.image_reference import inspect
from scripts.review_image_reference_export import MAX_JSON_BYTES, _read, _record


def review(export: Path, earlier: Path, submitted: Path) -> dict:
    data = json.loads(_read(export, MAX_JSON_BYTES), object_pairs_hook=_record)
    if (set(data) != {"comparison", "pngs"}
            or data["comparison"].get("schema") not in {reference_perspective.SCHEMA, reference_heif.SCHEMA}):
        raise ValueError("Expected a perspective comparison JSON export")
    earlier_heif = data["comparison"]["schema"] == reference_heif.SCHEMA
    with TemporaryDirectory(prefix="perspective-review-") as workspace:
        folder = Path(workspace) / "evidence"
        result = inspect(earlier, submitted, evidence_dir=folder, allow_proportional_resize=True, allow_perspective=True,
                         allow_earlier_heif=earlier_heif)
        if result != data["comparison"]:
            raise ValueError("The original images do not reproduce the comparison and its native coordinates")
        assets = {path.name: base64.b64encode(path.read_bytes()).decode("ascii") for path in sorted(folder.glob("*.png"))}
        if assets != data["pngs"]:
            raise ValueError("The exported masks or crops differ from the source-bound evidence")
    return {
        "source_hashes_verified": 2, "comparison": result["status"], "regions": len(result["regions"]),
        "compared_pixels": result.get("perspective_check", {}).get("compared_pixels", 0), "verified_assets": len(assets),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("export", "earlier", "submitted"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.export, args.earlier, args.submitted), sort_keys=True))
