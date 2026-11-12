"""Replay earlier-copy overlays against frozen resized-image evidence journals."""
from __future__ import annotations

import argparse
import io
import json
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.evidence.image_reference import inspect
from scripts import image_reference_resize_transfer as trial

panel = trial.panel
PUBLISHED = Path(__file__).resolve().parents[2] / "docs/evaluation-panels/image-reference-resize-transfer-2026-11-11.json"
ARTIFACTS = {"reference_crop", "submitted_crop", "submitted_overlay"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def png(evidence, asset, mode):
    data = (evidence / asset["file"]).read_bytes()
    require(panel.shared.digest(data) == asset["sha256"], "Evidence PNG digest differs")
    with Image.open(io.BytesIO(data)) as source:
        require(source.mode == mode and source.size == (asset["width"], asset["height"]), "Evidence PNG geometry differs")
        return np.asarray(source).copy()


def native_slice(pixels, bounds):
    x, y, w, h = (bounds[field] for field in ("x", "y", "w", "h"))
    return pixels[y:y+h, x:x+w]


def verify_assets(evidence, record, reference, submitted):
    if not record["regions"]:
        require(not list(evidence.glob("*.png")) and "mask" not in record, "Empty comparison has unexpected evidence")
        return np.zeros(submitted.shape[:2], dtype=bool), 0
    gray = png(evidence, record["mask"], "L")
    require(gray.shape == submitted.shape[:2] and np.all((gray == 0) | (gray == 255)), "Mask is outside native coordinates")
    mask = gray == 255
    expected_files = {record["mask"]["file"]}
    for index, region in enumerate(record["regions"], 1):
        for role, image, bounds in (("reference_crop", reference, region["reference_region"]),
                                    ("submitted_crop", submitted, region)):
            asset = region[role]
            require(np.array_equal(png(evidence, asset, "RGB"), native_slice(image, bounds)), "Native crop pixels differ")
            expected_files.add(asset["file"])
        asset = region["submitted_overlay"]
        require(asset["file"] == f"submitted-overlay-{index:02d}.png", "Unexpected overlay identifier")
        expected_files.add(asset["file"])
        selection = native_slice(gray, region)
        with Image.new("RGBA", selection.shape[::-1]) as expected:
            expected.paste((255, 173, 46, 144), (0, 0, expected.width, expected.height), Image.fromarray(selection))
            require(np.array_equal(png(evidence, asset, "RGBA"), np.asarray(expected)), "Overlay differs from submitted mask pixels")
    require(expected_files == {path.name for path in evidence.glob("*.png")}, "Extra or missing PNG artifacts")
    return mask, len(record["regions"])


def replay(intake, work):
    sources = list(panel.paired.inputs(intake))
    rows = []
    counts = Counter()
    for index, (case, native, _) in enumerate(sources):
        reference = native["original"]
        donor = sources[(index + 1) % len(sources)][1]["original"]
        reference_body, _ = panel.paired.encoded(reference, "png")
        earlier = work / "earlier.png"
        earlier.write_bytes(reference_body)
        for role, pixels, truth, scale in trial.cases(case, native, donor):
            for fmt in panel.FORMATS:
                body, decoded = panel.paired.encoded(pixels, fmt)
                submitted = np.asarray(decoded)
                questioned = work / "submitted"
                questioned.write_bytes(body)
                with tempfile.TemporaryDirectory(dir=work) as temp:
                    evidence = Path(temp) / "evidence"
                    result = inspect(earlier, questioned, evidence_dir=evidence, allow_proportional_resize=True)
                    require(result["schema"] == "chipmunk.image-reference-comparison/3"
                            and result["reference"]["sha256"] == panel.shared.digest(reference_body)
                            and result["submitted"]["sha256"] == panel.shared.digest(body), "Comparison route or file binding differs")
                    mask, overlays = verify_assets(evidence, result, reference, submitted)
                rows.append({"slot": case["slot"], "role": role, "format": fmt, "scale": scale,
                             "source_sha256": panel.shared.digest(reference.tobytes()), "input_sha256": panel.shared.digest(body),
                             "truth_sha256": panel.shared.digest(truth.tobytes()), "status": result["status"],
                             "incumbent_status": "integer_alignment_unresolved",
                             "regions": [{key: value for key, value in region.items() if key not in ARTIFACTS}
                                         for region in result["regions"]],
                             "mask_sha256": panel.shared.digest(mask.tobytes()) if result["status"] == "comparable" else None,
                             "candidate": panel.metrics.measure(mask, truth),
                             "incumbent": panel.metrics.measure(np.zeros_like(truth), truth)})
                counts["views"] += 1
                counts["overlays_verified"] += overlays
                counts["exact_crops_verified"] += 2 * overlays
                counts["views_with_overlays" if overlays else result["status"] + "_without_overlay"] += 1
    return rows, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("earlier", "later", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    expected = json.loads(PUBLISHED.read_bytes())
    require(trial.freeze((args.earlier, args.later)) == expected["frozen_protocol"], "Frozen resize protocol differs")
    cv2.setNumThreads(1)
    counts = {}
    with tempfile.TemporaryDirectory(prefix="chipmunk-overlay-review-") as directory:
        scratch = Path(directory)
        for name, intake in (("earlier", args.earlier), ("later", args.later)):
            rows, counts[name] = replay(intake, scratch)
            journal = scratch / f"{name}.json"
            panel.shared.write(journal, rows)
            require(len(rows) == 192 and panel.shared.file_digest(journal) == expected["journal_sha256"][name]
                    and trial.rollup(rows) == expected["results"][name], "Selected regions differ from the frozen journal")
            print(name, dict(counts[name]), flush=True)
    panel.shared.write(args.output, {"schema": "chipmunk.image-reference-overlay-audit/1",
                                    "comparison_receipt_sha256": panel.shared.file_digest(PUBLISHED),
                                    "checked_journal_sha256": expected["journal_sha256"], "counts": counts})


if __name__ == "__main__":
    main()
