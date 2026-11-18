# SPDX-License-Identifier: Apache-2.0
"""Test separate edit locations against an earlier PNG on public scene-text inputs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from app.bench import independent, independent_locations
from scripts import image_text_erase_localization_inputs as intake

SCHEMA = "chipmunk.image-multiple-edit-replay/1"
QUALITY = 75
RESIDUAL_FLOOR = 8
MIN_COMPONENT_PIXELS = 32
EDIT_RULE = {"minimum_iou": .30, "minimum_recall": .50}
SPLIT = "confirmation"
CONTROL_NAMES = ("unchanged_export", "ordinary_second_export")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, data) -> None:
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")


def save_mask(directory, name, mask):
    import numpy as np

    height, width = mask.shape
    body = f"P4\n{width} {height}\n".encode("ascii") + np.packbits(mask, axis=1).tobytes()
    (directory / name).write_bytes(body)
    return {"file": name, "sha256": sha(body)}


def second_edit(original, protected, stem):
    """Insert a separated ellipse of ordinary copied pixels, avoiding the AI change."""
    import cv2
    import numpy as np

    height, width = protected.shape
    ys, xs = np.ogrid[-16:17, -24:25]
    shape = (ys / 16) ** 2 + (xs / 24) ** 2 <= 1
    forbidden = cv2.dilate(protected.astype(np.uint8), np.ones((41, 41), dtype=np.uint8))
    candidates = [(x, y) for y in range(24, height - 24, 40) for x in range(32, width - 32, 48)]
    candidates.sort(key=lambda pos: sha(f"{SCHEMA}/{stem}/{pos}".encode()))
    for x, y in candidates:
        if forbidden[y-16:y+17, x-24:x+25].any():
            continue
        target = np.s_[y-16:y+17, x-24:x+25]
        source = np.s_[height-y-17:height-y+16, width-x-25:width-x+24]
        delta = np.abs(original[target].astype(np.int16) - original[source].astype(np.int16)).max(axis=2)
        if np.mean(delta[shape] >= 20) < .80:
            continue
        result, exact = original.copy(), np.zeros_like(protected)
        result[target][shape] = original[source][shape]
        exact[target] = shape & (delta > 0)
        return result, exact
    return None


def jpeg(rgb) -> bytes:
    from PIL import Image

    with io.BytesIO() as output:
        Image.fromarray(rgb).save(output, format="JPEG", quality=QUALITY, subsampling=2)
        return output.getvalue()


def source_bytes(source, stem, record):
    names = (f"text-erase-{stem}.png", f"bg-{stem}.png", f"mask-{stem}.png")
    bodies = [(source / "intake" / name).read_bytes() for name in names]
    expected = [record[f"{kind}/{stem}.png"] for kind in intake.FIELDS]
    if [sha(data) for data in bodies] != expected:
        raise ValueError("Source bytes differ from the original publisher inventory")
    return bodies


def prepare(source: Path, run: Path):
    import cv2
    import numpy as np
    from PIL import Image

    inputs = json.loads((source / "intake/intake.json").read_bytes())
    if inputs["revision"] != intake.REVISION:
        raise ValueError("Use source files from the pinned publisher revision")
    cv2.setNumThreads(1)
    masks = run / "masks"
    masks.mkdir(mode=0o700)
    deliveries = run / "deliveries"
    deliveries.mkdir(mode=0o700)
    records, masks_by_case, excluded = [], [], []
    for record in inputs["dispositions"]:
        stem = record["case"]
        if record["reason"] is not None:
            excluded.append({"source": stem, "reason": record["reason"]})
            continue
        bodies = source_bytes(source, stem, record["assets"])
        _, pair = intake.prepare_pair(bodies)
        if pair is None:
            raise ValueError("A previously qualified source no longer repeats")
        original, edited, first_mask = pair
        second = second_edit(original, first_mask, stem)
        if second is None:
            excluded.append({"source": stem, "reason": "no_distinct_second_edit"})
            continue
        pixels, second_mask = second
        combined = np.where(first_mask[..., None], edited, pixels)
        width, height = original.shape[1], original.shape[0]
        Image.fromarray(original).save(deliveries / f"{stem}.png")
        prior = sha((deliveries / f"{stem}.png").read_bytes())
        exact_masks = [save_mask(masks, f"{stem}-edit-{index}.pbm", mask)
                       for index, mask in enumerate((first_mask, second_mask), 1)]
        clean = jpeg(original)
        with Image.open(io.BytesIO(clean), formats=["JPEG"]) as image:
            ordinary_second = jpeg(np.asarray(image))
        versions = dict(zip(CONTROL_NAMES, (clean, ordinary_second), strict=True))
        versions["two_staged_changes"] = jpeg(combined)
        for condition, data in versions.items():
            case_id = f"{stem}-{condition}"
            (deliveries / f"{case_id}.jpg").write_bytes(data)
            label = int(condition not in CONTROL_NAMES)
            source_hash = sha(bodies[0])
            records.append({"id": case_id, "modality": "image", "split": SPLIT, "label": label, "group": stem,
                            "condition": condition, "family": "public-scene-text", "input_sha256": sha(data),
                            "source_sha256": source_hash, "earlier_copy_sha256": prior, "license": intake.LICENSE,
                            "localization_truth": {"unit": "pixels", "source_units": width * height,
                                                   "target_units": int((first_mask | second_mask).sum()) if label else 0}})
            masks_by_case.append({"id": case_id, "input_sha256": sha(data), "extent": [width, height],
                                  "instance_masks": exact_masks if label else []})
    manifest = {"schema": independent.SCHEMA, "cases": records}
    commitment = independent.audit_manifest(manifest)["manifest_sha256"]
    truth = {"schema": independent_locations.TRUTH_SCHEMA, "split": SPLIT,
             "manifest_sha256": commitment, "cases": masks_by_case}
    write(run / "manifest.json", manifest)
    write(run / "truth.json", truth)
    write(run / "freeze.json", {"schema": SCHEMA, "scope": "staged changes under a matched delivery",
                                "manifest_sha256": commitment, "truth_sha256": independent.digest(truth),
                                "code_sha256": sha(Path(__file__).read_bytes()), "match": EDIT_RULE,
                                "excluded": excluded, "input_manifest_sha256": sha((source / "plan.json").read_bytes())})


def current(earlier, submitted, directory):
    import numpy as np
    from PIL import Image

    from app.evidence.image_reference import inspect

    result = inspect(earlier, submitted, evidence_dir=directory, allow_proportional_resize=True,
                     allow_oriented_crop=True, allow_oriented_resize=True, allow_exact_jpeg_identity=True)
    if result["status"] != "comparable":
        return None, result["status"]
    if not result.get("mask"):
        return [], None
    with Image.open(directory / "difference-mask.png", formats=["PNG"]) as image:
        selection = np.asarray(image) != 0
    return components(selection), None


def components(selection):
    import cv2
    import numpy as np

    _, labels, areas, _ = cv2.connectedComponentsWithStats(selection.astype(np.uint8), connectivity=8)
    chosen = sorted(range(1, len(areas)), key=lambda index: (-areas[index, 4], index))
    return [labels == index for index in chosen[:5] if areas[index, 4] >= MIN_COMPONENT_PIXELS]


def candidate(earlier, submitted):
    import cv2
    import numpy as np
    from PIL import Image

    from app.detectors.image import reference_exact_jpeg

    a, b = earlier.read_bytes(), submitted.read_bytes()
    with Image.open(io.BytesIO(a), formats=["PNG"]) as source, Image.open(io.BytesIO(b), formats=["JPEG"]) as questioned:
        if not reference_exact_jpeg.qualified_carriers(source, questioned, a, b):
            return None, "different_delivery_profile"
        observed = np.asarray(questioned)
        expected = reference_exact_jpeg.encoded_reference(np.asarray(source), questioned)
        predeclared_second = reference_exact_jpeg.encoded_reference(expected, questioned)
    if np.array_equal(observed, expected) or np.array_equal(observed, predeclared_second):
        return [], None
    proof = reference_exact_jpeg.alignment_proof(expected, observed)
    if proof is None or proof["exact_tiles"] * 20 < proof["full_tiles"] * 17:
        return None, "unresolved_exact_delivery"
    delta = np.abs(expected.astype(np.int16) - observed.astype(np.int16)).max(axis=2)
    joined = cv2.morphologyEx((delta >= RESIDUAL_FLOOR).astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # The plotted mask never includes equal decoded pixels, even inside a closed region.
    return components(joined.astype(bool) & (delta > 0)), None


def measure(run: Path, output: Path):
    import cv2

    from app.detectors.image import reference_alignment, reference_compare, reference_exact_jpeg
    from app.evidence import image_reference

    cv2.setNumThreads(1)
    manifest, truth, freeze = (json.loads((run / f"{name}.json").read_bytes()) for name in ("manifest", "truth", "freeze"))
    if (independent.digest(manifest) != freeze["manifest_sha256"] or independent.digest(truth) != freeze["truth_sha256"]
            or sha(Path(__file__).read_bytes()) != freeze["code_sha256"] or freeze["match"] != EDIT_RULE):
        raise ValueError("The evaluation inputs, rules or code changed after the freeze")
    module_shas = {module.__name__: sha(Path(module.__file__).read_bytes())
                   for module in (reference_alignment, reference_compare, reference_exact_jpeg, image_reference)}
    entries = {method: [] for method in ("current", "exact_delivery")}
    for case in manifest["cases"]:
        case_id = case["id"]
        earlier, submitted = (run / "deliveries" / name for name in (f"{case['group']}.png", f"{case_id}.jpg"))
        if sha(earlier.read_bytes()) != case["earlier_copy_sha256"] or sha(submitted.read_bytes()) != case["input_sha256"]:
            raise ValueError("An evaluation delivery changed")
        with TemporaryDirectory(dir=run) as staging:
            readings = (current(earlier, submitted, Path(staging) / "evidence"), candidate(earlier, submitted))
        for (method, rows), (proposals, reason) in zip(entries.items(), readings, strict=True):
            if proposals is None:
                outcome = {"status": "not_applicable", "reason": reason}
                stats = {**outcome, "response": None}
            else:
                refs = [save_mask(run / "masks", f"{case_id}-{method}-{i}.pbm", shape)
                        for i, shape in enumerate(proposals, 1)]
                outcome = {"status": "ok", "instance_masks": refs}
                stats = {"status": "ok", "response": len(proposals)}
            rows.append({"id": case_id, "input_sha256": case["input_sha256"], **stats, "localization": outcome})
    receipts = [{"manifest_sha256": freeze["manifest_sha256"], "split": SPLIT, "system": method,
                 "implementation_sha256": independent.digest({"code": freeze["code_sha256"], "modules": module_shas,
                                                              "method": method}),
                 "method_kind": "model_free", "weight_sha256": [], "score_kind": "raw",
                 "response_units": "count of displayed regions", "predictions": rows}
                for method, rows in entries.items()]
    for method, report in zip(entries, receipts, strict=True):
        write(run / f"{method}.json", report)
    pair = independent_locations.compare_events(manifest, truth, *receipts, mask_directory=run / "masks", **EDIT_RULE)
    write(output, {"schema": SCHEMA, "freeze_sha256": independent.digest(freeze), "geometry": pair,
                   "source_groups": len({case["group"] for case in manifest["cases"]}),
                   "planned_files": len(manifest["cases"]), "excluded_before_scoring": len(freeze["excluded"]),
                   "application_code": module_shas})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prep = subcommands.add_parser("prepare")
    prep.add_argument("source", type=Path)
    prep.add_argument("run", type=Path)
    score = subcommands.add_parser("measure")
    score.add_argument("run", type=Path)
    score.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        args.run.mkdir(mode=0o700)
        prepare(args.source, args.run)
    else:
        measure(args.run, args.output)
