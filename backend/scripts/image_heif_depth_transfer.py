# SPDX-License-Identifier: Apache-2.0
"""Freeze and check linked-thumbnail evidence on SDR 10-bit HEIC deliveries."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
from pathlib import Path

SCHEMA = "chipmunk.heif-thumbnail-depth-transfer/1"
PREVIOUS = "docs/evaluation-panels/image-heif-thumbnail-2026-11-16.json.gz"
DEPTHS = ((8, 8), (10, 8), (10, 10))
RECIPES = ((320, 82, 74), (192, 66, 58))
CONTROL_FILES = (
    "basic-8bit-srgb.heic", "camera-style-10bit-422.hif", "display-p3-10bit.heic",
    "hdr-pq-10bit-bt2020.heic", "thumbnail-embedded.heic",
)
TREATMENTS = {
    "sharp": ("sharp", "ordinary"), "feather": ("feather", "ordinary"),
    "ordinary": ("ordinary", "ordinary"), "refreshed_edit": ("sharp", "sharp"),
    "tone": ("local_tone", "local_tone"), "blur": ("local_blur", "local_blur"),
    "copy": ("local_copy", "local_copy"),
}
EDITS = {"sharp", "feather"}
MIN_PAIRED_RECALL_PCT = 90
MAX_EXTRA_CONTROLS = 2


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, result):
    with path.open("xb") as stream:
        output = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        stream.write(gzip.compress(output, mtime=0) if path.suffix == ".gz" else output)


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def freeze(source, previous, output):
    manifest = load(source / "manifest.json")
    prior = load(previous)
    controls = {row["file"]: row["sha256"] for row in prior["external"]}
    groups = [{key: row[key] for key in ("tag", "source", "family")}
              for row in manifest["groups"] if not row["tag"].endswith("-00")]
    sources = {row["source"] for row in groups}
    if (len(sources) != 11 or len(groups) != 22 or sources & {row["source"] for row in prior["rows"]}
            or {row["family"] for row in groups} != {"lama", "migan"}):
        raise ValueError("Expected 11 distinct held photos with both edit methods")
    write(output, {"schema": SCHEMA, "source_manifest_sha256": digest((source / "manifest.json").read_bytes()),
                   "previous_panel_sha256": digest(previous.read_bytes()), "groups": groups,
                   "depths": DEPTHS, "recipes": RECIPES, "treatments": TREATMENTS,
                   "format_controls": {name: controls[name] for name in CONTROL_FILES},
                   "minimum_paired_recall_pct": MIN_PAIRED_RECALL_PCT, "maximum_extra_controls": MAX_EXTRA_CONTROLS})


def located(row):
    truth, selected, overlap = (row[field] for field in ("truth", "selected", "intersection"))
    if not 0 <= overlap <= min(truth, selected):
        raise ValueError("Invalid localization intersection")
    return bool(truth and overlap * 2 >= truth and overlap * 10 >= (truth + selected - overlap) * 3)


def count(plan, rows):
    expected = {(group["tag"], edge, name, full, thumb) for group in plan["groups"]
                for edge, _, _ in RECIPES for name in TREATMENTS for full, thumb in DEPTHS}
    cases = {(row["group"], row["edge"], row["treatment"], *row["depths"]): row for row in rows}
    if len(cases) != len(rows) or cases.keys() != expected:
        raise ValueError("The delivered and frozen case rosters differ")

    outcomes = {}
    for edge, _, _ in RECIPES:
        for full, thumb in DEPTHS[1:]:
            prior_useful = kept = found = added_false = prior_false = false = unassessed = misplaced = edits = ordinary = 0
            for tag, e, treatment, a, b in cases:
                if (e, a, b) != (edge, full, thumb):
                    continue
                old = cases[tag, e, treatment, 8, 8]
                row = cases[tag, e, treatment, a, b]
                unassessed += row["status"] != "comparable"
                if treatment in EDITS:
                    edits += 1
                    useful = located(row)
                    old_useful = located(old)
                    prior_useful += old_useful
                    kept += old_useful and useful
                    found += useful
                    misplaced += bool(row["regions"]) and not useful
                else:
                    ordinary += 1
                    found_any = bool(row["regions"])
                    prior_any = bool(old["regions"])
                    prior_false += prior_any
                    false += found_any
                    added_false += found_any and not prior_any
            outcomes[f"{edge}-{full}-{thumb}"] = {
                "edits": edits, "ordinary": ordinary, "baseline_located": prior_useful, "located": found, "kept": kept,
                "misplaced": misplaced, "baseline_ordinary_flags": prior_false, "ordinary_flags": false,
                "additional_ordinary_flags": added_false, "unassessed": unassessed,
                "qualified": prior_useful > 0 and kept * 100 >= MIN_PAIRED_RECALL_PCT * prior_useful
                and added_false <= MAX_EXTRA_CONTROLS,
            }
    return outcomes


def run(plan_file, source, controls, library, output):
    import ctypes as c

    import cv2
    import numpy as np
    from PIL import Image
    from pillow_heif import libheif_info, register_heif_opener

    from app.provenance import heif_thumbnail
    from app.provenance import image_thumbnail_localization as localization
    from scripts.image_heif_thumbnail_panel import Encoder, check, pixels

    class DepthEncoder(Encoder):
        def pixels(self, data, cleanup):
            if data.dtype == np.uint8:
                return super().pixels(data, cleanup)
            height, width, _ = data.shape
            image = c.c_void_p()
            check(self.image_create(width, height, 1, 14, c.byref(image)))  # RGB in little-endian 16-bit words.
            cleanup.callback(self.image_release, image)
            check(self.image_add_plane(image, 10, width, height, 10))
            stride = c.c_int()
            plane = self.image_get_plane(image, 10, c.byref(stride))
            row_bytes = 6 * width
            if not plane or stride.value < row_bytes:
                raise ValueError("Cannot allocate a 10-bit RGB fixture")
            contiguous = np.ascontiguousarray(data, dtype="<u2")
            for y in range(height):
                c.memmove(plane + y * stride.value, contiguous.ctypes.data + y * row_bytes, row_bytes)
            return image

    cv2.setNumThreads(1)
    register_heif_opener()
    plan = load(plan_file)
    manifest_path = source / "manifest.json"
    if plan["schema"] != SCHEMA or digest(manifest_path.read_bytes()) != plan["source_manifest_sha256"]:
        raise ValueError("The held image manifest differs from the frozen plan")
    output.mkdir(mode=0o700)
    fixture = output / "delivery.heic"
    encoder = DepthEncoder(library)
    manifest = load(manifest_path)
    deliveries = {(row["group"], row["role"]): row for row in manifest["images"] if row["quality"] == 90}
    rows, formats, proofs = [], [], set()

    def measure(body, truth, depths, *, enabled=True):
        with heif_thumbnail.read_pair(body, allow_10bit_sdr=enabled) as (status, pair):
            if pair is None:
                return {"status": status, "regions": [], "truth": int(truth.sum()), "intersection": 0, "selected": 0}
            main, thumb = pair
            if depths is not None and (main.info["bit_depth"], thumb.info["bit_depth"]) != depths:
                raise ValueError("Encoder produced different bit depths")
            with Image.open(io.BytesIO(body)) as displayed:
                if displayed.mode != "RGB" or displayed.tobytes() != main.tobytes():
                    raise ValueError("Decoded primary pixels differ from the upload viewer")
            evidence, _ = localization.measure_heif(main, thumb, minimum_component_fraction=localization.CONFIRMATION_COMPONENT_FRACTION)
            regions = [row["submitted"] for row in evidence.get("regions", [])[:3]]
            selected = np.zeros_like(truth)
            for box in regions:
                x, y, w, h = (box[key] for key in ("x", "y", "w", "h"))
                selected[y:y + h, x:x + w] = True
            return {"status": evidence["status"], "regions": regions, "truth": int(truth.sum()),
                    "selected": int(selected.sum()), "intersection": int((selected & truth).sum())}

    for group in plan["groups"]:
        tag = group["tag"]
        entries = {name: deliveries[tag, name] for names in TREATMENTS.values() for name in names}
        for row in entries.values():
            if any(digest((source / row[field]).read_bytes()) != row[pin]
                   for field, pin in (("file", "sha256"), ("mask", "mask_sha256"))):
                raise ValueError("A source image or truth mask differs from the manifest")
        rgb = {name: pixels(source / row["file"]) for name, row in entries.items()}
        ten = {name: ((values.astype(np.uint32) * 1023 + 127) // 255).astype(np.uint16) for name, values in rgb.items()}
        pictures = {8: rgb, 10: ten}
        for treatment, (current, thumbnail) in TREATMENTS.items():
            truth = (pixels(source / entries[current]["mask"], mask=True) > 0 if treatment in EDITS
                     else np.zeros(rgb[current].shape[:2], dtype=bool))
            for recipe in RECIPES:
                edge = recipe[0]
                for full, thumb in DEPTHS:
                    encoder.write(fixture, pictures[full][current], pictures[thumb][thumbnail], recipe)
                    body = fixture.read_bytes()
                    reading = measure(body, truth, (full, thumb))
                    earlier = measure(body, truth, (full, thumb), enabled=False)
                    if (full == 8 and earlier != reading) or (full == 10 and earlier["status"] != "unsupported_carrier"):
                        raise ValueError("Earlier reader returned unexpected evidence")
                    rows.append({"group": tag, "source": group["source"], "edge": edge, "treatment": treatment,
                                 "depths": [full, thumb], "sha256": digest(body), "prior_status": earlier["status"], **reading})
                    proof = f"{full}-{thumb}-{treatment}"
                    should_save = ((treatment == "ordinary" and reading["status"] == "comparable" and not reading["regions"])
                                   or (treatment == "sharp" and located(reading)))
                    if full == 10 and edge == 320 and proof not in proofs and should_save:
                        proofs.add(proof)
                        shutil.copyfile(fixture, output / f"proof-{proof}.heic")
        print(tag, len(rows), "encoded files verified", flush=True)

    for name, original_hash in plan["format_controls"].items():
        body = (controls / name).read_bytes()
        if digest(body) != original_hash:
            raise ValueError("A format control differs from its public hash")
        result = measure(body, np.zeros((1, 1), dtype=bool), None)
        if result["regions"]:
            raise ValueError("A published format control produced a false region")
        formats.append({"file": name, "sha256": original_hash, "status": result["status"]})
    fixture.unlink()
    totals = count(plan, rows)
    write(output / "receipt.json.gz", {"schema": SCHEMA, "plan_sha256": digest(plan_file.read_bytes()),
          "codec": libheif_info(), "rows": rows, "format_controls": formats, "totals": totals})
    print(json.dumps(totals, sort_keys=True))


def review(plan_file, receipt):
    plan = load(plan_file)
    evidence = load(receipt)
    if evidence["schema"] != SCHEMA or evidence["plan_sha256"] != digest(plan_file.read_bytes()):
        raise ValueError("Review needs the manuscript's frozen measurement plan")
    summary = count(plan, evidence["rows"])
    if summary != evidence["totals"] or {row["file"]: row["sha256"] for row in evidence["format_controls"]} != plan["format_controls"]:
        raise ValueError("The held result differs from its measurements")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(required=True, dest="mode")
    for mode, names in (("freeze", ("source", "previous", "output")),
                        ("run", ("plan_file", "source", "controls", "library", "output")),
                        ("review", ("plan_file", "receipt"))):
        args = modes.add_parser(mode)
        for name in names:
            args.add_argument(name, type=Path)
    kwargs = vars(parser.parse_args())
    {"freeze": freeze, "run": run, "review": review}[kwargs.pop("mode")](**kwargs)
