# SPDX-License-Identifier: Apache-2.0
"""Freeze and replay exact source-pixel copy locations on lossless WebP."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.image_exact_turned_copy_transfer import EDIT_PROFILES, useful

SCHEMA = "chipmunk.image-webp-exact-copy-transfer/1"
SHAPES = ((512, 512), (600, 512), (640, 576), (768, 576), (512, 704))
EDITS = ("straight", *EDIT_PROFILES)
ORDINARY = ("ordinary", "tone", "local_blur", "mirrored_copy")
GATE = {"minimum_added_edits": 48, "minimum_groups": 15,
        "maximum_added_controls": 2, "maximum_mislocated_additions": 1}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def write(path, record):
    raw = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def freeze(cameras, background, repetitions, output):
    manifests = {"cameras": cameras / "freeze.json", "background": background / "manifest.json",
                 "repetitions": repetitions / "freeze.json"}
    roots = {key: path.parent for key, path in manifests.items()}
    camera, background_bank, repeat = (read(manifests[key]) for key in roots)
    source_ids = {str(row["group"]): row["sha256"] for row in camera["sources"]}
    groups = {row["tag"]: row["source"] for row in background_bank["groups"] if row["family"] == "lama"}
    selected = {
        "cameras": [(str(r["group"]), r["file"], r["sha256"], source_ids[str(r["group"])])
                    for r in camera["rows"] if r["geometry"] == "medium" and r["role"] == "ordinary"],
        "background": [(r["group"], r["file"], r["sha256"], groups[r["group"]])
                       for r in background_bank["images"] if r["group"] in groups and r["role"] == "ordinary" and r["quality"] == 90],
        "repetitions": [(str(r["group"]), r["file"], r["sha256"], r["sha256"])
                        for r in repeat["rows"] if r["role"] == "repetition" and r["quality"] == 90],
    }
    if [len(rows) for rows in selected.values()] != [10, 12, 9]:
        raise ValueError("Expected 22 camera sources and nine repeated-structure controls")
    all_rows = [dict(bank=bank, id=group, file=name, sha256=pin, source=source)
                for bank, rows in selected.items() for group, name, pin, source in sorted(rows)]
    if len({row["source"] for row in all_rows}) != len(all_rows):
        raise ValueError("Source identities overlap")
    write(output, {"schema": SCHEMA, "sources": all_rows, "shapes": SHAPES, "gate": GATE,
                   "manifests": {key: digest(path.read_bytes()) for key, path in manifests.items()}})


def summarize(plan, rows, format_controls):
    expected = {(row["bank"], row["id"], role) for row in plan["sources"]
                for role in (ORDINARY if row["bank"] == "repetitions" else (*EDITS, *ORDINARY))}
    if len(rows) != len(expected) or {(r["bank"], r["source"], r["role"]) for r in rows} != expected:
        raise ValueError("The case table differs from the frozen roster")
    result = {}
    useful_groups = set()
    for row in rows:
        group = (row["bank"], row["source"])
        counts = result.setdefault(row["role"], Counter(cases=0, prior=0, combined=0, additions=0,
                                                        additional_control=0, mislocated=0))
        counts["cases"] += 1
        geometry = row["truth"]
        prior = useful(row["baseline"], geometry)
        extra = useful(row["exact"], geometry)
        counts["prior"] += prior
        counts["combined"] += prior or extra
        counts["additions"] += extra and not prior
        counts["additional_control"] += not geometry and not row["baseline"] and bool(row["exact"])
        counts["mislocated"] += bool(geometry and row["exact"] and not extra)
        if extra and not prior:
            useful_groups.add(group)
    totals = {"added_edits": sum(r["additions"] for r in result.values()), "useful_groups": len(useful_groups),
              "added_controls": sum(r["additional_control"] for r in result.values()),
              "mislocated_additions": sum(r["mislocated"] for r in result.values())}
    passed = (totals["added_edits"] >= GATE["minimum_added_edits"] and totals["useful_groups"] >= GATE["minimum_groups"]
              and totals["added_controls"] <= GATE["maximum_added_controls"]
              and totals["mislocated_additions"] <= GATE["maximum_mislocated_additions"]
              and format_controls == dict.fromkeys(("lossy", "alpha", "animated", "extended", "too_large"), "unsupported_profile"))
    return {"by_role": result, "totals": totals, "passed": passed}


def run(plan, cameras, background, repetitions, output):
    import io

    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    from app.detectors.image import copy_affine, copy_similarity
    from app.evidence import image_copy_affine_binding, image_packets_v7
    from app.evidence.image_copy_similarity_binding import V8, supported_size
    from app.evidence.image_exact_small_copy_source import review, verify, webp_rgb
    from scripts.image_exact_turned_copy_transfer import cases

    def encoded(pixels, **options):
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="WEBP", **options)
        return stream.getvalue()

    def trials(pixels, index, repeat):
        for role, view, truth in cases(pixels, index):
            if role in ORDINARY or not repeat and role in EDITS:
                yield role, view, truth
        if not repeat:
            height, width = pixels.shape[:2]
            a, b = (82, 92, 32, 34), (width - 112, height - 126, 32, 34)
            view = pixels.copy()
            view[a[1]:a[1]+a[3], a[0]:a[0]+a[2]] = pixels[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]
            yield "straight", view, (a, b)

    cv2.setNumThreads(1)
    roots = {"cameras": cameras, "background": background, "repetitions": repetitions}
    lock = read(plan)
    output.mkdir(mode=0o700)
    rows, saved = [], set()
    for index, row in enumerate(lock["sources"]):
        data = (roots[row["bank"]] / row["file"]).read_bytes()
        if digest(data) != row["sha256"]:
            raise ValueError("An input differs from the frozen JPEG source")
        with Image.open(io.BytesIO(data), formats=["JPEG"]) as image:
            rgb = np.array(ImageOps.fit(image.convert("RGB"), SHAPES[index % len(SHAPES)], Image.Resampling.LANCZOS))
        height, width = rgb.shape[:2]
        for role, pixels, truth in trials(rgb, index, row["bank"] == "repetitions"):
            delivered = encoded(pixels, lossless=True, method=1)
            decoded = webp_rgb(delivered)
            if decoded is None or not np.array_equal(decoded, pixels):
                raise ValueError("Lossless WebP changed the staged pixels")
            status, pairs, _ = ("unsupported_geometry", [], None)
            baseline = []
            if supported_size(width, height, version=V8):
                status, pairs, _ = image_packets_v7.selected_translation_pairs(decoded)
                baseline.extend(pairs)
                baseline.extend(copy_similarity.analyse(decoded)[1])
            if image_copy_affine_binding.supported(width, height):
                for proposal in copy_affine.candidates(decoded):
                    a, b, w, h = image_packets_v7._crop_box(proposal.box_a, width, height)
                    masks = copy_affine.dense_pair(copy_affine.luma(decoded), np.array(proposal.affine.matrix), (a, b, a+w, b+h))
                    if masks is not None:
                        baseline.append(tuple(cv2.boundingRect(mask.astype(np.uint8)) for mask in masks))
            exact, extra, turns = {"status": "not_needed", "measurement": None}, [], 0
            if status in {"unsupported_geometry", "no_pair"}:
                exact, extra, turns = review(delivered, webp=True, include_rotation=True)
            if extra:
                from app.evidence.image_copy_similarity_binding import PNG_COPY_TRANSFORMS

                regions = {name: dict(zip("xywh", box, strict=True)) for name, box in zip(("area_a", "area_b"), extra[0], strict=True)}
                verify(delivered, {"regions": regions, "transform": PNG_COPY_TRANSFORMS[turns]}, exact["measurement"], webp=True)
            if role not in saved and ((truth and useful(extra, truth) and not useful(baseline, truth))
                                      or role == "ordinary" and extra == baseline == []):
                (output / f"proof-{role}.webp").write_bytes(delivered)
                saved.add(role)
            rows.append({"bank": row["bank"], "source": row["id"], "role": role, "sha256": digest(delivered),
                         "rgb_sha256": digest(decoded.tobytes()), "size": [width, height], "truth": truth,
                         "baseline": baseline, "exact": extra, "review": exact, "turns": turns})
        print(row["bank"], row["id"], len(rows), flush=True)
    rgba = np.dstack((rgb, np.full(rgb.shape[:2], 80, dtype=np.uint8)))
    other = np.ascontiguousarray(rgb[::-1])
    png = Image.fromarray(rgb)
    animated = io.BytesIO()
    png.save(animated, format="WEBP", lossless=True, save_all=True, append_images=[Image.fromarray(other)], duration=[100, 100])
    extended = io.BytesIO()
    png.save(extended, format="WEBP", lossless=True, xmp=b"<x:xmpmeta/>")
    large = np.array(png.resize((769, 512)))
    variants = {"lossy": encoded(rgb), "alpha": encoded(rgba, lossless=True), "animated": animated.getvalue(),
                "extended": extended.getvalue(), "too_large": encoded(large, lossless=True)}
    controls = {name: review(data, webp=True, include_rotation=True)[0]["status"] for name, data in variants.items()}
    totals = summarize(lock, rows, controls)
    write(output / "receipt.json.gz", {"schema": SCHEMA, "plan": lock, "rows": rows, "format_controls": controls, "totals": totals})
    print(json.dumps(totals, sort_keys=True))


def review(receipt):
    record = read(receipt)
    plan = record["plan"]
    if (plan["schema"] != SCHEMA or plan["gate"] != GATE or plan["shapes"] != [list(size) for size in SHAPES]
            or Counter(row["bank"] for row in plan["sources"]) != {"cameras": 10, "background": 12, "repetitions": 9}
            or len({row["source"] for row in plan["sources"]}) != 31):
        raise ValueError("The evaluation does not use the frozen sources, geometry and release criteria")
    summary = summarize(plan, record["rows"], record["format_controls"])
    if record["schema"] != SCHEMA or record["totals"] != summary:
        raise ValueError("The numeric evidence differs from its saved totals")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command, fields in (("freeze", ("cameras", "background", "repetitions", "output")),
                            ("run", ("plan", "cameras", "background", "repetitions", "output")), ("review", ("receipt",))):
        child = sub.add_parser(command)
        for field in fields:
            child.add_argument(field, type=Path)
    args = vars(parser.parse_args())
    {"freeze": freeze, "run": run, "review": review}[args.pop("command")](**args)
