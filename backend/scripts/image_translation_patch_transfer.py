"""Compare geometric and dense translated-copy evidence on source-disjoint images."""

from __future__ import annotations

import argparse
import hashlib
import io
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.detectors.image import copy_similarity
from app.detectors.image.copy_move import iou
from app.evidence.image_packets_v7 import translation_pairs
from scripts import image_copy_transfer_v298 as prior
from scripts import image_fresh_source_cache_v39 as cache

SCHEMA = "chipmunk.image-translation-transfer/1"
SALT = "native-translation-copy-holdout-2026-11-11"
GROUPS = 16
FORMATS = ("png", "jpeg90")
ROLES = ("original", "tone", "blur", "copy_96", "copy_144")
HARD_CONTROL_SHA256 = "d608c4c158b36b6afc4f586a1752b4087b7d7beb606b44885b60d7e578561c5d"
COPY_TRANSFER_SHA256 = "df4f7902a3cee6d906fc6ffaa72d0e3e585e6eb7e98be4f9daeff582745e7716"
EXPOSED_SHA256 = frozenset({"4c397fdce4bb242910cb6f14a03a008f74a02070fe66f826c5753829df964577",
                            "ad439b2d5b6d4169d1eaf8e74f2f8d3104924ebd45d722b850d1236287b9f733"})
read, sha, write = prior.grouping.read, prior.grouping.sha, prior.grouping.write


def implementation():
    return {Path(path).name: sha(Path(path)) for path in (__file__, copy_similarity.__file__,
            translation_pairs.__code__.co_filename, prior.__file__, cache.__file__)}


def plan(hard, hard_roster, exposed):
    pilot = prior.pilot
    receipt_name = "image-fresh-source-cache-v39-results.json"
    receipt = prior.read_exact(pilot.PANELS / receipt_name, prior.RECEIPTS[receipt_name])
    root = pilot.DATA / "image-fresh-source-cache-v39"
    graph = prior.read_exact(root / "private-groups.json", receipt["private_groups_sha256"])
    catalog = prior.read_exact(root / "plan.json", receipt["private_plan_sha256"])
    by_id = {row["source_id"]: row for row in catalog["items"]}
    rows = {row["key"]: row for row in graph["sources"]}
    hard_cases = prior.read_exact(hard, HARD_CONTROL_SHA256)
    if sha(hard_roster) != hard_cases["shortlist_sha256"]:
        raise ValueError("The hard-control attribution shortlist differs from its frozen hash")
    control_rows = {row["slot"]: row["source"] for row in read(hard_roster)["rows"]}
    controls = [{"source": control_rows[case["slot"]], "category": case["category"]} for case in hard_cases["real_cases"]]
    if len(controls) != 8 or any(rows[row["source"]["key"]] != row["source"] for row in controls):
        raise ValueError("The eight hard-control sources must match the verified photo cache")
    excluded = {entry["key"] for entry in control_rows.values()}
    for folder, name in (("image-flux-photo-preparation-v276", "image-flux-photo-preparation-freeze-v276.json"),
                         ("image-flux-photo-expansion-v285", "image-flux-photo-expansion-v285-freeze.json")):
        reference = prior.read_exact(pilot.PANELS / name, prior.RECEIPTS[name])
        used = prior.read_exact(pilot.DATA / folder / "private-inputs.json", reference["private_inputs_sha256"])
        excluded.update(key for row in used["rows"] for key in row["group"])
    earlier = prior.read_exact(pilot.PANELS / "image-copy-transfer-v298-complete.json", COPY_TRANSFER_SHA256)
    bound = prior.read_exact(prior.TRIAL / "private-freeze.json", earlier["freeze_sha256"])
    if len(exposed) != len(EXPOSED_SHA256) or {sha(path) for path in exposed} != EXPOSED_SHA256:
        raise ValueError("Both frozen prior reflection rosters are required")
    for table in [bound, *(read(path) for path in exposed)]:
        excluded.update(key for row in table["rows"] for key in row["component"])
    available = []
    for group in graph["candidate_only_unlinked_components"]:
        if excluded.intersection(group):
            continue
        qualified = [rows[key] for key in group if rows[key]["license_url"] == pilot.CC_BY
                     and max(rows[key]["size"]) >= 576 and min(rows[key]["size"]) >= 480]
        if qualified:
            available.append(min(qualified, key=source_order))
    selected = sorted(available, key=source_order)[:GROUPS]
    if len(selected) != GROUPS or len({row["author"] for row in selected}) != GROUPS:
        raise ValueError("The frozen panel requires 16 distinct attributed author groups")
    for row in [*selected, *(control["source"] for control in controls)]:
        source, item = prior.source_path(row), by_id[row["source_id"]]
        if row["license_url"] != pilot.CC_BY or cache.pixel_record(item, source, row["source_jpeg_sha256"]) != row:
            raise ValueError("A selected source or its license no longer matches the cached record")
    return {"schema": SCHEMA, "implementation": implementation(), "available_groups": len(available),
            "sources": selected, "hard_controls": controls, "formats": list(FORMATS), "roles": list(ROLES),
            "runtime": {"opencv": cv2.__version__, "numpy": np.__version__},
            "roster_hashes": {"cache_graph": sha(root / "private-groups.json"), "hard_control_plan": sha(hard),
                              "other_panels": [sha(path) for path in exposed]},
            "useful_min_iou_each": 0.5,
            "comparison": "compare independent arms plus dense match with geometric fallback on no dense match"}


def source_order(row):
    return hashlib.sha256(f"{SALT}/{row['source_jpeg_sha256']}".encode()).hexdigest()


def original(row):
    path = prior.source_path(row)
    if sha(path) != row["source_jpeg_sha256"]:
        raise ValueError("A frozen source has different bytes")
    with Image.open(path) as image:
        source = image.convert("RGB")
    if hashlib.sha256(source.tobytes()).hexdigest() != row["rgb_sha256"]:
        raise ValueError("A frozen source decoded differently")
    width, height = source.size
    left, top = max(0, (width - 768) // 2), max(0, (height - 768) // 2)
    return source.crop((left, top, left + min(width, 768), top + min(height, 768)))


def views(source, slot):
    rgb = np.asarray(source)
    width, height = source.size
    yield "original", source, None
    yield "tone", ImageEnhance.Contrast(source).enhance(1.1), None
    blur = source.copy()
    box = (0, 39, 144, 183)
    blur.paste(source.filter(ImageFilter.GaussianBlur(1.25)).crop(box), box[:2])
    yield "blur", blur, None
    for side in (96, 144):
        left = 0 if slot % 2 == 0 else 45 + slot % 3
        top = 39 + slot % 4
        x, y = width - side - 42 - slot % 5, height - side - 38 - slot % 3
        pixels = rgb[y:y+side, x:x+side]
        edited = rgb.copy()
        edited[top:top+side, left:left+side] = pixels
        if not np.any(rgb != edited) or not np.array_equal(edited[y:y+side, x:x+side], pixels):
            raise ValueError("The donor must be unchanged and the target visibly altered")
        yield f"copy_{side}", Image.fromarray(edited), ((left, top, side, side), (x, y, side, side))


def matched(pairs, truth):
    if truth is None:
        return False
    first, second = truth
    return any(max(min(iou(a, first), iou(b, second)), min(iou(a, second), iou(b, first))) >= 0.5 for a, b in pairs)


def assess(image, fmt, truth):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG" if fmt == "png" else "JPEG", quality=90, subsampling=2)
    encoded = buffer.getvalue()
    with Image.open(io.BytesIO(encoded)) as source:
        rgb = np.asarray(source.convert("RGB"))
    before = time.monotonic()
    baseline = translation_pairs(rgb)
    between = time.monotonic()
    dense = copy_similarity.analyse_translation(rgb)
    end = time.monotonic()
    selected = dense if dense[0] != "no_pair" else baseline
    observations = {}
    for name, (status, pairs), elapsed in (
        ("geometric", baseline, between - before), ("native_patch", dense, end - between),
        ("combined", selected, end - before if dense[0] == "no_pair" else end - between),
    ):
        observations[name] = {"status": status, "pairs": pairs, "useful": matched(pairs, truth), "elapsed_s": elapsed}
    return hashlib.sha256(encoded).hexdigest(), observations


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["format"], row["role"]].append(row)
    results = {}
    for (fmt, role), cases in groups.items():
        values = results.setdefault(fmt, {}).setdefault(role, {})
        for name in ("geometric", "native_patch", "combined"):
            responses = [row["responses"][name] for row in cases]
            values[name] = {"views": len(responses), "flagged": sum(bool(r["pairs"]) for r in responses),
                            "useful": sum(r["useful"] for r in responses),
                            "unassessed": sum(r["status"] == copy_similarity.MISSING for r in responses),
                            "median_ms": round(statistics.median(r["elapsed_s"] for r in responses) * 1000, 2)}
        transitions = [(row["responses"]["geometric"]["useful"], row["responses"]["combined"]["useful"]) for row in cases]
        values["combined_vs_geometric"] = {"gains": transitions.count((False, True)),
                                             "losses": transitions.count((True, False))}
    return results


def run(destination):
    frozen = read(destination / "freeze.json")
    if frozen["schema"] != SCHEMA or frozen["implementation"] != implementation():
        raise ValueError("Frozen code commitments differ from this run")
    cv2.setNumThreads(1)
    rows = []
    inputs = [(slot, list(views(original(row), slot))) for slot, row in enumerate(frozen["sources"])]
    inputs += [(GROUPS+slot, [("hard_control", original(row["source"]), None)])
               for slot, row in enumerate(frozen["hard_controls"])]
    for slot, cases in inputs:
        for role, image, truth in cases:
            for fmt in FORMATS:
                digest, responses = assess(image, fmt, truth)
                rows.append({"slot": slot, "role": role, "format": fmt, "input_sha256": digest,
                             "truth": truth, "responses": responses})
    journal = destination / "rows.json"
    write(journal, rows)
    write(destination / "result.json", {"schema": SCHEMA, "main_source_groups": len(frozen["sources"]),
          "hard_control_groups": len(frozen["hard_controls"]), "views": len(rows), "freeze_sha256": sha(destination / "freeze.json"),
          "journal_sha256": sha(journal), "results": aggregate(rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    freeze = modes.add_parser("freeze")
    freeze.add_argument("destination", type=Path)
    freeze.add_argument("hard_freeze", type=Path)
    freeze.add_argument("hard_roster", type=Path)
    freeze.add_argument("--exposed", type=Path, action="append", default=[])
    execute = modes.add_parser("run")
    execute.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        frozen = plan(args.hard_freeze, args.hard_roster, args.exposed)
        args.destination.mkdir(mode=0o700)
        write(args.destination / "freeze.json", frozen)
    else:
        run(args.destination)


if __name__ == "__main__":
    main()
