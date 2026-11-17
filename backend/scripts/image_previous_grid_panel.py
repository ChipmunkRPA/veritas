# SPDX-License-Identifier: Apache-2.0
"""Build and compare cropped JPEGs whose earlier block grid can survive local editing."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.image-previous-grid/1"
QUALITIES = (75, 90)
POSITIVES = {"sharp", "feather"}
ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "docs/evaluation-panels"
TARGET = 12


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def save(path, payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with path.open("xb") as handle:
        handle.write(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw + b"\n")


def new_sources(metadata, retained, output):
    import yaml

    from scripts import image_bg_real_insertion_transfer as intake
    from scripts.image_cross_quality_sources import FAMILIES, known_sources

    roster = intake.roster
    keys, _, prior = known_sources(retained, PUBLIC / "image-cross-quality-sources-2026-11-16.json")
    extra = PUBLIC / "image-removal-peer-sources-2026-11-16.json"
    for record in read(extra)["accepted"]:
        keys.update(roster.source_keys(record["source"]))
    hashes = {name: roster.sha(metadata / name) for name in roster.METADATA}
    if any(hashes[name] != entry[2] for name, entry in roster.METADATA.items()):
        raise ValueError("The publisher metadata differs from the pinned edition")
    raw = roster.table(metadata / "manifest_v2.csv")
    images = roster.indexed(raw, "sample_id")
    credits = roster.indexed((r for r in roster.table(metadata / "attribution.csv") if r["role"] == "recipient"), "bgreal_id")
    sources = roster.indexed(roster.table(metadata / "recipients.csv"), "source_image_id")
    splits = {name: set(row) for name, row in yaml.safe_load((metadata / "splits_v2.yaml").read_bytes()).items()}
    families = defaultdict(list)
    for edit in raw:
        role = FAMILIES.get((edit["generator"], edit["edit_kind"]))
        if (not role or edit["family"] != "F5" or not roster.base_png(edit) or min(roster.dimensions(edit)) < 336
                or not .002 <= float(edit["edit_area"]) <= .10):
            continue
        name = edit["source_image_id"]
        try:
            original = roster.one(images, f"{name}_A0", "No verified original")
            case = roster.pair(edit, original, roster.one(credits, name, "No clear photo credits"),
                               roster.one(sources, name, "No fixed source record"), splits)
        except ValueError:
            continue
        families[role].append(case | {"generator": role})
    accepted = []
    for role in sorted(families):
        ordered = sorted(families[role], key=lambda case: (case["publisher_split"] not in roster.TEST_SPLITS,
                                                           digest(f"{SCHEMA}/{case['recipient']}".encode())))
        choices, _ = roster.select(ordered, keys)
        selected = choices[:2 * TARGET]
        accepted.extend(selected)
        for row in selected:
            keys.update(roster.source_keys(row))
    save(output, {"schema": SCHEMA, "publisher_revision": roster.REVISION, "metadata": hashes,
                  "reserved": accepted, "exposure_sha256": prior | {"later_removal": roster.sha(extra)}})
    print(dict(Counter(row["generator"] for row in accepted)), flush=True)


def qualified_download(case, deadline):
    import requests

    from scripts import image_bg_real_insertion_transfer as intake

    if case["photo_attribution"]["License"] != "https://creativecommons.org/licenses/by/2.0/":
        raise ValueError("Unexpected source photograph terms")
    with requests.Session() as session:
        session.trust_env = False
        data = {role: intake.transport.download(session, case[role]["path"], case[role]["sha256"], deadline)[0]
                for role in ("original", "edited", "semantic_mask")}
    decoded = {role: intake.inputs.png_pixels(body, case["size"]) for role, body in data.items()}
    exact, _ = intake.inputs.edit_truth(decoded["original"], decoded["edited"], decoded["semantic_mask"],
                                       case["semantic_edit_area"])
    if not .0015 <= exact.mean() <= .10:
        raise ValueError("Edit area falls outside the source plan")
    marks = {role: intake.inputs.fingerprint(data[role], decoded[role]) for role in ("original", "edited")}
    return data, marks


def acquire(plan, retained, output):
    import time
    from concurrent.futures import ThreadPoolExecutor

    import requests

    from scripts import image_bg_real_insertion_transfer as intake
    from scripts.image_cross_quality_sources import known_sources

    roster, images = intake.roster, intake.inputs
    frozen = read(plan)
    keys, fingerprints, _ = known_sources(retained, PUBLIC / "image-cross-quality-sources-2026-11-16.json")
    for record in read(PUBLIC / "image-removal-peer-sources-2026-11-16.json")["accepted"]:
        keys.update(roster.source_keys(record["source"]))
        for f in record["fingerprints"].values():
            fingerprints[0].append(f)
            fingerprints[1].add(f["encoded_sha256"])
            fingerprints[2].add(f["rgb_sha256"])
    output.mkdir(mode=0o700)
    deadline = time.monotonic() + 360

    def hydrate(case):
        try:
            data, marks = qualified_download(case, deadline)
            if images.prior_matches(case, marks, keys, fingerprints):
                return case, None, None, "prior source"
            return case, data, marks, None
        except (ValueError, OSError, RuntimeError, requests.RequestException) as error:
            return case, None, None, type(error).__name__

    accepted, counts, excluded = [], Counter(), Counter()
    # Retain the order of the frozen manifest regardless of retrieval timing.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for case, data, marks, reason in pool.map(hydrate, frozen["reserved"]):
            family = case["generator"]
            if counts[family] >= TARGET:
                continue
            if reason:
                excluded[reason] += 1
                continue
            if any(images.near(mark, other) for mark in marks.values() for r in accepted for other in r["fingerprints"].values()):
                excluded["within-panel duplicate"] += 1
                continue
            name = f"{len(accepted):02d}"
            root = output / name
            root.mkdir()
            for role, body in data.items():
                (root / f"{role}.png").write_bytes(body)
            accepted.append({"folder": name, "source": case, "fingerprints": marks})
            counts[family] += 1
    save(output / "qualification.json", {"schema": SCHEMA, "source_plan_sha256": digest(plan.read_bytes()),
                                         "accepted": accepted, "count": dict(counts), "excluded": dict(excluded)})
    print(dict(counts), dict(excluded), flush=True)


def restore(roster, output):
    import time
    from concurrent.futures import ThreadPoolExecutor

    bundle = read(roster)
    accepted = bundle["accepted"]
    if bundle["schema"] != SCHEMA or [row["folder"] for row in accepted] != [f"{i:02d}" for i in range(len(accepted))]:
        raise ValueError("Unsupported source roster")
    output.mkdir(mode=0o700)

    def restore_case(record):
        media, marks = qualified_download(record["source"], time.monotonic() + 60)
        if marks != record["fingerprints"]:
            raise ValueError("Downloaded media differ from the recorded source signatures")
        folder = output / record["folder"]
        folder.mkdir()
        for role, content in media.items():
            (folder / f"{role}.png").write_bytes(content)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(restore_case, accepted):
            pass
    save(output / "qualification.json", bundle)


def encode(image, quality):
    from io import BytesIO

    from PIL import Image

    with BytesIO() as out:
        Image.fromarray(image).save(out, "JPEG", quality=quality, subsampling=2)
        return out.getvalue()


def decode(body):
    from io import BytesIO

    import numpy as np
    from PIL import Image

    with Image.open(BytesIO(body)) as image:
        return np.array(image.convert("RGB"))


def authored(original, generated, seed):
    import numpy as np
    from PIL import Image

    from scripts.image_region_rank_inputs import views

    q1 = (50, 55, 60, 65)[int(seed[:2], 16) % 4]
    dx, dy = (int(seed[index:index + 2], 16) % 7 + 1 for index in (2, 4))
    height, width = original.shape[:2]
    # Padding lies outside the delivered crop. Keeping the native dimensions
    # also preserves the existing regional check's JPEG admission.
    older = np.pad(original, ((dy, 0), (dx, 0), (0, 0)), mode="reflect")
    parent = encode(older, q1)
    prior = decode(parent)[dy:dy + height, dx:dx + width]
    mask = np.any(original != generated, axis=2)
    fabricated = np.where(mask[..., None], generated, prior)
    captures = views(prior, fabricated, mask)
    image = Image.fromarray(prior)
    resized = image.resize(tuple(int(axis * .82) for axis in image.size), Image.Resampling.BICUBIC)
    captures["global_resize"] = np.asarray(resized.resize(image.size, Image.Resampling.BICUBIC))
    captures["direct_original"] = original
    captures["direct_edit"] = generated
    return q1, [8 - dx, 8 - dy], parent, prior, captures


def measure_box(box, truth):
    left, top, w, h = box
    return {"reported": w * h, "intersection": int(truth[top:top + h, left:left + w].sum()), "truth": int(truth.sum())}


def useful(a):
    return bool(a and a["truth"] and 10 * a["intersection"] >= 3 * (a["reported"] + a["truth"] - a["intersection"])
                and 2 * a["intersection"] >= a["truth"])


def sources(path, extra=()):
    from scripts.image_region_rank_inputs import bg_sources

    if not extra:
        for item in read(path / "qualification.json")["accepted"]:
            case = item["source"]
            yield (path / item["folder"], case["original"]["sha256"], case["generator"], case["edited"]["sha256"])
        return
    first, second, metadata = extra
    import csv

    with metadata.open(newline="") as stream:
        index = {row["image_path"]: row for row in csv.DictReader(stream)}
    for name, media in (("image-bg-real-acquisition-v225", first), ("image-bg-real-backup-intake-v262", second)):
        qualification = path / name / "qualified-private.json"
        # Reuse archived receipts and verify the packed generator and source bytes.
        for record, checked in zip(read(qualification)["qualified"], bg_sources(qualification, media, index), strict=True):
            files = record["measurement"]["files"]
            yield (media / Path(files["original"]["file"]).parent, checked[1], checked[2], files["edited"]["sha256"])


def measure_inputs(part, output, archives=()):
    import io
    from dataclasses import asdict

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from app.detectors.image import jpeg_dq, jpeg_previous_grid, regional
    from app.detectors.image.adapters.regional import RegionalDetector, repeat_components
    from app.detectors.image.regional_support import readout
    from app.evidence.image_regional_jpeg import evaluated_jpeg, reencode

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    output.mkdir(mode=0o700)
    model = None

    def incumbent(body, truth, quality):
        nonlocal model
        anchors = []
        physical = jpeg_dq.analyse(body)
        if physical is not None and physical.width % 8 == physical.height % 8 == 0 and (area := physical.localised):
            anchors.append(measure_box(tuple(8 * coordinate for coordinate in (area.x, area.y, area.w, area.h)), truth))
        if evaluated_jpeg(body, quality=quality):
            if model is None:
                model = regional.load_regional(regional.MODEL_ID, "cpu", "float32").obj
            with Image.open(io.BytesIO(body)) as image:
                source = image.convert("RGB")
            rgb = np.array(source)

            def reading(value):
                guess = model.predict(value)
                return readout(guess.raw_mask, guess.candidate_eligibility["eligible"], RegionalDetector.REGION_THRESHOLD, body, rgb)

            native = reading(source)
            if quality == 90 and native.regions:
                with Image.open(io.BytesIO(reencode(source))) as image:
                    repeat = reading(image.convert("RGB"))
                regions, _, _ = repeat_components(native, repeat.selected_grid_mask)
                masks = [layer for region, layer in zip(native.regions[:2], native.component_masks, strict=True)
                         if region in regions]
            else:
                masks = native.component_masks
            for mask in masks:
                active = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(source.size, Image.Resampling.NEAREST)) > 0
                anchors.append({"reported": int(active.sum()), "intersection": int((active & truth).sum()), "truth": int(truth.sum())})
        return anchors

    rows = []
    for number, (folder, source, family, child) in enumerate(sources(part, archives), 1):
        parent_path, child_path = folder / "original.png", folder / "edited.png"
        original_bytes = parent_path.read_bytes()
        if digest(original_bytes) != source or digest(child_path.read_bytes()) != child:
            raise ValueError("The comparison inputs no longer match the source commitments")
        with Image.open(parent_path) as p, Image.open(child_path) as c:
            original, generated = [np.array(im.convert("RGB")) for im in (p, c)]
        captured = output / source[:12]
        captured.mkdir()
        q1, phase, parent, prior, views = authored(original, generated, source)
        (captured / "source.png").write_bytes(original_bytes)
        (captured / "source.jpg").write_bytes(parent)
        parent_sha = digest(parent)
        for role, pixels in views.items():
            direct = role in {"direct_original", "direct_edit"}
            if role in POSITIVES:
                truth = np.any(prior != pixels, axis=2)
            elif role == "direct_edit":
                truth = np.any(original != pixels, axis=2)
            else:
                truth = np.zeros(pixels.shape[:2], dtype=bool)
            truth_path = captured / f"{role}.png"
            Image.fromarray(truth.astype(np.uint8) * 255).save(truth_path)
            for quality in QUALITIES:
                data = encode(pixels, quality)
                target = captured / f"{role}-{quality}.jpg"
                target.write_bytes(data)
                model_view = jpeg_previous_grid.measure(data)
                record = asdict(model_view) if model_view else None
                if record:
                    for proposal in record["regions"].values():
                        if proposal:
                            proposal["numbers"] = measure_box(proposal["box"], truth)
                rows.append({"source": source, "parent_delivery_sha256": source if direct else parent_sha,
                             "parent": (captured / ("source.png" if direct else "source.jpg")).relative_to(output).as_posix(),
                             "generator": family, "role": role, "quality": quality,
                             "previous_quality": None if direct else q1, "phase": None if direct else phase,
                             "file": target.relative_to(output).as_posix(), "input_sha256": digest(data),
                             "mask": truth_path.relative_to(output).as_posix(), "mask_sha256": digest(truth_path.read_bytes()),
                             "pixels_changed": int(truth.sum()), "shape": list(truth.shape),
                             "current": incumbent(data, truth, quality), "new": record})
        print(f"Measured source {number}: {source[:12]}", flush=True)
    save(output / "measurements.json.gz", {"schema": SCHEMA, "rows": rows,
                                            "measurement_sha256": digest(Path(jpeg_previous_grid.__file__).read_bytes())})


def tally(rows, arm, point):
    counts = defaultdict(Counter)
    for row in rows:
        outcome = counts[row["role"]]
        record = row["new"]
        selected = record["regions"].get(arm) if record else None
        flagged = bool(selected and selected["score"] > point)
        current = bool(row["current"])
        found = any(map(useful, row["current"]))
        new = bool(flagged and useful(selected["numbers"]))
        outcome.update(deliveries=1, measured=bool(record), correct_phase=bool(record and record["phase"] == row["phase"]),
                       current_flags=current, current_correct=found, new_flags=flagged and not current, new_correct=new and not found,
                       top_region_correct=bool(selected and useful(selected["numbers"])), correct_after_empty_baseline=new and not current,
                       misplaced_after_empty_baseline=flagged and not current and row["pixels_changed"] > 0 and not new)
    return counts


def freeze(development, output):
    data = read(development)
    arms = sorted({key for row in data["rows"] if row["new"] for key in row["new"]["regions"]})
    values = {}
    for quality in QUALITIES:
        cohort = [row for row in data["rows"] if row["quality"] == quality]
        for arm in arms:
            maxima = defaultdict(float)
            for row in cohort:
                if row["pixels_changed"] or row["current"] or not row["new"]:
                    continue
                if selected := row["new"]["regions"][arm]:
                    maxima[row["source"]] = max(maxima[row["source"]], selected["score"])
            ranked = sorted([*maxima.values(), 0., 0., 0., 0.], reverse=True)
            values[f"{quality}/{arm}"] = {"stricter": ranked[1], "wider": ranked[3]}
    save(output, {"schema": SCHEMA, "points": values, "development_sha256": digest(development.read_bytes()),
                  "measurement_sha256": data["measurement_sha256"], "development_groups": sorted({r["source"] for r in data["rows"]})})


def results(rows, points):
    output = {}
    for entry, cutoffs in points.items():
        quality, arm = entry.split("/", 1)
        deliveries = [row for row in rows if row["quality"] == int(quality)]
        for name, point in cutoffs.items():
            output[f"{entry}/{name}"] = tally(deliveries, arm, point)
    return output


def assess(fixed, held, output):
    frozen, measured = read(fixed), read(held)
    if (frozen["measurement_sha256"] != measured["measurement_sha256"]
            or set(frozen["development_groups"]) & {row["source"] for row in measured["rows"]}):
        raise ValueError("Confirmation must use the frozen measurements and independent sources")
    result = {"schema": SCHEMA, "freeze": frozen, "held_sha256": digest(held.read_bytes()), "rows": measured["rows"],
              "outcomes": results(measured["rows"], frozen["points"])}
    save(output, result)


def bindings(receipt, output):
    """Independently read the encoded inputs, parents and exact masks on disk."""
    import numpy as np
    from PIL import Image

    from app.bench import independent
    from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs

    rows = read(receipt)["rows"]
    root = receipt.parent
    cases, index = [], []
    for row in rows:
        case_id = row["file"].removesuffix(".jpg").replace("/", "-")
        mask_data = (root / row["mask"]).read_bytes()
        if digest(mask_data) != row["mask_sha256"]:
            raise ValueError("A source-pixel truth map changed")
        with Image.open(root / row["mask"]) as image:
            pixels = np.asarray(image, dtype=np.uint8)
        truth = pixels > 0
        if (not np.all((pixels == 0) | (pixels == 255)) or list(truth.shape) != row["shape"]
                or int(truth.sum()) != row["pixels_changed"]):
            raise ValueError("Unexpected truth map shape or label count")
        if row["new"]:
            for mark in row["new"]["regions"].values():
                if mark and measure_box(mark["box"], truth) != mark["numbers"]:
                    raise ValueError("A recorded rectangle does not match the source-pixel truth")
        cases.append({"id": case_id, "modality": "image", "split": "confirmation", "label": int(bool(truth.any())),
                      "group": row["source"], "lineage": [row["source"]], "family": row["generator"],
                      "condition": f'{row["role"]}-jpeg{row["quality"]}', "input_sha256": row["input_sha256"],
                      "source_sha256": row["parent_delivery_sha256"], "localization_truth": {
                          "unit": "pixels", "source_units": truth.size, "target_units": row["pixels_changed"]},
                      "license": {"id": "BG-REAL CC BY 4.0; source photographs CC BY 2.0",
                                  "evaluation_permitted": True, "redistribution_permitted": False}})
        index.append({"id": case_id, "input": row["file"], "source": row["parent"]})
    document = {"schema": independent.SCHEMA, "cases": cases}
    file_index = {"schema": INDEX_SCHEMA, "manifest_sha256": independent.audit_manifest(document)["manifest_sha256"],
                  "cases": index}
    verified = verify_inputs(document, file_index, root)
    save(output, {"manifest": document, "file_index": file_index, "verified": verified})
    print(verified, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="action", required=True)
    for command, args in (("plan", ("metadata", "retained", "output")), ("acquire", ("plan", "retained", "output")),
                          ("measure", ("part", "output")), ("freeze", ("development", "output")),
                          ("assess", ("fixed", "held", "output")), ("review", ("receipt",)),
                          ("bindings", ("receipt", "output")), ("restore", ("roster", "output"))):
        action = actions.add_parser(command)
        for arg in args:
            action.add_argument(arg, type=Path)
        if command == "measure":
            action.add_argument("--archives", nargs=3, type=Path, default=())
    values = vars(cli.parse_args())
    command = values.pop("action")
    if command == "review":
        data = read(values["receipt"])
        checked = results(data["rows"], data["freeze"]["points"])
        if data["schema"] != SCHEMA or data["outcomes"] != checked:
            raise ValueError("The localization receipt does not recount")
        print(json.dumps(checked, sort_keys=True))
    else:
        {"plan": new_sources, "acquire": acquire, "measure": measure_inputs, "freeze": freeze,
         "assess": assess, "bindings": bindings, "restore": restore}[command](**values)
