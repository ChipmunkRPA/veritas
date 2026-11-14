# SPDX-License-Identifier: Apache-2.0
"""Recheck padding stability on publisher-anchored PNG edits from other photo groups."""
from __future__ import annotations

import argparse
import io
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from scripts import image_edge_context_transfer as parent
from scripts.image_small_region_repeat import CHECKPOINT, read, require, save, sha, useful

ROOT = parent.ROOT
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")
PARENT = parent.PUBLIC / "image-edge-context-transfer-2026-11-14.json"
SCHEMA = "chipmunk.image-edge-context-new-sources/1"
FAMILIES = ("lama", "sd15", "kandinsky")
TARGET = {"lama": 10, "sd15": 10, "kandinsky": 6}
ROLES = ("sharp", "feathered", "original", "local_tone", "local_blur", "local_copy")
ARMS = ("current", "edge", "supported", "flip", "flip_and_edge")
METHOD = {
    "roster": "Pinned BG-REAL assets, no downstream detector score used. Exclude credited parent and retained exposures.",
    "roles": list(ROLES),
    "delivery": "RGB PNG at original 768-side geometry. Native publisher edits for F5. A fixed in-mask "
                "quarter-width by quarter-height native patch for G2. Fixed eight-pixel inward feather and matched controls.",
    "comparison": "Current, edge-only, and original components with half support as in the parent. Also separately "
                  "apply the local app horizontal reflection rule and require support from both repeat views.",
    "useful": "IoU at least 0.30 and exact changed-pixel recall at least 0.50 on individual edited deliveries.",
}


def public_roster(path):
    plan_path = path / "plan.json"
    frozen, intake = read(plan_path), read(path / "intake.json")
    require(intake["plan_sha256"] == sha(plan_path.read_bytes()), "The acquired roster has changed")
    sources = []
    for record in intake["results"]:
        if record["status"] != "accepted":
            continue
        case = frozen["inputs"][record["slot"]]
        fields = ("slot", "generator", "recipient", "size", "publisher_split", "original", "edited", "semantic_mask")
        public = {key: case[key] for key in fields}
        if "semantic_edit_area" in case:
            public["semantic_edit_area"] = case["semantic_edit_area"]
        if "processed" in case:
            public["processed"] = case["processed"]
        sources.append(public | {"proof": {key: record[key] for key in ("box", "truth_sha256", "assets")}})
    return sources


def plan(metadata, archive):
    import yaml

    from scripts import image_bg_real_insertion_transfer as intake
    from scripts import image_three_class_patch_fresh as fresh

    roster = intake.roster
    keys, known, historical = intake.exposure()
    seen = {case["source_sha256"] for case in read(PARENT)["freeze"]["cases"]}
    for file in (archive / "image-bg-real-small-insertions-2026-11-11/trial/plan.json",
                 archive / "image-bg-real-remote-context-2026-11-11/trial/plan.json",
                 archive / "image-background-patch-transfer-2026-11-11/intake/plan.json"):
        historical[file.relative_to(archive).as_posix()] = roster.sha(file)
        for case in read(file)["cases"]:
            keys.update(roster.source_keys(case))
            seen.add(case["original"]["sha256"])
    for name, (_, size, digest) in roster.METADATA.items():
        path = metadata / name
        require(path.stat().st_size == size and roster.sha(path) == digest, f"Pinned {name} changed")
    rows = roster.table(metadata / "manifest_v2.csv")
    index = roster.indexed(rows, "sample_id")
    credits = roster.indexed((r for r in roster.table(metadata / "attribution.csv") if r["role"] == "recipient"), "bgreal_id")
    recipients = roster.indexed(roster.table(metadata / "recipients.csv"), "source_image_id")
    split = {key: set(values) for key, values in yaml.safe_load((metadata / "splits_v2.yaml").read_bytes()).items()}
    models = {row["key"]: row for row in roster.table(metadata / "model_revisions.csv")}
    require(models["lama"] == roster.MODEL and models["sd15_inpaint"]["revision"] == intake.GENERATOR_REVISION,
            "The F5 generator pins changed")
    require(models["kandinsky_inpaint"]["revision"] == fresh.bounded.GENERATORS["kandinsky"][4]
            and models["kandinsky_inpaint"]["license"] == "apache-2.0", "The Kandinsky generator pin changed")
    candidates = defaultdict(list)
    for edit in rows:
        if edit["family"] not in {"F5", "G2"} or not roster.base_png(edit):
            continue
        dims = roster.dimensions(edit)
        if max(dims) != 768 or min(dims) < 512:
            continue
        src = edit["source_image_id"]
        try:
            credit = roster.one(credits, src, "A single checked credit is required")
            if edit["family"] == "G2" and edit["generator"] == models["kandinsky_inpaint"]["repo"]:
                case = fresh.full_background(edit, index, credits, split, "kandinsky")
                family = "kandinsky"
            elif edit["family"] == "F5" and .004 <= float(edit["edit_area"]) <= .12:
                options = {("big-lama", "object_removal"): "lama", (intake.GENERATOR, "object_insertion"): "sd15"}
                family = options.get((edit["generator"], edit["edit_kind"]))
                if family is None:
                    continue
                case = roster.pair(edit, roster.one(index, f"{src}_A0", "Original is required"), credit,
                                   roster.one(recipients, src, "A single source is required"), split)
            else:
                continue
        except ValueError:
            continue
        if case["original"]["sha256"] not in seen:
            candidates[family].append(case | {"generator": family})
    queue = []
    availability = {}
    for family in FAMILIES:
        distinct, _ = roster.select(candidates[family], keys)
        distinct.sort(key=lambda case: sha(f"{SCHEMA}/{case['recipient']}".encode()))
        availability[family] = len(distinct)
        queue.extend(distinct[:2 * TARGET[family]])
    return {"schema": SCHEMA, "method": METHOD, "checkpoint": CHECKPOINT, "publisher_revision": roster.REVISION,
            "publisher_metadata": {name: entry[2] for name, entry in roster.METADATA.items()},
            "historical": historical, "parent_sha256": sha(PARENT.read_bytes()),
            "availability": availability, "target": TARGET,
            "generators": {key: models[key] for key in ("lama", "sd15_inpaint", "kandinsky_inpaint")},
            "inputs": [row | {"slot": i} for i, row in enumerate(queue)]}, (keys, known)


def read_source(case, blobs):
    import numpy as np

    from scripts import image_bg_real_acquisition_v221 as pixels
    from scripts import image_three_class_patch_fresh as fresh

    if case["generator"] == "kandinsky":
        views, truth, box, identities = fresh.bind(case, blobs)
    else:
        original, edited, semantic = (pixels.png_pixels(blobs[role], case["size"])
                                      for role in ("original", "edited", "semantic_mask"))
        truth, _ = pixels.edit_truth(original, edited, semantic, case["semantic_edit_area"])
        views = {"original": original, "edited": edited}
        identities = {role: pixels.fingerprint(blobs[role], rgb) for role, rgb in views.items()}
        box = None
    require(np.array_equal(truth, np.any(views["edited"] != views["original"], axis=2)), "Exact edit changed")
    return views, truth, list(box) if box else None, identities


def acquire(path, metadata, archive):
    import requests

    from scripts import image_bg_real_acquisition_v221 as pixels
    from scripts import image_bg_real_roster_v218 as roster
    from scripts import image_bg_real_transport_v225 as transport

    frozen, (keys, known) = plan(metadata, archive)
    path.mkdir(mode=0o700)
    save(path / "plan.json", frozen)
    results, accepted, names = [], Counter(), set()
    with (path / "intake.jsonl").open("x") as journal, requests.Session() as session:
        session.trust_env = False
        for case in frozen["inputs"]:
            family, slot = case["generator"], case["slot"]
            creator = case["photo_attribution"]["Author"].strip().casefold()
            if accepted[family] >= TARGET[family]:
                continue
            record = {"slot": slot, "family": family}
            try:
                require(creator not in names and not roster.source_keys(case) & keys, "Known photographer or photo")
                roles = ("original", "edited", "semantic_mask")
                if family == "kandinsky":
                    roles += ("processed",)
                bodies = {role: transport.download(session, case[role]["path"], case[role]["sha256"],
                                                    time.monotonic() + 32)[0] for role in roles}
                _, exact, box, identities = read_source(case, bodies)
                require(not pixels.prior_matches(case, identities, keys, known), "Prior exact or near source match")
                folder = path / str(slot)
                folder.mkdir(mode=0o700)
                for role, body in bodies.items():
                    (folder / f"{role}.png").write_bytes(body)
                record |= {"status": "accepted", "box": box, "truth_sha256": sha(exact.tobytes()),
                           "assets": {role: sha(body) for role, body in bodies.items()}}
                names.add(creator)
                keys.update(roster.source_keys(case))
                for ident in identities.values():
                    known[0].append(ident)
                    known[1].add(ident["encoded_sha256"])
                    known[2].add(ident["rgb_sha256"])
                accepted[family] += 1
            except (OSError, ValueError, requests.RequestException) as error:
                record |= {"status": "rejected", "reason": str(error)}
            results.append(record)
            print(json.dumps(record, sort_keys=True), file=journal, flush=True)
            print(f"{family} {slot}: {record['status']} ({accepted[family]}/{TARGET[family]})", flush=True)
    save(path / "intake.json", {"plan_sha256": sha((path / "plan.json").read_bytes()), "accepted": dict(accepted),
                                "results": results})


def deliveries(path, sources):
    import numpy as np
    from PIL import Image

    from scripts.image_residual_encoder_transfer import authored

    for case in sources:
        record = case["proof"]
        bodies = {role: (path / str(case["slot"]) / f"{role}.png").read_bytes() for role in record["assets"]}
        require({role: sha(body) for role, body in bodies.items()} == record["assets"], "Retained photo bytes changed")
        views, truth, box, _ = read_source(case, bodies)
        require(box == record["box"] and sha(truth.tobytes()) == record["truth_sha256"], "Known mask changed")
        original, group = views["original"], case["original"]["sha256"]
        ordinary, edited = authored(views, truth, group)
        for role in ROLES:
            pixels = (ordinary | edited)[role]
            truth = np.any(pixels != original, axis=2) if role in edited else np.zeros(original.shape[:2], dtype=bool)
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            yield case["generator"], group, group, role, buffer.getvalue(), pixels, truth, None


def hydrate(path, receipt):
    """Retrieve only the accepted benchmark sources at the pinned revision."""
    import requests

    from scripts import image_bg_real_roster_v218 as roster
    from scripts import image_bg_real_transport_v225 as transport

    require(receipt["publisher_revision"] == roster.REVISION and summary(receipt) == receipt["summary"],
            "Cannot reconstruct an unsupported receipt")
    path.mkdir(mode=0o700)
    with requests.Session() as session:
        session.trust_env = False
        for case in receipt["sources"]:
            folder = path / str(case["slot"])
            folder.mkdir(mode=0o700)
            for role, digest in case["proof"]["assets"].items():
                body, _ = transport.download(session, case[role]["path"], digest, time.monotonic() + 32)
                (folder / f"{role}.png").write_bytes(body)


def measure(model, item):
    import numpy as np
    from PIL import Image

    from app.detectors.image import regional, regional_support

    row = parent.measure(model, item, None)
    data, rgb, truth = item[4:7]
    if not row["support"]:
        return row | {"flip_shared": [], "flip_patch_sha256": None}

    def readout(prediction, source):
        return regional_support.readout(prediction.raw_mask, prediction.candidate_eligibility["eligible"],
                                        regional.PIXEL_THRESHOLD, data, source)

    original = model.predict(Image.fromarray(rgb))
    measured = readout(original, rgb)
    repeated = np.ascontiguousarray(rgb[:, ::-1])
    reflected = model.predict(Image.fromarray(repeated))
    aligned = readout(reflected, repeated).selected_grid_mask[:, ::-1]
    counts = [(int(mask.sum()), int((mask & aligned).sum())) for mask in measured.component_masks]
    require([size for size, _ in counts] == [part[0] for part in row["support"]]
            and measured.source_mask(*rgb.shape[:2][::-1]).sum() == row["current"][0]
            and (measured.source_mask(*rgb.shape[:2][::-1]) & truth).sum() == row["current"][1],
            "Current region changed on reflection replay")
    return row | {"flip_shared": [shared for _, shared in counts],
                  "flip_patch_sha256": reflected.binding["actual_patch_float32_sha256"]}


def summary(receipt):
    groups, counts = defaultdict(set), defaultdict(Counter)
    flagged, affected = defaultdict(set), defaultdict(set)
    require(receipt["schema"] == SCHEMA and receipt["method"] == METHOD and receipt["checkpoint"] == CHECKPOINT,
            "Study contract differs")
    indexed = {case["original"]["sha256"]: case for case in receipt["sources"]}
    require(len(indexed) == len(receipt["sources"]) == len({case["slot"] for case in receipt["sources"]}),
            "Repeated source or slot")
    for case, row in zip(receipt["cases"], receipt["rows"], strict=True):
        family, group, role = (case[k] for k in ("cohort", "group", "role"))
        width, height = case["size"]
        target, parts = case["truth_pixels"], row["support"]
        current = row["current"]
        source = indexed[group]
        require(source["generator"] == family and source["size"] == case["size"]
                and row["body_sha256"] == case["body_sha256"] and max(width, height) == 768 and min(width, height) >= 512
                and bool(target) == (role in ("sharp", "feathered")) and role not in groups[(family, group)],
                "Source roles, geometry or identity changed")
        groups[(family, group)].add(role)
        rec = row["flip_shared"]
        require(len(rec) == len(parts) <= 2 and all(0 <= shared <= area and 0 <= flip <= area
                for (area, shared, _, _), flip in zip(parts, rec, strict=True))
                and [sum(part[k] for part in parts) for k in (2, 3)] == current, "Baseline component binding differs")

        def retained(parts, reflections, *, edge, flip):
            chosen = [p for p, reflection in zip(parts, reflections, strict=True)
                      if (not edge or 2 * p[1] >= p[0]) and (not flip or 2 * reflection >= p[0])]
            return [sum(p[i] for p in chosen) for i in (2, 3)]

        measures = (current, row["edge"], retained(parts, rec, edge=True, flip=False),
                    retained(parts, rec, edge=False, flip=True), retained(parts, rec, edge=True, flip=True))
        baseline = useful(current, target)
        local = useful(measures[3], target)
        for arm, observed in zip(ARMS, measures, strict=True):
            area, overlap = observed
            require(0 <= area <= width*height and 0 <= overlap <= min(area, target), "Exact pixel overlap is invalid")
            good = useful(observed, target)
            key = family, arm
            counts[key].update({role: 1, role + "_flags": bool(area), "useful": good, role + "_useful": good,
                                "misplaced": bool(target and area and not good), "lost_from_current": baseline and not good,
                                "gained_from_current": good and not baseline, "lost_from_flip": local and not good,
                                "gained_from_flip": good and not local})
            if not target and area:
                flagged[key].add(group)
            if good:
                affected[key].add(group)
    require({group for _, group in groups} == indexed.keys()
            and all(family in FAMILIES and roles == set(ROLES) for (family, _), roles in groups.items()),
            "Incomplete source roles")
    flattened = {f"{family}/{arm}": dict(row) | {"ordinary_groups": len(flagged[key]), "useful_groups": len(affected[key])}
                 for key, row in counts.items() for family, arm in (key,)}
    return {"groups": dict(Counter(f for f, _ in groups)), "counts": flattened}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("acquire", "measure", "review", "hydrate", "verify-inputs"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--metadata", type=Path, default=ARCHIVE / "image-bg-real-metadata-v218")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/weights/native-regional-v1/regional-model.safetensors")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        receipt = read(args.path)
        result = summary(receipt)
        require(result == receipt["summary"], "Receipt counts changed")
    elif args.phase == "acquire":
        acquire(args.path, args.metadata, args.archive)
        return
    elif args.phase == "hydrate":
        hydrate(args.path, read(args.receipt))
        return
    elif args.phase == "verify-inputs":
        receipt = read(args.receipt)
        require(summary(receipt) == receipt["summary"], "Numeric receipt differs")
        actual = [parent.identity(item) for item in deliveries(args.path, receipt["sources"])]
        require(actual == receipt["cases"], "Reconstructed inputs or masks differ")
        result = {"verified_inputs": len(actual)}
    else:
        import cv2
        import torch

        from app.detectors.image import regional

        cv2.setNumThreads(1)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        published = read(args.receipt) if args.receipt else None
        sources = published["sources"] if published else public_roster(args.path)
        frozen = [parent.identity(item) for item in deliveries(args.path, sources)]
        if published:
            require(frozen == published["cases"], "Input replay differs from the receipt")
        else:
            require(Counter(case["generator"] for case in sources) == TARGET, "A planned source group is missing")
        receipt = {"schema": SCHEMA, "method": METHOD, "checkpoint": CHECKPOINT, "cases": frozen,
                   "sources": sources, "publisher_revision": "aaba6cd1ced4ce3c846be9641fed092adb4919d1",
                   "acquisition_sha256": (published["acquisition_sha256"] if published
                                          else sha((args.path / "intake.json").read_bytes())),
                   "producer_sha256": sha(Path(__file__).read_bytes()), "rows": []}
        save(args.path / "freeze.json", receipt)
        model = regional.load_model(args.checkpoint)
        with (args.path / "scores.jsonl").open("x") as journal:
            for i, (item, case) in enumerate(zip(deliveries(args.path, sources), frozen, strict=True), 1):
                require(parent.identity(item) == case, "Measured source differs from the preflight")
                score = measure(model, item)
                print(json.dumps(score, sort_keys=True), file=journal, flush=True)
                receipt["rows"].append(score)
                if i % 12 == 0:
                    print(f"Measured {i}/{len(frozen)}", flush=True)
        result = summary(receipt)
        save(args.path / "receipt.json", receipt | {"summary": result})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
