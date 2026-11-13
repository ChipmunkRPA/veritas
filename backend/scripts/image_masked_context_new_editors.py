# SPDX-License-Identifier: Apache-2.0
"""Apply frozen masked-context image checks to fresh neural object removals."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml

from scripts import image_masked_context_transfer as context
from scripts import image_three_class_patch_fresh as fresh

SCHEMA = "chipmunk.image-masked-context-new-editors/1"
FAMILIES = ("lama", "sdxl", "kandinsky")
POINTS_SHA = "1c3f8fd558ac06c672fb07a3cc59e5cedbb6e30175b12dd0b2fec567536f4a29"
DESIGN = {
    "scope": "Six new BG-REAL LaMa neural object-removal photos, plus a single paired diffusion photo across two editor variants if available.",
    "selection": "Use metadata rights, source overlap and a salted roster order before media or detector scores.",
    "delivery": "Same JPEG75 4:2:0 delivery and both frozen context points. No calibration or fitting.",
    "controls": "Untouched, same-site local blur/tone/copy, resized round trip, intermediate JPEG; diffusion includes matched local/global processing.",
    "useful": context.DESIGN["useful"],
    "no_admission_reason": "Small one-family external transfer and one shared diffusion source cannot establish independent cross-family breadth.",
}
read, write, require, sha = context.read, context.write, context.require, context.sha
roster, pixels = fresh.roster, fresh.pixels


def exposure(args, metadata, prior):
    keys, known, bindings, recipients, names = fresh.exposure(args, metadata, prior)
    earlier, exposed = args.output_fresh, fresh.read(args.output_fresh / "plan.json")
    intake = read(earlier / "intake.json")
    for key, filename in (("plan_sha256", "plan.json"), ("journal_sha256", "intake.jsonl")):
        require(intake[key] == roster.sha(earlier / filename), "Prior transfer intake changed")
        bindings[f"prior_transfer/{filename}"] = intake[key]
    journal = [json.loads(line) for line in (earlier / "intake.jsonl").read_text().splitlines()]
    for case, row in zip(exposed["cases"], journal, strict=True):
        if row["status"] != "qualified":
            continue
        keys.update(roster.source_keys(case))
        recipients.add(case["recipient"])
        names.add(case["photo_attribution"]["Author"].strip().casefold())
        for info in row["fingerprints"].values():
            known[0].append(info)
            known[1].add(info["encoded_sha256"])
            known[2].add(info["rgb_sha256"])
    return keys, known, bindings, recipients, names


def source_plan(args, prior):
    metadata = args.archive / "image-bg-real-metadata-v218"
    for filename, (_, size, digest) in roster.METADATA.items():
        asset = metadata / filename
        require(asset.stat().st_size == size and roster.sha(asset) == digest, "Pinned publisher metadata changed")
    keys, known, bindings, recipients, names = exposure(args, metadata, prior)
    sources = roster.table(metadata / "manifest_v2.csv")
    source_index = roster.indexed(sources, "sample_id")
    people = roster.indexed(roster.table(metadata / "recipients.csv"), "source_image_id")
    credits = roster.indexed((r for r in roster.table(metadata / "attribution.csv") if r["role"] == "recipient"), "bgreal_id")
    memberships = {key: set(rows) for key, rows in yaml.safe_load((metadata / "splits_v2.yaml").read_bytes()).items()}
    models = {row["key"]: row for row in roster.table(metadata / "model_revisions.csv")}
    require(models["lama"] == roster.MODEL, "LaMa rights record differs")
    for name in FAMILIES[1:]:
        _, _, _, model_id, revision, license_id = fresh.bounded.GENERATORS[name]
        require(models[model_id]["revision"] == revision and models[model_id]["license"] == license_id,
                "Diffusion rights record differs")
    candidates = defaultdict(list)
    for row in sources:
        source = row["source_image_id"]
        if source in recipients or not roster.base_png(row):
            continue
        try:
            if (row["family"] == "F5" and row["generator"] == "big-lama" and row["edit_kind"] == "object_removal"
                    and .004 <= float(row["edit_area"]) <= .08):
                item = roster.pair(row, roster.one(source_index, f"{source}_A0", "Original required"),
                                   roster.one(credits, source, "Photo rights required"),
                                   roster.one(people, source, "Source identity required"), memberships)
                family = "lama"
            elif row["family"] in ("G1", "G2"):
                family = "sdxl" if row["family"] == "G1" else "kandinsky"
                require(row["generator"] == models[fresh.bounded.GENERATORS[family][3]]["repo"], "Unexpected generator record")
                item = fresh.full_background(row, source_index, credits, memberships, family)
            else:
                continue
            if item["photo_attribution"]["Author"].strip().casefold() not in names:
                candidates[family].append(item | {"generator": family})
        except ValueError:
            continue
    ordered = {name: roster.select(candidates[name], keys)[0] for name in FAMILIES}
    require([len(ordered[name]) for name in FAMILIES] == [18, 1, 1], "Expected exact untouched metadata counts")
    lama = sorted(ordered["lama"], key=lambda item: sha(f"{SCHEMA}/{item['recipient']}".encode()))[:8]
    paired = [ordered[key][0] for key in FAMILIES[1:]]
    require(paired[0]["recipient"] == paired[1]["recipient"], "Expected one shared diffusion source")
    require(not any(roster.source_keys(item) & roster.source_keys(paired[0]) for item in lama), "Source group overlap")
    return {
        "schema": SCHEMA, "design": DESIGN, "publisher_revision": roster.REVISION,
        "exposure_bindings": bindings, "eligible": {name: len(values) for name, values in ordered.items()},
        "model_records": {name: models[name] for name in ("lama", "sdxl_inpaint", "kandinsky_inpaint")},
        "cases": [item | {"slot": index} for index, item in enumerate(lama + paired)],
    }, known


def bound(item, bodies):
    if item["generator"] != "lama":
        return fresh.bind(item, bodies)
    a, b, support = (pixels.png_pixels(bodies[key], item["size"]) for key in ("original", "edited", "semantic_mask"))
    exact, _ = pixels.edit_truth(a, b, support, item["semantic_edit_area"])
    require(.004 <= float(exact.mean()) <= .08, "Exact removal area outside declared bounds")
    native = {"original": a, "edited": b}
    signatures = {key: pixels.fingerprint(bodies[key], image) for key, image in native.items()}
    return native, exact, None, signatures


def bind_views(item, bodies, known, shared_source):
    native, exact, crop, signatures = bound(item, bodies)
    tested = {key: value for key, value in signatures.items() if key != "original" or not shared_source}
    duplicate_reference = ([], known[1], known[2]) if shared_source else known
    require(not pixels.prior_matches(item, tested, set(), duplicate_reference), "Previously exposed source or edit")
    for key, info in tested.items():
        known[1].add(info["encoded_sha256"])
        known[2].add(info["rgb_sha256"])
        # The paired diffusion challenges intentionally share a source. Compare their exact identities but
        # keep the first source's thumbnails out of the duplicate filter for the same declared photo.
        if item["generator"] == "lama" or key == "original":
            known[0].append(info)
    return native, exact, {"crop": list(crop) if crop else None, "mask_sha256": sha(exact.tobytes()),
                           "source_pixels": exact.size, "changed_pixels": int(exact.sum()), "fingerprints": signatures}


def acquire(args, plan, known):
    directory = args.output / "assets"
    directory.mkdir(mode=0o700)
    results, count = [], 0
    deadline = time.monotonic() + 270
    paired_source = None
    with requests.Session() as session:
        session.trust_env = False
        for item in plan["cases"]:
            slot, family = item["slot"], item["generator"]
            row = {"slot": slot, "family": family, "proofs": {}}
            if family == "lama" and count == 6:
                row["status"] = "reserve_unopened"
            else:
                roles = ("original", "edited", "semantic_mask") if family == "lama" else fresh.ASSETS
                try:
                    bodies = {}
                    for role in roles:
                        asset = item[role]
                        body, proof = fresh.transport.download(session, asset["path"], asset["sha256"], deadline)
                        bodies[role] = body
                        row["proofs"][role] = proof
                    shared_source = paired_source is not None and family != "lama"
                    if shared_source:
                        require(sha(bodies["original"]) == paired_source, "Paired earlier photo changed")
                    _, _, info = bind_views(item, bodies, known, shared_source)
                    for role, body in bodies.items():
                        (directory / f"{slot:02d}-{role}.png").write_bytes(body)
                    row.update(status="qualified", **info)
                    if family == "lama":
                        count += 1
                    else:
                        paired_source = sha(bodies["original"])
                except (ValueError, requests.RequestException) as error:
                    row.update(status="excluded", reason=str(error) if isinstance(error, ValueError) else type(error).__name__)
            print(slot, family, row["status"], row.get("reason", ""), flush=True)
            results.append(row)
    return {"rows": results, "plan_sha256": roster.sha(args.output / "plan.json")}


def views(args, plan, known):
    intake = read(args.output / "intake.json")
    require(intake["plan_sha256"] == roster.sha(args.output / "plan.json"), "Frozen intake differs")
    paired_source = None
    for item, row in zip(plan["cases"], intake["rows"], strict=True):
        if row["status"] != "qualified":
            continue
        slot, family = item["slot"], item["generator"]
        require(row["slot"] == slot and row["family"] == family, "Source attempt differs")
        bodies = {}
        for role, proof in row["proofs"].items():
            body = (args.output / "assets" / f"{slot:02d}-{role}.png").read_bytes()
            require(proof["resolver_revision"] == roster.REVISION and len(body) == proof["bytes"]
                    and sha(body) == proof["sha256"] == (item[role]["sha256"] or proof["publisher_asset_commitment"]),
                    "Qualified asset differs")
            bodies[role] = body
        shared_source = paired_source is not None and family != "lama"
        require(not shared_source or paired_source == sha(bodies["original"]), "Paired earlier photo changed")
        native, exact, witness = bind_views(item, bodies, known, shared_source)
        require(all(row[key] == value for key, value in witness.items()), "Native evidence binding differs")
        if family == "lama":
            controls, edits = context.prior.base.authored(native, exact, item["recipient"])
            for role, rgb in (controls | edits).items():
                body = context.prior.base.encode(rgb, 75)
                truth = np.any(rgb != native["original"], axis=2) if role in context.prior.HELD_POSITIVE else np.zeros(exact.shape, bool)
                yield {"source": item["recipient"], "family": family, "slot": slot, "role": role,
                       "input_sha256": sha(body), "truth_sha256": sha(truth.tobytes())}, body, truth
        else:
            paired_source = sha(bodies["original"])
            for key, body, mask, _ in fresh.views(item, native, exact):
                yield key, body, mask


def measures(args, plan, known, head, points):
    measure, positive_control, calls = context.extractor(args.checkpoint, head)
    rows = []
    for key, body, truth in views(args, plan, known):
        fields, rgb = measure(body)
        arms = context.assess(fields, truth, points, {})
        reference = context.prior.base.comparison.format_reference("jpeg75", body, rgb)
        box = reference["box"] if reference and reference["selected"] else None
        overlap = context.prior.overlap(truth, box)
        arms["current_luma"] = {"box": box, "flagged": box is not None, "score": None, "overlap": overlap,
                                "useful": context.prior.base.geometry.useful(overlap)}
        rows.append(key | {"arms": arms})
        if len(rows) % 8 == 0:
            print("scored", len(rows), flush=True)
    return rows, positive_control, calls


def report(rows):
    totals, groups = defaultdict(Counter), defaultdict(set)
    for row in rows:
        positive = row["role"] in context.prior.HELD_POSITIVE
        role = row["role"] if positive else "ordinary"
        for arm, hit in row["arms"].items():
            key = f"{row['family']}/{role}/{arm}"
            totals[key].update(files=1, flags=hit["flagged"], useful=hit["flagged"] and hit["useful"],
                               useful_before=hit["useful"], misplaced=positive and hit["flagged"] and not hit["useful"])
            if not positive and hit["flagged"]:
                groups[key].add(row["source"])
    return {"strata": totals, "ordinary_flagged_groups": {key: len(value) for key, value in groups.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "acquire", "run", "review"))
    for key in ("archive", "prior-probe", "residual", "early-f5", "later-f5", "checkpoint", "output-fresh", "masked", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    args.exclude = []
    cv2.setNumThreads(1)
    lock, heads, old = fresh.previous(args)
    points = read(args.masked / "points.json")
    points_sha = roster.sha(args.masked / "points.json")
    require(points_sha == POINTS_SHA and points["three_class"] == old["three_class"], "Frozen model point differs")
    plan, known = source_plan(args, lock)
    plan.update(runner_sha256=roster.sha(Path(__file__)), masked_runner_sha256=roster.sha(Path(context.__file__)), points_sha256=points_sha)
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(args.output / "plan.json", plan)
        return
    require(plan == read(args.output / "plan.json"), "Prescoring freeze differs")
    if args.stage == "acquire":
        write(args.output / "intake.json", acquire(args, plan, known))
        return
    intake = read(args.output / "intake.json")
    qualified = Counter(row["family"] for row in intake["rows"] if row["status"] == "qualified")
    require(qualified["lama"] == 6, "Six new object-removal groups required")
    path = args.output / "measurements.json"
    if args.stage == "run":
        start = time.monotonic()
        rows, control, calls = measures(args, plan, known, heads["three_class"], points)
        write(path, rows)
        write(args.output / "result.json", report(rows) | {
            "schema": SCHEMA, "design": DESIGN, "point_sha256": points_sha, "points": points,
            "measurements_sha256": roster.sha(path), "qualified": qualified, "eligible": plan["eligible"],
            "intake_sha256": roster.sha(args.output / "intake.json"), "plan_sha256": roster.sha(args.output / "plan.json"),
            "delivery_calls": calls, "intervention_control": control, "seconds": round(time.monotonic() - start, 2),
            "frozen_diffusion_source_count": 1,
        })
        return
    result, rows, count = read(args.output / "result.json"), read(path), 0
    require(result["measurements_sha256"] == roster.sha(path), "Exact measurement journal differs")
    for row, (key, _, truth) in zip(rows, views(args, plan, known), strict=True):
        require(all(row[k] == value for k, value in key.items()), "Exact source or mask differs")
        for arm, hit in row["arms"].items():
            overlap = context.prior.overlap(truth, hit["box"])
            require(overlap == hit["overlap"] and hit["useful"] == context.prior.base.geometry.useful(overlap)
                    and hit["flagged"] == (bool(hit["box"]) and (arm == "current_luma" or hit["score"] > points[arm])),
                    "Native overlap or saved flag differs")
            count += 1
    require(all(result[key] == value for key, value in report(rows).items()), "Outcomes differ")
    write(args.publication, result | {"audit": {"exact_deliveries_reproduced": len(rows), "native_measurements": count,
                                             "model_replayed": False}})


if __name__ == "__main__":
    main()
