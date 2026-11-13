# SPDX-License-Identifier: Apache-2.0
"""Transfer frozen JPEG patch heads to new source photos with no fitting."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml

from scripts import image_background_patch_transfer as bounded
from scripts import image_bg_real_acquisition_v221 as pixels
from scripts import image_bg_real_insertion_transfer as insertions
from scripts import image_bg_real_roster_v218 as roster
from scripts import image_bg_real_transport_v225 as transport
from scripts import image_jpeg_texture_rank_transfer as latest
from scripts import image_three_class_patch_probe as probe

SCHEMA = "chipmunk.image-three-class-patch-fresh/1"
PREVIOUS = {
    "freeze.json": "7041774c64d953496e4b6bafa5ab214f821c12cb8ea78b96d13b2f72bb1e9559",
    "heads.json": "5c17bdc8f5ba93e4ca942cc1dd68caf937b509e819b726c09a7239c27a9e3956",
    "points.json": "a6e9abc477534ecca746a6f96b2b7aca8d8f1f4b5f18336e5d0b5e7f86f8cb52",
}
FAMILIES = ("kandinsky", "sd15")
ASSETS = ("original", "edited", "processed", "semantic_mask")
CONTROLS = ("original", "local_processing", "global_processing", "local_blur", "local_copy", "local_tone")
CASES_PER_FAMILY, ATTEMPTS = 6, {"kandinsky": 7, "sd15": 8}
DESIGN = {
    "scope": "New credited photographs within the same publisher, excluding known local exposure. Fixed heads and original ordinary-only operating points.",
    "family_targets": {family: CASES_PER_FAMILY for family in FAMILIES},
    "maximum_ordered_attempts_per_family": ATTEMPTS,
    "insertion_native_semantic_area": [.004, .08],
    "background_edits": "Use the prior deterministic native 1/16-area rectangle inside the publisher's edited mask. Preserve all source pixels outside the measured footprint.",
    "feather": "Eight-pixel cosine transition contained in the exact sharp edit mask.",
    "encoding": "JPEG75 4:2:0, identical final encode for every view.",
    "controls": list(CONTROLS),
    "gate": probe.DESIGN["gate"],
    "useful": probe.DESIGN["useful"],
    "paired_diagnostic": "Compare median inpainted log-odds on the same token sites at 50% or more exact feathered occupancy. The region and truth are known to this diagnostic only.",
    "ranking_diagnostic": "Count useful proposed rectangles before the prior operating point; report whether the highest rectangle equals that of the untouched photo.",
    "confidence_weight": 0,
}
read, write, require, sha = probe.read, probe.write, probe.require, probe.sha


def previous(args):
    require(all(roster.sha(args.prior_probe / name) == digest for name, digest in PREVIOUS.items()),
            "Prior fit, heads or points changed")
    lock = read(args.prior_probe / "freeze.json")
    require(lock["runner_sha256"] == roster.sha(Path(probe.__file__)) and probe.checkpoint(args.checkpoint) == lock["checkpoint"],
            "Prior encoder and geometry must remain identical")
    heads = {key: np.asarray(value) for key, value in read(args.prior_probe / "heads.json").items()}
    points = read(args.prior_probe / "points.json")
    require(set(heads) == set(points) == set(probe.HEADS), "Previous head set differs")
    return lock, heads, points


def exposure(args, metadata, prior):
    old = argparse.Namespace(archive=args.archive, prior_f5=args.early_f5)
    keys, known, bindings, recipients, names = latest.exposure(old, metadata, read(args.residual / "freeze.json"))
    bindings["residual"] = roster.sha(args.residual / "freeze.json")
    require(bindings["residual"] == latest.FIT_FREEZE, "Earlier exposure bank differs")
    for rows in prior["profile"].values():
        for item in rows:
            keys.update({("owner", item["author"]), ("photo", item["photo"]), ("original_sha256", item["original_sha256"])})
            recipients.add(item["recipient"])
            names.add(item["credit"]["Author"].strip().casefold())
            known[0].append({"size": item["size"], "sketch": item["sketch"]})
            known[1].add(item["original_sha256"])
            known[2].update(item[key] for key in ("original_rgb_sha256", "edit_sha256", "process_sha256", "global_processing_sha256"))
    for folder in (args.later_f5, *args.exclude):
        path = folder / "plan.json"
        plan = read(path)
        bindings[f"excluded_{len(bindings)}"] = roster.sha(path)
        for case in plan["cases"]:
            keys.update(roster.source_keys(case))
            recipients.add(case["recipient"])
            names.add(case["photo_attribution"]["Author"].strip().casefold())
        for case, native, _ in latest.held_inputs(folder, plan):
            for rgb in native.values():
                info = pixels.fingerprint(b"", rgb)
                known[0].append(info)
                known[2].add(info["rgb_sha256"])
            known[1].add(case["original"]["sha256"])
    return keys, known, bindings, recipients, names


def full_background(edit, index, attribution, memberships, family):
    tag, control, control_generator, model_key, _, _ = bounded.GENERATORS[family]
    identity = edit["source_image_id"]
    split = bounded.SPLIT
    require(roster.SOURCE.fullmatch(identity) and edit["sample_id"] == f"{identity}_{tag}", "Unexpected background edit")
    original = roster.one(index, f"{identity}_A0", "Exact source PNG required")
    processed = roster.one(index, f"{identity}_{control}", "Exact processing PNG required")
    size = roster.dimensions(edit)
    require(256 <= min(size) <= max(size) <= 768 and all(
        row["source_image_id"] == identity and row["split"] == split and roster.base_png(row)
        and row["sample_id"] in memberships[split] and roster.dimensions(row) == size and roster.SHA.fullmatch(row["sha256"])
        for row in (edit, original, processed)
    ), "Unmatched native PNG or split")
    require(original["family"] == "A0" and original["label"] == "0" and original["generator"] == "none"
            and processed["family"] == "C2" and processed["label"] == "0" and processed["generator"] == control_generator
            and edit["family"] == tag and edit["label"] == "1" and not any(row["donor_id"] for row in (original, processed, edit)),
            "Incorrect original or generated control labels")
    credit = roster.one(attribution, identity, "Exact photo attribution required")
    owner, photo = roster.attribution(credit, identity)
    image_dir, mask_dir = Path("v2/images") / identity[3], Path("v2/masks/edit") / identity[3]
    fields = {
        key: {"path": roster.named_png(row["image_path"], image_dir, f"{identity}_{suffix}.png"), "sha256": row["sha256"]}
        for key, row, suffix in (("original", original, "A0"), ("edited", edit, tag), ("processed", processed, control))
    }
    fields["semantic_mask"] = {
        "path": roster.named_png(edit["mask_edit_path"], mask_dir, f"{identity}_{tag}_edit.png"), "sha256": None,
    }
    return fields | {"recipient": identity, "size": list(size), "publisher_split": split, "photo_attribution": credit,
                     "flickr_photo_id": photo, "flickr_owner": owner, "editor_record": model_key}


def planned(args):
    prior, _, _ = previous(args)
    metadata = args.archive / "image-bg-real-metadata-v218"
    for name, (_, size, digest) in roster.METADATA.items():
        path = metadata / name
        require(path.stat().st_size == size and roster.sha(path) == digest, "Pinned publisher metadata changed")
    old_keys, known, old_bindings, old_recipients, old_names = exposure(args, metadata, prior)
    sources = roster.table(metadata / "manifest_v2.csv")
    source_index = roster.indexed(sources, "sample_id")
    recipient_index = roster.indexed(roster.table(metadata / "recipients.csv"), "source_image_id")
    attribution = roster.indexed((row for row in roster.table(metadata / "attribution.csv") if row["role"] == "recipient"), "bgreal_id")
    models = {row["key"]: row for row in roster.table(metadata / "model_revisions.csv")}
    memberships = {key: set(rows) for key, rows in yaml.safe_load((metadata / "splits_v2.yaml").read_bytes()).items()}
    config = bounded.GENERATORS["kandinsky"]
    generators = {"kandinsky": models[config[3]], "sd15": models["sd15_inpaint"]}
    require(generators["kandinsky"]["revision"] == config[4] and generators["kandinsky"]["license"] == config[5],
            "Pinned background generator record differs")
    require(generators["sd15"] == {"key": "sd15_inpaint", "repo": insertions.GENERATOR, "revision": insertions.GENERATOR_REVISION,
                                  "license": "creativeml-openrail-m"}, "Pinned insertion generator record differs")
    choices, eligible_counts, rejections = [], {}, Counter()
    keys, names = old_keys.copy(), set(old_names)
    for family in FAMILIES:
        candidates = []
        model = generators[family]
        family_key = "F5" if family == "sd15" else bounded.GENERATORS[family][0]
        for row in sources:
            identity = row["source_image_id"]
            if (row["family"] != family_key or not roster.base_png(row) or row["generator"] != model["repo"]
                    or identity in old_recipients):
                continue
            try:
                if family == "sd15":
                    if (row["split"] not in roster.TEST_SPLITS or row["edit_kind"] != "object_insertion"
                            or not .004 <= float(row["edit_area"]) <= .08):
                        continue
                    item = roster.pair(row, roster.one(source_index, f"{identity}_A0", "Exact original required"),
                                       roster.one(attribution, identity, "Exact licence required"),
                                       roster.one(recipient_index, identity, "Exact recipient required"), memberships)
                    item["processed"] = latest.matched_processing(source_index, item, memberships)
                else:
                    item = full_background(row, source_index, attribution, memberships, family)
                if item["photo_attribution"]["Author"].strip().casefold() not in names:
                    candidates.append(item)
            except ValueError as exc:
                rejections[f"{family}/{exc}"] += 1
        unique, excluded = roster.select(candidates, keys)
        rejections[f"{family}/prior_or_shared_identity"] += len(excluded)
        unique.sort(key=lambda item: sha(f"{SCHEMA}/{family}/{item['recipient']}".encode()))
        eligible_counts[family] = len(unique)
        count = 0
        for item in unique:
            name = item["photo_attribution"]["Author"].strip().casefold()
            if name in names:
                continue
            names.add(name)
            keys.update(roster.source_keys(item))
            choices.append(item | {"generator": family, "slot": len(choices)})
            count += 1
            if count == ATTEMPTS[family]:
                break
        require(count == ATTEMPTS[family], f"Insufficient fresh metadata for {family}")
    code = (Path(__file__), Path(probe.__file__), Path(transport.__file__), Path(bounded.__file__),
            Path(probe.base.__file__), Path(probe.base.comparison.jpeg_dq.__file__))
    return {
        "schema": SCHEMA, "design": DESIGN, "publisher_revision": roster.REVISION, "models": generators,
        "prior_probe": PREVIOUS, "prior_exposure": old_bindings, "eligible_candidates": eligible_counts,
        "metadata_rejections": dict(rejections), "cases": choices, "code_sha256": {p.name: roster.sha(p) for p in code},
    }, known


def bind(item, bodies):
    original, edited, processed, mask = (pixels.png_pixels(bodies[key], item["size"]) for key in ASSETS)
    native = {"original": original, "edited": edited, "processed": processed}
    if item["generator"] == "sd15":
        truth, _ = pixels.edit_truth(original, edited, mask, item["semantic_edit_area"])
        crop = None
        require(.004 <= float(truth.mean()) <= .08, "Exact insertion outside frozen bounds")
    else:
        support = mask[..., 0] > 0
        require(np.array_equal(mask[..., 0], mask[..., 1]) and np.array_equal(mask[..., 0], mask[..., 2])
                and np.isin(mask, (0, 255)).all(), "Semantic mask is nonbinary")
        crop = bounded.region(support, item["recipient"])
        composed, truth = bounded.compose(native, crop)
        edited = composed["edited"]
    local = np.where(truth[..., None], processed, original)
    require(np.array_equal(truth, np.any(edited != original, axis=2)), "Edit pixels differ from footprint")
    fingerprints = {role: pixels.fingerprint(bodies[role], value) for role, value in native.items()}
    return {"original": original, "edited": edited, "global_processing": processed, "local_processing": local}, truth, crop, fingerprints


def acquire(output, plan, known):
    directory = output / "assets"
    directory.mkdir(mode=0o700)
    qualified, rows = Counter(), []
    deadline = time.monotonic() + 320
    with requests.Session() as session, (output / "intake.jsonl").open("x") as journal:
        session.trust_env = False
        for item in plan["cases"]:
            slot, family = item["slot"], item["generator"]
            result = {"slot": slot, "family": family, "receipts": {}}
            if qualified[family] == CASES_PER_FAMILY:
                result["status"] = "reserve_unopened"
            else:
                try:
                    bodies = {}
                    for role in ASSETS:
                        ref = item[role]
                        body, receipt = transport.download(session, ref["path"], ref["sha256"], deadline)
                        bodies[role] = body
                        result["receipts"][role] = receipt
                    _, truth, crop, fingerprints = bind(item, bodies)
                    duplicates = pixels.prior_matches(item, fingerprints, set(), known)
                    require(not duplicates, ",".join(duplicates))
                    for role, body in bodies.items():
                        (directory / f"{slot:02d}-{role}.png").write_bytes(body)
                    for info in fingerprints.values():
                        known[0].append(info)
                        known[1].add(info["encoded_sha256"])
                        known[2].add(info["rgb_sha256"])
                    result.update(status="qualified", edit_box=crop, exact_sha256=sha(truth.tobytes()),
                                  exact_pixels=int(truth.sum()), image_pixels=int(truth.size), fingerprints=fingerprints)
                    qualified[family] += 1
                except (ValueError, requests.RequestException) as exc:
                    result.update(status="excluded", reason=str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
            journal.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            journal.flush()
            rows.append(result)
            print(slot, family, result["status"], result.get("reason", ""), flush=True)
    return {"qualified": dict(qualified), "status": dict(Counter(row["status"] for row in rows)),
            "plan_sha256": roster.sha(output / "plan.json"), "journal_sha256": roster.sha(output / "intake.jsonl")}


def groups(output, plan, known):
    intake = read(output / "intake.json")
    journal = output / "intake.jsonl"
    require(intake["plan_sha256"] == roster.sha(output / "plan.json") and intake["journal_sha256"] == roster.sha(journal),
            "Source acquisition differs from its commitments")
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    require(intake["qualified"] == dict(Counter(row["family"] for row in records if row["status"] == "qualified")),
            "Source acquisition count differs")
    for item, row in zip(plan["cases"], records, strict=True):
        slot = item["slot"]
        require(slot == row["slot"] and item["generator"] == row["family"], "Source identity differs")
        if row["status"] != "qualified":
            continue
        bodies = {}
        for role in ASSETS:
            body = (output / "assets" / f"{slot:02d}-{role}.png").read_bytes()
            proof = row["receipts"][role]
            require(proof["resolver_revision"] == roster.REVISION and len(body) == proof["bytes"]
                    and sha(body) == proof["sha256"] == (item[role]["sha256"] or proof["publisher_asset_commitment"]),
                    "Publisher byte proof differs")
            bodies[role] = body
        native, truth, crop, fingerprints = bind(item, bodies)
        require(not pixels.prior_matches(item, fingerprints, set(), known), "Source overlaps recorded exposure")
        for info in fingerprints.values():
            known[0].append(info)
            known[1].add(info["encoded_sha256"])
            known[2].add(info["rgb_sha256"])
        require(row["edit_box"] == (list(crop) if crop else None) and row["exact_sha256"] == sha(truth.tobytes())
                and row["fingerprints"] == fingerprints and row["exact_pixels"] == int(truth.sum()),
                "Native region or source changed")
        yield item, native, truth


def views(item, native, truth):
    controls, edits = probe.base.authored(native, truth, item["recipient"])
    local = {"original": native["original"], "local_processing": native["local_processing"],
             "global_processing": native["global_processing"], **{role: controls[role] for role in CONTROLS[3:]}, **edits}
    feather = np.any(edits["feathered"] != native["original"], axis=2)
    sites = probe.patch_share(feather) >= .50
    for role, rgb in local.items():
        body = probe.base.encode(rgb, probe.QUALITY)
        exact = np.any(rgb != native["original"], axis=2) if role in probe.HELD_POSITIVE else np.zeros(truth.shape, bool)
        yield {
            "source": item["recipient"], "slot": item["slot"], "family": item["generator"], "role": role,
            "input_sha256": sha(body), "truth_sha256": sha(exact.tobytes()), "exact_pixels": int(exact.sum()),
        }, body, exact, sites


def locate(fields, body, rgb, truth, points):
    arms = {name: probe.proposal(field) for name, field in fields.items()}
    current = probe.base.comparison.format_reference(f"jpeg{probe.QUALITY}", body, rgb)
    arms["current_luma"] = {"box": current["box"] if current and current["selected"] else None, "score": None}
    for name, row in arms.items():
        row["flagged"] = bool(row["box"]) and (name == "current_luma" or row["score"] > points[name])
        row["overlap"] = probe.overlap(truth, row["box"])
        row["useful"] = probe.base.geometry.useful(row["overlap"])
    return arms


def summarize(rows):
    result = probe.summarize(rows)
    sources = {}
    for row in rows:
        sources.setdefault(row["source"], {})[row["role"]] = row
    results = {}
    for family in FAMILIES:
        matching = [group for group in sources.values() if group["original"]["family"] == family]
        for name in probe.HEADS:
            support = [group for group in matching if group["original"]["exact_token_medians"][name] is not None]
            for control in ("original", "local_processing", "local_blur", "local_copy", "local_tone"):
                changes = [
                    g["feathered"]["exact_token_medians"][name] - g[control]["exact_token_medians"][name] for g in support
                ]
                results[f"{family}/{name}/{control}"] = {
                    "groups": len(changes), "feathered_higher": sum(value > 0 for value in changes),
                    "median_paired_gap": float(np.median(changes)) if changes else None,
                }
    same_box = {name: sum(g["feathered"]["arms"][name]["box"] == g["original"]["arms"][name]["box"] for g in sources.values())
                for name in probe.HEADS}
    return result | {"known_site_diagnostic": results, "same_feathered_box_as_original": same_box}


def run(args, plan, known, heads, points):
    start = time.monotonic()
    prepared = read(args.output / "intake.json")
    require(prepared["qualified"] == {family: CASES_PER_FAMILY for family in FAMILIES}, "Whole frozen panel required")
    read_features, forwards = probe.extractor(args.checkpoint)
    rows = []
    for item, native, truth in groups(args.output, plan, known):
        for row, body, exact, sites in views(item, native, truth):
            features, rgb = read_features(body)
            fields = {name: probe.score_map(features, weights) for name, weights in heads.items()}
            arms = locate(fields, body, rgb, exact, points)
            medians = {name: float(np.median(field[sites])) if sites.any() else None for name, field in fields.items()}
            rows.append(row | {"arms": arms, "exact_token_medians": medians, "diagnostic_token_count": int(sites.sum())})
        print("scored group", item["slot"], item["generator"], flush=True)
    write(args.output / "measurements.json", rows)
    return summarize(rows) | {
        "schema": SCHEMA, "design": DESIGN, "points": points, "fit_count": 0, "calibration_count": 0,
        "deliveries": len(rows), "forwards": dict(forwards), "intake": prepared, "prior_probe": PREVIOUS,
        "publisher_revision": roster.REVISION, "checkpoint": probe.REV,
        "elapsed_s": round(time.monotonic() - start, 2), "process_peak_rss_bytes_mac": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "measurements_sha256": roster.sha(args.output / "measurements.json"),
    }


def review(args, plan, known, points):
    result = read(args.output / "results.json")
    path = args.output / "measurements.json"
    require(roster.sha(path) == result["measurements_sha256"] and points == result["points"], "Frozen scores differ")
    rows = read(path)
    rebinding = (view for item, native, truth in groups(args.output, plan, known) for view in views(item, native, truth))
    checks, reference = 0, 0
    for row, (identity, body, exact, sites) in zip(rows, rebinding, strict=True):
        require(all(row[key] == value for key, value in identity.items()) and int(sites.sum()) == row["diagnostic_token_count"],
                "Exact JPEG, edit or native diagnostic sites changed")
        for name, data in row["arms"].items():
            if name == "current_luma":
                measured = probe.base.comparison.format_reference(f"jpeg{probe.QUALITY}", body, probe.base.decode(body))
                expected = measured["box"] if measured and measured["selected"] else None
                require(expected == data["box"], "Current luma rectangle changed")
                reference += 1
            reported = probe.overlap(exact, data["box"])
            require(reported == data["overlap"] and probe.base.geometry.useful(reported) == data["useful"]
                    and data["flagged"] == (bool(data["box"]) and (name == "current_luma" or data["score"] > points[name])),
                    "Exact bounds, overlap or flag differs")
            checks += 1
    require(all(result[key] == value for key, value in summarize(rows).items()), "Held totals differ")
    return result | {"audit": {"exact_delivery_truth_and_diagnostic_grid": len(rows), "native_boxes_and_flags": checks,
                               "current_luma_repeat": reference, "neural_inference_repeated": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "acquire", "run", "review"))
    for name in ("archive", "prior-probe", "residual", "early-f5", "later-f5", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    plan, known = planned(args)
    if args.stage == "plan":
        args.output.mkdir(mode=0o700)
        write(args.output / "plan.json", plan)
        print(plan["eligible_candidates"], plan["metadata_rejections"], flush=True)
        return
    require(plan == read(args.output / "plan.json"), "Private source and scoring freeze differs")
    if args.stage == "acquire":
        write(args.output / "intake.json", acquire(args.output, plan, known))
        return
    _, heads, points = previous(args)
    if args.stage == "run":
        write(args.output / "results.json", run(args, plan, known, heads, points))
        return
    require(args.publication is not None, "Supply an aggregate publication path")
    write(args.publication, review(args, plan, known, points))


if __name__ == "__main__":
    main()
