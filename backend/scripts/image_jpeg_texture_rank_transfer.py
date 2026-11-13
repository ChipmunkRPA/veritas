# SPDX-License-Identifier: Apache-2.0
"""Test within-image texture-matched ranking of a frozen residual encoder."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import requests
import torch
import yaml
from scipy.spatial.distance import cdist

from scripts import image_bg_real_acquisition_v221 as pixels
from scripts import image_bg_real_insertion_transfer as intake
from scripts import image_bg_real_roster_v218 as roster
from scripts import image_bg_real_transport_v225 as transport
from scripts import image_residual_encoder_transfer as old

SCHEMA = "chipmunk.image-jpeg-texture-rank-transfer/1"
FIT_FREEZE = "7ecd5f065260606e193a837513022ae79f9337740cfe0b8a9a7e968b56f86417"
FIT_WEIGHTS = "aa63b8a0a3123fb175e57341947d8392858498646aa92b4d46ab12d844396dc6"
MODELS = {"lama": ("big-lama", "object_removal"), "sd15": (intake.GENERATOR, "object_insertion")}
TARGETS = {"lama": 8, "sd15": 8}
ARMS = ("frozen", "texture_matched", "current_luma")
REMOTE, MATCHES, MINIMUM = 8, 12, 8
MAX_DISTANCE, SPREAD_FLOOR = 1.6, .18
SCALE_FLOORS = np.array([8., .28, .28, .18, .035])
DESIGN = {
    "geometry": old.DESIGN["proposal"], "quality": list(old.QUALITIES), "source_targets": TARGETS,
    "source_range": [.004, .08], "jpeg_chroma": "4:2:0", "texture": ["coarse luma mean", "log coarse luma spread",
        "log coarse gradient energy", "horizontal/vertical gradient balance", "saturated sample share"],
    "texture_scale_floors": SCALE_FLOORS.tolist(), "texture_window_px": 35, "coarse_gaussian_sigma": 2.5,
    "minimum_chebyshev_distance_cells": REMOTE, "nearest_references": MATCHES, "minimum_references": MINIMUM,
    "maximum_texture_distance": MAX_DISTANCE, "residual_scale_floor": SPREAD_FLOOR,
    "score": "Subtract median and divide by scaled MAD of distant cells matched for coarse texture. Rank compact rectangles with the unchanged baseline geometry.",
    "valid_rectangle": "All cells in candidate and ring have enough matched support.",
    "point": "Second-largest ordinary per-source maximum in the original 12 calibration sources, separate by quality and ranking arm.",
    "advance": "At least three additional useful feathered source groups across both families, at most two additional ordinary groups and four wrong edit locations, versus current luma; separately report ranking delta versus frozen encoder.",
    "confidence_weight": 0,
}


def exposure(args, metadata, profile):
    keys, known, commits = intake.exposure()
    prior_f5 = old.read(args.prior_f5 / "plan.json")
    prior_objects = old.read(args.prior_f5 / "acquired.json")
    commits["prior_f5"] = roster.sha(args.prior_f5 / "plan.json")
    commits["prior_f5_objects"] = roster.sha(args.prior_f5 / "acquired.json")
    old.require(prior_objects["plan_sha256"] == commits["prior_f5"] and len(prior_f5["rows"]) == 12,
                "Prior F5 exposure changed")
    by_source = {row["bgreal_id"]: row for row in roster.table(metadata / "attribution.csv") if row["role"] == "recipient"}
    names, recipients = set(), set()
    for row, assets in zip(prior_f5["rows"], prior_objects["rows"], strict=True):
        source = row["source_id"]
        credit = by_source[source]
        landing_owner, photo = roster.landing_identity(credit["OriginalLandingURL"])
        profile_owner = roster.canonical_owner(credit["AuthorProfileURL"].rstrip("/").rsplit("/", 1)[1])
        old.require(roster.static_photo_id(credit["OriginalURL"]) == photo, "Prior F5 photo identity differs")
        keys.update({("owner", landing_owner), ("owner", profile_owner), ("photo", photo),
                     ("original_sha256", row["source"]["sha256"])})
        names.add(credit["Author"].casefold())
        recipients.add(source)
        for key in ("source", "edited"):
            receipt = assets[key]
            body = (args.prior_f5 / "objects" / receipt["sha256"]).read_bytes()
            old.require(old.sha(body) == receipt["sha256"] == row[key]["sha256"], "Prior F5 asset changed")
            known[0].append(pixels.fingerprint(body, pixels.png_pixels(body, row["shape"])))
    for rows in profile["sources"].values():
        for row in rows:
            keys.update({("owner", row["author"]), ("photo", row["photo"]), ("original_sha256", row["original_sha256"])})
            names.add(row["attribution"]["Author"].casefold())
            recipients.add(row["recipient"])
            known[0].append({"size": row["size"], "sketch": row["sketch"]})
            known[1].add(row["original_sha256"])
            known[2].update(row["native"].values())
    folder = args.archive / "image-bg-real-remote-context-2026-11-11" / "trial"
    _, plan, attempts = old.insertion.checked_inputs(folder)
    commits["remote_context"] = roster.sha(folder / "plan.json")
    commits["remote_context_journal"] = roster.sha(folder / "attempts.jsonl")
    for row, attempt in zip(plan["cases"], attempts, strict=True):
        keys.update(roster.source_keys(row))
        recipients.add(row["recipient"])
        names.add(row["photo_attribution"]["Author"].casefold())
        for info in attempt.get("fingerprints", {}).values():
            known[0].append(info)
            known[1].add(info["encoded_sha256"])
            known[2].add(info["rgb_sha256"])
    return keys, known, commits, recipients, names


def matched_processing(controls, item, memberships):
    source, split = item["recipient"], item["publisher_split"]
    row = roster.one(controls, f"{source}_C2", "Exact publisher processing control required")
    old.require(roster.base_png(row) and row["source_image_id"] == source and row["family"] == "C2" and
                row["label"] == "0" and row["generator"] == "sd15_autoencoder" and row["split"] == split and
                float(row["edit_area"]) == 0. and "F5" in row["control_of"].split(",") and
                row["sample_id"] in memberships[split] and roster.dimensions(row) == tuple(item["size"]) and
                roster.SHA.fullmatch(row["sha256"]), "Publisher processing control differs")
    path = roster.named_png(row["image_path"], PurePosixPath("v2/images") / source[3], f"{source}_C2.png")
    return {"path": path, "sha256": row["sha256"]}


def source_plan(args):
    metadata = args.archive / "image-bg-real-metadata-v218"
    for name, (_, size, digest) in roster.METADATA.items():
        path = metadata / name
        old.require(path.stat().st_size == size and roster.sha(path) == digest, "Pinned publisher metadata changed")
    fit = args.encoder / "freeze.json"
    old.require(roster.sha(fit) == FIT_FREEZE and roster.sha(args.encoder / "encoder.pt") == FIT_WEIGHTS,
                "Frozen encoder release differs")
    keys, known, commits, seen, names = exposure(args, metadata, old.read(fit))
    models = roster.indexed(roster.table(metadata / "model_revisions.csv"), "key")
    sd15 = roster.one(models, "sd15_inpaint", "Expected exact SD1.5 record")
    old.require(roster.one(models, "lama", "Expected exact LaMa record") == roster.MODEL and
                sd15 == {"key": "sd15_inpaint", "repo": intake.GENERATOR, "revision": intake.GENERATOR_REVISION,
                         "license": "creativeml-openrail-m"}, "Pinned edit generator differs")
    memberships = {key: set(rows) for key, rows in yaml.safe_load((metadata / "splits_v2.yaml").read_bytes()).items()}
    raw = roster.table(metadata / "manifest_v2.csv")
    original = roster.indexed(raw, "sample_id")
    recipients = roster.indexed(roster.table(metadata / "recipients.csv"), "source_image_id")
    licenses = roster.indexed((row for row in roster.table(metadata / "attribution.csv") if row["role"] == "recipient"), "bgreal_id")
    choices, counts, rejects = [], {}, Counter()
    for family, (model, kind) in MODELS.items():
        options = []
        for row in raw:
            source = row["source_image_id"]
            if (source in seen or row["family"] != "F5" or not roster.base_png(row) or row["generator"] != model or
                    row["edit_kind"] != kind or not .004 <= float(row["edit_area"]) <= .08):
                continue
            try:
                credit = roster.one(licenses, source, "Single photo attribution required")
                item = roster.pair(row, roster.one(original, f"{source}_A0", "Single original required"),
                                   credit, roster.one(recipients, source, "Single recipient required"), memberships)
                if family == "sd15":
                    item["processed"] = matched_processing(original, item, memberships)
                options.append(item)
            except ValueError as exc:
                rejects[f"{family}/{exc}"] += 1
        eligible, _ = roster.select(options, keys)
        eligible = [item for item in eligible if item["photo_attribution"]["Author"].casefold() not in names]
        eligible.sort(key=lambda item: (item["publisher_split"] not in roster.TEST_SPLITS,
                                        old.sha(f"{SCHEMA}/{family}/{item['recipient']}".encode())))
        counts[family] = len(eligible)
        selected = 0
        for item in eligible:
            name = item["photo_attribution"]["Author"].casefold()
            if name in names:
                continue
            keys.update(roster.source_keys(item))
            names.add(name)
            choices.append(item | {"slot": len(choices), "generator": family})
            selected += 1
            if selected == TARGETS[family]:
                break
    result = {"schema": SCHEMA, "design": DESIGN, "code_sha256": roster.sha(Path(__file__)),
              "roster_counts": counts, "metadata_rejections": rejects, "dataset_revision": roster.REVISION,
              "encoder_freeze_sha256": FIT_FREEZE, "encoder_sha256": FIT_WEIGHTS, "exposure_sha256": commits,
              "cases": choices}
    return result, known


def bind(case, bodies, receipts):
    for role, body in bodies.items():
        proof = receipts[role]
        old.require(proof["resolver_revision"] == roster.REVISION and old.sha(body) == proof["sha256"] and
                    len(body) == proof["bytes"] and proof["sha256"] == (case[role]["sha256"] or proof["publisher_asset_commitment"]),
                    "Publisher asset does not reproduce its commitment")
    native = {role: pixels.png_pixels(body, case["size"]) for role, body in bodies.items()}
    truth, _ = pixels.edit_truth(native["original"], native["edited"], native.pop("semantic_mask"), case["semantic_edit_area"])
    old.require(.004 <= truth.mean() <= .08, "Local exact edit is outside the frozen size bound")
    fingerprints = {role: pixels.fingerprint(bodies[role], rgb) for role, rgb in native.items()}
    if "processed" in native:
        native["global_processing"] = native.pop("processed")
    return native, truth, fingerprints


def acquire(args, plan, known):
    destination = args.output
    old.require(plan == old.read(destination / "plan.json"), "Source or code freeze changed")
    assets = destination / "assets"
    assets.mkdir(mode=0o700)
    deadline = time.monotonic() + 240
    rows = []
    with requests.Session() as session:
        session.trust_env = False
        for case in plan["cases"]:
            row = {"slot": case["slot"], "family": case["generator"], "receipts": {}}
            try:
                bodies = {}
                roles = ("original", "edited", "semantic_mask") + (("processed",) if "processed" in case else ())
                for role in roles:
                    ref = case[role]
                    body, proof = transport.download(session, ref["path"], ref["sha256"], deadline)
                    bodies[role] = body
                    row["receipts"][role] = proof
                _, truth, fingerprints = bind(case, bodies, row["receipts"])
                matches = pixels.prior_matches(case, fingerprints, set(), known)
                pixels.check(not matches, ",".join(matches))
                for role, body in bodies.items():
                    (assets / f"{case['slot']:02d}-{role}.png").write_bytes(body)
                for info in fingerprints.values():
                    known[0].append(info)
                    known[1].add(info["encoded_sha256"])
                    known[2].add(info["rgb_sha256"])
                row.update(status="qualified", exact_pixels=int(truth.sum()), exact_sha256=old.sha(truth.tobytes()))
            except (pixels.Rejected, requests.RequestException, ValueError) as exc:
                row.update(status="excluded", reason=str(exc) if isinstance(exc, (ValueError, pixels.Rejected)) else type(exc).__name__)
            rows.append(row)
            print("input", case["slot"], case["generator"], row["status"], row.get("reason", ""), flush=True)
    old.write(destination / "intake.json", {"schema": SCHEMA, "plan_sha256": roster.sha(destination / "plan.json"), "rows": rows})


def texture(rgb, field_shape):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    coarse = cv2.GaussianBlur(gray, (0, 0), 2.5)
    ksize = (old.RECEPTIVE, old.RECEPTIVE)
    centers = tuple(slice(old.CENTER, old.CENTER + old.STRIDE * size, old.STRIDE) for size in field_shape)

    def means(field):
        return cv2.boxFilter(field, -1, ksize)[centers]

    light = means(coarse)
    spread = np.sqrt(np.maximum(0, means(coarse**2) - light**2))
    dy, dx = np.abs(np.gradient(coarse))
    h, v = means(dx), means(dy)
    sat = means(((gray < 8) | (gray > 247)).astype(np.float32))
    return np.stack((light, np.log(spread + .6), np.log(h + v + .2), (h - v) / (h + v + .2), sat), axis=-1)


def matched_field(rgb, field):
    context = texture(rgb, field.shape).reshape(-1, len(SCALE_FLOORS))
    scale = np.maximum(np.diff(np.percentile(context, (25, 75), axis=0), axis=0)[0], SCALE_FLOORS)
    rowcol = np.column_stack(np.unravel_index(np.arange(field.size), field.shape))
    logits = field.ravel()
    normalized = context / scale
    output = np.full(field.size, np.nan)
    for start in range(0, field.size, 256):
        selected = slice(start, start + 256)
        distance = cdist(normalized[selected], normalized)
        gap = cdist(rowcol[selected], rowcol, "chebyshev")
        distance[(gap <= REMOTE) | (distance > MAX_DISTANCE)] = np.inf
        neighbors = np.argpartition(distance, MATCHES - 1, axis=1)[:, :MATCHES]
        supported = np.isfinite(np.take_along_axis(distance, neighbors, axis=1))
        accepted = np.flatnonzero(supported.sum(axis=1) >= MINIMUM)
        for local in accepted:
            refs = logits[neighbors[local, supported[local]]]
            median = np.median(refs)
            scale = max(SPREAD_FLOOR, 1.4826 * np.median(np.abs(refs - median)))
            output[start + local] = np.clip((logits[start + local] - median) / scale, -8., 8.)
    return output.reshape(field.shape)


def matched_proposal(field, shape):
    rows, cols = field.shape
    valid = np.isfinite(field)
    if not valid.any():
        return None
    ii = np.pad(np.nan_to_num(field).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    count = np.pad(valid.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    median = np.median(field[valid])
    best = None
    for h in old.SIDES:
        for w in old.SIDES:
            area = h * w
            if max(h, w) > min(h, w) * 2 or area * old.STRIDE**2 > .10 * shape[0] * shape[1] or h > rows or w > cols:
                continue
            x, y = np.meshgrid(np.arange(cols - w + 1), np.arange(rows - h + 1))
            margin = max(2, min(h, w) // 4)
            l, t, r, b = np.maximum(x - margin, 0), np.maximum(y - margin, 0), np.minimum(x + w + margin, cols), np.minimum(y + h + margin, rows)
            core = old.sums(ii, x, y, x + w, y + h)
            ring_count = (r - l) * (b - t) - area
            ring = (old.sums(ii, l, t, r, b) - core) / np.maximum(1, ring_count)
            supported = (old.sums(count, l, t, r, b) == (r - l) * (b - t)) & (ring_count >= area * .2)
            scores = np.where(supported, (core / area - np.maximum(median, ring)) * area**.10, -np.inf)
            idx = np.unravel_index(scores.argmax(), scores.shape)
            score = float(scores[idx])
            if np.isfinite(score) and (best is None or score > best["score"]):
                left, upper = (int(axis[idx]) * old.STRIDE + old.CENTER - old.STRIDE // 2 for axis in (x, y))
                best = {"box": [left, upper, left + w * old.STRIDE, upper + h * old.STRIDE], "score": score}
    return best


def predict(rgb, model):
    with torch.inference_mode():
        return model(torch.from_numpy(rgb).permute(2, 0, 1)[None])[0].numpy()


def assess(inputs, model, points=None):
    rows = []
    for item, native, exact in inputs:
        ordinary, generated = old.authored(native, exact, item["recipient"])
        for role, rgb in (ordinary | generated).items():
            truth = np.any(native["original"] != rgb, axis=2) if role in old.POSITIVE else np.zeros(rgb.shape[:2], bool)
            for quality in old.QUALITIES:
                body = old.encode(rgb, quality)
                delivered = old.decode(body)
                field = predict(delivered, model)
                matched = matched_field(delivered, field)
                choices = {"frozen": old.proposal(field, truth.shape), "texture_matched": matched_proposal(matched, truth.shape)}
                if points is not None:
                    current = old.comparison.format_reference(f"jpeg{quality}", body, delivered)
                    choices["current_luma"] = current if current and current["selected"] else None
                result = {}
                for arm, proposal in choices.items():
                    box = proposal["box"] if proposal else None
                    score = proposal.get("score") if proposal else None
                    area = old.insertion.measure(old.bounded.boxed(box, truth.shape, box is not None), truth)
                    hit = {"box": box, "score": score, "useful": old.geometry.useful(area), "overlap": area}
                    if points is not None:
                        point = points[str(quality)].get(arm)
                        hit["flagged"] = bool(box) and (arm == "current_luma" or (point is not None and score > point))
                    result[arm] = hit
                rows.append({"source": item["recipient"], "generator": item["generator"], "role": role, "quality": quality,
                             "result": result, "input_sha256": old.sha(body), "truth_sha256": old.sha(truth.tobytes()),
                             "valid_cells": int(np.isfinite(matched).sum()), "cells": int(matched.size)})
        print("measurement", len(rows), item["generator"], flush=True)
    return rows


def held_inputs(folder, plan):
    intake = old.read(folder / "intake.json")
    old.require(intake["plan_sha256"] == roster.sha(folder / "plan.json"), "Acquired panel differs from freeze")
    for case, row in zip(plan["cases"], intake["rows"], strict=True):
        old.require(case["slot"] == row["slot"] and case["generator"] == row["family"], "Held acquisition order changed")
        if row["status"] == "qualified":
            assets = {role: (folder / "assets" / f"{case['slot']:02d}-{role}.png").read_bytes() for role in row["receipts"]}
            native, exact, _ = bind(case, assets, row["receipts"])
            old.require(int(exact.sum()) == row["exact_pixels"] and old.sha(exact.tobytes()) == row["exact_sha256"], "Held exact mask differs")
            yield case, native, exact


def thresholds(rows):
    groups = defaultdict(dict)
    for row in rows:
        for arm, data in row["result"].items():
            key, source = (str(row["quality"]), arm), row["source"]
            group = groups[key]
            score = data["score"] if data["score"] is not None else -float("inf")
            group[source] = max(group.get(source, -float("inf")), score)
    points = defaultdict(dict)
    for (quality, arm), maxima in groups.items():
        old.require(len(maxima) == 12, "Calibration source count differs")
        value = sorted(maxima.values())[-2]
        points[quality][arm] = value if np.isfinite(value) else None
    return points


def summarize(rows):
    sections = defaultdict(Counter)
    overall = {}
    for quality in old.QUALITIES:
        ordinary = defaultdict(set)
        extra = set()
        wrong = 0
        for row in (r for r in rows if r["quality"] == quality):
            family, role, source = (row[key] for key in ("generator", "role", "source"))
            for arm, result in row["result"].items():
                control = role not in old.POSITIVE
                table = sections[f"{quality}/{family}/{role}/{arm}"]
                table.update(views=1, ranked=result["box"] is not None, useful_before_point=result["useful"],
                             flagged=result["flagged"], useful=result["flagged"] and result["useful"])
                if control and result["flagged"]:
                    ordinary[arm].add(source)
                if arm == "texture_matched" and not control and result["flagged"]:
                    wrong += not result["useful"]
                    if role == "feathered" and result["useful"] and not row["result"]["current_luma"]["useful"]:
                        extra.add((source, family))
        outcome = {"additional_feathered_groups": len(extra), "additional_families": len({f for _, f in extra}),
                   "additional_ordinary_groups": len(ordinary["texture_matched"] - ordinary["current_luma"]),
                   "wrong_edit_views": wrong}
        outcome["passes"] = (outcome["additional_feathered_groups"] >= 3 and outcome["additional_families"] == 2 and
                             outcome["additional_ordinary_groups"] <= 2 and wrong <= 4)
        overall[str(quality)] = {"ordinary_groups": {arm: len(ordinary[arm]) for arm in ARMS}, "outcome": outcome}
    return {"by_quality": overall, "strata": sections}


def run(args, plan):
    start = time.monotonic()
    old.require(plan == old.read(args.output / "plan.json"), "Source or code freeze changed")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    model = old.Encoder().eval()
    model.load_state_dict(torch.load(args.encoder / "encoder.pt", weights_only=True))
    args.rosters = roster.DATA
    calibration = assess(old.frozen_inputs(args, "calibration", old.read(args.encoder / "freeze.json")), model)
    old.write(args.output / "calibration.json", calibration)
    points = thresholds(calibration)
    old.write(args.output / "points.json", points)
    old.require(dict(points) and all(points[q]["frozen"] == v for q, v in old.read(args.encoder / "points.json")["points"].items()),
                "Frozen comparator point differs")
    held = assess(held_inputs(args.output, plan), model, points)
    old.write(args.output / "held.json", held)
    summary = {"schema": SCHEMA, "design": DESIGN, "eligible": plan["roster_counts"], "points": points,
               "qualified": Counter(r["family"] for r in old.read(args.output / "intake.json")["rows"] if r["status"] == "qualified"),
               "commitments": {name: roster.sha(args.output / name) for name in ("plan.json", "intake.json", "points.json", "held.json", "calibration.json")},
               "code_sha256": plan["code_sha256"], "encoder_sha256": FIT_WEIGHTS,
               "elapsed_s": round(time.monotonic() - start, 2), "rows": len(held), "calibration_rows": len(calibration), **summarize(held)}
    old.write(args.output / "result.json", summary)
    print(summary["by_quality"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "acquire", "run"))
    for key in ("archive", "encoder", "prior-f5", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    args = parser.parse_args()
    plan, known = source_plan(args)
    if args.stage == "plan":
        args.output.mkdir(mode=0o700)
        old.write(args.output / "plan.json", plan)
        print(json.dumps({"eligible": plan["roster_counts"], "planned": Counter(r["generator"] for r in plan["cases"]),
                          "rejected": plan["metadata_rejections"], "sha256": roster.sha(args.output / "plan.json")}), flush=True)
    elif args.stage == "acquire":
        acquire(args, plan, known)
    else:
        run(args, plan)


if __name__ == "__main__":
    main()
