# SPDX-License-Identifier: Apache-2.0
"""Compare three- and two-class heads using matched inpainting controls."""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.special import logsumexp

from scripts import image_background_patch_score as bounded
from scripts import image_bg_real_acquisition_v221 as source
from scripts import image_bg_real_insertion_score as insertions
from scripts import image_bg_real_roster_v218 as roster
from scripts import image_jpeg_texture_rank_transfer as recent
from scripts import image_residual_encoder_transfer as base
from scripts.review_image_residual_encoder_transfer import overlap

SCHEMA = "chipmunk.image-three-class-patch-probe/1"
REV = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
LAYERS = (3, 7)
PATCH, TILES, BATCH = 14, 224, 4
SEED, SITES, QUALITY, RIDGE = 418, 12, 75, .03
HEADS = ("binary", "three_class")
FIT_LABELS = ("authentic", "processing", "inpainted")
HELD_POSITIVE = ("sharp", "feathered")
COHORTS = (
    ("fit", "bounded_fit", "image-background-patch-transfer-2026-11-11/intake", "bounded"),
    ("fit", "sd15_fit", "image-bg-real-small-insertions-2026-11-11/trial", "insertion"),
    ("calibration", "sd15_cal", "image-bg-real-remote-context-2026-11-11/trial", "insertion"),
    ("held", "bounded_held", "image-wavelet-region-transfer-2026-11-11/holdout", "bounded"),
    ("held", "sd15_held", None, "later"),
)
DESIGN = {
    "delivery": "standard 4:2:0 JPEG75, same output recipe for every case",
    "features": {"layers_one_based": list(LAYERS), "patch_px": PATCH, "tile_px": TILES, "native": True,
                 "conditioning": "Unit-normalize each hidden patch token before concatenation. Fixed reflect padding at bottom and right."},
    "fit": {"source_roles": [list(row) for row in COHORTS], "sites_per_source": SITES, "minimum_edit_occupancy": .75,
            "minimum_matched_processing_occupancy": .60, "ridge": RIDGE, "balanced_triplets": True,
            "arms": {"binary": "Fit original and processing as one negative class.",
                     "three_class": "Fit original, processing and inpainting as separate labels. Compare inpainted logit with logsumexp of both negative logits."}},
    "calibration": "Score the original and the exact-site plus full-frame autoencoder controls, with local blur and conventional copy. Strictly above second-largest ordinary source maximum per arm.",
    "held": "Score sharp/feathered edits with unchanged, exact-site processing, full processing, matched blur, same-photo copy and color adjustments.",
    "geometry": {"patch_sides": [3, 4, 5, 6, 8, 10, 12, 14], "maximum_aspect_ratio": 2, "maximum_area_share": .10,
                 "score": "Mean patch score in region minus higher of image median or adjacent ring, same for both heads."},
    "comparator": "Current JPEG aligned luma check on identical deliveries.",
    "useful": {"iou": .30, "recall": .50}, "gate": {"additional_useful_feathered_groups": 3, "minimum_families": 2,
                                                     "additional_ordinary_groups": 2, "misplaced_edit_flags": 4},
    "confidence_weight": 0,
}
read, write, require, sha = base.read, base.write, base.require, base.sha


def checkpoint(path):
    entries = yaml.safe_load(Path("backend/app/models.yaml").read_bytes())["models"]
    row = next(item for item in entries if item["id"] == "dinov2.small")
    require(row["licence"]["spdx"] == "Apache-2.0" and row["source"]["revision"] == REV, "Commercial checkpoint record changed")
    files = row["source"]["files"]
    require(all((path / item["path"]).stat().st_size == item["bytes"] and roster.sha(path / item["path"]) == item["sha256"]
                for item in files), "Exact pinned local encoder files required")
    return {"revision": REV, "license": "Apache-2.0", "files": files}


def raw_cases(folder, kind):
    if kind == "bounded":
        plan = read(folder / "plan.json")["cases"]
        for row, native, exact in bounded.inputs(folder):
            yield plan[row["slot"]], row["generator"], native, exact
    elif kind == "insertion":
        _, plan, attempts = insertions.checked_inputs(folder)
        for slot, _, native, exact in insertions.panels(folder, plan, attempts):
            yield plan["cases"][slot], "sd15", native | {"global_processing": native["processed"]}, exact
    else:
        plan = read(folder / "plan.json")
        for item, native, exact in recent.held_inputs(folder, plan):
            if item["generator"] == "sd15":
                yield item, "sd15", native, exact


def patch_share(mask):
    rows, cols = (n // PATCH for n in mask.shape)
    return mask[:rows*PATCH, :cols*PATCH].reshape(rows, PATCH, cols, PATCH).mean(axis=(1, 3))


def cases(args, phase):
    for role, cohort, relative, kind in COHORTS:
        if role != phase:
            continue
        folder = args.archive / relative if relative else args.later
        for item, family, native, truth in raw_cases(folder, kind):
            credit = item["photo_attribution"]
            require(credit["License"] == roster.PHOTO_LICENCE and bool(credit["Author"]), "Original photo rights missing")
            original, edited, processed = (native[key] for key in ("original", "edited", "global_processing"))
            local = np.where(truth[..., None], processed, original)
            require(np.array_equal(truth, np.any(edited != original, axis=2)), "Exact changed pixels differ")
            sites = (patch_share(truth) >= .75) & (patch_share(np.any(local != original, axis=2)) >= .60)
            finger = source.fingerprint(b"", original)
            info = {"cohort": cohort, "family": family, "credit": credit, "author": item["flickr_owner"].casefold(),
                    "photo": item["flickr_photo_id"], "recipient": item["recipient"], "size": finger["size"], "sketch": finger["sketch"],
                    "original_sha256": item["original"]["sha256"], "original_rgb_sha256": finger["rgb_sha256"],
                    "mask_sha256": sha(truth.tobytes()), "edit_sha256": sha(edited.tobytes()), "process_sha256": sha(local.tobytes()),
                    "global_processing_sha256": sha(processed.tobytes()), "matched_sites": int(sites.sum())}
            yield info, {"original": original, "edited": edited, "global_processing": processed, "localized_processing": local}, truth, sites


def source_profile(args):
    panels = {role: [item for item, *_ in cases(args, role)] for role in ("fit", "calibration", "held")}
    all_rows = [item for rows in panels.values() for item in rows]
    require([len(v) for v in panels.values()] == [20, 8, 20] and sum(r["matched_sites"] < SITES for r in panels["fit"]) == 2,
            "Frozen source counts or matched fit admission changed")
    for key in ("photo", "author", "recipient", "original_sha256", "original_rgb_sha256"):
        require(len({item[key] for item in all_rows}) == len(all_rows), "Source identity overlaps")
    require(not any(source.near(a, b) for a, b in combinations(all_rows, 2)), "Source thumbnail overlaps")
    return panels


def split_tiles(rgb):
    h, w = rgb.shape[:2]
    added = tuple((-dim) % TILES for dim in (h, w))
    image = np.pad(rgb, ((0, added[0]), (0, added[1]), (0, 0)), mode="reflect")
    rows, cols = (dim // TILES for dim in image.shape[:2])
    tiles = image.reshape(rows, TILES, cols, TILES, 3).transpose(0, 2, 1, 3, 4)
    return tiles.reshape(-1, TILES, TILES, 3), (rows, cols)


def stitch(values, dims, shape):
    rows, cols = dims
    width = TILES // PATCH
    channels = values.shape[-1]
    grid = values.reshape(rows, cols, width, width, channels).transpose(0, 2, 1, 3, 4)
    grid = grid.reshape(rows*width, cols*width, channels)
    height, length = shape[:2]
    return grid[:height // PATCH, :length // PATCH]


def extractor(path):
    import torch
    from transformers import Dinov2Model

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    encoder = Dinov2Model.from_pretrained(path, local_files_only=True, use_safetensors=True).cpu().eval()
    require(encoder.config.patch_size == PATCH and encoder.config.hidden_size == 384, "Pinned model geometry changed")
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]
    calls = Counter()

    @torch.inference_mode()
    def read_features(body):
        rgb = base.decode(body)
        tiles, dims = split_tiles(rgb)
        chunks = []
        for first in range(0, len(tiles), BATCH):
            batch = torch.from_numpy(tiles[first:first+BATCH]).permute(0, 3, 1, 2).float() / 255
            outputs = encoder(pixel_values=(batch - mean) / std, output_hidden_states=True)
            patches = [torch.nn.functional.normalize(outputs.hidden_states[layer][:, 1:], dim=-1) for layer in LAYERS]
            result = torch.cat(patches, dim=-1).numpy()
            require(np.isfinite(result).all(), "Nonfinite frozen dense features")
            chunks.append(result)
            calls.update(batches=1, tiles=len(batch))
        calls.update(deliveries=1)
        return stitch(np.concatenate(chunks), dims, rgb.shape), rgb

    return read_features, calls


def bound_cases(args, profile, phase):
    for (item, values, truth, sites), expected in zip(cases(args, phase), profile[phase], strict=True):
        require(item == expected, "Source triple changed since freeze")
        yield item, values, truth, sites


def fit_head(features, labels, classes):
    target = np.eye(classes)[labels]
    matrix = np.column_stack((features, np.ones(len(features))))
    width = matrix.shape[-1]

    def objective(flat):
        weights = flat.reshape(width, classes)
        output = matrix @ weights
        total = logsumexp(output, axis=1)
        p = np.exp(output - total[:, None])
        loss = float((total - (target * output).sum(axis=1)).mean()) + RIDGE * np.square(weights[:-1]).sum() / 2
        grad = matrix.T @ (p - target) / len(matrix)
        grad[:-1] += RIDGE * weights[:-1]
        return loss, grad.ravel()

    result = minimize(objective, np.zeros(width*classes), method="L-BFGS-B", jac=True, options={"maxiter": 160, "gtol": 1e-7})
    require(result.success and np.isfinite(result.x).all(), "Frozen logistic head did not converge")
    return result.x.reshape(width, classes), {"loss": float(result.fun), "iterations": int(result.nit)}


def fit(args, profile, read_features):
    gathered = []
    rows = []
    for item, native, _, sites in bound_cases(args, profile, "fit"):
        locations = np.argwhere(sites)
        if len(locations) < SITES:
            rows.append({"recipient": item["recipient"], "family": item["family"], "status": "too_few_matched_sites"})
            continue
        seed = int(sha(f"{SCHEMA}/{SEED}/{item['recipient']}".encode())[:12], 16)
        sampled = locations[np.random.default_rng(seed).choice(len(locations), SITES, replace=False)]
        y, x = sampled.T
        names = ("original", "localized_processing", "edited")
        bodies = [base.encode(native[name], QUALITY) for name in names]
        data = np.stack([read_features(body)[0][y, x] for body in bodies], axis=1)
        gathered.append(data)
        rows.append({"recipient": item["recipient"], "family": item["family"], "status": "fit", "locations": sampled.tolist(),
                     "delivery_sha256": dict(zip(FIT_LABELS, [sha(body) for body in bodies], strict=True)),
                     "matched_feature_sha256": sha(data.tobytes())})
        print("fit source", len(rows), flush=True)
    matrix = np.concatenate(gathered).reshape(-1, len(LAYERS)*384).astype(np.float64)
    labels = np.tile(np.arange(3), len(matrix)//3)
    require(len(matrix) == 18 * 3 * SITES, "Fit source count changed")
    result, status = {}, {}
    for name, target, classes in (("binary", labels == 2, 2), ("three_class", labels, 3)):
        result[name], status[name] = fit_head(matrix, target.astype(int), classes)
    write(args.output / "fit-inputs.json", rows)
    write(args.output / "heads.json", {key: weights.tolist() for key, weights in result.items()})
    return result, status


def proposal(field):
    rows, cols = field.shape
    center = float(np.median(field))
    table = np.pad(field.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best = {"box": None, "score": None}
    for h in DESIGN["geometry"]["patch_sides"]:
        for w in DESIGN["geometry"]["patch_sides"]:
            area = h*w
            if max(h, w) > 2*min(h, w) or area > .10*rows*cols or h + 2 > rows or w + 2 > cols:
                continue
            x, y = np.meshgrid(np.arange(1, cols-w), np.arange(1, rows-h))
            core = base.sums(table, x, y, x+w, y+h)
            ring = base.sums(table, x-1, y-1, x+w+1, y+h+1) - core
            score = (core / area - np.maximum(center, ring / ((h+2)*(w+2)-area))) * area**.10
            i, j = np.unravel_index(score.argmax(), score.shape)
            value = float(score[i, j])
            if best["score"] is None or value > best["score"]:
                best = {"box": [int(p)*PATCH for p in (x[i,j],y[i,j],x[i,j]+w,y[i,j]+h)], "score": value}
    return best


def score_map(features, weights):
    output = features @ weights[:-1] + weights[-1]
    return output[..., -1] - logsumexp(output[..., :-1], axis=-1)


def views(args, profile, phase):
    for item, native, truth, _ in bound_cases(args, profile, phase):
        controls, edits = base.authored(native, truth, item["recipient"])
        ordinary = {"original": native["original"], "local_processing": native["localized_processing"],
                    "global_processing": native["global_processing"], **{r: controls[r] for r in ("local_blur", "local_copy")}}
        if phase == "held":
            ordinary["local_tone"] = controls["local_tone"]
            ordinary.update(edits)
        for role, rgb in ordinary.items():
            body = base.encode(rgb, QUALITY)
            exact = np.any(rgb != native["original"], axis=2) if role in HELD_POSITIVE else np.zeros(rgb.shape[:2], bool)
            yield {"source": item["recipient"], "cohort": item["cohort"], "family": item["family"], "role": role,
                   "input_sha256": sha(body), "truth_sha256": sha(exact.tobytes())}, body, exact


def measure(args, profile, phase, extractor, heads, points=None):
    rows = []
    for item, body, truth in views(args, profile, phase):
        features, rgb = extractor(body)
        arms = {name: proposal(score_map(features, weights)) for name, weights in heads.items()}
        if points is not None:
            reference = base.comparison.format_reference(f"jpeg{QUALITY}", body, rgb)
            arms["current_luma"] = {"box": reference["box"] if reference and reference["selected"] else None, "score": None}
        for name, hit in arms.items():
            evidence = overlap(truth, hit["box"])
            hit["overlap"] = evidence
            hit["useful"] = base.geometry.useful(evidence)
            if points is not None:
                hit["flagged"] = bool(hit["box"]) and (name == "current_luma" or hit["score"] > points[name])
        rows.append(item | {"arms": arms})
        if len(rows) % 16 == 0:
            print(phase, len(rows), flush=True)
    return rows


def operating_points(rows):
    points = {}
    for arm in HEADS:
        groups = defaultdict(lambda: -float("inf"))
        for row in rows:
            score = row["arms"][arm]["score"]
            groups[row["source"]] = max(groups[row["source"]], score if score is not None else -float("inf"))
        require(len(groups) == 8, "Expected independent ordinary-only calibration groups")
        points[arm] = sorted(groups.values())[-2]
    return points


def summarize(rows):
    scope, ordinary, edits, misplaced = defaultdict(Counter), defaultdict(set), defaultdict(set), Counter()
    for row in rows:
        role, group, family = (row[k] for k in ("role", "source", "family"))
        for name, hit in row["arms"].items():
            useful = hit["flagged"] and hit["useful"]
            scope[f"{family}/{role}/{name}"].update(views=1, useful=useful, flagged=hit["flagged"],
                                                    useful_before_point=hit["useful"])
            if role in HELD_POSITIVE:
                misplaced[name] += hit["flagged"] and not useful
                if useful and role == "feathered":
                    edits[name].add((family, group))
            elif hit["flagged"]:
                ordinary[name].add(group)
    result = {}
    for name in HEADS:
        new = edits[name] - edits["current_luma"]
        actual = {"additional_useful_feathered_groups": len(new), "minimum_families": len({family for family, _ in new}),
                  "additional_ordinary_groups": len(ordinary[name] - ordinary["current_luma"]), "misplaced_edit_flags": misplaced[name]}
        bounds = DESIGN["gate"]
        result[name] = {"actual": actual, "useful_feathered_vs_binary": len(edits[name] - edits["binary"]),
                        "ordinary_groups": len(ordinary[name]), "passes": all(actual[key] >= value if key in
                        {"additional_useful_feathered_groups", "minimum_families"} else actual[key] <= value for key, value in bounds.items())}
    return {"strata": scope, "outcome": result}


def audit(args, profile, phase, results, points=None):
    replayed = 0
    for (identity, _, truth), saved in zip(views(args, profile, phase), results, strict=True):
        require(all(saved[key] == value for key, value in identity.items()), "JPEG or exact changed mask differs")
        for name, arm in saved["arms"].items():
            match = overlap(truth, arm["box"])
            useful = match["iou"] is not None and match["iou"] >= .30 and match["recall"] >= .50
            require(arm["overlap"] == match and arm["useful"] == useful, "Native overlap differs")
            if points is not None:
                require(arm["flagged"] == (bool(arm["box"]) and (name == "current_luma" or arm["score"] > points[name])), "Flag differs")
            replayed += 1
    return replayed


def audit_fit(args, profile, rows):
    count = 0
    for (item, native, _, eligible), row in zip(bound_cases(args, profile, "fit"), rows, strict=True):
        require(item["recipient"] == row["recipient"] and item["family"] == row["family"], "Fit source role differs")
        locations = np.argwhere(eligible)
        if len(locations) < SITES:
            require(row["status"] == "too_few_matched_sites", "Ineligible triplet was fitted")
            continue
        seed = int(sha(f"{SCHEMA}/{SEED}/{item['recipient']}".encode())[:12], 16)
        chosen = locations[np.random.default_rng(seed).choice(len(locations), SITES, replace=False)]
        require(row["status"] == "fit" and row["locations"] == chosen.tolist(), "Fit source-pixel coordinates differ")
        keys = ("original", "localized_processing", "edited")
        delivery = dict(zip(FIT_LABELS, [sha(base.encode(native[key], QUALITY)) for key in keys], strict=True))
        require(delivery == row["delivery_sha256"], "Fit delivery bytes differ")
        count += len(keys)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    for key in ("archive", "later", "checkpoint", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    folder = args.output
    profile = source_profile(args)
    lock = {"schema": SCHEMA, "design": DESIGN, "profile": profile, "checkpoint": checkpoint(args.checkpoint),
            "runner_sha256": roster.sha(Path(__file__)), "versions": {k: version(k) for k in ("torch", "transformers", "numpy", "Pillow", "scipy")}}
    if args.stage == "freeze":
        folder.mkdir(mode=0o700)
        write(folder / "freeze.json", lock)
        print(roster.sha(folder / "freeze.json"), {key: len(val) for key, val in profile.items()}, flush=True)
        return
    require(read(folder / "freeze.json") == lock, "Frozen source, implementation or checkpoint changed")
    if args.stage == "run":
        start = time.monotonic()
        features, calls = extractor(args.checkpoint)
        heads, fitted = fit(args, profile, features)
        ordinary = measure(args, profile, "calibration", features, heads)
        points = operating_points(ordinary)
        write(folder / "calibration.json", ordinary)
        write(folder / "points.json", points)
        print("points", points, flush=True)
        held = measure(args, profile, "held", features, heads, points)
        write(folder / "held.json", held)
        results = {"schema": SCHEMA, "design": DESIGN, "fit": fitted, "points": points, "counts": calls,
                   "sources": {role: dict(Counter(row["family"] for row in values)) for role, values in profile.items()},
                   "held_rows": len(held), "calibration_rows": len(ordinary), "seconds": round(time.monotonic()-start, 2),
                   "commitments": {name: roster.sha(folder / name) for name in ("freeze.json", "heads.json", "fit-inputs.json", "calibration.json", "points.json", "held.json")},
                   "checkpoint_sha256": lock["checkpoint"]["files"][-1]["sha256"], "runner_sha256": lock["runner_sha256"], **summarize(held)}
        write(folder / "result.json", results)
        print(results["outcome"], calls, flush=True)
        return
    results = read(folder / "result.json")
    require(all(roster.sha(folder / name) == expected for name, expected in results["commitments"].items()), "Result journal differs")
    cal, held = (read(folder / f"{name}.json") for name in ("calibration", "held"))
    points = operating_points(cal)
    require(points == results["points"] == read(folder / "points.json"), "Calibration does not reproduce")
    replayed = audit(args, profile, "calibration", cal) + audit(args, profile, "held", held, points)
    require(summarize(held) == {key: results[key] for key in ("strata", "outcome")}, "Held arithmetic differs")
    results["audit"] = {"exact_input_and_mask_bindings": len(cal)+len(held), "native_overlap_readbacks": replayed,
                        "fit_deliveries_and_token_coordinates": audit_fit(args, profile, read(folder / "fit-inputs.json"))}
    require(args.publication is not None, "Review output path required")
    write(args.publication, results)
    print(results["audit"], flush=True)


if __name__ == "__main__":
    main()
