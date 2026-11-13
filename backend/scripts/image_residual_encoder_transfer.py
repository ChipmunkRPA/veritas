# SPDX-License-Identifier: Apache-2.0
"""Learn a small native-pixel residual encoder from matched, attributed edit pairs."""

from __future__ import annotations

import argparse
import io
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.nn import functional as F

from scripts import image_background_patch_score as bounded
from scripts import image_bg_real_acquisition_v221 as source
from scripts import image_bg_real_insertion_score as insertion
from scripts import image_bg_real_transport_v225 as transport
from scripts import image_edge_spread_transfer as geometry
from scripts import image_residual_color_transfer as comparison
from scripts.image_source_groups_v22 import near

SCHEMA = "chipmunk.image-residual-encoder-transfer/1"
ROSTERS = {
    "fit": {"diffusion_bounded_fit": 12, "sd15_insertions": 8},
    "calibration": {"backup_originals": 12},
    "held": {"diffusion_bounded_held": 12, "lama_removals": 24},
}
ARCHIVES = {
    "diffusion_bounded_fit": "image-background-patch-transfer-2026-11-11/intake",
    "sd15_insertions": "image-bg-real-small-insertions-2026-11-11/trial",
    "diffusion_bounded_held": "image-wavelet-region-transfer-2026-11-11/holdout",
    "backup_originals": "image-bg-real-backup-intake-v262",
    "lama_removals": "image-bg-real-acquisition-v225",
}
QUALITIES = (75, 90)
STRIDE, RECEPTIVE, CENTER = 8, 35, 17
SIDES = (5, 7, 9, 12, 15, 18, 22, 26)
SEED, SITES, EPOCHS, BATCH = 6153, 8, 10, 16
POSITIVE = ("sharp", "feathered")
CONTROL = ("original", "local_blur", "local_tone", "local_copy", "global_resize", "second_jpeg")
PROCESSING = ("local_processing", "global_processing")
GATE = {"added_feathered_sources": 5, "added_families": 2, "extra_ordinary_groups": 3, "wrong_edit_views": 4}
DESIGN = {
    "roles": ROSTERS, "quality": list(QUALITIES), "subsampling": "4:2:0",
    "fit": {"seed": SEED, "epochs": EPOCHS, "batch_pairs": BATCH, "sites_per_source_quality": SITES,
            "positive_minimum_inner_occupancy": .65, "optimizer": "AdamW .001, weight decay .0001",
            "loss": "group-balanced logistic loss plus .25 times matched contrastive logistic loss"},
    "encoder": {"input": "native RGB minus valid 3x3 RGB mean, scaled by 12/255", "convolution_channels": [12, 16, 20],
                "convolution_widths": [5, 3, 3], "stride": STRIDE, "receptive_field": RECEPTIVE},
    "proposal": {"side_cells": list(SIDES), "minimum_aspect_ratio": .5, "maximum_area_share": .10,
                 "score": "mean logit minus larger of global median and local ring mean, weighted by area^0.10"},
    "calibration": "Strictly above second-highest ordinary source maximum, independently for each JPEG quality.",
    "useful": {"iou": .30, "recall": .50}, "advance": GATE, "confidence_weight": 0,
}
read, write, require, sha = geometry.read, geometry.write, geometry.require, source.digest


def decoded_png(body, case):
    return source.png_pixels(body, case["size"])


def transport_cases(roster, name, positives):
    folder = roster / ARCHIVES[name]
    sealed_path = folder / "qualified-private.json"
    summary = read(folder / ("summary-v225.json" if positives else "summary.json"))
    sealed = read(sealed_path)
    require(sha(sealed_path.read_bytes()) == summary["qualified_private_sha256"], "Qualified source index changed")
    require(sealed["acquisition_journal_sha256"] == sha((folder / "attempts.jsonl").read_bytes()), "Source journal changed")
    cache = Path(sealed["cache"])
    for entry in sealed["qualified"]:
        case, measured = entry["source"], entry["measurement"]
        files = measured["files"]
        original = decoded_png(transport.verify_file(cache, case, "original", files["original"]), case)
        views, truth = {"original": original}, None
        if positives:
            require(measured["measured"]["semantic_mask_binding"] == "publisher_asset_commitment", "Unknown mask lineage")
            edited, mask = (decoded_png(transport.verify_file(cache, case, role, files[role]), case)
                            for role in ("edited", "semantic_mask"))
            truth, support = source.edit_truth(original, edited, mask, case["semantic_edit_area"])
            truth_file = (cache / entry["truth_file"]).read_bytes()
            require(sha(truth_file) == entry["truth_png_sha256"] and
                    np.array_equal(truth, decoded_png(truth_file, case)[:, :, 0] > 0) and
                    int(truth.sum()) == measured["measured"]["changed_pixels"] and
                    support == measured["measured"]["support_pixels"], "Exact LaMa edit changed")
            views["edited"] = edited
        yield case, "lama" if positives else "ordinary", views, truth


def source_cases(args, name):
    if name in ("backup_originals", "lama_removals"):
        yield from transport_cases(args.rosters, name, name == "lama_removals")
        return
    folder = args.archive / ARCHIVES[name]
    if name == "sd15_insertions":
        _, plan, attempts = insertion.checked_inputs(folder)
        for slot, _, views, truth in insertion.panels(folder, plan, attempts):
            yield plan["cases"][slot], "sd15", {"original": views["original"], "edited": views["edited"],
                                               "global_processing": views["processed"]}, truth
        return
    cases = read(folder / "plan.json")["cases"]
    for row, views, truth in bounded.inputs(folder):
        yield cases[row["slot"]], row["generator"], {key: val for key, val in views.items() if key != "blur"}, truth


def encode(rgb, quality):
    with io.BytesIO() as output:
        Image.fromarray(rgb).save(output, "JPEG", quality=quality, subsampling=2)
        return output.getvalue()


def decode(body):
    with Image.open(io.BytesIO(body), formats=["JPEG"]) as image:
        return np.array(image.convert("RGB"))


def describe(case, family, name, views, truth):
    attribution = case["photo_attribution"]
    require(attribution["License"] == "https://creativecommons.org/licenses/by/2.0/" and attribution["Author"] and
            attribution["OriginalLandingURL"].rstrip("/").endswith("/" + case["flickr_photo_id"]) and
            attribution["bgreal_id"] == case["recipient"], "Missing original photo rights or attribution")
    native = {key: sha(rgb.tobytes()) for key, rgb in views.items()}
    for pixels in views.values():
        require(pixels.shape == (*reversed(case["size"]), 3), "Native source geometry differs")
    if truth is not None:
        require(np.array_equal(np.any(views["edited"] != views["original"], axis=2), truth) and
                .001 <= truth.mean() <= .15, "Source lacks a bounded exact edit")
    fingerprint = source.fingerprint(b"", views["original"])
    return {"cohort": name, "generator": family, "recipient": case["recipient"], "author": case["flickr_owner"].casefold(),
            "photo": case["flickr_photo_id"], "attribution": attribution, "original_sha256": case["original"]["sha256"],
            "publisher_split": case["publisher_split"], "native": native, "size": case["size"],
            "sketch": fingerprint["sketch"], "exact_mask_sha256": sha(truth.tobytes()) if truth is not None else None}


def inputs(args, part):
    for name in ROSTERS[part]:
        for case, family, views, truth in source_cases(args, name):
            yield describe(case, family, name, views, truth), views, truth


def freeze(args):
    parts = {key: [row for row, _, _ in inputs(args, key)] for key in ROSTERS}
    all_rows = [row for rows in parts.values() for row in rows]
    for key, rows in parts.items():
        require(Counter(row["cohort"] for row in rows) == ROSTERS[key], "Expected exact source roster counts")
    for key in ("recipient", "author", "photo", "original_sha256"):
        require(len({r[key] for r in all_rows}) == len(all_rows), f"Repeated source identity: {key}")
    require(len({r["native"]["original"] for r in all_rows}) == len(all_rows), "Repeated original RGB")
    require(not any(near(a, b) for a, b in combinations(all_rows, 2)), "Informative source thumbnails overlap")
    paths = (Path(__file__), Path(source.__file__), Path(transport.__file__), Path(bounded.__file__), Path(insertion.__file__),
             Path(comparison.__file__), Path(comparison.jpeg_dq.__file__), Path(geometry.__file__))
    profile = {"schema": SCHEMA, "design": DESIGN, "sources": parts,
               "code_sha256": {path.name: sha(path.read_bytes()) for path in paths},
               "dependencies": {package: version(package) for package in ("torch", "numpy", "Pillow", "opencv-python-headless")}}
    write(args.output / "freeze.json", profile)
    print("freeze", sha((args.output / "freeze.json").read_bytes()), {p: len(rows) for p, rows in parts.items()}, flush=True)


def frozen_inputs(args, part, profile):
    for (row, views, truth), expected in zip(inputs(args, part), profile["sources"][part], strict=True):
        require(row == expected, "Frozen source identity or native edit differs")
        yield row, views, truth


def authored(views, truth, identity):
    original = views["original"]
    height, width = original.shape[:2]
    photo = Image.fromarray(original)
    if truth is None:
        # Calibration retouching uses an ordinary region chosen independently of any edit pair.
        rng = np.random.default_rng(int(sha(identity.encode())[:12], 16))
        x, y = rng.uniform(.26, .74, 2)
        rx, ry = rng.uniform(.095, .15, 2)
        yy, xx = np.ogrid[:height, :width]
        support = ((xx / width - x) / rx) ** 2 + ((yy / height - y) / ry) ** 2 < 1
    else:
        # The three matched local nuisances use the very same footprint as the generated render.
        support = truth
    dist = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    alpha = ((1 - np.cos(np.clip(dist / 8, 0, 1) * np.pi)) / 2).astype(np.float32)[..., None]

    def blend(replacement):
        return np.rint(original * (1 - alpha) + replacement * alpha).astype(np.uint8)

    blur = np.asarray(photo.filter(ImageFilter.GaussianBlur(1.35)))
    tone = np.clip(np.asarray(ImageEnhance.Color(photo).enhance(.78)).astype(np.int16) + [10, -4, 8], 0, 255)
    copied = np.roll(original, (height // 7, width // 5), axis=(0, 1))
    down = photo.resize((round(.78 * width), round(.78 * height)), Image.Resampling.LANCZOS)
    restored = np.asarray(down.resize(photo.size, Image.Resampling.LANCZOS))
    ordinary = dict(zip(CONTROL, (original, blend(blur), blend(tone), blend(copied), restored, decode(encode(original, 60))),
                        strict=True))
    if "global_processing" in views:
        ordinary.update(local_processing=blend(views["global_processing"]), global_processing=views["global_processing"])
    if truth is None:
        return ordinary, {}
    generated = {"sharp": views["edited"], "feathered": blend(views["edited"])}
    for role, rgb in generated.items():
        actual = np.any(rgb != original, axis=2)
        require(actual.any() and not (actual & ~truth).any(), f"{role} edit exceeded exact support")
    return ordinary, generated


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 12, 5, stride=2), nn.SiLU(),
                                    nn.Conv2d(12, 16, 3, stride=2), nn.SiLU(),
                                    nn.Conv2d(16, 20, 3, stride=2), nn.SiLU(),
                                    nn.AvgPool2d(3, stride=1), nn.Conv2d(20, 1, 1))

    def forward(self, rgb):
        pixels = rgb.to(torch.float32)
        residual = pixels[:, :, 1:-1, 1:-1] - F.avg_pool2d(pixels, 3, stride=1)
        return self.layers(residual * (12. / 255.))[:, 0]


def sums(integral, left, top, right, bottom):
    return integral[bottom, right] - integral[bottom, left] - integral[top, right] + integral[top, left]


def grid(mask, radius):
    height, width = mask.shape
    x, y = np.meshgrid(np.arange(CENTER, width - CENTER, STRIDE), np.arange(CENTER, height - CENTER, STRIDE))
    ii = np.pad(mask.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    occupancy = sums(ii, x - radius, y - radius, x + radius + 1, y + radius + 1) / (2 * radius + 1) ** 2
    return np.stack((x, y), axis=-1), occupancy


def site_pairs(original, sharp, feather, seed):
    xy, interior = grid(np.any(original != feather, axis=2), 8)
    _, remote = grid(np.any(original != sharp, axis=2), CENTER)
    possible = xy[interior >= .65]
    outside = xy[remote == 0]
    require(len(possible) >= SITES and len(outside) >= SITES, "No exact interior or remote fit sites")
    rng = np.random.default_rng(int(sha(seed.encode())[:12], 16))
    return possible[rng.choice(len(possible), SITES, replace=False)], outside[rng.choice(len(outside), SITES, replace=False)]


def cut(rgb, positions):
    return np.stack([rgb[y-CENTER:y+CENTER+1, x-CENTER:x+CENTER+1].transpose((2, 0, 1)) for x, y in positions])


def fit(args, profile):
    pairs, fit_roster = [], []
    for row, views, truth in frozen_inputs(args, "fit", profile):
        ordinary, generated = authored(views, truth, row["recipient"])
        source_counts = []
        for quality in QUALITIES:
            near_sites, remote_sites = site_pairs(ordinary["original"], generated["sharp"], generated["feathered"],
                                                 f"{SCHEMA}/{row['recipient']}/{quality}")
            rendered = {key: decode(encode(rgb, quality)) for key, rgb in (ordinary | generated).items()}
            near_positives = np.stack([cut(rendered[key], near_sites) for key in POSITIVE], axis=1)
            ordinary_sites = np.stack([cut(rendered[key], near_sites) for key in CONTROL], axis=1)
            remote = np.stack([cut(rendered[key], remote_sites) for key in POSITIVE], axis=1)
            # Give each source the same batch shape. All fit groups have a published processing control.
            matched = np.stack([cut(rendered[key], near_sites) for key in PROCESSING], axis=1)
            combined = np.concatenate((near_positives, ordinary_sites, remote, matched), axis=1)
            pairs.append(combined)
            source_counts.append({"quality": quality, "near_sites": near_sites.tolist(), "remote_sites": remote_sites.tolist(),
                                  "patch_sha256": sha(combined.tobytes())})
        fit_roster.append({"recipient": row["recipient"], "generator": row["generator"], "sites": source_counts})
    write(args.output / "fit-inputs.json", fit_roster)
    patches = torch.from_numpy(np.concatenate(pairs))
    torch.manual_seed(SEED)
    model = Encoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    history = []
    for epoch in range(EPOCHS):
        losses = []
        for indices in torch.randperm(len(patches)).split(BATCH):
            group = patches[indices]
            scores = model(group.flatten(0, 1)).reshape(group.shape[:2])
            positives, negatives = scores[:, :len(POSITIVE)], scores[:, len(POSITIVE):]
            loss = (.5 * (F.softplus(-positives).mean() + F.softplus(negatives).mean())
                    + .25 * F.softplus(negatives[:, None, :] - positives[:, :, None]).mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        print("epoch", epoch + 1, "loss", round(history[-1], 5), flush=True)
    model.eval()
    path = args.output / "encoder.pt"
    torch.save(model.state_dict(), path)
    receipt = {"groups": len(fit_roster), "matched_sites": len(patches), "patches": int(patches.shape[0] * patches.shape[1]),
               "parameters": sum(p.numel() for p in model.parameters()), "loss": history, "checkpoint_sha256": sha(path.read_bytes()),
               "fit_inputs_sha256": sha((args.output / "fit-inputs.json").read_bytes())}
    write(args.output / "fit.json", receipt)
    return model


def proposal(field, shape):
    rows, cols = field.shape
    ii = np.pad(field.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    median = np.median(field)
    top = None
    for h in SIDES:
        for w in SIDES:
            area = h * w
            if max(h, w) > min(h, w) * 2 or area * STRIDE**2 > .10 * shape[0] * shape[1] or h > rows or w > cols:
                continue
            x, y = np.meshgrid(np.arange(cols - w + 1), np.arange(rows - h + 1))
            margin = max(2, min(h, w) // 4)
            l, t, r, b = np.maximum(x - margin, 0), np.maximum(y - margin, 0), np.minimum(x + w + margin, cols), np.minimum(y + h + margin, rows)
            core = sums(ii, x, y, x + w, y + h)
            ring_count = (r - l) * (b - t) - area
            ring = (sums(ii, l, t, r, b) - core) / np.maximum(1, ring_count)
            scores = np.where(ring_count >= area * .2, (core / area - np.maximum(median, ring)) * area**.10, -np.inf)
            idx = np.unravel_index(scores.argmax(), scores.shape)
            score = float(scores[idx])
            if np.isfinite(score) and (top is None or score > top["score"]):
                left, upper = int(x[idx]) * STRIDE + CENTER - STRIDE // 2, int(y[idx]) * STRIDE + CENTER - STRIDE // 2
                top = {"box": [left, upper, left + w * STRIDE, upper + h * STRIDE], "score": score}
    return top


def measurements(args, profile, model, part, points=None):
    rows = []
    for info, views, truth in frozen_inputs(args, part, profile):
        ordinary, generated = authored(views, truth, info["recipient"])
        for role, rgb in (ordinary | generated).items():
            exact = np.any(rgb != views["original"], axis=2) if role in POSITIVE else np.zeros(rgb.shape[:2], dtype=bool)
            for quality in QUALITIES:
                body = encode(rgb, quality)
                delivered = decode(body)
                with torch.inference_mode():
                    field = model(torch.from_numpy(delivered).permute(2, 0, 1)[None])[0].numpy()
                hit = proposal(field, exact.shape)
                require(hit is not None, "No supported native candidate region")
                area = insertion.measure(bounded.boxed(hit["box"], exact.shape, True), exact)
                candidate = {**hit, "overlap": area, "useful": geometry.useful(area)}
                row = {"source": info["recipient"], "generator": info["generator"], "cohort": info["cohort"], "role": role,
                       "quality": quality, "shape": list(exact.shape), "input_sha256": sha(body),
                       "truth_sha256": sha(exact.tobytes()), "candidate": candidate}
                if points is not None:
                    candidate["flagged"] = hit["score"] > points[str(quality)]
                    reference = comparison.format_reference(f"jpeg{quality}", body, delivered)
                    box = reference["box"] if reference and reference["selected"] else None
                    overlap = insertion.measure(bounded.boxed(box, exact.shape, box is not None), exact)
                    row["reference"] = {"box": box, "flagged": box is not None, "overlap": overlap,
                                        "useful": geometry.useful(overlap), "assessed": reference is not None}
                rows.append(row)
        print(part, len(rows), info["cohort"], flush=True)
    return rows


def calibration(rows):
    maxima = defaultdict(dict)
    for row in rows:
        quality, group = str(row["quality"]), row["source"]
        value = row["candidate"]["score"]
        maxima[quality][group] = max(value, maxima[quality].get(group, -float("inf")))
    require(all(len(values) == 12 for values in maxima.values()) and set(maxima) == {str(q) for q in QUALITIES},
            "Calibration requires twelve separate groups at each quality")
    return {quality: sorted(values.values())[-2] for quality, values in maxima.items()}, maxima


def summarize(rows):
    by_quality, strata = {}, defaultdict(Counter)
    for quality in QUALITIES:
        counts, groups, added, added_families = Counter(), defaultdict(set), set(), set()
        section = [r for r in rows if r["quality"] == quality]
        for row in section:
            role, family, source_id = (row[key] for key in ("role", "generator", "source"))
            candidate, reference = row["candidate"], row["reference"]
            target = "edits" if role in POSITIVE else "ordinary"
            counts[target] += 1
            for method, result in (("candidate", candidate), ("reference", reference)):
                has_useful = result["flagged"] and result["useful"]
                strata[f"{quality}/{family}/{role}/{method}"].update(views=1, flagged=result["flagged"], useful=has_useful,
                                                                      useful_before_point=result["useful"])
                counts[f"{method}_{target}_flags"] += result["flagged"]
                counts[f"{method}_{target}_useful"] += has_useful
                if target == "ordinary" and result["flagged"]:
                    groups[method].add(source_id)
            if target == "edits" and candidate["flagged"] and not candidate["useful"]:
                counts["wrong_edit_views"] += 1
            if role == "feathered" and candidate["flagged"] and candidate["useful"] and not reference["useful"]:
                added.add(source_id)
                added_families.add(family)
        actual = {"added_feathered_sources": len(added), "added_families": len(added_families),
                  "extra_ordinary_groups": len(groups["candidate"] - groups["reference"]), "wrong_edit_views": counts["wrong_edit_views"]}
        passed = all(actual[key] >= value if key.startswith("added") else actual[key] <= value for key, value in GATE.items())
        by_quality[quality] = {"counts": counts, "ordinary_groups": {key: len(groups[key]) for key in ("candidate", "reference")},
                               "advance": {"actual": actual, "passed": passed}}
    return {"qualities": by_quality, "strata": strata}


def run(args):
    start = time.monotonic()
    path = args.output / "freeze.json"
    profile = read(path)
    freeze_sha = sha(path.read_bytes())
    require(freeze_sha == args.expected and profile["design"] == DESIGN and
            profile["code_sha256"][Path(__file__).name] == sha(Path(__file__).read_bytes()), "Expected frozen protocol changed")
    for name, digest in profile["code_sha256"].items():
        code = Path(comparison.jpeg_dq.__file__) if name == "jpeg_dq.py" else Path(__file__).with_name(name)
        require(sha(code.read_bytes()) == digest, f"Frozen dependency changed: {name}")
    require(all(version(name) == pin for name, pin in profile["dependencies"].items()), "Frozen library changed")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    model = fit(args, profile)
    cal = measurements(args, profile, model, "calibration")
    write(args.output / "calibration.json", cal)
    points, maxima = calibration(cal)
    point_path = args.output / "points.json"
    write(point_path, {"points": points, "maxima": maxima})
    print("points", points, flush=True)
    rows = measurements(args, profile, model, "held", points)
    write(args.output / "held.json", rows)
    aggregate = {"schema": SCHEMA, "freeze_sha256": freeze_sha, "design": DESIGN, "fit": read(args.output / "fit.json"),
                 "source_counts": {key: dict(Counter(r["generator"] for r in data)) for key, data in profile["sources"].items()},
                 "code_sha256": profile["code_sha256"], "dependencies": profile["dependencies"], "points": points,
                 "artifact_sha256": {name: sha((args.output / name).read_bytes()) for name in ("calibration.json", "held.json", "points.json")},
                 "elapsed_s": round(time.monotonic() - start, 2), "held_rows": len(rows), "calibration_rows": len(cal), **summarize(rows)}
    write(args.output / "aggregate.json", aggregate)
    print(aggregate["qualities"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run"))
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--rosters", default=Path(__file__).resolve().parents[1] / "data" / "independent-evaluation", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected", help="Required freeze SHA-256 for run")
    args = parser.parse_args()
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        freeze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
