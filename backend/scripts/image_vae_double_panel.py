# SPDX-License-Identifier: Apache-2.0
"""Separate same-photo global reconstruction from a local inpainted edit."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from importlib.metadata import version
from inspect import getfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.detectors.image.adapters.loaders import _entry, _file

from scripts import image_residual_encoder_transfer as base
from scripts import image_three_class_patch_probe as prior
from scripts.image_region_rank_features import FrozenEncoder
from scripts.review_image_residual_encoder_transfer import overlap

PHASES = ("fit", "calibration", "held")
FILES = {
    "bounded_fit": ("image-background-patch-transfer-2026-11-11/intake", "bounded"),
    "sd15_mixed": ("image-bg-real-small-insertions-2026-11-11/trial", "insertion"),
    "sd15_cal": ("image-bg-real-remote-context-2026-11-11/trial", "insertion"),
    "bounded_held": ("image-wavelet-region-transfer-2026-11-11/holdout", "bounded"),
}
CODECS = ("jpeg60", "jpeg90_then_jpeg60")
POSITIVES = ("original_outside", "vae_outside")
NEGATIVES = ("pristine", "global_vae", "local_vae", "local_blur", "local_copy")
ARMS = ("single_binary", "single_three_class", "paired_binary", "paired_three_class")
CAP = 6


def sha(value):
    return hashlib.sha256(value).hexdigest()


def rank(value):
    return sha(f"chipmunk-vae-double-v1/{value}".encode())


def write(path, item):
    with path.open("x") as f:
        json.dump(item, f, sort_keys=True, indent=2, allow_nan=False, ensure_ascii=False)
        f.write("\n")


def read(path):
    return json.loads(path.read_bytes())


def load_groups(root):
    groups = {}
    for name, (folder, kind) in FILES.items():
        groups[name] = []
        for info, family, images, truth in prior.raw_cases(root / folder, kind):
            if family not in ("sdxl", "kandinsky", "sd15") or info["photo_attribution"]["License"] != prior.roster.PHOTO_LICENCE:
                raise ValueError("Unqualified photo or generator")
            original, edited, reconstructed = (images[k] for k in ("original", "edited", "global_processing"))
            if not np.array_equal(truth, np.any(edited != original, axis=2)):
                raise ValueError("The splice truth is not exact on the native pixel clock")
            changed = np.any(reconstructed != original, axis=2)
            paired = np.where(truth[..., None], edited, reconstructed)
            if not np.array_equal(paired[truth], edited[truth]) or not np.array_equal(paired[~truth], reconstructed[~truth]):
                raise ValueError("Matched geometry and reconstruction lineage differ")
            src = {"recipient":info["recipient"], "photo":info["flickr_photo_id"],
                   "author":info["flickr_owner"].casefold(), "credited_author":info["photo_attribution"]["Author"].strip().casefold(),
                   "family":family, "archive_cohort":name, "shape":list(truth.shape),
                   "original":sha(original.tobytes()), "edited":sha(edited.tobytes()), "reconstructed":sha(reconstructed.tobytes()),
                   "truth":sha(truth.tobytes()), "paired":sha(paired.tobytes()), "edit_pixels":int(truth.sum()),
                   "changed_outside":int((changed & ~truth).sum())}
            if not src["edit_pixels"] or not src["changed_outside"]:
                raise ValueError("Insufficient authored outside reconstruction")
            materials = {"pristine":original, "global_vae":reconstructed, "original_outside":edited, "vae_outside":paired}
            groups[name].append((src, materials, truth, changed))
    for items in groups.values():
        items.sort(key=lambda item: rank(item[0]["recipient"]))
    sd = groups["sd15_mixed"]
    rows = {"fit":groups["bounded_fit"] + sd[:4], "calibration":groups["sd15_cal"], "held":groups["bounded_held"] + sd[4:]}
    sizes = [len(rows[k]) for k in PHASES]
    if sizes != [16, 8, 16]:
        raise ValueError(f"Unexpected archived source counts {sizes}")
    if any(item[0]["changed_outside"] < 4 * item[0]["edit_pixels"] for role in PHASES[1:] for item in rows[role]):
        raise ValueError("Insufficient reconstructed support at evaluation")
    fields = ("recipient", "photo", "author", "credited_author", "original")
    for key in fields:
        products = [{item[0][key] for item in rows[role]} for role in PHASES]
        if len(set.union(*products)) != sum(map(len, products)):
            raise ValueError(f"Source role overlap on {key}")
    return rows


def processing(root):
    for phase, items in load_groups(root).items():
        for source, materials, truth, changed in items:
            original, recon = materials["pristine"], materials["global_vae"]
            if phase != "fit":
                ordinary, _ = base.authored({"original":original, "edited":materials["original_outside"],
                                             "global_processing":recon}, truth, source["recipient"])
                materials["local_vae"] = np.where(truth[..., None], recon, original)
                for name in ("local_blur", "local_copy"):
                    materials[name] = ordinary[name]
            yield phase, source, materials, truth, changed


def encode(rgb, codec):
    def jpeg(values, quality):
        buf = io.BytesIO()
        Image.fromarray(values, "RGB").save(buf, format="JPEG", quality=quality, subsampling=2, optimize=False, progressive=False)
        return buf.getvalue()
    previous = None
    if codec == CODECS[1]:
        previous = jpeg(rgb, 90)
        rgb = np.asarray(Image.open(io.BytesIO(previous)).convert("RGB"))
    return jpeg(rgb, 60), sha(previous) if previous else None


def train_sites(truth, changed, source):
    if source["changed_outside"] < 4 * source["edit_pixels"]:
        return None
    positive = prior.patch_share(truth)
    processing_share = prior.patch_share(changed)
    inside = np.argwhere((positive >= .65) & (processing_share >= .60))
    outside = np.argwhere((positive == 0) & (processing_share >= .60))
    if not len(inside) or len(outside) < CAP:
        return None
    rng = np.random.default_rng(int(rank(source["recipient"])[:12], 16))
    size = min(CAP, len(inside))
    return inside[rng.choice(len(inside), size, replace=False)], outside[rng.choice(len(outside), size, replace=False)]


def checkpoint():
    model = _entry("dinov2.small")
    return prior.checkpoint(_file(model, "config.json").parent)


def implementation():
    files = (Path(__file__), Path(prior.__file__), Path(base.__file__), Path(getfile(FrozenEncoder)), Path(getfile(overlap)))
    return {file.name: sha(file.read_bytes()) for file in files}


def frozen(root):
    cases = {}
    for phase, source, views, truth, changed in processing(root):
        sites = train_sites(truth, changed, source) if phase == "fit" else None
        selection = None if sites is None else [array.tolist() for array in sites]
        encodings = {}
        cases[source["recipient"]] = {"phase":phase, "info":source, "fit_sites":selection, "deliveries":encodings}
        for codec in CODECS:
            names = list(views) if phase != "calibration" else NEGATIVES
            for name in names:
                body, initial = encode(views[name], codec)
                encodings[codec + "/" + name] = [sha(body), initial]
    if sum(row["fit_sites"] is not None for row in cases.values()) < 13:
        raise ValueError("Too few distinct fit groups met patch support")
    return {"cases": cases, "model": checkpoint(), "implementation": implementation(),
            "versions": {name: version(name) for name in ("numpy", "pillow", "torch", "transformers")},
            "resize":False, "native_cells":14, "tile_px":224, "tile_step_px":168,
            "matching":{"iou":.30, "recall":.5},
            "cutoff":"Fit-only training, then strictly exceed the second-highest control-group maximum per arm and delivery.",
            "source_terms":{"dataset":"BG-REAL v2, CC BY 4.0 contributions, attribution CC BY 2.0 photos.",
                            "generators":"Kandinsky Apache-2.0, SD1.5 CreativeML OpenRAIL-M, SDXL OpenRAIL++-M."}}


def rows_for_fit(features, sites):
    inside, outside = (np.asarray(values) for values in sites)
    matrix, labels = [], []
    for name, pair in (("pristine",(0,0)), ("global_vae",(1,1)), ("original_outside",(2,0)), ("vae_outside",(2,1))):
        for coords, label in zip((inside, outside), pair, strict=True):
            y, x = coords.T
            matrix.append(features[name][y,x])
            labels.extend([label] * len(x))
    return np.concatenate(matrix), np.asarray(labels)


def measure(root, plan_path, output, step, prior_file):
    import resource

    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    began = time.perf_counter()
    freeze = read(plan_path)
    if freeze["implementation"] != implementation() or freeze["model"] != checkpoint():
        raise ValueError("Frozen runner or permitted encoder changed")
    freeze_hash = sha(plan_path.read_bytes())
    encoder = FrozenEncoder("cpu")
    training, rows = defaultdict(list), []
    heads, points = {}, {}
    if step != "fit":
        prior_run = read(prior_file)
        preceding = {"calibration": "fit", "held": "calibration"}
        if prior_run["freeze_sha256"] != freeze_hash or prior_run["phase"] != preceding[step]:
            raise ValueError("The preceding phase must use the current source and method freeze")
        heads = {arm:np.asarray(value, np.float64) for arm, value in prior_run["heads"].items()}
        points = prior_run.get("points",{})
    n = 0
    for phase, source, views, truth, _ in processing(root):
        if phase != step:
            continue
        group = freeze["cases"][source["recipient"]]
        if group["phase"] != phase or group["info"] != source:
            raise ValueError("Frozen source changed")
        if step == "fit" and group["fit_sites"] is None:
            continue
        for codec in CODECS:
            features = {}
            names = tuple(views) if step != "calibration" else NEGATIVES
            for name in names:
                body, initial = encode(views[name], codec)
                identity = [sha(body), initial]
                if identity != group["deliveries"][f"{codec}/{name}"]:
                    raise ValueError("Native delivery changed")
                rgb = np.asarray(Image.open(io.BytesIO(body)).convert("RGB"))
                features[name] = encoder(rgb)
                n += 1
                if step == "fit":
                    continue
                target = truth if name in POSITIVES else np.zeros_like(truth)
                arms = {}
                for arm, weights in heads.items():
                    pick = prior.proposal(prior.score_map(features[name], weights))
                    arms[arm] = {**pick, **overlap(target, pick["box"])}
                if step == "held":
                    current = base.comparison.format_reference("jpeg60", body, rgb)
                    box = current["box"] if current and current["selected"] else None
                    arms["current_luma"] = {"box":box, "score":None, **overlap(target, box)}
                rows.append({"source":source["recipient"],"family":source["family"],"delivery":codec,
                             "view":name,"sha256":identity[0],"model":arms})
            if step == "fit":
                training[codec].append(rows_for_fit(features, group["fit_sites"]))
        print(step, source["recipient"], source["family"], n, flush=True)
    result = {"phase":step, "freeze_sha256":freeze_hash, "rows":rows}
    if step == "fit":
        for scope, selected in (("single", CODECS[:1]), ("paired", CODECS)):
            examples = [item for codec in selected for item in training[codec]]
            matrix = np.concatenate([item[0] for item in examples]).astype(np.float64)
            labels = np.concatenate([item[1] for item in examples])
            for title, target, classes in (("binary", labels == 2, 2), ("three_class", labels, 3)):
                weights, stats = prior.fit_head(matrix, target.astype(int), classes)
                arm = f"{scope}_{title}"
                heads[arm] = weights
                result[arm] = {**stats, "tokens":len(labels), "classes":np.bincount(labels).tolist()}
    if step == "calibration":
        for codec in CODECS:
            points[codec] = {}
            for arm in ARMS:
                maxima = defaultdict(lambda: -np.inf)
                for item in rows:
                    if item["delivery"] == codec and item["model"][arm]["score"] is not None:
                        maxima[item["source"]] = max(maxima[item["source"]], item["model"][arm]["score"])
                if len(maxima) != 8:
                    raise ValueError("Incomplete ordinary calibration")
                points[codec][arm] = sorted(maxima.values())[-2]
    if step == "held":
        for item in rows:
            for arm, loc in item["model"].items():
                loc["flagged"] = bool(loc["box"]) if arm == "current_luma" else (
                    loc["score"] is not None and loc["score"] > points[item["delivery"]][arm])
    result.update(heads={arm:value.tolist() for arm,value in heads.items()}, points=points,
                  views_measured=n, seconds=round(time.perf_counter()-began, 3),
                  maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(output, result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("freeze", *PHASES))
    ap.add_argument("root",type=Path)
    ap.add_argument("plan",type=Path)
    ap.add_argument("--out",type=Path)
    ap.add_argument("--prior",type=Path)
    args = ap.parse_args()
    if args.phase == "freeze":
        plan = frozen(args.root)
        write(args.plan, plan)
        print(dict(Counter(info["phase"] for info in plan["cases"].values())), flush=True)
    else:
        if args.out is None or args.phase in ("calibration", "held") and args.prior is None:
            ap.error("measurement needs --out and calibration/held need --prior")
        measure(args.root, args.plan, args.out, args.phase, args.prior)


if __name__ == "__main__":
    main()
