# SPDX-License-Identifier: Apache-2.0
"""Test source-paired image-label supervision for local JPEG edit proposals."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from scripts import image_vae_double_panel as panel
from scripts.image_reconstruction_phase_transfer import WEIGHTS, WEIGHTS_SHA256, phase_padding
from scripts.image_region_rank_features import CELL, FrozenEncoder
from scripts.image_three_class_patch_probe import proposal
from scripts.review_image_residual_encoder_transfer import overlap

ARM = ("appearance", "reconstruction", "joint")
PAIRS = (("original_outside", "pristine"), ("vae_outside", "global_vae"))
TEMPERATURE = .35
SEED = 20261118
ADVANCE = {"minimum_locations": 4, "minimum_source_groups": 3, "maximum_added_control_flags": 4,
           "maximum_added_control_groups": 3}


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def dump(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def read(path):
    return json.loads(path.read_bytes())


def plan(sources, weights, output):
    original = panel.frozen(sources)
    rows = {key: {field: value for field, value in case.items() if field != "fit_sites"}
            for key, case in original["cases"].items()}
    config, model = weights / "config.json", weights / WEIGHTS
    if sha(model.read_bytes()) != WEIGHTS_SHA256:
        raise ValueError("The reconstruction checkpoint differs")
    dump(output, {"cases": rows, "encoder": original["model"],
                  "reconstructor": {"config": sha(config.read_bytes()), "weights": WEIGHTS_SHA256},
                  "versions": original["versions"], "implementation": sha(Path(__file__).read_bytes()),
                  "paired_image_labels": PAIRS, "advance": ADVANCE, "seed": SEED,
                  "calibration": "Strictly exceed the second-highest ordinary source maximum per delivery and arm."})


class FeatureViews:
    def __init__(self, weights, lock):
        from diffusers import AutoencoderTiny

        if lock["encoder"] != panel.checkpoint() or lock["reconstructor"] != {
            "config": sha((weights / "config.json").read_bytes()), "weights": sha((weights / WEIGHTS).read_bytes()),
        }:
            raise ValueError("A licensed input asset changed before extraction")
        self.encoder = FrozenEncoder("cpu")
        self.reconstruction = AutoencoderTiny.from_pretrained(weights, local_files_only=True, use_safetensors=True).eval()

    @torch.inference_mode()
    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        pixels = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)[None]) / 127.5 - 1
        pads = phase_padding((h, w), (0, 0))
        left, _, top, _ = pads
        native = F.pad(pixels, pads, mode="reflect")
        decoded = self.reconstruction.decode(self.reconstruction.encode(native).latents).sample
        if decoded.shape != native.shape:
            raise ValueError("The reconstruction changed the original coordinate grid")
        decoded = decoded[:, :, top:top + h, left:left + w].clamp(-1, 1)

        def pool(x, size=CELL):
            return F.avg_pool2d(x, size, stride=CELL)

        error = (decoded - pixels).abs()
        energy = pool(error)
        detail = pool((pixels - F.avg_pool2d(F.pad(pixels, (1, 1, 1, 1), mode="reflect"), 3, 1)).abs())
        normalized = torch.log((energy + .004) / (detail + .010))
        within = F.max_pool2d(error, CELL) / (energy + .004)
        combined = torch.cat((torch.log(energy + .004), normalized, within), dim=1)
        residuals = combined[0].permute(1, 2, 0).numpy()
        appearance = self.encoder(rgb)
        if appearance.shape[:2] != residuals.shape[:2]:
            raise ValueError("Visual and reconstruction features disagree on native cells")
        result = {"appearance": appearance, "reconstruction": residuals}
        for name, features in result.items():
            result[name] = features - np.median(features, axis=(0, 1), keepdims=True)
        result["joint"] = np.concatenate([result[name] for name in ARM[:2]], axis=2)
        return result


def inputs(sources, lock, phase):
    for split, source, materials, truth, _ in panel.processing(sources):
        if split != phase:
            continue
        expected = lock["cases"][source["recipient"]]
        if expected["phase"] != split or expected["info"] != source:
            raise ValueError("The source file or held split differs")
        for codec in panel.CODECS:
            scope = tuple(materials) if phase != "calibration" else panel.NEGATIVES
            for name in scope:
                encoded, parent = panel.encode(materials[name], codec)
                identity = sha(encoded)
                if [identity, parent] != expected["deliveries"][codec + "/" + name]:
                    raise ValueError("JPEG delivery differs from the input plan")
                decoded = np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB"))
                yield (source["recipient"], source["family"], codec, name, identity), decoded, truth


def fit_head(examples):
    raw = np.concatenate([array.reshape(-1, array.shape[-1]) for array in examples.values()])
    scale = np.maximum(np.sqrt(np.mean(raw ** 2, axis=0)), .005)
    groups = defaultdict(dict)
    for (source, codec, name), array in examples.items():
        groups[source, codec][name] = torch.from_numpy(np.clip(array / scale, -8, 8).astype(np.float32))
    weights = torch.zeros(scale.shape, requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam((weights, bias), lr=.028)
    for _ in range(75):
        total = []
        for group in groups.values():
            summaries = {}
            for name, array in group.items():
                field = (array @ weights + bias)[None, None]
                values = F.avg_pool2d(field, 2, stride=1).flatten()
                summaries[name] = TEMPERATURE * (torch.logsumexp(values / TEMPERATURE, dim=0) - np.log(values.numel()))
            for pos, neg in PAIRS:
                total.append(F.softplus(-summaries[pos]) + F.softplus(summaries[neg])
                             + F.softplus(1 - (summaries[pos] - summaries[neg])))
        loss = torch.stack(total).mean() + .008 * (weights.square().sum())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {"scale": scale.tolist(), "weights": weights.detach().tolist(), "bias": float(bias.detach()), "loss": float(loss.detach())}


def train(sources, lock, model):
    fit = defaultdict(dict)
    for idx, ((source, _, codec, name, _), rgb, _truth) in enumerate(inputs(sources, lock, "fit"), 1):
        # Only whole-image labels enter the fit. Pixel annotations are evaluated later.
        for arm, local in model(rgb).items():
            fit[arm][source, codec, name] = local
        if idx % 8 == 0:
            print("fit", idx, flush=True)
    return {arm: fit_head(fit[arm]) for arm in ARM}


def positions(sources, lock, model, phase, heads):
    rows = []
    for idx, ((source, family, codec, name, identity), rgb, truth) in enumerate(inputs(sources, lock, phase), 1):
        readings = {}
        for arm, features in model(rgb).items():
            head = heads[arm]
            field = np.clip(features / np.asarray(head["scale"]), -8, 8) @ head["weights"] + head["bias"]
            region = proposal(field)
            mask = truth if name in panel.POSITIVES else np.zeros_like(truth)
            readings[arm] = region | overlap(mask, region["box"])
        rows.append({"source": source, "family": family, "delivery": codec, "view": name,
                     "sha256": identity, "model": readings})
        if idx % 8 == 0:
            print(phase, idx, flush=True)
    return rows


def cutoffs(rows):
    values = defaultdict(dict)
    for row in rows:
        for arm, reading in row["model"].items():
            by_source = values[row["delivery"], arm]
            by_source[row["source"]] = max(reading["score"], by_source.get(row["source"], -np.inf))
    if not all(len(groups) == 8 for groups in values.values()) or len(values) != len(panel.CODECS) * len(ARM):
        raise ValueError("The ordinary calibration rows are incomplete")
    return {codec: {arm: sorted(values[codec, arm].values())[-2] for arm in ARM} for codec in panel.CODECS}


def decision(rows, baseline, cut):
    old = {(r["source"], r["delivery"], r["view"], r["sha256"]): r for r in baseline["rows"]}
    if set(old) != {(r["source"], r["delivery"], r["view"], r["sha256"]) for r in rows}:
        raise ValueError("Historical outputs belong to a different JPEG cohort")
    counts = defaultdict(Counter)
    groups = defaultdict(set)
    for row in rows:
        key = (row["source"], row["delivery"], row["view"], row["sha256"])
        reference = old[key]["model"]["current_luma"]
        positive = row["view"] in panel.POSITIVES
        for arm, item in row["model"].items():
            found = item["score"] > cut[row["delivery"]][arm]
            useful = positive and item["iou"] >= .3 and item["recall"] >= .5
            prior = positive and reference["flagged"] and reference["iou"] >= .3 and reference["recall"] >= .5
            item.update(flagged=found, useful=found and useful)
            count = counts[row["delivery"], arm]
            count.update(positives=positive, controls=not positive, location_before_cutoff=useful,
                         useful=found and useful, additional=found and useful and not prior,
                         ordinary=not positive and found, new_ordinary=not positive and found and not reference["flagged"],
                         misplaced=positive and found and not useful)
            if found and useful and not prior:
                groups[row["delivery"], arm, "useful"].add(row["source"])
            if found and not positive and not reference["flagged"]:
                groups[row["delivery"], arm, "ordinary"].add(row["source"])
    output = {}
    for (codec, arm), items in counts.items():
        output[f"{codec}/{arm}"] = dict(items, source_groups=len(groups[codec, arm, "useful"]),
                                       added_control_groups=len(groups[codec, arm, "ordinary"]))
    return output


def run(sources, weights, frozen, baseline, out):
    lock = read(frozen)
    if lock["implementation"] != sha(Path(__file__).read_bytes()) or lock["advance"] != ADVANCE:
        raise ValueError("The experimental method freeze differs")
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    model = FeatureViews(weights, lock)
    out.mkdir(mode=0o700)
    heads = train(sources, lock, model)
    dump(out / "fit.json", heads)
    calibration = positions(sources, lock, model, "calibration", heads)
    cuts = cutoffs(calibration)
    dump(out / "calibration.json", {"points": cuts, "rows": calibration, "plan_sha256": sha(frozen.read_bytes())})
    held = positions(sources, lock, model, "held", heads)
    result = decision(held, read(baseline), cuts)
    dump(out / "held.json", {"points": cuts, "rows": held, "counts": result, "plan_sha256": sha(frozen.read_bytes()),
                            "runtime_seconds": round(time.monotonic() - started, 2)})
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    for cmd, names in (("plan", ("sources", "weights", "output")),
                       ("run", ("sources", "weights", "frozen", "baseline", "out"))):
        p = sub.add_parser(cmd)
        for name in names:
            p.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    {"plan": plan, "run": run}[args.pop("command")](**args)
