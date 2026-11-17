# SPDX-License-Identifier: Apache-2.0
"""Fit image-matched JPEG boundary features and freeze the native region comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from app.bench import image_jpeg_boundary as review
from app.detectors.image import jpeg_boundary_localizer as detector
from scripts import image_cross_quality_fit as geometry
from scripts import image_jpeg_band_transfer as previous
from scripts import image_region_rank_inputs as files
from scripts.image_region_rank_compare import incumbent

SEED, EPOCHS, BATCH, PATCH = 170643, 13, 12, 30
CLASSICAL = ("telea_fill", "navier_stokes_fill")


def groups(root, journal, *, families=None):
    manifest = files.read(root / "manifest.json")
    baseline = None if journal is None else incumbent(journal)[1]
    deliveries = defaultdict(dict)
    for item in manifest["images"]:
        key = item["role"], item["quality"]
        if key in deliveries[item["group"]]:
            raise ValueError("Duplicate JPEG delivery")
        deliveries[item["group"]][key] = item
    for group in manifest["groups"]:
        if families is not None and group["family"] not in families:
            continue
        cases = deliveries[group["tag"]]
        roles = CLASSICAL if group["family"] == "classical_fill" else (*files.POSITIVE, *files.CONTROLS)
        if set(cases) != {(role, q) for role in roles for q in review.QUALITIES}:
            raise ValueError("Incomplete matched JPEG deliveries")
        yield root, group, cases, baseline


def read_case(root, row):
    data = (root / row["file"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError("Delivered JPEG differs from its manifest")
    return detector.features(data)


def label_maps(truth, shape):
    density = previous.fraction(truth, shape)
    raw = (density >= 0.12).astype(np.uint8)
    contour = cv2.morphologyEx(raw, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
    return np.stack((np.where(raw, 0.4 + 0.6 * density, 0), contour)).astype(np.float32)


def learning_pairs(rows):
    cache, packed = {}, []

    def reuse(path, record):
        key = path / record["file"]
        if key not in cache:
            cache[key] = read_case(path, record).astype(np.float16)
        return cache[key]

    for root, group, cases, _, extra in rows:
        for q in review.QUALITIES:
            controls = [reuse(root, cases[role, q]) for role in files.CONTROLS]
            if extra:
                source, _, records, _ = extra
                controls.extend(reuse(source, records[role, q]) for role in CLASSICAL)
            for role in files.POSITIVE:
                record = cases[role, q]
                sample = reuse(root, record)
                truth = geometry.load_mask(root, record)
                labels = label_maps(truth, sample.shape[1:])
                if (
                    any(item.shape != sample.shape for item in controls)
                    or tuple(group["shape"]) != truth.shape
                ):
                    raise ValueError("The matched training crops use different source grids")
                packed.append((sample, controls, labels, np.argwhere(labels[0] > 0.1)))
    return packed


def patch(record, random, index):
    edit, controls, labels, positions = record
    height, width = labels.shape[1:]
    if len(positions) and index % 4 != 3:
        center = positions[random.integers(len(positions))]
        y, x = np.clip(
            center + random.integers(-7, 8, size=2) - PATCH // 2, 0, (height - PATCH, width - PATCH)
        )
    else:
        y, x = random.integers((height - PATCH + 1, width - PATCH + 1))
    section = (slice(None), slice(y, y + PATCH), slice(x, x + PATCH))
    other = controls[random.integers(len(controls))]
    return edit[section], other[section], labels[section]


def _balanced(logits, label, importance=None):
    positive = label > 0.1
    other = (~positive).to(torch.float32)
    if importance is not None:
        other = other * importance
    weights = 0.52 * positive / positive.sum().clamp(min=1) + 0.48 * other / other.sum().clamp(min=1)
    return F.binary_cross_entropy_with_logits(logits, label, weight=weights, reduction="sum")


def train(examples, output, *, paired):
    torch.manual_seed(SEED)
    random = np.random.default_rng(SEED)
    model = detector.model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.006)
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        order = random.permutation(len(examples))
        for start in range(0, len(order), BATCH):
            chosen = order[start : start + BATCH]
            positive, ordinary, target = zip(
                *(patch(examples[i], random, int(i) + epoch) for i in chosen), strict=True
            )
            inputs = torch.from_numpy(np.asarray((*positive, *ordinary), dtype=np.float32))
            targets = torch.from_numpy(np.stack(target))
            labels = torch.cat((targets, torch.zeros_like(targets)))
            logits = model(inputs)
            loss = _balanced(logits[:, 0], labels[:, 0])
            if paired:
                # Give authentic strong edges more weight when learning a forensic contour.
                ordinary_edge = 1.0 + 2.0 * (inputs[:, -2] + inputs[:, -7]).clamp(0.0, 2.0)
                loss = loss + 0.55 * _balanced(logits[:, 1], labels[:, 1], ordinary_edge)
                edited, matching = logits.chunk(2)
                active = (targets > 0.1).to(torch.float32)
                joined = F.softplus(1.0 + matching - edited) * active
                loss = loss + 0.40 * joined.sum() / active.sum().clamp(min=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(chosen)
        losses.append(round(total / len(examples), 7))
        print(f"{'paired' if paired else 'plain'} epoch {epoch + 1}/{EPOCHS}, loss {losses[-1]}", flush=True)
    save_file(model.state_dict(), output)
    return model.eval(), losses


def measure(rows, plain, paired):
    outputs = []
    with torch.inference_mode():
        for index, (root, group, cases, baseline) in enumerate(rows, 1):
            for record in cases.values():
                reference = baseline[record["file"]] if baseline is not None else None
                if reference and reference["sha256"] != record["sha256"]:
                    raise ValueError("The existing check assessed different JPEG bytes")
                current = {k: reference[k] for k in ("jpeg", "regional")} if reference else {}
                truth = geometry.load_mask(root, record)
                value = torch.from_numpy(read_case(root, record)[None])
                p = plain(value).numpy()[0, 0]
                response, contour = paired(value).numpy()[0]
                checks = {}
                for name, args in zip(review.ARMS, ((p,), (response,), (response, contour)), strict=True):
                    score, mask = detector.proposals(*args)
                    checks[name] = {"score": score, **previous.region(mask, truth)}
                outputs.append(
                    {k: record[k] for k in ("file", "sha256", "quality", "role", "mask_sha256")}
                    | {
                        "family": group["family"],
                        "source_group": group["rgb_sha256"],
                        "shape": group["shape"],
                        "current": current,
                        "checks": checks,
                    }
                )
            if index % 8 == 0:
                print(f"Measured {index}/{len(rows)} source/delivery groups", flush=True)
    return outputs


def method():
    return {Path(path).name: files.sha(Path(path)) for path in (__file__, detector.__file__, review.__file__)}


def development(args):
    output = args.output
    output.mkdir()
    earlier_fit, earlier_calibration = previous.partition([(Path(a), Path(b)) for a, b in args.input])
    old, extras = [], {}
    for region_size in ("full", "partial"):
        original = args.previous / ("holdout" if region_size == "full" else "partial")
        log = args.previous / (
            "holdout-reference.jsonl" if region_size == "full" else "partial-reference-complete.jsonl"
        )
        additions = list(groups(args.controls / f"old-ordinary-{region_size}", None))
        extras[original] = {item[1]["rgb_sha256"]: item for item in additions}
        old.extend(groups(original, log, families={"lama"}))
    keys = sorted(
        {row[1]["rgb_sha256"] for row in old},
        key=lambda key: hashlib.sha256(f"boundary/{key}".encode()).hexdigest(),
    )
    calibration_keys = set(keys[:8])
    fitting = earlier_fit + [row for row in old if row[1]["rgb_sha256"] not in calibration_keys]
    calibration = earlier_calibration + [row for row in old if row[1]["rgb_sha256"] in calibration_keys]
    fit_keys, check_keys = ({g["rgb_sha256"] for _, g, _, _ in rows} for rows in (fitting, calibration))
    if fit_keys & check_keys or len(keys) != 24:
        raise ValueError("An independent photo cannot be used for fitting and selection")
    print(f"Pinned {len(fit_keys)} fitting and {len(check_keys)} calibration photos", flush=True)
    examples = learning_pairs(
        [(*item, extras.get(item[0], {}).get(item[1]["rgb_sha256"])) for item in fitting]
    )
    print(f"Prepared {len(examples)} matched training pairs", flush=True)
    plain, l0 = train(examples, output / "plain.safetensors", paired=False)
    paired, l1 = train(examples, output / "paired.safetensors", paired=True)
    del examples
    calibration.extend(
        item for source in extras.values() for key, item in source.items() if key in calibration_keys
    )
    rows = measure(calibration, plain, paired)
    points, chosen = review.freeze(rows)
    masks = sorted({files.sha(root / "manifest.json") for root, _, _, _ in (*fitting, *calibration)})
    result = {
        "schema": review.SCHEMA,
        "freeze": None,
        "source_method": method(),
        "used_sources": sorted(fit_keys | check_keys),
        "fit_sources": sorted(fit_keys),
        "selection_sources": sorted(check_keys),
        "model_pins": {name: files.sha(output / f"{name}.safetensors") for name in ("plain", "paired")},
        "source_manifests": masks,
        "fit_losses": {"plain": l0, "paired": l1},
        "rows": rows,
        "points": points,
        "chosen": chosen,
        "outcomes": review.summarize(rows, points, chosen),
    }
    files.save(output / "selection.json.gz", result)
    print(json.dumps(chosen, sort_keys=True), flush=True)


def assess(args):
    lock = args.fit / "selection.json.gz"
    saved = review.read(lock)
    if saved["source_method"] != method():
        raise ValueError("The held method differs from the calibration freeze")
    models = []
    for name, digest in saved["model_pins"].items():
        path = args.fit / f"{name}.safetensors"
        if files.sha(path) != digest:
            raise ValueError("The detector weights differ from the calibration freeze")
        model = detector.model()
        model.load_state_dict(load_file(path))
        models.append((name, model.eval()))
    panels = [Path(p) for p, _ in args.input]
    groups_list = [item for p, j in args.input for item in groups(Path(p), Path(j))]
    if {item[1]["rgb_sha256"] for item in groups_list} & set(saved["used_sources"]):
        raise ValueError("A held original was used for development")
    rows = measure(groups_list, **dict(models))
    value = {key: saved[key] for key in ("schema", "source_method", "model_pins", "points", "chosen")}
    value.update(
        freeze=files.sha(lock),
        rows=rows,
        source_manifests=[files.sha(p / "manifest.json") for p in panels],
        outcomes=review.summarize(rows, saved["points"], saved["chosen"]),
    )
    files.save(args.output, value)
    print(
        json.dumps({key: item["selected"] for key, item in value["outcomes"].items()}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    command = argparse.ArgumentParser(description=__doc__)
    steps = command.add_subparsers(dest="step", required=True)
    fit = steps.add_parser("fit")
    fit.add_argument("previous", type=Path)
    fit.add_argument("controls", type=Path)
    held = steps.add_parser("held")
    held.add_argument("fit", type=Path)
    for parser in (fit, held):
        parser.add_argument("output", type=Path)
        parser.add_argument(
            "--input",
            action="append",
            nargs=2,
            metavar=("JPEG_DIRECTORY", "REFERENCE_JOURNAL"),
            required=True,
        )
    args = command.parse_args()
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    (development if args.step == "fit" else assess)(args)
