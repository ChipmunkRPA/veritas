# SPDX-License-Identifier: Apache-2.0
"""Fit independent JPEG frequency bands, freeze selection, and count held locations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import image_cross_quality_fit as shared
from scripts import image_region_rank_compare as rank
from scripts import image_region_rank_inputs as files

SCHEMA = "chipmunk.image-jpeg-band-transfer/1"
ARMS = ("coarse", "middle", "fine", "combined")
SEED = 191013
EPOCHS, PATCH, BATCH = 14, 26, 16
OPERATING_POINTS = (0, 1, 2)
METHOD = {
    "fit": "Train three separate small convolutional responses on signed relationships within separate native JPEG bands. "
           "The two delivery qualities and ordinary versions use the same changed-area training support and paired measures.",
    "location": "Describe the highest supported connected native region per arm. The combined arm averages the logits from the three bands.",
    "calibration": "Compare each branch and the combined arm at three ordinary source-group budgets per quality. "
                   "Prefer correct additions on feathered edits, then correct additions overall, then fewer ordinary and misplaced additions.",
    "selection": "Prefer correctly localized areas on source-separated JPEG75 and JPEG90. Preserve existing successful evidence. "
                 "Qualifying locations require 0.30 IoU and 0.50 recall of exact pre-encode changed pixels.",
}


def models_and_rosters(inputs):
    for study, reference in inputs:
        manifest = files.read(study / "manifest.json")
        _, baseline = rank.incumbent(reference)
        yield study, manifest, baseline


def partition(inputs):
    families = defaultdict(list)
    for study, manifest, reference in models_and_rosters(inputs):
        for group, cases in shared.images(manifest, ("transfer", "fresh")):
            family = group["family"]
            if family not in {"big-lama", "lama", "sd15"} or min(group["shape"]) < 384:
                continue
            families["lama" if family in {"lama", "big-lama"} else family].append((study, group, cases, reference))
    fitting, calibration = [], []
    hashes = [entry[1]["rgb_sha256"] for rows in families.values() for entry in rows]
    if len(hashes) != len(set(hashes)) or len(families) < 2:
        raise ValueError("Need unique, separately attributed groups from both editor families")
    for rows in families.values():
        rows.sort(key=lambda entry: hashlib.sha256(f"{SCHEMA}/{entry[1]['rgb_sha256']}".encode()).hexdigest())
        reserved = math.ceil(.27 * len(rows))
        fitting.extend(rows[reserved:])
        calibration.extend(rows[:reserved])
    return fitting, calibration


def read_coefficients(root, row):
    from app.detectors.image import jpeg_band_localizer as localizer

    body = (root / row["file"]).read_bytes()
    if hashlib.sha256(body).hexdigest() != row["sha256"]:
        raise ValueError("Delivered JPEG differs from the source manifest")
    return localizer.features(body)


def fraction(mask, shape):
    from app.detectors.image.jpeg_band_localizer import CELL

    rows, cols = shape
    return mask[:rows * CELL, :cols * CELL].reshape(rows, CELL, cols, CELL).mean(axis=(1, 3), dtype="float32")


def training_examples(fitting):
    for index, (root, _, cases, _) in enumerate(fitting, 1):
        support = shared.load_mask(root, cases["sharp", 75])
        for role in (*files.POSITIVE, *files.CONTROLS):
            truth = shared.load_mask(root, cases[role, 75]) if role in files.POSITIVE else support & False
            for quality in files.QUALITY:
                bands = tuple(item.astype("float16") for item in read_coefficients(root, cases[role, quality]))
                shape = bands[0].shape[1:]
                yield bands, fraction(truth, shape), fraction(support, shape)
        if index % 8 == 0:
            print(f"Read exact fitting inputs for {index}/{len(fitting)} photos", flush=True)


def crop(bands, targets, support, rng, *, anchored):
    import numpy as np

    height, width = targets.shape
    positions = np.argwhere(support > .10)
    if anchored and positions.size:
        center = positions[rng.integers(len(positions))]
        y, x = np.clip(center + rng.integers(-5, 6, size=2) - PATCH // 2, 0, (height - PATCH, width - PATCH))
    else:
        y, x = rng.integers((height - PATCH + 1, width - PATCH + 1))
    target = targets[y:y + PATCH, x:x + PATCH]
    return tuple(values[:, y:y + PATCH, x:x + PATCH] for values in bands), target


def train(fitting, weights):
    import cv2
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from torch.nn import functional as F

    from app.detectors.image import jpeg_band_localizer as detector

    torch.manual_seed(SEED)
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    assert detector.ARMS == ARMS
    module = detector.model()
    optimizer = torch.optim.AdamW(module.parameters(), lr=.003, weight_decay=.008)
    examples = list(training_examples(fitting))
    rng = np.random.default_rng(SEED)
    losses = []
    for epoch in range(EPOCHS):
        module.train()
        order = rng.permutation(len(examples))
        total = np.zeros(len(detector.BANDS), dtype=np.float64)
        updates = 0
        for start in range(0, len(order), BATCH):
            pairs = [crop(*examples[i], rng, anchored=(epoch % 4 != 3 or i % 2 == 0)) for i in order[start:start + BATCH]]
            labels = torch.from_numpy(np.stack([target for _, target in pairs])[:, None])
            positive = labels > .10
            negative = ~positive
            weights_map = .55 * positive / positive.sum().clamp(min=1) + .45 * negative / negative.sum().clamp(min=1)
            # Mixed pixels retain their measured share, with stronger supervision for the interior.
            labels = torch.where(positive, .40 + .60 * labels, 0.)
            optimizer.zero_grad(set_to_none=True)
            loss = []
            for band, layer in enumerate(module):
                data = torch.from_numpy(np.stack([values[band] for values, _ in pairs]).astype(np.float32))
                value = F.binary_cross_entropy_with_logits(layer(data), labels, weight=weights_map, reduction="sum")
                loss.append(value)
            torch.stack(loss).sum().backward()
            optimizer.step()
            total += np.array([value.item() for value in loss])
            updates += 1
        losses.append([round(float(item / updates), 7) for item in total])
        print(f"Fitted epoch {epoch + 1}/{EPOCHS}, per-band loss {losses[-1]}", flush=True)
    save_file({name: tensor.cpu() for name, tensor in module.state_dict().items()}, weights)
    return module.eval(), losses


def region(mask, truth):
    import numpy as np

    from app.detectors.image.jpeg_band_localizer import CELL

    native = np.repeat(np.repeat(mask, CELL, axis=0), CELL, axis=1)
    y, x = np.where(native)
    height, width = native.shape
    return {"box": [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if x.size else None,
            "selected": int(native.sum()), "intersection": int((truth[:height, :width] & native).sum()),
            "truth": int(truth.sum()), "mask_sha256": hashlib.sha256(native.tobytes()).hexdigest()}


def score(groups, module):
    import cv2
    import numpy as np
    import torch

    from app.detectors.image import jpeg_band_localizer as detector

    cv2.setNumThreads(1)
    measured = []
    with torch.inference_mode():
        for index, (root, group, cases, baseline) in enumerate(groups, 1):
            for row in cases.values():
                original = baseline[row["file"]]
                if original["sha256"] != row["sha256"]:
                    raise ValueError("Incumbent scores refer to different bytes")
                bands = read_coefficients(root, row)
                heat = [head(torch.from_numpy(values[None])).numpy()[0, 0] for head, values in zip(module, bands, strict=True)]
                heat.append(np.mean(heat, axis=0))
                truth = shared.load_mask(root, row)
                checks = {}
                for name, plane in zip(ARMS, heat, strict=True):
                    value, mask = detector.strongest(plane)
                    checks[name] = {"score": value, **region(mask, truth)}
                measured.append({key: row[key] for key in ("file", "sha256", "quality", "role", "group", "mask_sha256")} |
                    {"family": group["family"], "source_group": group["rgb_sha256"], "shape": group["shape"],
                     "current": {key: original[key] for key in ("jpeg", "regional")}, "checks": checks})
            if index % 8 == 0:
                print(f"Compared {index}/{len(groups)} photo/delivery groups", flush=True)
    return measured


def outcome(rows, arm, point):
    counts = Counter()
    ordinary_groups = set()
    for row in rows:
        positive = row["role"] in files.POSITIVE
        existing = rank.current_flags(row)
        useful = positive and any(map(rank.useful, row["current"].values()))
        check = row["checks"][arm]
        reported = check["score"] is not None and check["score"] > point
        new = reported and not existing
        located = positive and rank.useful(check)
        counts.update(deliveries=1, edits=positive, ordinary=not positive, baseline_correct=useful,
                      baseline_ordinary=not positive and existing, top_correct=located,
                      standalone_correct=located and reported, added_correct=located and new,
                      added_feather=located and new and row["role"] == "feather", added_wrong=positive and new and not located,
                      added_ordinary=not positive and new)
        if new and not positive:
            ordinary_groups.add(row["source_group"])
    return dict(counts) | {"added_ordinary_groups": len(ordinary_groups)}


def selection(calibration):
    selected = {}
    for quality in files.QUALITY:
        rows = [row for row in calibration if row["quality"] == quality]
        options = []
        for arm in ARMS:
            groups = defaultdict(lambda: -1e6)
            for row in rows:
                if row["role"] in files.CONTROLS and not rank.current_flags(row):
                    value = row["checks"][arm]["score"]
                    if value is not None:
                        key = row["source_group"]
                        groups[key] = max(groups[key], value)
            maxima = sorted(groups.values(), reverse=True)
            if len(maxima) < 3:
                raise ValueError("No usable controls for the JPEG operating point")
            for budget in OPERATING_POINTS:
                point = maxima[budget]
                counts = outcome(rows, arm, point)
                order = (+counts["added_feather"], counts["added_correct"], -counts["added_ordinary_groups"],
                         -counts["added_wrong"], -budget, -ARMS.index(arm))
                options.append((order, {"arm": arm, "point": point, "budget": budget, "outcome": counts}))
        selected[str(quality)] = max(options, key=lambda choice: choice[0])[1]
    return selected


def implementation():
    model_file = Path(__file__).resolve().parents[1] / "app/detectors/image/jpeg_band_localizer.py"
    return {path.name: files.sha(path) for path in (model_file, Path(__file__))}


def freeze(inputs, output):
    output.mkdir(mode=0o700)
    fitting, calibration = partition(inputs)
    weights = output / "bands.safetensors"
    module, losses = train(fitting, weights)
    rows = score(calibration, module)
    selected = selection(rows)
    record = {"schema": SCHEMA, "method": METHOD, "code": implementation(), "weights": files.sha(weights),
              "manifests": [files.sha(media / "manifest.json") for media, _ in inputs],
              "fitting": [group["rgb_sha256"] for _, group, _, _ in fitting],
              "calibration": rows, "selected": selected, "epoch_loss": losses}
    files.save(output / "freeze.json.gz", record)
    print(json.dumps(selected, indent=2), flush=True)


def held_counts(rows, chosen):
    all_results = {}
    for quality in files.QUALITY:
        selected = chosen[str(quality)]
        for family in sorted({row["family"] for row in rows}):
            subset = [row for row in rows if row["quality"] == quality and row["family"] == family]
            all_results[f"{quality}/{family}"] = outcome(subset, selected["arm"], selected["point"])
    return all_results


def assess(locked, media, reference, output):
    import torch
    from safetensors.torch import load_file

    from app.detectors.image.jpeg_band_localizer import model

    freeze_path = locked / "freeze.json.gz"
    packet = files.read(freeze_path)
    validate(packet)
    if packet["code"] != implementation() or files.sha(locked / "bands.safetensors") != packet["weights"]:
        raise ValueError("The method or weights differ from the frozen calibration")
    study, manifest, ref = next(models_and_rosters(((media, reference),)))
    groups = [(study, group, cases, ref) for group, cases in shared.images(manifest, ("fresh", "confirmation"))]
    exposed = set(packet["fitting"]) | {row["source_group"] for row in packet["calibration"]}
    if not groups or exposed & {group["rgb_sha256"] for _, group, _, _ in groups}:
        raise ValueError("The confirmation sources overlap development or are empty")
    torch.set_num_threads(3)
    module = model().eval()
    module.load_state_dict(load_file(locked / "bands.safetensors"), strict=True)
    rows = score(groups, module)
    result = held_counts(rows, packet["selected"])
    files.save(output, {"schema": SCHEMA, "frozen": files.sha(freeze_path), "manifest": files.sha(study / "manifest.json"),
                        "selected": packet["selected"], "rows": rows, "summary": result})
    print(json.dumps(result, indent=2), flush=True)


def validate(packet):
    if packet["schema"] != SCHEMA:
        raise ValueError("Unknown image confirmation receipt")
    if "calibration" in packet:
        assert selection(packet["calibration"]) == packet["selected"]
        assert not set(packet["fitting"]) & {row["source_group"] for row in packet["calibration"]}
        return packet["selected"]
    assert held_counts(packet["rows"], packet["selected"]) == packet["summary"]
    return packet["summary"]


if __name__ == "__main__":
    command = argparse.ArgumentParser(description=__doc__)
    modes = command.add_subparsers(dest="step", required=True)
    fit = modes.add_parser("freeze")
    fit.add_argument("output", type=Path)
    fit.add_argument("--inputs", nargs=2, type=Path, action="append", required=True)
    held = modes.add_parser("assess")
    for field in ("locked", "media", "reference", "output"):
        held.add_argument(field, type=Path)
    review = modes.add_parser("review")
    review.add_argument("receipt", type=Path)
    values = vars(command.parse_args())
    step = values.pop("step")
    if step == "review":
        print(json.dumps(validate(files.read(values["receipt"])), indent=2))
    else:
        {"freeze": freeze, "assess": assess}[step](**values)
