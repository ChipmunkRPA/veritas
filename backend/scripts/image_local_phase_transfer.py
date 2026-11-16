# SPDX-License-Identifier: MIT
"""Compare localized Fourier-amplitude and phase evidence on real image edits."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import image_regional_lossy_webp_transfer as sources

SCHEMA = "chipmunk.image-local-phase-transfer/1"
BANKS = ("tgif-pairs-v2", "tgif-confirmation-v4", "tgif-confirmation-v5")
ROLES = ("fit", "point", "held")
DELIVERIES = ("png", "webp90")
ARMS = ("amplitude", "phase", "combined")
PATCH, HOP = 32, 16
METHOD = {
    "features": "Float opponent-color high-pass residuals. Measure local Fourier power and normalized luma/chroma "
                "cross-spectrum at two radial scales in 32-pixel Hann windows on a 16-pixel step.",
    "fit": "Eight native images per editor in each of three source banks. Fit a separate weighted logistic model per "
           "format and feature arm. Use exact edit cells, the same cells in three ordinary variants, "
           "and distant exterior cells in the edited file. Every source and class receives equal total weight.",
    "selection": "Fix the cell gate at the 97th percentile of ordinary fitting cells. Rank connected components of "
                 "at least two cells by their 75th percentile score. Use the second highest ordinary-image-group "
                 "maximum as the strict operating point. Keep only the highest-ranked candidate.",
    "transfer": "Use the last source bank only after both fitting and operating-point selection. Compare each held "
                "proposal with exact changed pixels, and count ordinary and misplaced alerts separately.",
    "useful": "Edit recall of at least one half and intersection over union of at least 0.30.",
}


def features(pixels):
    import cv2
    import numpy as np

    rgb = pixels.astype(np.float32)
    red, green, blue = np.moveaxis(rgb, -1, 0)
    channels = np.stack((.299 * red + .587 * green + .114 * blue, red - green, blue - green))
    high = channels - np.stack([cv2.GaussianBlur(channel, (0, 0), 1.15) for channel in channels])
    tiles = np.lib.stride_tricks.sliding_window_view(high, (PATCH, PATCH), axis=(1, 2))[:, ::HOP, ::HOP]
    taper = np.outer(np.hanning(PATCH), np.hanning(PATCH)).astype(np.float32)
    spectrum = np.fft.rfft2(tiles * taper[None, None, None, :, :], axes=(-2, -1))
    upper = np.hypot(np.fft.fftfreq(PATCH)[:, None], np.fft.rfftfreq(PATCH)[None, :])
    log_power, phase = [], []
    for lo, hi in ((.125, .25), (.25, .42)):
        fourier = spectrum[..., (upper >= lo) & (upper < hi)]
        modulus = np.abs(fourier)
        log_power.extend(np.log1p(values.mean(axis=-1)) for values in np.square(modulus))
        for index in (1, 2):
            cross = fourier[0] * np.conj(fourier[index])
            normalized = cross / (np.abs(cross) + 1.0)
            phase.extend((normalized.real.mean(axis=-1), np.abs(normalized.imag).mean(axis=-1),
                          np.abs(normalized.mean(axis=-1))))
    # Control broad illumination and texture intensity in all arms.
    luma = np.lib.stride_tricks.sliding_window_view(channels[0], (PATCH, PATCH))[::HOP, ::HOP]
    shape = luma.shape[:2]
    scene = [luma.mean(axis=(-2, -1)) / 255, luma.std(axis=(-2, -1)) / 255]
    return {"amplitude": np.stack(scene + log_power, axis=-1).reshape(-1, 8).astype(np.float64),
            "phase": np.stack(scene + phase, axis=-1).reshape(-1, 14).astype(np.float64),
            "combined": np.stack(scene + log_power + phase, axis=-1).reshape(-1, 20).astype(np.float64)}, shape


def train(records):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    fits = {}
    for carrier in DELIVERIES:
        for arm in ARMS:
            batches, labels, weights = [], [], []
            for group in (row for row in records if row["fold"] == "fit"):
                variants = group["views"][carrier]
                fraction = group["occupancy"]
                interior = np.flatnonzero(fraction >= .55)
                outside = np.flatnonzero(fraction == 0)
                if not len(interior):
                    continue
                chosen = interior[np.argsort(-fraction[interior], kind="stable")[:64]]
                if len(outside) > len(chosen):
                    outside = outside[(np.arange(len(chosen)) * len(outside)) // len(chosen)]
                for label, view, indices in [(1, "edited", chosen), (0, "edited", outside)] + [
                    (0, view, chosen) for view in sources.ROLES if view != "edited"
                ]:
                    data = variants[view][arm][indices]
                    batches.extend(data)
                    labels.extend([label] * len(data))
                    weights.extend([1 / ((1 if label else 4) * len(data))] * len(data))
            design = np.asarray(batches)
            center, scale = design.mean(axis=0), np.maximum(.00001, design.std(axis=0))
            model = LogisticRegression(C=.2, solver="lbfgs", max_iter=500, tol=1e-8)
            fitted = model.fit((design - center) / scale, labels, sample_weight=weights)
            assert model.n_iter_[0] < model.max_iter
            raw = fitted.decision_function((design - center) / scale)
            gate = float(np.quantile(raw[np.asarray(labels) == 0], .97, method="higher"))
            fits[carrier, arm] = {"center": center, "scale": scale, "beta": fitted.coef_[0], "offset": fitted.intercept_[0],
                                  "cell_gate": gate, "fit_positive_cells": labels.count(1), "fit_negative_cells": labels.count(0)}
    return fits


def select(matrix, shape, size, fit):
    import cv2
    import numpy as np

    response = (((matrix - fit["center"]) / fit["scale"]) @ fit["beta"] + fit["offset"]).reshape(shape)
    count, indices, stats, _ = cv2.connectedComponentsWithStats((response > fit["cell_gate"]).astype(np.uint8), connectivity=4)
    candidates = [(float(np.quantile(response[indices == i], .75)), -i, i) for i in range(1, count)
                  if stats[i, cv2.CC_STAT_AREA] >= 2]
    h, w = size
    mask = np.zeros((h, w), dtype=bool)
    if not candidates:
        return None, mask
    strength, _, index = max(candidates)
    projected = np.repeat(np.repeat(indices == index, HOP, axis=0), HOP, axis=1)
    edge = (PATCH - HOP) // 2
    mask[edge:edge + projected.shape[0], edge:edge + projected.shape[1]] = projected
    return round(strength, 8), mask


def qualify(rows, fits):
    from collections import defaultdict

    for (carrier, arm), fit in fits.items():
        maxima = defaultdict(lambda: float("-inf"))
        for case in (group for group in rows if group["fold"] == "point"):
            for view in sources.ROLES:
                if view == "edited":
                    continue
                score, _ = select(case["views"][carrier][view][arm], case["shape"], case["pixel_shape"], fit)
                maxima[case["source"]["source_id"]] = max(maxima[case["source"]["source_id"]],
                                                           score if score is not None else float("-inf"))
        ranked = sorted(maxima.values())
        assert len(ranked) == 24
        fit["point"] = ranked[-2] if ranked[-2] != float("-inf") else fit["cell_gate"]


def summarize(rows, models):
    results = defaultdict(Counter)
    for row in rows:
        key = row["format"], row["arm"]
        tally = results["/".join(key)]
        positive = row["role"] == "edited"
        point = models["/".join(key)]["point"]
        flag = row["score"] is not None and row["score"] > point
        correct = sources.useful(row["candidate"], row["true_pixels"] if positive else 0)
        tally["edited" if positive else "ordinary"] += 1
        tally["ranked_useful"] += correct
        tally["useful"] += flag and correct
        tally["ordinary_flags"] += flag and not positive
        tally["wrong_edit_flags"] += flag and positive and not correct
    return {key: dict(counts) for key, counts in sorted(results.items())}


def validate(report):
    freeze, models, rows = (report[key] for key in ("freeze", "fit", "rows"))
    assert freeze["schema"] == SCHEMA and freeze["method"] == METHOD
    roster = {row["source_id"]: row for row in freeze["sources"]}
    assert len(roster) == len(freeze["sources"]) == 72
    assert Counter((row["fold"], row["family"]) for row in roster.values()) == {
        (fold, family): 8 for fold in ROLES for family in sources.FAMILIES}
    assert all(row["bank"] == BANKS[ROLES.index(row["fold"])] for row in roster.values())
    assert set(models) == {f"{form}/{arm}" for form in DELIVERIES for arm in ARMS}
    assert all(math.isfinite(fit["point"]) and math.isfinite(fit["cell_gate"]) for fit in models.values())
    cases = set()
    for row in rows:
        anchor = roster[row["source_id"]]
        carrier, arm, role = row["format"], row["arm"], row["role"]
        assert anchor["fold"] == "held" and carrier in DELIVERIES and arm in ARMS and role in sources.ROLES
        assert all(row[key] == value for key, value in anchor.items() if key not in {"fold", "pins"})
        assert row["input_sha256"] == anchor["pins"][f"{carrier}/{role}"]
        cases.add((row["source_id"], carrier, arm, role))
        truth = row["true_pixels"] if role == "edited" else 0
        result = row["candidate"]
        assert row["score"] is None or math.isfinite(row["score"])
        assert bool(result["pixels"]) == (row["score"] is not None)
        assert 0 <= result["overlap"] <= min(truth, result["pixels"]) <= max(truth, result["pixels"]) <= math.prod(row["size"])
    assert len(cases) == len(rows) == 24 * len(DELIVERIES) * len(ARMS) * len(sources.ROLES)
    summary = summarize(rows, models)
    assert summary == report["summary"]
    return summary


def score(source_root, output):
    import io

    import numpy as np
    from PIL import Image

    output.mkdir(mode=0o700)
    records, roster = [], []
    for source, views in sources.inputs(source_root, banks=BANKS, per_family=8):
        fold = ROLES[BANKS.index(source["bank"])]
        truth = views["edited"][1]
        occupancy = np.lib.stride_tricks.sliding_window_view(truth, (PATCH, PATCH))[::HOP, ::HOP].mean(axis=(-2, -1))
        measured, pins = {}, {}
        for carrier in DELIVERIES:
            measured[carrier] = {}
            for name, (pixels, _) in views.items():
                body = sources.encode(pixels, carrier)
                decoded = np.asarray(Image.open(io.BytesIO(body)).convert("RGB"))
                descriptors, shape = features(decoded)
                assert shape == occupancy.shape
                measured[carrier][name] = descriptors
                pins[f"{carrier}/{name}"] = sources.sha(body)
        records.append({"source": source, "fold": fold, "shape": occupancy.shape, "pixel_shape": truth.shape,
                        "truth": truth, "occupancy": occupancy.flatten(), "views": measured, "pins": pins})
        roster.append(source | {"fold": fold, "pins": pins})
        if len(records) % 12 == 0:
            print(f"Measured local features on {len(records)} of 72 photo groups.", flush=True)
    frozen = {"schema": SCHEMA, "method": METHOD, "sources": roster,
              "manifests": {bank: sources.file_sha(source_root / bank / "manifest.json") for bank in BANKS}}
    sources.save(output / "freeze.json", frozen)
    models = train(records)
    qualify(records, models)
    model_pins = {"/".join(key): {name: float(value) if isinstance(value, np.float64)
                                else value.tolist() if isinstance(value, np.ndarray) else value for name, value in fit.items()}
                  for key, fit in models.items()}
    sources.save(output / "fit.json", model_pins)
    results = []
    for group in (row for row in records if row["fold"] == "held"):
        for (carrier, arm), fit in models.items():
            for role, view in group["views"][carrier].items():
                raw, mask = select(view[arm], group["shape"], group["pixel_shape"], fit)
                truth = group["truth"] if role == "edited" else np.zeros(group["pixel_shape"], dtype=bool)
                results.append(group["source"] | {"format": carrier, "arm": arm, "role": role,
                               "input_sha256": group["pins"][f"{carrier}/{role}"], "score": raw,
                               "candidate": sources._bounds(mask, truth)})
    receipt = {"freeze": frozen, "fit": model_pins, "rows": results, "summary": summarize(results, model_pins)}
    validate(receipt)
    sources.save(output / "receipt.json.gz", receipt)
    print(json.dumps(receipt["summary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("score")
    run.add_argument("sources", type=Path)
    run.add_argument("output", type=Path)
    commands.add_parser("review").add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.action == "score":
        score(args.sources, args.output)
    else:
        print(json.dumps(validate(sources.read(args.receipt)), indent=2))
