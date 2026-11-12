# SPDX-License-Identifier: Apache-2.0
"""Compare content-matched and spatial references for native fine-detail changes."""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from app.detectors.image import sharp_boundary
from scripts import image_native_tile_agreement_transfer as panel

SCHEMA = "chipmunk.image-matched-detail-transfer/1"
CELL = 16
FEATURES = ("luma_detail_ratio", "chroma_detail_ratio", "chroma_luma_balance", "luma_lag_one", "luma_chroma_correlation")
SIGMAS = (0.85, 2.4)
CONTEXT_SCALE_FLOOR = np.array([12., .25, .25, .18])
FEATURE_SCALE_FLOOR = np.array([.16, .16, .20, .06, .06])
SIDES = (3, 4, 6, 8, 10, 12, 16)
AREA_POWER = .18
REMOTE_DISTANCE = 6
LOCAL_MIN_DISTANCE = 2
MATCHED_SUPPORT = 18
MINIMUM_SUPPORT = 8
MAX_MATCH_DISTANCE = 2.5
ARMS = ("spatial", "matched")
CALIBRATION_PANELS = ("cal",)
HELD_PANELS = ("held_lama", "held_generated", "screen")
PANEL_COUNTS = {"cal": 32, "held_lama": 72, "held_generated": 30, "screen": 36}
BASELINES = ("sharp_boundary", "research_baseline", "previous_tile_agreement")
GATE = {"minimum_added_primary_feathered": 2, "minimum_added_other_feathered": 1, "maximum_lost_useful": 1,
        "maximum_extra_photo_groups": 2, "maximum_extra_screen_groups": 1}
DESIGN = {"cell": CELL, "features": list(FEATURES), "gaussian_sigmas": list(SIGMAS), "context_scale_floor": CONTEXT_SCALE_FLOOR.tolist(),
          "feature_scale_floor": FEATURE_SCALE_FLOOR.tolist(), "support": [MINIMUM_SUPPORT, MATCHED_SUPPORT],
          "spatial_exclusion": LOCAL_MIN_DISTANCE, "matched_exclusion": REMOTE_DISTANCE,
          "maximum_context_distance": MAX_MATCH_DISTANCE, "rectangle_side_cells": list(SIDES), "area_power": AREA_POWER,
          "eligibility": "one upright RGB PNG, short side at least 384, long side at most 768",
          "reference": "spatial nearest cells or distant cells matched on coarse luma, variation, gradient and direction",
          "statistic": "signed standardized residual by feature; same-sign regional mean less opposing surround agreement",
          "point": "strictly exceed the second-highest ordinary calibration group maximum",
          "useful": {"iou": .30, "recall": .50}, "confidence_weight": 0, "gate": GATE}
require, read, write, sha, file_sha = panel.require, panel.read, panel.write, panel.sha, panel.file_sha


def windows(values):
    h, w = (n // CELL for n in values.shape)
    return values[:h * CELL, :w * CELL].reshape(h, CELL, w, CELL).transpose(0, 2, 1, 3)


def centered_covariance(a, b):
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    b = b - b.mean(axis=(-2, -1), keepdims=True)
    denom = np.sqrt(np.mean(a * a, axis=(-2, -1)) * np.mean(b * b, axis=(-2, -1)))
    return np.mean(a * b, axis=(-2, -1)) / np.maximum(denom, .16)


def measurements(rgb):
    r, g, b = np.moveaxis(rgb.astype(np.float32), -1, 0)
    y, u, v = .299 * r + .587 * g + .114 * b, r - g, b - g

    def smooth(data, sigma):
        return cv2.GaussianBlur(data, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT_101)

    def bands(data):
        fine = smooth(data, SIGMAS[0])
        coarse = smooth(data, SIGMAS[1])
        return windows(data - fine), windows(fine - coarse), windows(coarse)

    yh, ym, yc = bands(y)
    uh, um, _ = bands(u)
    vh, vm, _ = bands(v)
    sy, sx = np.abs(np.diff(yc, axis=-2)).mean(axis=(-2, -1)), np.abs(np.diff(yc, axis=-1)).mean(axis=(-2, -1))
    context = np.stack((yc.mean(axis=(-2, -1)), np.log(yc.std(axis=(-2, -1)) + .75),
                        np.log(sx + sy + .2), (sx - sy) / (sx + sy + .2)), axis=-1)

    def power(tile):
        return np.mean(np.minimum(np.abs(tile), 15.) ** 2, axis=(-2, -1)) + .16

    py, pm, pc, pcm = power(yh), power(ym), (power(uh) + power(vh)) / 2, (power(um) + power(vm)) / 2
    lag = (centered_covariance(yh[..., :-1], yh[..., 1:])
           + centered_covariance(yh[..., :-1, :], yh[..., 1:, :])) / 2
    cross = (np.abs(centered_covariance(yh, uh)) + np.abs(centered_covariance(yh, vh))) / 2
    features = np.stack((np.log(py / pm), np.log(pc / pcm), np.log(pc / py), lag, cross), axis=-1)
    clip = np.mean((windows(y) < 3) | (windows(y) > 252), axis=(-2, -1))
    return context, features, clip < .15


def references(context, features, valid):
    height, width = valid.shape
    indices = np.flatnonzero(valid.ravel())
    output = np.full((2, height, width, len(FEATURES)), np.nan, dtype=np.float64)
    if len(indices) <= MATCHED_SUPPORT:
        return output
    rowcol = np.column_stack(np.unravel_index(indices, valid.shape))
    location = cdist(rowcol, rowcol, "chebyshev")
    active = context.reshape(-1, context.shape[-1])[indices]
    scale = np.maximum(np.diff(np.percentile(active, (25, 75), axis=0), axis=0)[0], CONTEXT_SCALE_FLOOR)
    distance = cdist(active / scale, active / scale)
    tie_order = np.arange(len(indices)) * (1e-9 / len(indices))
    spatial = np.where(location > LOCAL_MIN_DISTANCE, location + tie_order, np.inf)
    matched = np.where((location > REMOTE_DISTANCE) & (distance <= MAX_MATCH_DISTANCE), distance + tie_order, np.inf)
    chosen = features.reshape(-1, len(FEATURES))[indices]
    for index, values in enumerate((spatial, matched)):
        near = np.argpartition(values, MATCHED_SUPPORT - 1, axis=1)[:, :MATCHED_SUPPORT]
        mask = np.take_along_axis(values, near, axis=1) < np.inf
        flat = output[index].reshape(-1, len(FEATURES))
        for i in np.flatnonzero(mask.sum(axis=1) >= MINIMUM_SUPPORT):
            refs = chosen[near[i, mask[i]]]
            center = np.median(refs, axis=0)
            spread = np.median(np.abs(refs - center), axis=0) * 1.4826
            flat[indices[i]] = np.clip((chosen[i] - center) / np.maximum(spread, FEATURE_SCALE_FLOOR), -5., 5.)
    return output


def integral(values):
    return np.pad(values.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)) + ((0, 0),) * (values.ndim - 2))


def rectangle(table, top, left, bottom, right):
    return table[bottom, right] - table[top, right] - table[bottom, left] + table[top, left]


def shapes(height, width):
    for h in SIDES:
        for w in SIDES:
            if max(h, w) <= 2 * min(h, w) and h + 2 <= height and w + 2 <= width:
                top, left = np.mgrid[1:height - h, 1:width - w]
                yield top, left, top + h, left + w, h * w


def select(scores):
    height, width, channels = scores.shape
    # Positive and negative deviations require spatial agreement in the same feature.
    values = np.nan_to_num(scores)
    observed = np.isfinite(scores[..., 0]).astype(np.int64)
    signed = np.concatenate((values, -values), axis=2)
    response, support, count = integral(signed), integral((signed > .75).astype(np.int64)), integral(observed)
    best, box, channel, core, context, agreeing, eligible = None, None, None, None, None, None, None
    for top, left, bottom, right, area in shapes(height, width):
        n = rectangle(count, top, left, bottom, right)
        ring_n = rectangle(count, top - 1, left - 1, bottom + 1, right + 1) - n
        a = rectangle(response, top, left, bottom, right)
        ring = rectangle(response, top - 1, left - 1, bottom + 1, right + 1) - a
        core_scores = a / np.maximum(n[..., None], 1)
        ring_scores = ring / np.maximum(ring_n[..., None], 1)
        agreement = rectangle(support, top, left, bottom, right)
        good = (n[..., None] * 5 >= 4 * area) & (ring_n[..., None] >= 6) & (5 * agreement >= 3 * n[..., None])
        rank = np.where(good, (core_scores - np.maximum(0, ring_scores) / 2) * (area / 9) ** AREA_POWER, -np.inf)
        i, j, k = np.unravel_index(rank.argmax(), rank.shape)
        value = float(rank[i, j, k])
        if np.isfinite(value) and (best is None or value > best):
            best, channel = value, FEATURES[k % channels] + (":higher" if k < channels else ":lower")
            box = [CELL * int(q[i, j]) for q in (left, top, right, bottom)]
            core, context = float(core_scores[i, j, k]), float(ring_scores[i, j, k])
            agreeing, eligible = int(agreement[i, j, k]), int(n[i, j])
    return {"score": best, "box": box, "feature": channel, "core": core, "ring": context,
            "agreeing_cells": agreeing, "eligible_cells": eligible, "assessed_cells": int(observed.sum()),
            "score_map_sha256": sha(np.asarray(scores, dtype="<f8").tobytes())}


def pixels(box, mask):
    if box is None:
        return {"reported": 0, "overlap": 0, "target": int(mask.sum())}
    x, y, right, bottom = box
    return {"reported": (right - x) * (bottom - y), "overlap": int(mask[y:bottom, x:right].sum()), "target": int(mask.sum())}


def useful(values):
    target, overlap, reported = (values[key] for key in ("target", "overlap", "reported"))
    return bool(target and 2 * overlap >= target and 10 * overlap >= 3 * (reported + target - overlap))


def geometry(mask):
    target = int(mask.sum())
    if not target:
        return {"useful_proposals": 0, "best_iou": 0.}
    table = integral(mask.astype(np.int64))
    proposals, best = 0, 0.
    for top, left, bottom, right, area in shapes(*(side // CELL for side in mask.shape)):
        overlap = rectangle(table, top * CELL, left * CELL, bottom * CELL, right * CELL)
        union = area * CELL ** 2 + target - overlap
        proposals += int(((2 * overlap >= target) & (10 * overlap >= 3 * union)).sum())
        best = max(best, float(np.max(overlap / union)))
    return {"useful_proposals": proposals, "best_iou": best}


def defaults():
    local = Path("backend/data/independent-evaluation")
    return local, local / "image-native-tile-agreement-2026-11-12"


def input_args(args):
    return SimpleNamespace(commons=args.commons, primary=args.primary, backup=args.backup,
                           legacy=args.legacy, alternate=args.alternate, previous=args.previous, run=args.tile_run)


def source_paths(args):
    return {"commons": args.commons / "acquisition.json", "primary": args.primary / "qualified-private.json",
            "backup": args.backup / "qualified-private.json", "earlier": args.legacy / "result.json",
            "alternate": args.alternate / "result.json", "layout": args.tile_run / "layout-plan.json",
            "captures": args.tile_run / "captures.json"}


def frozen_panel(args):
    source_lock = read(args.tile_run / "freeze.json")
    prior = read(args.tile_run / "result.json")
    require(prior["freeze_sha256"] == file_sha(args.tile_run / "freeze.json")
            and prior["held_sha256"] == file_sha(args.tile_run / "held.json"), "Previous exact benchmark commitments differ")
    require(source_lock["source_roles_sha256"] == {key: file_sha(path) for key, path in source_paths(args).items()},
            "Qualified panel assets differ")
    return source_lock


def freeze(args):
    prior = frozen_panel(args)
    cases = {part: prior["cases"][part] for part in CALIBRATION_PANELS + HELD_PANELS}
    require({name: len(items) for name, items in cases.items()} == PANEL_COUNTS, "Unexpected exact panel denominator")
    sources = {part: {row["rgb"] for row in cases[part] if row["key"][-1] == "original"} for part in cases}
    for i, left in enumerate(sources):
        require(not any(sources[left] & sources[right] for right in tuple(sources)[i + 1:]), "Exact original images cross roles")
    return {"schema": SCHEMA, "design": DESIGN, "source_commitment": file_sha(args.tile_run / "freeze.json"),
            "previous_result_sha256": file_sha(args.tile_run / "result.json"),
            "previous_calibration_fit_sha256": file_sha(args.previous / "freeze.json"),
            "code_sha256": {p.name: file_sha(p) for p in (Path(__file__), Path(panel.__file__), Path(sharp_boundary.__file__))},
            "source_roles_sha256": prior["source_roles_sha256"], "cases": cases}


def joined_cases(args, panels, lock):
    for name in panels:
        for case, commitment in zip(panel.views(input_args(args), name), lock["cases"][name], strict=True):
            require(panel.old.identity(case) == commitment, "Native RGB, exact changed mask or delivered PNG differs")
            yield case, commitment


def response_summary(found, mask, point):
    area = pixels(found["box"], mask)
    return found | {"pixels": area, "flag": None if point is None else found["score"] is not None and found["score"] > point}


def baselines(row, rgb, mask, points):
    height, width = rgb.shape[:2]
    sharp = sharp_boundary.measure(rgb) if sharp_boundary.supported_size(width, height) else None
    original = {"sharp_boundary": {"box": list(sharp.box) if sharp else None, "score": sharp.score if sharp else None,
                                   "flag": bool(sharp and sharp.score > sharp_boundary.REVIEW_POINT)}}
    prior = row["responses"]["research_baseline"]
    old = panel.BASE.baseline
    boxes = [(x0, y0, x1, y1) for y in old.crops.STARTS for x in old.crops.STARTS
             for y0, y1 in (old.crops.boundaries(height, y),) for x0, x1 in (old.crops.boundaries(width, x),)]
    replay = old.inspect(prior["scores"], boxes, points)
    require(replay["box"] == prior["box"] and replay["remote_control"] == prior["flag"], "Historical crop decision differs")
    original["research_baseline"] = {"box": prior["box"], "score": replay["relative"], "flag": prior["flag"]}
    original["previous_tile_agreement"] = {key: row["responses"]["agreement"][key] for key in ("box", "score", "flag")}
    for found in original.values():
        found["pixels"] = pixels(found["box"], mask)
    return original


def observations(args, panels, lock, points=None):
    old = {tuple(row["key"]): row for row in read(args.tile_run / "held.json")} if points else {}
    prior_points = read(args.tile_run / "freeze.json")["baseline_points"]
    for (_, _, _, rgb, mask), committed in joined_cases(args, panels, lock):
        key = tuple(committed["key"])
        maps = references(*measurements(rgb))
        found = {arm: response_summary(select(values), mask, points[arm] if points else None)
                 for arm, values in zip(ARMS, maps, strict=True)}
        if points:
            require(all(old[key][name] == committed[name] for name in committed), "Previous comparator was measured on another input")
            found |= baselines(old[key], rgb, mask, prior_points)
        yield {"key": list(key), "family": committed["family"], "target": int(mask.sum()),
               "shape": list(mask.shape), "responses": found, "geometry": geometry(mask)}


def operating_points(rows):
    groups = {arm: {} for arm in ARMS}
    for row in rows:
        group = row["key"][1]
        for arm, maxima in groups.items():
            value = row["responses"][arm]["score"]
            maxima[group] = max(maxima.get(group, 0.), value if value is not None else 0.)
    require(all(len(rows) == 8 for rows in groups.values()), "Incomplete calibration group maxima")
    return {arm: sorted(maxima.values())[-2] for arm, maxima in groups.items()}


def aggregate(rows):
    summary = defaultdict(lambda: defaultdict(Counter))
    control_groups = {arm: set() for arm in ARMS + BASELINES}
    changes = {"added_primary_feathered": [], "added_other_feathered": [], "lost_useful": []}
    for row in rows:
        name, group, role = row["key"]
        key = f"{name}/{role}"
        results = row["responses"]
        for arm, found in results.items():
            summary[key][arm].update({"cases": 1, "assessed": found["score"] is not None, "flagged": found["flag"],
                                      "useful": found["flag"] and useful(found["pixels"]),
                                      "useful_before_point": useful(found["pixels"])})
            if not row["target"] and found["flag"]:
                control_groups[arm].add((name, group))
        before, after = (results[arm]["flag"] and useful(results[arm]["pixels"]) for arm in ("research_baseline", "matched"))
        if before and not after:
            changes["lost_useful"].append(row["key"])
        if after and not before and role in {"feathered_edit", "edit"}:
            changes["added_primary_feathered" if name == "held_lama" else "added_other_feathered"].append(row["key"])
    added = control_groups["matched"] - control_groups["research_baseline"]
    extra_photo = sum(name != "screen" for name, _ in added)
    extra_screen = len(added) - extra_photo
    passed = (len(changes["added_primary_feathered"]) >= GATE["minimum_added_primary_feathered"]
              and len(changes["added_other_feathered"]) >= GATE["minimum_added_other_feathered"]
              and len(changes["lost_useful"]) <= GATE["maximum_lost_useful"]
              and extra_photo <= GATE["maximum_extra_photo_groups"] and extra_screen <= GATE["maximum_extra_screen_groups"])
    return {"by_panel_role": {role: dict(arms) for role, arms in summary.items()},
            "ordinary_groups": {arm: dict(Counter(name for name, _ in groups)) for arm, groups in control_groups.items()},
            "candidate_oracle_useful": sum(row["geometry"]["useful_proposals"] > 0 for row in rows if row["target"]),
            "changes_against_research_baseline": changes, "extra_photo_groups": extra_photo,
            "extra_screen_groups": extra_screen, "passed": passed}


def main():
    local, tile = defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "calibrate", "held"))
    parser.add_argument("run", type=Path)
    defaults_by_name = {"tile-run": tile, "commons": local / "image-affine-native-scale-2026-11-11",
                        "primary": local / "image-bg-real-acquisition-v225", "backup": local / "image-bg-real-backup-intake-v262",
                        "legacy": local / "image-residual-patch-held-2026-11-11", "previous": local / "image-crop-layout-2026-11-12",
                        "alternate": None}
    for key, value in defaults_by_name.items():
        parser.add_argument("--" + key, type=Path, default=value, required=value is None)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    lock, root = freeze(args), args.run
    frozen = root / "freeze.json"
    if args.stage == "freeze":
        root.mkdir(mode=0o700)
        write(frozen, lock)
        print(file_sha(frozen), PANEL_COUNTS, flush=True)
        return
    require(lock == read(frozen), "Current design differs from the frozen method or source roles")
    start = time.monotonic()
    cal_path = root / "calibration.json"
    if args.stage == "calibrate":
        rows = list(observations(args, CALIBRATION_PANELS, lock))
        points = operating_points(rows)
        write(cal_path, {"freeze_sha256": file_sha(frozen), "rows": rows, "points": points})
        print(points, time.monotonic() - start, flush=True)
        return
    cal = read(cal_path)
    require(cal["freeze_sha256"] == file_sha(frozen) and operating_points(cal["rows"]) == cal["points"],
            "Held evaluation requires sealed ordinary calibration")
    rows = list(observations(args, HELD_PANELS, lock, cal["points"]))
    held = root / "held.json"
    write(held, rows)
    result = {"schema": SCHEMA, "freeze_sha256": file_sha(frozen), "calibration_sha256": file_sha(cal_path),
              "held_sha256": file_sha(held), "result": aggregate(rows), "elapsed_s": time.monotonic() - start}
    write(root / "result.json", result)
    print(result["result"]["changes_against_research_baseline"], result["result"]["passed"], flush=True)


if __name__ == "__main__":
    main()
