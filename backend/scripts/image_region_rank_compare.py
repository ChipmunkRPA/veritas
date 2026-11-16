"""Fit region ranks against same-site controls and the strongest wrong patches."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from scripts.image_region_rank_features import CELL
from scripts.image_region_rank_inputs import CONTROLS, POSITIVE, QUALITY
from scripts.video_face_temporal_fit import read, save, sha

SCHEMA = "chipmunk.image-region-ranking-comparison/1"
SIZES = (2, 3, 4, 6, 8, 10, 12, 16, 20)
ARMS = ("visual", "residual", "combined", "mined")
POINTS = {"strict": 0, "wider": 2}
ROLES = {"fit": 24, "calibration": 24, "held": 48, "transfer": 36}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integral(values):
    import numpy as np

    pad = [(1, 0), (1, 0), *([(0, 0)] * (values.ndim - 2))]
    return np.pad(values, pad).cumsum(axis=0, dtype=np.float64).cumsum(axis=1)


def window(table, boxes):
    left, top, right, bottom = boxes.T
    return table[bottom, right] + table[top, left] - table[bottom, left] - table[top, right]


@lru_cache(maxsize=5)
def layout(shape):
    import numpy as np

    h, w = (n // CELL for n in shape)
    windows = []
    for height in SIZES:
        for width in SIZES:
            area = height * width * CELL**2 / (shape[0] * shape[1])
            if .0007 <= area <= .14 and width <= w and height <= h and max(height, width) <= 3 * min(height, width):
                y, x = np.mgrid[:h-height+1, :w-width+1]
                windows.append(np.stack((x, y, x + width, y + height), axis=2).reshape(-1, 4).astype(np.int32))
    boxes = np.concatenate(windows)
    spread = np.clip(boxes + [-2, -2, 2, 2], 0, [w, h, w, h]).astype(np.int32)
    size = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    ring = np.prod(spread[:, 2:] - spread[:, :2], axis=1) - size
    return boxes, spread, size, ring


def regions(grid, geometry, indices):
    import numpy as np

    boxes, rings, size, ring_size = geometry
    table = integral(grid)
    central = window(table, boxes[indices])
    surrounding = window(table, rings[indices]) - central
    inside = central / size[indices, None]
    return np.concatenate((inside, inside - surrounding / ring_size[indices, None]), axis=1)


def all_scores(grid, geometry, head):
    import numpy as np

    boxes, rings, size, ring_size = geometry
    effective = np.asarray(head["weights"]) / head["scale"]
    inside_weight, contrast_weight = np.split(effective, 2)
    combined = integral(grid @ (inside_weight + contrast_weight))
    contrast = integral(grid @ contrast_weight)
    return (window(combined, boxes) / size - (window(contrast, rings) - window(contrast, boxes)) / ring_size
            - np.dot(effective, head["center"]))


def load_grid(cache, row):
    import numpy as np

    with np.load(cache / (row["file"].replace("/", "-") + ".npz"), allow_pickle=False) as packet:
        require(packet["source"].item() == row["sha256"], "Feature grid differs from the encoded source")
        value = np.concatenate((packet["visual"], packet["residual"]), axis=2).astype(np.float32)
        require(np.isfinite(value).all(), "Feature grid contains a non-finite result")
        return value


def exact_mask(media, row):
    import numpy as np
    from PIL import Image

    path = media / row["mask"]
    require(sha(path) == row["mask_sha256"], "Truth image differs from the checked mask")
    with Image.open(path) as image:
        return np.array(image.convert("L")) > 0


def best_boxes(mask, geometry):
    import numpy as np

    boxes, _, size, _ = geometry
    shared = window(integral(mask), boxes * CELL)
    truth = int(mask.sum())
    overlap = shared / (size * CELL**2 + truth - shared)
    admissible = np.flatnonzero((overlap >= .30) & (2 * shared >= truth))
    return admissible[np.argsort(-overlap[admissible], kind="stable")[:8]], overlap


def train(media, cache, groups, cases, quality, *, first=None):
    import numpy as np

    columns = defaultdict(dict)
    for row in cases:
        if row["quality"] == quality and row["group"] in groups:
            columns[row["group"]][row["role"]] = row
    positive, negative, group_ids = [], [], []

    def add(edited_sites, decoys, identity):
        positive.extend(edited_sites)
        negative.extend(decoys)
        group_ids.extend([identity] * len(edited_sites))

    for identity, views in columns.items():
        shape = tuple(groups[identity]["shape"])
        geometry = layout(shape)
        random = np.random.default_rng(int(groups[identity]["source"][:12], 16))
        tiles = {key: load_grid(cache, row) for key, row in views.items()}
        for key in POSITIVE:
            truth = exact_mask(media, views[key])
            good, overlap = best_boxes(truth, geometry)
            if not len(good):
                continue
            evidence = regions(tiles[key], geometry, good)
            other = np.flatnonzero(overlap <= .10)
            random_boxes = random.choice(other, size=len(good), replace=False)
            add(evidence, regions(tiles[key], geometry, random_boxes), identity)
            for control in CONTROLS:
                add(evidence, regions(tiles[control], geometry, good), identity)
            if first is not None:
                for lane in (key, *CONTROLS):
                    scores = all_scores(tiles[lane], geometry, first)
                    if lane == key:
                        scores[overlap > .10] = -np.inf
                    ranked = np.argsort(-scores, kind="stable")[:len(good)]
                    add(evidence, regions(tiles[lane], geometry, ranked), identity)
    require(len(set(group_ids)) >= 16, "Too few fitting groups have known regions on the frozen native grid")
    edited, decoys = np.asarray(positive), np.asarray(negative)
    weights = Counter(group_ids)
    balance = np.array([1 / (len(weights) * weights[key]) for key in group_ids])
    center = balance @ ((edited + decoys) / 2)
    scale = np.maximum(np.sqrt(balance @ ((edited - center)**2 + (decoys - center)**2) / 2), .001)
    p, n = (edited - center) / scale, (decoys - center) / scale
    visual = [*range(32), *range(len(center)//2, len(center)//2+32)]
    residual = [key for key in range(len(center)) if key not in visual]
    arms = {"visual": visual, "residual": residual, "combined": list(range(len(center)))} if first is None else {
        "mined": list(range(len(center))),
    }
    trained = {}
    for arm, indices in arms.items():
        delta = p[:, indices] - n[:, indices]
        scatter = (delta.T * balance) @ delta + .3 * np.eye(len(indices))
        vector = np.zeros(len(center))
        vector[indices] = np.linalg.solve(scatter, balance @ delta)
        trained[arm] = {"weights": vector.tolist(), "center": center.tolist(), "scale": scale.tolist(),
                        "matched_pairs": len(group_ids), "fitted_groups": len(weights)}
    return trained


def incumbent(path):
    with path.open() as stream:
        header = json.loads(next(stream))
        return header, {item["file"]: item for line in stream if line.endswith("\n") for item in (json.loads(line),)}


def score_rows(study, cache, groups, cases, heads, baseline):
    import numpy as np

    result = []
    for row in cases:
        group = groups.get(row["group"])
        if group is None:
            continue
        geometry = layout(tuple(group["shape"]))
        grid = load_grid(cache, row)
        named = {}
        for arm, head in heads[str(row["quality"])].items():
            scores = all_scores(grid, geometry, head)
            greatest = int(np.argmax(scores))
            named[arm] = {"score": float(scores[greatest]), "box": (geometry[0][greatest] * CELL).tolist()}
        mask = exact_mask(study, row)
        for outcome in named.values():
            x, y, right, bottom = outcome["box"]
            outcome.update(selected=(right - x) * (bottom - y), intersection=int(mask[y:bottom, x:right].sum()), truth=int(mask.sum()))
        old = baseline[row["file"]]
        require(old["sha256"] == row["sha256"], "Reference result differs from evaluated image")
        result.append({**row, "split": group["split"], "family": group["family"], "shape": group["shape"],
                       "checks": named, "current": {name: old[name] for name in ("jpeg", "regional")}})
        if len(result) % 96 == 0:
            print(f"Ranked proposals for {len(result)} images.", flush=True)
    return result


def useful(metric):
    if not metric:
        return False
    count, correct, truth = (metric[key] for key in ("selected", "intersection", "truth"))
    return truth > 0 and 2 * correct >= truth and 10 * correct >= 3 * (truth + count - correct)


def current_flags(row):
    return any(metric and metric["selected"] > 0 for metric in row["current"].values())


def points(rows):
    levels = {}
    picks = {}
    for q in QUALITY:
        view = [row for row in rows if row["quality"] == q]
        for arm in ARMS:
            ordinary = {}
            for row in view:
                if row["role"] not in POSITIVE and not current_flags(row):
                    key, value = row["group"], row["checks"][arm]["score"]
                    ordinary[key] = max(ordinary.get(key, -float("inf")), value)
            require(len(ordinary) >= 16, "Too few available ordinary operating-point groups")
            ordered = sorted(ordinary.values(), reverse=True)
            levels[f"{q}/{arm}"] = {key: ordered[allowed] for key, allowed in POINTS.items()}

        def rank(arm, view=view, quality=q):
            added = [row for row in view if row["role"] in POSITIVE
                     and row["checks"][arm]["score"] > levels[f"{quality}/{arm}"]["wider"]]
            new = [row for row in added if useful(row["checks"][arm]) and not any(map(useful, row["current"].values()))]
            return (-sum(row["role"] == "feather" for row in new), -len(new),
                    sum(not useful(row["checks"][arm]) for row in added), ARMS.index(arm))

        picks[str(q)] = min(ARMS, key=rank)
    return levels, picks


def summarize(rows, levels):
    totals = defaultdict(Counter)
    added_groups = defaultdict(set)
    for row in rows:
        current_ok = any(map(useful, row["current"].values()))
        current_present = current_flags(row)
        for cohort in ("all", row["family"]):
            for arm in ARMS:
                key = f"{row['split']}/{row['quality']}/{cohort}/{row['role']}/{arm}"
                result = totals[key]
                response = row["checks"][arm]
                own_ok = useful(response)
                result.update(total=1, current_flagged=current_present, current_useful=current_ok,
                              regional_assessed=row["current"]["regional"] is not None,
                              physical_assessed=row["current"]["jpeg"] is not None, top_useful=own_ok)
                for label, cutoff in levels[f"{row['quality']}/{arm}"].items():
                    flag = response["score"] > cutoff
                    result[f"{label}_flagged"] += flag
                    result[f"{label}_combined_useful"] += current_ok or (flag and own_ok)
                    result[f"{label}_misplaced"] += flag and not own_ok and row["role"] in POSITIVE
                    if flag and not current_present and row["role"] not in POSITIVE:
                        added_groups[key, label].add(row["group"])
    for (key, label), groups in added_groups.items():
        totals[key][f"{label}_added_ordinary_groups"] = len(groups)
    return totals


def fit(study):
    media = study / "deliveries"
    plan = read(media / "manifest.json")
    fit_groups = {row["tag"]: row for row in plan["groups"] if row["split"] == "fit"}
    choice_groups = {row["tag"]: row for row in plan["groups"] if row["split"] == "calibration"}
    baseline_header, baseline = incumbent(study / "reference.jsonl")
    require(all(row["file"] in baseline for row in plan["images"] if row["group"] in choice_groups),
            "The current-check calibration comparison is incomplete")
    heads = {}
    for q in QUALITY:
        bank = study / "rank-fit"
        models = train(media, bank, fit_groups, plan["images"], q)
        models.update(train(media, bank, fit_groups, plan["images"], q, first=models["combined"]))
        heads[str(q)] = models
    observed = score_rows(media, study / "rank-choice", choice_groups, plan["images"], heads, baseline)
    limits, chosen = points(observed)
    freeze = {"schema": SCHEMA, "manifest_sha256": sha(media / "manifest.json"), "fitter_sha256": sha(Path(__file__)),
              "heads": heads, "levels": limits, "chosen": chosen, "calibration": observed,
              "reference": baseline_header, "features": read(study / "rank-fit/features.json"),
              "roles": ROLES, "held_responses_at_freeze": 0}
    save(study / "rank-freeze.json", freeze)
    print(f"Locked separate JPEG operating points and ranked checks {chosen}.", flush=True)


def assess(study):
    media = study / "deliveries"
    plan = read(media / "manifest.json")
    freeze = read(study / "rank-freeze.json")
    require(freeze["manifest_sha256"] == sha(media / "manifest.json") and freeze["fitter_sha256"] == sha(Path(__file__)),
            "Frozen comparison changed")
    baseline_header, baseline = incumbent(study / "reference.jsonl")
    require(baseline_header == freeze["reference"], "Existing detector reference changed")
    output = []
    for split in ("held", "transfer"):
        bank = study / ("rank-" + split)
        header = read(bank / "features.json")
        require(all(header[key] == freeze["features"][key] for key in header if key != "roles"),
                "Encoder or native feature geometry changed after fitting")
        group = {row["tag"]: row for row in plan["groups"] if row["split"] == split}
        output.extend(score_rows(media, bank, group, plan["images"], freeze["heads"], baseline))
    result = {"schema": SCHEMA, "freeze": freeze, "source_groups": plan["groups"], "source_manifests": plan["source_manifests"],
              "held": output, "summary": summarize(output, freeze["levels"])}
    verify(result)
    save(study / "rank-result.json.gz", result)


def verify(result):
    locked = result["freeze"]
    require(result["schema"] == locked["schema"] == SCHEMA and locked["roles"] == ROLES,
            "Unknown study roles or schema")
    levels, chosen = points(locked["calibration"])
    require(levels == locked["levels"] and chosen == locked["chosen"], "Frozen operating-point arithmetic differs")
    groups = result["source_groups"]
    require(Counter(row["split"] for row in groups) == ROLES, "Image source roles are incomplete")
    for key in ("tag", "source", "rgb_sha256"):
        require(len({row[key] for row in groups}) == len(groups), "Repeated source identity")
    rows = result["held"]
    plan = {(group["tag"], role, quality) for group in groups if group["split"] != "fit"
            for role in (*POSITIVE, *CONTROLS) for quality in QUALITY}
    measured = [(row["group"], row["role"], row["quality"]) for row in locked["calibration"] + rows]
    require(set(measured) == plan and len(measured) == len(plan), "Image delivery coverage changed")
    for row in rows:
        height, width = row["shape"]
        for outcome in row["checks"].values():
            l, t, r, b = outcome["box"]
            require(0 <= l < r <= width and 0 <= t < b <= height and all(key % CELL == 0 for key in outcome["box"])
                    and (r - l) * (b - t) == outcome["selected"] and outcome["truth"] == row["changed_pixels"]
                    and 0 <= outcome["intersection"] <= min(outcome["selected"], outcome["truth"]),
                    "Invalid native evidence geometry")
    require(summarize(rows, levels) == result["summary"], "Held location arithmetic differs")
    return {"images": len(rows), "source_roles": ROLES, "selected_by_quality": chosen}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("fit", "assess", "verify"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.step == "fit":
        fit(args.path)
    elif args.step == "assess":
        assess(args.path)
    else:
        print(json.dumps(verify(read(args.path)), sort_keys=True))
