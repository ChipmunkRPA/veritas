"""Fit and replay source-disjoint face-residual and current-score comparisons."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from app.evidence import whole_face_binding as deployed
from scripts.audit_video_face_temporal import public_roster
from scripts.video_face_temporal_fit import read, save, sha

SCHEMA = "chipmunk.face-residual-comparison/1"
POINTS = ("wider", "strict")
VIEWS = {
    "appearance": ("appearance",), "spatial": ("spatial",), "temporal": ("temporal",),
    "residuals": ("spatial", "temporal"), "learned_visual": ("appearance", "spatial", "temporal"),
    "current_readings": ("current",), "combined": ("current", "appearance", "spatial", "temporal"),
}
DESIGN = {
    "roles": {"fit": 60, "selection": 30, "held": 24, "camera": 8},
    "fit": "For each available view, balance origins and source groups. Standardize by fitted controls and normalize blocks. "
           "Learn one ridge head from all available individual clips and matched content differences within each delivered source.",
    "comparison": "Use the current pooled same-picture fallback in the baseline. Compare standalone points and retaining all "
                  "baseline flags while adding each candidate on remaining videos. Use one point across both encodes.",
    "selection": "Choose standalone points at the baseline's ordinary source-group count. Set fallback points at a maximum of "
                 "two added ordinary selection groups at the wider point and zero at the stricter point. Rank arms by tandem "
                 "swap gains at the wider point, then fewer ordinary group additions, then total swap gains.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def journal(path):
    with path.open() as stream:
        header = json.loads(next(stream))
        return header, [json.loads(line) for line in stream]


def current(packet):
    if packet is None:
        return None
    require(packet["schema"] == deployed.METHOD, "Expected the unmodified paired reference readings")
    samples = packet["samples"]
    require([r["frame_index"] for r in samples] == list(deployed.sample_indices(range(100))), "Paired sample schedule changed")
    require(all(r["pts"] * Fraction(packet["time_base"]) == Fraction(r["frame_index"], 25) for r in samples),
            "Paired reference clock changed")
    raw = [[float.fromhex(r["responses_hex"][arm]) if r["responses_hex"][arm] is not None else None
            for r in samples] for arm in range(2)]
    paired = deployed.paired_score(*raw)
    require(paired == float.fromhex(packet["score_hex"]), "The stored reference score differs")
    return [paired, deployed.same_frame_score(*raw)]


def prior_measurements(folder, feature_path, references, *, families=None):
    header, rows = journal(feature_path)
    manifest = folder / "manifest.json"
    require(sha(manifest) == header["manifest_sha256"], "Feature journal uses a different source manifest")
    planned = [row for row in read(manifest)["rows"] if "file" in row and row["delivery"] != "middle"]
    require(len(planned) == len(rows) == len({row["file"] for row in rows}), "Incomplete or repeated feature journal")
    rows = {row["file"]: row for row in rows}
    result = []
    for case in planned:
        item = rows[case["file"]]
        reference = references[case["sha256"]]
        require(item["sha256"] == case["sha256"], "Readout does not match the delivered file")
        merged = {key: case[key] for key in ("file", "group", "family", "delivery", "sha256", "source_sha256")}
        merged.update(time_base=item["time_base"], frames=item["frames"], features=item["features"],
                      reference=reference, current=current(reference))
        if families is None or case["family"] in families:
            result.append(merged)
    return header, result


def fitted_point(ordinary, allowance):
    ordered = sorted(ordinary, reverse=True)
    if len(ordered) <= allowance:
        return None
    return ordered[allowance]


def baseline(row):
    values = row["current"]
    return values is not None and ((values[0] is not None and values[0] > deployed.REVIEW_POINT)
                                  or (values[1] is not None and values[1] > deployed.SAME_FRAME_POINT))


def group_scores(rows, method, *, remaining=False):
    groups = defaultdict(list)
    for row in rows:
        score = row["scores"][method]
        if row["family"] == "ordinary" and score is not None and (not remaining or not baseline(row)):
            groups[row["group"]].append(score)
    return [max(x) for x in groups.values()]


def points_for(rows, method):
    limit = len({row["group"] for row in rows if row["family"] == "ordinary" and baseline(row)})
    all_points = group_scores(rows, method)
    remaining = group_scores(rows, method, remaining=True)
    return {"standalone": fitted_point(all_points, limit),
            "wider": fitted_point(remaining, 2), "strict": fitted_point(remaining, 0)}


def above(score, point):
    return score is not None and point is not None and score > point


def selection_order(rows, points):
    def rank(method):
        gains = [row for row in rows if not baseline(row) and above(row["scores"][method], points[method]["wider"])]
        extra = {row["group"] for row in gains if row["family"] == "ordinary"}
        true = [row for row in gains if row["family"] != "ordinary"]
        return -sum(row["delivery"] == "tandem" for row in true), len(extra), -len(true), list(VIEWS).index(method)

    return sorted(VIEWS, key=rank)


def features(row, method):
    blocks = []
    for name in VIEWS[method]:
        view = row["current"] if name == "current" else (row.get("features") or {}).get(name)
        if view is None or any(value is None for value in view):
            return None
        blocks.append(view)
    return blocks


def fit_head(rows, method):
    import numpy as np

    measured = [(row, blocks) for row in rows if (blocks := features(row, method)) is not None]
    weights = Counter((row["group"], row["family"] != "ordinary") for row, _ in measured)
    role_counts = Counter(label for _, label in weights)
    if min(role_counts.values(), default=0) < 16 or len(role_counts) != 2:
        raise ValueError(f"Too few measured source groups for {method}")
    block_lengths = [len(block) for block in measured[0][1]]
    penalty = np.concatenate([np.repeat(np.sqrt(length), length) for length in block_lengths])
    matrix = np.stack([np.concatenate(blocks) for _, blocks in measured])
    labels = np.array([row["family"] != "ordinary" for row, _ in measured], dtype=bool)
    weight = np.array([.5 / (weights[row["group"], label] * role_counts[label])
                       for (row, _), label in zip(measured, labels, strict=True)])
    center = weight @ matrix
    scale = np.maximum(np.sqrt(weight @ (matrix - center) ** 2), 1e-5) * penalty
    normalized = (matrix - center) / scale
    genuine = {(row["group"], row["delivery"]): item for (row, _), item in zip(measured, normalized, strict=True)
               if row["family"] == "ordinary"}
    pairs = np.stack([item - genuine[key] for (row, _), item in zip(measured, normalized, strict=True)
                      if row["family"] != "ordinary" and (key := (row["group"], row["delivery"])) in genuine])
    require(len(pairs) >= 32, f"Too few matched readings for {method}")
    equation = (normalized.T * weight) @ normalized + pairs.T @ pairs / len(pairs) + .50 * np.eye(matrix.shape[1])
    target = (normalized.T * weight) @ (labels.astype(float) - .5) + pairs.mean(axis=0)
    return {"center": center.tolist(), "scale": scale.tolist(), "weight": np.linalg.solve(equation, target).tolist(),
            "measured": len(measured), "pairs": len(pairs)}


def score_rows(rows, heads):
    def score(row, method):
        blocks = features(row, method)
        if blocks is None:
            return None
        head = heads[method]
        flat = (value for block in blocks for value in block)
        return math.fsum((value - center) * weight / scale for value, center, scale, weight in
                         zip(flat, head["center"], head["scale"], head["weight"], strict=True))

    return [{key: value for key, value in row.items() if key != "features"}
            | {"scores": {method: score(row, method) for method in VIEWS}} for row in rows]


def tally(rows, points):
    counters = defaultdict(Counter)
    new_groups = defaultdict(set)
    for row in rows:
        original = baseline(row)
        family = row["family"]
        cell = (row["delivery"], family)
        counters[cell]["planned"] += 1
        counters[cell]["baseline"] += original
        counters[cell]["baseline_unassessed"] += row["current"] is None
        for method, limit in points.items():
            response = row["scores"][method]
            counters[cell][f"{method}/unassessed"] += response is None
            counters[cell][f"{method}/standalone"] += above(response, limit["standalone"])
            for point in POINTS:
                flag = above(response, limit[point])
                counters[cell][f"{method}/{point}"] += original or flag
                if flag and not original:
                    new_groups[cell + (method, point)].add(row["group"])
    for (*cell, method, point), group in new_groups.items():
        counters[tuple(cell)][f"{method}/{point}/added_source_groups"] = len(group)
    return {"/".join(key): dict(count) for key, count in sorted(counters.items())}


def old_reference(path):
    receipt = read(path)
    return {row["sha256"]: row["reference"] for row in receipt["training"] + receipt["held"] + receipt["camera_transfer"]["rows"]}


def freeze(args):
    archive = old_reference(args.previous)
    headers, fit_groups, choice = [], [], []
    for directory, journal_file, rows in ((args.first, args.study / "fit-a.jsonl", fit_groups),
                                          (args.second, args.study / "fit-b.jsonl", fit_groups),
                                          (args.selection, args.study / "selection.jsonl", choice)):
        header, measured = prior_measurements(directory, journal_file, archive)
        rows.extend(measured)
        headers.append(header)
    training_keys = {row["group"] for row in fit_groups}
    selection_keys = {row["group"] for row in choice}
    require(len(training_keys) == 60 and len(selection_keys) == 30 and not training_keys & selection_keys,
            "Training and selection roles overlap or are incomplete")
    heads = {method: fit_head(fit_groups, method) for method in VIEWS}
    scores = score_rows(choice, heads)
    points = {key: points_for(scores, key) for key in VIEWS}
    targets = read(args.study / "fresh-roster.json")
    require(not {row["group"] for row in targets["groups"]} & (training_keys | selection_keys), "A held target has project exposure")
    save(args.study / "comparison-freeze.json", {"schema": SCHEMA, "design": DESIGN, "heads": heads, "points": points,
             "selection": scores, "selection_order": selection_order(scores, points), "headers": headers,
             "fit_groups": sorted(training_keys), "selection_groups": sorted(selection_keys),
             "fit_sources_sha256": [sha(args.study / name) for name in ("fit-a.jsonl", "fit-b.jsonl", "selection.jsonl")],
             "held_plan_sha256": sha(args.study / "fresh-roster.json"), "previous_sha256": sha(args.previous),
             "fitter_sha256": sha(Path(__file__)), "held_responses_at_freeze": 0})
    print(f"Froze {len(VIEWS)} combinations on {len(training_keys)} fitting and {len(selection_keys)} point-selection groups.")


def assess(args):
    freeze_path = args.study / "comparison-freeze.json"
    lock = read(freeze_path)
    targets = args.study / "fresh-roster.json"
    require(lock["schema"] == SCHEMA and lock["fitter_sha256"] == sha(Path(__file__))
            and lock["held_plan_sha256"] == sha(targets) and lock["previous_sha256"] == sha(args.previous), "Comparison inputs changed")
    ref_header, rows = journal(args.study / "fresh-reference.jsonl")
    manifest = args.study / "fresh/manifest.json"
    require(ref_header["manifest_sha256"] == sha(manifest), "Current reference uses another holdout")
    prior = {row["sha256"]: row["review"] for row in rows}
    panels = {}
    for name, directory, file, references in (("held", args.study / "fresh", args.study / "held.jsonl", prior),
                                                ("camera", args.camera, args.study / "camera.jsonl", old_reference(args.previous))):
        header, measured = prior_measurements(directory, file, references)
        require(header["weights"] == lock["headers"][0]["weights"] and header["method"] == lock["headers"][0]["method"]
                and header["code_sha256"] == lock["headers"][0]["code_sha256"], "The held feature extractor changed")
        scores = score_rows(measured, lock["heads"])
        panels[name] = {"header": header, "scores": scores, "summary": tally(scores, lock["points"])}
    result = {"schema": SCHEMA, "freeze": lock, "freeze_sha256": sha(freeze_path), "reference_header": ref_header,
              "acquisition": public_roster(read(targets)), "panels": panels}
    verify(result)
    save(args.study / "result.json.gz", result)


def verify(result):
    lock, panels = result["freeze"], result["panels"]
    require(result["schema"] == lock["schema"] == SCHEMA and lock["design"] == DESIGN, "Unknown residual comparison")
    choice = lock["selection"]
    points = {method: points_for(choice, method) for method in VIEWS}
    require(points == lock["points"] and selection_order(choice, points) == lock["selection_order"], "Selection arithmetic differs")
    require(set(panels) == {"held", "camera"}, "Incomplete held and camera results")
    groups_by_role = [set(lock["fit_groups"]), set(lock["selection_groups"])]
    for role in ("held", "camera"):
        panel = panels[role]
        rows = panel["scores"]
        require(tally(rows, points) == panel["summary"], "Held decision counts differ")
        expected = {"ordinary", "inswapper", "uniface"} if role == "held" else {"ordinary"}
        keys = {(row["group"], row["family"], row["delivery"]) for row in rows}
        sources = {row["group"] for row in rows}
        require(len(sources) == DESIGN["roles"][role] and len(keys) == len(rows)
                and keys == {(key, family, codec) for key in sources for family in expected for codec in ("direct", "tandem")},
                "Held recordings or their paired re-encodes are incomplete")
        groups_by_role.append(sources)
        for row in rows:
            require(current(row["reference"]) == row["current"] and set(row["scores"]) == set(VIEWS)
                    and all(x is None or math.isfinite(x) for x in row["scores"].values()), "Invalid or unbound face response")
            if row["frames"]:
                indices = [spot["index"] for spot in row["frames"]]
                require(all(i in range(5, 100, 6) for i in indices) and all(b - a == 6 for a, b in zip(indices, indices[1:]))
                        and all(pos["pts"] * Fraction(row["time_base"]) == Fraction(pos["index"], 25) for pos in row["frames"]),
                        "Native residual frame clock differs")
    require(sum(map(len, groups_by_role)) == len(set.union(*groups_by_role)), "A source appears in more than one role")
    return {"held": len(panels["held"]["scores"]), "camera": len(panels["camera"]["scores"]), "selected": lock["selection_order"][0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "assess", "verify"))
    parser.add_argument("study", type=Path)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--camera", type=Path)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    if args.phase == "verify":
        print(json.dumps(verify(read(args.study)), sort_keys=True))
    elif args.phase == "assess":
        assess(args)
    else:
        freeze(args)
