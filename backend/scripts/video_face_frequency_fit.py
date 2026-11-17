"""Compare frequency and appearance evidence for strongly compressed face swaps."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import median

from scripts import video_face_residual_fit as residual
from scripts.video_face_temporal_fit import read, save, sha

SCHEMA = "chipmunk.face-frequency-comparison/1"
VIEWS = {
    "appearance": ("appearance",),
    "relative_appearance": ("relative_appearance",),
    "frequency": ("absolute", "relative"),
    "appearance_frequency": ("relative_appearance", "absolute", "relative"),
    "paired_frequency": ("paired", "relative_appearance", "absolute", "relative"),
}
SCOPE = ("all", "strong")
QUALITY = (30., 46.)
WIDER_ALLOWANCE = {"original": 2, "camera": 1}
DESIGN = {
    "input": "Compare spatial band energies and appearance on the same uninterrupted aligned face track.",
    "quality": "The scoped fit and readout require measured median inter-picture quantization from 30 through 46.",
    "fit": "Give each source group equal weight within each class. Use same-target, same-encode score differences and a fixed ridge penalty.",
    "calibration": "Freeze fallback points from the remaining ordinary source maxima and a replacement with a matched alert budget.",
    "selection": "Rank by recovered strongly recompressed swap groups at the wider point, then camera alerts, then total swaps and ordinary groups.",
}


def is_strong(qp):
    return qp is not None and QUALITY[0] <= qp <= QUALITY[1]


def qp(quantization, time_base):
    if quantization is None:
        return None
    frames = quantization["frames"]
    tb = Fraction(time_base)
    if (len(frames) != 100 or any(tick * tb != Fraction(i, 25) or kind not in ("I", "P", "B")
                                 or type(value) is not int or not 0 <= value <= 51
                                 for i, (tick, kind, value) in enumerate(frames))):
        raise ValueError("Quantization readings differ from the verified native picture clock")
    inter = [value for _, kind, value in frames if kind != "I"]
    if len(inter) < 50 or median(inter) != quantization["inter_picture_median"]:
        raise ValueError("Incorrect inter-picture quantization median")
    return median(inter)


def attach(folder, file, references, *, camera=False):
    from scripts.video_face_frequency_readout import SCHEMA as FREQUENCY_SCHEMA

    header, measured = residual.journal(file)
    manifest = folder / "manifest.json"
    source = {row["file"]: row for row in read(manifest)["rows"] if "file" in row and row["delivery"] in ("direct", "tandem")}
    if (header["schema"] != FREQUENCY_SCHEMA or sha(manifest) != header["manifest_sha256"]
            or len(measured) != len(source) or {row["file"] for row in measured} != set(source)):
        raise ValueError("A detail journal does not cover its committed delivery panel")
    result = []
    for row in measured:
        expected = source[row["file"]]
        if any(row[key] != expected[key] for key in ("group", "family", "delivery", "sha256", "source_sha256")):
            raise ValueError("Frequency responses differ from their source identities")
        prior = references[row["sha256"]]
        current = residual.current(prior)
        if prior is not None:
            prior = prior | {"samples": [{key: item[key] for key in ("frame_index", "pts", "responses_hex")}
                                         for item in prior["samples"]]}
        result.append(row | {"reference": prior, "current": current, "qp": qp(row["quantization"], row["time_base"]),
                             "origin": "camera" if camera else "original"})
    return header, result


def _blocks(row, view, scope):
    response = row["features"]
    if response is None or (scope == "strong" and not is_strong(row["qp"])):
        return None
    if "paired" in VIEWS[view] and (row["current"] is None or None in row["current"]):
        return None
    appearance = response["appearance"]
    if len(appearance) != 144:
        raise ValueError("Unexpected three-zone appearance response width")
    inner, ring, outer = (appearance[i:i + 48] for i in (0, 48, 96))
    rel = [a - b for a, b in zip(inner, ring, strict=True)] + [a - b for a, b in zip(ring, outer, strict=True)]
    values = response | {"relative_appearance": rel, "paired": row["current"]}
    return [values[name] for name in VIEWS[view]]


def fit_head(rows, view, scope):
    import numpy as np

    samples = [(row, blocks) for row in rows if (blocks := _blocks(row, view, scope)) is not None]
    counts = Counter((row["group"], row["family"] != "ordinary") for row, _ in samples)
    groups = Counter(label for _, label in counts)
    if len(groups) != 2 or min(groups.values()) < 12:
        raise ValueError(f"Too few existing matched training groups for {view}/{scope}")
    labels = np.asarray([row["family"] != "ordinary" for row, _ in samples])
    weights = np.array([.5 / (groups[label] * counts[row["group"], label])
                        for (row, _), label in zip(samples, labels, strict=True)])
    matrix = np.stack([np.concatenate(blocks) for _, blocks in samples])
    penalty = np.concatenate([np.full(len(block), np.sqrt(len(block))) for block in samples[0][1]])
    center = weights @ matrix
    scale = np.maximum(np.sqrt(weights @ (matrix - center) ** 2), 1e-5) * penalty
    data = (matrix - center) / scale
    ordinary = {(row["group"], row["delivery"]): x for (row, _), x in zip(samples, data, strict=True)
                if row["family"] == "ordinary"}
    paired = np.stack([x - ordinary[key] for (row, _), x in zip(samples, data, strict=True)
                       if row["family"] != "ordinary" and (key := (row["group"], row["delivery"])) in ordinary])
    equation = (data.T * weights) @ data + paired.T @ paired / len(paired) + .50 * np.eye(data.shape[1])
    target = (data.T * weights) @ (labels.astype(float) - .5) + paired.mean(axis=0)
    return {"center": center.tolist(), "scale": scale.tolist(), "weight": np.linalg.solve(equation, target).tolist(),
            "observations": len(samples), "matched_pairs": len(paired)}


def score_rows(rows, heads):
    def score(row, key):
        view, scope = key.split("/")
        blocks = _blocks(row, view, scope)
        if blocks is None:
            return None
        head = heads[key]
        return math.fsum((x - center) * weight / scale for x, center, scale, weight in zip(
            (value for block in blocks for value in block), head["center"], head["scale"], head["weight"], strict=True))

    return [{key: value for key, value in row.items() if key != "features"}
            | {"scores": {key: score(row, key) for key in heads}} for row in rows]


def points(rows, key):
    maximum = defaultdict(dict)
    assessed, baseline_flags = {}, set()
    for row in rows:
        score = row["scores"][key]
        if row["family"] != "ordinary" or score is None:
            continue
        group, origin = row["group"], row["origin"]
        assessed[group] = max(assessed.get(group, -math.inf), score)
        if residual.baseline(row):
            baseline_flags.add(group)
        else:
            maximum[origin][group] = max(maximum[origin].get(group, -math.inf), score)
    fallback = {}
    for policy, allowance in (("strict", {key: 0 for key in WIDER_ALLOWANCE}), ("wider", WIDER_ALLOWANCE)):
        limits = [residual.fitted_point(maximum[origin].values(), count) for origin, count in allowance.items()]
        fallback[policy] = max(limits) if all(limit is not None for limit in limits) else None
    return fallback | {"substitute": residual.fitted_point(assessed.values(), len(baseline_flags))}


def selection_order(rows, calibration):
    def rank(key):
        gains = [row for row in rows if not residual.baseline(row)
                 and residual.above(row["scores"][key], calibration[key]["wider"])]
        swaps = [row for row in gains if row["family"] != "ordinary"]
        tandem = {row["group"] for row in swaps if row["delivery"] == "tandem"}
        camera = {row["group"] for row in gains if row["origin"] == "camera"}
        ordinary = {row["group"] for row in gains if row["family"] == "ordinary"}
        return -len(tandem), len(camera), -len(swaps), len(ordinary), key

    return sorted(calibration, key=rank)


def tally(rows, calibration):
    output = defaultdict(Counter)
    for row in rows:
        cell = output["/".join((row["origin"], row["delivery"], row["family"]))]
        base = residual.baseline(row)
        cell.update(total=1, current=base, current_unassessed=row["current"] is None)
        for key, limits in calibration.items():
            score = row["scores"][key]
            cell[f"{key}/unassessed"] += score is None
            for mode, point in limits.items():
                detected = residual.above(score, point)
                if mode == "substitute":
                    flag = base if score is None or point is None else detected
                else:
                    flag = base or detected
                cell.update({f"{key}/{mode}": flag, f"{key}/{mode}/added": flag and not base,
                             f"{key}/{mode}/lost": base and not flag})
    return {key: dict(value) for key, value in sorted(output.items())}


def _reference_journal(path):
    header, rows = residual.journal(path)
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Repeated delivery in paired face reference")
    return header, {row["sha256"]: row["review"] for row in rows}


def freeze(study, previous, second, earlier):
    base = read(previous / "result.json.gz")
    refs = residual.old_reference(earlier)
    for panel in base["panels"].values():
        refs.update({row["sha256"]: row["reference"] for row in panel["scores"]})
    paths = ((previous / "first/delivered", "fit-a", False), (second, "fit-b", False),
             (previous / "quality/pair", "selection", False), (previous / "fresh", "selection-b", False),
             (previous / "quality/camera-pair", "selection-camera", True))
    headers, observed = zip(*(attach(folder, study / f"{name}.jsonl", refs, camera=camera)
                              for folder, name, camera in paths), strict=True)
    train = [row for panel in observed[:2] for row in panel]
    select = [row for panel in observed[2:] for row in panel]
    fit_groups, select_groups = ({row["group"] for row in rows} for rows in (train, select))
    roster_path = study / "held-roster.json"
    fresh = read(roster_path)
    held_groups = {row["group"] for row in fresh["groups"]}
    assert len(fit_groups) == 60 and len(select_groups) == 62 and len(held_groups) == 32
    assert not fit_groups & select_groups and not held_groups & (fit_groups | select_groups)
    assert all(all(header[key] == headers[0][key] for key in ("weights", "frequency_sha256", "track_sha256", "readout_sha256"))
               for header in headers)
    heads = {f"{view}/{scope}": fit_head(train, view, scope) for view in VIEWS for scope in SCOPE}
    choices = score_rows(select, heads)
    calibration = {key: points(choices, key) for key in heads}
    ordered = selection_order(choices, calibration)
    lock = {"schema": SCHEMA, "design": DESIGN, "fit_groups": sorted(fit_groups), "held_groups": sorted(held_groups),
            "fit_and_selection_sha256": {name: sha(study / f"{name}.jsonl") for _, name, _ in paths},
            "roster_sha256": sha(roster_path), "camera_manifest_sha256": sha(study / "camera/manifest.json"),
            "code_sha256": sha(Path(__file__)), "headers": headers, "selection": choices, "selection_order": ordered,
            "heads": heads, "points": calibration, "held_readings_at_freeze": 0}
    save(study / "freeze.json.gz", lock)
    print(json.dumps({"fit": len(fit_groups), "selection": len(select_groups), "held": len(held_groups),
                      "preferred": ordered[0], "ordered": ordered}, sort_keys=True), flush=True)


def assess(study, camera_roster=None):
    from scripts.audit_video_face_temporal import public_roster

    lock = read(study / "freeze.json.gz")
    roster = study / "held-roster.json"
    if (lock["code_sha256"] != sha(Path(__file__)) or lock["roster_sha256"] != sha(roster)
            or lock["camera_manifest_sha256"] != sha(study / "camera/manifest.json")):
        raise ValueError("Locked training or held input changed")
    panels = {}
    for key in ("held", "camera"):
        folder = study / key
        reference, current = _reference_journal(study / f"{key}-reference.jsonl")
        header, rows = attach(folder, study / f"{key}.jsonl", current, camera=key == "camera")
        if (reference["manifest_sha256"] != header["manifest_sha256"]
                or any(header[k] != lock["headers"][0][k] for k in ("weights", "frequency_sha256", "track_sha256", "readout_sha256"))):
            raise ValueError("A held readout differs from its fitted assets or native source clocks")
        rows = score_rows(rows, lock["heads"])
        panels[key] = {"header": header, "reference": reference, "rows": rows, "summary": tally(rows, lock["points"])}
    acquisition = public_roster(read(roster))
    cameras = [{k: v for k, v in item.items() if k != "path"} for item in read(camera_roster)] if camera_roster else []
    result = {"schema": SCHEMA, "freeze": lock, "panels": panels, "source_roster": acquisition, "camera_roster": cameras}
    verify(result)
    save(study / "result.json.gz", result)


def verify(result):
    lock = result["freeze"]
    assert result["schema"] == lock["schema"] == SCHEMA and lock["design"] == DESIGN and lock["held_readings_at_freeze"] == 0
    names = {f"{view}/{scope}" for view in VIEWS for scope in SCOPE}
    choices = lock["selection"]
    calibration = {key: points(choices, key) for key in names}
    assert calibration == lock["points"] and selection_order(choices, calibration) == lock["selection_order"]
    assert set(lock["heads"]) == names and set(result["panels"]) == {"held", "camera"}
    fit_keys, selected, held_keys = set(lock["fit_groups"]), {row["group"] for row in choices}, set(lock["held_groups"])
    assert not fit_keys & selected and not held_keys & (selected | fit_keys)
    for key, panel in result["panels"].items():
        rows = panel["rows"]
        assert panel["summary"] == tally(rows, calibration)
        groups = {row["group"] for row in rows}
        media = {"ordinary", "inswapper", "uniface"} if key == "held" else {"ordinary"}
        assert groups == held_keys if key == "held" else groups.isdisjoint(fit_keys | selected | held_keys)
        assert len({row["sha256"] for row in rows}) == len(rows) == len(groups) * 2 * len(media)
        assert {(row["group"], row["family"], row["delivery"]) for row in rows} == {
            (group, family, delivery) for group in groups for family in media for delivery in ("direct", "tandem")}
        for row in rows:
            assert residual.current(row["reference"]) == row["current"] and set(row["scores"]) == names
            assert row["qp"] == qp(row["quantization"], row["time_base"])
            assert all(value is None or (math.isfinite(value) and (not name.endswith("/strong") or is_strong(row["qp"])))
                       for name, value in row["scores"].items())
            tb = Fraction(row["time_base"])
            assert all(frame["pts"] * tb == Fraction(frame["index"], 25) for frame in row["frames"])
    return {"selected": lock["selection_order"][0], **{key: panel["summary"] for key, panel in result["panels"].items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    fitting = phases.add_parser("freeze")
    for field in ("study", "previous", "second", "earlier"):
        fitting.add_argument(field, type=Path)
    measurement = phases.add_parser("assess")
    measurement.add_argument("study", type=Path)
    measurement.add_argument("--camera-roster", type=Path)
    replay = phases.add_parser("verify")
    replay.add_argument("result", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.study, args.previous, args.second, args.earlier)
    elif args.phase == "assess":
        assess(args.study, args.camera_roster)
    else:
        print(json.dumps(verify(read(args.result)), sort_keys=True))
