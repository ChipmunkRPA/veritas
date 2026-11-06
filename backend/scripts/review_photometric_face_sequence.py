"""Freeze source-grouped development comparisons and audit fresh video localization."""
from __future__ import annotations

import argparse
import ast
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from app.detectors.video.decode import FrameTable
from app.evidence.face_intervals import RULE, bind_tracks, localize
from scripts.prepare_video_upgrade_eval import sha

MODES = ("current", "raw", "normalized")


def feature_references(paths: list[Path]) -> set[str]:
    """Accept only the reviewed SHA-helper import repair, with identical executable definitions."""
    current = Path(__file__).with_name("score_face_sequence_candidates.py")

    def comparable(path):
        body = ast.parse(path.read_text()).body
        definitions = [ast.dump(n, include_attributes=False) for n in body
                       if not isinstance(n, (ast.Import, ast.ImportFrom))]
        imports = []
        for node in body:
            if isinstance(node, ast.ImportFrom) and node.module == "prepare_video_upgrade_eval":
                node.module = "scripts.prepare_video_upgrade_eval"
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.dump(node, include_attributes=False))
        return definitions, sorted(imports)

    expected = comparable(current)
    accepted = {sha(current)}
    for path in paths:
        if comparable(path) != expected:
            raise ValueError("Development feature runtime differs beyond the reviewed SHA-helper import repair")
        accepted.add(sha(path))
    return accepted


def load(manifest: Path, baseline: Path, candidate: Path) -> tuple[dict, list[dict]]:
    sources = json.loads(manifest.read_text())["items"]
    with baseline.open() as stream:
        first = json.loads(next(stream))
        current = [json.loads(line) for line in stream]
    with candidate.open() as stream:
        header = json.loads(next(stream))
        proposals = [json.loads(line) for line in stream]
    expected = [r for r in sources if r["split"] == first["split"]]
    if (first["manifest_sha256"] != sha(manifest) or header["manifest_sha256"] != sha(manifest)
            or header["baseline_sha256"] != sha(baseline) or header["split"] != first["split"]
            or len(current) != len(expected) or len(proposals) != len(expected)):
        raise ValueError("Incomplete or mismatched video comparison denominator")
    rows = []
    for item, row, proposal in zip(expected, current, proposals, strict=True):
        if (any(row.get(k) != v for k, v in item.items()) or proposal["id"] != item["id"]
                or proposal["source_sha256"] != item.get("sha256")):
            raise ValueError("Video comparison changed a prepared source or its labels")
        points = sorted(row.get("points") or [], key=lambda p: p["frame_index"])
        span = None
        if points:
            table = FrameTable.from_json(row["table"])
            tracks = bind_tracks(points, row["exhibits"], table)
            intervals = localize(tracks, RULE)
            if intervals:
                span = [intervals[0].during[0].time_s, intervals[0].during[-1].time_s]
            else:
                peak = max(range(len(points)), key=lambda i: points[i]["score"])
                if points[peak]["score"] >= .5:
                    span = [table.seconds(points[max(0, peak-1)]["frame_index"]),
                            table.seconds(points[min(len(points)-1, peak+1)]["frame_index"])]
        if proposal["status"] not in {"ok", "not_applicable", "failed"}:
            raise ValueError("Invalid candidate case status")
        if proposal["status"] != "ok" and any(proposal.get(k) for k in ("raw_windows", "normalized_windows")):
            raise ValueError("A failed or unavailable candidate cannot supply review windows")
        for key in ("raw_windows", "normalized_windows"):
            for window in proposal.get(key, []):
                if (not math.isfinite(window["response"]) or not 0 <= window["start_s"] < window["end_s"]
                        or not points or window["end_s"] > table.seconds(len(table)-1) + 1e-6):
                    raise ValueError("Invalid candidate source-time window")
        rows.append({**item, "current_range": span, "baseline_scored": bool(points),
                     "baseline_status": row.get("check_status", "failed"), "proposal": proposal})
    return header, rows


def overlap(predicted: list[float] | None, actual: list[float] | None) -> float:
    if predicted is None or actual is None:
        return 0.
    union = max(predicted[1], actual[1])-min(predicted[0], actual[0])
    if union <= 0:
        return 0.
    return max(0., min(predicted[1], actual[1])-max(predicted[0], actual[0])) / union


def cases(rows: list[dict], mode: str, threshold: float | None) -> list[dict]:
    if mode not in MODES:
        raise ValueError("Unknown comparison mode")
    result = []
    for row in rows:
        windows = (row["proposal"].get(mode+"_windows", [])
                   if mode != "current" and row["proposal"]["status"] == "ok" else [])
        proposed = None
        if windows and threshold is not None:
            selected = max(windows, key=lambda w: (w["response"], -w["start_s"]))
            if selected["response"] >= threshold:
                proposed = [selected["start_s"], selected["end_s"]]
        span = proposed or row["current_range"]
        iou = overlap(span, row.get("edit_s")) if row["label"] else 0.
        result.append({"id": row["id"], "group": row["group"], "family": row["family"], "label": row["label"],
                       "source_status": row["status"], "candidate_status": row["proposal"]["status"],
                       "baseline_status": row["baseline_status"], "baseline_scored": row["baseline_scored"],
                       "windows_available": bool(windows), "range_s": span, "known_edit_s": row.get("edit_s"),
                       "used_candidate": proposed is not None, "iou": iou, "localized": iou >= .3})
    return result


def metrics(rows: list[dict]) -> dict:
    original = [r for r in rows if not r["label"]]
    edited = [r for r in rows if r["label"]]
    hard = [r for r in original if r["family"] != "original_videos"]
    return {"planned": len(rows), "source_groups": len({r["group"] for r in rows}),
            "originals": len(original), "edits": len(edited), "hard_controls": len(hard),
            "original_flags": sum(r["range_s"] is not None for r in original),
            "hard_control_flags": sum(r["range_s"] is not None for r in hard),
            "localized_edits": sum(r["localized"] for r in edited),
            "missed_localizations": sum(not r["localized"] for r in edited),
            "mean_edit_iou": sum(r["iou"] for r in edited) / len(edited) if edited else None,
            "baseline_scored_cases": sum(r["baseline_scored"] for r in rows),
            "baseline_failures": sum(r["baseline_status"] == "failed" for r in rows),
            "baseline_unassessed": sum(not r["baseline_scored"] for r in rows),
            "baseline_status_counts": dict(Counter(r["baseline_status"] for r in rows)),
            "candidate_failures": sum(r["candidate_status"] == "failed" for r in rows),
            "candidate_status_counts": dict(Counter(r["candidate_status"] for r in rows)),
            "candidate_window_cases": sum(r["windows_available"] for r in rows),
            "unavailable_sources": sum(r["source_status"] != "ready" for r in rows)}


def choose(rows: list[dict], mode: str) -> float | None:
    base = metrics(cases(rows, "current", None))
    values = {w["response"] for r in rows for w in r["proposal"].get(mode+"_windows", []) if w["response"] > 0}
    # None is a real, conservative candidate: preserve the current review check.
    thresholds = [None, *sorted(values, reverse=True)]

    def rank(threshold):
        result = metrics(cases(rows, mode, threshold))
        acceptable = (result["original_flags"] <= base["original_flags"] + 1
                      and result["hard_control_flags"] <= base["hard_control_flags"] + 1)
        return (acceptable, result["localized_edits"], -result["original_flags"], -result["hard_control_flags"],
                result["mean_edit_iou"] or 0., threshold if threshold is not None else math.inf)

    return max(thresholds, key=rank)


def summarize(rows: list[dict], thresholds: dict, *, confirmation: bool) -> dict:
    selected = {mode: cases(rows, mode, thresholds.get(mode)) for mode in MODES}
    result = {mode: metrics(items) for mode, items in selected.items()}
    base = result["current"]
    baseline = {r["id"]: r for r in selected["current"]}
    for mode in ("raw", "normalized"):
        values = result[mode]
        differences = {}
        for row in selected[mode]:
            if row["label"]:
                differences.setdefault(row["group"], []).append(row["iou"] - baseline[row["id"]]["iou"])
        delta = [float(np.mean(d)) for d in differences.values()]
        interval = None
        if len(delta) >= 2:
            sample = np.random.default_rng(20261105).choice(delta, (5000, len(delta)), replace=True).mean(axis=1)
            interval = np.quantile(sample, [.025, .975]).tolist()
        values["paired_iou"] = {"candidate_minus_current": float(np.mean(delta)) if delta else None,
                                "source_group_bootstrap_95_percent": interval, "source_groups": len(delta)}
        values["meets_count_tradeoff"] = (values["localized_edits"] > base["localized_edits"]
            and values["original_flags"] <= base["original_flags"] + 1
            and values["hard_control_flags"] <= base["hard_control_flags"] + 1
            and values["mean_edit_iou"] is not None and base["mean_edit_iou"] is not None
            and values["mean_edit_iou"] >= base["mean_edit_iou"] and values["candidate_failures"] == 0)
        values["qualifies_for_scoped_local_adoption"] = confirmation and values["meets_count_tradeoff"]
    result["cases"] = selected
    return result


def write(path: Path, value: dict) -> None:
    with path.open("x") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")


def freeze(args) -> dict:
    accepted = feature_references(args.feature_reference)
    headers, rows = [], []
    for baseline, candidate in zip(args.baseline, args.candidate, strict=True):
        header, part = load(args.manifest, baseline, candidate)
        if header["role"] != "consumed development":
            raise ValueError("Only consumed development inputs may select the comparison rules")
        headers.append(header)
        rows.extend(part)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Development receipts repeat a source case")
    if len({r["group"] for r in rows}) < 20:
        raise ValueError("At least twenty consumed development source groups are required")
    if sum(not r["label"] for r in rows) < 20 or sum(bool(r["label"]) for r in rows) < 20:
        raise ValueError("Development needs at least twenty genuine and twenty manipulated cases")
    first = headers[0]
    runner = sha(Path(__file__).with_name("score_photometric_face_sequence.py"))
    for header in headers:
        if (header["runner_sha256"] != runner or header["feature_runtime_sha256"] not in accepted
                or header["normalization"] != first["normalization"] or header["weights"] != first["weights"]):
            raise ValueError("Development methods, feature runtime, or checkpoints differ")
    future = json.loads(args.confirmation_manifest.read_text())["items"]
    if {r["group"] for r in rows} & {r["group"] for r in future}:
        raise ValueError("Video source groups cross development and confirmation")
    thresholds = {mode: choose(rows, mode) for mode in ("raw", "normalized")}
    development = summarize(rows, thresholds, confirmation=False)
    eligible = [mode for mode in ("raw", "normalized") if development[mode]["meets_count_tradeoff"]]
    primary = max(eligible, key=lambda mode: (development[mode]["localized_edits"],
                  -development[mode]["original_flags"], development[mode]["mean_edit_iou"])) if eligible else None
    record = {"schema": "chipmunk.photometric-decision/1", "development_groups": sorted({r["group"] for r in rows}),
              "confirmation_manifest_sha256": sha(args.confirmation_manifest),
              "candidate_runner_sha256": runner, "feature_runtime_sha256": sha(Path(__file__).with_name("score_face_sequence_candidates.py")),
              "reviewer_sha256": sha(Path(__file__)), "weights": first["weights"],
              "normalization": first["normalization"], "thresholds": thresholds,
              "accepted_development_feature_sha256": sorted(accepted),
              "feature_runtime_note": "SHA-helper import packaging/format repairs only; non-import ASTs and other import declarations match.",
              "development_receipts_sha256": [sha(p) for p in args.candidate],
              "baseline_receipts_sha256": [sha(p) for p in args.baseline],
              "development": development, "primary_candidate": primary,
              "primary_selection": "Among development candidates meeting the frozen count tradeoff: most localized edits, "
                                   "then fewer genuine flags, then higher mean IoU. Secondary test outcomes cannot select a default.",
              "adoption": "More edits localized at IoU >=0.30, nondecreasing mean edited IoU, at most one extra "
                          "genuine review cue and one extra ordinary-edit cue, and no unexplained candidate execution failures. "
                          "Only a fresh confirmation result may qualify; all comparisons concern inspection ranges, not AI confidence.",
              "scope": "Review interval localization only; no customer confidence, calibration, or default changes."}
    write(args.output, record)
    return record


def review(args) -> dict:
    lock = json.loads(args.lock.read_text())
    header, rows = load(args.manifest, args.baseline, args.candidate)
    if (lock["confirmation_manifest_sha256"] != sha(args.manifest)
            or lock["reviewer_sha256"] != sha(Path(__file__))
            or header["lock_sha256"] != sha(args.lock) or header["role"] != "confirmation"
            or header["runner_sha256"] != lock["candidate_runner_sha256"]
            or header["feature_runtime_sha256"] != lock["feature_runtime_sha256"]
            or header["weights"] != lock["weights"] or header["normalization"] != lock["normalization"]
            or set(lock["development_groups"]) & {r["group"] for r in rows}):
        raise ValueError("Fresh video results do not match frozen development and source identities")
    result = {"schema": "chipmunk.photometric-review/1", "lock_sha256": sha(args.lock),
              "manifest_sha256": sha(args.manifest), "candidate_sha256": sha(args.candidate),
              "baseline_sha256": sha(args.baseline), "confirmation": summarize(rows, lock["thresholds"], confirmation=True),
              "primary_candidate": lock["primary_candidate"],
              "limits": "Matched, short clips from two swap families. Training/subject overlap beyond declared source and donor IDs "
                        "is unknown. Review ranges and feature responses do not prove AI generation or supply overall confidence. "
                        "Source groups, including two swaps and two controls, are resampled together for paired IoU uncertainty."}
    for mode in ("raw", "normalized"):
        result["confirmation"][mode]["qualifies_for_scoped_local_adoption"] = (
            mode == lock["primary_candidate"] and result["confirmation"][mode]["meets_count_tradeoff"])
    result["families"] = {family: {mode: metrics(cases([r for r in rows if r["family"] == family],
                                                     mode, lock["thresholds"].get(mode))) for mode in MODES}
                          for family in sorted({r["family"] for r in rows})}
    write(args.output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("freeze")
    select.add_argument("manifest", type=Path)
    select.add_argument("confirmation_manifest", type=Path)
    select.add_argument("output", type=Path)
    select.add_argument("--baseline", type=Path, action="append", required=True)
    select.add_argument("--candidate", type=Path, action="append", required=True)
    select.add_argument("--feature-reference", type=Path, action="append", default=[])
    result = commands.add_parser("review")
    for name in ("manifest", "baseline", "candidate", "lock", "output"):
        result.add_argument(name, type=Path)
    args = parser.parse_args()
    record = freeze(args) if args.command == "freeze" else review(args)
    values = record["development"] if args.command == "freeze" else record["confirmation"]
    print(json.dumps({k: v for k, v in values.items() if k != "cases"}, indent=2))
