# SPDX-License-Identifier: MIT
"""Freeze and validate an additional context check on eye-hold proposals."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from scripts import video_eye_hold_panel as first
from scripts import video_eye_texture_hold_panel as texture_panel
from scripts.video_cropped_reverse_panel import load, save, sha

SCHEMA = "chipmunk.eye-hold-context-check/1"
ARMS = {
    "collar": (.015, .035, .065, .100),
    "opposite": (.025, .055, .090, .140),
    "both": (.015, .035, .065, .100),
}
CONTEXT_ROLES = ("whole_hold", "picture_jump")


def admits(reading, arm, point):
    if reading is None:
        return False
    if arm == "collar":
        return reading["collar_gap"] >= point
    if arm == "opposite":
        return reading["opposite_gap"] >= point
    if arm == "both":
        return reading["collar_gap"] >= point and reading["opposite_gap"] >= .055
    raise ValueError("Unsupported eye-context comparison")


def counts(rows, choice=None):
    useful, ordinary, misplaced, global_flags = 0, 0, 0, 0
    groups = set()
    for row in rows:
        before, truth = row["texture"], row["truth"]
        if before is None or choice and not admits(row["context"], *choice):
            continue
        if first._matches(before, truth):
            useful += 1
            groups.add(row["group"])
        elif truth:
            misplaced += 1
        else:
            ordinary += 1
            global_flags += row["role"] in CONTEXT_ROLES
    return {"useful": useful, "groups": len(groups), "ordinary": ordinary, "misplaced": misplaced,
            "global_flags": global_flags}


def choose(rows):
    trials = [{"arm": arm, "point": point, **counts(rows, (arm, point))}
              for arm, points in ARMS.items() for point in points]
    before = counts(rows)
    # Retain at least half of the old useful locations during development.
    # Among those variants, minimize ordinary and whole-picture alerts.
    viable = [arm for arm in trials if 2 * arm["useful"] >= before["useful"]]
    selected = min(viable, key=lambda arm: (
        arm["global_flags"], arm["ordinary"], arm["misplaced"], -arm["useful"], -arm["groups"],
    ), default=None)
    return {"unfiltered": before, "trials": trials, "selected": selected}


def _pins():
    source = Path(__file__).parent
    return texture_panel._pins() | {path.name: sha(path) for path in
                                   (Path(__file__), source / "video_eye_context_guard.py")}


def _layout(groups, media):
    return {row["index"]: (row, media / f"{row['index']:02d}-ordinary.mp4") for row in groups}


def develop(priors, base_folders, output):
    import cv2

    from app.detectors.video import mouth_local_hold
    from scripts import video_eye_context_guard as guard
    from scripts import video_eye_texture_hold as texture

    cv2.setNumThreads(1)
    detector, weight = first._locator()
    rows = []
    commitments = []
    for prior, media in zip(priors, base_folders, strict=True):
        ledger = load(prior)
        cases = ledger["rows"]
        layout = _layout(ledger["roster"]["groups"], media)
        normalized = ledger.get("focus") == texture_panel.SCHEMA
        commitments.append({"file": prior.name, "sha256": sha(prior), "groups": len(layout)})
        for index, (source, asset) in layout.items():
            local = [case for case in cases if case["index"] == index]
            if not local:
                continue
            if sha(asset) != source["files"]["ordinary"]["sha256"]:
                raise ValueError("A development source differs from its pinned publisher bytes")
            prepared = first._prepare(asset, source["offset"])
            with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
                for case in local:
                    point = {"context": None, "texture": case["candidate"] if normalized else None}
                    if point["texture"] is not None or not normalized:
                        video = Path(scratch) / case["delivery"]
                        first._encode(prepared, case["role"], source["eye_boxes"], video)
                        if sha(video) != case["sha256"]:
                            raise ValueError("A rerendered development input differs from its held commitment")
                        _, frames = mouth_local_hold.native(video)
                        if not normalized:
                            _, point["texture"] = texture.observe(frames, detector, texture_panel.POINT, guard=True)
                        point["context"] = guard.measure(frames, detector, point["texture"])
                        video.unlink()
                    rows.append({key: case[key] for key in ("group", "role", "truth", "sha256")} | point)
        print("Context development", prior.name, len(rows), flush=True)
    selection = choose(rows)
    save(output, {"schema": SCHEMA, "rows": rows, "selection": selection, "weight_sha256": weight,
                  "priors": commitments, "method_sha256": _pins()})
    print(selection["selected"], flush=True)


def freeze(roster, sources, development, prior_receipts, output):
    before = load(development)
    if before["method_sha256"] != _pins() or before["selection"] != choose(before["rows"]):
        raise ValueError("The frozen development sweep changed")
    winner = before["selection"]["selected"]
    if winner is None:
        raise ValueError("The context comparison did not retain half the development edits")
    plans = [load(path) for path in prior_receipts]
    for prior in plans:
        texture_panel._distinct(load(roster)["groups"], prior["roster"]["groups"])
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        base = Path(scratch) / "native-authoring.json"
        first.freeze(roster, sources, base)
        plan = load(base)
    save(output, plan | {"focus": SCHEMA, "development_sha256": sha(development), "selection": winner,
                         "priors": before["priors"], "method_sha256": _pins()})
    print("Frozen eye context", sha(output), flush=True)


def score(plan_path, sources, output):
    import cv2

    from app.detectors.video import mouth_local_hold
    from scripts import video_eye_context_guard as guard
    from scripts import video_eye_texture_hold as texture

    cv2.setNumThreads(1)
    plan = load(plan_path)
    if plan["focus"] != SCHEMA or plan["method_sha256"] != _pins() or plan["gate"] != first.GATE:
        raise ValueError("The prospective context rule changed")
    detector, weight = first._locator()
    if weight != plan["geometry_weight_sha256"]:
        raise ValueError("The pinned face locator changed")
    winner = plan["selection"]
    rows = []
    expected = [row for row in plan["cases"] if row["split"] == "held"]
    for index, (group, asset) in _layout(plan["roster"]["groups"], sources).items():
        if group["role"] != "held":
            continue
        if sha(asset) != group["files"]["ordinary"]["sha256"]:
            raise ValueError("A held capture differs from the pinned publisher data")
        prepared = first._prepare(asset, group["offset"])
        with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
            for case in (row for row in expected if row["index"] == index):
                video = Path(scratch) / case["delivery"]
                first._encode(prepared, case["role"], group["eye_boxes"], video)
                baseline = first._measure(video, detector)
                table, frames = mouth_local_hold.native(video)
                status, proposal = texture.observe(frames, detector, texture_panel.POINT, guard=True)
                context = guard.measure(frames, detector, proposal)
                passed = proposal is not None and admits(context, winner["arm"], winner["point"])
                match = proposal if passed else None
                decision = status if status != "matched" or passed else "no_match"
                rows.append(case | baseline | {"texture": proposal, "context": context, "candidate": match, "status": decision,
                                               "verified_anchors": first._anchors(video, table, frames, match) if match else {}})
                video.unlink()
        save(output, plan | {"rows": rows, "split": "held", "complete": len(rows) == len(expected),
                             "plan_sha256": sha(plan_path), "summary": first.summary(rows),
                             "texture_unfiltered": counts(rows)})
        print("Held eye context", len(rows), "/", len(expected), flush=True)


def verify(receipt, development):
    rows = load(receipt)
    dev = load(development)
    if (sha(development) != rows["development_sha256"] or rows["focus"] != SCHEMA or dev["schema"] != SCHEMA
            or rows["selection"] != choose(dev["rows"])["selected"] or rows["texture_unfiltered"] != counts(rows["rows"])):
        raise ValueError("The reported context selection is inconsistent")
    chosen = rows["selection"]
    for row in rows["rows"]:
        accepted = admits(row["context"], chosen["arm"], chosen["point"])
        assert row["candidate"] == (row["texture"] if accepted else None)
    first.verify(receipt)
    print("Unfiltered texture check", rows["texture_unfiltered"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    tune = commands.add_parser("develop")
    tune.add_argument("output", type=Path)
    tune.add_argument("--prior", type=Path, action="append", required=True)
    tune.add_argument("--sources", type=Path, action="append", required=True)
    prescribe = commands.add_parser("freeze")
    prescribe.add_argument("roster", type=Path)
    prescribe.add_argument("sources", type=Path)
    prescribe.add_argument("development", type=Path)
    prescribe.add_argument("plan", type=Path)
    prescribe.add_argument("--prior", type=Path, action="append", required=True)
    measure = commands.add_parser("score")
    measure.add_argument("plan", type=Path)
    measure.add_argument("sources", type=Path)
    measure.add_argument("receipt", type=Path)
    check = commands.add_parser("verify")
    check.add_argument("receipt", type=Path)
    check.add_argument("development", type=Path)
    args = parser.parse_args()
    if args.action == "develop":
        develop(args.prior, args.sources, args.output)
    elif args.action == "freeze":
        freeze(args.roster, args.sources, args.development, args.prior, args.plan)
    elif args.action == "score":
        score(args.plan, args.sources, args.receipt)
    else:
        verify(args.receipt, args.development)


if __name__ == "__main__":
    main()
