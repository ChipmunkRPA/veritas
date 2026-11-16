# SPDX-License-Identifier: MIT
"""Test whether clean speech responses can train short-edit localization in noise."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_local_speech_normality as panels
from scripts import audio_short_edit_detector_training as base
from scripts import audio_short_edit_ssl_transfer as features

SOURCE = base.SOURCE
SCHEMA = "chipmunk.audio-clean-teacher-transfer/1"
ARMS = ("paired_labels", "aligned_teacher", "shifted_teacher")
SHIFT = 110
METHOD = {
    "teacher": "Fit a separate clean waveform teacher for each of three disjoint fitting-reader folds. "
               "Read paired dry classes at their exact sample clocks. Keep a group's centered teacher responses "
               "only when its synthetic core scores above every matched dry control at the same time.",
    "student": "Keep the earlier temporal network, paired noisy and dry supervision, seeds and optimizer. "
               "For the noisy view, add a Huber penalty against the held-out dry logits after subtracting "
               "each channel's per-clip mean. Weight edit interiors four times for the content response "
               "and use the same weight near the known joins for the transition response. "
               "Scale the two penalties by 0.30 and 0.075.",
    "comparison": "Remove the teacher term for the first arm. Shift the same clean traces by 1.10 seconds "
                  "for the timing control. Every arm sees only one noisy upload at inference.",
    "calibration": "Each arm and fit direction uses separate noisy genuine file maxima and its isolated rooms. "
                   "Keep both the existing strict and wider point estimators.",
    "overlap": "Correct ranges cover half the labeled edit and at least 30% of the union on the delivered sample clock.",
}


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def save(path, value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data + b"\n")


def runtime(speech):
    import torch

    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free

    SOURCE.check(SOURCE.path_sha(speech) == features.SPEECH, "Speech admission asset changed")
    network_free()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    return SpeechActivity(speech)


def _teachers(data, plan, family, frame_features):
    import numpy as np
    import torch

    fitting = [row for row in plan["sources"] if row["role"] == "fit"]
    readers = sorted({row["reader_sha256"] for row in fitting})
    assert len(readers) == 6
    dry = {(row["group"], row["kind"]): row for row in plan["rows"]
           if row["role"] == "fit" and row["condition"] == "dry"}

    def dry_features(path, row):
        return frame_features(path, dry[row["group"], row["kind"]])

    clean, region, edge = base.fit_input(data, plan, family, frame_features)
    traces = torch.empty((len(fitting), 4, 2, base.FRAMES))
    fit_notes = []
    for first in range(0, len(readers), 2):
        withheld = set(readers[first:first + 2])
        indices = [i for i, source in enumerate(fitting) if source["reader_sha256"] in withheld]
        train_roster = [row for row in fitting if row["reader_sha256"] not in withheld]
        model, center, scale, _ = base.train(data, plan | {"sources": train_roster}, family, None, feature_of=dry_features)
        normalized = (clean[indices, 0] - center[:, None]) / scale[:, None]
        with torch.inference_mode():
            trace = model(torch.from_numpy(normalized).flatten(0, 1)).reshape(len(indices), 4, 2, base.FRAMES)
            traces[indices] = trace
        fit_notes.append({"held_reader_hashes": sorted(withheld), "held_prompts": len(indices), "fit_prompts": len(train_roster)})
    centered = traces - traces.mean(dim=-1, keepdim=True)
    region, edge = torch.from_numpy(region), torch.from_numpy(edge)
    mid = (centered[:, :, 0] * region[:, None]).sum(-1) / region.sum(-1, keepdim=True)
    reliable = mid[:, -1] > mid[:, :-1].amax(dim=1)
    SOURCE.check(np.isfinite(centered.numpy()).all(), "Teacher returned nonfinite frame readings")
    return centered, reliable, (1 + 3 * region, 1 + 3 * edge), fit_notes


def _objective(target, reliable, weights, shifted):
    import torch
    from torch.nn.functional import smooth_l1_loss

    if shifted:
        target = target.roll(SHIFT, dims=-1)

    def objective(_model, _center, _scale, indices, local, edges):
        response = torch.stack((local, edges), dim=2)
        response = response - response.mean(dim=-1, keepdim=True)
        penalties = smooth_l1_loss(response, target[indices], reduction="none")
        content = (penalties[:, :, 0] * weights[0][indices, None]).mean(dim=(1, 2))
        boundary = (penalties[:, :, 1] * weights[1][indices, None]).mean(dim=(1, 2))
        return ((.30 * content + .075 * boundary) * reliable[indices]).mean()

    return objective


def _rows(plan, role, family, *, calibration=False):
    return [row for row in base.relevant(plan, role, family, calibration=calibration)
            if row["condition"] == "competing_speech"]


def _attach(rows, readings):
    return [{"kind": row["kind"], "group": row["group"], "pcm_sha256": row["pcm_sha256"],
             "truth": [row["truth"]] if row["truth"] else [], "best": measured["best"]}
            for row, measured in zip(rows, readings, strict=True)]


def fit(args):
    import torch
    from safetensors.torch import save_file

    plan, digest = base.load_plan(args.data)
    origin, feature_of = features.feature_reader(args.cache, plan, digest)
    sources = {name: SOURCE.path_sha(getattr(args, name)) for name in ("retained_plan", "fresh_plan", "stereo_plan")}
    freeze = {
        "schema": SCHEMA, "method": METHOD, "arms": ARMS, "source_sha256": digest, "encoding": origin["encoder"],
        "feature_ledger_sha256": SOURCE.path_sha(args.cache / "features.json"), "fitting_code_sha256": SOURCE.path_sha(Path(__file__)),
        "previous_serving_receipt_sha256": SOURCE.path_sha(panels.PRIMARY),
        "held_serving_receipt_sha256": SOURCE.path_sha(args.current), "held_input_plans": sources,
    }
    args.output.mkdir(mode=0o700)
    save(args.output / "freeze.json", freeze)
    activity = runtime(args.speech)
    for known, unknown in (base.sources.FAMILIES, base.sources.FAMILIES[::-1]):
        trace, reliable, weights_by_frame, folds = _teachers(args.data, plan, known, feature_of)
        for arm in ARMS:
            objective = _objective(trace, reliable, weights_by_frame, arm == ARMS[2]) if arm != ARMS[0] else None
            model, center, scale, losses = base.train(args.data, plan, known, None, feature_of=feature_of,
                                                     extra_objective=objective)
            target = args.output / f"{known}-{arm}.safetensors"
            save_file({**model.state_dict(), "feature_mean": torch.from_numpy(center),
                       "feature_scale": torch.from_numpy(scale)}, str(target))
            ordinary = _rows(plan, "calibration", known, calibration=True)
            measured = base.score_rows(model, (center, scale), args.data, ordinary, None, activity, feature_of=feature_of)
            save(target.with_suffix(".json"), {
                "trained_family": known, "held_family": unknown, "arm": arm,
                "file": target.name, "sha256": SOURCE.path_sha(target), "teacher_folds": folds,
                "reliable_teacher_prompts": int(reliable.sum()), "fit_loss": losses,
                "points": base.scores_at_points(ordinary, measured), "calibration": _attach(ordinary, measured),
            })
            print(f"Sealed {known} {arm} with {int(reliable.sum())}/{len(reliable)} eligible clean teacher prompts.", flush=True)


def _head(root, path):
    from safetensors.torch import load_file

    seal = read(path)
    weights = root / seal["file"]
    SOURCE.check(SOURCE.path_sha(weights) == seal["sha256"], "Frozen head weights differ")
    tensors = load_file(str(weights))
    center, scale = [tensors.pop(key).numpy() for key in ("feature_mean", "feature_scale")]
    model = base.model().eval()
    model.load_state_dict(tensors, strict=True)
    return seal, model, center, scale


def score(args):
    import torch

    plan, digest = base.load_plan(args.data)
    origin, feature_of = features.feature_reader(args.cache, plan, digest)
    freeze = read(args.fit / "freeze.json")
    actual_plans = {f"{name}_plan": SOURCE.path_sha(getattr(args, name) / "freeze.json")
                    for name in ("retained", "fresh", "stereo")}
    SOURCE.check(freeze["schema"] == SCHEMA and freeze["method"] == METHOD and freeze["source_sha256"] == digest
                 and freeze["encoding"] == origin["encoder"] and freeze["held_input_plans"] == actual_plans
                 and freeze["previous_serving_receipt_sha256"] == SOURCE.path_sha(panels.PRIMARY)
                 and freeze["held_serving_receipt_sha256"] == SOURCE.path_sha(args.current)
                 and freeze["fitting_code_sha256"] == SOURCE.path_sha(Path(__file__))
                 and freeze["feature_ledger_sha256"] == SOURCE.path_sha(args.cache / "features.json"),
                 "Evaluation inputs differ from the sealed fit")
    heads = [_head(args.fit, args.fit / f"{known}-{arm}.json")
             for known in base.sources.FAMILIES for arm in ARMS]
    activity = runtime(args.speech)
    encode = features.frozen_encoder(args.encoder)
    prior = read(args.current)
    current = {(row["panel"], row["pcm_sha256"]): row["current"] for row in prior["rows"]}
    primary, retained = SOURCE.read(panels.PRIMARY), read(panels.SINE)
    known = {row["pcm_sha256"]: row for row in plan["rows"]}
    measured, cache = [], {}
    with torch.inference_mode():
        for row, pcm, saved in panels.inputs(args, plan, primary, retained):
            key = row["pcm_sha256"]
            if key not in cache:
                raw = feature_of(args.data, known[key]) if key in known else encode(pcm)
                speech = activity.spans(pcm)
                result = {}
                for seal, model, center, scale in heads:
                    local, edge = model(torch.from_numpy(((raw - center[:, None]) / scale[:, None])[None]))[0].numpy()
                    result[f"{seal['trained_family']}/{seal['arm']}"] = base.best_interval(local, edge, speech)
                cache[key] = result
            app_reading = current[row["panel"], key]
            if saved is not None:
                SOURCE.check(saved == app_reading, "Serving response changed for the exact input")
            measured.append({k: row[k] for k in ("pcm_sha256", "panel", "room", "group", "kind", "truth")}
                            | {"current": app_reading, "candidates": cache[key]})
            if len(measured) % 40 == 0:
                print(f"Measured {len(measured)} held mixtures.", flush=True)
    SOURCE.check(Counter((row["panel"], row["pcm_sha256"]) for row in measured)
                 == Counter((row["panel"], row["pcm_sha256"]) for row in prior["rows"]), "Held cases changed or are missing")
    document = {"freeze": freeze, "heads": [head[0] for head in heads], "rows": measured}
    document["results"] = recount(document)
    save(args.output, document)
    print(json.dumps(document["results"], sort_keys=True), flush=True)


def _overlap(found, truth):
    if found is None:
        return False
    left, right = found
    return any((inside := max(0, min(right, b) - max(left, a))) * 2 >= b - a
               and 10 * inside >= 3 * (max(right, b) - min(left, a)) for a, b in truth)


def _response(row, armed, family, point):
    if armed == "current":
        return row["current"]["range"], row["current"]["assessed"]
    selected = row["candidates"][f"{family}/{armed}"]
    return (selected["range"] if selected and selected["score"] > point else None), selected is not None


def _summarize(rows, candidate, family, point, *, fallback=False):
    counters = Counter({key: 0 for key in ("edits", "useful", "outside", "missed", "ordinary", "ordinary_alerts",
                                          "isolated", "isolated_alerts", "unassessed", "useful_before_cutoff", "clear_useful")})
    controls = defaultdict(list)
    decisions = []
    for row in rows:
        selected, assessed = _response(row, candidate, family, point)
        if fallback and row["current"]["range"] is not None:
            selected, assessed = row["current"]["range"], True
        counters["unassessed"] += not assessed
        if row["truth"]:
            counters["edits"] += 1
            correctly = _overlap(selected, row["truth"])
            counters["useful"] += correctly
            counters["outside"] += selected is not None and not correctly
            counters["missed"] += not correctly
            if candidate != "current":
                proposal = row["candidates"][f"{family}/{candidate}"]
                counters["useful_before_cutoff"] += bool(proposal) and _overlap(proposal["range"], row["truth"])
            if correctly:
                decisions.append((row, selected))
        else:
            key = "isolated" if row["kind"] == "background" else "ordinary"
            counters[key] += 1
            counters[key + "_alerts"] += selected is not None
            if selected is not None and key == "ordinary":
                controls[row["group"]].append(selected)
    counters["clear_useful"] = sum(not any(_overlap(selected, row["truth"]) for selected in controls[row["group"]])
                                    for row, _ in decisions)
    return dict(counters)


def recount(document):
    heads = document["heads"]
    assert len({(head["trained_family"], head["arm"]) for head in heads}) == len(heads) == 6
    result = {}
    for head in heads:
        calibration = head["calibration"]
        assert all(not row["truth"] for row in calibration)
        assert base.scores_at_points([r | {"truth": None} for r in calibration], calibration) == head["points"]
        family, other, arm = (head[key] for key in ("trained_family", "held_family", "arm"))
        selected = defaultdict(list)
        for row in document["rows"]:
            if (row["panel"] == "phoneme" and row["kind"].startswith(("edited/", "resampled/"))
                    and row["kind"].split("/", 1)[1] != other):
                continue
            selected[row["panel"]].append(row)
        result[f"{family}/{arm}"] = {panel: {
            "current": _summarize(rows, "current", family, 0),
            **{name: _summarize(rows, arm, family, point) for name, point in head["points"].items()},
            "fallback": _summarize(rows, arm, family, head["points"]["wider"], fallback=True),
        } for panel, rows in sorted(selected.items())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    fitting = commands.add_parser("fit")
    for name in ("data", "cache", "speech", "current", "retained_plan", "fresh_plan", "stereo_plan", "output"):
        fitting.add_argument(name, type=Path)
    held = commands.add_parser("score")
    for name in ("data", "cache", "speech", "current", "retained", "fresh", "stereo", "encoder", "fit", "output"):
        held.add_argument(name, type=Path)
    review = commands.add_parser("review")
    review.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "fit":
        fit(args)
    elif args.phase == "score":
        score(args)
    else:
        document = read(args.receipt)
        actual = recount(document)
        SOURCE.check(actual == document["results"], "Reported transfer outcomes differ")
        print(json.dumps(actual, sort_keys=True))


if __name__ == "__main__":
    main()
