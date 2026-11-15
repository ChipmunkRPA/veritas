# SPDX-License-Identifier: MIT
"""Test a clean-feature reconstruction before short speech-edit localization."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts import audio_short_edit_detector_training as detector
from scripts import audio_short_edit_ssl_transfer as layers

SOURCE = detector.SOURCE
SCHEMA = "chipmunk.audio-short-edit-noise-projection/1"
OFFSETS = (-2, 0, 2)
PENALTY = .25
FALSE_GROUP_BUDGETS = (0, 1, 2, 4)
METHOD = {
    "input": layers.EXPERIMENT,
    "fit": "Only untouched voices from the original fitting readers. Standardize on paired clean and mixed "
           "frames. Fit a ridge predictor for the clean-minus-mixed residual and a zero target on clean frames. "
           "The input stacks the current frame and frames at ±20 ms, with edge hold and an intercept. "
           "All frame pairs have equal weight. The ridge penalty is relative to the number of fit frames.",
    "context_offsets_in_head_frames": list(OFFSETS),
    "ridge_penalty_per_frame": PENALTY,
    "inference": "Apply the single fitted map to each uploaded clip before the original localized head. "
                 "Neither a dry recording, an isolated background nor an overlap label is needed at inference.",
    "comparisons": "Freeze the paired fit, then fit and calibrate the same short-edit classifier in two held-generator "
                   "directions. Reuse the prior windows and admission. Report clean-feature errors, true locations, "
                   "matched-control collisions and background flags. Transfer to the existing disjoint meeting/generator panel.",
}


def contexts(values):
    import numpy as np

    clock = np.arange(values.shape[1])
    return np.concatenate([values[:, np.clip(clock + offset, 0, values.shape[1] - 1)].T for offset in OFFSETS], axis=1)


def ridge(plan, data, raw):
    import numpy as np

    fit = {(row["group"], row["condition"]): row for row in plan["rows"]
           if row["role"] == "fit" and row["kind"] == "untouched"}
    groups = sorted({group for group, _ in fit})
    dry, noisy = (np.stack([raw(data, fit[group, condition]) for group in groups]).astype(np.float64)
                  for condition in ("dry", "competing_speech"))
    features = np.concatenate((dry, noisy))
    mean = features.mean(axis=(0, 2))
    scale = np.maximum(.001, features.std(axis=(0, 2)))
    standardized = (features - mean[None, :, None]) / scale[None, :, None]
    matrix = np.vstack([contexts(values) for values in standardized])
    design = np.column_stack((matrix, np.ones(len(matrix))))
    residual = standardized[:len(groups)] - standardized[len(groups):]
    target = np.vstack((np.zeros((len(matrix) // 2, len(mean))), residual.transpose(0, 2, 1).reshape(-1, len(mean))))
    regularizer = np.eye(design.shape[1]) * (PENALTY * len(design))
    regularizer[-1, -1] = 0.
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
    return np.vstack((mean, scale, weights)).astype("<f8"), len(groups)


def project(values, state):
    import numpy as np

    mean, scale = state[:2]
    normalized = (values - mean[:, None]) / scale[:, None]
    residual = contexts(normalized) @ state[2:-1] + state[-1]
    return (values + residual.T * scale[:, None]).astype("<f4")


def aligned_distances(plan, data, raw, state):
    import numpy as np

    held = {(row["group"], row["condition"], row["kind"]): row for row in plan["rows"]
            if row["role"] == "held" and row["kind"] != "background"}
    scale = state[1, :, None]
    distances = []
    for (group, condition, kind), mixed in held.items():
        if condition != "competing_speech":
            continue
        target = raw(data, held[group, "dry", kind])
        values = raw(data, mixed)
        output = project(values, state)
        region = mixed["truth"]
        columns = (slice(0, detector.FRAMES) if region is None else
                   slice((region[0] + detector.HOP - 1) // detector.HOP, (region[1] + detector.HOP - 1) // detector.HOP))
        before = (values[:, columns] - target[:, columns]) / scale
        after = (output[:, columns] - target[:, columns]) / scale
        distances.append({"kind": kind, "group": group, "before_mse": round(float(np.mean(before ** 2)), 8),
                          "after_mse": round(float(np.mean(after ** 2)), 8)})
    return distances


def runtime(threads):
    import torch

    from scripts.audio_strong_traffic_view import network_free

    network_free()
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    return torch


def train(args):
    import numpy as np
    from safetensors.torch import save_file

    from app.detectors.audio.upgraded import SpeechActivity

    torch = runtime(2)
    plan, digest = detector.load_plan(args.data)
    origin, raw = layers.feature_reader(args.cache, plan, digest)
    SOURCE.check(SOURCE.path_sha(args.speech) == layers.SPEECH, "Speech activity asset differs")
    args.output.mkdir(mode=0o700)
    state, groups = ridge(plan, args.data, raw)
    projection = args.output / "projection.npy"
    np.save(projection, state, allow_pickle=False)
    lock = origin | {"schema": SCHEMA, "method": METHOD, "clean_fit_groups": groups,
                     "projection_sha256": SOURCE.path_sha(projection),
                     "feature_ledger_sha256": SOURCE.path_sha(args.cache / "features.json")}
    SOURCE.write(args.output / "freeze.json", lock)

    def features(data, row):
        return project(raw(data, row), state)

    activity = SpeechActivity(args.speech)
    fitted = []
    for known, unknown in (detector.sources.FAMILIES, detector.sources.FAMILIES[::-1]):
        classifier, mean, scale, losses = detector.train(args.data, plan, known, None, feature_of=features)
        weights = args.output / f"{known}.safetensors"
        save_file({**classifier.state_dict(), "feature_mean": torch.from_numpy(mean), "feature_scale": torch.from_numpy(scale)},
                  str(weights))
        rows = detector.relevant(plan, "calibration", known, calibration=True)
        scores = detector.score_rows(classifier, (mean, scale), args.data, rows, None, activity, feature_of=features)
        seal = {"trained_family": known, "held_family": unknown, "calibration_scores": scores, "losses": losses,
                "points": detector.scores_at_points(rows, scores), "weights_sha256": SOURCE.path_sha(weights)}
        SOURCE.write(args.output / f"{known}-points.json", seal)
        fitted.append((classifier, (mean, scale), seal))
        print(f"Sealed the {known} fit before opening held scores.", flush=True)
    cases = []
    for classifier, norm, seal in fitted:
        rows = detector.relevant(plan, "held", seal["held_family"])
        scores = detector.score_rows(classifier, norm, args.data, rows, None, activity, feature_of=features)
        cases.append(seal | {"held_scores": scores})
    SOURCE.write(args.output / "candidate.json", {"freeze": lock, "cases": cases,
                                                  "aligned_distances": aligned_distances(plan, args.data, raw, state)})


def transfer(args):
    import numpy as np

    from app.detectors.audio.upgraded import SpeechActivity

    torch = runtime(3)
    rows, excluded, digest = layers.transfer_inputs(args.data, args.roster)
    args.output.mkdir(mode=0o700)
    candidate = SOURCE.read(args.fit / "candidate.json")
    fit, projection = candidate["freeze"], args.fit / "projection.npy"
    SOURCE.check(fit["schema"] == SCHEMA and fit["method"] == METHOD and fit["source_sha256"] == digest
                 and SOURCE.path_sha(projection) == fit["projection_sha256"]
                 and SOURCE.path_sha(args.speech) == layers.SPEECH, "Transfer fit or speech commitments differ")
    state = np.load(projection, allow_pickle=False)
    baseline = SOURCE.read(args.previous)
    SOURCE.check([row["pcm_sha256"] for row in rows] == baseline["freeze"]["planned_pcm_sha256"],
                 "Previous comparison used different transfer PCM")
    layers.transfer_counts(baseline)
    heads = {case["trained_family"]: layers.load_head(args.fit, case) for case in candidate["cases"]}
    head_specs = {f"projection/{case['trained_family']}": {key: case[key] for key in ("weights_sha256", "points")}
                  for case in candidate["cases"]}
    freeze = baseline["freeze"] | {"head_specs": baseline["freeze"]["head_specs"] | head_specs,
                                  "projection_sha256": fit["projection_sha256"], "excluded_groups": len(excluded)}
    SOURCE.write(args.output / "freeze.json", freeze)
    activity, encode = SpeechActivity(args.speech), layers.frozen_encoder(args.encoder)
    output = []
    with torch.inference_mode():
        for row, previous in zip(rows, baseline["rows"], strict=True):
            raw = (args.roster / "pcm" / (row["id"] + ".pcm")).read_bytes()
            SOURCE.check(len(raw) == 2 * detector.SAMPLES and SOURCE.digest(raw) == row["pcm_sha256"]
                         and all(row[key] == previous[key] for key in ("group", "arm", "family", "truth", "pcm_sha256")),
                         "Transfer source binding differs")
            audio = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.
            features = project(encode(audio), state)
            spans = activity.spans(audio)
            scores = {}
            for name, (classifier, (mean, scale)) in heads.items():
                values = (features - mean[:, None]) / scale[:, None]
                local, edge = classifier(torch.from_numpy(values[None]))[0].numpy()
                scores[f"projection/{name}"] = detector.best_interval(local, edge, spans)
            output.append(previous | {"heads": previous["heads"] | scores})
        record = {"freeze": freeze, "rows": output}
        record["counts"] = layers.transfer_counts(record)
        SOURCE.write(args.output / "receipt.json", record)
        print(f"Measured {len(output)} committed transfer recordings.", flush=True)


def matched_location_collisions(rows, calls, condition_key, control_key):
    """Count an ordinary match at the true edit clock even if the proposed boxes differ."""
    from collections import Counter, defaultdict

    from scripts.review_audio_stock_voice_transfer import useful

    grouped = defaultdict(list)
    for row, call in zip(rows, calls, strict=True):
        if row["truth"] is None and call is not None:
            grouped[(row["group"], condition_key(row))].append((control_key(row), call))
    counts = defaultdict(Counter)
    for row, call in zip(rows, calls, strict=True):
        truth = row["truth"]
        if truth is None or call is None or not useful(call, truth):
            continue
        condition = condition_key(row)
        report = counts[condition]
        report["useful"] += 1
        controls = grouped[row["group"], condition]
        report["exact_control_collision"] += any(call == control for _, control in controls)
        collided = {kind for kind, control in controls if useful(control, truth)}
        report["edit_clock_control_collision"] += bool(collided)
        report["edit_clock_background_collision"] += bool(collided & {"background", "background_only"})
        report["useful_without_edit_clock_collision"] += not collided
    return dict(counts)


def fixed_false_group_comparison(rows, scores, control_key):
    """Compare retrospective ranking at equal ordinary-group and isolated-background budgets."""
    ordinary, background = {}, []
    for row, measured in zip(rows, scores, strict=True):
        item = measured["best"]
        if item is None or row["truth"] is not None:
            continue
        score = item["score"]
        if control_key(row) in {"background", "background_only"}:
            background.append(score)
        else:
            group = row["group"]
            ordinary[group] = max(ordinary.get(group, score), score)
    ranked = sorted(ordinary.values(), reverse=True)
    SOURCE.check(len(ranked) > max(FALSE_GROUP_BUDGETS) and bool(background), "Too few assessed ordinary or background groups")
    points = {}
    for budget in FALSE_GROUP_BUDGETS:
        # Strictly exceed the next ordinary group and any isolated background.
        # Ties can leave the actual ordinary-group count below the budget.
        point = max(ranked[budget], max(background))
        calls = detector.verdicts(rows, scores, point)
        false_groups = {row["group"] for row, call in zip(rows, calls, strict=True)
                        if call is not None and row["truth"] is None and control_key(row) not in {"background", "background_only"}}
        clocks = matched_location_collisions(rows, calls, lambda row: "competing_speech", control_key)
        points[budget] = {"ordinary_flagged_groups": len(false_groups), **clocks.get("competing_speech", {
            "useful": 0, "exact_control_collision": 0, "edit_clock_control_collision": 0,
            "edit_clock_background_collision": 0, "useful_without_edit_clock_collision": 0,
        })}
    return points


def candidate_collisions(plan, candidates):
    report, equal_groups = {}, {}
    for name, candidate in candidates.items():
        for case in candidate["cases"]:
            family = case["held_family"]
            held = detector.relevant(plan, "held", family)
            for policy, point in case["points"].items():
                calls = detector.verdicts(held, case["held_scores"], point)
                report[f"{name}/{family}/{policy}"] = matched_location_collisions(
                    held, calls, lambda row: row["condition"], lambda row: row["kind"])
            wet = [(row, score) for row, score in zip(held, case["held_scores"], strict=True)
                   if row["condition"] == "competing_speech"]
            rows, scores = zip(*wet, strict=True)
            equal_groups[f"{name}/{family}"] = fixed_false_group_comparison(rows, scores, lambda row: row["kind"])
    return report, equal_groups


def review_transfer(record):
    totals = layers.transfer_counts(record)
    rows = record["rows"]
    reporting = {
        "incumbent": matched_location_collisions(
            rows, [row["incumbent"]["range"] for row in rows], lambda row: "competing_speech", lambda row: row["arm"]),
    }
    equal_groups = {}
    for key, spec in record["freeze"]["head_specs"].items():
        scores = [{"pcm_sha256": row["pcm_sha256"], "best": row["heads"][key]} for row in rows]
        equal_groups[key] = fixed_false_group_comparison(rows, scores, lambda row: row["arm"])
        for name, point in spec["points"].items():
            calls = detector.verdicts(rows, scores, point)
            reporting[f"{key}/{name}"] = matched_location_collisions(
                rows, calls, lambda row: "competing_speech", lambda row: row["arm"])
    return {"totals": totals, "matched_edit_clocks": reporting, "retrospective_equal_false_groups": equal_groups}


def review(baseline, before, after):
    import json
    import math
    from statistics import median

    prior = detector.result(baseline | {"candidate": before}, schema=layers.SCHEMA, method=layers.EXPERIMENT)
    revised = detector.result(baseline | {"candidate": after}, schema=SCHEMA, method=METHOD)
    SOURCE.check(before["feature_ledger_sha256"] == after["freeze"]["feature_ledger_sha256"],
                 "The baseline and candidate read different input features")
    entries = after["aligned_distances"]
    expected = {(row["group"], row["kind"]) for row in baseline["sources"]["rows"]
                if row["condition"] == "competing_speech" and row["role"] == "held" and row["kind"] != "background"}
    SOURCE.check({(row["group"], row["kind"]) for row in entries} == expected and len(entries) == len(expected),
                 "Paired reconstruction rows differ from the held roster")
    measurements = {}
    for prefix in ("untouched", "genuine_splice", "resampled", "edited"):
        cases = [row for row in entries if row["kind"].startswith(prefix)]
        SOURCE.check(all(math.isfinite(row[key]) and row[key] > 0 for row in cases for key in ("before_mse", "after_mse")),
                     "Non-finite or invalid pair distance")
        measurements[prefix] = {
            "pairs": len(cases),
            "closer_to_clean": sum(row["after_mse"] < row["before_mse"] for row in cases),
            "median_relative_mse_reduction": round(median(1. - row["after_mse"] / row["before_mse"] for row in cases), 8),
        }
    collisions, equal_groups = candidate_collisions(baseline["sources"], {"original_layers": before, "projection": after})
    incumbent = {row["pcm_sha256"]: row["selected"] for row in baseline["incumbent"]["results"]}
    for family in detector.sources.FAMILIES:
        wet = [row for row in detector.relevant(baseline["sources"], "held", family) if row["condition"] == "competing_speech"]
        collisions[f"incumbent/{family}"] = matched_location_collisions(
            wet, [incumbent[row["pcm_sha256"]] for row in wet], lambda row: row["condition"], lambda row: row["kind"])
    print(json.dumps({"original_layers": prior, "projection": revised, "paired_distances": measurements,
                      "matched_edit_clocks": collisions, "retrospective_equal_false_groups": equal_groups}, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase, arguments in (("train", ("data", "cache", "speech", "output")),
                             ("transfer", ("data", "roster", "encoder", "speech", "fit", "previous", "output")),
                             ("review", ("baseline", "before", "after")), ("review-transfer", ("receipt",))):
        command = phases.add_parser(phase)
        for argument in arguments:
            command.add_argument(argument, type=Path)
    args = parser.parse_args()
    if args.phase == "train":
        train(args)
    elif args.phase == "transfer":
        transfer(args)
    elif args.phase == "review":
        review(*(SOURCE.read(path) for path in (args.baseline, args.before, args.after)))
    else:
        import json

        print(json.dumps(review_transfer(SOURCE.read(args.receipt)), indent=2, sort_keys=True))
