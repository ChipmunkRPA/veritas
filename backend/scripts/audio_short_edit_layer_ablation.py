# SPDX-License-Identifier: MIT
"""Test earlier speech-encoder features for locating edits under louder competing speech."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts import audio_competing_speech_ranking as transfer
from scripts import audio_short_edit_detector_training as detector
from scripts import audio_short_edit_noise_projection as projection
from scripts import audio_short_edit_ssl_transfer as frozen
from scripts import review_audio_random_clock_stock as random_stock

SOURCE = detector.SOURCE
SCHEMA = "chipmunk.audio-short-edit-layer-ablation/1"
TRANSFER_SCHEMA = "chipmunk.audio-short-edit-layer-ablation-transfer/1"
RANDOM_SCHEMA = "chipmunk.audio-short-edit-layer-ablation-random-clock/1"
RANDOM_ROSTER = transfer.ROOM_BASELINE.with_name("audio-random-clock-stock-2026-11-13.json")
ORIGINAL_FIT = transfer.ROOM_BASELINE.with_name("audio-short-edit-speech-layers-2026-11-14.json")
ARMS = {"layer4": frozen.WIDTHS[0], "layers4_and8": sum(frozen.WIDTHS[:2])}
METHOD = frozen.EXPERIMENT | {
    "ablation": "Keep layer 4 alone or keep layers 4 and 8. Zero the omitted projection columns before fitting "
                "and inference. Keep projection columns, head shape, initialization, optimizer, data and native "
                "time grid fixed. Read a single original waveform. Both arms get their own genuine-only points.",
    "transfer": "Use the previously committed independent AMI and stock banks. Keep dry, 0 dB and -5 dB separate. "
                "Compare useful ranges, unassessed files and edit-clock collisions with the current short review.",
}


def mask(values, width):
    selected = values.copy()
    selected[width:] = 0
    return selected


def feature_view(read, width):
    def view(data, row):
        return mask(read(data, row), width)

    return view


def fit(args):
    from safetensors.torch import save_file

    from app.detectors.audio.upgraded import SpeechActivity

    torch = projection.runtime(2)
    plan, digest = detector.load_plan(args.data)
    origin, feature_of = frozen.feature_reader(args.cache, plan, digest)
    SOURCE.check(SOURCE.path_sha(args.speech) == frozen.SPEECH, "Speech activity asset differs")
    activity = SpeechActivity(args.speech)
    args.output.mkdir(mode=0o700)
    weights = {}
    for arm, width in ARMS.items():
        directory = args.output / arm
        directory.mkdir(mode=0o700)
        selected = feature_view(feature_of, width)
        cases = []
        for known, unknown in (detector.sources.FAMILIES, detector.sources.FAMILIES[::-1]):
            model, mean, scale, trace = detector.train(args.data, plan, known, None, feature_of=selected)
            target = directory / (known + ".safetensors")
            save_file({**model.state_dict(), "feature_mean": torch.from_numpy(mean),
                       "feature_scale": torch.from_numpy(scale)}, str(target))
            calibration = detector.relevant(plan, "calibration", known, calibration=True)
            held = detector.relevant(plan, "held", unknown)
            scored = detector.score_rows(model, (mean, scale), args.data, calibration, None, activity, feature_of=selected)
            points = detector.scores_at_points(calibration, scored)
            cases.append({"trained_family": known, "held_family": unknown, "weights_sha256": SOURCE.path_sha(target),
                          "loss_trace": trace, "calibration_scores": scored, "points": points,
                          "held_scores": detector.score_rows(model, (mean, scale), args.data, held, None, activity,
                                                             feature_of=selected)})
            print(f"{arm} trained on {known}. Scored {len(held)} held recordings.", flush=True)
        weights[arm] = {"freeze": origin | {"schema": SCHEMA, "method": METHOD | {"arm": arm}}, "cases": cases,
                        "feature_ledger_sha256": SOURCE.path_sha(args.cache / "features.json")}
    SOURCE.write(args.output / "candidate.json", weights)


def review(baseline, original, variants):
    SOURCE.check(set(variants) == set(ARMS), "Missing layer arm")
    counts = {"original": detector.result(baseline | {"candidate": original}, schema=frozen.SCHEMA, method=frozen.EXPERIMENT)}
    for arm in ARMS:
        candidate = variants[arm]
        SOURCE.check(candidate["feature_ledger_sha256"] == original["feature_ledger_sha256"]
                     and all(candidate["freeze"][key] == original["freeze"][key]
                             for key in ("source_sha256", "speech_sha256", "seed", "epochs"))
                     and all(candidate["freeze"]["encoder"].get(key) == value
                             for key, value in original["freeze"]["encoder"].items()),
                     "The fitted ablation used different model features or sources")
        counts[arm] = detector.result(baseline | {"candidate": candidate}, schema=SCHEMA, method=METHOD | {"arm": arm})
    clocks, budgets = projection.candidate_collisions(baseline["sources"], {"original": original, **variants})
    return {"counts": counts, "matched_edit_clocks": clocks, "equal_alarm_budgets": budgets}


def points(fits, plan):
    result = {}
    for arm, fitted in fits.items():
        for case in fitted["cases"]:
            family = case["trained_family"]
            rows = detector.relevant(plan, "calibration", family, calibration=True)
            readout = detector.scores_at_points(rows, case["calibration_scores"])
            SOURCE.check(readout == case["points"], "The sealed point differs")
            result[f"{arm}/{family}"] = readout
    return result


def score_transfer(args):
    import numpy as np

    from app.detectors.audio.upgraded import SpeechActivity

    torch = projection.runtime(3)
    artifact = args.fit / "candidate.json"
    fitted = {"original": SOURCE.read(ORIGINAL_FIT), **SOURCE.read(artifact)}
    plan, digest = detector.load_plan(args.data)
    SOURCE.check(all(fit["freeze"]["source_sha256"] == digest for fit in fitted.values())
                 and SOURCE.path_sha(args.speech) == frozen.SPEECH, "Transfer source or speech asset differs")
    specs = points(fitted, plan)
    paths = {"original": args.original_fit, **{arm: args.fit / arm for arm in ARMS}}
    heads = {f"{arm}/{case['trained_family']}": (ARMS.get(arm, detector.FREQUENCIES), frozen.load_head(paths[arm], case))
             for arm, reading in fitted.items() for case in reading["cases"]}
    encoder, activity = frozen.frozen_encoder(args.encoder), SpeechActivity(args.speech)
    random = args.random_stock is not None
    inputs = random_cases(args) if random else transfer.transfer_cases(args.data, args.rooms, args.stock)
    rows = []
    with torch.inference_mode():
        for row, signal in inputs:
            pcm = signal.astype(np.float32) / 32768.
            features, speech = encoder(pcm), activity.spans(pcm)
            readings = {}
            for name, (width, (head, (mean, scale))) in heads.items():
                values = (mask(features, width) - mean[:, None]) / scale[:, None]
                local, edge = head(torch.from_numpy(values[None]))[0].numpy()
                readings[name] = detector.best_interval(local, edge, speech)
            rows.append(row | {"heads": readings})
            if len(rows) % 24 == 0:
                print(f"Verified and scored {len(rows)} transfer recordings.", flush=True)
    results = random_results(rows, specs) if random else transfer.transfer_scores(rows, specs)
    SOURCE.write(args.output, {"schema": RANDOM_SCHEMA if random else TRANSFER_SCHEMA,
                               "fit_sha256": SOURCE.path_sha(artifact), "original_sha256": SOURCE.path_sha(ORIGINAL_FIT),
                               "points": specs, "rows": rows, "results": results})


def random_cases(args):
    import numpy as np

    root = args.random_stock
    reference = SOURCE.read(RANDOM_ROSTER)
    freeze = root / "placements-freeze.json"
    SOURCE.check(SOURCE.path_sha(freeze) == reference["native_freeze_sha256"], "Random-clock source manifest differs")
    native = SOURCE.read(freeze)
    plan, _ = detector.load_plan(args.data)
    room, _, _ = frozen.transfer_inputs(args.data, args.rooms)
    stock = SOURCE.read(args.stock / "freeze.json")
    SOURCE.check(SOURCE.path_sha(args.stock / "freeze.json") == SOURCE.read(transfer.STOCK_BASELINE)["native_freeze_sha256"],
                 "Previous stock roster differs")
    used_rooms = {item["background"]["room"] for item in plan["sources"]} | {row["room"] for row in room}
    used_rooms.update(item["meeting"] for item in stock["capture_metadata"])
    reader_pins = {item["reader_sha256"] for item in plan["sources"]}
    readers = {(row.get(side) or {}).get("canonical_speaker", "").split(":")[-1] for row in room for side in ("host", "donor")}
    readers.update(str(row["sources"][side]) for row in stock["cases"] for side in ("host_reader", "donor_reader") if side in row["sources"])
    members = {str(row[side]) for row in native["source_commitments"] for side in ("reader", "donor_reader")}
    SOURCE.check(not used_rooms.intersection(row["meeting"] for row in native["capture_commitments"])
                 and not readers & members and all(detector.sources.order(voice) not in reader_pins for voice in members),
                 "Random-clock speakers or recordings overlap a previous role")
    used_pcm = {row["pcm_sha256"] for row in plan["rows"] + room + stock["cases"]}
    with np.load(root / "placements-native.npz", allow_pickle=False) as bank:
        signal = bank["native"]
    SOURCE.check(signal.shape == (83, detector.SAMPLES) and signal.dtype == np.dtype("<i2"), "Random-clock PCM geometry differs")
    for data, bound, row in zip(signal, native["native_manifest_rows"], reference["rows"], strict=True):
        SOURCE.check(all(row[key] == value for key, value in bound.items()) and row["pcm_sha256"] not in used_pcm
                     and SOURCE.digest(data.tobytes()) == row["pcm_sha256"], "Random-clock PCM or role differs")
        yield random_identity(row), data


def random_identity(row):
    return {"pcm_sha256": row["pcm_sha256"], "group": row["group"], "arm": row["kind"], "truth": row["edit_clock"],
            "condition": f"{row['snr']}dB" if row["snr"] is not None else "isolated"}


def random_results(rows, specs):
    reference = SOURCE.read(RANDOM_ROSTER)
    SOURCE.check(random_stock.verify(reference) == reference["summary"], "Published random-clock comparison differs")
    SOURCE.check([{key: value for key, value in row.items() if key != "heads"} for row in rows]
                 == [random_identity(row) for row in reference["rows"]], "Random-clock identities differ")
    SOURCE.check(all(set(row["heads"]) == set(specs) for row in rows), "Random-clock fitted heads differ")
    incumbent = [random_stock.selected(row, "existing_short") for row in reference["rows"]]
    calls = [list(window) if flagged else None for window, _, flagged in incumbent]
    policies = {"incumbent": (calls, [window is not None for window, _, _ in incumbent])}
    for name, cutoffs in specs.items():
        measured = [{"pcm_sha256": row["pcm_sha256"], "best": row["heads"][name]} for row in rows]
        for level, point in cutoffs.items():
            decisions = detector.verdicts(rows, measured, point)
            policies[f"{name}/{level}"] = decisions, [row["best"] is not None for row in measured]
    results = {}
    for name, (decisions, coverage) in policies.items():
        counts, paired = {}, {}
        for row, interval, assessed, current in zip(rows, decisions, coverage, calls, strict=True):
            category = row["arm"] if row["truth"] is not None else "background" if row["arm"] == "isolated" else "ordinary"
            cell = counts.setdefault(f"{row['condition']}/{category}", Counter())
            cell.update(files=1, flagged=interval is not None, unassessed=not assessed)
            if row["truth"] is not None:
                cell["useful"] += interval is not None and random_stock.prior.useful(interval, row["truth"])
            by_condition = paired.setdefault(row["condition"], ([], [], []))
            by_condition[0].append(row | {"kind": "background" if category == "background" else row["arm"]})
            by_condition[1].append(current)
            by_condition[2].append(interval)
        clocks = projection.matched_location_collisions(rows, decisions, lambda row: row["condition"], lambda row: row["arm"])
        results[name] = {"counts": {key: dict(tally) for key, tally in counts.items()}, "matched_edit_clocks": clocks,
                         "additions_if_incumbent_is_retained": {condition: detector.paired_gains(*paired[condition]) for condition in paired}}
    return results


def review_transfer(record, artifact, baseline):
    SOURCE.check(record["schema"] in {TRANSFER_SCHEMA, RANDOM_SCHEMA} and record["fit_sha256"] == SOURCE.path_sha(artifact)
                 and record["original_sha256"] == SOURCE.path_sha(ORIGINAL_FIT),
                 "Transfer refers to a different fit")
    fitted = {"original": SOURCE.read(ORIGINAL_FIT), **SOURCE.read(artifact)}
    specs = points(fitted, baseline["sources"])
    SOURCE.check(specs == record["points"], "Transfer points differ from the fit")
    recount = random_results if record["schema"] == RANDOM_SCHEMA else transfer.transfer_scores
    results = recount(record["rows"], specs)
    SOURCE.check(results == record["results"], "Transfer counts differ")
    return results


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="phase", required=True)
    for phase, fields in (("fit", ("data", "cache", "speech", "output")),
                          ("transfer", ("data", "rooms", "stock", "encoder", "speech", "fit", "output")),
                          ("review", ("baseline", "original", "candidate")),
                          ("review-transfer", ("record", "candidate", "baseline"))):
        command = commands.add_parser(phase)
        for field in fields:
            command.add_argument(field, type=Path)
        if phase == "transfer":
            command.add_argument("--original-fit", type=Path, required=True)
            command.add_argument("--random-stock", type=Path)
    args = cli.parse_args()
    if args.phase == "fit":
        fit(args)
    elif args.phase == "transfer":
        score_transfer(args)
    else:
        output = (review(SOURCE.read(args.baseline), SOURCE.read(args.original), SOURCE.read(args.candidate))
                  if args.phase == "review" else review_transfer(SOURCE.read(args.record), args.candidate, SOURCE.read(args.baseline)))
        print(json.dumps(output, indent=2, sort_keys=True))
