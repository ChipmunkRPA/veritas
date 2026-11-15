# SPDX-License-Identifier: MIT
"""Fit short voice-edit proposals against matched ordinary speech and isolated backgrounds."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts import audio_short_edit_detector_training as detector
from scripts import audio_short_edit_noise_projection as projection
from scripts import audio_short_edit_ssl_transfer as layers
from scripts import review_audio_stock_voice_transfer as stock
from scripts.review_audio_stock_voice_transfer import useful

SOURCE = detector.SOURCE
SCHEMA = "chipmunk.audio-competing-speech-ranking/1"
TRANSFER_SCHEMA = "chipmunk.audio-competing-speech-ranking-transfer/1"
PANELS = Path(__file__).resolve().parents[2] / "docs" / "evaluation-panels"
ROOM_BASELINE = PANELS / "audio-short-edit-speech-layer-transfer-2026-11-14.json"
STOCK_BASELINE = PANELS / "audio-stock-voice-transfer-2026-11-13.json"
METHOD = {
    "inputs": "Same frozen encoder, small two-output network, 16 epochs, and independent family and speaker roles as the prior study.",
    "target": "Rank a speech-admitted, truth-overlapping half-to-one-second window from the synthetic clip above the strongest window in its three matched genuine clips and isolated background.",
    "clock": "Limit control windows to those usefully overlapping the same labeled edit clock. Use controls only during fitting and evaluation.",
    "whole_file": "Consider every eligible proposal in each of the four control files. Add the same rank loss with the same margin and weight.",
    "loss": "Add 0.45 times softplus(0.5 + highest control proposal - best useful synthetic proposal) on admitted matched sets. Exclude sets lacking either a supported positive or an eligible control from this loss.",
    "assessment": "Keep prior speech admission, interval readout and genuine-only calibration. Run each trained model on one submitted clip at a time.",
}
ARMS = ("clock", "whole_file")
PROPOSALS = [(start, start + width) for width in detector.LENGTHS
             for start in range(0, detector.FRAMES - width + 1, detector.STRIDE)]


def ranking_objectives(data, plan, family, activity, feature_of):
    import numpy as np
    import torch
    from torch.nn import functional as F

    from app.detectors.audio.content import window_in_speech

    groups = [row for row in plan["sources"] if row["role"] == "fit"]
    lookup = {(row["group"], row["kind"]): row for row in plan["rows"]
              if row["role"] == "fit" and row["condition"] == "competing_speech"}
    kinds = ("untouched", "genuine_splice", f"resampled/{family}", f"edited/{family}", "background")
    intervals = [(a * detector.HOP, b * detector.HOP) for a, b in PROPOSALS]
    background, coverage, matching = [], [], []
    for group in groups:
        members = [lookup[group["group"], kind] for kind in kinds]
        background.append(feature_of(data, members[-1]))
        matching.append([useful(span, group["truth"]) for span in intervals])
        gates = []
        for row in members:
            speech = activity.spans(detector.pcm(data, row))
            gates.append([window_in_speech(speech, a / detector.RATE, b / detector.RATE, fraction=.5) for a, b in intervals])
        coverage.append(gates)
    backgrounds = torch.from_numpy(np.stack(background))
    support, correct = torch.tensor(coverage), torch.tensor(matching)
    target = support[:, 3] & correct
    ordinary = support[:, (0, 1, 2, 4), :]
    starts, stops = torch.tensor(PROPOSALS).unbind(dim=1)
    width = stops - starts
    scopes = {"clock": ordinary & correct[:, None, :], "whole_file": ordinary}

    def objective(mask):
        qualified = target.any(1) & mask.any(dim=(1, 2))

        def loss(classifier, mean, scale, indices, local, boundary):
            valid = qualified[indices]
            if not valid.any():
                return local.new_zeros(())
            accepted = indices[valid]
            waveform = (backgrounds[accepted] - torch.from_numpy(mean)[None, :, None]) / torch.from_numpy(scale)[None, :, None]
            isolated, isolated_edge = classifier(waveform).unbind(dim=1)
            local = torch.cat((local[valid], isolated[:, None, :]), dim=1)
            boundary = torch.cat((boundary[valid], isolated_edge[:, None, :]), dim=1)
            prefix = F.pad(local.cumsum(-1), (1, 0))
            mean_response = (prefix[..., stops] - prefix[..., starts]) / width
            pooled = F.max_pool1d(F.pad(boundary, (0, 1), value=-torch.inf), kernel_size=9, stride=1, padding=4)
            proposals = mean_response + .075 * (pooled[..., starts] + pooled[..., stops])
            positives = proposals[:, 3].masked_fill(~target[accepted], -torch.inf).amax(-1)
            controls = proposals[:, (0, 1, 2, 4)].masked_fill(~mask[accepted], -torch.inf).amax(dim=(1, 2))
            return .45 * F.softplus(.5 + controls - positives).mean()

        return loss, int(qualified.sum())

    return {arm: objective(mask) for arm, mask in scopes.items()}


def fit(args):
    from safetensors.torch import save_file

    from app.detectors.audio.upgraded import SpeechActivity

    torch = projection.runtime(2)
    plan, digest = detector.load_plan(args.data)
    origin, feature_of = layers.feature_reader(args.cache, plan, digest)
    SOURCE.check(SOURCE.path_sha(args.speech) == layers.SPEECH, "Speech activity asset differs")
    args.output.mkdir(mode=0o700)
    lock = origin | {"schema": SCHEMA, "method": METHOD,
                     "feature_ledger_sha256": SOURCE.path_sha(args.cache / "features.json")}
    SOURCE.write(args.output / "freeze.json", lock)
    activity = SpeechActivity(args.speech)
    fitted = []
    for known, unknown in (detector.sources.FAMILIES, detector.sources.FAMILIES[::-1]):
        objectives = ranking_objectives(args.data, plan, known, activity, feature_of)
        for arm, (objective, qualified) in objectives.items():
            name = f"{arm}-{known}"
            classifier, mean, scale, trace = detector.train(args.data, plan, known, None, feature_of=feature_of,
                                                           extra_objective=objective)
            weights = args.output / (name + ".safetensors")
            save_file(classifier.state_dict() | {"feature_mean": torch.from_numpy(mean), "feature_scale": torch.from_numpy(scale)},
                      str(weights))
            calibration = detector.relevant(plan, "calibration", known, calibration=True)
            scores = detector.score_rows(classifier, (mean, scale), args.data, calibration, None, activity, feature_of=feature_of)
            seal = {"trained_family": known, "held_family": unknown, "arm": arm, "qualified_fit_groups": qualified,
                    "calibration_scores": scores, "losses": trace, "points": detector.scores_at_points(calibration, scores),
                    "weights_sha256": SOURCE.path_sha(weights)}
            SOURCE.write(args.output / (name + "-points.json"), seal)
            fitted.append((classifier, (mean, scale), seal))
            print(f"Sealed {arm}, {known}, with {qualified} matched fitting groups.", flush=True)
    cases = []
    for classifier, norm, seal in fitted:
        held = detector.relevant(plan, "held", seal["held_family"])
        scores = detector.score_rows(classifier, norm, args.data, held, None, activity, feature_of=feature_of)
        cases.append(seal | {"held_scores": scores})
    SOURCE.write(args.output / "candidate.json", {"freeze": lock, "cases": cases})


def review(baseline, before, candidate):
    SOURCE.check(candidate["freeze"]["schema"] == SCHEMA and candidate["freeze"]["method"] == METHOD, "Experiment differs")
    plan = baseline["sources"]
    original = detector.result(baseline | {"candidate": before}, schema=layers.SCHEMA, method=layers.EXPERIMENT)
    results = {"prior": original}
    candidates = {"prior": before}
    for arm in ARMS:
        subset = candidate | {"cases": [case for case in candidate["cases"] if case["arm"] == arm]}
        SOURCE.check(len(subset["cases"]) == len(detector.sources.FAMILIES), "Missing family arm")
        results[arm] = detector.result(baseline | {"candidate": subset}, schema=SCHEMA, method=METHOD)
        candidates[arm] = subset
    clocks, paired = projection.candidate_collisions(plan, candidates)
    print(json.dumps({"counts": results, "matched_edit_clocks": clocks, "equal_alarm_budgets": paired}, indent=2, sort_keys=True))


def transfer_cases(data, rooms, stock_bank):
    import numpy as np

    room_rows, _, digest = layers.transfer_inputs(data, rooms)
    baseline = SOURCE.read(ROOM_BASELINE)
    layers.transfer_counts(baseline)
    for row, prior in zip(room_rows, baseline["rows"], strict=True):
        raw = (rooms / "pcm" / (row["id"] + ".pcm")).read_bytes()
        keys = ("group", "arm", "family", "truth", "pcm_sha256")
        SOURCE.check(len(raw) == detector.SAMPLES * 2 and SOURCE.digest(raw) == row["pcm_sha256"]
                     and all(row[key] == prior[key] for key in keys), "Separate room PCM differs")
        projected = {key: row[key] for key in keys}
        yield projected | {"bank": "room_transfer", "condition": "-5dB"}, np.frombuffer(raw, "<i2")
    stock_doc = SOURCE.read(STOCK_BASELINE)
    stock.check(stock_doc)
    freeze = stock_bank / "freeze.json"
    SOURCE.check(SOURCE.path_sha(freeze) == stock_doc["native_freeze_sha256"], "Stock voice source manifest differs")
    plan, current = detector.load_plan(data)
    SOURCE.check(current == digest, "Fit source manifest differs between transfers")
    native = SOURCE.read(freeze)
    capture_rooms = {item["meeting"] for item in native["capture_metadata"]}
    consumed_rooms = {item["background"]["room"] for item in plan["sources"]} | {row["room"] for row in room_rows}
    audio = np.load(stock_bank / "signals.npy", mmap_mode="r", allow_pickle=False)
    SOURCE.check(not capture_rooms & consumed_rooms and audio.shape == (len(stock_doc["rows"]), detector.SAMPLES)
                 and audio.dtype == np.dtype("<i2"), "Stock voice room or PCM geometry differs")
    previous_pcm = {row["pcm_sha256"] for row in plan["rows"] + room_rows}
    readers = {item["reader_sha256"] for item in plan["sources"]}
    for row, prior, signal in zip(native["cases"], stock_doc["rows"], audio, strict=True):
        keys = ("group", "kind", "condition", "truth", "pcm_sha256")
        source = row["sources"]
        source_readers = [source[key] for key in ("host_reader", "donor_reader") if key in source]
        SOURCE.check(all(row[key] == prior[key] for key in keys) and row["pcm_sha256"] not in previous_pcm
                     and all(detector.sources.order(str(voice)) not in readers for voice in source_readers)
                     and SOURCE.digest(signal.tobytes()) == row["pcm_sha256"], "Stock voice source or matched PCM differs")
        yield {"bank": "stock_transfer", "condition": row["condition"], "group": row["group"],
               "arm": row["kind"], "family": row["kind"] if row["truth"] else None, "truth": row["truth"],
               "pcm_sha256": row["pcm_sha256"]}, signal


def transfer_scores(rows, specs):
    room, stock_rows = SOURCE.read(ROOM_BASELINE), SOURCE.read(STOCK_BASELINE)
    known = {row["pcm_sha256"]: row for row in room["rows"] + stock_rows["rows"]}
    cohorts = {}
    for row in rows:
        base = known.pop(row["pcm_sha256"])
        expected = base.get("arm", base.get("kind"))
        bank = "room_transfer" if "arm" in base else "stock_transfer"
        family = base.get("family") if bank == "room_transfer" else (expected if base["truth"] is not None else None)
        SOURCE.check(row["bank"] == bank and row["arm"] == expected and row["group"] == base["group"]
                     and row["truth"] == base["truth"] and row["condition"] == base.get("condition", "-5dB")
                     and row["family"] == family and set(row["heads"]) == set(specs),
                     "Transfer result changed its frozen source labels")
        if bank == "room_transfer":
            incumbent = base["incumbent"]["range"]
            assessed = base["incumbent"]["assessed"]
        else:
            selected = stock.strongest(base, range(len(stock_rows["windows"])))
            incumbent = (stock_rows["windows"][-selected[1]] if selected and selected[0] > stock_rows["points"]["existing_short"] else None)
            assessed = selected is not None
        cohorts.setdefault(bank, []).append((row, incumbent, assessed))
    SOURCE.check(not known, "Missing transfer cases")
    summary = {}
    for bank, values in cohorts.items():
        cases, incumbent, incumbent_coverage = zip(*values, strict=True)
        policies = {"incumbent": incumbent}
        for head, by_level in specs.items():
            scored = [{"pcm_sha256": row["pcm_sha256"], "best": row["heads"][head]} for row in cases]
            for level, cut in by_level.items():
                policies[f"{head}/{level}"] = detector.verdicts(cases, scored, cut)
        report = {}
        for policy, calls in policies.items():
            outcomes = {}
            for row, window, covered in zip(cases, calls, incumbent_coverage, strict=True):
                category = row["family"] if row["truth"] is not None else (
                    "background" if row["arm"] in {"background", "background_only"} else "ordinary")
                cell = outcomes.setdefault(f"{row['condition']}/{category}", Counter())
                head = policy.rsplit("/", 1)[0]
                assessed = covered if policy == "incumbent" else row["heads"][head] is not None
                cell.update(files=1, flagged=window is not None, unassessed=not assessed)
                if row["truth"] is not None:
                    cell["useful"] += window is not None and useful(window, row["truth"])
            clocks = projection.matched_location_collisions(cases, calls, lambda row: row["condition"], lambda row: row["arm"])
            report[policy] = {"counts": {k: dict(v) for k, v in outcomes.items()}, "matched_edit_clocks": clocks}
        summary[bank] = report
    return summary


def transfer(args):
    import numpy as np
    from app.detectors.audio.upgraded import SpeechActivity

    torch = projection.runtime(3)
    trained = args.fit / "candidate.json"
    document = SOURCE.read(trained)
    plan, digest = detector.load_plan(args.data)
    SOURCE.check(document["freeze"]["source_sha256"] == digest and document["freeze"]["method"] == METHOD
                 and SOURCE.path_sha(args.speech) == layers.SPEECH, "Transfer assets differ")
    heads, specs = {}, {}
    for case in document["cases"]:
        calibration = detector.relevant(plan, "calibration", case["trained_family"], calibration=True)
        SOURCE.check(case["points"] == detector.scores_at_points(calibration, case["calibration_scores"]), "Calibration differs")
        name = f"{case['arm']}-{case['trained_family']}"
        heads[name] = layers.load_head(args.fit, case | {"trained_family": name})
        specs[name] = case["points"]
    activity, encoder = SpeechActivity(args.speech), layers.frozen_encoder(args.encoder)
    records = []
    with torch.inference_mode():
        for row, signal in transfer_cases(args.data, args.rooms, args.stock):
            wave = signal.astype(np.float32) / 32768.
            features, speech = encoder(wave), activity.spans(wave)
            readings = {}
            for name, (head, (mean, scale)) in heads.items():
                local, edge = head(torch.from_numpy(((features - mean[:, None]) / scale[:, None])[None]))[0].numpy()
                readings[name] = detector.best_interval(local, edge, speech)
            records.append(row | {"heads": readings})
            if len(records) % 24 == 0:
                print(f"Verified and scored {len(records)} transfer recordings.", flush=True)
    result = {"schema": TRANSFER_SCHEMA, "fit_sha256": SOURCE.path_sha(trained), "points": specs, "rows": records,
              "results": transfer_scores(records, specs)}
    SOURCE.write(args.output, result)


def review_transfer(record, candidate):
    document = SOURCE.read(candidate)
    specs = {f"{case['arm']}-{case['trained_family']}": case["points"] for case in document["cases"]}
    SOURCE.check(record["schema"] == TRANSFER_SCHEMA and record["fit_sha256"] == SOURCE.path_sha(candidate)
                 and record["points"] == specs, "Transfer fit or points differ")
    checked = transfer_scores(record["rows"], specs)
    SOURCE.check(checked == record["results"], "Transfer counts differ")
    comparisons = paired_room_results(record["rows"], specs)
    print(json.dumps({"counts": checked, "paired_frozen_baseline": comparisons}, indent=2, sort_keys=True))


def paired_room_results(rows, points):
    baseline = SOURCE.read(ROOM_BASELINE)
    compared = [row for row in rows if row["bank"] == "room_transfer"]
    old_rows = {row["pcm_sha256"]: row for row in baseline["rows"]}
    comparison = {}
    for family in detector.sources.FAMILIES:
        old_head = f"frozen_speech_layers/{family}"
        old_scores = [{"pcm_sha256": row["pcm_sha256"], "best": old_rows[row["pcm_sha256"]]["heads"][old_head]} for row in compared]
        for arm in ARMS:
            head = f"{arm}-{family}"
            scores = [{"pcm_sha256": row["pcm_sha256"], "best": row["heads"][head]} for row in compared]
            for level in points[head]:
                old = detector.verdicts(compared, old_scores, baseline["freeze"]["head_specs"][old_head]["points"][level])
                new = detector.verdicts(compared, scores, points[head][level])
                changes = Counter()
                for row, before, after in zip(compared, old, new, strict=True):
                    if truth := row["truth"]:
                        previous = before is not None and useful(before, truth)
                        current = after is not None and useful(after, truth)
                        changes.update(new_localization=current and not previous, lost_localization=previous and not current)
                    elif row["arm"] != "background_only":
                        changes.update(new_ordinary_flag=after is not None and before is None,
                                       cleared_ordinary_flag=before is not None and after is None)
                comparison[f"{head}/{level}"] = dict(changes)
    return comparison


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    phases = cli.add_subparsers(dest="phase", required=True)
    train = phases.add_parser("fit")
    for field in ("data", "cache", "speech", "output"):
        train.add_argument(field, type=Path)
    audit = phases.add_parser("review")
    for field in ("baseline", "before", "candidate"):
        audit.add_argument(field, type=Path)
    testing = phases.add_parser("transfer")
    for field in ("data", "rooms", "stock", "encoder", "speech", "fit", "output"):
        testing.add_argument(field, type=Path)
    portable = phases.add_parser("review-transfer")
    for field in ("record", "candidate"):
        portable.add_argument(field, type=Path)
    args = cli.parse_args()
    if args.phase == "fit":
        fit(args)
    elif args.phase == "review":
        review(SOURCE.read(args.baseline), SOURCE.read(args.before), SOURCE.read(args.candidate))
    elif args.phase == "transfer":
        transfer(args)
    else:
        review_transfer(SOURCE.read(args.record), args.candidate)
