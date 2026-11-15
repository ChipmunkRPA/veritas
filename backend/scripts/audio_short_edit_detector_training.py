# SPDX-License-Identifier: MIT
"""Train and evaluate a small short-edit detector with matched controls."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_short_edit_training_sources as sources

SOURCE = sources.SOURCE
RATE, SAMPLES = sources.RATE, sources.SAMPLES
HOP = 160
FRAMES = SAMPLES // HOP
FREQUENCIES = 64
LENGTHS = (50, 75, 100)
STRIDE = 10
SEED = 341519
EPOCHS = 16
SCHEMA = "chipmunk.audio-short-edit-detector/1"
METHOD = {
    "data": "Train two independent directions. Each sees the synthetic substitutions and same-word resampling "
            "from just one generator plus shared genuine controls. Match dry and noisy training within each prompt. "
            "The operating-point readers and held readers are separate from fitting.",
    "inputs": "Use centered 25 ms frames every 10 ms and fixed 64 triangular Mel power bands. Standardize "
              "each band using only the respective fit pool. Teach exact per-frame content labels and splice "
              "edges within 30 ms. Optimize binary losses, paired synthetic-versus-genuine rankings, "
              "and consistency of the local margins between the dry and loud competing-speech views.",
    "readout": "Propose .50, .75 and 1.00 s intervals every 100 ms. Use mean content response plus 0.15 times "
               "the mean of the strongest learned edge responses within 40 ms of the two endpoints. "
               "Require at least half detected speech and return the strongest span, with earlier/shorter ties.",
    "points": "Use only genuine files from the calibration readers, excluding the other generator's resampling. "
              "Seal one strict cutoff at the maximum genuine score and one wider cutoff at the 95th percentile "
              "of genuine foreground scores, bounded below by the highest background-only score. Both "
              "use a strict comparison. Cutoffs are shared across dry and noisy views and are not probabilities.",
    "held": "Each direction scores the generator that contributed no training positives. Keep speakers and "
            "meeting captures outside fit/calibration. Report distinct locations, matched-control collisions, "
            "false alarms (including the isolated backgrounds), and misses. Compare original PCM to the "
            "current timed review, and earlier saved whole-model methods only on their own cohorts.",
}


def load_plan(root):
    path = root / "sources.json"
    plan = SOURCE.read(path)
    SOURCE.check(plan["schema"] == sources.SCHEMA and plan["method"] == sources.METHOD, "Source freeze differs")
    validate_plan(plan)
    return plan, SOURCE.path_sha(path)


def validate_plan(plan):
    groups = {x["group"]: x for x in plan["sources"]}
    sources_by_role = defaultdict(set)
    rooms = defaultdict(set)
    reader_lengths = Counter()
    expected_rows = defaultdict(set)
    for group, source in groups.items():
        sources_by_role[source["role"]].add(source["reader_sha256"])
        rooms[source["role"]].add(source["background"]["room"])
        left, right = source["truth"]
        SOURCE.check(0 < left < right < SAMPLES and 2 * source["background"]["support"] >= right-left,
                     "Unexpected foreground or background bounds")
        reader_lengths[source["reader_sha256"], right - left >= 9600] += 1
        for condition in sources.CONDITIONS:
            for kind in sources.KINDS:
                if kind != "background" or condition == "competing_speech":
                    expected_rows[group].add((condition, kind))
    SOURCE.check(len(groups) == len(plan["sources"]) == sources.PROMPTS * sum(sources.ROLES.values()),
                 "Unexpected source cohort")
    SOURCE.check({role: len(group) for role, group in sources_by_role.items()} == sources.ROLES
                 and sum(map(len, sources_by_role.values())) == len(set.union(*sources_by_role.values()))
                 and sum(map(len, rooms.values())) == len(set.union(*rooms.values())), "Speaker or recording role leakage")
    SOURCE.check({role: len(captures) for role, captures in rooms.items()} == {"fit": 1, "calibration": 1, "held": 3}
                 and all(reader_lengths[reader, longer] == sources.PROMPTS // 2
                         for reader in set.union(*sources_by_role.values()) for longer in (True, False)),
                 "Unexpected duration balance or recording roles")
    SOURCE.check(len({x["input_sha256"]["real"] for x in groups.values()}) == len(groups),
                 "Foreground recording reused")
    for row in plan["rows"]:
        group = groups[row["group"]]
        expected = group["truth"] if row["kind"].startswith("edited/") else None
        SOURCE.check(row["truth"] == expected and row["role"] == group["role"], "Role or truth differs")
        expected_rows[row["group"]].remove((row["condition"], row["kind"]))
    SOURCE.check(not any(expected_rows.values()), "Some variants are missing")


def mel_filterbank():
    import numpy as np

    frequencies = np.fft.rfftfreq(512, 1 / RATE)
    bands = 700 * (np.exp(np.linspace(np.log1p(50 / 700), np.log1p(7900 / 700), FREQUENCIES + 2)) - 1)
    a, b, c = bands[:-2, None], bands[1:-1, None], bands[2:, None]
    return np.maximum(0, np.minimum((frequencies - a) / (b - a), (c - frequencies) / (c - b))).astype(np.float32)


def features(signal, bank):
    import numpy as np

    padded = np.pad(signal, (200, 200), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, 400)[::HOP][:FRAMES]
    power = abs(np.fft.rfft(frames * np.hanning(400), n=512))**2
    value = np.log(np.maximum(1e-10, power @ bank.T))
    return value.T.astype(np.float32)


def pcm(root, row):
    import numpy as np

    key = row["pcm_sha256"]
    raw = (root / "pcm" / f"{key}.pcm").read_bytes()
    SOURCE.check(len(raw) == SAMPLES * 2 and SOURCE.digest(raw) == key, "PCM commitment differs")
    return np.frombuffer(raw, "<i2").astype(np.float32) / 32768.


def model():
    from torch import nn

    return nn.Sequential(
        nn.Conv1d(FREQUENCIES, 24, 1), nn.GELU(),
        nn.Conv1d(24, 24, 5, padding=2, groups=24), nn.Conv1d(24, 24, 1), nn.GELU(),
        nn.Conv1d(24, 24, 5, dilation=4, padding=8, groups=24), nn.Conv1d(24, 24, 1), nn.GELU(),
        nn.Conv1d(24, 24, 5, dilation=12, padding=24, groups=24), nn.Conv1d(24, 16, 1), nn.GELU(),
        nn.Conv1d(16, 2, 1),
    )


def fit_input(root, plan, generator, feature_of):
    import numpy as np

    wanted = ("untouched", "genuine_splice", f"resampled/{generator}", f"edited/{generator}")
    selected = {(row["group"], row["condition"], row["kind"]): row for row in plan["rows"] if row["role"] == "fit"}
    records, masks, edges = [], [], []
    clock = np.arange(FRAMES) * HOP
    for g in (source for source in plan["sources"] if source["role"] == "fit"):
        group, (left, right) = g["group"], g["truth"]
        records.append([[feature_of(root, selected[group, condition, kind]) for kind in wanted]
                        for condition in sources.CONDITIONS])
        masks.append((clock >= left) & (clock < right))
        edges.append((abs(clock-left) <= 480) | (abs(clock-right) <= 480))
    return np.array(records), np.array(masks, np.float32), np.array(edges, np.float32)


def train(root, plan, generator, bank, *, feature_of=None, extra_objective=None):
    import numpy as np
    import torch
    from torch.nn import functional as F

    if feature_of is None:
        def feature_of(data, row):
            return features(pcm(data, row), bank)
    x, region, edge = fit_input(root, plan, generator, feature_of)
    axes = (0, 1, 2, 4)
    mean = x.mean(axis=axes, dtype=np.float64).astype(np.float32)
    scale = np.maximum(.001, x.std(axis=axes, dtype=np.float64)).astype(np.float32)
    x = torch.from_numpy((x - mean[:, None]) / scale[:, None])
    region, edge = [torch.from_numpy(array) for array in (region, edge)]
    torch.manual_seed(SEED)
    classifier = model().train()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=.002, weight_decay=.001)
    trace = []
    for _ in range(EPOCHS):
        losses = []
        for indices in torch.randperm(len(x)).split(6):
            batch = x[indices]
            prediction = classifier(batch.flatten(0, 2)).reshape(*batch.shape[:3], 2, FRAMES)
            local, boundary = prediction.unbind(dim=3)
            interval = region[indices, None, None, :]
            marks = edge[indices, None, None, :]
            labels = torch.zeros_like(local)
            labels[:, :, -1:, :] = interval
            boundary_labels = torch.zeros_like(boundary)
            boundary_labels[:, :, -1:, :] = marks
            interior_loss = F.binary_cross_entropy_with_logits(local, labels, pos_weight=torch.tensor(8.))
            edge_loss = F.binary_cross_entropy_with_logits(boundary, boundary_labels, pos_weight=torch.tensor(20.))
            means = (local * interval).sum(dim=-1) / interval.sum(dim=-1)
            margins = means[:, :, -1:] - means[:, :, 1:3]
            rank_loss = F.softplus(-margins).mean()
            consistency = F.smooth_l1_loss(margins[:, 0, :], margins[:, 1, :])
            loss = interior_loss + .25 * edge_loss + .35 * rank_loss + .12 * consistency
            if extra_objective is not None:
                loss = loss + extra_objective(classifier, mean, scale, indices, local[:, 1], boundary[:, 1])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        trace.append(round(sum(losses)/len(losses), 8))
    return classifier.eval(), mean, scale, trace


def best_interval(local, edge, spans):
    from app.detectors.audio.content import window_in_speech

    peak = None
    for duration in LENGTHS:
        for start in range(0, FRAMES - duration + 1, STRIDE):
            end = start + duration
            lo, hi = start * HOP, end * HOP
            if not window_in_speech(spans, lo / RATE, hi / RATE, fraction=.5):
                continue
            onset = max(edge[max(0, start-4):min(FRAMES, start+5)])
            tail = max(edge[max(0, end-4):min(FRAMES, end+5)])
            strength = float(local[start:end].mean()) + .075 * float(onset + tail)
            proposal = (strength, -lo, -hi)
            if peak is None or proposal > peak:
                peak = proposal
    return None if peak is None else {"score": round(peak[0], 8), "range": [-peak[1], -peak[2]]}


def score_rows(classifier, norm, root, rows, bank, activity, *, feature_of=None, select=best_interval):
    import torch

    mean, scale = norm
    cache, result = {}, []
    with torch.inference_mode():
        for row in rows:
            key = row["pcm_sha256"]
            if key not in cache:
                signal = pcm(root, row)
                spans = activity.spans(signal)
                raw = features(signal, bank) if feature_of is None else feature_of(root, row)
                value = (raw - mean[:, None]) / scale[:, None]
                local, edges = classifier(torch.from_numpy(value[None]))[0].numpy()
                cache[key] = select(local, edges, spans)
            result.append({"pcm_sha256": key, "best": cache[key]})
    return result


def relevant(plan, role, generator, *, calibration=False):
    kinds = {"untouched", "genuine_splice", "background", f"resampled/{generator}"}
    if not calibration:
        kinds.add(f"edited/{generator}")
    return [r for r in plan["rows"] if r["role"] == role and r["kind"] in kinds]


def scores_at_points(rows, responses):
    foreground, backgrounds = [], []
    for row, measured in zip(rows, responses, strict=True):
        SOURCE.check(row["pcm_sha256"] == measured["pcm_sha256"] and row["truth"] is None,
                     "Operating point received ineligible data")
        if measured["best"] is not None:
            (backgrounds if row["kind"] == "background" else foreground).append(measured["best"]["score"])
    SOURCE.check(bool(foreground), "No ordinary scores for calibration")
    tail = sorted(foreground)[math.ceil(.95 * len(foreground)) - 1]
    highest_background = max(backgrounds, default=-math.inf)
    return {"strict": max(max(foreground), highest_background), "wider": max(tail, highest_background)}


def study(args):
    import torch
    from safetensors.torch import save_file

    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_second_classifier_overlap import SPEECH_SHA256
    from scripts.audio_strong_traffic_view import network_free

    plan, digest = load_plan(args.data)
    SOURCE.check(SOURCE.path_sha(args.speech) == SPEECH_SHA256, "Speech activity weights differ")
    args.output.mkdir(mode=0o700)
    lock = {"schema": SCHEMA, "method": METHOD, "source_sha256": digest, "seed": SEED, "epochs": EPOCHS,
            "speech_sha256": SPEECH_SHA256}
    SOURCE.write(args.output / "freeze.json", lock)
    network_free()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    bank, activity = mel_filterbank(), SpeechActivity(args.speech)
    fitted = {}
    for trained_family, test_family in (sources.FAMILIES, sources.FAMILIES[::-1]):
        classifier, mean, scale, losses = train(args.data, plan, trained_family, bank)
        weights = args.output / f"{trained_family}.safetensors"
        save_file({**classifier.state_dict(), "feature_mean": torch.from_numpy(mean),
                   "feature_scale": torch.from_numpy(scale)}, str(weights))
        calibration = relevant(plan, "calibration", trained_family, calibration=True)
        scores = score_rows(classifier, (mean, scale), args.data, calibration, bank, activity)
        points = scores_at_points(calibration, scores)
        seal = {"trained_family": trained_family, "held_family": test_family, "weights_sha256": SOURCE.path_sha(weights),
                "losses": losses, "points": points, "calibration_scores": scores}
        SOURCE.write(args.output / f"{trained_family}-points.json", seal)
        fitted[trained_family] = classifier, (mean, scale), seal
        print(f"Sealed {trained_family} fit and calibration before scoring {test_family}.", flush=True)
    cases = []
    for classifier, norm, seal in fitted.values():
        held = relevant(plan, "held", seal["held_family"])
        scored = score_rows(classifier, norm, args.data, held, bank, activity)
        cases.append(seal | {"held_scores": scored})
        print(f"Scored {len(scored)} held examples for {seal['held_family']}.", flush=True)
    SOURCE.write(args.output / "candidate.json", {"freeze": lock, "cases": cases})


def incumbent(args):
    import torch

    from app.detectors.audio import short_voice_review
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    plan, digest = load_plan(args.data)
    network_free()
    torch.set_num_threads(2)
    model = load_voice({"weights": args.weights, "assets": ASSETS})
    activity = SpeechActivity(args.weights / "silero_vad.jit")

    class Deadline:
        def check(self, _):
            pass  # Repeatable offline accuracy comparison.

    selected = [row for row in plan["rows"] if row["role"] == "held" and row["condition"] == "competing_speech"]
    output, cache = [], {}
    for row in selected:
        key = row["pcm_sha256"]
        if key not in cache:
            signal = pcm(args.data, row)
            response = short_voice_review.scan(signal, activity.spans(signal), model, Deadline(), codec="pcm_s16le")
            item = response["selected"]
            selected_range = [item["start_sample"], item["end_sample"]] if item is not None else None
            cache[key] = {"pcm_sha256": key, "selected": selected_range, "assessed": bool(response["responses"])}
        output.append(cache[key])
        if len(output) % 24 == 0:
            print(f"Incumbent measured {len(output)}/{len(selected)} held mixtures.", flush=True)
    SOURCE.write(args.output, {"source_sha256": digest, "assets": ASSETS, "results": output})


def summarize(rows, decisions, assessment, controls):
    from scripts.review_audio_stock_voice_transfer import useful

    selection = {(row["group"], row["condition"], row["kind"]): decision for row, decision in zip(rows, decisions, strict=True)}
    counts = defaultdict(Counter)
    readers = defaultdict(set)
    for row, selected, assessed in zip(rows, decisions, assessment, strict=True):
        condition, kind, group = row["condition"], row["kind"], row["group"]
        key = condition, kind.split("/", 1)[0]
        if kind == "genuine_splice" or kind == "untouched" or kind.startswith("resampled/"):
            key = condition, "genuine"
        tally = counts[key]
        tally["files"] += 1
        tally["assessed"] += assessed
        tally["flags"] += selected is not None
        if row["truth"] is not None:
            correct = selected is not None and useful(selected, row["truth"])
            tally["useful"] += correct
            tally["misplaced"] += selected is not None and not correct
            if correct:
                collided = any(selection.get((group, condition, name)) == selected for name in
                               ("untouched", "genuine_splice", kind.replace("edited/", "resampled/"), "background"))
                tally["useful_without_same_control_range"] += not collided
                if not collided:
                    readers[key].add(controls[group])
    return [dict(condition=kind[0], kind=kind[1], **total,
                 useful_readers_without_collision=len(readers[kind])) for kind, total in sorted(counts.items())]


def verdicts(rows, scored, point):
    results = []
    for row, measured in zip(rows, scored, strict=True):
        SOURCE.check(row["pcm_sha256"] == measured["pcm_sha256"], "Measured input differs")
        if (item := measured["best"]) is not None:
            a, b = item["range"]
            SOURCE.check(a % (STRIDE * HOP) == 0 and (b-a) in [HOP * length for length in LENGTHS]
                         and 0 <= a < b <= SAMPLES
                         and math.isfinite(item["score"]), "Invalid localization")
        results.append(item["range"] if item is not None and item["score"] > point else None)
    return results


def paired_gains(rows, baseline, candidates):
    from scripts.review_audio_stock_voice_transfer import useful

    totals = Counter()
    for row, current, trial in zip(rows, baseline, candidates, strict=True):
        if row["truth"] is not None:
            reached = trial is not None and useful(trial, row["truth"])
            prior = current is not None and useful(current, row["truth"])
            totals["additional_useful"] += reached and not prior
            totals["additional_misplaced"] += trial is not None and not reached and current is None
        elif row["kind"] == "background":
            totals["additional_background_flags"] += trial is not None and current is None
        else:
            totals["additional_genuine_flags"] += trial is not None and current is None
    return dict(totals)


def matched_order(rows, measured):
    records = {(row["group"], row["condition"], row["kind"]): score["best"]
               for row, score in zip(rows, measured, strict=True)}
    counts = defaultdict(Counter)
    for row, score in zip(rows, measured, strict=True):
        if row["truth"] is None:
            continue
        group, condition, kind = row["group"], row["condition"], row["kind"]
        controls = [records[group, condition, key] for key in
                    ("untouched", "genuine_splice", kind.replace("edited/", "resampled/"))]
        if score["best"] is not None and all(item is not None for item in controls):
            counts[condition]["fully_assessed_groups"] += 1
            counts[condition]["edit_exceeds_all_three_controls"] += all(
                score["best"]["score"] > item["score"] for item in controls)
    return dict(counts)


def result(record, *, schema=SCHEMA, method=METHOD):
    plan = record["sources"]
    SOURCE.check(plan["schema"] == sources.SCHEMA and plan["method"] == sources.METHOD, "Source policy differs")
    validate_plan(plan)
    candidate, incumbent = record["candidate"], record["incumbent"]
    encoded = json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n"
    digest = SOURCE.digest(encoded.encode("utf-8"))
    SOURCE.check(candidate["freeze"]["schema"] == schema and candidate["freeze"]["method"] == method
                 and candidate["freeze"]["source_sha256"] == incumbent["source_sha256"] == digest,
                 "Analysis commitments differ")
    SOURCE.check(candidate["freeze"]["seed"] == SEED and candidate["freeze"]["epochs"] == EPOCHS,
                 "Training settings differ")
    controls = {row["group"]: row["reader_sha256"] for row in plan["sources"]}
    compared = [row for row in plan["rows"] if row["role"] == "held" and row["condition"] == "competing_speech"]
    baseline = {}
    for row, score in zip(compared, incumbent["results"], strict=True):
        SOURCE.check(row["pcm_sha256"] == score["pcm_sha256"], "Incumbent PCM differs")
        baseline[row["pcm_sha256"]] = score
    policies = {}
    SOURCE.check({(r["trained_family"], r["held_family"]) for r in candidate["cases"]} ==
                 {sources.FAMILIES, sources.FAMILIES[::-1]} and len(candidate["cases"]) == 2, "Generator roles differ")
    for case in candidate["cases"]:
        held = relevant(plan, "held", case["held_family"])
        cal = relevant(plan, "calibration", case["trained_family"], calibration=True)
        points = scores_at_points(cal, case["calibration_scores"])
        SOURCE.check(case["points"] == points, "Cutoffs failed replay")
        common = [row for row in held if row["condition"] == "competing_speech"]
        base = [baseline[row["pcm_sha256"]] for row in common]
        selected = [score["selected"] for score in base]
        policies[case["held_family"]] = {
            "incumbent": summarize(common, selected, [score["assessed"] for score in base], controls),
            "matched_score_order": matched_order(held, case["held_scores"]),
        }
        for mode, point in points.items():
            calls = verdicts(held, case["held_scores"], point)
            assessed = [score["best"] is not None for score in case["held_scores"]]
            wet_calls = [call for row, call in zip(held, calls, strict=True) if row["condition"] == "competing_speech"]
            policies[case["held_family"]][mode] = {"standalone": summarize(held, calls, assessed, controls),
                                                   "supplement": paired_gains(common, selected, wet_calls)}
    return policies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    for name, fields in (("study", ("data", "output", "speech")),
                         ("incumbent", ("data", "output", "weights")),
                         ("combine", ("data", "candidate", "incumbent", "output")),
                         ("review", ("receipt",))):
        command = commands.add_parser(name)
        for field in fields:
            command.add_argument(field, type=Path)
    args = parser.parse_args()
    if args.phase == "study":
        study(args)
    elif args.phase == "incumbent":
        incumbent(args)
    elif args.phase == "combine":
        plan, digest = load_plan(args.data)
        candidate_data, incumbent_data = SOURCE.read(args.candidate), SOURCE.read(args.incumbent)
        SOURCE.check(digest == candidate_data["freeze"]["source_sha256"] == incumbent_data["source_sha256"],
                     "Wrong source freeze")
        record = {"sources": plan, "candidate": candidate_data, "incumbent": incumbent_data}
        record["results"] = result(record)
        SOURCE.write(args.output, record)
    else:
        record = SOURCE.read(args.receipt)
        out = result(record)
        SOURCE.check(out == record["results"], "Published counts failed replay")
        print(out)


if __name__ == "__main__":
    main()
