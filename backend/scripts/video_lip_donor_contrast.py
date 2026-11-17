# SPDX-License-Identifier: Apache-2.0
"""Compare mouth-speech agreement learned from matched recordings of the same speaker."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts import video_lip_sequence_panel as previous

SCHEMA = "chipmunk.video-lip-donor-contrast/1"
DELIVERIES = previous.DELIVERIES
ROLE = "different_words"
POINTS = {"strict": 0, "wider": 2}
BINS = 16
SEED = 111617
EPOCHS = 28
RATE = 16_000
WIDTHS = (6_400, 7_680, 8_960, 10_240, 11_520, 12_800)
LAGS = tuple(range(-3, 4))
ARMS = ("geometry", "mouth_texture")
METHODS = ("local", "bracketed")


def plan_confirmation(parent_path, frozen_path, sources, output):
    from scripts.lombard_mouth_source_intake import index_zip

    load, save, sha = previous.panel.read, previous.panel.save, previous.panel.sha
    parent, frozen = load(parent_path), load(frozen_path)
    if frozen["source"]["plan"] != sha(parent_path):
        raise ValueError("The confirmation must exclude all original study inputs")
    seen = [row[key] for row in parent["rows"] for key in ("target", "same_words", "different_words") if key in row]
    for name in ("video-lip-correspondence-sources-2026-11-16.json", "video-lip-stereo-source-plan-2026-11-16.json"):
        path = previous.prior.ROOT / "docs/evaluation-panels" / name
        if path.is_file():
            historical = json.loads(path.read_bytes())
            seen.extend(row[key] for row in historical.get("cases", historical.get("rows", []))
                        for key in ("target", "same_words", "different_words") if key in row)
    used_ids = {row["id"] for row in seen}
    used_audio = {row["audio_sha256"] for row in seen}
    used_pictures = {row["front_sha256"] for row in seen if "front_sha256" in row}
    excluded = set(frozen["excluded_speakers"])
    choices = defaultdict(dict)
    rows = []
    with (ZipFile(sources / "publisher-json.zip") as labels, ZipFile(sources / "alignment.zip") as alignment,
          ZipFile(sources / "lombardgrid_audio.zip") as audio, ZipFile(sources / "lombardgrid_front.zip") as front):
        eye, ear = index_zip(front, ".mov"), index_zip(audio, ".wav")
        bounds = index_zip(alignment, ".json")
        for metadata in index_zip(labels, ".json").values():
            for row in json.loads(labels.read(metadata)):
                speaker = previous.prior.keyed(row["SPKR"])
                condition = row["COND"]
                uid = f"{row['SPKR']}_{condition}_{row['UTTERANCE']}"
                if (speaker in excluded or uid in used_ids or not all(uid in source for source in (eye, ear, bounds))
                        or condition not in {"p", "l"} or row["STATUS"] != "CORRECT"):
                    continue
                record = alignment.read(bounds[uid])
                if span := previous.prior.parent.word_bounds(json.loads(record)[uid]):
                    choices[speaker, condition][uid] = span, record
        # Rank original source groups once, without reading any candidate response.
        order = sorted(choices, key=lambda key: (hashlib.sha256(f"{SCHEMA}/{key[0]}".encode()).hexdigest(), key[1]))
        for speaker, condition in order:
            if case := previous.prior.matched_records(choices[speaker, condition], audio, front, ear, eye,
                                                    used_audio, used_pictures):
                rows.append({"speaker": speaker, "condition": condition, "split": "held", **case})
                used_audio.update(item["audio_sha256"] for item in case.values())
                used_pictures.add(case["target"]["front_sha256"])
                if len(rows) == 40:
                    break
    print(f"Qualified {len(rows)} of {len(order)} speaker and recording-condition groups.", flush=True)
    if len(rows) < 24 or len({row["speaker"] for row in rows}) < 12:
        raise ValueError("Too few independent matched recordings for confirmation")
    save(output, {"schema": previous.SCHEMA, "method": previous.STUDY_METHOD, "rows": rows, "publisher": parent["publisher"],
                  "counts": {"held": {"recordings": len(rows)}}, "reference_plan": sha(parent_path), "frozen_fit": sha(frozen_path)})
    print(f"{len(rows)} new source pictures from {len({row['speaker'] for row in rows})} speakers.", flush=True)


def observations(grid, times, centers):
    where = np.searchsorted(times, centers, side="right") - 1
    appropriate = (where >= 0) & (where < len(times) - 1)
    where = where.clip(0, len(times) - 2)
    relative = ((centers - times[where]) / (times[where + 1] - times[where])).clip(0, 1)[..., None]
    values = (1 - relative) * grid["features"][where] + relative * grid["features"][where + 1]
    measured = appropriate & grid["active"][where] & grid["active"][where + 1]
    return values, measured


def intervals(begin, end):
    return np.asarray(begin)[..., None] + ((np.arange(BINS) + .5) / BINS) * (np.asarray(end) - np.asarray(begin))[..., None]


def whole_audio(grid, positions):
    return observations(grid, np.arange(len(grid["active"])) * 160, positions)


def packet(row, words):
    original = previous._arrays(row)
    start, stop = row["truth"]
    return original | {"words": words, "positions": intervals(start, stop), "length": stop - start}


def prepare_fit(plan, source, arm):
    by_picture = {row["target"]["front_sha256"]: row for row in plan["rows"] if row["split"] == "fit"}
    records = defaultdict(list)
    for row in source:
        label = by_picture[row["picture_sha256"]]
        if not (label["speaker"] == row["speaker"] and label["condition"] == row["condition"]
                and label["target"]["span"] == row["truth"]):
            raise ValueError("Fit landmarks and audio differ from the publisher record")
        if row["visual"][arm] and row["audio"]:
            records[row["speaker"], row["condition"]].append(packet(row, label["target"]["id"].split("_")[-1][3:5]))
    mouths, sounds, mismatches, sources = [], [], [], set()
    rng = np.random.default_rng(SEED)
    coverage = Counter()
    for speech in records.values():
        for target in speech:
            video, ok = observations(target["visual"][arm], target["times"], target["positions"])
            if ok.sum() < 12:
                continue
            other_words = [source for source in speech if all(x != y for x, y in zip(source["words"], target["words"], strict=True))
                           and .75 <= source["length"] / target["length"] <= 1.25]
            if not other_words:
                continue
            same_words = [source for source in speech if source is not target and source["words"] == target["words"]
                          and .75 <= source["length"] / target["length"] <= 1.25]
            # Keep same speaker and capture condition in all negative pairs.
            donors = ([target] * 4 + same_words[:2], [other_words[i] for i in rng.permutation(len(other_words))[:4]])
            for label, examples in enumerate(donors):
                for i, donor in enumerate(examples):
                    delta = int(rng.integers(-480, 481)) if i else 0
                    audio, active = whole_audio(donor["audio"], donor["positions"] + delta)
                    if active.sum() >= 9:
                        mouths.append(video)
                        sounds.append(audio)
                        mismatches.append(label)
                        sources.add((target["speaker"], target["picture_sha256"]))
                        coverage["mismatched" if label else "synchronized" if donor is target else "same_words"] += 1
    if len(sources) < 50 or set(mismatches) != {0, 1}:
        raise ValueError("Too few source-matched examples for both branches")
    return (*[torch.as_tensor(np.array(rows), dtype=torch.float32) for rows in (mouths, sounds, mismatches)],
            dict(coverage), len(sources), len({speaker for speaker, _ in sources}))


class Similarity(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.visual = nn.Sequential(nn.Conv1d(channels, 20, 3, padding=1), nn.GELU(), nn.Conv1d(20, 12, 3, padding=1))
        self.audio = nn.Sequential(nn.Conv1d(14, 20, 3, padding=1), nn.GELU(), nn.Conv1d(20, 12, 3, padding=1))
        self.interaction = nn.Sequential(nn.Conv1d(24, 12, 3, padding=1), nn.GELU(), nn.Conv1d(12, 1, 1))

    def forward(self, picture, audio):
        visual = F.normalize(self.visual(picture.transpose(1, 2)), dim=1)
        speech = F.normalize(self.audio(audio.transpose(1, 2)), dim=1)
        state = self.interaction(torch.cat((visual * speech, torch.abs(visual - speech)), dim=1))[:, 0]
        return (state.mean(dim=1) + state.topk(3).values.mean(dim=1)) / 2


def standardize(data, parameters):
    return ((data - parameters[0]) / parameters[1]).clamp(-5, 5)


def fit(plan, source, arm):
    picture, audio, labels, counts, targets, speakers = prepare_fit(plan, source, arm)
    scales = tuple((data.mean(dim=(0, 1)), data.std(dim=(0, 1)).clamp(min=.05)) for data in (picture, audio))
    picture, audio = (standardize(data, scale) for data, scale in zip((picture, audio), scales, strict=True))
    model = Similarity(picture.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.004)
    for epoch in range(EPOCHS):
        for indices in torch.randperm(len(labels)).split(48):
            rgb, wav = F.dropout(picture[indices], .15, training=True), F.dropout(audio[indices], .15, training=True)
            value = F.binary_cross_entropy_with_logits(model(rgb, wav), labels[indices])
            optimizer.zero_grad()
            value.backward()
            optimizer.step()
        if (epoch + 1) % 7 == 0:
            print(f"{arm} epoch {epoch + 1}/{EPOCHS}: loss {value.item():.4f}", flush=True)
    record = {"weights": {key: value.tolist() for key, value in model.state_dict().items()},
              "scale": [[part.tolist() for part in scale] for scale in scales],
              "pairs": counts, "targets": targets, "speakers": speakers}
    return (model.eval(), scales), record


@torch.inference_mode()
def measure(model, row, acoustic, arm):
    visual = row["visual"][arm]
    if visual is None or acoustic is None:
        return {}
    net, scales = model
    times = row["times"]
    edge = round(np.median(np.diff(times)))
    spans = np.array([(a, a + duration) for duration in WIDTHS for a in times[5:-5:2] if a + duration <= times[-6]])
    if not len(spans):
        return {}
    positions = intervals(spans[:, 0], spans[:, 1])
    features, support = observations(visual, times, positions)
    audio, active = whole_audio(acoustic, positions[:, None] + np.asarray(LAGS)[None, :, None] * edge)
    eligible = (support.sum(axis=-1) >= 12)[:, None] & (active.sum(axis=-1) >= 9)
    norm_mouth = standardize(torch.from_numpy(features).float(), scales[0])
    norm_audio = standardize(torch.from_numpy(audio).float(), scales[1])
    scores = net(norm_mouth[:, None].expand(-1, len(LAGS), -1, -1).flatten(0, 1),
                 norm_audio.flatten(0, 1)).numpy().reshape(eligible.shape)
    counts = eligible.sum(axis=0)
    lanes = [i for i, n in enumerate(counts) if n >= max(12, round(len(spans) * .45))]
    if not lanes:
        return {}
    lane = min(lanes, key=lambda i: (float(np.median(scores[eligible[:, i], i])), abs(LAGS[i]), LAGS[i]))
    scores = np.min(np.where(eligible[:, max(0, lane-1):lane+2], scores[:, max(0, lane-1):lane+2], np.inf), axis=1)
    median = np.median(scores[np.isfinite(scores)])
    direct, bracketed = None, None
    for index, (begin, end) in enumerate(spans):
        if not np.isfinite(scores[index]) or not eligible[index, lane]:
            continue
        left = (spans[:, 1] <= begin + edge) & (spans[:, 1] >= begin - 3 * edge) & np.isfinite(scores)
        right = (spans[:, 0] >= end - edge) & (spans[:, 0] <= end + 3 * edge) & np.isfinite(scores)
        audio_range = [int(value + edge * LAGS[lane]) for value in (begin, end)]
        observation = {"audio": audio_range, "video": [int(begin), int(end)], "lag_samples": int(edge * LAGS[lane]),
                       "frames": [int(np.argmin(np.abs(times - time))) for time in (begin, (begin + end) / 2, end)]}
        if direct is None or scores[index] > direct["score"]:
            direct = observation | {"score": round(float(scores[index]), 6)}
        if np.any(left) and np.any(right):
            value = scores[index] - max(float(np.median(scores[left])), float(np.median(scores[right])), float(median))
            if bracketed is None or value > bracketed["score"]:
                bracketed = observation | {"score": round(float(value), 6)}
    return {key: value for key, value in zip(METHODS, (direct, bracketed), strict=True) if value is not None}


def measure_rows(models, capture):
    for raw in capture["rows"]:
        row = previous._arrays(raw)
        for delivery, cases in raw["cases"].items():
            for role, case in sorted(cases.items()):
                acoustic = {key: np.asarray(value) for key, value in case["audio"].items()} if case["audio"] else None
                measured = {f"{arm}/{method}": value for arm in ARMS
                            for method, value in measure(models[arm], row, acoustic, arm).items()}
                yield {key: raw[key] for key in ("speaker", "picture_sha256", "truth", "pts", "time_base")} | {
                    "delivery": delivery, "role": role, "baseline": case["current_span"],
                    "current_status": case["current_status"], "pcm_sha256": case["source_pcm_sha256"],
                    "seconds_length_samples": raw["samples"], "measured": measured}


def useful(interval, truth):
    if not interval:
        return False
    shared = max(0, min(interval[1], truth[1]) - max(interval[0], truth[0]))
    return 5 * shared >= 2 * (interval[1] - interval[0] + truth[1] - truth[0] - shared)


def baseline(row):
    span = row["baseline"]
    if span is not None and (len(span) != 2 or any(type(point) is not int for point in span)):
        raise ValueError("The current timing range is not on the same decoded audio clock")
    return span


def recount(rows, arm, point):
    counts = Counter()
    new = set()
    controls = set()
    for row in rows:
        role, truth = row["role"], row["truth"]
        former = baseline(row)
        proposal = row["measured"].get(arm)
        shown = proposal["audio"] if not former and proposal and proposal["score"] > point else None
        if role == ROLE:
            correct_before, correct_after = useful(former, truth), useful(shown, truth)
            counts.update(edits=1, baseline=correct_before, combined=correct_before or correct_after,
                          misplaced=bool(shown) and not correct_after)
            if correct_after:
                new.add(row["speaker"])
        else:
            category = "whole_delay" if role.startswith("global_") else "controls" if role in {"original", "gain"} else "same_words"
            counts.update({category: 1, f"{category}_baseline": bool(former), f"{category}_combined": bool(former or shown)})
            if shown:
                controls.add(row["speaker"])
    return dict(counts) | {"additional_speakers": len(new), "additional_control_speakers": len(controls)}


def freeze_points(rows):
    options, choices = {}, {}
    for carrier in DELIVERIES:
        cohort = [row for row in rows if row["delivery"] == carrier]
        scores = []
        for arm in (f"{features}/{rule}" for features in ARMS for rule in METHODS):
            by_speaker = defaultdict(lambda: -1e10)
            for row in cohort:
                if row["role"] != ROLE and not baseline(row) and (entry := row["measured"].get(arm)):
                    key = row["speaker"]
                    by_speaker[key] = max(by_speaker[key], entry["score"])
            points = sorted(by_speaker.values(), reverse=True)
            if len(points) <= max(POINTS.values()):
                continue
            for name, offset in POINTS.items():
                key = f"{carrier}/{arm}/{name}"
                mark = points[offset]
                outcome = recount(cohort, arm, mark)
                costs = (-outcome["combined"], outcome["additional_control_speakers"], outcome["misplaced"], len(scores))
                scores.append((costs, key))
                options[key] = mark
        choices[carrier] = min(scores)[1]
    return options, choices


def run(folder, output):
    load, save, sha = previous.panel.read, previous.panel.save, previous.panel.sha
    paths = {name: folder / f"{name}.json.gz" for name in ("plan", "fit", "calibration", "held")}
    plan, train, tune = (load(paths[name]) for name in ("plan", "fit", "calibration"))
    if {row["speaker"] for row in train["rows"]} & {row["speaker"] for row in tune["rows"]}:
        raise ValueError("Calibration includes a training speaker")
    if train["plan_sha256"] != sha(paths["plan"]) or tune["plan_sha256"] != sha(paths["plan"]):
        raise ValueError("The original plan changed")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    models, weights = {}, {}
    for arm in ARMS:
        models[arm], weights[arm] = fit(plan, train["rows"], arm)
    calibration = list(measure_rows(models, tune))
    points, chosen = freeze_points(calibration)
    seen = sorted({row["speaker"] for data in (train, tune) for row in data["rows"]})
    fixed = {"schema": SCHEMA, "source": {key: sha(paths[key]) for key in ("plan", "fit", "calibration")},
             "weights": weights, "points": points, "chosen": chosen, "excluded_speakers": seen, "code": sha(Path(__file__))}
    save(output / "freeze.json.gz", fixed)
    print(json.dumps({"chosen": chosen, "weights": {k: v["pairs"] for k, v in weights.items()}}, sort_keys=True), flush=True)
    held = load(paths["held"])
    if held["plan_sha256"] != sha(paths["plan"]) or {row["speaker"] for row in held["rows"]} & set(seen):
        raise ValueError("Held inputs differ from their frozen and separated plan")
    measured = list(measure_rows(models, held))
    totals = {}
    for key, point in points.items():
        delivery, features, rule, _ = key.split("/")
        totals[key] = recount([row for row in measured if row["delivery"] == delivery], f"{features}/{rule}", point)
    save(output / "result.json.gz", {"freeze": fixed, "calibration": calibration, "held": measured,
                                      "held_sha256": sha(paths["held"]), "total": totals})
    print(json.dumps(totals, sort_keys=True), flush=True)


def assess_confirmation(frozen_path, plan_path, captured, output):
    load, save, sha = previous.panel.read, previous.panel.save, previous.panel.sha
    frozen, plan, payload = map(load, (frozen_path, plan_path, captured))
    if (plan["frozen_fit"] != sha(frozen_path) or payload["plan_sha256"] != sha(plan_path)
            or {row["speaker"] for row in payload["rows"]} & set(frozen["excluded_speakers"])):
        raise ValueError("Confirmation inputs do not match their separated plan")
    torch.set_num_threads(2)
    models = {}
    for arm, values in frozen["weights"].items():
        scales = tuple(tuple(torch.tensor(value) for value in element) for element in values["scale"])
        model = Similarity(len(scales[0][0]))
        model.load_state_dict({name: torch.tensor(value) for name, value in values["weights"].items()})
        models[arm] = model.eval(), scales
    results = list(measure_rows(models, payload))
    totals = {}
    for key, point in frozen["points"].items():
        delivery, features, rule, _ = key.split("/")
        totals[key] = recount([row for row in results if row["delivery"] == delivery], f"{features}/{rule}", point)
    save(output, {"freeze": frozen, "freeze_sha256": sha(frozen_path), "confirmation_sources_sha256": sha(plan_path),
                  "capture_sha256": sha(captured), "held": results, "total": totals})
    print(json.dumps(totals, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name, arguments in (("fit", ("folder", "output")),
                            ("plan", ("parent_path", "frozen_path", "sources", "output")),
                            ("assess", ("frozen_path", "plan_path", "captured", "output"))):
        mode = commands.add_parser(name)
        for operand in arguments:
            mode.add_argument(operand, type=Path)
    flags = vars(parser.parse_args())
    action = flags.pop("action")
    if action == "fit":
        flags["output"].mkdir(mode=0o700)
    {"fit": run, "plan": plan_confirmation, "assess": assess_confirmation}[action](**flags)
