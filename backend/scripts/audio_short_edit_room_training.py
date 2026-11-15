# SPDX-License-Identifier: MIT
"""Train a short speech edit localizer on paired levels and room responses."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from scripts import audio_short_edit_detector_training as base
from scripts import audio_two_voice_components as two_voice

SOURCE = base.SOURCE
PHONEME = two_voice.PHONEME
STOCK = two_voice.prior.STOCK
ROOM_TRANSFER = PHONEME.with_name("audio-short-edit-speech-layer-transfer-2026-11-14.json")
RATE, SAMPLES = base.RATE, base.SAMPLES
CONDITIONS = ("dry", "level", "room", "competing")
ARMS = ("dry_only", "paired", "paired_centered")
FAMILIES = base.sources.FAMILIES
STOCK_FAMILIES = ("piper", "kitten")
SPEECH = two_voice.SCORING.SPEECH_SHA256
SCHEMA = "chipmunk.audio-short-edit-room-training/1"
METHOD = {
    "sources": "Use the original 48 PhonemeDF fitting prompts, 16 calibration prompts and 24 held prompts. "
    "Use separate speaker groups and recorded background sessions within this study. Add eight held stock "
    "reader groups with two previously published generators. New room renders, prior original source banks.",
    "views": "Deliver dry, dry 8 dB quieter, isolated room-rendered, and room-rendered with the original "
    "competing conversation 5 dB louder. Match the untouched room level to dry, align the dominant impulse "
    "peak and use fixed gains and integer stems across all states of each matched source. Use Surrey for "
    "fit and calibration, BBC for PhonemeDF held, and Huddersfield for stock held.",
    "fit": "Use the same small 64-band temporal architecture, 16 epochs, deterministic seed and matched "
    "genuine-control content, boundary and local ranking objectives in three comparisons. Compare dry only; "
    "all four views with a dry-relative margin penalty; and the same paired fit after subtracting each clip's "
    "per-band median over detected speech windows. Balance one training generator per fold.",
    "readout": "Use the existing timed window search and speech gate. Freeze separate control-only maximum "
    "and nearest-rank 95th percentile from all four views of the calibration readers, bounded below by the "
    "highest isolated competing-voice control. Compare on the other "
    "PhonemeDF generator and both previously unseen stock generators. Use original-file times and report "
    "exact matched-control collisions, coverage, extra hits and false alarms. No held controls set a point.",
}


def contract():
    return {"schema": SCHEMA, "method": METHOD, "phoneme_sha256": SOURCE.path_sha(PHONEME),
            "stock_sha256": SOURCE.path_sha(STOCK), "speech_sha256": SPEECH, "seed": base.SEED, "epochs": base.EPOCHS}


def phoneme_groups(root):
    plan, _ = base.load_plan(root)
    SOURCE.check(all(plan[k] == v for k, v in SOURCE.read(PHONEME)["sources"].items()), "PhonemeDF roster differs")
    rows = {(row["group"], row["condition"], row["kind"]): row for row in plan["rows"]}
    for unit in plan["sources"]:
        group = unit["group"]
        dry = {kind: rows[group, "dry", kind] for kind in base.sources.KINDS if kind != "background"}
        buffer = {kind: base.pcm(root, row) for kind, row in dry.items()}
        noise = rows[group, "competing_speech", "background"]
        tag = {"panel": "phoneme", "id": group, "role": unit["role"], "speaker": unit["reader_sha256"],
               "capture": unit["background"]["room"], "span": unit["truth"],
               "inputs": {kind: row["pcm_sha256"] for kind, row in dry.items()}, "noise_sha256": noise["pcm_sha256"]}
        yield tag, buffer, base.pcm(root, noise)


def stock_groups(root):
    import numpy as np

    parent = SOURCE.read(STOCK)
    two_voice.prior.stock.verify(parent)
    path = root / "placements-freeze.json"
    SOURCE.check(SOURCE.path_sha(path) == parent["native_freeze_sha256"], "Stock plan differs")
    roster = SOURCE.read(path)
    data = defaultdict(dict)
    with np.load(root / "placements-native.npz", allow_pickle=False) as samples:
        dry, native = samples["dry"], samples["native"]
    SOURCE.check(all(a.shape == (83, SAMPLES) and a.dtype == np.dtype("<i2") for a in (dry, native)), "Stock clocks differ")
    kinds = {"untouched": "untouched", "same_reader_splice": "genuine_splice", "piper": "edited/piper", "kitten": "edited/kitten"}
    for original, mixture, row in zip(dry, native, parent["rows"], strict=True):
        if row["snr"] == -5 and row["kind"] in kinds:
            SOURCE.check(SOURCE.digest(original.tobytes()) == row["dry_pcm_sha256"]
                         and SOURCE.digest(mixture.tobytes()) == row["pcm_sha256"], "Stock PCM changed")
            label = kinds[row["kind"]]
            data[row["group"]][label] = (row, original, mixture.astype(np.int32) - original.astype(np.int32))
    for group, values in sorted(data.items()):
        bits = values["untouched"][2]
        words = values["edited/piper"][0]["edit_clock"]
        capture = values["untouched"][0]["capture"]
        SOURCE.check(all(np.array_equal(x[2], bits) for x in values.values()) and
                     values["edited/kitten"][0]["edit_clock"] == values["genuine_splice"][0]["ordinary_splice_clock"] == words,
                     "Stock ordinary and synthetic variants are not matched")
        tag = {"panel": "stock", "id": group, "role": "held", "span": words,
               "speaker": base.sources.order(str(roster["source_commitments"][group]["reader"])),
               "capture": roster["capture_commitments"][capture]["meeting"],
               "inputs": {kind: x[0]["dry_pcm_sha256"] for kind, x in values.items()},
               "noise_sha256": SOURCE.digest(bits.astype("<i4").tobytes())}
        yield tag, {kind: x[1].astype(np.float32) / 32768 for kind, x in values.items()}, bits.astype(np.float32) / 32768


def room_assets(archive):
    import numpy as np

    from scripts import audio_blind_huddersfield_transfer as huddersfield
    from scripts.audio_measured_spatial_transfer import load_rigs

    rooms, asset = load_rigs(archive)
    third, receipt = huddersfield.impulses(archive)
    rooms["huddersfield"] = third
    asset["huddersfield"] = receipt
    return {room: {name: np.asarray(value).mean(axis=1) for name, value in positions.items()}
            for room, positions in rooms.items()}, asset


def render(tag, stems, noise, rooms):
    import numpy as np
    from scipy.signal import fftconvolve

    room = "huddersfield" if tag["panel"] == "stock" else "bbc" if tag["role"] == "held" else "surrey"
    positions = ("front", "left", "right") if room != "huddersfield" else ("S01", "S03", "S07")
    position = positions[tag["id"] % len(positions)]
    impulse = rooms[room][position]
    arrival = int(np.argmax(abs(impulse)))
    start, stop = tag["span"]
    reference = stems["untouched"]
    for samples in stems.values():
        SOURCE.check(np.array_equal(samples[:start], reference[:start]) and np.array_equal(samples[stop:], reference[stop:]),
                     "Controlled voice changes outside the staged edit")
    filtered = {kind: fftconvolve(signal.astype(np.float64), impulse)[arrival:arrival + SAMPLES]
                for kind, signal in stems.items()}
    before = np.linalg.norm(reference.astype(np.float64))
    after = np.linalg.norm(filtered["untouched"])
    background = np.linalg.norm(noise.astype(np.float64))
    SOURCE.check(min(before, after, background) > 0, "A matched voice or background is silent")
    filtered = {kind: signal * (before / after) for kind, signal in filtered.items()}
    noise = noise.astype(np.float64) * (10 ** .25 * before / background)
    peak = max(max(np.max(abs(v)) for v in stems.values()),
               max(np.max(abs(y) + abs(noise)) for y in filtered.values()))
    gain = min(32768., 25000 / peak)
    dry = {name: np.rint(signal * gain).astype(np.int32) for name, signal in stems.items()}
    level = {name: np.rint(signal * (gain * 10 ** -.4)).astype(np.int32) for name, signal in stems.items()}
    only_room = {name: np.rint(signal * gain).astype(np.int32) for name, signal in filtered.items()}
    noise = np.rint(noise * gain).astype(np.int32)
    competing = {name: signal + noise for name, signal in only_room.items()}
    competing["background"] = noise
    output = dict(zip(CONDITIONS, (dry, level, only_room, competing), strict=True))
    for condition in output.values():
        SOURCE.check(all(np.max(abs(samples)) < 32768 and len(samples) == SAMPLES for samples in condition.values()),
                     "A matched delivered signal clips or loses its sample clock")
    return {"room": room, "position": position, "arrival": arrival}, output


def freeze(args):
    args.output.mkdir(mode=0o700)
    pcms = args.output / "pcm"
    pcms.mkdir(mode=0o700)
    rooms, receipts = room_assets(args.rooms)
    groups = []
    for load in (phoneme_groups(args.phoneme), stock_groups(args.stock)):
        for tag, native, background in load:
            delivery, encoded = render(tag, native, background, rooms)
            material = {}
            for arm, waves in encoded.items():
                material[arm] = {}
                for kind, samples in waves.items():
                    raw = samples.astype("<i2").tobytes()
                    witness = SOURCE.digest(raw)
                    target = pcms / f"{witness}.pcm"
                    if not target.exists():
                        target.write_bytes(raw)
                    material[arm][kind] = witness
            groups.append(tag | {"delivery": delivery, "observations": material})
    plan = {"contract": contract(), "rooms": receipts, "groups": groups}
    validate(plan)
    SOURCE.write(args.output / "freeze.json", plan)
    print(f"Froze {len(groups)} source groups and {sum(len(g['inputs']) * len(CONDITIONS) + 1 for g in groups)} deliveries",
          flush=True)


def validate(plan):
    SOURCE.check(plan["contract"] == contract(), "Study source or method differs")
    counts = Counter()
    speakers, recordings = defaultdict(set), defaultdict(set)
    for group in plan["groups"]:
        panel, role = group["panel"], group["role"]
        counts[panel, role] += 1
        speakers[role].add(group["speaker"])
        recordings[role].add(group["capture"])
        a, b = group["span"]
        SOURCE.check(0 < a < b <= SAMPLES and set(group["observations"]) == set(CONDITIONS), "Source geometry differs")
        for condition, states in group["observations"].items():
            expected = set(group["inputs"]) | ({"background"} if condition == "competing" else set())
            SOURCE.check(set(states) == expected and all(two_voice.prior.stock.prior.sha256(d) for d in states.values()),
                         "A delivered voice state or PCM witness differs")
    SOURCE.check(counts == {("phoneme", "fit"): 48, ("phoneme", "calibration"): 16,
                            ("phoneme", "held"): 24, ("stock", "held"): 8}, "Frozen panel sizes differ")
    for kind in (speakers, recordings):
        SOURCE.check(sum(map(len, kind.values())) == len(set.union(*kind.values())), "A reader or recording crosses roles")


def case_for(group, condition, kind):
    return {"panel": group["panel"], "group": group["id"], "speaker": group["speaker"], "condition": condition,
            "kind": kind, "truth": group["span"] if kind.startswith("edited/") else None,
            "pcm_sha256": group["observations"][condition][kind]}


def trial_cases(plan, role, trained):
    unknown = next(family for family in FAMILIES if family != trained)
    for group in plan["groups"]:
        if group["role"] != role:
            continue
        controls = ["untouched", "genuine_splice"]
        if group["panel"] == "phoneme":
            seen = trained if role != "held" else unknown
            controls.append(f"resampled/{seen}")
            additions = [f"edited/{seen}"] if role != "calibration" else []
        else:
            additions = [f"edited/{family}" for family in STOCK_FAMILIES]
        for condition in CONDITIONS:
            for kind in (*controls, *additions):
                yield case_for(group, condition, kind)
            if condition == "competing":
                yield case_for(group, condition, "background")


def runtime(speech_path):
    import torch

    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free

    SOURCE.check(SOURCE.path_sha(speech_path) == SPEECH, "Speech activity weights changed")
    network_free()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    return SpeechActivity(speech_path)


def feature_inputs(folder, speech):
    import numpy as np

    mel = base.mel_filterbank()
    clock = np.arange(base.FRAMES) * base.HOP / RATE

    @lru_cache(maxsize=1800)
    def read(digest):
        samples = base.pcm(folder, {"pcm_sha256": digest})
        return base.features(samples, mel), speech.spans(samples)

    def extract(digest, centered):
        features, spans = read(digest)
        if centered and spans:
            frames = np.zeros(base.FRAMES, dtype=bool)
            for a, b in spans:
                frames |= (clock >= a) & (clock < b)
            if np.any(frames):
                features = features - np.median(features[:, frames], axis=1, keepdims=True)
        return features, spans

    return extract


def training_input(plan, family, arm, extract):
    import numpy as np

    conditions = CONDITIONS[:1] if arm == "dry_only" else CONDITIONS
    kinds = ("untouched", "genuine_splice", f"resampled/{family}", f"edited/{family}")
    data, intervals, endpoints = [], [], []
    clock = np.arange(base.FRAMES) * base.HOP
    for group in plan["groups"]:
        if group["role"] != "fit":
            continue
        data.append([[extract(group["observations"][condition][kind], arm == "paired_centered")[0]
                      for kind in kinds] for condition in conditions])
        start, stop = group["span"]
        intervals.append((clock >= start) & (clock < stop))
        endpoints.append((abs(clock - start) <= 480) | (abs(clock - stop) <= 480))
    return np.asarray(data), np.asarray(intervals, np.float32), np.asarray(endpoints, np.float32)


def train_model(plan, family, arm, extract):
    import numpy as np
    import torch
    from torch.nn import functional as F

    features, intervals, endpoints = training_input(plan, family, arm, extract)
    axes = (0, 1, 2, 4)
    mean = features.mean(axis=axes, dtype=np.float64).astype(np.float32)
    scale = np.maximum(.001, features.std(axis=axes, dtype=np.float64)).astype(np.float32)
    samples = torch.from_numpy((features - mean[:, None]) / scale[:, None])
    regions, edges = map(torch.from_numpy, (intervals, endpoints))
    torch.manual_seed(base.SEED)
    model = base.model().train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    trace = []
    for _ in range(base.EPOCHS):
        losses = []
        for indices in torch.randperm(len(samples)).split(6):
            batch = samples[indices]
            window = regions[indices, None, None, :]
            content, boundary = model(batch.flatten(0, 2)).reshape(*batch.shape[:3], 2, base.FRAMES).unbind(dim=3)
            y = torch.zeros_like(content)
            y[:, :, -1:, :] = window
            z = torch.zeros_like(boundary)
            z[:, :, -1:, :] = edges[indices, None, None, :]
            loss = F.binary_cross_entropy_with_logits(content, y, pos_weight=torch.tensor(8.))
            loss += .25 * F.binary_cross_entropy_with_logits(boundary, z, pos_weight=torch.tensor(20.))
            local_means = (content * window).sum(dim=-1) / window.sum(dim=-1)
            margins = local_means[:, :, -1:] - local_means[:, :, 1:3]
            loss += .35 * F.softplus(-margins).mean()
            if arm != "dry_only":
                loss += .12 * F.smooth_l1_loss(margins[:, 1:], margins[:, :1].expand_as(margins[:, 1:]))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
        trace.append(round(sum(losses) / len(losses), 8))
    return model.eval(), mean, scale, trace


def score_rows(model, mean, scale, rows, arm, extract):
    import torch

    readings = []
    cache = {}
    with torch.inference_mode():
        for row in rows:
            digest = row["pcm_sha256"]
            if digest not in cache:
                signal, spans = extract(digest, arm == "paired_centered")
                if spans:
                    content, edge = model(torch.from_numpy(((signal - mean[:, None]) / scale[:, None])[None]))[0].numpy()
                    cache[digest] = base.best_interval(content, edge, spans)
                else:
                    cache[digest] = None
            readings.append({"pcm_sha256": digest, "best": cache[digest]})
    return readings


def plan_hash(plan):
    return SOURCE.digest((json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def fitted(args):
    import torch
    from safetensors.torch import save_file

    plan = SOURCE.read(args.inputs / "freeze.json")
    validate(plan)
    speech = runtime(args.speech)
    extract = feature_inputs(args.inputs, speech)
    args.output.mkdir(mode=0o700)
    lock = {"plan_sha256": plan_hash(plan), "contract": contract(), "trials": []}
    for family in FAMILIES:
        cases = list(trial_cases(plan, "calibration", family))
        for arm in ARMS:
            model, mean, scale, trace = train_model(plan, family, arm, extract)
            weights = args.output / f"{family}-{arm}.safetensors"
            save_file({**model.state_dict(), "feature_mean": torch.from_numpy(mean), "feature_scale": torch.from_numpy(scale)},
                      str(weights))
            scores = score_rows(model, mean, scale, cases, arm, extract)
            lock["trials"].append({"family": family, "arm": arm, "loss": trace, "weights_sha256": SOURCE.path_sha(weights),
                                   "calibration": scores, "points": base.scores_at_points(cases, scores)})
            print(f"Fitted {family} / {arm} and sealed calibration", flush=True)
    SOURCE.write(args.output / "fit.json", lock)


def verify_fit(plan, lock):
    SOURCE.check(lock["plan_sha256"] == plan_hash(plan) and lock["contract"] == contract(), "Fit came from another plan")
    SOURCE.check([(trial["family"], trial["arm"]) for trial in lock["trials"]] ==
                 [(family, arm) for family in FAMILIES for arm in ARMS], "Fit inventory differs")
    for trial in lock["trials"]:
        cases = list(trial_cases(plan, "calibration", trial["family"]))
        points = base.scores_at_points(cases, trial["calibration"])
        SOURCE.check(points == trial["points"] and len(trial["loss"]) == base.EPOCHS
                     and all(math.isfinite(loss) for loss in trial["loss"]), "Calibration or training trace differs")
        base.verdicts(cases, trial["calibration"], points["strict"])


def load_trial(folder, trial):
    from safetensors.torch import load_file

    name = f"{trial['family']}-{trial['arm']}.safetensors"
    path = folder / name
    SOURCE.check(SOURCE.path_sha(path) == trial["weights_sha256"], "Held model differs from calibration model")
    state = load_file(str(path))
    mean, scale = (state.pop(key).numpy() for key in ("feature_mean", "feature_scale"))
    model = base.model().eval()
    model.load_state_dict(state, strict=True)
    return model, mean, scale


def confirm(args):
    plan = SOURCE.read(args.inputs / "freeze.json")
    validate(plan)
    fitted_path = args.fit / "fit.json"
    lock = SOURCE.read(fitted_path)
    verify_fit(plan, lock)
    speech = runtime(args.speech)
    extract = feature_inputs(args.inputs, speech)
    args.output.mkdir(mode=0o700)
    commitment = {"plan_sha256": plan_hash(plan), "fit_sha256": SOURCE.path_sha(fitted_path)}
    SOURCE.write(args.output / "freeze.json", commitment)
    measurements = []
    for trial in lock["trials"]:
        family, arm = trial["family"], trial["arm"]
        model, mean, scale = load_trial(args.fit, trial)
        cases = list(trial_cases(plan, "held", family))
        scored = score_rows(model, mean, scale, cases, arm, extract)
        measurements.append({"family": family, "arm": arm, "scores": scored})
        print(f"Held {family} / {arm} scored {len(scored)} deliveries", flush=True)
    SOURCE.write(args.output / "held.json", {**commitment, "trials": measurements})


def transfer_cases():
    for row in SOURCE.read(ROOM_TRANSFER)["rows"]:
        kind = f"edited/{row['family']}" if row["arm"] == "synthetic" else row["arm"]
        yield {"panel": "recorded_rooms", "group": row["group"], "condition": "competing", "truth": row["truth"],
               "kind": "background" if kind == "background_only" else kind, "pcm_sha256": row["pcm_sha256"]}


def transfer(args):
    from scripts.audio_short_edit_ssl_transfer import transfer_inputs

    plan = SOURCE.read(args.inputs / "freeze.json")
    validate(plan)
    lock = SOURCE.read(args.fit / "fit.json")
    verify_fit(plan, lock)
    original, _, _ = transfer_inputs(args.phoneme, args.recorded)
    primary_rooms = {row["capture"] for row in plan["groups"]}
    primary_audio = {digest for row in plan["groups"] for view in row["observations"].values() for digest in view.values()}
    cases = list(transfer_cases())
    args.output.mkdir(mode=0o700)
    waves = args.output / "pcm"
    waves.mkdir(mode=0o700)
    for row, case in zip(original, cases, strict=True):
        digest = case["pcm_sha256"]
        signal = (args.recorded / "pcm" / (row["id"] + ".pcm")).read_bytes()
        SOURCE.check(row["room"] not in primary_rooms and digest not in primary_audio and
                     len(signal) == SAMPLES * 2 and SOURCE.digest(signal) == digest == row["pcm_sha256"],
                     "Transfer participant capture or exact waveform overlaps the primary plan")
        destination = waves / f"{digest}.pcm"
        if not destination.exists():
            destination.write_bytes(signal)
    commitment = {"plan_sha256": plan_hash(plan), "fit_sha256": plan_hash(lock),
                  "transfer_sha256": SOURCE.path_sha(ROOM_TRANSFER)}
    SOURCE.write(args.output / "freeze.json", commitment)
    extract = feature_inputs(args.output, runtime(args.speech))
    trials = []
    for trial in lock["trials"]:
        model, mean, scale = load_trial(args.fit, trial)
        arm = trial["arm"]
        trials.append({"family": trial["family"], "arm": arm, "scores": score_rows(model, mean, scale, cases, arm, extract)})
    SOURCE.write(args.output / "transfer.json", {**commitment, "trials": trials})
    print(f"Read all six frozen models on {len(cases)} disjoint recorded-room deliveries", flush=True)


def current(args):
    import torch

    from app.detectors.audio import short_voice_review
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    plan = SOURCE.read(args.inputs / "freeze.json")
    validate(plan)
    voice = load_voice({"weights": args.weights, "assets": ASSETS})
    speech = runtime(args.weights / "silero_vad.jit")
    torch.set_num_threads(2)

    class Deadline:
        def check(self, _stage):
            pass  # Keep timing budgets separate from this offline comparison.

    rows = []
    for item in current_cases(plan):
        signal = base.pcm(args.inputs, item)
        reading = short_voice_review.scan(signal, speech.spans(signal), voice, Deadline(), codec="pcm_s16le")
        selected = reading["selected"]
        result = {"window": [selected["start_sample"], selected["end_sample"]] if selected else None,
                  "speech_counts": reading["speech_counts"], "responses": reading["responses"]}
        rows.append(item | result)
        if len(rows) % 24 == 0:
            print(f"Current scan measured {len(rows)} held competing-speech inputs", flush=True)
    SOURCE.write(args.output, {"plan_sha256": plan_hash(plan), "assets": ASSETS, "rows": rows})


def current_cases(plan):
    for group in plan["groups"]:
        if group["role"] == "held":
            for name in sorted(group["observations"]["competing"]):
                yield case_for(group, "competing", name)


def case_key(row):
    return row["panel"], row["group"], row["condition"], row["kind"]


def current_readings(plan, dumped):
    SOURCE.check(dumped["plan_sha256"] == plan_hash(plan), "App scan used another source plan")
    result = {}
    for reference, row in zip(current_cases(plan), dumped["rows"], strict=True):
        SOURCE.check(all(row[k] == v for k, v in reference.items()), "App scan measured another item")
        selected, _, assessed = two_voice.app_reading(row)
        result[case_key(row)] = selected, assessed
    return result


def summarize(cases, measured, selected):
    rows = defaultdict(Counter)
    controls = defaultdict(list)
    for case, location in zip(cases, selected, strict=True):
        if case["truth"] is None and case["kind"] != "background":
            controls[case_key(case)[:3]].append(location)
    useful = two_voice.SCORING._useful
    for case, item, location in zip(cases, measured, selected, strict=True):
        count = rows[case["panel"], case["condition"]]
        truth = case["truth"]
        if truth:
            count["edited"] += 1
            count["edited_assessed"] += item is not None
            accurate = useful(location, truth)
            count["useful"] += accurate
            count["wrong_time"] += location is not None and not accurate
            count["useful_without_matched_control"] += accurate and not any(
                useful(w, truth) for w in controls[case_key(case)[:3]])
        else:
            role = "background" if case["kind"] == "background" else "ordinary"
            count[role] += 1
            count[f"{role}_assessed"] += item is not None
            count[f"{role}_flags"] += location is not None
    return {"/".join(key): dict(count) for key, count in sorted(rows.items())}


def differences(cases, reference, candidate):
    result = defaultdict(Counter)
    useful = two_voice.SCORING._useful
    for case, old, new in zip(cases, reference, candidate, strict=True):
        slot = result[case["panel"], case["condition"]]
        if case["truth"] is not None:
            before, after = (useful(item, case["truth"]) for item in (old, new))
            slot["additional_useful"] += after and not before
            slot["lost_useful"] += before and not after
        else:
            role = "background" if case["kind"] == "background" else "ordinary"
            slot[f"additional_{role}_flags"] += old is None and new is not None
            slot[f"cleared_{role}_flags"] += old is not None and new is None
    return {"/".join(key): dict(value) for key, value in sorted(result.items())}


def report(payload):
    plan, fit, held = (payload[key] for key in ("plan", "fit", "held"))
    validate(plan)
    verify_fit(plan, fit)
    SOURCE.check(held["plan_sha256"] == plan_hash(plan) and held["fit_sha256"] == plan_hash(fit),
                 "Held phase did not use the sealed fit")
    app = current_readings(plan, payload["current"])
    decisions = {}
    aggregate = {"results": {}, "versus_dry_fit": {}, "versus_current": {}, "current": {}, "current_with_fallback": {}}
    for trial, measured in zip(fit["trials"], held["trials"], strict=True):
        family, arm = trial["family"], trial["arm"]
        SOURCE.check((family, arm) == (measured["family"], measured["arm"]), "Held trial row metadata differs")
        cases = list(trial_cases(plan, "held", family))
        acoustic = [item["best"] for item in measured["scores"]]
        competing = [i for i, case in enumerate(cases) if case["condition"] == "competing"]
        competing_cases = [cases[i] for i in competing]
        previous = [app[case_key(case)] for case in competing_cases]
        if family not in aggregate["current"]:
            aggregate["current"][family] = summarize(competing_cases, [True if v[1] else None for v in previous],
                                                     [v[0] for v in previous])
        for level, point in trial["points"].items():
            choices = base.verdicts(cases, measured["scores"], point)
            label = f"{family}/{arm}/{level}"
            decisions[family, arm, level] = choices
            aggregate["results"][label] = summarize(cases, acoustic, choices)
            if arm != "dry_only":
                aggregate["versus_dry_fit"][label] = differences(cases, decisions[family, "dry_only", level], choices)
            current_intervals = [v[0] for v in previous]
            candidate_intervals = [choices[i] for i in competing]
            aggregate["versus_current"][label] = differences(competing_cases, current_intervals, candidate_intervals)
            aggregate["current_with_fallback"][label] = summarize(
                competing_cases, [True if old[1] or acoustic[i] is not None else None
                                  for i, old in zip(competing, previous, strict=True)],
                [old if old is not None else new for old, new in zip(current_intervals, candidate_intervals, strict=True)],
            )
    if "transfer" in payload:
        extra = payload["transfer"]
        SOURCE.check(extra["plan_sha256"] == plan_hash(plan) and extra["fit_sha256"] == plan_hash(fit) and
                     extra["transfer_sha256"] == SOURCE.path_sha(ROOM_TRANSFER), "Transfer used another model or input plan")
        cases = list(transfer_cases())
        old = [row["incumbent"] for row in SOURCE.read(ROOM_TRANSFER)["rows"]]
        incumbent = [row["range"] for row in old]
        aggregate["recorded_room_transfer"] = {"current": summarize(cases, [True if row["assessed"] else None for row in old],
                                                                    incumbent)}
        for trial, measurement in zip(fit["trials"], extra["trials"], strict=True):
            family, arm = trial["family"], trial["arm"]
            SOURCE.check((family, arm) == (measurement["family"], measurement["arm"]), "Transfer trial metadata differs")
            for setting, point in trial["points"].items():
                locations = base.verdicts(cases, measurement["scores"], point)
                aggregate["recorded_room_transfer"][f"{family}/{arm}/{setting}"] = summarize(
                    cases, [item["best"] for item in measurement["scores"]], locations)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("freeze")
    for field in ("output", "phoneme", "stock", "rooms"):
        setup.add_argument(field, type=Path)
    for mode, fields in (("fit", ("inputs", "output", "speech")),
                         ("held", ("inputs", "fit", "output", "speech")),
                         ("transfer", ("inputs", "fit", "output", "speech", "phoneme", "recorded")),
                         ("current", ("inputs", "output", "weights")),
                         ("combine", ("plan", "fit", "held", "current", "output")),
                         ("review", ("receipt",))):
        entry = commands.add_parser(mode)
        for field in fields:
            entry.add_argument(field, type=Path)
        if mode == "combine":
            entry.add_argument("--transfer", type=Path)
    args = parser.parse_args()
    if args.command == "combine":
        record = {key: SOURCE.read(getattr(args, key)) for key in ("plan", "fit", "held", "current")}
        if args.transfer is not None:
            record["transfer"] = SOURCE.read(args.transfer)
        record["result"] = report(record)
        SOURCE.write(args.output, record)
    elif args.command == "review":
        open_receipt = gzip.open if args.receipt.suffix == ".gz" else Path.open
        with open_receipt(args.receipt, "rt", encoding="utf-8") as stream:
            record = json.load(stream)
        counts = report(record)
        SOURCE.check(counts == record["result"], "Saved result disagrees with replay")
        print(f"Replayed {len(record['plan']['groups'])} matched source groups and {len(record['fit']['trials'])} frozen trials")
        for section, values in counts.items():
            for label, item in values.items():
                print(section, label, item)
    else:
        {"freeze": freeze, "fit": fitted, "held": confirm, "current": current, "transfer": transfer}[args.command](args)


if __name__ == "__main__":
    main()
