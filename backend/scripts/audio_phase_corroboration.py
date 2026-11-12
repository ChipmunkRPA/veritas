"""Evaluate a lightweight phase-domain veto on disjoint recorded-room groups."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal.windows import hann

from app.detectors.audio import calibration
from app.detectors.audio.decode import SAMPLE_RATE, decode_audio, wav_bytes
from scripts import audio_mono_dereverb_transfer as incumbent
from scripts.audio_mono_dereverb_transfer import read, sha, write

SCHEMA = "chipmunk.audio-phase-corroboration/1"
BANDS = ((180, 600), (600, 1250), (1250, 2500), (2500, 4500))
FFT, HOP, COUNT = 512, 128, 3 * SAMPLE_RATE
WINDOW = hann(FFT, sym=False)
FREQUENCIES = np.fft.rfftfreq(FFT, 1 / SAMPLE_RATE)
LEFT, RIGHT = (a.ravel() for a in np.meshgrid(np.arange(4, 64, 4), np.arange(5, 70, 6)))
SUM = LEFT + RIGHT
DELIVERY = {
    "opus": ["-ar", "48000", "-c:a", "libopus", "-b:a", "24k", "-vbr", "off"],
    "restored": ["-af", "afftdn=nr=9:nf=-42,acompressor=threshold=0.125:ratio=3:attack=15:release=120",
                 "-c:a", "pcm_s16le"],
}
METHOD = {
    "frame": [FFT, HOP], "bands_hz": BANDS, "active_energy_percentile": 40,
    "phase": "Magnitude-weighted phase curvature coherence and normalized cross-bin bicoherence.",
    "control": "Magnitude flatness and robust spectral centroid on the same active frames and frequency bands.",
    "pooling": "Median and 90th percentile for bicoherence; median and interquartile range for other measurements.",
    "split": "Salted ordering within each source family, with half the genuine groups and two synthetic groups per family in fit.",
    "salt": "chipmunk-audio-phase-20261111-v1",
    "ridge": 0.25,
    "fit": "Centered archived scenes and clean clips. Balance both origin classes and groups within each class.",
    "point": "Largest observed held-selection score preserving all clean synthetic AI flags and at least 90% of mixed synthetic AI flags.",
    "policy": "Withhold existing AI direction when the independent response falls strictly below the fixed point.",
    "advance": "On held centered scenes clear at least four genuine flags in at least three groups across both rooms, "
               "lose at most two synthetic flags, change no clean decision. Offset and each delivery may lose at most "
               "one true flag each and must leave no more than the baseline false flags.",
    "delivery": DELIVERY,
}


def integer(samples):
    if samples.shape != (COUNT,) or not np.isfinite(samples).all():
        raise ValueError("Invalid three-second samples")
    values = np.rint(samples * 32768)
    return np.clip(values, -32768, 32767).astype("<i2").tobytes()


def sample_bytes(parent, row):
    data = (parent / "mono" / f"{row['id']:03}.pcm").read_bytes()
    if len(data) != COUNT * 2 or sha(data) != row["mono_sha256"]:
        raise ValueError("Input does not match its frozen mono digest")
    return data


def features(data):
    signal = np.frombuffer(data, "<i2").astype(float) / 32768
    frames = np.lib.stride_tricks.sliding_window_view(signal, FFT)[::HOP] * WINDOW
    energy = np.sum(frames * frames, axis=1)
    active = energy > max(float(np.percentile(energy, 40)), 1e-10)
    if active.sum() < 12:
        return None
    spec = np.fft.rfft(frames[active])
    amps = np.abs(spec)
    unit = spec / np.maximum(amps, 1e-12)
    curvature = unit[:, :-2] * unit[:, 2:] * np.conj(unit[:, 1:-1])**2
    weight = np.sqrt(amps[:, :-2] * amps[:, 2:]) * amps[:, 1:-1]
    pair = spec[:, LEFT] * spec[:, RIGHT]
    cross = np.abs(np.sum(pair * np.conj(spec[:, SUM]), axis=0))
    bicoherence = cross / np.sqrt(np.maximum(np.sum(np.abs(pair)**2, axis=0)
                                          * np.sum(amps[:, SUM]**2, axis=0), 1e-24))
    phase, magnitude = [], []
    for low, high in BANDS:
        freq = (low <= FREQUENCIES) & (high > FREQUENCIES)
        inner = freq[1:-1]
        triples = (FREQUENCIES[SUM] >= low) & (FREQUENCIES[SUM] < high)
        local = np.abs(np.sum(curvature[:, inner] * weight[:, inner], axis=1)) / np.maximum(
            np.sum(weight[:, inner], axis=1), 1e-12)
        phase.extend((np.median(local), np.subtract(*np.percentile(local, [75, 25])),
                      np.median(bicoherence[triples]), np.percentile(bicoherence[triples], 90)))
        power = np.maximum(amps[:, freq]**2, 1e-15)
        flatness = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)
        centroid = np.sum(power * (FREQUENCIES[freq] - low) / (high - low), axis=1) / np.sum(power, axis=1)
        for values in (flatness, centroid):
            magnitude.extend((np.median(values), np.subtract(*np.percentile(values, [75, 25]))))
    return {"phase": list(map(float, phase)), "magnitude": list(map(float, magnitude))}


def parent_receipts(parent):
    frozen, roster, scores = (read(parent / name) for name in ("freeze.json", "roster.json", "scores.json"))
    aggregate = read(parent / "aggregate.json")
    expected = {"freeze_sha256": sha(parent / "freeze.json"), "scores_sha256": sha(parent / "scores.json"),
                "roster_sha256": sha(parent / "roster.json"), "aggregate_sha256": sha(parent / "aggregate.json")}
    if (expected["freeze_sha256"] != scores["freeze_sha256"] or expected["freeze_sha256"] != aggregate["freeze_sha256"]
            or expected["scores_sha256"] != aggregate["scores_sha256"] or expected["roster_sha256"] != frozen["roster_sha256"]
            or frozen["calibration_sha256"] != calibration.SHA256):
        raise ValueError("The source panel, score receipts or serving calibration changed")
    if any((row["id"], row["mono_sha256"]) != (score["id"], score["mono_sha256"])
           for row, score in zip(roster, scores["rows"], strict=True)):
        raise ValueError("Baseline decisions are not bound to the source PCM")
    return roster, scores["rows"], expected


def freeze(folder, parent):
    roster, _, bound = parent_receipts(parent)
    groups = defaultdict(list)
    for row in roster:
        sample_bytes(parent, row)
        if row["role"] == "archived" and row["placement"] == "clean":
            groups[row["family"]].append(row["group"])
    fit = {group for family, values in groups.items() for group in sorted(
        values, key=lambda g: hashlib.sha256((METHOD["salt"] + g).encode()).digest()
    )[:4 if family == "genuine" else 2]}
    selection = {g for values in groups.values() for g in values} - fit
    held = {row["group"] for row in roster if row["role"] == "transfer" and row["placement"] == "clean"}
    if (len(fit), len(selection), len(held)) != (10, 7, 16) or held & (fit | selection):
        raise ValueError("Expected three group-disjoint source roles")
    variants = []
    for row in roster:
        if row["role"] == "transfer" and (row["placement"] == "clean"
                or (row["placement"] == "center" and row["mode"] == "simultaneous")):
            variants.append(row["id"])
    if len(variants) != 48:
        raise ValueError("Expected three matched delivery variants per held foreground")
    folder.mkdir(parents=True)
    write(folder / "freeze.json", {"schema": SCHEMA, "method": METHOD, "parent": bound,
          "runner_sha256": sha(Path(__file__)), "fit": sorted(fit), "selection": sorted(selection),
          "held": sorted(held), "delivery_rows": variants})
    print(f"Frozen {len(roster)} original and {len(variants) * len(DELIVERY)} transformed views", flush=True)


def bound(folder, parent):
    frozen = read(folder / "freeze.json")
    roster, scores, binding = parent_receipts(parent)
    if (frozen["runner_sha256"] != sha(Path(__file__)) or frozen["parent"] != binding
            or frozen["method"] != json.loads(json.dumps(METHOD))):
        raise ValueError("The frozen protocol or exact source panel changed")
    return frozen, roster, scores


def fit_head(fit_rows, key):
    x = np.array([f[key] for _, f in fit_rows])
    y = np.array([row["family"] != "genuine" for row, _ in fit_rows])
    counts = Counter(row["group"] for row, _ in fit_rows)
    group_classes = Counter({row["group"]: row["family"] != "genuine" for row, _ in fit_rows}.values())
    w = np.array([.5 / (counts[row["group"]] * group_classes[label])
                  for (row, _), label in zip(fit_rows, y, strict=True)])
    mean = w @ x
    sd = np.maximum(np.sqrt(w @ (x - mean)**2), 1e-4)
    normalized = (x - mean) / sd
    coef = np.linalg.solve((normalized.T * w) @ normalized + np.eye(x.shape[1]) * METHOD["ridge"],
                           normalized.T @ (w * (y - .5)))
    return {"mean": mean.tolist(), "sd": sd.tolist(), "coef": coef.tolist()}


def response(head, features):
    return float(((np.asarray(features) - head["mean"]) / head["sd"]) @ head["coef"])


def train(folder, parent):
    lock, rows, scores = bound(folder, parent)
    fit_ids, select_ids = set(lock["fit"]), set(lock["selection"])
    inputs = [(row, features(sample_bytes(parent, row))) for row in rows
              if row["role"] == "archived" and row["group"] in fit_ids | select_ids]
    fitted = [(row, f) for row, f in inputs if row["group"] in fit_ids and f is not None]
    expected_groups = {row["group"] for row, _ in fitted}
    if expected_groups != fit_ids:
        raise ValueError("A fitting source group has no usable features")
    models = {}
    for key in ("phase", "magnitude"):
        head = fit_head(fitted, key)
        pos = defaultdict(list)
        for row, f in inputs:
            if (row["group"] in select_ids and row["family"] != "genuine" and
                    scores[row["id"]]["baseline"]["decision"] == "ai"):
                if f is None:
                    raise ValueError("A selection case has no feature response")
                pos[row["placement"]].append(response(head, f[key]))
        if len(pos["clean"]) != 3 or len(pos["center"]) < 5:
            raise ValueError("The point-selection slice is too small")
        cutoff = min(min(pos["clean"]), sorted(pos["center"])[len(pos["center"]) // 10])
        models[key] = {"head": head, "cutoff": cutoff, "selected_positive": dict(Counter(
            {key: len(values) for key, values in pos.items()}))}
    write(folder / "models.json", {"freeze_sha256": sha(folder / "freeze.json"), "models": models})
    print("Both candidate heads and points are frozen", flush=True)


def transformed(folder, original, row, name):
    source = folder / "work.wav"
    suffix = "opus" if name == "opus" else "wav"
    path = folder / "delivery" / f"{row['id']:03}-{name}.{suffix}"
    source.write_bytes(wav_bytes(np.frombuffer(original, "<i2").astype(np.float32) / 32768))
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source), *DELIVERY[name], str(path)],
                   check=True, timeout=30)
    digest = sha(path)
    decoded = decode_audio(path, source_sha256=digest, max_seconds=3)
    if len(decoded.pcm) != COUNT:
        raise ValueError(f"{name} changed the intended sample count")
    return integer(decoded.pcm), digest, decoded.mapping["timeline"]


def score(folder, parent):
    from app.detectors.audio.upgraded import SpeechActivity
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from app.engine.boundary import Deadline
    from scripts.run_audio_specialists_v15 import load_voice

    lock, rows, scores = bound(folder, parent)
    record = read(folder / "models.json")
    if record["freeze_sha256"] != sha(folder / "freeze.json"):
        raise ValueError("Heads differ from the frozen source split")
    incumbent.INCUMBENT.network_free()
    weights = incumbent.INCUMBENT.WEIGHTS
    voice = load_voice({"weights": str(weights), "assets": incumbent.INCUMBENT.ASSETS})
    gate = SpeechActivity(weights / "silero_vad.jit")
    detector = PellaVoiceDetector()
    forward = Counter()

    def count_forward(*_):
        forward["classifier"] += 1

    hook = voice.register_forward_pre_hook(count_forward)

    def baseline(data):
        pcm = np.frombuffer(data, "<i2").astype(np.float32) / 32768
        spans = gate.spans(pcm)
        cached = SimpleNamespace(obj=SimpleNamespace(spans=lambda _: spans))
        ctx = SimpleNamespace(profile="full", lease=SimpleNamespace(get=lambda _: cached),
                              outputs=SimpleNamespace(audio=AudioTrack("analysed", pcm, COUNT)))
        windows = detector._windows(ctx)
        if not windows:
            return {"decision": "unassessed", "raw": None}
        if len(windows) != 1 or windows[0][0] != 0 or not np.array_equal(windows[0][1], pcm):
            raise ValueError("The serving speech gate changed the measurement interval")
        raw, = voice.predict([pcm])
        return {"decision": calibration.evaluate(raw, pcm, windows, voice, Deadline(90))["decision"], "raw": raw}

    def observe(data, result, row, **extra):
        f = features(data)
        return {"id": row["id"], "sha256": sha(data), "baseline": result, "features": f,
                "response": {key: response(model["head"], f[key]) if f else None
                             for key, model in record["models"].items()}, **extra}

    observed = []
    (folder / "delivery").mkdir()
    try:
        for row, saved in zip(rows, scores, strict=True):
            raw = sample_bytes(parent, row)
            observed.append(observe(raw, saved["baseline"], row, view="parent"))
            if row["id"] in lock["delivery_rows"]:
                for name in DELIVERY:
                    output, digest, timeline = transformed(folder, raw, row, name)
                    observed.append(observe(output, baseline(output), row, view=name,
                                            delivery_sha256=digest, timeline=timeline))
                    if forward["classifier"] % 20 == 0:
                        print(f"{len(observed)} views, {forward['classifier']} classifier forwards", flush=True)
    finally:
        hook.remove()
        (folder / "work.wav").unlink(missing_ok=True)
    write(folder / "scores.json", {"models_sha256": sha(folder / "models.json"), "observed": observed,
                                  "classifier_forwards": forward["classifier"]})


def aggregate(folder, parent):
    lock, rows, _ = bound(folder, parent)
    models = read(folder / "models.json")
    scores = read(folder / "scores.json")
    if scores["models_sha256"] != sha(folder / "models.json") or models["freeze_sha256"] != sha(folder / "freeze.json"):
        raise ValueError("Scores and points were produced under a different lock")
    results = {}
    for key, model in models["models"].items():
        cells, true_groups, rooms = defaultdict(Counter), set(), set()
        for seen in scores["observed"]:
            row = rows[seen["id"]]
            role = ("fit" if row["group"] in lock["fit"] else "selection" if row["group"] in lock["selection"] else
                    "held" if row["role"] == "transfer" else "archived-native")
            genuine = row["family"] == "genuine"
            kind = "genuine" if genuine else "synthetic"
            view, treatment = seen["view"], row["placement"]
            base, value = seen["baseline"]["decision"], seen["response"][key]
            withdraw = base == "ai" and value is not None and value < model["cutoff"]
            cell = cells[f"{role}/{view}/{treatment}/{kind}"]
            cell.update(planned=1, assessed=int(value is not None), **{f"baseline_{base}": 1}, withdrawn=int(withdraw))
            if role == "held" and view == "parent" and treatment == "center" and genuine and withdraw:
                true_groups.add(row["group"])
                rooms.add(row["room"])
        clean_losses = sum(v["withdrawn"] for k, v in cells.items() if k.startswith("held/") and "/clean/" in k)
        delivered_losses = {name: cells[f"held/{name}/center/synthetic"]["withdrawn"] for name in DELIVERY}
        corrections = cells["held/parent/center/genuine"]["withdrawn"]
        losses = cells["held/parent/center/synthetic"]["withdrawn"]
        advance = (corrections >= 4 and len(true_groups) >= 3 and len(rooms) == 2 and losses <= 2 and clean_losses == 0
                   and all(loss <= 1 for loss in delivered_losses.values())
                   and cells["held/parent/offset/synthetic"]["withdrawn"] <= 1)
        results[key] = {"cells": cells, "corrected_groups": len(true_groups), "corrected_rooms": len(rooms),
                        "advance": advance, "selection": model["selected_positive"]}
    write(folder / "aggregate.json", {"schema": SCHEMA, "method": METHOD,
          "freeze_sha256": sha(folder / "freeze.json"), "models_sha256": sha(folder / "models.json"),
          "scores_sha256": sha(folder / "scores.json"), "views": len(scores["observed"]),
          "new_classifier_forwards": scores["classifier_forwards"], "arms": results})
    print({k: v["advance"] for k, v in results.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "train", "score", "aggregate"])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    {"freeze": freeze, "train": train, "score": score, "aggregate": aggregate}[args.stage](args.folder, args.parent)


if __name__ == "__main__":
    main()
