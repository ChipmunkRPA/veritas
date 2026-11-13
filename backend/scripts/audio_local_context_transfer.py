"""Test whether a short speech interval changes under restricted model context."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import wave
from pathlib import Path

import numpy as np

from scripts.audio_overlap_pitch_transfer import canonical, read, require, useful

ROOT = Path(__file__).resolve().parents[2]
RATE, SAMPLES, STRIDE = 16000, 96000, 320
SPAN, GUARD, STEP = 12800, 5120, 1280
CROP, CROP_STARTS = 32000, tuple(range(0, 64001, 12800))
ENCODER = ROOT / "models/research/wavlm-v13"
VAD = ROOT / "models/weights/audio-upgrade/silero_vad.jit"
VAD_HASH = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
SCHEMA = "chipmunk.audio-local-context-transfer/1"
METHOD = {
    "inputs": "One six-second, 16 kHz mono PCM upload. Compare a full WavLM Base Plus forward with six 2-second crops of the same original. Use shared convolution-frame centers 200 + 320i. Ignore crop frames within 320 ms of an edge and retain the smallest disagreement when two crops cover a frame. No clean copy, stems, synthesis metadata, noise level or truth enters inference.",
    "measure": "At each shared frame, subtract layer 4 cosine distance from layer 12 cosine distance between full and restricted context. The score for a proposed 0.80-second interval on the 80 ms grid is median signed difference inside minus the larger of two flank medians. Each 320 ms flank excludes the nearest 80 ms. Require 60 percent of interval and flank centers to have both eligible context and speech support. The strongest supported interval wins with earlier time resolving ties.",
    "point": "Strict maximum on the 20 ordinary or isolated-background development clips at 0 and -5 dB. Each generator family in held comes from a previously scored source bank. ElevenLabs and Kokoro are absent from development. Preserve the fixed prior incumbent point and revalidate the exact two-second native input hashes.",
    "gate": "At least two net additional useful held synthetic locations across at least two families, with at most one additional ordinary source group and no isolated-background false flag. Useful range covers at least 50 percent of the edit and meets 0.30 IoU. Report fixed and descriptive same-number-of-false-group counts, ordinary/synthetic availability and exact source ranges.",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pcm(path):
    if path.suffix == ".wav":
        with wave.open(str(path), "rb") as audio:
            require(audio.getparams()[:4] == (1, 2, RATE, SAMPLES), "Native WAV format differs")
            raw = audio.readframes(SAMPLES)
    else:
        raw = path.read_bytes()
    require(len(raw) == SAMPLES * 2, "Native PCM extent differs")
    return raw


def bindings(args):
    intake = read(args.intake)
    reference, roster, baseline = (read(args.held / name) for name in ("freeze.json", "roster.json", "current.json"))
    published = read(args.prior)
    require(published["commitments"]["freeze"] == file_sha(args.held / "freeze.json")
            and published["commitments"]["roster"] == file_sha(args.held / "roster.json")
            and published["commitments"]["current"] == file_sha(args.held / "current.json")
            and reference["roster_sha256"] == file_sha(args.held / "roster.json")
            and baseline["freeze"] == file_sha(args.held / "freeze.json"), "Frozen replication inputs differ")
    rows, paths = [], {}
    for item in intake["deliveries"]:
        if item["role"] != "development" or item["level"] not in (0, -5):
            continue
        name = f"{item['case']}_{item['level']}_{item['kind']}"
        path = args.development_wavs / (name + ".wav")
        require(sha(pcm(path)) == item["pcm_sha256"], f"Development PCM differs: {name}")
        rows.append({"id": name, "group": item["case"], "role": "development", "arm": item["kind"],
                     "family": "chatterbox", "truth": item["source_edit_range"], "pcm_sha256": item["pcm_sha256"]})
        paths[name] = path
    require(len(rows) == 28, "Incomplete development bank")
    for item, current in zip(roster, baseline["records"], strict=True):
        name = item["id"]
        path = args.held / "pcm" / (name + ".pcm")
        raw = pcm(path)
        require(current["id"] == name and current["pcm_sha256"] == item["pcm_sha256"] == sha(raw), "Held PCM differs")
        for entry in current["windows"]:
            require(sha(raw[2 * entry["start"]:2 * entry["end"]]) == entry["pcm_sha256"], "Incumbent native window differs")
        measured = [item for item in current["windows"] if item["score"] is not None]
        best = max(measured, key=lambda item: item["score"], default=None)
        rows.append({"id": name, "role": "held", "group": item["group"], "arm": item["arm"], "family": item["family"],
                     "truth": item["truth"], "pcm_sha256": item["pcm_sha256"],
                     "incumbent": None if best is None else [best["score"], [best["start"], best["end"]]]})
        paths[name] = path
    require(len(rows) == len(paths) == 80, "Unexpected number of unique deliveries")
    return rows, paths, reference["points"]["current"]


def normalize(states):
    states = states.astype(np.float64)
    return states / np.maximum(np.linalg.norm(states, axis=-1, keepdims=True), 1e-10)


def encode(model, values):
    import torch

    data = torch.from_numpy(values)
    states = model(data, attention_mask=torch.ones_like(data, dtype=torch.long), output_hidden_states=True, return_dict=True).hidden_states
    require(len(states) == 13, "Expected exact encoder state stack")
    return tuple(normalize(states[layer].numpy()) for layer in (4, 12))


def drift(model, samples):
    import torch

    with torch.inference_mode():
        full = encode(model, samples[None, :])
        local = encode(model, np.stack([samples[start:start + CROP] for start in CROP_STARTS]))
    require(full[0].shape == full[1].shape == (1, 299, 768) and local[0].shape == local[1].shape == (6, 99, 768),
            "Full and local encoder clocks differ")
    result = np.full(299, np.inf)
    local_centers = 200 + STRIDE * np.arange(99)
    available = (local_centers >= GUARD) & (local_centers < CROP - GUARD)
    for position, start in enumerate(CROP_STARTS):
        centers = start // STRIDE + np.flatnonzero(available)
        early, late = (np.einsum("ij,ij->i", a[0, centers], b[position, available]) for a, b in zip(full, local, strict=True))
        result[centers] = np.minimum(result[centers], early - late)
    return result


def nominate(values, speech):
    centers = 200 + STRIDE * np.arange(len(values))
    admitted = np.isfinite(values) & speech
    best, examined = None, 0
    flank, gap = 5120, 1280
    for start in range(flank + gap, SAMPLES - SPAN - flank - gap + 1, STEP):
        stop = start + SPAN
        pieces = [(start - flank - gap, start - gap), (start, stop), (stop + gap, stop + gap + flank)]
        scores = []
        for a, b in pieces:
            selected = admitted & (centers >= a) & (centers < b)
            if selected.sum() < .60 * ((b - a) // STRIDE):
                break
            scores.append(float(np.median(values[selected])))
        if len(scores) != 3:
            continue
        examined += 1
        score = round(scores[1] - max(scores[0], scores[2]), 8)
        if best is None or score > best[0]:
            best = [score, [start, stop]]
    return best, int(admitted.sum()), examined


def measure(args, rows, paths):
    import torch

    from app.detectors.audio.upgraded import SpeechActivity
    from scripts import audio_wavlm_features_v13 as wavlm
    from scripts.audio_strong_traffic_view import network_free

    network_free()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    encoder = wavlm.load(args.encoder)
    activity = SpeechActivity(args.vad)
    centers = (200 + STRIDE * np.arange(299)) / RATE
    output = []
    for row in rows:
        raw = pcm(paths[row["id"]])
        require(sha(raw) == row["pcm_sha256"], "PCM changed after freeze")
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
        speech = np.zeros(299, dtype=bool)
        for start, stop in activity.spans(values):
            speech |= (centers >= start) & (centers < stop)
        best, admitted, examined = nominate(drift(encoder, values), speech)
        output.append(row | {"candidate": best, "admitted_frames": admitted, "examined_ranges": examined})
        if len(output) % 8 == 0:
            print(f"Measured {len(output)}/{len(rows)} exact original waveforms", flush=True)
    return output


def metrics(rows, arm, point):
    ordinary = set()
    found = {}
    output = {"localized": 0, "misplaced": 0, "unassessed": 0, "unassessed_synthetic": 0,
              "useful_before_point": 0, "false_files": 0, "isolated_background_flags": 0}
    for row in rows:
        result = row[arm]
        positive = row["arm"].startswith("synthetic")
        if result is None:
            output["unassessed"] += 1
            output["unassessed_synthetic"] += int(positive)
            continue
        overlap = positive and useful(result[1], row["truth"])
        output["useful_before_point"] += int(overlap)
        if result[0] <= point:
            continue
        if overlap:
            output["localized"] += 1
            found[row["family"]] = found.get(row["family"], 0) + 1
        elif positive:
            output["misplaced"] += 1
        else:
            output["false_files"] += 1
            output["isolated_background_flags"] += int(row["arm"] == "background_only")
            ordinary.add(row["group"])
    return output | {"false_groups": len(ordinary), "localized_families": found}


def report(held, points):
    output = {"fixed": {arm: metrics(held, arm, point) for arm, point in points.items()}, "equal_false_groups": {}}
    for arm in points:
        controls = {}
        for row in held:
            value = row[arm]
            if value is None or row["arm"].startswith("synthetic"):
                continue
            group = row["group"]
            controls[group] = max(controls.get(group, -float("inf")), value[0])
        ordered = sorted(controls.values(), reverse=True)
        output["equal_false_groups"][arm] = {str(n): metrics(held, arm, ordered[n]) for n in range(min(4, len(ordered)))}
    candidate, old = (output["fixed"][arm] for arm in ("candidate", "incumbent"))
    output["promotion_gate"] = (candidate["localized"] >= old["localized"] + 2
                                and len(candidate["localized_families"]) >= 2
                                and candidate["false_groups"] <= old["false_groups"] + 1
                                and not candidate["isolated_background_flags"])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "score", "replay"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--development-wavs", type=Path, required=True)
    parser.add_argument("--held", type=Path, default=ROOT / "backend/data/independent-evaluation/audio-remote-neighbor-replication-2026-11-12")
    panel = ROOT / "docs/evaluation-panels"
    parser.add_argument("--intake", type=Path, default=panel / "audio-phoneme-short-edit-intake-2026-11-13.json")
    parser.add_argument("--prior", type=Path, default=panel / "audio-remote-neighbor-replication-2026-11-12.json")
    parser.add_argument("--encoder", type=Path, default=ENCODER)
    parser.add_argument("--vad", type=Path, default=VAD)
    args = parser.parse_args()
    from scripts import audio_wavlm_features_v13 as wavlm

    rows, paths, point = bindings(args)
    vad_hash = file_sha(args.vad)
    require(vad_hash == VAD_HASH, "Speech activity asset differs")
    lock = {"schema": SCHEMA, "method": METHOD, "roster_sha256": sha(canonical(rows)), "runner_sha256": file_sha(Path(__file__)),
            "preceding_receipts": {"intake": file_sha(args.intake), "held": file_sha(args.prior)},
            "assets": wavlm.assets(args.encoder), "speech_activity_sha256": vad_hash,
            "runtime": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "transformers")},
            "incumbent_point": point}
    frozen, fitted, finished = (args.run / name for name in ("freeze.json", "fit.json", "result.json"))
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        frozen.write_bytes(canonical(lock))
        print("Locked 28 development and 52 family-transfer recordings before inference")
        return
    require(frozen.read_bytes() == canonical(lock), "Frozen experiment changed")
    if args.phase == "fit":
        require(not fitted.exists() and not finished.exists(), "Development was already scored")
        values = measure(args, [r for r in rows if r["role"] == "development"], paths)
        controls = [r["candidate"][0] for r in values if r["candidate"] is not None and not r["arm"].startswith("synthetic")]
        require(controls, "No admitted development controls")
        fitted.write_bytes(canonical({"freeze_sha256": file_sha(frozen), "point": max(controls),
                                      "control_count": len(controls), "rows": values}))
        print(f"Sealed {max(controls)} from {len(controls)} ordinary controls before held inference", flush=True)
        return
    require(fitted.exists(), "Development point is not sealed")
    fit = read(fitted)
    require(fit["freeze_sha256"] == file_sha(frozen), "Point uses a different freeze")
    selected = [r for r in rows if r["role"] == "held"]
    if args.phase == "replay":
        prior = read(finished)
        require(prior["freeze"] == lock and prior["fit_sha256"] == file_sha(fitted), "Result receipt inputs differ")
        held = prior["held"]
        require([{k: r[k] for k in original} for r, original in zip(held, selected, strict=True)] == selected,
                "Result native source bindings differ")
    else:
        require(not finished.exists(), "Held sources already scored")
        held = measure(args, selected, paths)
    points = {"candidate": fit["point"], "incumbent": point}
    result = {"freeze": lock, "fit_sha256": file_sha(fitted), "points": points, "development": fit["rows"],
              "held": held, "summary": report(held, points)}
    if args.phase == "score":
        finished.write_bytes(canonical(result))
    else:
        require(finished.read_bytes() == canonical(result), "Result decisions differ")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
