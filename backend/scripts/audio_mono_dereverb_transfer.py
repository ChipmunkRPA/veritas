"""Compare delayed-prediction views of single-file voice screening under room speech."""
from __future__ import annotations

import argparse
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import istft, stft

from scripts import audio_harmonic_overlap_transfer as harmonic
from scripts import audio_stereo_agreement as agreement

RATE, SAMPLES = harmonic.harmonic.RATE, harmonic.harmonic.SAMPLES
sha, read, write = harmonic.sha, harmonic.read, harmonic.write
INCUMBENT = harmonic.voice.incumbent
HOP, FFT, DELAY, TAPS, ITERATIONS, BLEND = 128, 512, 6, 10, 3, .8
METHOD = {"fft": FFT, "hop": HOP, "delay": DELAY, "taps": TAPS, "iterations": ITERATIONS,
          "subtract_fraction": BLEND, "active_frequencies_hz": [150, 6500],
          "variance_smoothing_frames": 9, "relative_variance_floor": .01, "diagonal_loading": .001,
          "target_foreground_snr_db": -5, "full_profile_window_samples": SAMPLES,
          "suppression": "An AI result becomes inconclusive only if an admitted conditioned view loses AI "
                         "and its raw log odds decrease by more than ln(2).",
          "rescue": "An inconclusive result becomes AI only if an admitted conditioned view leans AI.",
          "transfer": "Each fixed rule must gain in both transfer rooms on at least three foreground groups, "
                      "change no clean or native cases, and preserve the transfer's other class. "
                      "Suppression allows at most two lost synthetic flags and requires at least three total "
                      "genuine false flags on this panel. Rescue allows at most one added false flag and "
                      "requires gains in at least two synthetic voice families. No apparent success with "
                      "fewer than six assessed genuine mixtures per room and mode."}


def decode(raw):
    if len(raw) != 2 * SAMPLES:
        raise ValueError("Three seconds of exact signed mono PCM are required")
    return np.frombuffer(raw, "<i2").astype(np.float64)


def spectra(pcm):
    return stft(pcm, fs=RATE, nperseg=FFT, noverlap=FFT - HOP, boundary="zeros", padded=True)[2]


def predictors(spec):
    offset = DELAY + TAPS - 1
    design = np.stack([spec[:, offset - DELAY - lag:spec.shape[1] - DELAY - lag]
                       for lag in range(TAPS)], axis=-1)
    return offset, design


def fit(pcm):
    """Fit a fixed single-channel complex delayed linear predictor to mixture alone."""
    spec = spectra(pcm)
    offset, design = predictors(spec)
    target = spec[:, offset:]
    power_floor = np.maximum(.01 * np.mean(abs(spec)**2, axis=1, keepdims=True), 1e-10)
    residual = target
    for _ in range(ITERATIONS):
        inverse = 1 / np.maximum(uniform_filter1d(abs(residual)**2, 9, axis=1, mode="nearest"), power_floor)
        cov = np.einsum("ftp,ft,ftq->fpq", design.conj(), inverse, design)
        rhs = np.einsum("ftp,ft,ft->fp", design.conj(), inverse, target)
        load = .001 * cov.trace(axis1=1, axis2=2).real / TAPS + 1e-10
        weights = np.linalg.solve(cov + load[:, None, None] * np.eye(TAPS), rhs[..., None])[..., 0]
        residual = target - np.einsum("ftp,fp->ft", design, weights)
    bands = np.fft.rfftfreq(FFT, 1 / RATE)
    weights[(bands < 150) | (bands > 6500)] = 0
    if not np.isfinite(weights).all():
        raise ValueError("Delayed prediction coefficients must be finite")
    return weights


def apply(pcm, weights):
    spec = spectra(pcm)
    offset, design = predictors(spec)
    spec[:, offset:] -= BLEND * np.einsum("ftp,fp->ft", design, weights)
    waveform = istft(spec, fs=RATE, nperseg=FFT, noverlap=FFT - HOP, boundary=True)[1][:SAMPLES]
    if waveform.shape != (SAMPLES,) or not np.isfinite(waveform).all():
        raise ValueError("Conditioned samples differ from the exact mono clock")
    return waveform


def mix(foreground, backgrounds):
    voice = decode(foreground)
    scaled = [noise * np.linalg.norm(voice) / np.linalg.norm(noise) * 10**.25 for noise in backgrounds]
    peak = max(float(np.max(abs(voice) + abs(noise))) for noise in scaled)
    gain = min(1., 31_129 / peak)
    clean = np.rint(voice * gain).astype("<i2").tobytes()
    results = []
    for noise in scaled:
        reference = np.rint(noise * gain)
        measured = 20 * math.log10(np.linalg.norm(decode(clean)) / np.linalg.norm(reference))
        if abs(measured + 5) > .02:
            raise ValueError("Fresh mixture differs from its planned −5 dB")
        results.append(((decode(clean) + reference).astype("<i2").tobytes(), measured))
    return clean, results


def freeze(folder, archived, harmonic_run, ami, annotations):
    _, archive_rows, archive_gates, archive_scores, archive_parents = agreement.inputs(archived)
    held_result = harmonic.voice.PANEL / "audio-harmonic-overlap-2026-11-11.json"
    held = read(held_result)
    lock, held_rows = harmonic.frozen(harmonic_run)
    if (sha(annotations) != lock["annotations_sha256"] or
            sha(harmonic_run / "freeze.json") != held["freeze_sha256"]):
        raise ValueError("Expected exact published parent room and annotation binding")
    voices, voice_parent = harmonic.foregrounds()
    chosen = {}
    for raw, row in voices:
        chosen.setdefault(row["voice"], (raw, row))
    if len(chosen) != 16 or {row["group_id"] for _, row in chosen.values()} & {row["group"] for row in archive_rows}:
        raise ValueError("The transfer roster must contain 16 disjoint declared voices")
    (folder / "pcm").mkdir(parents=True)
    rows, sources, room_cuts = [], {}, []

    def save(raw, **metadata):
        index = len(rows)
        path = folder / "pcm" / f"{index:03}.pcm"
        path.write_bytes(raw)
        rows.append({"id": index, "sha256": sha(raw), **metadata})
        return index

    archived_id = {}
    for row, gate, baseline in zip(archive_rows, archive_gates, archive_scores, strict=True):
        agreement.paired.pcm_views(archived, row)
        raw = (archived / "pcm" / f"{row['id']:03}.pcm").read_bytes()
        observed = baseline.get("mono")
        if bool(observed) != gate["admitted"]["mono"]:
            raise ValueError("Archived speech gate and incumbent response differ")
        archived_id[row["id"]] = save(raw, role="retrospective", room=row.get("room"), mode=row.get("mode"),
            treatment=row["treatment"], family=row["family"], group=row["group"], baseline=observed,
            clean_id=archived_id.get(row.get("clean_id")))
    with ZipFile(annotations) as corpus:
        for room, info in lock["rooms"].items():
            if info["role"] != "confirmation":
                continue
            mic = ami / f"{room}.Array1-01.wav"
            digest = sha(mic)
            if digest != info["archive_sha256"] or room in {row["room"] for row in rows}:
                raise ValueError("Fresh room must be source-separated from retrospective rooms")
            sources[room] = {"sha256": digest, "url": info["official_url"]}
            old_cuts = [row["source_start_sample"] for row in held_rows if row.get("room") == room and "source_start_sample" in row]
            available = harmonic.windows(corpus, room, old_cuts)
            occupancy = harmonic.room_activity(corpus, room)
            for mode, pick in (("sequential", "minimal"), ("simultaneous", "high")):
                begin = available[pick][0]
                raw, metadata = harmonic.room_cut(mic, begin, occupancy)
                overlap = sum(end - start for start, end in metadata["annotated_overlap"])
                native_id = save(raw, role="transfer", room=room, mode=mode, treatment="native",
                                 family="genuine", group=room, baseline=None, source=metadata)
                room_cuts.append((room, mode, native_id, decode(raw), overlap))
    if len(room_cuts) != 4:
        raise ValueError("Two new sampled cuts from each of two natural room recordings are required")
    for raw, row in chosen.values():
        clean, mixed = mix(raw, [cut[3] for cut in room_cuts])
        base = {"role": "transfer", "family": row["lane"], "group": row["group_id"], "baseline": None}
        clean_id = save(clean, room=None, mode=None, treatment="clean", **base)
        for (room, mode, native_id, _, overlap), (pcm, snr) in zip(room_cuts, mixed, strict=True):
            save(pcm, room=room, mode=mode, treatment="mixed", clean_id=clean_id, native_id=native_id,
                 snr_db=snr, overlap_samples=overlap, **base)
    write(folder / "roster.json", rows)
    record = {"schema": "chipmunk.audio-mono-dereverb/1", "method": METHOD, "rooms": sources,
              "roster_sha256": sha(folder / "roster.json"), "code_sha256": sha(Path(__file__)),
              "incumbent_assets": INCUMBENT.ASSETS, "calibration_sha256": sha(INCUMBENT.CALIBRATION),
              "archive": archive_parents, "annotations_sha256": sha(annotations),
              "earlier_room_cut_freeze_sha256": sha(harmonic_run / "freeze.json"),
              "transfer_source_manifest_sha256": voice_parent,
              "planned": {role: dict(Counter(row["treatment"] for row in rows if row["role"] == role))
                          for role in ("retrospective", "transfer")}}
    write(folder / "freeze.json", record)
    print(record["planned"], flush=True)


def frozen(folder):
    lock, rows = (read(folder / name) for name in ("freeze.json", "roster.json"))
    if (lock["code_sha256"] != sha(Path(__file__)) or lock["roster_sha256"] != sha(folder / "roster.json")
            or lock["method"] != METHOD or lock["incumbent_assets"] != INCUMBENT.ASSETS
            or lock["calibration_sha256"] != sha(INCUMBENT.CALIBRATION)):
        raise ValueError("The source, method or serving calibration changed after freeze")
    return lock, rows


def exact(folder, row):
    raw = (folder / "pcm" / f"{row['id']:03}.pcm").read_bytes()
    if sha(raw) != row["sha256"]:
        raise ValueError("An input waveform changed")
    return decode(raw)


def component_audit(pcm, output, clean, weights):
    noise = pcm - clean
    a, b = apply(clean, weights), apply(noise, weights)
    error = float(max(abs(a + b - output)))
    if error > 1e-7:
        raise ValueError("The known components do not reconstruct the conditioned waveform")
    return {"snr_gain_db": 20 * math.log10(np.linalg.norm(a) / np.linalg.norm(b)
                                          * np.linalg.norm(noise) / np.linalg.norm(clean)),
            "voice_level_change_db": 20 * math.log10(np.linalg.norm(a) / np.linalg.norm(clean)),
            "recombination_max_pcm_error": error}


def score(folder):
    from app.detectors.audio import calibration
    from app.detectors.audio.upgraded import SpeechActivity
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from app.engine.boundary import Deadline
    from scripts.run_audio_specialists_v15 import load_voice

    lock, rows = frozen(folder)
    INCUMBENT.network_free()
    voice = load_voice({"weights": str(INCUMBENT.WEIGHTS), "assets": lock["incumbent_assets"]})
    vad = SpeechActivity(INCUMBENT.WEIGHTS / "silero_vad.jit")
    vad.model = activity_counter = agreement.paired.stereo.gate.CountedVAD(vad.model)
    bundle = calibration.load()
    detector = PellaVoiceDetector()
    count = Counter()

    def track_forward(*_):
        count["classifier"] += 1

    hook = voice.register_forward_pre_hook(track_forward)

    def measure(values, *, baseline):
        waveform = (values / 32768).astype(np.float32)
        spans = vad.spans(waveform)
        cached = SimpleNamespace(obj=SimpleNamespace(spans=lambda _: spans))
        ctx = SimpleNamespace(profile="full", lease=SimpleNamespace(get=lambda _: cached),
                              outputs=SimpleNamespace(audio=AudioTrack("analysed", waveform, SAMPLES)))
        windows = detector._windows(ctx)
        if not windows:
            return {"decision": "unassessed", "raw": None, "probe_status": "not_admitted"}
        if len(windows) != 1 or windows[0][0] != 0 or not np.array_equal(waveform, windows[0][1]):
            raise ValueError("Serving window changed the exact waveform")
        raw, = voice.predict([waveform])
        if baseline:
            evaluation = calibration.evaluate(raw, waveform, windows, voice, Deadline(120))
            return {**{k: evaluation[k] for k in ("decision", "probe_status", "stability_log_odds_change")}, "raw": raw}
        decision = calibration.initial_decision(calibration.probability(raw, bundle), bundle)
        return {"decision": decision, "raw": raw, "probe_status": "not_required_for_comparison"}
    start, observed = time.monotonic(), []
    try:
        for row in rows:
            pcm = exact(folder, row)
            if row["role"] == "retrospective":
                incumbent = row["baseline"] or {"decision": "unassessed", "raw": None, "probe_status": "not_admitted"}
            else:
                incumbent = measure(pcm, baseline=True)
            weights = fit(pcm)
            output = apply(pcm, weights)
            quantized = np.rint(output)
            detail = {"id": row["id"], "source_sha256": row["sha256"], "incumbent": incumbent}
            if row["treatment"] == "mixed":
                detail["acoustic"] = component_audit(pcm, output, exact(folder, rows[row["clean_id"]]), weights)
            if incumbent["decision"] not in {"ai", "inconclusive"}:
                detail["candidate"] = {"decision": "unassessed", "raw": None, "probe_status": "incumbent_outside_comparison"}
            elif quantized.min() < -32768 or quantized.max() > 32767:
                detail["candidate"] = {"decision": "unassessed", "raw": None, "probe_status": "clipped"}
            else:
                detail["output_sha256"] = sha(quantized.astype("<i2").tobytes())
                detail["candidate"] = measure(quantized, baseline=False)
            observed.append(detail)
            if len(observed) % 24 == 0:
                print(f"Scored {len(observed)}/{len(rows)} ({count['classifier']} actual classifier forwards)", flush=True)
    finally:
        hook.remove()
    write(folder / "scores.json", {"freeze_sha256": sha(folder / "freeze.json"), "rows": observed,
          "classifier_forwards": count["classifier"], "speech_activity_forwards": activity_counter.calls,
          "elapsed_seconds": time.monotonic() - start, "peak_rss_bytes_darwin": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def summarize(folder):
    from app.detectors.audio import calibration

    lock, rows = frozen(folder)
    scores = read(folder / "scores.json")
    if scores["freeze_sha256"] != sha(folder / "freeze.json"):
        raise ValueError("Scores are not bound to the frozen roster")
    cells, effects, groups, rooms, families, snr, voice_level = (defaultdict(Counter), defaultdict(Counter),
        defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(list), defaultdict(list))
    for row, measured in zip(rows, scores["rows"], strict=True):
        if (row["id"], row["sha256"]) != (measured["id"], measured["source_sha256"]):
            raise ValueError("A score is bound to a different recording")
        role, family, room, treatment, mode = (row[key] for key in ("role", "family", "room", "treatment", "mode"))
        label = "genuine" if family == "genuine" else "synthetic"
        key = "/".join(str(part) for part in (role, room or "-", treatment, mode or "-", label))
        a, b = (measured[key] for key in ("incumbent", "candidate"))
        suppressed = a["decision"] == "ai" and b["decision"] not in {"ai", "unassessed"} and (
            calibration.logit(a["raw"]) - calibration.logit(b["raw"]) > math.log(2))
        rescued = a["decision"] == "inconclusive" and b["decision"] == "ai"
        tally = cells[key]
        tally.update(planned=1, **{f"incumbent_{a['decision']}": 1, f"conditioned_{b['decision']}": 1})
        tally.update(suppressed=int(suppressed), rescued=int(rescued))
        for change, observed in (("suppression", suppressed), ("rescue", rescued)):
            if observed:
                effects[role, change].update({f"{treatment}_{label}": 1})
                if treatment == "mixed":
                    groups[role, change, label].add(row["group"])
                    rooms[role, change, label].add(room)
                    families[role, change, label].add(family)
        if "acoustic" in measured:
            snr[key].append(measured["acoustic"]["snr_gain_db"])
            voice_level[key].append(measured["acoustic"]["voice_level_change_db"])
    coverage = all(cells[key]["planned"] - cells[key]["incumbent_unassessed"] >= 6
                   for key in cells if key.startswith("transfer/") and "/mixed/" in key and key.endswith("/genuine"))
    s, r = (effects["transfer", name] for name in ("suppression", "rescue"))
    def common(name, tally, target):
        return (coverage and len(rooms["transfer", name, target]) == 2
                and not any(value for key, value in tally.items() if not key.startswith("mixed_")))
    advances = {
        "suppression": common("suppression", s, "genuine") and len(groups["transfer", "suppression", "genuine"]) >= 3
                       and s["mixed_synthetic"] <= 2,
        "rescue": common("rescue", r, "synthetic") and len(groups["transfer", "rescue", "synthetic"]) >= 3
                  and len(families["transfer", "rescue", "synthetic"]) >= 2 and r["mixed_genuine"] <= 1,
    }
    result = {"freeze": lock, "freeze_sha256": sha(folder / "freeze.json"), "scores_sha256": sha(folder / "scores.json"),
              "cells": cells, "effects": {"/".join(key): value for key, value in effects.items()},
              "affected_groups": {"/".join(key): len(value) for key, value in groups.items()},
              "median_snr_gain_db": {key: median(values) for key, values in snr.items()},
              "median_voice_level_change_db": {key: median(values) for key, values in voice_level.items()},
              "transfer_genuine_coverage_met": coverage, "advance": advances,
              **{key: scores[key] for key in ("classifier_forwards", "speech_activity_forwards", "elapsed_seconds", "peak_rss_bytes_darwin")}}
    write(folder / "aggregate.json", result)
    print({"advance": advances, "effects": result["effects"], "genuine_coverage": coverage}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "score", "readout"))
    parser.add_argument("folder", type=Path)
    for name in ("archived", "harmonic_run", "ami", "annotations"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.folder, args.archived, args.harmonic_run, args.ami, args.annotations)
    elif args.action == "score":
        score(args.folder)
    else:
        summarize(args.folder)


if __name__ == "__main__":
    main()
