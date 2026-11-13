"""Compare a blind stereo directional mask with earlier voice evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft

from scripts import audio_blind_spatial_transfer as native
from scripts import audio_unassigned_stereo_voice as previous
from scripts.audio_blind_spatial_voice import WINDOWS

SCHEMA = "chipmunk.audio-spatial-mask-voice/1"
RATE = native.RATE
STEP = native.STEP
FFT = native.WINDOW
MAX_LAG = 22
MIN_SEPARATION = 4
RELATIVE_PEAK = 0.38
UPDATES = 8
SHRINKAGE = 0.20
MAX_OUTPUTS = 3
MIN_OUTPUT_FRACTION = 0.015
READOUTS = ("mono", "earlier_either", "directional")
LOCATIONS = ("surrey", *previous.LOCATIONS)
METHOD = {
    "comparison": "Native mono and prior AuxIVA maximum, independently frozen on the ordinary Surrey set. Use the same two-second windows and speech gate for the new direction masks. Score each anonymous output separately and take the largest admitted response. In the additive comparison keep all mono findings.",
    "analysis": "Use the existing sample-clock-preserving 1024-sample square-root Hann transform with 256 sample hop. Propose up to three fixed interchannel lag modes on the submitted six-second track, from -22 through 22 samples. Score summed positive phase agreement cubed over 250–2400 Hz, weighted by square root of cross-channel magnitude and capped at its 85th percentile. Require local lag maxima, four samples of separation, a second peak at least 0.38 of the strongest and nondegenerate joint covariance.",
    "masks": "Seed per-frequency complex two-channel angular Gaussian scatter matrices with each phase-aligned broadband lag and its weighted relative amplitude, limited to plus or minus 12 dB. Eight weighted EM updates on unit channel vectors. Shrink 20% of normalized covariance back to the fixed frequency-specific seed and blend 15% equal mixture weights. Use masks from 125–6400 Hz with nearest fitted edge masks outside. Reconstruct the mixture mean through each mask with unchanged native clock, unity total and no normalizing gain. Decline clipped views and those with less than 1.5% of mixture energy.",
    "timing": "Pick the strongest admitted response with the earlier window breaking ties. Require at least 0.30 interval IoU and coverage of at least half the authored edit for a useful result. Report repeated source groups by layout, false ordinary clips and groups, misplaced flags, unassessed windows, and equal ordinary-group ranking as retrospective only.",
    "gate": "Require at least two net additional useful held edit source groups across two generator families, no added background-only flagged group in any layout, and at most one added ordinary flagged group in each layout. A pass on reused recordings supports fresh transfer only. All outputs are anonymous and no candidate contributes to the service score.",
}


def hash_file(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def lag_anchors(spectrum):
    frequency = np.fft.rfftfreq(FFT, 1 / RATE)
    band = (frequency >= 250) & (frequency <= 2400)
    left, right = np.moveaxis(spectrum[band], -1, 0)
    cross = right * left.conj()
    magnitude = abs(cross)
    if np.sum(magnitude) <= 1:
        return []
    unit_phase = cross / np.maximum(magnitude, 1e-12)
    weight = np.sqrt(magnitude)
    np.minimum(weight, np.percentile(weight, 85), out=weight)
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    aligned = (unit_phase[None] * np.exp(2j * np.pi * lags[:, None, None] * frequency[None, band, None] / RATE)).real
    agreement = np.maximum(aligned, 0) ** 3
    scores = np.sum(agreement * weight[None], axis=(1, 2))
    peaks = [i for i, value in enumerate(scores) if value >= scores[max(0, i - 1)]
             and value >= scores[min(i + 1, len(scores) - 1)]]
    selected = []
    for i in sorted(peaks, key=lambda j: (-scores[j], abs(lags[j]), lags[j])):
        if selected and (scores[i] < RELATIVE_PEAK * scores[selected[0]]
                         or any(abs(lags[i] - lags[j]) < MIN_SEPARATION for j in selected)):
            continue
        selected.append(i)
        if len(selected) == MAX_OUTPUTS:
            break
    result = []
    for i in selected:
        support = agreement[i] ** 2 * weight
        ratio = np.log(np.maximum(abs(right), 1e-12)) - np.log(np.maximum(abs(left), 1e-12))
        gain = float(np.exp(np.clip(np.sum(support * np.clip(ratio, -2, 2)) / max(np.sum(support), 1e-12),
                                    -math.log(4), math.log(4))))
        result.append((int(lags[i]), gain))
    return sorted(result)


def angular_masks(spectrum, frequencies, anchors):
    identity = np.eye(2, dtype=np.complex128)
    seeds = np.array([[np.ones(len(frequencies)), gain * np.exp(-2j * np.pi * frequencies * lag / RATE)]
                      for lag, gain in anchors]).transpose(0, 2, 1)
    seeds /= np.linalg.norm(seeds, axis=-1, keepdims=True)
    seeds = 1.76 * np.einsum("kfi,kfj->kfij", seeds, seeds.conj()) + 0.12 * identity
    covariance = seeds.copy()
    weights = np.full(covariance.shape[:2], 1 / len(anchors))
    norm = np.linalg.norm(spectrum, axis=-1)
    unit = spectrum / np.maximum(norm[..., None], 1e-12)
    reliable = norm >= np.maximum(np.median(norm, axis=1, keepdims=True) * .15, 1e-9)
    for step in range(UPDATES + 1):
        a, b, d = covariance[..., 0, 0].real, covariance[..., 0, 1], covariance[..., 1, 1].real
        determinant = np.maximum(a * d - abs(b) ** 2, 1e-12)
        left, right = np.moveaxis(unit, -1, 0)
        quadratic = np.maximum((d[..., None] * abs(left) ** 2 + a[..., None] * abs(right) ** 2
                                - 2 * np.real(b[..., None] * left.conj() * right)) / determinant[..., None], 1e-12)
        logits = np.log(weights / determinant)[..., None] - 2 * np.log(quadratic)
        raw = np.exp(logits - logits.max(axis=0))
        posterior = raw / raw.sum(axis=0)
        if step == UPDATES:
            return posterior
        supported = posterior * reliable
        scatter = np.einsum("kft,fti,ftj->kfij", supported / quadratic, unit, unit.conj(), optimize=True)
        trace = scatter.trace(axis1=-2, axis2=-1).real[..., None, None]
        normalized = np.where(trace > 1e-12, 2 * scatter / np.maximum(trace, 1e-12), seeds)
        covariance = (1 - SHRINKAGE) * normalized + SHRINKAGE * seeds
        total = supported.sum(axis=2)
        proportions = total / np.maximum(total.sum(axis=0), 1e-12)
        weights = np.where(total.sum(axis=0) > 0, 0.85 * proportions + 0.15 / len(anchors), 1 / len(anchors))
    raise AssertionError("The fixed update count did not produce a mask")


def reconstruct(spectrum, window, inverse):
    blocks = inverse(spectrum.transpose(1, 2, 0), n=FFT, axis=-1) * window
    output = np.zeros((native.SAMPLES + FFT, spectrum.shape[-1]))
    weight = np.zeros(native.SAMPLES + FFT)
    for index, block in enumerate(blocks):
        start = index * STEP
        output[start:start + FFT] += block.T
        weight[start:start + FFT] += window ** 2
    section = slice(FFT // 2, FFT // 2 + native.SAMPLES)
    return output[section] / weight[section, None]


def separate(stereo, *, fft=rfft, inverse=irfft):
    spectrum, window = native.analyze(stereo, fft)
    frequencies = np.fft.rfftfreq(FFT, 1 / RATE)
    band = (frequencies >= 125) & (frequencies <= 6400)
    samples = spectrum[band].reshape(-1, 2)
    eigenvalues = np.linalg.eigvalsh(samples.T.conj() @ samples)
    if eigenvalues[0] <= 1e-4 * eigenvalues[1] or eigenvalues[1] <= 1:
        return [], []
    anchors = lag_anchors(spectrum)
    if len(anchors) < 2:
        return [], []
    fitted = np.flatnonzero(band)
    masks = angular_masks(spectrum[band], frequencies[band], anchors)
    full = masks[:, np.clip(np.arange(len(frequencies)) - fitted[0], 0, len(fitted) - 1)]
    mixture = spectrum.mean(axis=2)
    waves = reconstruct(mixture[..., None] * full.transpose(1, 2, 0), window, inverse)
    native.require(np.max(abs(waves.sum(axis=1) - stereo.mean(axis=1))) < 1e-6,
                   "The spatial masks do not reconstruct the original mean")
    power = np.mean(stereo.mean(axis=1) ** 2)
    quantized = []
    for wave in waves.T:
        rounded = np.rint(wave)
        usable = np.mean(rounded ** 2) >= MIN_OUTPUT_FRACTION * power and np.all((rounded >= -32768) & (rounded <= 32767))
        quantized.append(rounded.astype("<i2") if usable else None)
    return quantized, [lag for lag, _ in anchors]


def freeze_inputs(args):
    old, rows = previous.bound_rosters(args.pilot, args.room)
    old_frozen = native.read(args.prior / "freeze.json")
    prior = args.prior / "private-windows.jsonl"
    with prior.open() as stream:
        header = stream.readline().strip()
        readings = [json.loads(line) for line in stream]
    public = native.read(Path("docs/evaluation-panels/audio-unassigned-stereo-voice-2026-11-13.json"))
    native.require(old_frozen == old and header == hash_file(args.prior / "freeze.json") and len(readings) == len(rows),
                   "Prior stereo baseline differs from its freeze")
    native.require(public["private_ledger_sha256"] == hash_file(prior), "Prior voice ledger differs from publication")
    return {"schema": SCHEMA, "method": METHOD, "assets": old["assets"], "sources": old["sources"],
            "prior_ledger_sha256": hash_file(prior), "source_code_sha256": hash_file(Path(__file__)),
            "reviewer_sha256": hash_file(Path(__file__).with_name("review_audio_spatial_mask_voice_transfer.py")),
            "windows": [list(window) for window in WINDOWS]}, rows, readings


def row_source(index, entry, prior):
    folder, location, kind, source = entry
    expected = previous.metadata(index, location, kind, source, prior["route"])
    native.require(expected == {**prior, "reads": []}, "Reused baseline case metadata differs")
    stereo = native.load_pcm(folder, source, "stereo")
    mono = native.load_pcm(folder, source, "original")
    foreground = native.load_pcm(folder, source, "foreground")
    background = native.load_pcm(folder, source, "background")
    native.require(np.array_equal(stereo, foreground + background) and np.array_equal(mono, np.rint(stereo.mean(axis=1))),
                   "Original acoustic mixture differs")
    for (start, end), (digest, _) in zip(WINDOWS, prior["reads"][0], strict=True):
        native.require(digest == native.sha(mono[start:end].astype("<i2").tobytes()), "Prior mono window differs")
    return expected, stereo, foreground.mean(axis=1), mono


def acoustic_view(entry, prior, index, *, alternate=False):
    row, stereo, target, mono = row_source(index, entry, prior)
    waves, lags = separate(stereo, **({"fft": np.fft.rfft, "inverse": np.fft.irfft} if alternate else {}))
    row["lag_modes_samples"] = lags
    row["reads"] = []
    truth = slice(*row["truth"]) if row["truth"] else slice(0, 0)
    row["original_snr_db"] = [native.snr(target, mono), native.snr(target[truth], mono[truth])]
    row["candidate_snr_db"] = [None if wave is None else [native.snr(target, wave), native.snr(target[truth], wave[truth])]
                               for wave in waves]
    row["baselines"] = {"mono": [pair[1] for pair in prior["reads"][0]],
                        "earlier_either": [max((view[i][1] for view in prior["reads"][1:]
                                                if view is not None and view[i][1] is not None), default=None)
                                           for i in range(len(WINDOWS))]}
    return row, waves


def rows_for(args, lock, sources, readings, role, *, audit=False):
    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.upgraded import SpeechActivity

    native.require(hash_file(args.weights / "silero_vad.jit") == lock["assets"]["silero_vad.jit"], "Speech activity weights differ")
    gate = SpeechActivity(args.weights / "silero_vad.jit")
    model = None
    if not audit:
        from scripts.audio_strong_traffic_view import network_free
        from scripts.run_audio_specialists_v15 import load_voice

        network_free()
        model = load_voice({"weights": str(args.weights), "assets": lock["assets"]})
    selected = [i for i, (_, location, _, _) in enumerate(sources) if (location == "surrey") == (role == "development")]
    path = args.output / f"{role}.json"
    old = iter(native.read(path)["rows"]) if audit else None
    rows, cache = [], {}
    started = time.monotonic()
    for index in selected:
        row, waves = acoustic_view(sources[index], readings[index], index, alternate=audit)
        known = next(old) if audit else None
        for track, integers in enumerate(waves):
            if integers is None:
                row["reads"].append(None)
                continue
            pcm = integers.astype(np.float32) / 32768
            spans = gate.spans(pcm)
            entries = []
            for slot, (start, end) in enumerate(WINDOWS):
                digest = native.sha(integers[start:end].tobytes())
                eligible = window_in_speech(spans, start / RATE, end / RATE)
                response = None
                if eligible and audit:
                    response = known["reads"][track][slot][1]
                    native.require(math.isfinite(response), "Speech admission differs from original run")
                elif eligible:
                    if digest not in cache:
                        (probability,) = model.predict([pcm[start:end]])
                        native.require(math.isfinite(probability) and 0 <= probability <= 1, "Invalid voice response")
                        cache[digest] = float(logit(probability))
                    response = cache[digest]
                entries.append([digest, response])
            row["reads"].append(entries)
        if audit:
            native.require(row == known, f"Source waveform, admission or acoustic measure changed on case {index}")
        rows.append(row)
        if len(rows) % 10 == 0 or len(rows) == len(selected):
            print(role, len(rows), len(selected), "cases and", len(cache), "forwards in", round(time.monotonic() - started, 2), flush=True)
    if audit:
        return {"cases": len(rows), "waveforms": sum(sum(view is not None for view in row["reads"]) for row in rows)}
    native.write(path, {"freeze_sha256": hash_file(args.output / "freeze.json"), "forwards": len(cache), "rows": rows})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "held", "audit"))
    parser.add_argument("pilot", type=Path)
    parser.add_argument("room", type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("models/weights/audio-upgrade"))
    args = parser.parse_args()
    lock, sources, readings = freeze_inputs(args)
    path = args.output / "freeze.json"
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        native.write(path, lock)
        return
    native.require(native.read(path) == lock, "Frozen inputs changed")
    from scripts import review_audio_spatial_mask_voice_transfer as review

    if args.phase == "development":
        rows = rows_for(args, lock, sources, readings, args.phase)
        native.write(args.output / "points.json", {"points": review.points(rows),
                     "development_sha256": hash_file(args.output / "development.json")})
    elif args.phase == "held":
        previous_points = native.read(args.output / "points.json")
        native.require(previous_points["development_sha256"] == hash_file(args.output / "development.json")
                       and previous_points["points"] == review.points(native.read(args.output / "development.json")["rows"]),
                       "Operating points changed before confirmation")
        rows_for(args, lock, sources, readings, args.phase)
    else:
        receipt = {role: rows_for(args, lock, sources, readings, role, audit=True) for role in ("development", "held")}
        native.write(args.output / "source-audit.json", receipt)


if __name__ == "__main__":
    main()
