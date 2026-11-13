"""Assess first-formant changes on the original clock of noisy short speech."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from scripts.audio_local_context_transfer import bindings, file_sha, pcm
from scripts.audio_overlap_pitch_transfer import canonical, digest, inputs, key, read, require, wav_pcm
from scripts.review_audio_formant_local_transfer import ARMS, report, selection_points

ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
RATE, LPC_RATE = 16_000, 16_000 * 2 / 3
FRAME, HOP, CENTER = 320, 160, 240
INTERIOR, FLANK, GUARD, STEP = 40, 24, 3, 4
SCHEMA = "chipmunk.audio-formant-local-transfer/1"
METHOD = {
    "input": "One six-second mono 16 kHz PCM16 upload. No isolated source, reference copy, edit range, speech label, or noise level enters the formant measurement.",
    "tracking": "Resample by 2/3, apply 50 Hz preemphasis, and use a 30 ms Hann window on 15 ms source ticks. Retain the two lowest distinct LPC resonances within 200–1200 Hz and 500–3000 Hz, separated by at least 200 Hz with bandwidths between 20 and 900 Hz. Require at least 0.40 periodic autocorrelation between 70 and 400 Hz on the same native original frame. Compare order 10 to order 12 as a check on tracker dependence.",
    "spans": "Measure 600 ms spans on a 60 ms source-clock grid. Each comparison flank uses 360 ms, separated by a 45 ms guard. Require at least six tracked frames in each flank and ten inside. Each descriptor takes the smaller positive decrease against either flank. Select the largest response with earlier time resolving ties.",
    "arms": "Narrowing arm uses order 10 log F1 bandwidth. The two-order arm retains frames with F1 and F2 tracking agreement within 120 Hz and F1 bandwidth agreement within 100 Hz. It takes the median of the positive drops in log F1 bandwidth, log F1 and log F1/F2. Trackers are correlated and the two-order agreement measures numerical stability only.",
    "point": "For each candidate arm use the maximum assessed ordinary or isolated-background score on the primary development bank across dry, +5, 0 and -5 dB. Flag strictly above this frozen point. Never tune on replication or transfer. On matched primary and replication bytes, retain the prior frozen short response and exact range.",
    "gate": "For optional follow-up, add at least two useful locations across at least two generators in the three-generator -5 dB replication, at most one additional ordinary group and no new room-only flag over incumbent. Also locate at least two useful edits at -5 dB and two at -10 dB on the additional paired-source transfer without more than one ordinary group flagged at either level. Useful means 0.30 temporal IoU and at least half the known edit covered.",
}


@lru_cache(maxsize=1)
def hanning():
    return np.hanning(FRAME)


def _lpc(correlation, order):
    coefficients = np.zeros((len(correlation), order + 1))
    coefficients[:, 0] = 1.
    error = correlation[:, 0] * 1.0001
    for degree in range(1, order + 1):
        prior = coefficients[:, 1:degree].copy()
        reflected = -(correlation[:, degree] + np.sum(prior * correlation[:, degree - 1:0:-1], axis=1))
        reflected = np.clip(reflected / np.maximum(error, 1e-12), -.999, .999)
        coefficients[:, 1:degree] = prior + reflected[:, None] * prior[:, ::-1]
        coefficients[:, degree] = reflected
        error *= 1 - reflected**2
    return coefficients


def _resonances(coefficients):
    roots = np.roots(coefficients)
    roots = roots[np.imag(roots) > 0]
    frequencies = np.angle(roots) * LPC_RATE / (2 * np.pi)
    widths = -np.log(np.maximum(np.abs(roots), 1e-12)) * LPC_RATE / np.pi
    eligible = (frequencies >= 200) & (frequencies <= 3000) & (widths >= 20) & (widths <= 900)
    indices = np.flatnonzero(eligible)
    if len(indices) < 2:
        return (np.nan,) * 3
    first, second = indices[np.argsort(frequencies[indices])[:2]]
    a, b = frequencies[[first, second]]
    if a > 1200 or b < 500 or b - a < 200:
        return (np.nan,) * 3
    return a, b, widths[first]


def track(samples):
    signal = resample_poly(samples.astype(np.float64) / 32768., 2, 3)
    original_frames = np.lib.stride_tricks.sliding_window_view(signal, FRAME)[::HOP]
    centered = original_frames - original_frames.mean(axis=1, keepdims=True)
    energy = np.sum(centered**2, axis=1)
    ac = np.fft.irfft(abs(np.fft.rfft(centered, n=1024))**2, n=1024)
    pitch_min, pitch_max = int(LPC_RATE / 400), int(LPC_RATE / 70)
    admitted = (energy >= FRAME * .001**2) & (ac[:, pitch_min:pitch_max + 1].max(axis=1) >= .40 * energy)
    difference = signal - np.exp(-2 * np.pi * 50 / LPC_RATE) * np.r_[signal[0], signal[:-1]]
    selected = np.lib.stride_tricks.sliding_window_view(difference, FRAME)[::HOP][admitted] * hanning()
    spectrum = np.fft.rfft(selected, n=1024)
    autocorrelation = np.fft.irfft(abs(spectrum)**2, n=1024)[:, :13]
    values = []
    for order in (10, 12):
        result = np.full((len(original_frames), 3), np.nan)
        if admitted.any():
            result[admitted] = [_resonances(row) for row in _lpc(autocorrelation, order)]
        values.append(result)
    primary, alternate = values
    agreements = np.all(abs(primary - alternate) <= (120., 120., 100.), axis=1)
    agreement = np.where(agreements[:, None], (primary + alternate) / 2, np.nan)
    return primary, agreement


def _measure(frames, multifeature):
    valid = np.isfinite(frames).all(axis=1)
    log_f1, log_f2, log_bw = np.log(frames).T
    columns = np.column_stack((log_bw, log_f1, log_f1 - log_f2)) if multifeature else log_bw[:, None]
    count, best = 0, None
    for start in range(FLANK + GUARD, len(frames) - INTERIOR - FLANK - GUARD + 1, STEP):
        end = start + INTERIOR
        slices = (slice(start - FLANK - GUARD, start - GUARD), slice(start, end), slice(end + GUARD, end + GUARD + FLANK))
        coverage = [int(valid[span].sum()) for span in slices]
        if any(observed < minimum for observed, minimum in zip(coverage, (6, 10, 6), strict=True)):
            continue
        count += 1
        medians = np.round([np.median(columns[span][valid[span]], axis=0) for span in slices], 7)
        score = round(float(np.median(np.maximum(0., np.minimum(medians[0] - medians[1], medians[2] - medians[1])))), 7)
        if best is None or score > best[0]:
            native_start = start * 240 + CENTER
            best = [score, [native_start, native_start + INTERIOR * 240], coverage, medians.tolist()]
    return {"tracked": int(valid.sum()), "eligible_ranges": count, "best": best}


def locate(raw):
    pcm16 = np.frombuffer(raw, dtype="<i2")
    primary, stable = track(pcm16)
    return dict(zip(ARMS, (_measure(primary, False), _measure(stable, True)), strict=True))


def roster(args):
    earlier_lock, primary = inputs(args)
    replica, paths, replica_point = bindings(args)
    rows = []
    for item in primary:
        role = item["role"]
        template = "{}_{}_{}_submitted.wav" if role == "transfer" else "{}_{}_{}.wav"
        paths[(role, *key(item))] = (args.transfer_wavs if role == "transfer" else args.primary_wavs) / template.format(*key(item))
        incumbent = item.get("incumbent", {}).get("short", {}).get("best")
        rows.append({"role": role, "id": [role, *key(item)], "group": item["case"], "arm": item["kind"],
                     "truth": item["truth"], "sha256": item["sha256"], "level": item["level"], "family": "chatterbox",
                     "incumbent": None if incumbent is None else [incumbent["score"], incumbent["range"]]})
    for item in replica:
        if item["role"] != "held":
            continue
        paths[("replication", item["id"])] = paths.pop(item["id"])
        rows.append({"role": "replication", "id": ["replication", item["id"]], "group": item["group"], "arm": item["arm"],
                     "truth": item["truth"], "sha256": item["pcm_sha256"], "level": -5, "family": item["family"],
                     "incumbent": item["incumbent"]})
    require(len(rows) == 232, "Unexpected panel size")
    lock = {"schema": SCHEMA, "method": METHOD, "runner_sha256": file_sha(Path(__file__)),
            "input_pins": earlier_lock["input_pins"] | {"replication": file_sha(args.prior)},
            "input_loaders": [file_sha(Path(__file__).with_name(name)) for name in
                              ("audio_overlap_pitch_transfer.py", "audio_local_context_transfer.py", "review_audio_formant_local_transfer.py")],
            "numpy": np.__version__, "rows_sha256": digest(canonical(rows)), "rows": len(rows),
            "incumbent_points": {"assessment": read(args.baseline)["operating_points"]["points"]["short"],
                                 "replication": replica_point}}
    return rows, paths, lock


def measure(rows, paths, *, development):
    result = []
    for row in rows:
        if (row["role"] == "development") != development:
            continue
        path = paths[tuple(row["id"])]
        raw = pcm(path) if row["role"] == "replication" else wav_pcm(path)
        require(digest(raw) == row["sha256"], "Source changed after freeze")
        result.append(row | {"measures": locate(raw)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "score", "replay"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--primary-wavs", type=Path, required=True)
    parser.add_argument("--transfer-wavs", type=Path, required=True)
    parser.add_argument("--held", type=Path, default=ROOT / "backend/data/independent-evaluation/audio-remote-neighbor-replication-2026-11-12")
    for name, stem in (("intake", "audio-phoneme-short-edit-intake-2026-11-13"), ("baseline", "audio-phoneme-short-edit-score-2026-11-13"),
                       ("transfer", "audio-clean-reference-transfer-2026-11-13"), ("prior", "audio-remote-neighbor-replication-2026-11-12")):
        parser.add_argument(f"--{name}", type=Path, default=PANELS / f"{stem}.json")
    args = parser.parse_args()
    args.development_wavs = args.primary_wavs
    rows, paths, lock = roster(args)
    frozen, fit_path, receipt = (args.run / name for name in ("freeze.json", "development.json", "public.json"))
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        frozen.write_bytes(canonical(lock))
        print(f"Froze {len(rows)} original PCM deliveries before candidate inference")
        return
    require(frozen.read_bytes() == canonical(lock), "Frozen comparison inputs differ")
    if args.phase == "fit":
        fit = measure(rows, paths, development=True)
        points = selection_points(fit)
        with fit_path.open("xb") as dest:
            dest.write(canonical({"rows": fit, "points": points}))
        print(f"Sealed development before held inference. {points}")
        return
    fitting = read(fit_path)
    if args.phase == "replay":
        require(fitting["rows"] == measure(rows, paths, development=True), "Native development replay differs")
    held = measure(rows, paths, development=False)
    public = {"freeze": lock, "fitting_sha256": file_sha(fit_path), "development": fitting, "held": held,
              "summary": report(held, fitting["points"], lock["incumbent_points"])}
    if args.phase == "score":
        with receipt.open("xb") as dest:
            dest.write(canonical(public))
    else:
        require(receipt.read_bytes() == canonical(public), "Native result replay differs")
    print(f"{args.phase} verified {len(held)} held deliveries. {public['summary']['gate_pass']}")


if __name__ == "__main__":
    main()
