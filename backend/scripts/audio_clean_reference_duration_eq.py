"""Test a physical full-recording EQ guard on clean-reference duration evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from scripts import audio_clean_reference_duration as base
from scripts.audio_overlap_pitch_transfer import canonical, read, require, useful

SCHEMA = "chipmunk.audio-clean-reference-duration-eq/1"
FFT, HOP = 2048, 1024
BANDS = ((300, 850), (850, 1450), (1450, 2100), (2100, 2800), (2800, 3700))
METHOD = {
    "guard": "Measure whole-recording shared-signal transfer in five frequency bands. Fit signed amplitude gain to the clean earlier waveform with a tapered FFT and energy-weighted real cross spectra. Require a broad tilt of at least .35 and temporal agreement in at least two of three four-second parts, including first and last. A supported EQ explanation suppresses only the long comparison's local flag and contributes no AI confidence.",
    "coverage": "Read primary 12-second development and held input hashes unchanged. Preserve the exact previous numeric lane and status for the primary study. Supplementary stress uses the exact same source prompts/background with two distinct global EQ curves: two-pole lowpass 1900 Hz and 5 dB high-shelf above 2000 Hz. No metric or signal from the base held ledger influences the fixed guard arithmetic.",
    "gate": "Across all three held levels, decline all wrong references, suppress at least three of the four original EQ false findings at -5 dB with zero suppressed correct real/synthetic ranges at that level, and suppress at least half of any alternate-curve flags at each level. Report secondary readout separately. Do not claim untreated EQ controls are new source groups.",
}


def spectra(signal):
    starts = np.lib.stride_tricks.sliding_window_view(signal.astype(np.float64), FFT)[::HOP]
    return np.fft.rfft(starts * np.hanning(FFT))


def band_gain(x, y, mask):
    x, y = x[:, mask], y[:, mask]
    power = np.sum(abs(x) ** 2, axis=1)
    weights = np.minimum(1., np.quantile(power, .85) / np.maximum(power, 1e-6))
    return float(np.dot(weights, np.sum(np.real(np.conj(x) * y), axis=1)) / max(1e-6, np.dot(weights, power)))


def profile(first, second):
    frequencies = np.fft.rfftfreq(FFT, 1 / base.RATE)
    masks = [(frequencies >= low) & (frequencies < high) for low, high in BANDS]
    vectors = []
    for offset in (0, 4 * base.RATE, 8 * base.RATE):
        end = offset + 4 * base.RATE
        x, y = (spectra(signal[offset:end]) for signal in (first, second))
        gains = np.array([band_gain(x, y, mask) for mask in masks])
        if min(gains[:2]) <= .25:
            vectors.append(None)
        else:
            vectors.append(gains / np.median(gains[:2]))
    return vectors


def eq_explanation(first, second):
    vectors = profile(first, second)
    earlier, middle, later = vectors
    if earlier is None or later is None:
        return False, vectors
    typical = np.median([row for row in vectors if row is not None], axis=0)
    # The tilt must have the same sign and approximately the same shape across the file.
    tilt = typical[-1] - typical[0]
    direction = float(np.sign(tilt))
    consistent = all(direction * (vector[-1] - vector[0]) >= .25
                     and float(np.median(abs(vector - typical))) <= .18 for vector in (earlier, later))
    if middle is not None:
        consistent &= (direction * (middle[-1] - middle[0]) >= .15
                       and float(np.median(abs(middle - typical))) <= .22)
    return bool(consistent and abs(tilt) >= .35 and direction * (typical[-1] - typical[2]) >= .18), vectors


def alternate(first, original, style):
    first = first.astype(np.float64)
    original = original.astype(np.float64)
    cutoff = {"other_lowpass": 1900, "high_shelf": 2000}[style]
    sos = butter(2, cutoff, fs=base.RATE, btype="lowpass", output="sos")
    low = sosfiltfilt(sos, first)
    changed = low if style == "other_lowpass" else low + (first - low) * 10 ** (5 / 20)
    changed = base.short.pair.normed(changed, first)
    submitted = np.rint(changed + original - first)
    require(np.max(abs(submitted)) <= 32767, "EQ stress render clips")
    return submitted.astype("<i2")


def observe(row, first, second, comparison):
    explained, profiles = eq_explanation(first, second)
    original = comparison["corroborated"]
    return {**row, "admitted": comparison["admitted"], "prior_range": original["samples"] if original else None,
            "eq_explains": explained, "profiles": [None if p is None else [round(float(v), 8) for v in p] for p in profiles],
            "retained_range": original["samples"] if original and not explained else None}


def summary(rows):
    output = {}
    for role in ("development", "assessment"):
        output[role] = {}
        for level in base.LEVELS:
            counts = Counter()
            for row in rows:
                if row["role"] != role or row["level"] != level:
                    continue
                kind = row["kind"]
                counts[f"rows:{kind}"] += 1
                counts[f"explained:{kind}"] += int(row["eq_explains"])
                counts[f"admitted:{kind}"] += int(row["admitted"])
                if row["prior_range"]:
                    counts[f"prior:{kind}"] += 1
                    if row["retained_range"]:
                        counts[f"retained:{kind}"] += 1
                    elif row["truth"] and useful(row["prior_range"], row["truth"]):
                        counts["useful_removed"] += 1
            output[role][str(level)] = dict(counts)
    held = output["assessment"]["-5"]
    output["gate_pass"] = (held.get("prior:global_eq", 0) - held.get("retained:global_eq", 0) >= 3
                           and not held.get("useful_removed")
                           and all(all(not output["assessment"][str(level)].get(f"admitted:{kind}") for kind in base.WRONG)
                                   for level in base.LEVELS)
                           and all(all(2 * output["assessment"][str(level)].get(f"retained:{style}", 0) <=
                                       output["assessment"][str(level)].get(f"prior:{style}", 0)
                                       for style in ("other_lowpass", "high_shelf")) for level in base.LEVELS))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "replay"))
    for arg in ("cache", "old-cache", "record", "odss-cache", "ami", "gain-ami", "prior", "run"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    native = read(args.prior / "result.json")
    lock = {"schema": SCHEMA, "method": METHOD, "runner": base.digest(Path(__file__).read_bytes()),
            "primary_result": base.digest((args.prior / "result.json").read_bytes()), "primary_freeze": native["freeze"]}
    pin = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        pin.write_bytes(canonical(lock))
        print("Locked the signal-shape guard and two alternate EQ stresses before inference")
        return
    require(pin.read_bytes() == canonical(lock), "Frozen guard inputs differ")
    rows, pairs, _ = base.render(args)
    prior = {row["id"]: row for row in native["rows"]}
    require(len(rows) == len(prior) == 495, "Prior duration roster differs")
    measurements = []
    for row in rows:
        earlier, submitted = pairs[row["id"]]
        previous = prior[row["id"]]
        require(all(previous[key] == value for key, value in row.items()), "Prior source bytes differ")
        measurements.append(observe(row, earlier, submitted, previous["comparison"]))
        if row["kind"] == "untouched":
            for style in ("other_lowpass", "high_shelf"):
                variant = alternate(earlier, submitted, style)
                item = row | {"kind": style, "id": row["id"].replace("untouched", style),
                              "submitted_sha256": base.digest(variant.tobytes())}
                measurements.append(observe(item, earlier, variant, base.compare(earlier, variant)))
    result = {"freeze": lock, "rows": measurements, "summary": summary(measurements)}
    path = args.run / "result.json"
    if args.phase == "score":
        require(not path.exists(), "Guard was already scored")
        path.write_bytes(canonical(result))
    else:
        require(path.read_bytes() == canonical(result), "Guard replay differs")
    print(result["summary"]["gate_pass"])


if __name__ == "__main__":
    main()
