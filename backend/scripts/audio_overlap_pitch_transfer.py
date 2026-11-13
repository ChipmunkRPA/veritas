"""Freeze and assess one blind harmonic-change cue on fixed noisy voice edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

RATE, SAMPLES, FRAME, HOP, NFFT = 16_000, 96_000, 1024, 320, 4096
SCHEMA = "chipmunk.audio-overlap-pitch-transfer/1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "docs/evaluation-panels"
PITCH = 65 * 2 ** (np.arange(141) / 60)
KINDS_EXCLUDED = {"different_reader_reference", "same_words_rerecorded_reference"}
METHOD = {
    "input": "Only the 16 kHz mono submitted waveform. Peak-safe same-word replacements in retained AMI mixtures. Earlier copies, known levels, truths and model scores cannot enter candidate inference.",
    "pitch": "A 64 ms Hann FFT advances 20 ms. Subtract a smoothed log magnitude. Seek two distinct harmonic series, with the second series supported at least 26 Hz from bins of the strongest series. Require at least five supporting partials, each at least 2.5 percent of the peak magnitude between 250 and 3600 Hz, and reject octave/subharmonic overlap.",
    "cue": "Compare two pitch distributions across both ends of 0.80 s candidates on an 80 ms grid. Each endpoint uses two 140 ms regions and a 20 ms guard. Require concurrent series in both regions. Score movement of the quieter series only when the stronger series stays relatively continuous.",
    "operating_point": "Use the maximum candidate score on the 38 ordinary/background primary development deliveries, including all noise levels and dry. Use a strict point. Seal development and the point before scoring held inputs. Compare the previously frozen incumbent and background-reduced points on identical primary inputs.",
    "scope": "Primary assessment reuses the earlier primary bank with two held reader groups; transfer uses three other held readers and two other AMI sessions from the clean-reference study. Their other-method outcomes were known. Count ordinary waveform replacements as genuine for synthetic detection, together with unchanged, gain, genuine resampling, EQ and isolated background.",
    "gate": "At -5 dB gain at least two useful primary localizations above the incumbent at no more than one added ordinary flag and no background-only flag. Transfer requires at least two of six at both -5 and -10 dB with at most one ordinary flag each. Useful means at least half of the true edit covered and at least .30 temporal IoU.",
}


def require(test, reason):
    if not test:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read(path):
    return json.loads(path.read_bytes())


@lru_cache(maxsize=1)
def frequency_grid():
    fundamental = PITCH[:, None]
    harmonic = np.arange(2, 17)[None, :]
    frequency = fundamental * harmonic
    bins = np.rint(frequency * NFFT / RATE).astype(np.int32)
    return fundamental, frequency, bins, (frequency >= 250) & (frequency <= 3600)


def tracks(pcm):
    fundamental, frequency, bins, valid = frequency_grid()
    frames = np.lib.stride_tricks.sliding_window_view(pcm.astype(np.float64) / 32768, FRAME)[::HOP]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(FRAME), n=NFFT))
    log_amplitude = np.log(np.maximum(spectrum, .0001))
    whitened = log_amplitude - gaussian_filter1d(log_amplitude, 10, axis=1, mode="nearest")
    # Broadens the narrow harmonic peak by one FFT bin for moderate pitch motion.
    floor = .025 * spectrum[:, 64:922].max(axis=1)
    peaks = np.maximum.reduce([
        np.where(spectrum[:, bins + offset] >= floor[:, None, None], whitened[:, bins + offset], 0.)
        for offset in (-1, 0, 1)
    ])
    peaks = np.clip(peaks, 0., 3.)
    enough_partials = ((peaks >= .20) & valid).sum(axis=2) >= 5
    strengths = np.where(enough_partials, (peaks * valid).sum(axis=2) / valid.sum(axis=1), 0)
    first = strengths.argmax(axis=1)
    first_hz = PITCH[first, None, None]
    multiple = frequency[None, :, :] / first_hz
    off_comb = abs(multiple - np.rint(multiple)) * first_hz >= 26
    residual = valid & off_comb
    free = residual.sum(axis=2)
    relative = fundamental.T / first_hz[:, :, 0]
    distinct = (abs(np.log2(relative)) >= .12) & (abs(abs(np.log2(relative)) - 1) >= .10)
    second_scores = (peaks * residual).sum(axis=2) / np.maximum(free, 1)
    eligible = (((peaks >= .20) & residual).sum(axis=2) >= 5) & (free * 2 >= valid.sum(axis=1)) & distinct
    second_scores = np.where(eligible, second_scores, 0)
    second = second_scores.argmax(axis=1)
    index = np.arange(len(frames))
    usable = (strengths[index, first] >= .30) & (second_scores[index, second] >= .25)
    usable &= np.sqrt(np.mean(frames**2, axis=1)) >= .001
    return first, second, usable


def histogram(track, usable, start, end):
    cells = usable[start:end]
    if np.count_nonzero(cells) < 3:
        return None
    counts = np.bincount(track[start:end][cells], minlength=len(PITCH)).astype(np.float64)
    return gaussian_filter1d(counts / counts.sum(), 3, mode="constant")


def endpoint_changes(first, second, usable):
    endpoints = {}
    for boundary in range(8, len(usable) - 7):
        leading, trailing = (boundary - 8, boundary - 1), (boundary + 1, boundary + 8)
        histograms = [[histogram(series, usable, *bounds) for bounds in (leading, trailing)] for series in (first, second)]
        if any(value is None for row in histograms for value in row):
            continue
        dominant, secondary = (float(np.abs(before - after).sum() / 2) for before, after in histograms)
        if dominant <= .60:
            endpoints[boundary] = (max(0., secondary - dominant), dominant, secondary)
    return endpoints


def locate(pcm):
    first, second, usable = tracks(pcm)
    endpoints = endpoint_changes(first, second, usable)
    best, count = None, 0
    for start in range(8, len(usable) - 47, 4):
        stop = start + 40
        if start not in endpoints or stop not in endpoints:
            continue
        count += 1
        score = round(min(endpoints[start][0], endpoints[stop][0]), 8)
        if best is None or score > best["score"]:
            at = start * HOP + FRAME // 2
            best = {"score": score, "range": [at, at + 40 * HOP],
                    "ends": [[round(v, 8) for v in endpoints[tick]] for tick in (start, stop)]}
    return {"dual_pitch_frames": int(usable.sum()), "eligible_ranges": count, "best": best}


def key(row):
    return row["case"], row["level"], row["kind"]


def wav_pcm(path):
    with wave.open(str(path), "rb") as stream:
        require(stream.getparams()[:4] == (1, 2, RATE, SAMPLES), f"Source clock differs: {path}")
        raw = stream.readframes(SAMPLES)
    require(len(raw) == SAMPLES * 2, f"Source PCM is truncated: {path}")
    return raw


def inputs(args):
    primary = read(args.intake)
    baseline = read(args.baseline)
    transfer = read(args.transfer)
    require(len(primary["deliveries"]) == 108 and len(transfer["assessment"]) == 90, "Unexpected panel size")
    baseline_rows = {key(row): row for role in ("development", "assessment") for row in baseline[role]["rows"]}
    points = baseline["operating_points"]["points"]
    deliveries = []
    for row in primary["deliveries"]:
        prior = baseline_rows[key(row)]
        require(prior["pcm_sha256"] == row["pcm_sha256"] and prior["role"] == row["role"]
                and prior["source_edit_range"] == row["source_edit_range"], "Incumbent source bindings differ")
        path = args.primary_wavs / "{}_{}_{}.wav".format(*key(row))
        incumbents = {}
        for arm in ("full", "short", "reduced_short"):
            by_time = [item for item in prior["readings"][arm] if item["score"] is not None]
            best = max(by_time, key=lambda item: item["score"], default=None)
            incumbents[arm] = {"best": best, "point": points[arm]}
        raw = wav_pcm(path)
        require(digest(raw) == row["pcm_sha256"], f"Original WAV samples differ: {path}")
        for arm in ("full", "short"):
            for entry in prior["readings"][arm]:
                first, end = entry["range"]
                require(digest(raw[2 * first:2 * end]) == entry["window_sha256"], "Incumbent native window differs")
        deliveries.append({"case": row["case"], "role": row["role"], "level": row["level"], "kind": row["kind"],
                           "truth": row["source_edit_range"], "sha256": row["pcm_sha256"], "incumbent": incumbents})
    primary_count = len(deliveries)
    for row in transfer["assessment"]:
        if row["kind"] in KINDS_EXCLUDED:
            continue
        path = args.transfer_wavs / "{}_{}_{}_submitted.wav".format(*key(row))
        require(digest(wav_pcm(path)) == row["submitted_sha256"], f"Original transfer WAV samples differ: {path}")
        deliveries.append({"case": row["case"], "role": "transfer", "level": row["level"], "kind": row["kind"],
                           "truth": row["truth"], "sha256": row["submitted_sha256"]})
    lock = {"schema": SCHEMA, "method": METHOD, "runner_sha256": digest(Path(__file__).read_bytes()),
            "input_pins": {name: digest(path.read_bytes()) for name, path in (("intake", args.intake),
                                                                             ("baseline", args.baseline),
                                                                             ("transfer", args.transfer))},
            "delivery_sha256": digest(canonical(deliveries)), "primary": primary_count, "transfer": len(deliveries) - primary_count}
    return lock, deliveries


def measure(args, rows, role):
    result = []
    for row in rows:
        if row["role"] != role:
            continue
        path = (args.transfer_wavs / "{}_{}_{}_submitted.wav".format(*key(row)) if role == "transfer"
                else args.primary_wavs / "{}_{}_{}.wav".format(*key(row)))
        raw = wav_pcm(path)
        require(digest(raw) == row["sha256"], "Original changed between freeze and inference")
        result.append({**row, "candidate": locate(np.frombuffer(raw, dtype="<i2"))})
    return result


def useful(found, truth):
    a, b = found
    c, d = truth
    overlap = max(0, min(b, d) - max(a, c))
    return 2 * overlap >= d - c and 10 * overlap >= 3 * (max(b, d) - min(a, c))


def report(rows, point):
    output = {}
    for role in ("development", "assessment", "transfer"):
        levels = sorted({str(row["level"]) for row in rows if row["role"] == role})
        output[role] = {}
        for level in levels:
            subset = [row for row in rows if row["role"] == role and str(row["level"]) == level]
            output[role][level] = {}
            for arm in (("candidate",) if role == "transfer" else ("candidate", "full", "short", "reduced_short")):
                counts = Counter()
                for row in subset:
                    spec = {"best": row["candidate"]["best"], "point": point} if arm == "candidate" else row["incumbent"][arm]
                    best = spec["best"]
                    positive = row["kind"].startswith("synthetic_")
                    counts["positives" if positive else "controls"] += 1
                    if best is None:
                        counts["positive_unassessed" if positive else "control_unassessed"] += 1
                        continue
                    hit = positive and useful(best["range"], row["truth"])
                    counts["candidate_range_hits"] += int(hit)
                    if best["score"] <= spec["point"]:
                        continue
                    if positive:
                        counts["localized" if hit else "misplaced"] += 1
                        counts[f"localized_{row['kind']}"] += int(hit)
                    else:
                        counts["false_flags"] += 1
                        counts[f"false_{row['kind']}"] += 1
                output[role][level][arm] = dict(counts)
    primary = output["assessment"]["-5"]
    earlier, candidate = primary["short"], primary["candidate"]
    output["gate_pass"] = (
        candidate.get("localized", 0) >= earlier.get("localized", 0) + 2
        and candidate.get("false_flags", 0) <= earlier.get("false_flags", 0) + 1
        and candidate.get("false_background", 0) == 0
        and all(output["transfer"][level]["candidate"].get("localized", 0) >= 2
                and output["transfer"][level]["candidate"].get("false_flags", 0) <= 1 for level in ("-5", "-10"))
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "score", "replay"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--primary-wavs", type=Path, required=True)
    parser.add_argument("--transfer-wavs", type=Path, required=True)
    for name, basename in (("intake", "audio-phoneme-short-edit-intake"), ("baseline", "audio-phoneme-short-edit-score"),
                           ("transfer", "audio-clean-reference-transfer")):
        parser.add_argument(f"--{name}", type=Path, default=DEFAULT / f"{basename}-2026-11-13.json")
    args = parser.parse_args()
    lock, roster = inputs(args)
    frozen, fitting, public = (args.run / name for name in ("freeze.json", "development.json", "public.json"))
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        frozen.write_bytes(canonical(lock))
        print(f"Froze {len(roster)} source deliveries before candidate inference")
        return
    require(frozen.read_bytes() == canonical(lock), "Frozen experiment inputs changed")
    if args.phase == "fit":
        require(not fitting.exists() and not public.exists(), "Development has already been sealed")
        development = measure(args, roster, "development")
        controls = [row["candidate"]["best"]["score"] for row in development
                    if not row["kind"].startswith("synthetic_") and row["candidate"]["best"] is not None]
        point = max(controls, default=1.)
        fitting.write_bytes(canonical({"rows": development, "point": point, "eligible_controls": len(controls)}))
        print(f"Sealed point {point} using {len(controls)} eligible ordinary controls before scoring")
        return
    fitted = read(fitting)
    if args.phase == "replay":
        require(measure(args, roster, "development") == fitted["rows"], "Development replay differs")
    assessment = measure(args, roster, "assessment") + measure(args, roster, "transfer")
    result = {"freeze": lock, "point": fitted["point"], "fitting_sha256": digest(fitting.read_bytes()),
              "development": fitted["rows"], "assessment": assessment,
              "summary": report(fitted["rows"] + assessment, fitted["point"])}
    if args.phase == "score":
        require(not public.exists(), "Held inference already has a receipt")
        public.write_bytes(canonical(result))
    else:
        require(public.read_bytes() == canonical(result), "Independent rerun differs")
    print(f"{args.phase}: {len(assessment)} held deliveries, gate={result['summary']['gate_pass']}")


if __name__ == "__main__":
    main()
