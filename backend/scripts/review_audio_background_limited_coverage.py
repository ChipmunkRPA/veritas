"""Reconstruct a retained street-noise trial and audit its interval measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import periodogram

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = "backend/app/detectors/audio/calibrations/local-speech-v3.json"
RATE, SAMPLES, SCALE = 16_000, 48_000, 32768
FOREGROUNDS = {"genuine", "clone", "expressive", "stock"}
LEVELS = {"-6", "6"}
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def exact_pcm(path, sha256):
    raw = path.read_bytes()
    require(len(raw) == 2 * SAMPLES and digest(raw) == sha256, "Original PCM differs from its exact commitment")
    return np.frombuffer(raw, dtype="<i2")


def power_by_band(values, bands):
    """Use a SciPy periodogram independently of the runner's FFT implementation."""
    frames = np.array([values[i:i + 512] for i in range(0, len(values) - 511, 256)])
    frequency, power = periodogram(frames, fs=RATE, window=np.hanning(512), detrend=False, axis=-1)
    return np.stack([(power[:, (frequency >= a) & (frequency < b)] * (RATE / 512)).sum(axis=1)
                     for a, b in zip(bands[:-1], bands[1:], strict=True)], axis=1)


def db(power):
    return 10 * np.log10(np.maximum(power, 1e-12))


def direction(observed, calibration, admitted):
    raw = observed["raw"]
    require(admitted == (raw is not None), "The recorded voice score differs from its admitted samples")
    if not admitted:
        require(observed["stability"] is None, "An unassessed input has a stability response")
        return "unassessed", 0
    require(type(raw) in (int, float) and 0 <= raw <= 1, "Invalid classifier response")
    clipped = max(1e-6, min(1 - 1e-6, raw))
    logit = math.log(clipped) - math.log1p(-clipped)
    fit = calibration["parameters"]
    probability = 1 / (1 + math.exp(-(fit["slope"] * logit + fit["intercept"])))
    supported = []
    for label, score in enumerate((probability, 1 - probability)):
        guards = calibration["guards"][str(label)]
        if (1 + sum(value is None or value >= score for value in guards)) / (1 + len(guards)) > calibration["alpha"]:
            supported.append(label)
    if supported == [0] and probability <= .2 + 1e-12:
        stability = observed["stability"]
        require(type(stability) in (int, float) and math.isfinite(stability) and stability >= 0,
                "Original-side response lacks a completed stability measurement")
        return ("inconclusive" if stability > math.log(2) else "original"), 2
    require(observed["stability"] is None, "Unexpected stability measurement")
    return ("ai" if supported == [1] and probability >= .8 else "inconclusive"), 1


def reconstruct(clean, noise, row, *, bands, center, mode):
    if mode == "speech_band":
        clean_energy, noise_energy = (float(power_by_band(values[center], bands).sum()) for values in (clean, noise))
    else:
        clean_energy, noise_energy = (sum(values[center] ** 2) for values in (clean, noise))
    gain = math.sqrt(clean_energy / noise_energy) * 10 ** (-int(row["condition"]) / 20)
    scaled = noise * gain
    mixed = clean + scaled
    attenuation = min(1., .95 / max(abs(mixed)))
    require(abs(gain - row["noise_gain"]) < 1e-12 and abs(attenuation - row["mix_gain"]) < 1e-12,
            "Recorded mixture gains differ from the isolated samples")
    return np.rint(mixed * attenuation * SCALE).astype("<i2"), scaled


def truth(clean, noise, *, bands, focus, rule):
    def means(values):
        return np.array([power_by_band(values[a:b], bands).sum(axis=1).mean() for a, b in focus])

    foreground, background = means(clean), means(noise)
    snr = db(foreground) - db(background)
    minimum_foreground = max(1e-6, .016 * max(foreground))
    return ["absent" if energy < minimum_foreground else "masked" if level <= rule["ground_truth_masking_snr_db"] else
            "audible" if level >= rule["ground_truth_audible_snr_db"] else "transition"
            for energy, level in zip(foreground, snr, strict=True)]


def measured(wave, recorded, *, rule, bands, refs, focus):
    spans = recorded["speech_spans"]
    require(all(type(a) is int and type(b) is int and 0 <= a < b <= SAMPLES for a, b in spans)
            and all(b <= c for (_, b), (c, _) in zip(spans, spans[1:])), "Invalid recorded speech sample clock")

    def speech(left, right):
        return sum(max(0, min(right, end) - max(left, start)) for start, end in spans)

    flanks = np.array([np.median(power_by_band(wave[a:b], bands), axis=0) for a, b in refs])
    require(np.allclose(db(flanks), recorded["ref_power_db"], rtol=0, atol=1e-9), "Reference band powers differ")
    reference = flanks.mean(axis=0)
    agreement = np.count_nonzero(abs(db(flanks[0]) - db(flanks[1])) <= rule["maximum_flank_band_difference_db"])
    if speech(focus[0][0], focus[-1][1]) < rule["required_center_speech_samples"]:
        scope = "unassessed_no_speech_anchor"
    elif any(speech(a, b) > rule["maximum_reference_speech_samples"] for a, b in refs):
        scope = "unassessed_reference_speech"
    elif min(db(flanks.sum(axis=1))) < rule["minimum_reference_dbfs"]:
        scope = "unassessed_reference_quiet"
    elif agreement < rule["minimum_agreeing_bands"]:
        scope = "unassessed_reference_changed"
    else:
        scope = "assessed"
    require(scope == recorded["scope"], "Reference eligibility differs")
    flags = []
    for (a, b), row in zip(focus, recorded["intervals"], strict=True):
        band = np.median(power_by_band(wave[a:b], bands), axis=0)
        excess = db(sum(band)) - db(sum(reference))
        same = np.count_nonzero(abs(db(band) - db(reference) - excess) <= rule["maximum_band_shape_difference_db"])
        flag = bool(scope == "assessed" and excess <= rule["maximum_mixed_power_excess_db"]
                    and same >= rule["minimum_agreeing_bands"])
        require((a, b, speech(a, b)) == (row["start_sample"], row["end_sample"], row["detected_speech_samples"])
                and abs(excess - row["excess_db"]) < 1e-9 and same == row["matching_bands"] and flag == row["flag"],
                "Recorded interval differs from its exact source measurement")
        flags.append(flag)
    return scope, flags, speech(0, SAMPLES) >= SAMPLES // 2


def keyed_counts(rows, fields):
    counts = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        require(key not in counts, "Repeated aggregate key")
        counts[key] = {k: value for k, value in row.items() if k not in fields}
    return counts


def review(folder, published=None):
    names = ("freeze", "roster", "measurements", "aggregate", "private-evidence")
    data = {name: (folder / f"{name}.json").read_bytes() for name in names}
    freeze, roster, measured_rows, aggregate, recorded_evidence = (json.loads(data[name]) for name in names)
    mode = freeze.get("mix_band", "full_band")
    require(freeze["schema"] == "chipmunk.audio-background-limited-freeze/1"
            and aggregate["schema"] == "chipmunk.audio-background-limited-evaluation/1"
            and mode in {"full_band", "speech_band"} and aggregate["freeze"] == freeze
            and digest(data["freeze"]) == measured_rows["freeze_sha256"] == aggregate["freeze_sha256"]
            and digest(data["roster"]) == freeze["roster_sha256"]
            and digest(data["measurements"]) == aggregate["measurements_sha256"]
            and digest(data["private-evidence"]) == aggregate["private_evidence_sha256"], "Trial commitments differ")
    if published is not None:
        report = json.loads(published.read_text())
        require(report["schema"] == "chipmunk.audio-background-limited-comparison/1"
                and report["runs"][mode] == {"completed_aggregate_sha256": digest(data["aggregate"]), "receipt": aggregate},
                "The completed trial differs from the published receipt")
    calibration_data = (ROOT / CALIBRATION).read_bytes()
    require(digest(calibration_data) == freeze["code_sha256"][CALIBRATION], "Voice calibration differs from the freeze")
    calibration = json.loads(calibration_data)
    bands, refs, focus, rule = (freeze[key] for key in ("bands_hz", "ref", "focus", "rule"))
    require((freeze["sample_rate"], freeze["samples"], freeze["fft"], freeze["hop"]) == (RATE, SAMPLES, 512, 256)
            and bands == [250, 500, 1000, 2000, 4000]
            and refs == [[0, 5600], [42400, SAMPLES]] and focus == [[a, a + 9200] for a in range(5600, 42400, 9200)]
            and len(roster) == freeze["views"], "Unsupported interval or spectral grid")
    center = slice(focus[0][0], focus[-1][1])
    pcm = [exact_pcm(folder / "pcm" / f"{i:03}.pcm", row["pcm_sha256"]) for i, row in enumerate(roster)]
    cells, scopes, outcomes, warning_overlap = (defaultdict(Counter) for _ in range(4))
    known_noises, evidence, hit_groups = {}, [], defaultdict(set)
    forwards, mixed_count = 0, 0
    for i, (row, observed) in enumerate(zip(roster, measured_rows["rows"], strict=True)):
        require((row["id"], row["pcm_sha256"]) == (i, observed["pcm_sha256"]) and observed["id"] == i,
                "Unordered or substituted case")
        samples = pcm[i]
        wave = samples / SCALE
        scope, flags, admitted = measured(wave, observed, rule=rule, bands=bands, refs=refs, focus=focus)
        status, calls = direction(observed, calibration, admitted)
        require(status == observed["current"], "Recorded voice direction differs from its calibration")
        forwards += calls
        family, condition, tape = (row[key] for key in ("family", "condition", "tape"))
        if condition == "isolated":
            require(family == "background" and row["noise_sha256"] == row["pcm_sha256"], "Wrong isolated background")
            labels = ["absent"] * len(focus)
        else:
            clean_id = row["clean_id"]
            require(family in FOREGROUNDS and (condition == "clean" or condition in LEVELS)
                    and 0 <= clean_id <= i and roster[clean_id]["condition"] == "clean"
                    and roster[clean_id]["group"] == row["group"], "Missing same-foreground clean reference")
            clean, noise = pcm[clean_id] / SCALE, np.zeros(SAMPLES)
            if condition != "clean":
                noise_sha = row["noise_sha256"]
                require(isinstance(noise_sha, str) and SHA256.fullmatch(noise_sha), "Invalid isolated noise commitment")
                if noise_sha not in known_noises:
                    known_noises[noise_sha] = exact_pcm(folder / "noise" / f"{noise_sha}.pcm", noise_sha) / SCALE
                reconstructed, noise = reconstruct(clean, known_noises[noise_sha], row, bands=bands, center=center, mode=mode)
                require(np.array_equal(reconstructed, samples), "A matched mixture differs at its original samples")
                mixed_count += 1
            labels = truth(clean, noise, bands=bands, focus=focus, rule=rule)
        kind = family if family in {"genuine", "background"} else "synthetic"
        scopes[(kind, condition)][scope] += 1
        outcomes[(kind, condition)][status] += 1
        if any(flags):
            warning_overlap[(kind, condition)][status] += 1
        for label, flagged, (a, b) in zip(labels, flags, focus, strict=True):
            cells[(kind, condition, tape or "clean", label)].update(planned=1, assessed=int(scope == "assessed"), flagged=int(flagged))
            if label == "masked" and flagged:
                hit_groups[kind].add(row["group"])
                evidence.append({"id": i, "family": kind, "tape": tape, "source_pcm_sha256": row["pcm_sha256"],
                                 "clip_pcm_sha256": digest(samples[a:b].tobytes()), "source_sample_span": [a, b],
                                 "ref_sample_spans": refs, "time_base": "1/16000"})
    sums = defaultdict(Counter)
    for (_, _, tape, label), count in cells.items():
        sums[("all", label)].update(count)
        sums[(tape, label)].update(count)
    scene_passes = 0
    for tape in freeze["background_commitments"]:
        masked = sums[(tape, "masked")]
        scene_passes += masked["planned"] > 0 and masked["flagged"] / masked["planned"] >= rule["minimum_masked_recall_per_scene"]
    summary = {key: {"current": count, "overlap_any_coverage_warning": warning_overlap[key]} for key, count in outcomes.items()}
    expected_checks = {"masked_recall": scene_passes >= rule["minimum_scenes_with_recall"],
                       "audible_false_rate": sums[("all", "audible")]["flagged"] / sums[("all", "audible")]["planned"] <= rule["maximum_false_audible_rate"],
                       "absent_false_rate": sums[("all", "absent")]["flagged"] / sums[("all", "absent")]["planned"] <= rule["maximum_false_absent_rate"],
                       "genuine_hit_groups": len(hit_groups["genuine"]) >= rule["minimum_genuine_groups_with_hits"],
                       "synthetic_hit_groups": len(hit_groups["synthetic"]) >= rule["minimum_synthetic_groups_with_hits"]}
    counts = ((cells, "cells", ("family", "condition", "tape", "truth")), (scopes, "scopes", ("family", "condition")),
              (summary, "current_results", ("family", "condition")), (sums, "interval_summary", ("tape", "truth")))
    require(all(value == keyed_counts(aggregate[key], fields) for value, key, fields in counts)
            and expected_checks == aggregate["checks"] and all(expected_checks.values()) == aggregate["advance"]
            and scene_passes == aggregate["scene_passes"] and {key: len(value) for key, value in hit_groups.items()} == aggregate["hit_group_counts"],
            "Grouped interval counts or advancement results differ")
    require(evidence == recorded_evidence and forwards == measured_rows["classifier_forwards"] == aggregate["classifier_forwards"]
            and math.ceil(SAMPLES / 512) * len(roster) == measured_rows["activity_forwards"] == aggregate["activity_forwards"]
            and mixed_count == aggregate["matched_mixtures_reconstructed"], "Clip or model call commitments differ")
    return {"mix_band": mode, "aggregate_sha256": digest(data["aggregate"]), "verified_views": len(roster),
            "verified_sample_bound_intervals": len(focus) * len(roster), "reconstructed_mixtures": mixed_count,
            "matched_clip_commitments": len(evidence), "verified_activity_forwards": aggregate["activity_forwards"],
            "verified_classifier_forwards": forwards, "advance": aggregate["advance"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--published", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.folder, args.published), sort_keys=True))


if __name__ == "__main__":
    main()
