"""Test source-waveform continuity against a same-recording earlier copy."""

from __future__ import annotations

import argparse
import wave
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
from scipy.signal import butter, sosfiltfilt

from scripts import audio_odss_matched_room as audio
from scripts import audio_phoneme_short_edit_intake as intake
from scripts import audio_phonemedf_matched_transfer as phoneme

SCHEMA = "chipmunk.audio-paired-speech-continuity/1"
SIZE, RATE = intake.SPAN, audio.RATE
FRAME, HOP = 3840, 1280
LEVELS = (0, -5, -10)
REFERENCE_LEVEL = 5
ARMS = ("correlation", "projection")
CONTROLS = ("untouched", "local_gain", "ordinary_tts", "ordinary_conversion", "global_eq")
POSITIVES = ("ordinary_replacement", "synthetic_tts", "synthetic_conversion")
PAIRS = {
    "development": ("IS1001b", "ES2012b"),
    "assessment": ("ES2007c", "IS1004c"),
}
AMI = {
    **dict(audio.AMI.values()),
    "IS1004c": "d9c81453ea85a228bf4b82e9757a19edfc760203d1a3af1c353473b3030613b6",
    "ES2007c": "895904c48583d74c311e42cf0bf49554e998cd5ab87b67268c97de09b856e619",
}
METHOD = {
    "input": "Compare the same six-second mono recording on an equal 16 kHz clock using an earlier copy as a separate input. The earlier recording carries +5 dB interfering speech. The submitted recording carries 0, -5, or -10 dB with a different AMI meeting in each role and pair. There is no attempt to align arbitrary recordings or infer a new clock.",
    "features": "Band-pass both inputs at 250–3800 Hz, remove local weighted DC and measure 240 ms Hann-window normalized correlation and signed least-squares reference gain at 80 ms hops. Limit analysis to reference windows at or above 25 percent of its 80th percentile RMS. Normalize each cue by its 70th percentile among those windows. Require reference support in at least 36 windows and broad positive matching; unrelated or same-words re-recorded reference declines.",
    "localization": "Measure a local deficit with the median over each nine-window sequence, requiring at least six assessed cells and coherent two-flank support. Each flank uses five cells separated by one guard cell. Report the 720 ms span centered on the nine centers with strongest deficit. Require strict positive over the development maximum of matched untouched, gain, genuine resampling and ordinary full-recording EQ controls.",
    "roles": "Reuse only the two previous source-panel development prompts for point selection. Select up to three held prompts from source readers not in any of the four earlier short-panel roles. Exclude the prior whole-utterance/ODSS readers and all exact old v9 source IDs. Use the publisher's matching word tier and the earlier source-duration, context and amplitude eligibility constraints. Prior speaker or training exposure is possible. Background captures and participants are disjoint from development.",
    "scope": "A temporary waveform change can result from ordinary editing or synthetic replacement. Count both as physical-change positives, with matched gain, resampling and EQ controls. Unrelated and same-text re-recorded references are admission controls. No model or customer scoring is used.",
}


def rank(value):
    return audio.digest(f"{SCHEMA}:{value}".encode())


def ancestry(args):
    payloads, keys, _, _ = intake.collect(args)
    alignments = intake.grid_inputs(args, keys, payloads)
    earlier, _, _, _, _ = intake.select(args, payloads, keys, alignments)
    whole, _, _ = phoneme.load_inputs(args.old_cache, args.record)
    excluded = {row["source_id"].split("-")[0] for row in whole}
    excluded.update(row["source_key"].split("-")[0] for row in earlier)
    odss, _ = audio.plan(args.odss_cache, False)
    excluded.update(row["speaker"] for row in odss)
    history = audio.ROOT / "backend/data/independent-evaluation/audio-v9/source-plan.json"
    exact = {value.removeprefix("librispeech:") for row in audio.read(history)["sources"]
             for field in ("source_key", "source_group") if (value := row.get(field) or "").startswith("librispeech:")}
    eligible = defaultdict(list)
    for key in sorted(keys, key=rank):
        speaker = key.split("-")[0]
        if speaker in excluded or key in exact or set(alignments[key]) != set(intake.ORIGINS):
            continue
        options = intake.possibilities(alignments[key], payloads, key)
        if options:
            eligible[speaker].append((key, min(options, key=lambda x: rank(f"{key}:{x[1]}:{x[5]}"))))
    audio.require(len(eligible) >= 3, "Need three distinct held source readers")
    rows = [{**row, "case": f"dev-{i}"} for i, row in enumerate(earlier) if row["role"] == "development"]
    for index, speaker in enumerate(sorted(eligible, key=rank)[:3]):
        key, (left, a, b, spans, _, count) = eligible[speaker][0]
        row = {"case": f"held-{index}", "role": "assessment", "source_key": key,
               "source_crop": [left, left + SIZE], "edit_range": [a, b], "word_count": count,
               "source_bindings": {origin: {"native_word_span": span, "pcm_sha256": payloads[origin][key][1],
                                           "alignment_sha256": alignments[key][origin][1]} for origin, span in spans.items()}}
        rows.append(row)
    return payloads, rows


def maskers(args, roster):
    results, pins = {}, {}
    annotation = args.ami / "ami_public_manual_1.6.2.zip"
    audio.require(audio.file_digest(annotation) == audio.ami.ANNOTATION_SHA256, "Pinned AMI annotations differ")
    with ZipFile(annotation) as archive:
        document = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        people = {node.get("observation"): {p.get("global_name") for p in node if p.get("global_name")}
                  for node in document.iter() if node.get("observation")}
        used_people = set()
        for role, pair in PAIRS.items():
            for index, meeting in enumerate(pair):
                audio.require(len(people[meeting]) == 4 and not used_people & people[meeting], "Participant roles overlap")
                used_people |= people[meeting]
                path = (args.ami if role == "development" else args.fresh_ami) / f"{meeting}.Array1-01.wav"
                audio.require(audio.file_digest(path) == AMI[meeting], "Pinned AMI capture differs")
                tracks = [audio.ami.track_words(archive.read(f"words/{meeting}.{channel}.words.xml")) for channel in "ABCD"]
                length = max(b for track in tracks for _, b in track)
                activity = np.zeros((4, length), dtype=bool)
                for row, track in zip(activity, tracks, strict=True):
                    for a, b in track:
                        row[a:b] = True
                active = activity.sum(axis=0)
                reserved = []
                with wave.open(str(path), "rb") as stream:
                    audio.require(stream.getparams()[:3] == (1, 2, RATE), "Native 16 kHz AMI PCM required")
                    for row in roster:
                        if row["role"] != role:
                            continue
                        case = row["case"]
                        a, b = row["edit_range"]
                        candidates = []
                        for tick in range(500, min(length, stream.getnframes() // 160) - 600, 25):
                            if any(abs(tick - other) < 600 for other in reserved):
                                continue
                            clip = active[tick:tick + 600]
                            overlap = int(np.count_nonzero(clip[(a + 159)//160:b//160] >= 2))
                            if overlap >= 25 and np.count_nonzero(clip) >= 450:
                                candidates.append((tick, overlap))
                        audio.require(candidates, "No annotated independent masker for the paired interval")
                        tick, overlap = min(candidates, key=lambda x: rank(f"{case}:{meeting}:{x[0]}"))
                        reserved.append(tick)
                        stream.setpos(tick * 160)
                        blob = stream.readframes(SIZE)
                        audio.require(len(blob) == 2 * SIZE, "AMI masker slice truncated")
                        results[case, index] = np.frombuffer(blob, dtype="<i2").astype(np.float64)
                        pins[f"{case}:{index}"] = {"capture": meeting, "start_sample": tick * 160, "overlap_cs": overlap,
                                                   "pcm_sha256": audio.digest(blob)}
    return results, pins


def dry_variants(row, payloads):
    key = row["source_key"]
    left, right = row["source_crop"]
    a, b = row["edit_range"]
    original = payloads["real"][key][2][left:right].astype(np.float64)
    segment = original[a:b]
    donor_position = (b + 800 if b + 800 + b - a <= SIZE else a - 800 - (b - a))
    audio.require(0 <= donor_position < SIZE - len(segment), "Missing ordinary replacement segment")
    others = {"ordinary_replacement": original[donor_position:donor_position + len(segment)],
              "local_gain": segment * 1.4}
    fake_full = payloads["ChatterboxTTS"][key][2].astype(np.float64)
    for family, origin in zip(intake.FAMILIES, intake.ORIGINS[1:], strict=True):
        c, d = row["source_bindings"][origin]["native_word_span"]
        donor = payloads[origin][key][2][c:d].astype(np.float64)
        others[f"synthetic_{family}"] = intake.resample(donor, b - a)
        others[f"ordinary_{family}"] = intake.resample(intake.resample(segment, len(donor)), b - a)
    variants = {"untouched": original}
    for kind, donor in others.items():
        if kind != "local_gain":
            donor = donor * audio.rms(segment) / audio.rms(donor)
        variants[kind] = intake.replace(original, donor, a, b)
    equalized = sosfiltfilt(butter(2, 2450, fs=RATE, btype="lowpass", output="sos"), original)
    variants["global_eq"] = equalized * audio.rms(original) / audio.rms(equalized)
    return variants, fake_full[:SIZE] if len(fake_full) >= SIZE else np.pad(fake_full, (0, SIZE - len(fake_full)))


def quantized(signal):
    values = np.rint(signal)
    audio.require(np.max(abs(values)) <= 24001, "Mix clipped")
    return values.astype("<i2")


def normed(x, target, level=0):
    return x * audio.rms(target) / audio.rms(x) * 10 ** (-level / 20)


def prepare(args):
    payloads, roster = ancestry(args)
    masks, records = maskers(args, roster)
    dry = {row["case"]: dry_variants(row, payloads) for row in roster}
    rows, clips = [], []
    for row in roster:
        case, role = row["case"], row["role"]
        variants, rerecorded = dry[case]
        original = variants["untouched"]
        other = next(other for other in roster if other["role"] == role and other["case"] != case)
        unrelated = dry[other["case"]][0]["untouched"]
        first_noise = normed(masks[case, 0], original, REFERENCE_LEVEL)
        second_noise = normed(masks[case, 1], original, min(LEVELS))
        earlier = {"same": original, "different_reader_reference": normed(unrelated, original),
                   "same_words_rerecorded_reference": normed(rerecorded, original)}
        peak = max(np.max(abs(x) + abs(first_noise)) for x in earlier.values())
        peak = max(peak, *(np.max(abs(x) + abs(second_noise)) for x in variants.values()))
        scale = 24000 / peak
        first = {kind: quantized((x + first_noise) * scale) for kind, x in earlier.items()}
        for level in LEVELS:
            noise = second_noise * 10 ** (-(level - min(LEVELS)) / 20)
            measured = float(20 * np.log10(audio.rms(quantized(original * scale).astype(np.float64))
                                         / audio.rms(quantized(noise * scale).astype(np.float64))))
            audio.require(abs(measured - level) < .02, "Submitted mixing level differs")
            for kind, query in variants.items():
                after = quantized((query + noise) * scale)
                before = first["same"]
                truth = row["edit_range"] if kind in POSITIVES else None
                clips.append((before, after))
                rows.append({"case": case, "role": role, "level": level, "kind": kind, "truth": truth,
                             "earlier_sha256": audio.digest(before.tobytes()), "submitted_sha256": audio.digest(after.tobytes()),
                             "measured_snr_db": round(measured, 6)})
            for kind in ("different_reader_reference", "same_words_rerecorded_reference"):
                after = quantized((original + noise) * scale)
                clips.append((first[kind], after))
                rows.append({"case": case, "role": role, "level": level, "kind": kind, "truth": None,
                             "earlier_sha256": audio.digest(first[kind].tobytes()),
                             "submitted_sha256": audio.digest(after.tobytes()), "measured_snr_db": round(measured, 6)})
    audio.require(len(rows) == 150, "Exact pair matrix differs")
    lock = {"schema": SCHEMA, "method": METHOD, "clock_hz": RATE, "reference_level_db": REFERENCE_LEVEL,
            "submission_levels_db": LEVELS, "code_sha256": audio.file_digest(Path(__file__)),
            "source_roster_sha256": audio.digest(audio.json_bytes(roster)), "masker_roster_sha256": audio.digest(audio.json_bytes(records)),
            "masker_archive_sha256": AMI, "masker_annotation_sha256": audio.ami.ANNOTATION_SHA256,
            "delivery_roster_sha256": audio.digest(audio.json_bytes(rows)), "native_source_intake_sha256": audio.file_digest(args.intake)}
    return lock, rows, clips


def profiles(earlier, submitted):
    sos = butter(2, (250, 3800), btype="bandpass", fs=RATE, output="sos")
    weight = np.hanning(FRAME)
    windows = []
    for pcm in (earlier, submitted):
        filtered = sosfiltfilt(sos, pcm.astype(np.float64))
        frames = np.lib.stride_tricks.sliding_window_view(filtered, FRAME)[::HOP]
        frames = frames - (frames @ weight / weight.sum())[:, None]
        windows.append(frames * np.sqrt(weight))
    before, after = windows
    ref = np.einsum("ij,ij->i", before, before)
    sub = np.einsum("ij,ij->i", after, after)
    joint = np.einsum("ij,ij->i", before, after)
    rms = np.sqrt(ref / weight.sum())
    active = (rms >= max(12, float(np.quantile(rms, .8)) * .25)) & (sub > 0)
    corr = joint / np.sqrt(np.maximum(ref * sub, 1e-10))
    gain = joint / np.maximum(ref, 1e-10)
    supported = bool(np.count_nonzero(active) >= 36 and np.count_nonzero(active & (corr >= .22)) >= 18
                     and np.quantile(corr[active], .7) >= .30)
    if not supported:
        return None, np.count_nonzero(active)
    response = {}
    for arm, values in (("correlation", corr), ("projection", gain)):
        normal = float(np.quantile(values[active], .7))
        if normal <= .05:
            response[arm] = None
            continue
        response[arm] = np.where(active, values / normal, np.nan)
    return response, int(np.count_nonzero(active))


def localize(values):
    if values is None:
        return None
    best = None
    for start in range(6, len(values) - 14):
        stop = start + 9
        sections = [part[np.isfinite(part)] for part in (values[start - 6:start - 1], values[start:stop], values[stop + 1:stop + 6])]
        if any(len(part) < count for part, count in zip(sections, (3, 6, 3), strict=True)):
            continue
        left, middle, right = map(float, map(np.median, sections))
        if min(left, right) < .55:
            continue
        value = round(min(left, right) - middle - .25 * abs(left - right), 8)
        if best is None or value > best["score"]:
            a = start * HOP + FRAME // 2 - HOP // 2
            best = {"score": value, "range": [a, a + 9 * HOP], "flanks": [round(left, 8), round(right, 8)],
                    "middle": round(middle, 8)}
    return best


def measure(rows, clips, role):
    measured = []
    for row, (first, second) in zip(rows, clips, strict=True):
        if row["role"] != role:
            continue
        views, active = profiles(first, second)
        diff = np.any((first != second).reshape(-1, 320), axis=1)
        boundaries = np.flatnonzero(np.diff(np.r_[False, diff, False]))
        baseline = [int(boundaries[0]) * 320, int(boundaries[-1]) * 320] if boundaries.size else None
        measured.append({**row, "supported_pair": views is not None, "active_windows": int(active),
                         "results": {arm: None if views is None else localize(views[arm]) for arm in ARMS},
                         "exact_comparator_extent": baseline, "exact_comparator_changed_bins": int(diff.sum())})
    return measured


def points(rows):
    return {arm: max(row["results"][arm]["score"] for row in rows
                     if row["kind"] in CONTROLS and row["results"][arm] is not None) for arm in ARMS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "score", "replay"))
    for arg in ("cache", "old-cache", "record", "odss-cache", "ami", "fresh-ami", "intake", "run"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    lock, rows, clips = prepare(args)
    freeze, plan, dev, point, held = (args.run / name for name in ("freeze.json", "plan.json", "development.json", "points.json", "public.json"))
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        audio.write(freeze, lock)
        audio.write(plan, rows)
        print(f"Froze {len(rows)} distinct-condition paired deliveries before measurement")
        return
    audio.require(freeze.read_bytes() == audio.json_bytes(lock) and plan.read_bytes() == audio.json_bytes(rows), "Frozen pair inputs changed")
    if args.phase == "fit":
        fit = measure(rows, clips, "development")
        audio.write(dev, fit)
        audio.write(point, {"freeze_sha256": audio.file_digest(freeze), "development_sha256": audio.file_digest(dev), "points": points(fit)})
        print(f"Sealed both points on {len(fit)} development pairs")
        return
    fit, cutoffs = audio.read(dev), audio.read(point)
    audio.require(cutoffs == {"freeze_sha256": audio.file_digest(freeze), "development_sha256": audio.file_digest(dev),
                             "points": points(fit)}, "Development-only points differ")
    assessment = measure(rows, clips, "assessment")
    result = {"freeze": lock, "development": fit, "operating_points": cutoffs, "assessment": assessment}
    if args.phase == "replay":
        audio.require(held.read_bytes() == audio.json_bytes(result) and fit == measure(rows, clips, "development"),
                      "Offline pair measurement replay differs")
        print(f"Replayed all {len(rows)} pair waveforms and outcomes")
        return
    audio.write(held, result)
    print(f"Scored {len(assessment)} held pairs")


if __name__ == "__main__":
    main()
