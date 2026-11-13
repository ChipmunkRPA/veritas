"""Freeze short same-word replacement test inputs with exact publisher clock bindings."""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.signal import resample_poly

from scripts import audio_odss_matched_room as audio
from scripts import audio_phonemedf_matched_transfer as prior
from scripts import replay_audio_odss_public as replay

SCHEMA = "chipmunk.audio-phoneme-short-edit/1"
TEXTGRID_BYTES = 386_093_119
TEXTGRID_MD5 = "dad7c124775cd1be4f4187282d8ed1e1"
VOICE_BYTES = 9_622_565_627
ORIGINS = ("real", "ChatterboxTTS", "ChatterboxVC")
FAMILIES = ("tts", "conversion")
PREVIOUS_PREFIX = 96 * 1024**2
PREFIX = 320 * 1024**2
SPAN = 6 * audio.RATE
LEVELS = (None, 5, 0, -5)
FADE = 320
METHOD = {
    "alignment": "Use publisher word intervals on native 16 kHz clocks. Require all three origins to have identical full normalized word sequences and choose a two- or three-word genuine interval of 0.64–1.02 seconds with duration ratio within 25 percent in both paired sources. Resample each donor span to the exact genuine span and report the native and resampled hashes.",
    "control": "The genuine timing control time-stretches the exact host phrase through the same per-family natural-to-donor and back-to-host rates used in the synthetic branch. The gain control scales the same genuine words by 1.4 with identical fades. The host and each edit version share one competing speech excerpt, one gain and exact samples outside the authored interval. Include unedited and masker-only controls.",
    "roles": "Choose two held source readers and two development source readers, with one prompt each, using only published alignment, integrity, lengths, word coverage, PCM activity and hashed ordering. Exclude all four prior PhonemeDF whole-utterance readers, the ODSS paired Hi-Fi readers, and every exact source identifier in the saved serving data. Report the stricter reader-exclusion census separately. Historical serving reader exposure can recur. Generator voices may share publisher reference speakers.",
    "delivery": "Crop the genuine utterance to six seconds with at least 1.3 seconds of context on each side of the replacement. Mix mono with independently annotated AMI competing speech. Verify that competing speech overlaps at least 200 ms of the edit interval. Mix at +5, 0 and −5 dB plus dry reference, with one shared peak-safe gain. This is controlled mono overlap with real recorded interfering speech, without a simulated or measured transfer function.",
}


def rank(value):
    return audio.digest(f"{SCHEMA}:{value}".encode())


def archive(cache, flavor, total, end, old=None, *, complete_md5=None):
    chunks = []
    digest = hashlib.md5(usedforsecurity=False) if complete_md5 else None
    for offset in range(0, end, prior.CHUNK):
        length = min(end - offset, prior.CHUNK)
        source = old if old and offset < PREVIOUS_PREFIX else cache
        path = source / flavor / f"{offset}.range"
        header = path.with_suffix(".headers") if source == old else path.with_suffix(".range.headers")
        expected = f"content-range: bytes {offset}-{offset + length - 1}/{total}"
        audio.require(path.stat().st_size == length and expected in header.read_text().casefold().splitlines(),
                      "Publisher byte range differs")
        if digest:
            digest.update(path.read_bytes())
        chunks.append(path)
    if complete_md5:
        audio.require(end == total and digest.hexdigest() == complete_md5, "Complete publisher TextGrid archive differs")
    return chunks


def collect(args):
    record = audio.read(args.record)
    release = {row["key"]: row["size"] for row in record["files"]}
    audio.require(record["id"] == prior.SOURCE_RECORD and record["metadata"]["license"]["id"] == "cc-by-4.0",
                  "Expected exact commercial-compatible publisher release")
    old, current = args.old_cache, args.cache
    payloads, inputs = {}, {}
    known = {"real": prior.ARCHIVES["natural"][1], "ChatterboxTTS": prior.ARCHIVES["chatterbox"][1], "ChatterboxVC": VOICE_BYTES}
    earliest = {}
    for origin in ORIGINS:
        size = known[origin]
        audio.require(release[f"PhonemeDF_{origin}.tar.gz"] == size, "Publisher archive identity differs")
        paths = archive(current, origin, size, PREFIX, old if origin != "ChatterboxVC" else None)
        inputs[origin] = [audio.file_digest(path) for path in paths]
        records = {}
        folder = "real/wav" if origin == "real" else origin
        for name, body in prior.archive_members(paths):
            match = re.fullmatch(rf"{folder}/(\d+-\d+-\d+)\.wav", name)
            if match:
                key, = match.groups()
                audio.require(key not in records, "Duplicate archive utterance")
                pcm = prior.pcm_bytes(body)
                records[key] = (audio.digest(body), audio.digest(pcm), np.frombuffer(pcm, dtype="<i2"))
        payloads[origin] = records
        earliest[origin] = len(records)
    available = set.intersection(*(set(value) for value in payloads.values()))
    return payloads, available, inputs, earliest


WORD = re.compile(r'intervals \[\d+\]:\s+xmin = ([0-9.]+)\s+xmax = ([0-9.]+)\s+text = "([^"]*)"')


def words(grid, samples):
    section = grid.decode("utf-8-sig").split("item [2]:", 1)[0]
    audio.require('name = "words"' in section, "Publisher TextGrid lacks its first word interval tier")
    labels = [(round(float(a) * audio.RATE), round(float(b) * audio.RATE), token.casefold())
              for a, b, token in WORD.findall(section) if token.strip()]
    if not labels or any(not 0 <= a < b <= samples for a, b, _ in labels):
        return None
    return labels


def grid_inputs(args, keys, payloads):
    paths = archive(args.cache, "textgrids", TEXTGRID_BYTES, TEXTGRID_BYTES, complete_md5=TEXTGRID_MD5)
    result = defaultdict(dict)
    for name, data in prior.archive_members(paths):
        match = re.fullmatch(r"textgrids/(real|ChatterboxTTS|ChatterboxVC)/(\d+-\d+-\d+)\.TextGrid", name)
        if not match:
            continue
        origin, key = match.groups()
        if key not in keys:
            continue
        native = payloads[origin][key][2]
        labels = words(data, len(native))
        if labels:
            result[key][origin] = (labels, audio.digest(data))
    return result


def possibilities(alignment, payloads, key):
    real = alignment["real"][0]
    sequence = [row[2] for row in real]
    if any([row[2] for row in labels] != sequence for labels, _ in alignment.values()):
        return []
    out = []
    genuine = payloads["real"][key][2]
    if len(genuine) < SPAN:
        return out
    for start in range(1, len(real) - 3):
        for count in (2, 3):
            spans = {origin: (labels[start][0], labels[start + count - 1][1]) for origin, (labels, _) in alignment.items()}
            a, b = spans["real"]
            width = b - a
            if not 10240 <= width <= 16320:
                continue
            if any(not .80 <= (hi - lo) / width <= 1.25 for lo, hi in spans.values()):
                continue
            left = max(0, min(len(genuine) - SPAN, (a + b - SPAN) // 2))
            if not left + 20800 <= a < b <= left + SPAN - 20800:
                continue
            if any(audio.rms(payloads[origin][key][2][lo:hi].astype(np.float64)) < 140
                   for origin, (lo, hi) in spans.items()):
                continue
            out.append((left, a - left, b - left, spans, start, count))
    return out


def select(args, payloads, keys, alignments):
    cache = args.old_cache
    previous, _, _ = prior.load_inputs(cache, args.record)
    readers = {item["source_id"].split("-")[0] for item in previous}
    old_roster, _ = audio.plan(args.odss_cache, False)
    readers.update(row["speaker"] for row in old_roster)
    v9 = audio.ROOT / "backend/data/independent-evaluation/audio-v9/source-plan.json"
    records = audio.read(v9)["sources"]
    historical = {speaker.split(":")[-1] for row in records
                  if (speaker := (row.get("canonical_speaker") or row.get("speaker") or "")).startswith("librispeech:")}
    exact_ancestry = {key.removeprefix("librispeech:") for row in records
                      for field in ("source_key", "source_group")
                      if (key := (row.get(field) or "")).startswith("librispeech:")}
    eligible = defaultdict(list)
    for key in sorted(keys, key=rank):
        speaker = key.split("-")[0]
        if speaker in readers or key in exact_ancestry or set(alignments[key]) != set(ORIGINS):
            continue
        if options := possibilities(alignments[key], payloads, key):
            option = min(options, key=lambda value: rank(f"{key}:{value[1]}:{value[5]}"))
            eligible[speaker].append((key, option))
    ranked = sorted(eligible, key=lambda speaker: (speaker not in historical, rank(speaker)))
    # Prioritize wholly unexposed readers in assessment when the prefix supplies any.
    ranked = ranked[:2] + sorted(ranked[2:], key=lambda speaker: (speaker in historical, rank(speaker)))
    audio.require(len(ranked) >= 4, f"Fewer than four fresh declared speaker groups meet all criteria: {len(ranked)}")
    chosen = []
    for index, speaker in enumerate(ranked[:4]):
        key, (left, a, b, spans, word_start, word_count) = eligible[speaker][0]
        role = "development" if index < 2 else "assessment"
        sources = {origin: {"wav_sha256": payloads[origin][key][0], "pcm_sha256": payloads[origin][key][1],
                            "grid_sha256": alignments[key][origin][1], "samples": len(payloads[origin][key][2]),
                            "native_word_span": spans[origin]} for origin in ORIGINS}
        labels = [row[2] for row in alignments[key]["real"][0][word_start:word_start + word_count]]
        chosen.append({"case": f"phoneme-reader-{index:02d}-prompt-0", "group": f"phoneme-reader-{index:02d}", "role": role, "source_key": key,
                       "words_sha256": audio.digest(" ".join(labels).encode()), "word_count": word_count,
                       "source_crop": [left, left + SPAN], "edit_range": [a, b], "source_bindings": sources})
    return chosen, len(eligible), len(set(eligible) - historical), len(readers), audio.file_digest(v9)


def replace(host, donor, a, b):
    width = b - a
    audio.require(len(donor) == width, "Exact replacement clock required")
    weights = np.r_[np.linspace(0, 1, FADE), np.ones(width - 2 * FADE), np.linspace(1, 0, FADE)]
    result = host.copy()
    result[a:b] = (1 - weights) * host[a:b] + weights * donor
    return result


def resample(signal, width):
    import math

    scale = math.gcd(len(signal), width)
    out = resample_poly(signal, width // scale, len(signal) // scale)
    audio.require(len(out) == width, "Alignment transform changed the destination clock")
    return out


def render(source, payloads, masker):
    key = source["source_key"]
    left, right = source["source_crop"]
    a, b = source["edit_range"]
    host = payloads["real"][key][2][left:right].astype(np.float64)
    original = host[a:b]
    noise = masker.astype(np.float64) * audio.rms(host) / audio.rms(masker.astype(np.float64))
    variants = {"untouched": host, "local_gain": replace(host, original * 1.4, a, b)}
    for family, origin in zip(FAMILIES, ORIGINS[1:], strict=True):
        x, y = source["source_bindings"][origin]["native_word_span"]
        fragment = payloads[origin][key][2][x:y].astype(np.float64)
        converted = resample(fragment, b - a)
        converted *= audio.rms(original) / audio.rms(converted)
        ordinary = resample(resample(original, len(fragment)), b - a)
        ordinary *= audio.rms(original) / audio.rms(ordinary)
        variants[f"ordinary_{family}"] = replace(host, ordinary, a, b)
        variants[f"synthetic_{family}"] = replace(host, converted, a, b)
    peak = max(np.max(abs(signal) + abs(noise) * 10 ** .25) for signal in variants.values())
    common = 24000 / peak
    signals = {kind: np.rint(signal * common).astype(np.int32) for kind, signal in variants.items()}
    for level in LEVELS:
        masker_pcm = None if level is None else np.rint(noise * common * 10 ** (-level / 20)).astype(np.int32)
        values = signals if level is None else {**{kind: signal + masker_pcm for kind, signal in signals.items()},
                                                "background": masker_pcm}
        measured = None if level is None else 20 * np.log10(audio.rms(signals["untouched"]) / audio.rms(masker_pcm))
        audio.require(measured is None or abs(measured - level) < .02, "Target/background mixture differs")
        for kind, signal in values.items():
            if kind != "background":
                audio.require(np.array_equal(signal[:a], values["untouched"][:a])
                              and np.array_equal(signal[b:], values["untouched"][b:]), "Source edit escaped its exact interval")
            audio.require(np.max(abs(signal)) <= 24001, "Signed delivery clips")
            yield level, kind, signal.astype("<i2"), measured


def _with_edit_overlap(annotation, ami_dir, roster, reserved):
    import wave

    selected, pcm = {}, {}
    with ZipFile(annotation) as archive:
        for role, (meeting, pin) in audio.AMI.items():
            tracks = [audio.ami.track_words(archive.read(f"words/{meeting}.{channel}.words.xml")) for channel in "ABCD"]
            count = max(b for track in tracks for _, b in track)
            voices = np.zeros((4, count), dtype=bool)
            for index, intervals in enumerate(tracks):
                for a, b in intervals:
                    voices[index, a:b] = True
            active = voices.sum(axis=0)
            taken = [row["start_sample"] // 160 for row in reserved.values() if row["meeting"] == meeting]
            original = ami_dir / f"{meeting}.Array1-01.wav"
            audio.require(audio.file_digest(original) == pin, "AMI parent differs")
            with wave.open(str(original), "rb") as stream:
                audio.require(stream.getparams()[:3] == (1, 2, audio.RATE), "Unexpected AMI PCM profile")
                for row in roster:
                    if row["role"] != role or "edit_range" not in row:
                        continue
                    start, stop = row["edit_range"]
                    candidates = []
                    for tick in range(500, min(count, stream.getnframes() // 160) - 600, 25):
                        if any(abs(tick - old) < 600 for old in taken):
                            continue
                        overlap = int(np.count_nonzero(active[tick + (start + 159)//160:tick + stop//160] >= 2))
                        if overlap >= 20 and np.count_nonzero(active[tick:tick + 600]) >= 450:
                            candidates.append((tick, overlap))
                    audio.require(candidates, "No disjoint annotated speech window covers the edit")
                    tick, overlap = min(candidates, key=lambda value: rank(f"{row['case']}:{value[0]}"))
                    taken.append(tick)
                    stream.setpos(tick * 160)
                    data = stream.readframes(SPAN)
                    audio.require(len(data) == 2 * SPAN, "Background clip is truncated")
                    case = row["case"]
                    selected[case] = {"meeting": meeting, "start_sample": tick * 160, "edited_overlap_centiseconds": overlap,
                                      "pcm_sha256": audio.digest(data)}
                    pcm[case] = np.frombuffer(data, dtype="<i2")
    return selected, pcm


def prepare(args):
    payloads, keys, input_ranges, origins = collect(args)
    alignments = grid_inputs(args, keys, payloads)
    roster, eligible, strictly_fresh, excluded, v9_sha = select(args, payloads, keys, alignments)
    earlier, _ = audio.plan(args.odss_cache, False)
    annotation = args.ami / "ami_public_manual_1.6.2.zip"
    old_backgrounds = replay.manifest(annotation, earlier)
    backgrounds, maskers = _with_edit_overlap(annotation, args.ami, roster, old_backgrounds)
    retained = []
    for source in roster:
        case = source["case"]
        a, b = source["edit_range"]
        for level, kind, pcm, measured in render(source, payloads, maskers[case]):
            pin = audio.digest(pcm.tobytes())
            retained.append({"case": case, "role": source["role"], "level": level, "kind": kind,
                             "source_edit_range": [a, b] if kind not in ("background", "untouched") else None,
                             "clock_hz": audio.RATE, "pcm_sha256": pin,
                             "measured_snr_db": None if level is None or kind == "background" else round(float(measured), 6)})
    audio.require(len(retained) == 108, "Unexpected frozen delivery count")
    receipt = {"schema": SCHEMA, "method": METHOD, "release_record_sha256": audio.file_digest(args.record),
               "runner_sha256": audio.file_digest(Path(__file__)), "bounded_wav_ranges": input_ranges,
               "common_utterance_ids": len(keys), "complete_member_counts": origins,
               "eligible_distinct_speakers": eligible, "eligible_speakers_outside_all_saved_v9_roles": strictly_fresh,
               "declared_speaker_exclusions": excluded, "v9_source_plan_sha256": v9_sha,
               "background_edit_overlap_centiseconds": sorted(item["edited_overlap_centiseconds"] for item in backgrounds.values()),
               "textgrid_archive_md5": TEXTGRID_MD5, "source_roster_sha256": audio.digest(audio.json_bytes(roster)),
               "background_roster_sha256": audio.digest(audio.json_bytes(backgrounds)), "deliveries": retained,
               "voice_classifier_forwards": 0}
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "replay"))
    for arg in ("cache", "old-cache", "record", "odss-cache", "ami", "receipt"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    outcome = prepare(args)
    if args.phase == "freeze":
        audio.write(args.receipt, outcome)
    else:
        audio.require(args.receipt.read_bytes() == audio.json_bytes(outcome), "Fresh exact publisher-source replay differs")
    print(f"{args.phase}: {len(outcome['deliveries'])} exact deliveries; {outcome['eligible_distinct_speakers']} eligible source readers; zero model forwards")


if __name__ == "__main__":
    main()
