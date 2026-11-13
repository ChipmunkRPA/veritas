"""Compare a clean earlier voice recording against noisy same-clock copies."""

from __future__ import annotations

import argparse
import wave
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.signal import butter, sosfiltfilt

from scripts import audio_odss_matched_room as audio
from scripts import audio_paired_speech_continuity_transfer as pair
from scripts import audio_phoneme_short_edit_intake as intake
from scripts import audio_phonemedf_matched_transfer as phoneme

SCHEMA = "chipmunk.audio-clean-reference-transfer/1"
LEVELS = (0, -5, -10)
RANGE = 11_520
RIG = {"development": ("ES2004a", "IS1009a"), "assessment": ("ES2005a", "ES2008a")}
WIDTH, HOP = 3840, 1280
PRIOR = audio.ROOT / "docs/evaluation-panels/audio-paired-speech-continuity-transfer-2026-11-13.json"
ARMS = ("prior_correlation", "coherent_projection")
METHOD = {
    "input": "A clean earlier copy of the same continuous six-second mono recording on the same 16 kHz sample clock. Test 0, -5 and -10 dB submission mixes with role-disjoint real competing speech. Include different-reader and same-words regenerated reference admission controls.",
    "controls": "Separate no-edit, local gain, same-speech resampling and global EQ controls. Ordinary different-performance, TTS and voice-converted same-word replacements are physical-change positives. The earlier copy is untouched in all positives.",
    "candidate": "A coherent match uses 240 ms Hann frames, 80 ms hops and three independent filtered bands from 300 to 3400 Hz. Evaluate local retained gain relative to both supported flanks. Reference-energy weighting reduces quiet frames and floor limits the coefficient extrapolation. Cross-band agreement uses the median of available signed deficits.",
    "baseline": "Compare with the earlier normalized-correlation cue and exact 20 ms PCM differences using the identical clean reference. The noisy-reference historical study is a separate experiment.",
    "thresholds": "Use the strict maximum of each route on all ordinary development controls at all noise levels; seal before assessment. A localized range covers at least half the changed interval and reaches 0.30 interval IoU.",
    "promotion": "Continue only if the candidate locates at least four of six synthetic deliveries at -5 dB with at most one ordinary control flag, two reader groups covered, and zero flags on unrelated reference controls. No customer confidence weight.",
}


def rank(key):
    return audio.digest(f"{SCHEMA}:{key}".encode())


def select(args, old_payloads, earlier):
    record = audio.read(args.record)
    sizes = {row["key"]: row["size"] for row in record["files"]}
    extent = 448 * 1024**2
    payloads, range_pins = {}, {}
    for origin in intake.ORIGINS:
        prefix = intake.archive(args.cache, origin, sizes[f"PhonemeDF_{origin}.tar.gz"], extent,
                                args.old_cache if origin != "ChatterboxVC" else None)
        members, folder = {}, "real/wav" if origin == "real" else origin
        for name, data in phoneme.archive_members(prefix):
            if name.startswith(f"{folder}/") and name.endswith(".wav"):
                pcm = phoneme.pcm_bytes(data)
                members[Path(name).stem] = (audio.digest(data), audio.digest(pcm), np.frombuffer(pcm, dtype="<i2"))
        payloads[origin] = members
        range_pins[origin] = [audio.file_digest(path) for path in prefix]
    keys = set.intersection(*(set(members) for members in payloads.values()))
    alignments = intake.grid_inputs(args, keys, payloads)
    whole, _, _ = phoneme.load_inputs(args.old_cache, args.record)
    short_old, _, _, _, _ = intake.select(args, old_payloads, set.intersection(
        *(set(values) for values in old_payloads.values())), intake.grid_inputs(
            args, set.intersection(*(set(values) for values in old_payloads.values())), old_payloads))
    excluded = {r["source_id"].split("-")[0] for r in whole} | {
        r["source_key"].split("-")[0] for r in earlier + short_old}
    history = audio.ROOT / "backend/data/independent-evaluation/audio-v9/source-plan.json"
    exact = {value.removeprefix("librispeech:") for row in audio.read(history)["sources"]
             for field in ("source_key", "source_group") if (value := row.get(field) or "").startswith("librispeech:")}
    eligible = defaultdict(list)
    for key in sorted(keys, key=rank):
        speaker = key.split("-")[0]
        if speaker not in excluded and key not in exact and set(alignments[key]) == set(intake.ORIGINS):
            options = intake.possibilities(alignments[key], payloads, key)
            if options:
                eligible[speaker].append((key, min(options, key=lambda x: rank(f"{key}:{x[1]}:{x[5]}"))))
    audio.require(len(eligible) >= 3, f"Only {len(eligible)} new declared reader groups qualify")
    roster = [row for row in earlier if row["role"] == "development"]
    for index, speaker in enumerate(sorted(eligible, key=rank)[:3]):
        key, (offset, a, b, spans, _, words) = eligible[speaker][0]
        roster.append({"case": f"held-{index}", "role": "assessment", "source_key": key, "edit_range": [a, b],
                       "source_crop": [offset, offset + pair.SIZE], "word_count": words,
                       "source_bindings": {origin: {"native_word_span": span, "pcm_sha256": payloads[origin][key][1],
                                                    "grid_sha256": alignments[key][origin][1]} for origin, span in spans.items()}})
    return payloads, roster, range_pins


def backgrounds(args, rows):
    public = audio.read(audio.ROOT / "docs/evaluation-panels/audio-gain-public-replay-2026-11-13.json")
    annotation = args.ami / "ami_public_manual_1.6.2.zip"
    audio.require(audio.file_digest(annotation) == audio.ami.ANNOTATION_SHA256, "AMI annotations differ")
    result, pins = {}, {}
    with ZipFile(annotation) as archive:
        from xml.etree import ElementTree
        meetings = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        people = {node.get("observation"): {child.get("global_name") for child in node if child.get("global_name")}
                  for node in meetings.iter() if node.get("observation")}
        used_people = set()
        for role, names in RIG.items():
            for index, row in enumerate(item for item in rows if item["role"] == role):
                meeting = names[index % len(names)]
                # Multiple prompts from the same meeting are allowed inside one role.
                if meeting not in pins:
                    audio.require(len(people[meeting]) == 4 and not used_people & people[meeting], "Competing-speech roles overlap")
                    used_people |= people[meeting]
                    name = f"{meeting}.Array1-01.wav"
                    path = args.gain_ami / name
                    pin = public["sources"][name]["sha256"]
                    audio.require(audio.file_digest(path) == pin, "AMI source changed")
                    pins[meeting] = {"sha256": pin, "excerpts": []}
                name = f"{meeting}.Array1-01.wav"
                path = args.gain_ami / name
                tracks = [audio.ami.track_words(archive.read(f"words/{meeting}.{letter}.words.xml")) for letter in "ABCD"]
                n = max(b for track in tracks for _, b in track)
                occupied = np.zeros((4, n), dtype=bool)
                for intervals, track in zip(tracks, occupied, strict=True):
                    for a, b in intervals:
                        track[a:b] = True
                active = occupied.sum(axis=0)
                start, stop = row["edit_range"]
                reserved = [item["start"] // 160 for item in pins[meeting]["excerpts"]]
                with wave.open(str(path), "rb") as stream:
                    audio.require(stream.getparams()[:3] == (1, 2, pair.RATE), "Unexpected AMI native format")
                    possible = []
                    for t in range(500, min(n, stream.getnframes() // 160) - 600, 25):
                        if any(abs(t - old) < 600 for old in reserved):
                            continue
                        content = active[t:t+600]
                        overlap = int(np.count_nonzero(content[(start+159)//160:stop//160] >= 2))
                        if overlap >= 25 and np.count_nonzero(content) >= 440:
                            possible.append((t, overlap))
                    audio.require(possible, "Insufficient recorded overlapping speech")
                    tick, overlap = min(possible, key=lambda cell: rank(f"{row['case']}:{meeting}:{cell[0]}"))
                    stream.setpos(tick * 160)
                    pcm = stream.readframes(pair.SIZE)
                    audio.require(len(pcm) == 2 * pair.SIZE, "Truncated AMI sample interval")
                result[row["case"]] = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
                pins[meeting]["excerpts"].append({"case": row["case"], "start": tick * 160, "overlap_centiseconds": overlap,
                                                    "pcm_sha256": audio.digest(pcm)})
    return result, pins


def inputs(args):
    old_payloads, earlier = pair.ancestry(args)
    sources, roster, range_pins = select(args, old_payloads, earlier)
    masks, background = backgrounds(args, roster)
    dry = {row["case"]: pair.dry_variants(row, sources) for row in roster}
    rows, inputs = [], []
    for row in roster:
        case, role = row["case"], row["role"]
        variants, rerecorded = dry[case]
        host = variants["untouched"]
        other = next(other for other in roster if other["role"] == role and other["case"] != case)
        references = {"same": host, "different_reader_reference": pair.normed(dry[other["case"]][0]["untouched"], host),
                      "same_words_rerecorded_reference": pair.normed(rerecorded, host)}
        loud = pair.normed(masks[case], host, min(LEVELS))
        peak = max(*(np.max(abs(x) + abs(loud)) for x in variants.values()), *(np.max(abs(x)) for x in references.values()))
        gain = 24000 / peak
        refs = {key: pair.quantized(x * gain) for key, x in references.items()}
        for level in LEVELS:
            noise = loud * 10 ** (-(level - min(LEVELS)) / 20)
            measured = float(20 * np.log10(audio.rms(pair.quantized(host * gain).astype(np.float64))
                                         / audio.rms(pair.quantized(noise * gain).astype(np.float64))))
            audio.require(abs(measured - level) < .02, "Unexpected mixing ratio")
            for kind, signal in {**variants, **{key: host for key in references if key != "same"}}.items():
                before = refs.get(kind, refs["same"])
                submitted = pair.quantized((signal + noise) * gain)
                inputs.append((before, submitted))
                rows.append({"case": case, "role": role, "kind": kind, "level": level,
                             "truth": row["edit_range"] if kind in pair.POSITIVES else None,
                             "earlier_sha256": audio.digest(before.tobytes()), "submitted_sha256": audio.digest(submitted.tobytes()),
                             "measured_snr_db": round(measured, 6)})
    audio.require(len(rows) == 150, "Unexpected source plan size")
    return {
        "schema": SCHEMA, "method": METHOD, "clock_hz": pair.RATE, "source_samples": pair.SIZE, "levels": LEVELS,
        "code_sha256": audio.file_digest(Path(__file__)), "prior_sha256": audio.file_digest(PRIOR),
        "source_roster_sha256": audio.digest(audio.json_bytes(roster)), "publisher_ranges": range_pins,
        "background_roster_sha256": audio.digest(audio.json_bytes(background)),
        "roster_sha256": audio.digest(audio.json_bytes(rows)),
    }, rows, inputs


def bands(before, after):
    weight = np.hanning(WIDTH)
    norm = weight.sum()
    values = []
    for low, high in ((300, 900), (900, 1750), (1750, 3400)):
        sos = butter(2, (low, high), fs=pair.RATE, btype="bandpass", output="sos")
        x, y = [np.lib.stride_tricks.sliding_window_view(sosfiltfilt(sos, v.astype(np.float64)), WIDTH)[::HOP]
                for v in (before, after)]
        x, y = (v - (v @ weight / norm)[:, None] for v in (x, y))
        power = (x*x) @ weight
        xy = (x*y) @ weight
        values.append((np.sqrt(power / norm), xy / np.maximum(power, 1e-8)))
    return values


def project(before, after):
    stats = bands(before, after)
    best = None
    centers = np.arange(73) * HOP + WIDTH // 2
    for start in range(8, len(centers) - 17):
        stop = start + 9
        score, support = [], []
        for strength, gain in stats:
            cutoff = max(12., .45 * np.quantile(strength, .85))
            ranges = (slice(start - 8, start - 1), slice(start, stop), slice(stop + 1, stop + 8))
            current = []
            for region in ranges:
                weight = np.minimum(strength[region], np.quantile(strength, .9)) ** 2
                mask = strength[region] >= cutoff
                if np.count_nonzero(mask) < 3:
                    break
                current.append(float(np.dot(weight[mask], gain[region][mask]) / weight[mask].sum()))
            if len(current) != 3:
                continue
            left, local, right = current
            if min(left, right) <= .25 or max(left, right) / min(left, right) > 2.2:
                continue
            score.append(1 - local / min(left, right))
            support.append([round(value, 8) for value in current])
        if len(score) >= 2:
            value = round(float(np.median(score)), 8)
            a = int(centers[start] - HOP // 2)
            if best is None or value > best["score"]:
                best = {"score": value, "range": [a, a + RANGE], "band_gains": support}
    return best


def measure(rows, clips, role):
    output = []
    for row, (a, b) in zip(rows, clips, strict=True):
        if row["role"] != role:
            continue
        values, active = pair.profiles(a, b)
        baseline = np.flatnonzero(np.diff(np.r_[False, np.any((a != b).reshape(-1, 320), axis=1), False]))
        output.append({**row, "active_windows": int(active), "supported_pair": values is not None,
                       "results": {"prior_correlation": None if values is None else pair.localize(values["correlation"]),
                                   "coherent_projection": None if values is None else project(a, b)},
                       "exact_range": [int(baseline[0])*320, int(baseline[-1])*320] if len(baseline) else None})
    return output


def points(rows):
    return {arm: max(r["results"][arm]["score"] for r in rows if r["kind"] in pair.CONTROLS and r["results"][arm] is not None)
            for arm in ARMS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "score", "replay"))
    for arg in ("cache", "old-cache", "record", "odss-cache", "ami", "gain-ami", "run"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    lock, rows, clips = inputs(args)
    freeze, dev, cutoffs, outcome = (args.run / f"{key}.json" for key in ("freeze", "development", "points", "public"))
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        audio.write(freeze, lock)
        print(f"Froze all {len(rows)} clean-reference pairs before measurement")
        return
    audio.require(freeze.read_bytes() == audio.json_bytes(lock), "Frozen source inputs or comparison code differ")
    if args.phase == "fit":
        development = measure(rows, clips, "development")
        audio.write(dev, development)
        audio.write(cutoffs, {"freeze_sha256": audio.file_digest(freeze), "development_sha256": audio.file_digest(dev),
                              "points": points(development)})
        print(f"Sealed both points on {len(development)} development pairs")
        return
    development, operating = audio.read(dev), audio.read(cutoffs)
    audio.require(operating == {"freeze_sha256": audio.file_digest(freeze), "development_sha256": audio.file_digest(dev),
                                "points": points(development)}, "Development-only points differ")
    held = measure(rows, clips, "assessment")
    result = {"freeze": lock, "development": development, "points": operating, "assessment": held}
    if args.phase == "replay":
        audio.require(outcome.read_bytes() == audio.json_bytes(result) and development == measure(rows, clips, "development"),
                      "Clean-reference replay differs")
        print(f"Replayed all {len(rows)} clean-reference pairs")
        return
    audio.write(outcome, result)
    print(f"Scored {len(held)} held clean-reference pairs")


if __name__ == "__main__":
    main()
