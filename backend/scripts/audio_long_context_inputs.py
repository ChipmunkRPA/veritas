"""Prepare paired six- and ten-second views of the same short speech edits."""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from scripts import audio_phoneme_short_edit_intake as intake
from scripts.audio_edit_direction_fresh import mix
from scripts.audio_focused_context_transfer import read, save
from scripts.prepare_natural_overlap_transfer import track_words

RATE = 16000
SAMPLES = {"short": 6 * RATE, "long": 10 * RATE}
CONTEXTS = tuple(SAMPLES)
GENERATORS = ("MeloTTS", "ChatterboxTTS")
ORIGINS = ("real", *GENERATORS)
KINDS = ("ordinary", *(f"resampled/{family}" for family in GENERATORS), *(f"edited/{family}" for family in GENERATORS))
PREFIX = 320 * 1024**2
SCHEMA = "chipmunk.audio-native-context-inputs/1"
PHRASE = re.compile(r"(\d+-\d+-\d+)\.wav")
GRID = re.compile(r"textgrids/(real|MeloTTS|ChatterboxTTS)/(\d+-\d+-\d+)\.TextGrid")
ANNOTATION_SHA256 = "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d"


def rank(*values):
    return hashlib.sha256("/".join((SCHEMA, *map(str, values))).encode()).hexdigest()


def source_archive(cache, origin, total, extent, expected=None):
    complete = intake.TEXTGRID_MD5 if origin == "textgrids" else None
    parts = intake.archive(cache, origin, total, extent, complete_md5=complete)
    if expected is not None and [intake.audio.file_digest(part) for part in parts] != expected:
        raise ValueError(f"Publisher prefix differs for {origin}")
    return parts


def sources(cache, panels, exclusions):
    manifest = read(panels / "audio-phonemedf-release-19522979.json")
    expected = read(panels / "audio-phoneme-short-edit-intake-2026-11-13.json")["bounded_wav_ranges"]
    expected["MeloTTS"] = read(panels / "audio-melo-short-source-census-2026-11-13.json")["melo_range_sha256"]
    if (manifest["id"], manifest["metadata"]["license"]["id"]) != (19522979, "cc-by-4.0"):
        raise ValueError("Unexpected PhonemeDF release")
    files = {row["key"]: row["size"] for row in manifest["files"]}
    signals = {}
    for origin in ORIGINS:
        chunks = source_archive(cache, origin, files[f"PhonemeDF_{origin}.tar.gz"], PREFIX, expected[origin])
        carrier = "real/wav" if origin == "real" else origin
        matched = {}
        for name, body in intake.prior.archive_members(chunks):
            path = Path(name)
            found = PHRASE.fullmatch(path.name)
            if not found or path.parent.as_posix() != carrier:
                continue
            key = found[1]
            if key.split("-", 1)[0] in exclusions or (origin != "real" and key not in signals["real"]):
                continue
            values = np.frombuffer(intake.prior.pcm_bytes(body), "<i2")
            if origin != "real" or len(values) >= SAMPLES["long"]:
                matched[key] = values
        signals[origin] = matched
        print(f"{origin} supplied {len(matched)} length-qualified or source-matched utterances", flush=True)
    common = set.intersection(*(set(signals[origin]) for origin in ORIGINS))
    alignments = defaultdict(dict)
    full = files["PhonemeDF_textgrids.tar.gz"]
    chunks = source_archive(cache, "textgrids", full, full)
    for name, body in intake.prior.archive_members(chunks):
        found = GRID.fullmatch(name)
        if found and found[2] in common:
            origin, key = found.groups()
            if spans := intake.words(body, len(signals[origin][key])):
                alignments[key][origin] = spans
    return signals, alignments, {origin: expected[origin] for origin in ORIGINS}


def placements(tiers, signals, key):
    if set(tiers) != set(ORIGINS):
        return
    words = tiers["real"]
    if any([part[2] for part in tiers[origin]] != [part[2] for part in words] for origin in GENERATORS):
        return
    for index in range(1, len(words) - 2):
        for count in (1, 2):
            spans = {origin: [tier[index][0], tier[index + count - 1][1]] for origin, tier in tiers.items()}
            a, b = spans["real"]
            if not 4000 <= b - a <= 9600 or any(not .8 <= (y - x) / (b - a) <= 1.25 for x, y in spans.values()):
                continue
            if any(intake.audio.rms(signals[origin][key][x:y].astype(np.float64)) < 140 for origin, (x, y) in spans.items()):
                continue
            lo = max(0, b + RATE - SAMPLES["long"])
            hi = min(a - RATE, len(signals["real"][key]) - SAMPLES["long"])
            context = min(range(lo, hi + 1, RATE // 10), key=lambda offset: rank(key, index, count, offset), default=None)
            if context is None:
                continue
            lo = max(context, b + RATE - SAMPLES["short"])
            hi = min(a - RATE, context + SAMPLES["long"] - SAMPLES["short"])
            short = min(range(lo, hi + 1, RATE // 10), key=lambda offset: rank(key, "short", index, count, offset), default=None)
            if short is not None:
                yield {"source": key, "reader": key.split("-", 1)[0], "offsets": {"short": short, "long": context},
                       "native_edit": spans}


def select(signals, alignments):
    ordered = []
    for key, tiers in alignments.items():
        choice = min(placements(tiers, signals, key), key=lambda row: rank(key, row["native_edit"]), default=None)
        if choice is not None:
            ordered.append(choice)
    counts = Counter()
    selected = []
    for row in sorted(ordered, key=lambda item: rank("utterance", item["source"])):
        if counts[row["reader"]] < 8:
            counts[row["reader"]] += 1
            selected.append(row)
    readers = sorted(counts, key=lambda owner: rank("reader-role", owner))
    if len(readers) < 4:
        raise ValueError(f"Only {len(readers)} readers qualify for a separate fit and calibration")
    calibration = set(readers[::4])
    for row in selected:
        row["role"] = "calibration" if row["reader"] in calibration else "fit"
    return selected


def room_bank(path, annotations, held):
    rooms = read(path / "rooms.json")
    forbidden = {row["room"][:6] for row in held["rooms"]["rooms"]}
    if intake.audio.file_digest(annotations) != ANNOTATION_SHA256:
        raise ValueError("The AMI annotation release differs")
    result = {}
    with ZipFile(annotations) as words:
        for row in sorted(rooms, key=lambda item: rank("room", item["room"])):
            capture = row["room"]
            if capture[:6] in forbidden:
                continue
            raw = (path / f"{capture}.pcm").read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["pcm_sha256"] or len(raw) != 2 * row["samples"]:
                raise ValueError(f"AMI recording does not match the exact archived source {capture}")
            active = np.zeros(row["samples"], dtype=bool)
            start = row["start_sample"]
            for channel in "ABCD":
                for left, right in track_words(words.read(f"words/{capture}.{channel}.words.xml")):
                    lo, hi = max(0, left * 160 - start), min(len(active), right * 160 - start)
                    if lo < hi:
                        active[lo:hi] = True
            result[capture] = (row, np.frombuffer(raw, "<i2").astype(np.float64), active)
    ids = list(result)
    if len(ids) < 4:
        raise ValueError("Need separate AMI meeting families for fitting and calibration")
    return result, {"fit": ids[2:], "calibration": ids[:2]}


def masked_room(row, rooms, allowed):
    offset = row["offsets"]["long"]
    a, b = [bound - offset for bound in row["native_edit"]["real"]]
    offers = []
    for room in allowed:
        source, background, speech = rooms[room]
        for start in range(0, len(background) - SAMPLES["long"] + 1, RATE // 10):
            coverage = int(speech[start + a:start + b].sum())
            if 2 * coverage >= b - a and speech[start:start + SAMPLES["long"]].sum() >= 4 * RATE:
                offers.append((rank("room-offset", row["source"], room, start), room, start, coverage,
                               background[start:start + SAMPLES["long"]], source["pcm_sha256"]))
    return min(offers, default=None)


def variants(row, signals):
    offset = row["offsets"]["long"]
    a, b = [bound - offset for bound in row["native_edit"]["real"]]
    host = signals["real"][row["source"]][offset:offset + SAMPLES["long"]].astype(np.float64)
    phrase = host[a:b]
    values = {"ordinary": host}
    for family in GENERATORS:
        x, y = row["native_edit"][family]
        donor = signals[family][row["source"]][x:y].astype(np.float64)
        reprocessed = intake.resample(intake.resample(phrase, len(donor)), len(phrase))
        edited = intake.resample(donor, len(phrase))
        for kind, replacement in (("resampled", reprocessed), ("edited", edited)):
            equivalent = replacement * (intake.audio.rms(phrase) / intake.audio.rms(replacement))
            values[f"{kind}/{family}"] = intake.replace(host, equivalent, a, b)
    stack = np.asarray([values[kind] for kind in KINDS])
    gain = min(1., 24000 / float(np.abs(stack).max()))
    return np.rint(stack * gain).astype("<i2"), (a, b)


def make_cases(output, selected, signals, rooms, folds):
    output.mkdir(mode=0o700)
    pcm = output / "pcm"
    pcm.mkdir()
    rows, metadata, per_fold = [], [], Counter()

    def keep(context, values):
        raw = values.astype("<i2").tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        (pcm / f"{digest}.pcm").write_bytes(raw)
        return {"context": context, "samples": len(values), "sha256": digest}

    for pair in selected:
        room = masked_room(pair, rooms, folds[pair["role"]])
        if room is None:
            continue
        _, meeting, offset, overlapping, masker, pin = room
        per_fold[pair["role"]] += 1
        levels = (-5, -10) if pair["role"] == "calibration" else ((-5, -10)[per_fold["fit"] % 2],)
        source, (a, b) = variants(pair, signals)
        metadata.append(pair | {"room": meeting, "room_offset": offset, "background_native_sha256": pin,
                                "room_speech_in_edit_samples": overlapping})
        assert all(np.array_equal(values[:a], source[0, :a]) and np.array_equal(values[b:], source[0, b:]) for values in source)
        short_start = pair["offsets"]["short"] - pair["offsets"]["long"]
        for level in (None, *levels):
            mixed, isolated = mix(source, masker, level)
            assert np.array_equal(mixed[:, :a], np.broadcast_to(mixed[0, :a], mixed[:, :a].shape))
            assert np.array_equal(mixed[:, b:], np.broadcast_to(mixed[0, b:], mixed[:, b:].shape))
            submitted = dict(zip(KINDS, mixed, strict=True))
            if isolated is not None:
                submitted["background"] = isolated
            for kind, waveform in submitted.items():
                short = waveform[short_start:short_start + SAMPLES["short"]]
                waveforms = [keep("short", short), keep("long", waveform)]
                truth = ({"short": [a-short_start,b-short_start], "long": [a,b]} if kind.startswith("edited/") else {})
                rows.append({"source": pair["source"], "reader": pair["reader"], "role": pair["role"], "condition": level,
                             "kind": kind, "room": meeting if level is not None else None, "inputs": waveforms, "truth": truth})
    return rows, metadata


def prepare(args):
    held = read(args.held)
    exclusions = {str(row["group"]) for row in held["origins"]}
    signals, alignments, pins = sources(args.cache, args.panels, exclusions)
    selected = select(signals, alignments)
    print(f"Qualified {len(selected)} matched utterances across {len({row['reader'] for row in selected})} readers", flush=True)
    rooms, folds = room_bank(args.backgrounds, args.annotations, held)
    rows, matched = make_cases(args.output, selected, signals, rooms, folds)
    if {row["role"] for row in rows} != {"fit", "calibration"}:
        raise ValueError("There is no usable separate calibration or fitting fold")
    save(args.output / "freeze.json", {"schema": SCHEMA, "source_pins": pins, "textgrids_md5": intake.TEXTGRID_MD5,
                                      "held_manifest_sha256": intake.audio.file_digest(args.held),
                                      "room_folds": folds, "room_sources": [entry[0] for entry in rooms.values()],
                                      "sources": matched, "rows": rows})
    print(f"Prepared {len(rows)} paired deliveries from {len(matched)} utterances", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("cache", "panels", "backgrounds", "annotations", "held", "output"):
        parser.add_argument(argument, type=Path)
    prepare(parser.parse_args())
