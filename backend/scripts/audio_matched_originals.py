"""Verify the original LibriTTS recordings for partially edited speech cases."""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import tarfile
import wave
from pathlib import Path

import numpy as np
from scipy import signal

from scripts import audio_focused_context_transfer as event
from scripts.audio_edit_direction_fresh import mix

SCHEMA = "chipmunk.same-utterance-originals/1"
ARCHIVE_MD5 = "0c3076c1e5245bb3f0af7d82087ee207"
RATE = 16000
CHECK_WINDOW = 6400
EDIT_MARGIN = 1600
MIN_CORRELATION = .94
MIN_MARGIN = .08
MATCHED_SCHEMA = "chipmunk.same-utterance-noise-controls/1"


def source_member(identifier):
    match = re.search(r"_(\d+)_(\d+)_(\d{6})_(\d{6})$", identifier)
    if match is None:
        raise ValueError(f"Unrecognized publisher identifier: {identifier}")
    speaker, chapter, *_ = match.groups()
    return f"LibriTTS/dev-clean/{speaker}/{chapter}/{'_'.join(match.groups())}.wav"


def _originals(source, roster):
    with source.open("rb") as archive:
        if hashlib.file_digest(archive, "md5").hexdigest() != ARCHIVE_MD5:
            raise ValueError("The archive differs from the published LibriTTS dev-clean checksum")
    with tarfile.open(source, "r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if member.isfile() and name in roster:
                with archive.extractfile(member) as stream:
                    yield name, stream.read()


def source_wave(wav):
    with wave.open(io.BytesIO(wav)) as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, 24000):
            raise ValueError("The publisher file is outside the expected native audio profile")
        samples = np.frombuffer(stream.readframes(stream.getnframes()), "<i2")
    return signal.resample_poly(samples.astype(np.float64) / 32768, 2, 3)


def edited_wave(folder, row):
    raw = (folder / "pcm" / f"{row['pcm_sha256']}.pcm").read_bytes()
    if len(raw) != 2 * row["samples"] or hashlib.sha256(raw).hexdigest() != row["pcm_sha256"]:
        raise ValueError("The edited recording differs from its frozen waveform")
    return np.frombuffer(raw, "<i2").astype(np.float64) / 32768


def unchanged_windows(samples, truth):
    cursor = 0
    bounds = []
    for start, stop in truth:
        bounds.append((cursor, max(cursor, start - EDIT_MARGIN)))
        cursor = min(samples, stop + EDIT_MARGIN)
    bounds.append((cursor, samples))
    for start, stop in bounds:
        while stop - start >= CHECK_WINDOW:
            # Each returned excerpt is wholly outside a known edited interval.
            yield start, start + CHECK_WINDOW
            start += CHECK_WINDOW


def locate(waveform, template):
    if len(waveform) < len(template) or float(np.std(template)) < .004:
        return None
    template = template - template.mean()
    cross = signal.correlate(waveform, template, mode="valid", method="fft")
    kernel = np.ones(len(template))
    sums = signal.fftconvolve(waveform, kernel, mode="valid")
    energy = signal.fftconvolve(np.square(waveform), kernel, mode="valid") - np.square(sums) / len(template)
    power = np.sqrt(np.maximum(energy, 0.) * float(template @ template))
    scores = np.zeros(cross.shape)
    np.divide(cross, power, out=scores, where=power > 1e-6)
    best = int(scores.argmax())
    alternative = max(scores[:max(0, best - RATE // 8)].max(initial=-1.), scores[best + RATE // 8:].max(initial=-1.))
    return {"original_start": best, "correlation": round(float(scores[best]), 6),
            "separation": round(float(scores[best] - alternative), 6)}


def _supported(item):
    return item["correlation"] >= MIN_CORRELATION and item["separation"] >= MIN_MARGIN


def compare(original, edited, truth):
    measurements = []
    for start, end in unchanged_windows(len(edited), truth):
        match = locate(original, edited[start:end])
        if match is not None:
            measurements.append(match | {"edited_samples": [start, end]})
    matching = [entry for entry in measurements if _supported(entry)]
    anchors = [entry["original_start"] for entry in matching]
    aligned = len(anchors) >= 2 and all(b - a >= CHECK_WINDOW // 2 for a, b in zip(anchors, anchors[1:]))
    return {"original_16khz_samples": len(original), "edited_samples": len(edited), "measurements": measurements,
            "verified_unchanged_windows": len(anchors), "same_utterance_confirmed": aligned}


def audit(source, media, output):
    frozen = media / "freeze.json"
    plan = event.read(frozen)
    cases = {row["id"].split("/", 1)[0]: row for row in plan["cases"] if (row["kind"], row["condition"]) == ("edited", "clean")}
    roster = {source_member(key): (key, row) for key, row in cases.items()}
    if len(cases) != len(roster):
        raise ValueError("At least two edited recordings point to one source file")
    results = {}
    for member, native in _originals(source, roster):
        identifier, row = roster[member]
        if identifier in results:
            raise ValueError(f"The archive repeats {member}")
        result = compare(source_wave(native), edited_wave(media, row), row["truth"])
        results[identifier] = result | {"member": member, "original_sha256": hashlib.sha256(native).hexdigest(),
                                        "edited_pcm_sha256": row["pcm_sha256"], "family": row["family"]}
        print(identifier, result["verified_unchanged_windows"], result["same_utterance_confirmed"], flush=True)
    with frozen.open("rb") as stream:
        plan_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    event.save(output, {"schema": SCHEMA, "archive_md5": ARCHIVE_MD5, "holdout_sha256": plan_sha,
                        "planned": len(roster), "located_in_archive": len(results),
                        "same_utterance_confirmed": sum(row["same_utterance_confirmed"] for row in results.values()),
                        "missing_originals": sorted(set(cases) - set(results)), "results": results})


def _shared_clock(record):
    center = record["edited_samples"] // 2
    anchor = min((item for item in record["measurements"] if _supported(item)),
                 key=lambda item: abs(sum(item["edited_samples"]) // 2 - center))
    delay = anchor["original_start"] - anchor["edited_samples"][0]
    first = max(0, -delay)
    last = min(record["edited_samples"], record["original_16khz_samples"] - delay)
    return first, last, delay, anchor


def prepare(source, media, measured, rooms, output):
    frozen = media / "freeze.json"
    plan, audit_record = event.read(frozen), event.read(measured)
    with frozen.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != audit_record["holdout_sha256"]:
            raise ValueError("The original audit uses a different evaluation plan")
    room_plan = event.read(rooms / "rooms.json")
    if room_plan != plan["rooms"]:
        raise ValueError("The matching experiment uses a different competing speech pool")
    origins = {row["id"]: row for row in plan["origins"]}
    edited = {case["id"].split("/", 1)[0]: case for case in plan["cases"]
              if case["kind"] == "edited" and case["condition"] == "clean"}
    background = {}
    for room in plan["rooms"]["rooms"]:
        samples = (rooms / f"{room['room']}.pcm").read_bytes()
        if hashlib.sha256(samples).hexdigest() != room["pcm_sha256"] or len(samples) != 28 * RATE:
            raise ValueError("Competing speech differs from the frozen source")
        background[room["room"]] = np.frombuffer(samples, "<i2")
    qualified = {record["member"]: (key, record) for key, record in audit_record["results"].items()
                 if record["same_utterance_confirmed"]}
    output.mkdir(mode=0o700)
    (output / "pcm").mkdir()
    cases, pairs, sources = [], [], []
    unavailable = {key: "insufficient_unchanged_witnesses" for key in edited if key not in {i for i, _ in qualified.values()}}

    def keep(key, values, kind, condition, room, group, labels, family=None):
        raw = values.astype("<i2").tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        (output / "pcm" / f"{digest}.pcm").write_bytes(raw)
        cases.append({"id": f"{key}/{kind}/{condition}", "group": group, "kind": kind, "condition": condition,
                      "room": room, "family": family, "truth": labels, "pcm_sha256": digest, "samples": len(values),
                      "corpus": "llama" if kind == "edited" else "libritts"})

    for member, native in _originals(source, qualified):
        key, measured_source = qualified[member]
        if hashlib.sha256(native).hexdigest() != measured_source["original_sha256"]:
            raise ValueError("The matched original does not match its audited bytes")
        row, source_meta = edited[key], origins[key]
        first, last, delay, anchor = _shared_clock(measured_source)
        span = last - first
        offset = (span - 6 * RATE) // 2
        if span < 9 * RATE or not any(first + offset <= a < b <= first + offset + 6 * RATE for a, b in row["truth"]):
            unavailable[key] = "insufficient_shared_scope"
            continue
        clean, modified = source_wave(native), edited_wave(media, row)
        original_at = anchor["original_start"]
        a, b = anchor["edited_samples"]
        amplitude = float(np.linalg.norm(modified[a:b]) / np.linalg.norm(clean[original_at:original_at + b - a]))
        waves = np.asarray([modified[first:last], clean[first + delay:last + delay] * amplitude])
        safety_gain = min(1., .92 / float(np.abs(waves).max()))
        waves = np.rint(waves * safety_gain * 32768).astype("<i2")
        labels = [[max(a, first) - first, min(b, last) - first] for a, b in row["truth"] if a < last and b > first]
        room, group, family = source_meta["room"], row["group"], row["family"]
        control = key + "-original"
        sources.extend(({"id": key, "group": group, "source_sha256": source_meta["source_sha256"], "family": family},
                        {"id": control, "group": group, "source_sha256": measured_source["original_sha256"], "family": None}))
        pairs.append({"edit": key, "ordinary": control, "edit_source_samples": [first, last],
                      "original_source_samples": [first + delay, last + delay], "shared_anchor": anchor,
                      "ordinary_amplitude": amplitude, "shared_gain": safety_gain, "room": room})
        keep(key, waves[0], "edited", "clean", None, group, labels, family)
        keep(control, waves[1], "ordinary", "clean", None, group, [])
        for db in (-5, -10):
            submitted, isolated = mix(waves, background[room][first:last], db)
            condition = f"overlap_{db}dB"
            keep(key, submitted[0], "edited", condition, room, group, labels, family)
            keep(control, submitted[1], "ordinary", condition, room, group, [])
            keep(key, isolated, "background", condition, room, group, [])
    event.save(output / "freeze.json", {"schema": MATCHED_SCHEMA, "parent_sha256": audit_record["holdout_sha256"],
                                        "original_archive_md5": ARCHIVE_MD5, "origins": sources, "rooms": room_plan,
                                        "pairs": pairs, "cases": cases, "unavailable": unavailable})
    print(f"Prepared {len(cases)} deliveries from {len(pairs)} waveform-verified original and edited pairs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_cli = commands.add_parser("audit")
    prepare_cli = commands.add_parser("prepare")
    for name in ("source", "media", "output"):
        audit_cli.add_argument(name, type=Path)
    for name in ("source", "media", "measured", "rooms", "output"):
        prepare_cli.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        audit(args.source, args.media, args.output)
    else:
        prepare(args.source, args.media, args.measured, args.rooms, args.output)
