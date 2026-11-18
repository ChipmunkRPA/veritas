# SPDX-License-Identifier: Apache-2.0
"""Freeze two stereo change checks and evaluate them against new recordings."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import tempfile
import wave
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.signal import resample_poly

from app.detectors.audio import stereo_ambience
from app.detectors.audio.decode import decode_audio
from app.evidence import stereo_collapse_binding, stereo_difference_binding
from app.evidence.audio_silence_binding import continuous_spans
from scripts.audio_native_alignment_panel import digest, iou, read, save, save_wav, segment

DOCS = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
SCHEMA = "chipmunk.stereo-ambience-panel/1"
ARCHIVE_SHA1 = "0f8088da2f45bd699bb6e39870dc8c6b624ddb96"
NATIVE = 48_000
CARRIERS = ("pcm", "aac160")
MINIMUM = {"reduction": 10.0, "spectrum": .26}
VOICES = {"development": ("ES2004a", "TS3004a"), "held": ("IS1006b", "ES2011b")}


def _rms(samples):
    return max(.01, float(np.sqrt(np.mean(samples ** 2))))


def _mix(room, speech, donor, alternate, number):
    """All local controls use precisely the same sample offsets as the positives."""
    durations = (.27, .38, .52, .68, .86, 1.04)
    start = int((1.31 + ((number * 31) % 59) / 17) * NATIVE)
    start = start - start % 3
    size = round(durations[number % len(durations)] * NATIVE)
    stop = start + size
    alpha = np.zeros(len(room))
    alpha[start:stop] = 1.
    edge = 480 + 120 * (number % 3)
    alpha[start:start + edge] = np.linspace(0., 1., edge)
    alpha[stop - edge:stop] = np.linspace(1., 0., edge)
    level = (-5, 4, 10, 15)[number % 4]
    speech_gain = _rms(room) / _rms(speech) * 10 ** (level / 20)
    donor = donor * (_rms(speech) / _rms(donor)) * speech_gain
    speech = speech * speech_gain
    imbalance = (0., .015, -.027, .06, -.01)[number % 5]
    room_mid = room.mean(axis=1)
    room_side = (room[:, 0] - room[:, 1]) / 2
    mid = room_mid + speech
    side = room_side + imbalance * speech
    other = (alternate[:, 0] - alternate[:, 1]) / 2
    local = slice(start, stop)
    other *= _rms(side[local]) / _rms(other[local])
    pan_width = .27 * NATIVE
    time = np.arange(len(room))
    pan = np.clip((time - start + pan_width) / (2 * pan_width), 0, 1)
    pan *= np.clip((stop + pan_width - time) / (2 * pan_width), 0, 1)
    headroom = min(1., 28000. / (sum(np.max(np.abs(x)) for x in (mid, side, other, donor)) + 1.))

    def encode(common, difference):
        return np.rint(headroom * np.column_stack((common + difference, common - difference))).astype("<i2")

    original = encode(mid, side)
    for role, common, difference, edited in (
        ("untouched_mix", mid, side, False),
        ("whole_mix_duck", mid * (1. - .8 * alpha), side * (1. - .8 * alpha), False),
        ("centered_speaker_change", mid + (donor - speech) * alpha, side + imbalance * (donor - speech) * alpha, False),
        ("gentle_pan", mid, side + .22 * pan * speech, False),
        ("wide_reduction", mid, side * (1. - .94 * alpha), True),
        ("mild_reduction", mid, side * (1. - .82 * alpha), True),
        ("background_only", mid, side - .92 * room_side * alpha, True),
        ("difference_replacement", mid, side * (1. - alpha) + other * alpha, True),
    ):
        pcm = encode(common, difference)
        if edited and (not np.array_equal(original[:start], pcm[:start]) or not np.array_equal(original[stop:], pcm[stop:])):
            raise ValueError("Changed audio escaped its truth interval")
        yield role, pcm, [start // 3, stop // 3] if edited else None, level, imbalance


def _voices(directory, split):
    names = VOICES[split]
    if split == "development":
        checksums = read(DOCS / "audio-aac-repetition-held-2026-11-17.json.gz")["foreground_sha256"]
        original = {f"{name}.Array1-05.wav": checksums[name] for name in names}
        names = list(original)
    else:
        original = read(DOCS / "audio-stereo-timing-change-held-2026-11-17.json.gz")["speech"]
        names = [f"{name}.Array1-{channel}.wav" for name in names for channel in ("05", "08")]
    files = {name: directory / name for name in names}
    if any(digest(path) != original[name] for name, path in files.items()):
        raise ValueError("Speech source differs from the recorded publisher checksum")
    return files, {name: original[name] for name in names}


def _scene_crops(group, room, alternate, number, files, voices, start, sources):
    size = len(room)
    voice_offset = 282 + 11 * number
    labels = [f"{voices[(number + index) % 2]}.Array1-05.wav" for index in range(2)]
    offsets = [voice_offset, voice_offset + 15]
    voice, donor = (resample_poly(segment(files[label], offset, size // 3, 16000).astype(float), 3, 1)
                    for label, offset in zip(labels, offsets, strict=True))
    base = {"group": group, "recording": sources, "background_start_seconds": start, "speech": labels, "speech_start_seconds": offsets}
    for role, audio, truth, level, imbalance in _mix(room, voice, donor, alternate, number):
        yield base | {"role": role, "truth": truth, "speech_over_background_db": level, "speech_channel_imbalance": imbalance}, audio


def _development(folders, files):
    def extract(paths, start, size):
        return np.column_stack([segment(path, start, size * NATIVE, NATIVE) for path in paths]).astype(float)

    development_scenes = {"DLIVING", "OOFFICE", "DKITCHEN", "TMETRO"}
    for folder in folders:
        scenes = read(folder / "recorded-sources.json")
        for source in scenes:
            scene = source["scene"]
            if scene not in development_scenes:
                continue
            development_scenes.remove(scene)
            paths = [folder / item["member"] for item in source["members"]]
            if any(digest(path) != item["sha256"] for path, item in zip(paths, source["members"], strict=True)):
                raise ValueError("Background differs from the recorded publisher checksum")

            for number, start in enumerate((4, 74, 141, 254, 286)):
                size = (7, 9, 11)[number % 3]
                key = f"{scene}-{number}"
                room = extract(paths, start, size)
                alternate = extract(paths, (start + 39) % 245, size)
                yield from _scene_crops(key, room, alternate, number, files, VOICES["development"], start, scene)
                if number < 4:
                    room = extract(paths, (start + 9) % 240, 30)
                    yield {"group": key, "recording": scene, "background_start_seconds": (start + 9) % 240,
                           "role": "recorded_environment", "truth": None}, room.astype("<i2")
    if development_scenes:
        raise ValueError(f"Missing development scenes: {sorted(development_scenes)}")


def _archive_pcm(raw):
    with wave.open(io.BytesIO(raw), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()) != (2, 2, 44100, 1323000):
            raise ValueError("Unexpected archive audio format")
        return np.frombuffer(stream.readframes(1323000), dtype="<i2").reshape(-1, 2)


def _held(archive, files):
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha1").hexdigest() != ARCHIVE_SHA1:
            raise ValueError("Holdout archive differs from its publisher checksum")
    with ZipFile(archive) as bundle:
        members = sorted(name for name in bundle.namelist() if re.fullmatch(r"scenes_stereo/[a-z]+[0-9]{2}\.wav", name))
        if len(members) != 100:
            raise ValueError("Expected 100 original DCASE recordings")
        for index, member in enumerate(members):
            identifier = Path(member).stem
            raw = bundle.read(member)
            background = {"member": member, "sha256": hashlib.sha256(raw).hexdigest()}
            stereo = _archive_pcm(raw)
            yield {"group": identifier, "recording": background, "role": "recorded_environment", "truth": None}, raw
            if index % 10 >= 4:
                continue
            number = (index // 10) * 4 + index % 10
            size = (7, 9, 11)[number % 3]
            start = 3 + number % 13
            donor_member = members[(index + 19) % len(members)]
            donor_bytes = bundle.read(donor_member)
            donor = _archive_pcm(donor_bytes)
            one, two = (resample_poly(samples[first * 44100:(first + size) * 44100].astype(float), 160, 147, axis=0)
                        for samples, first in ((stereo, start), (donor, 14)))
            sources = [background, {"member": donor_member, "sha256": hashlib.sha256(donor_bytes).hexdigest(), "start_seconds": 14}]
            yield from _scene_crops(identifier, one, two, number, files, VOICES["held"], start, sources)
    for position, name in enumerate(VOICES["held"]):
        for index, start in enumerate((43, 87, 126, 147, 201, 245, 270, 333)):
            labels = [f"{name}.Array1-{channel}.wav" for channel in ("05", "08")]
            stereo = np.column_stack([segment(files[label], start, 30 * 16000, 16000) for label in labels])
            samples = np.rint(resample_poly(stereo.astype(float), 3, 1, axis=0)).clip(-32768, 32767).astype("<i2")
            yield {"group": f"conversation-{position}-{index}", "speech": labels, "speech_start_seconds": [start],
                   "role": "recorded_conversation", "truth": None}, samples


def _delivery(info, samples, outputs, point):
    with tempfile.TemporaryDirectory(dir=outputs) as directory:
        path = Path(directory) / "stereo.wav"
        if isinstance(samples, bytes):
            path.write_bytes(samples)
        else:
            save_wav(path, samples)
        other = Path(directory) / "stereo.m4a"
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map_metadata", "-1",
                        "-c:a", "aac", "-b:a", "160k", str(other)], check=True, timeout=20)
        rows = []
        for carrier, input_path in zip(CARRIERS, (path, other), strict=True):
            sha = digest(input_path)
            decoded = decode_audio(input_path, source_sha256=sha, inspect_stereo=True)
            stereo = decoded.stereo_pcm
            context = decoded.mapping["stereo_context"]
            if stereo is None or not context["stereo_pcm_sha256"]:
                raise ValueError("Stereo delivery did not retain verified channels")
            if list(continuous_spans(decoded.mapping["timeline"])) != [(0, len(decoded.pcm))]:
                raise ValueError("Evaluation delivery had a discontinuous source clock")
            measured, measurements = stereo_ambience.search(stereo_ambience.frames(stereo))
            result = context[stereo_difference_binding.KEY]
            event = measurements["reduction"]
            expected = bool(event and event["score"] > stereo_difference_binding.OPERATING_POINT_DB)
            if (result["status"] != "unassessed") != measured or bool(result["selected"]) != expected:
                raise ValueError("Serving stereo decision differs from the measured candidate")
            if expected and any(result["selected"][0][key] != value for key, value in event.items()):
                raise ValueError("Serving stereo measurements differ from the measured candidate")
            exact = context[stereo_collapse_binding.KEY]["selected"]
            rows.append(info | {"carrier": carrier, "delivery_sha256": sha, "pcm_sha256": context["stereo_pcm_sha256"],
                                "samples": len(stereo), "assessed": measured, "measurements": measurements,
                                "exact_baseline": [[event["start_sample"], event["end_sample"]] for event in exact]})
            if point and info["group"] == point and info["role"] in ("wide_reduction", "untouched_mix"):
                (outputs / f"{point}-{info['role']}{input_path.suffix}").write_bytes(input_path.read_bytes())
        return rows


def thresholds(rows):
    chosen = {}
    for carrier in CARRIERS:
        chosen[carrier] = {}
        for kind, floor in MINIMUM.items():
            ordinary = [row for row in rows if row["carrier"] == carrier and row["truth"] is None]
            scores = [(row["measurements"][kind] or {}).get("score", 0.) for row in ordinary]
            chosen[carrier][kind] = max(floor, round(float(np.quantile(scores, .98, method="higher")), 4))
    return chosen


def _correct(span, truth):
    return bool(truth and iou(span, truth) >= .40)


def totals(rows, point):
    summary = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        carrier, role, truth = row["carrier"], row["role"], row["truth"]
        counts = summary[carrier][role]
        counts.update(n=1, assessed=row["assessed"], exact_flagged=bool(row["exact_baseline"]),
                      exact_correct=any(_correct(span, truth) for span in row["exact_baseline"]))
        for kind, result in row["measurements"].items():
            flagged = bool(result and result["score"] > point[carrier][kind])
            correct = flagged and _correct([result["start_sample"], result["end_sample"]], truth)
            counts.update({f"{kind}_flagged": flagged, f"{kind}_correct": correct})
    return summary


def measure(options):
    from app.detectors.audio import stereo

    split, output = options.split, options.output
    files, speech_hashes = _voices(options.speech_dir, split)
    inputs = _development(options.rooms, files) if split == "development" else _held(options.archive, files)
    method_sha = digest(Path(stereo_ambience.__file__))
    if split == "held":
        frozen = read(options.freeze)
        if frozen["implementation_sha256"] != method_sha or frozen["thresholds"] != thresholds(frozen["rows"]):
            raise ValueError("The held method and thresholds must match frozen development")
    output.mkdir(mode=0o700)
    rows, pending = [], set()
    next_progress = 80
    with ThreadPoolExecutor(max_workers=4) as pool:
        for info, samples in inputs:
            pending.add(pool.submit(_delivery, info, samples, output, options.keep_example))
            if len(pending) < 8:
                continue
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for item in done:
                rows.extend(item.result())
            if len(rows) >= next_progress:
                print(len(rows), "deliveries measured", flush=True)
                next_progress += 80
        for item in pending:
            rows.extend(item.result())
    rows.sort(key=lambda row: (row["group"], row["role"], row["carrier"]))
    point = thresholds(rows) if split == "development" else frozen["thresholds"]
    receipt = {"schema": SCHEMA, "split": split, "rows": rows, "thresholds": point, "totals": totals(rows, point),
               "speech_sha256": speech_hashes, "archive_sha1": ARCHIVE_SHA1 if split == "held" else None,
               "development_background_sources": [source for folder in options.rooms or []
                                                  for source in read(folder / "recorded-sources.json")],
               "calibration_sha256": digest(options.freeze) if split == "held" else None,
               "implementation_sha256": method_sha, "baseline_sha256": digest(Path(stereo.__file__)),
               "recipe_sha256": digest(Path(__file__)),
               "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]}
    save(output / "result.json.gz", receipt)
    print(json.dumps({"thresholds": point, "totals": receipt["totals"]}, sort_keys=True), flush=True)


def review(receipt, freeze):
    data = read(receipt)
    if data["schema"] != SCHEMA or totals(data["rows"], data["thresholds"]) != data["totals"]:
        raise ValueError("Published recording counts could not be reproduced")
    calibration = read(freeze) if freeze else data
    if (thresholds(calibration["rows"]) != data["thresholds"] or
            freeze and (digest(freeze) != data["calibration_sha256"] or
                        calibration["implementation_sha256"] != data["implementation_sha256"])):
        raise ValueError("Published calibration could not be reproduced")
    print(json.dumps({"thresholds": data["thresholds"], "totals": data["totals"]}, sort_keys=True))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="action", required=True)
    run = actions.add_parser("measure")
    run.add_argument("split", choices=VOICES)
    run.add_argument("speech_dir", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--rooms", nargs="+", type=Path)
    run.add_argument("--archive", type=Path)
    run.add_argument("--freeze", type=Path)
    run.add_argument("--keep-example")
    inspect = actions.add_parser("review")
    inspect.add_argument("receipt", type=Path)
    inspect.add_argument("--freeze", type=Path)
    options = cli.parse_args()
    if options.action == "measure":
        if options.split == "development" and not options.rooms:
            cli.error("Development requires --rooms")
        if options.split == "held" and (not options.archive or not options.freeze):
            cli.error("The held panel requires --archive and --freeze")
        measure(options)
    else:
        review(options.receipt, options.freeze)
