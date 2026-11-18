# SPDX-License-Identifier: Apache-2.0
"""Compare stereo sound reductions with source-located mouth measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

from scripts.video_lip_multishot_panel import CARRIERS, RATE, digest, ffmpeg, overlap, read, save
from scripts.video_silent_mouth_clock_panel import (
    GEOMETRY_SHA,
    VIDEO_ENCODING,
    fit_sound,
    pixels,
    sha,
    source_sound,
    treatments,
)

SCHEMA = "chipmunk.video-shared-sound/1"
ARCHIVE_SHA1 = "0f8088da2f45bd699bb6e39870dc8c6b624ddb96"
SNR_LEVELS = (8, -5)
ROLES = ("unmodified_mix", "whole_gain", "local_duck", "position_change", "background_reduction", "hard", "faded")
EDITS = {"hard", "faded"}
ARMS = ("direct", "difference_adjusted")
BUDGETS = (0, 1, 2)
STREAM_HASH = ("-map", "0:v:0", "-c:v", "copy", "-f", "streamhash", "pipe:1")


def acoustic_sources(archive):
    with archive.open("rb") as handle:
        if hashlib.file_digest(handle, "sha1").hexdigest() != ARCHIVE_SHA1:
            raise ValueError("The acoustic scene archive differs from its publisher checksum")
    with ZipFile(archive) as bundle:
        scenes = defaultdict(list)
        for name in sorted(bundle.namelist()):
            if match := re.fullmatch(r"scenes_stereo/([a-z]+)[0-9]{2}\.wav", name):
                scenes[match[1]].append(name)
        if len(scenes) != 10 or any(len(choices) != 10 for choices in scenes.values()):
            raise ValueError("The archive does not contain the planned stereo environments")
        for split, prior in (("development", 0), ("held", 3)):
            yield split, [scenes[key][prior + layer] for layer in range(2) for key in sorted(scenes)]


def mixes(pcm, bed, span, snr, number):
    import numpy as np

    start, stop = span
    source = pcm.astype(np.float64)
    room = bed.astype(np.float64)
    voice_rms = np.sqrt(np.mean(source[start:stop] ** 2))
    room_rms = np.sqrt(np.mean(room[start:stop] ** 2))
    if min(room_rms, voice_rms) < 8:
        raise ValueError("The source edits need measured soundtrack and ambience")
    room *= voice_rms / room_rms * 10 ** (-snr / 20)
    # Alternate a centered voice with a mild pan while keeping total gain constant.
    balance = ((1., 1.), (.83, 1.17), (1.17, .83))[number % 3]
    voice = source[:, None] * balance
    scale = min(1., 25_000 / np.maximum(np.abs(voice).max() + np.abs(room).max(), 1))
    source, voice, room = source * scale, voice * scale, room * scale
    full = voice + room
    fade = np.ones(len(source), dtype=np.float64)
    fade[start:stop] = 0
    ramp = np.linspace(1., 0., 400)
    fade[start:start + len(ramp)] = ramp
    fade[stop-len(ramp):stop] = ramp[::-1]
    soft = fade[:, None]
    hard = np.ones_like(soft)
    hard[start:stop] = 0
    options = {
        "unmodified_mix": full,
        "whole_gain": full * .6,
        "local_duck": (.3 + .7 * soft) * full,
        "position_change": room + soft * voice + (1 - soft) * source[:, None] * np.asarray([1.7, .3]),
        "background_reduction": voice + (.22 + .78 * soft) * room,
        "hard": voice * hard + room,
        "faded": voice * soft + room,
    }
    for role in ROLES:
        values = np.rint(options[role])
        if np.max(np.abs(values)) >= 32768:
            raise ValueError("The mixture clipped during preparation")
        yield role, values.astype("<i2")


def _mux(pictures, sound, carrier, output):
    options = ("-c:a", "pcm_s16le") if carrier == "pcm" else ("-c:a", "aac", "-ar", "48000", "-b:a", "160k")
    ffmpeg("-i", str(pictures), "-f", "s16le", "-ar", str(RATE), "-ac", "2", "-i", "pipe:0",
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *options,
           "-video_track_timescale", "30000", "-movie_timescale", "240000", str(output), data=sound.tobytes())


def observe(source, room, number, assets, output):
    from contextlib import closing

    import numpy as np
    from scipy.signal import resample_poly

    from app.detectors.audio.decode import decode_audio
    from app.detectors.video import shared_sound
    from app.detectors.video.decode import decode_stream, probe_frames
    from app.detectors.video.shots import detect_shots
    from app.detectors.video.viseme_review import sound_levels
    from app.detectors.video.viseme_worker import capture_mouth_frames
    from app.evidence import video_silent_mouth as silent
    from app.evidence.video_viseme_binding import presented_samples
    from scripts.audio_stereo_ambience_panel import _archive_pcm

    group = f"source-{number:02d}"
    original = probe_frames(source)
    if not 40 <= len(original) <= 120 or original.rotation or original.truncated or original.interpolated:
        raise ValueError("A planned source does not have a continuous short picture clock")
    pcm, shift = source_sound(decode_audio(source, source_sha256=digest(source)), original)
    pictures = output / f"{group}-pictures.mp4"
    ticks = [i * 1000 for i in range(len(original))]
    ffmpeg("-i", str(source), "-an", "-map", "0:v:0", "-vf", "settb=1/30000,setpts=N*1000",
           *VIDEO_ENCODING, "-bsf:v", "setts=duration=1000", str(pictures))
    table = probe_frames(pictures)
    reference_pixels = pixels(source, original)
    if table.tb != Fraction(1, 30_000) or list(table.pts) != ticks or pixels(pictures, table) != reference_pixels:
        raise ValueError("Picture preparation changed the planned clock or original image pixels")
    with closing(decode_stream(pictures, table)) as stream:
        shots = detect_shots(stream, table).as_json()
    if shots["frames_scanned"] != len(table) or shots["ranges"] != [[0, len(table)]]:
        raise ValueError("The picture does not reproduce the planned continuous shot")
    count = round(len(table) * RATE / 30)
    pcm = fit_sound(pcm, count)
    frames, _, _ = capture_mouth_frames(pictures, table, assets / "face_landmarker.task", motion=False, shots=[[0, len(table)]])
    frames = [{**frame, "sample": round(tick * RATE / 30000)} for tick, frame in zip(ticks, frames, strict=True)]
    start_sec = 4 + number % 13
    clip = _archive_pcm(room)[start_sec * 44100:(start_sec + 4) * 44100]
    bed = resample_poly(clip.astype(np.float64), 160, 147, axis=0)[:count]
    if len(bed) != count:
        raise ValueError("The recorded ambience does not extend through the presented frames")
    span = next(truth for role, _, truth in treatments(pcm, 0, number) if role == "muted")
    picture_hash = ffmpeg("-i", str(pictures), *STREAM_HASH).decode().strip()
    observations = []
    for snr in SNR_LEVELS:
        for role, audio in mixes(pcm, bed, span, snr, number):
            for carrier in CARRIERS:
                name = f"{group}-{snr:+d}-{carrier}-{role}.mov"
                output_path = output / name
                _mux(pictures, audio, carrier, output_path)
                native = probe_frames(output_path)
                if ((native.pts, native.time_base) != (table.pts, table.time_base)
                        or ffmpeg("-i", str(output_path), *STREAM_HASH).decode().strip() != picture_hash):
                    raise ValueError("A sound treatment changed a picture or its presented time")
                input_pin = digest(output_path)
                decoded = decode_audio(output_path, source_sha256=input_pin, retain_stereo=True)
                mapping = decoded.mapping
                stereo = decoded.stereo_pcm
                extent = presented_samples(mapping)
                if (extent != count or stereo is None or mapping["timeline"] != [
                        {"start_sample": 0, "end_sample": mapping["samples"], "source_pts": 0}]
                        or carrier == "pcm" and not np.array_equal(stereo, audio)):
                    raise ValueError("The delivered channels differ from the verified sample extent")
                stereo = stereo[:extent]
                values = shared_sound.levels(stereo)
                shared_sound.validate_levels(values, extent)
                assessed, audio_only, visual = shared_sound.scan(values, frames, extent)
                baseline = silent.record(sound_levels(decoded.pcm, extent), frames, extent, 0)
                observations.append({"group": group, "snr_db": snr, "carrier": carrier, "role": role,
                                     "delivery": name, "input_sha256": input_pin,
                                     "stereo_pcm_sha256": sha(stereo.astype("<i2", copy=False).tobytes()),
                                     "samples": extent, "activating_interval": span, "truth": span if role in EDITS else None,
                                     "audio_eligible": assessed, "audio_only": audio_only, "with_picture": visual,
                                     "mono_baseline": baseline[0]["span"] if baseline else None})
    provenance = {"group": group, "rgb_sha256": sha("\n".join(reference_pixels).encode()), "picture_stream": picture_hash,
                  "ticks": ticks, "duration_samples": count, "ambient_start_s": start_sec, "source_trim_samples": shift}
    return observations, provenance


def counts(rows, mode, arm, point):
    result = Counter()
    ordinary_groups, edit_groups = set(), set()
    for row in rows:
        finding = row[mode][arm]
        target = row["truth"]
        flag = finding is not None and finding["margin_millidb"] > point
        hit = bool(flag and target and overlap(finding["span"], target) >= .4)
        result.update(files=1, assessed=row["audio_eligible"], edits=bool(target), controls=target is None,
                      located=hit, missed=bool(target and not hit), misplaced=bool(target and flag and not hit),
                      ordinary_flags=target is None and flag,
                      mono_located=bool(row["mono_baseline"] and target and overlap(row["mono_baseline"], target) >= .4),
                      mono_ordinary_flags=target is None and bool(row["mono_baseline"]))
        if hit:
            edit_groups.add(row["group"])
        if target is None and flag:
            ordinary_groups.add(row["group"])
    return dict(result) | {"ordinary_groups": len(ordinary_groups), "edit_groups": len(edit_groups)}


def calibrate(rows):
    profiles, choices = {}, {}
    for carrier in CARRIERS:
        bank = [row for row in rows if row["carrier"] == carrier]
        options = []
        points = {}
        for mode in ("audio_only", "with_picture"):
            by_arm = points[mode] = {}
            for arm in ARMS:
                groups = {row["group"]: 0 for row in bank}
                for row in bank:
                    if row["truth"] is None and (finding := row[mode][arm]) is not None:
                        groups[row["group"]] = max(groups[row["group"]], finding["margin_millidb"])
                maxima = sorted(groups.values(), reverse=True)
                by_arm[arm] = {str(budget): max(3800, maxima[budget]) for budget in BUDGETS}
                if mode != "with_picture":
                    continue
                for budget, point in by_arm[arm].items():
                    observed = counts(bank, mode, arm, point)
                    low = counts([row for row in bank if row["snr_db"] < 0], mode, arm, point)
                    utility = (2 * observed["located"] + low["located"] - observed["misplaced"]
                               - 2 * observed["ordinary_flags"] - observed["ordinary_groups"])
                    options.append(((utility, -int(budget), -ARMS.index(arm)), {"arm": arm, "budget": budget, "point": point}))
        profiles[carrier] = points
        choices[carrier] = max(options, key=lambda row: row[0])[1]
    return profiles, choices


def report(rows, profiles, choices):
    return {f"{carrier}/{snr}": {
        "selected": counts(bank, "with_picture", choices[carrier]["arm"], choices[carrier]["point"]),
        "alternates": {mode: {arm: {budget: counts(bank, mode, arm, point) for budget, point in points.items()}
                              for arm, points in arms.items()} for mode, arms in profiles[carrier].items()},
    } for carrier in CARRIERS for snr in SNR_LEVELS
        if (bank := [row for row in rows if row["carrier"] == carrier and row["snr_db"] == snr])}


def measure(split, sources, archive, assets, output, *, freeze=None):
    from app.detectors.audio import stereo_ambience
    from app.detectors.video import shared_sound
    from app.evidence import video_silent_mouth

    if digest(assets / "face_landmarker.task") != GEOMETRY_SHA:
        raise ValueError("Source motion needs the same verified mouth landmark asset")
    sys.path.insert(0, str(assets / "runtime"))
    inventory = sources / "sources.json"
    roster = json.loads(inventory.read_bytes())
    clips = [row for row in roster["entries"] if row["split"] == split]
    programs = {key: digest(value) for key, value in (
        ("checks", Path(shared_sound.__file__)), ("recipe", Path(__file__)), ("landmarks", assets / "face_landmarker.task"),
        ("stereo_energy", Path(stereo_ambience.__file__)), ("mouth_motion", Path(video_silent_mouth.__file__)),
    )}
    frozen = read(freeze) if freeze else None
    if (split == "held") != (frozen is not None) or frozen and (
        frozen["programs"] != programs or frozen["source_sha256"] != digest(inventory)
        or {row["speaker"] for row in clips} & {row["speaker"] for row in frozen["sources"]}
    ):
        raise ValueError("The held evaluation needs its original freeze and unused speakers")
    rooms = dict(acoustic_sources(archive))[split]
    output.mkdir(mode=0o700)
    rows, proven = [], []
    with ZipFile(archive) as bundle:
        for number, (row, room) in enumerate(zip(clips, rooms[:len(clips)], strict=True), start=1):
            source = sources / row["name"]
            if digest(source) != row["sha256"]:
                raise ValueError("Source clip differs from its pinned publisher hash")
            raw = bundle.read(room)
            measurements, proof = observe(source, raw, number, assets, output)
            proven.append(row | proof | {"room": room, "room_sha256": sha(raw)})
            rows.extend(measurements)
            print(f"Measured {number}/{len(clips)} speakers, {len(rows)} paired deliveries", flush=True)
    profiles, choices = (frozen["profiles"], frozen["choices"]) if frozen else calibrate(rows)
    summary = report(rows, profiles, choices)
    save(output / "result.json.gz", {"schema": SCHEMA, "split": split, "programs": programs,
                                     "freeze_sha256": digest(freeze) if freeze else None, "source_sha256": digest(inventory),
                                     "archive_sha1": ARCHIVE_SHA1, "sources": proven, "rows": rows,
                                     "profiles": profiles, "choices": choices, "summary": summary})
    print(json.dumps({name: value["selected"] for name, value in summary.items()}, sort_keys=True), flush=True)


def review(result, freeze=None):
    data = read(result)
    rows = data["rows"]
    expected = {(row["group"], carrier, snr, role) for row in data["sources"]
                for carrier in CARRIERS for snr in SNR_LEVELS for role in ROLES}
    seen = {(row["group"], row["carrier"], row["snr_db"], row["role"]) for row in rows}
    if (data["schema"] != SCHEMA or seen != expected or len(rows) != len(seen)
            or data["archive_sha1"] != ARCHIVE_SHA1):
        raise ValueError("Missing or repeated paired case")
    for row in rows:
        a, b = row["activating_interval"]
        if (not 0 <= a < b <= row["samples"] or row["truth"] != ([a, b] if row["role"] in EDITS else None)
                or any(not 0 <= found["span"][0] < found["span"][1] <= row["samples"]
                       for mode in ("audio_only", "with_picture") for found in row[mode].values() if found)):
            raise ValueError("A reported range is outside its source extent")
    if freeze:
        review(freeze)
        earlier = read(freeze)
        fields = ("programs", "profiles", "choices", "source_sha256")
        if (data["split"] != "held" or data["freeze_sha256"] != digest(freeze)
                or any(data[key] != earlier[key] for key in fields)
                or any(set(x[key] for x in data["sources"]) & set(x[key] for x in earlier["sources"])
                       for key in ("speaker", "room"))):
            raise ValueError("Held inputs cross calibrated sources or the frozen rule")
    elif data["split"] != "development" or data["freeze_sha256"] is not None or calibrate(rows) != (data["profiles"], data["choices"]):
        raise ValueError("The recorded calibration does not reproduce")
    summary = report(rows, data["profiles"], data["choices"])
    if summary != data["summary"]:
        raise ValueError("The reported counts do not reproduce")
    return summary


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="action", required=True)
    drive = modes.add_parser("measure")
    drive.add_argument("split", choices=("development", "held"))
    for field in ("sources", "archive", "assets", "output"):
        drive.add_argument(field, type=Path)
    bounded = modes.add_parser("review")
    bounded.add_argument("result", type=Path)
    for tool in (drive, bounded):
        tool.add_argument("--freeze", type=Path)
    args = vars(cli.parse_args())
    if args.pop("action") == "measure":
        measure(**args)
    else:
        print(json.dumps(review(**args), sort_keys=True))
