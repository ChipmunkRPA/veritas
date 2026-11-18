# SPDX-License-Identifier: Apache-2.0
"""Replay the frozen speech-activity check on new mono video soundtracks."""

from __future__ import annotations

import argparse
from contextlib import closing
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

from scripts import video_shared_sound_panel as stereo
from scripts import video_speech_activity_gap_panel as activity
from scripts.video_stereo_covariance_panel import comparisons, file_hash, read, save

SCHEMA = "chipmunk.video-mono-activity/1"
EDIT_ROLES = {"hard", "faded", "clean_muted"}
CLEAN = 100


def treatments(pcm, room, target, snr):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    start, stop = target
    voice, bed = pcm.astype(np.float64), room.astype(np.float64)
    speech_rms = np.sqrt(np.mean(voice[start:stop] ** 2))
    ambient_rms = np.sqrt(np.mean(bed[start:stop] ** 2))
    if min(speech_rms, ambient_rms) < 8:
        raise ValueError("The source regions need both recorded speech and room sound")
    bed *= speech_rms / ambient_rms * 10 ** (-snr / 20)
    scale = min(1., 25_000 / (np.abs(voice).max() + np.abs(bed).max()))
    voice *= scale
    bed *= scale
    full = voice + bed
    hard = np.ones(len(voice))
    hard[start:stop] = 0
    soft = hard.copy()
    ramp = np.linspace(1., 0., 400)
    soft[start:start + len(ramp)] = ramp
    soft[stop-len(ramp):stop] = ramp[::-1]
    lowpass = sosfiltfilt(butter(2, 2300, fs=stereo.RATE, output="sos"), full)
    bank = {"unmodified_mix": full, "whole_gain": .6 * full, "local_duck": (.3 + .7 * soft) * full,
            "local_eq": soft * full + (1 - soft) * lowpass, "background_reduction": voice + (.22 + .78 * soft) * bed,
            "hard": voice * hard + bed, "faded": voice * soft + bed}
    if any(np.max(np.abs(audio)) >= 32768 for audio in bank.values()):
        raise ValueError("The mono test mix exceeded the signed playback range")
    return {name: np.rint(samples).astype("<i2") for name, samples in bank.items()}


def render_pictures(source, picture, assets):
    from app.detectors.video.decode import decode_stream, probe_frames
    from app.detectors.video.shots import detect_shots
    from scripts.video_covariance_sources import mouth_frames
    from scripts.video_silent_mouth_clock_panel import VIDEO_ENCODING, pixels

    original = probe_frames(source)
    ticks = [i * 1000 for i in range(len(original))]
    stereo.ffmpeg("-i", str(source), "-an", "-map", "0:v:0", "-vf", "settb=1/30000,setpts=N*1000",
                  *VIDEO_ENCODING, "-bsf:v", "setts=duration=1000", str(picture))
    table = probe_frames(picture)
    rgb = pixels(source, original)
    if (table.tb != Fraction(1, 30_000) or list(table.pts) != ticks or pixels(picture, table) != rgb):
        raise ValueError("The mono delivery changed the original frame pixels or frozen picture clock")
    with closing(decode_stream(picture, table)) as stream:
        shots = detect_shots(stream, table).as_json()
    if shots["frames_scanned"] != len(table) or shots["ranges"] != [[0, len(table)]]:
        raise ValueError("The reserved original is outside the supported single-shot picture scope")
    stream_hash = stereo.ffmpeg("-i", str(picture), *stereo.STREAM_HASH).decode().strip()
    clock = {"ticks": ticks, "picture_stream": stream_hash, "rgb_sha256": stereo.sha("\n".join(rgb).encode())}
    return original, table, clock | {"frames": mouth_frames(picture, clock, assets)}


def evaluate_delivery(picture, out, table, record, carrier, samples, vad, enhancer):
    import numpy as np

    from app.bench.video_speech_activity_gap import scan
    from app.detectors.audio.decode import decode_audio
    from app.detectors.video.decode import probe_frames
    from app.detectors.video.viseme_review import sound_levels
    from app.evidence.video_silent_mouth import record as near_silence
    from app.evidence.video_viseme_binding import audio_origin, presented_samples
    from scripts.audio_noise_view_v4 import enhance

    options = ("-c:a", "pcm_s16le") if carrier == "pcm" else ("-c:a", "aac", "-ar", "48000", "-b:a", "112k")
    stereo.ffmpeg("-i", str(picture), "-f", "s16le", "-ar", str(stereo.RATE), "-ac", "1", "-i", "pipe:0",
                  "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *options,
                  "-video_track_timescale", "30000", "-movie_timescale", "240000", str(out), data=samples.tobytes())
    native = probe_frames(out)
    if ((native.pts, native.time_base) != (table.pts, table.time_base)
            or stereo.ffmpeg("-i", str(out), *stereo.STREAM_HASH).decode().strip() != record["picture_stream"]):
        raise ValueError("The mono soundtrack treatment moved or changed source pictures")
    digest = file_hash(out)
    decoded = decode_audio(out, source_sha256=digest)
    extent = presented_samples(decoded.mapping)
    if (decoded.mapping["source_channels"] != 1 or audio_origin(decoded.mapping) != 0
            or extent != len(samples) or carrier == "pcm" and not np.array_equal(decoded.pcm * 32768, samples)):
        raise ValueError("The delivered mono soundtrack differs from the planned source sample clock")
    pcm = np.asarray(decoded.pcm[:extent], dtype="float32")
    reduced = enhance(pcm, enhancer)
    candidate = scan(activity.activity(vad, pcm), activity.activity(vad, reduced), record["frames"])
    measured = near_silence(sound_levels(pcm, extent), record["frames"], extent, 0)
    return {"delivery": out.name, "input_sha256": digest, "samples": extent,
            "mono_pcm_sha256": stereo.sha(np.rint(pcm * 32768).astype("<i2").tobytes()),
            "candidate": candidate, "near_silence": measured[0]["span"] if measured else None}


def source_rows(source, room, number, assets, output, vad, enhancer):
    import numpy as np
    from scipy.signal import resample_poly

    from app.detectors.audio.decode import decode_audio
    from scripts.audio_stereo_ambience_panel import _archive_pcm

    group = f"source-{number:02}"
    picture = output / f"{group}-pictures.mp4"
    original, table, clock = render_pictures(source, picture, assets)
    pcm, shift = stereo.source_sound(decode_audio(source, source_sha256=file_hash(source)), original)
    count = round(len(table) * stereo.RATE / 30)
    pcm = stereo.fit_sound(pcm, count)
    start = 4 + number % 13
    room = _archive_pcm(room)[start * 44100:(start + 4) * 44100, number % 2]
    bed = resample_poly(room.astype(np.float64), 160, 147)[:count]
    if len(bed) != count:
        raise ValueError("The recorded room sound does not cover the planned duration")
    target = next(truth for name, _, truth in stereo.treatments(pcm, 0, number) if name == "muted")
    clean = pcm.copy()
    clean[target[0]:target[1]] = 0
    mixes = {snr: treatments(pcm, bed, target, snr) for snr in stereo.SNR_LEVELS}
    mixes[CLEAN] = {"clean_unmodified": pcm.astype("<i2"), "clean_muted": clean.astype("<i2")}
    rows = []
    for snr, roles in mixes.items():
        for role, samples in roles.items():
            for carrier in stereo.CARRIERS:
                path = output / f"{group}-{snr:+d}-{carrier}-{role}.mov"
                measured = evaluate_delivery(picture, path, table, clock, carrier, samples, vad, enhancer)
                rows.append(measured | {"group": group, "role": role, "carrier": carrier, "snr_db": snr,
                                        "truth": target if role in EDIT_ROLES else None})
    return rows, {key: value for key, value in clock.items() if key != "frames"} | {
        "group": group, "duration_samples": count, "room_channel": number % 2, "ambient_start_s": start,
        "source_trim_samples": shift,
    }


def summary(rows, choices):
    def candidate(row):
        choice = choices[row["carrier"]]
        return activity.pick(row, choice["arm"], choice["point"])

    def paired(row):
        reading, assessed = candidate(row)
        return (row["near_silence"] or reading), assessed or row["near_silence"] is not None

    return {"candidate": comparisons(rows, candidate), "near_silence": comparisons(rows, lambda row: (row["near_silence"], None)),
            "preserve_near_silence": comparisons(rows, paired)}


def measure(plan, inventory, archive, assets, vad, enhancer, freeze, output):
    import torch

    from app.bench import video_speech_activity_gap as method
    from scripts import audio_noise_view_v4 as reducer
    from scripts import video_covariance_sources as sources
    from scripts import video_mono_activity_sources as reservation

    roster, files, settings = read(plan), read(inventory / "sources.json.gz"), read(freeze)
    choices = {carrier: arm["selected"] for carrier, arm in settings["decisions"].items()}
    if (roster["evaluation"] != reservation.SCHEMA or roster["operating_points"] != choices
            or any(arm["arm"] != "both" for arm in choices.values()) or file_hash(freeze) != roster["comparison_file_sha256"][reservation.FILES[3]]
            or files["plan_sha256"] != file_hash(plan) or len(files["entries"]) != sources.HELD_SPEAKERS
            or file_hash(Path(method.__file__)) != settings["assets"]["method"] or file_hash(vad) != activity.VAD_SHA):
        raise ValueError("The mono comparison requires frozen stereo settings, verified assets and the reserved sources")
    sources.check_archive(archive)
    sources.load_assets(assets)
    torch.set_num_threads(1)
    model = torch.jit.load(str(vad), map_location="cpu").eval()
    cleaned = reducer.load(enhancer)
    output.mkdir(mode=0o700)
    rows, sources_used = [], []
    with ZipFile(archive) as bundle:
        for number, (entry, planned) in enumerate(zip(files["entries"], roster["held"], strict=True), 1):
            clip = inventory / entry["name"]
            room = entry["room"]
            room_bytes = bundle.read(room["name"])
            if (entry["speaker"] != planned["speaker"] or room != planned["room"] or entry["name"] not in planned["options"]
                    or file_hash(clip) != entry["sha256"] or stereo.sha(room_bytes) != room["sha256"]):
                raise ValueError("A publisher input differs from its frozen mono reservation")
            group, clock = source_rows(clip, room_bytes, number, assets, output, model, cleaned)
            rows += group
            sources_used.append(entry | clock)
            print(f"Measured {number}/{len(files['entries'])} speakers ({len(rows)} exact files)", flush=True)
    document = {"schema": SCHEMA, "plan_sha256": file_hash(plan), "freeze_sha256": file_hash(freeze), "points": choices,
                "sources": sources_used, "rows": rows, "summary": summary(rows, choices)}
    save(output / "receipt.json.gz", document)
    print(document["summary"], flush=True)


def review(receipt, freeze, plan):
    document, settings, frozen = read(receipt), read(freeze), read(plan)
    choice = {carrier: arm["selected"] for carrier, arm in settings["decisions"].items()}
    if (document["schema"] != SCHEMA or document["freeze_sha256"] != file_hash(freeze) or document["points"] != choice
            or frozen["operating_points"] != choice or file_hash(plan) != document["plan_sha256"]
            or document["summary"] != summary(document["rows"], choice)):
        raise ValueError("The exported mono result does not reproduce with the frozen plan and points")
    print(document["summary"])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="mode", required=True)
    for function, fields in {measure: ("plan", "inventory", "archive", "assets", "vad", "enhancer", "freeze", "output"),
                             review: ("receipt", "freeze", "plan")}.items():
        parser = commands.add_parser(function.__name__)
        parser.set_defaults(action=function)
        for name in fields:
            parser.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    function = args.pop("action")
    args.pop("mode")
    function(**args)
