# SPDX-License-Identifier: Apache-2.0
"""Compare stereo repetition methods on exact AAC deliveries and matched controls."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.audio_native_alignment_panel import (
    DURATION,
    counts,
    deliveries,
    digest,
    read,
    save,
    save_wav,
    segment,
)

SCHEMA = "chipmunk.aac-stereo-repetition-panel/1"
RATES = (96, 160)
HELD_VOICES = {
    "ES2004a": "c6f0eafaedd3632da4cb3f14788ee83a8efae2b456f70eefa8e98b2b6992102a",
    "TS3004a": "3f12bd8637e86b937e7962d2df246c6ffe49d996a8a19412737ce02a8d7ce8b0",
}
SCENE_STARTS = (26, 59, 88, 112, 149, 178, 215, 255)
VOICE_STARTS = (69, 83, 116, 143, 168, 200, 229, 270)
SHIFTS = (5, -19, 26, -31, 9, -13, 20, 35)


def prepare(root, output):
    import numpy as np
    from scipy.signal import resample_poly

    output.mkdir(mode=0o700)
    scenes = read(root / "recorded-sources.json")
    if {row["scene"] for row in scenes} != {"DKITCHEN", "NFIELD", "STRAFFIC", "TMETRO"}:
        raise ValueError("Expected four held stereo recordings")
    voices = {name: root / f"{name}.Array1-05.wav" for name in HELD_VOICES}
    if any(digest(path) != HELD_VOICES[name] for name, path in voices.items()):
        raise ValueError("Foreground recordings differ from the pinned sources")
    rows = []
    for position, scene in enumerate(scenes):
        name = scene["scene"]
        files = [root / name / Path(row["member"]).name for row in scene["members"]]
        if any(digest(path) != member["sha256"] for path, member in zip(files, scene["members"], strict=True)):
            raise ValueError("Held background channels differ from the pinned sources")
        meeting = tuple(voices)[position % len(voices)]
        for index, start in enumerate(SCENE_STARTS):
            audio = np.column_stack([segment(path, start, DURATION, 48_000) for path in files])
            different = np.column_stack([segment(path, start + 7, 30720, 48_000) for path in files])
            voice_start = VOICE_STARTS[index] + 12 * (position // 2)
            voice = resample_poly(segment(voices[meeting], voice_start, DURATION // 3, 16_000).astype(float), 3, 1)
            original = list(deliveries(audio, different, voice, index, offset=SHIFTS[index]))
            first, second = original[-1][2]
            repeated_voice = voice.copy()
            repeated_voice[slice(*second)] = voice[slice(*first)]
            foreground = next(stereo for role, stereo, _ in deliveries(
                audio, different, repeated_voice, index, offset=SHIFTS[index]) if role == "ordinary_panned")
            variants = (*original, ("repeated_foreground", foreground, None),
                        ("same_channel_twice", np.repeat(audio[:, :1], 2, axis=1), None))
            for role, stereo, truth in variants:
                path = output / f"{name}-{index}-{role}.wav"
                save_wav(path, stereo)
                rows.append({"scene": name, "role": role, "start_seconds": start, "meeting": meeting,
                             "speech_start_seconds": voice_start, "actual_offset": SHIFTS[index], "truth": truth,
                             "delivery": path.name, "input_sha256": digest(path)})
        print(name, len(rows), flush=True)
    save(output / "result.json.gz", {"rows": rows, "split": "confirmation", "scenes": scenes,
                                    "foreground_sha256": HELD_VOICES})


def _compare(path, sha):
    import numpy as np

    from app.detectors.audio.decode import decode_audio, decode_native_stereo
    from app.evidence import (
        audio_aac_alignment,
        audio_exact_repetition,
        audio_native_alignment,
        audio_similarity,
        audio_waveform_similarity,
    )

    decoded = decode_audio(path, source_sha256=sha)
    pcm = decode_native_stereo(path, decoded.mapping, aac=True)
    if pcm is None or len(pcm) != DURATION * 4 or decoded.mapping["source_start_pts"] != 0:
        raise ValueError("AAC delivery differs from the zero-based presented sample clock")
    samples = np.frombuffer(pcm, dtype="<i2").reshape(-1, 2)
    mono = np.rint(decoded.pcm * 32768).astype("<i2").tobytes()
    procedures = (audio_exact_repetition.find_repeated_spans, audio_similarity.find_similar_spans,
                  audio_waveform_similarity.find_similar_spans)
    for method in procedures:
        serving = method(mono)
        if serving.spans:
            break
    responses = {"serving_mono": {"assessed": serving.status == "measured", "reason": serving.reason,
                                  "spans": [tuple((3 * a, 3 * b) for a, b in (p.first, p.second)) for p in serving.spans]}}
    for name, method in (("unchanged_stereo", audio_native_alignment), ("lower_band_stereo", audio_aac_alignment)):
        fitted, reason, pairs = method.search(samples)
        responses[name] = {"assessed": reason is None, "reason": reason, "spans": pairs,
                           "alignment": vars(fitted) if fitted else None}
    # When a serving check already reports a pair, retain that existing evidence first.
    responses["serving_plus_lower_band"] = (responses["serving_mono"] if serving.spans
                                           else responses["lower_band_stereo"])
    return responses, pcm, decoded.mapping


def _deliver(row, bitrate, source, output):
    import hashlib

    parent = source / row["delivery"]
    if digest(parent) != row["input_sha256"]:
        raise ValueError("Expected exact source WAV for AAC encoding")
    media = output / f"{parent.stem}-{bitrate}.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(parent), "-c:a", "aac", "-b:a", f"{bitrate}k",
                    "-map_metadata", "-1", str(media)], check=True, timeout=35)
    sha = digest(media)
    responses, pcm, mapping = _compare(media, sha)
    return {**row, "delivery": media.name, "parent_sha256": row["input_sha256"], "input_sha256": sha,
            "presented_pcm_sha256": hashlib.sha256(pcm).hexdigest(), "presented_samples": len(pcm) // 4,
            "decoded_mono_samples": mapping["samples"], "source_start_pts": mapping["source_start_pts"],
            "source_rate": mapping["source_sample_rate"], "observed_bit_rate": mapping["source_bit_rate"],
            "bitrate_setting": bitrate, "responses": responses}


def totals(rows):
    return {str(bitrate): counts(row for row in rows if row["bitrate_setting"] == bitrate) for bitrate in RATES}


def measure(source, output):
    from app.evidence import audio_aac_alignment, audio_native_alignment

    output.mkdir(mode=0o700)
    plan = read(source / "result.json.gz")
    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(_deliver, row, bitrate, source, output) for row in plan["rows"] for bitrate in RATES]
        for future in pending:
            rows.append(future.result())
            if len(rows) % 96 == 0:
                print(len(rows), "measurements complete", flush=True)
    save(output / "result.json.gz", {"schema": SCHEMA, "split": plan["split"], "rows": rows, "counts": totals(rows),
                                    "scenes": plan["scenes"], "foreground_sha256": plan["foreground_sha256"],
                                    "recipe_sha256": digest(Path(__file__)), "implementation_sha256": {
                                        "unchanged_stereo": digest(Path(audio_native_alignment.__file__)),
                                        "lower_band_stereo": digest(Path(audio_aac_alignment.__file__))},
                                    "encoder": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]})


def review(receipt):
    record = read(receipt)
    output = totals(record["rows"])
    if record["schema"] != SCHEMA or output != record["counts"]:
        raise ValueError("Measured AAC results do not recount")
    print(json.dumps(output, sort_keys=True))


def _audit_delivery(row, encoded, parents):
    import hashlib

    import numpy as np

    from app.evidence.audio_aac_alignment import search

    path = encoded / row["delivery"]
    parent = parents / f"{path.stem.rsplit('-', 1)[0]}.wav"
    if digest(path) != row["input_sha256"] or digest(parent) != row["parent_sha256"]:
        raise ValueError(f"Audio delivery or parent differs for {path.name}")
    probe = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "a:0",
         "-show_frames", "-show_entries", "frame=pts,nb_samples:stream=codec_name,profile,sample_rate,channels,"
         "time_base,start_pts,duration_ts", "-of", "json", str(path)], timeout=20))
    stream, = probe["streams"]
    count = row["presented_samples"]
    if stream != {"codec_name": "aac", "profile": "LC", "sample_rate": "48000", "channels": 2,
                  "time_base": "1/48000", "start_pts": 0, "duration_ts": count}:
        raise ValueError(f"Audio track clock differs for {path.name}")
    cursor = 0
    for frame in probe["frames"]:
        if frame["pts"] != cursor or frame["nb_samples"] <= 0:
            raise ValueError(f"Audio frame clock discontinuity in {path.name}")
        cursor += frame["nb_samples"]
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", "file", "-copyts", "-i", str(path),
         "-map", "0:a:0", "-c:a", "pcm_s16le", "-f", "s16le", "-"], timeout=20)
    if len(raw) != 4 * cursor or not 0 <= cursor - count < 1024:
        raise ValueError(f"Audio frame extent differs for {path.name}")
    presented = raw[:4 * count]
    if hashlib.sha256(presented).hexdigest() != row["presented_pcm_sha256"]:
        raise ValueError(f"Presented samples differ for {path.name}")
    fitted, reason, spans = search(np.frombuffer(presented, dtype="<i2").reshape(-1, 2))
    expected = row["responses"]["lower_band_stereo"]
    if (reason != expected["reason"] or (vars(fitted) if fitted else None) != expected["alignment"]
            or [[list(span) for span in pair] for pair in spans] != expected["spans"]):
        raise ValueError(f"Released measurement differs from the frozen reading for {path.name}")
    return len(probe["frames"]), cursor - count


def audit(receipt, parents, encoded, output):
    from app.evidence import audio_aac_alignment

    record = read(receipt)
    if record["schema"] != SCHEMA or record["counts"] != totals(record["rows"]):
        raise ValueError("The independent comparison counts differ")
    frames = padding = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for index, (frame_count, tail) in enumerate(pool.map(
            lambda row: _audit_delivery(row, encoded, parents), record["rows"]
        ), 1):
            frames += frame_count
            padding += tail
            if index % 96 == 0:
                print(index, "audio bindings checked", flush=True)
    save(output, {"schema": "chipmunk.aac-stereo-repetition-bindings/1", "result_sha256": digest(receipt),
                  "delivered_files_verified": len(record["rows"]), "parent_file_bindings_verified": len(record["rows"]),
                  "audio_frames_verified": frames, "unpresented_samples": padding,
                  "released_detector_sha256": digest(Path(audio_aac_alignment.__file__))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser("prepare")
    build.add_argument("root", type=Path)
    build.add_argument("output", type=Path)
    measure_ = actions.add_parser("measure")
    measure_.add_argument("source", type=Path)
    measure_.add_argument("output", type=Path)
    actions.add_parser("review").add_argument("receipt", type=Path)
    audit_ = actions.add_parser("audit")
    for name in ("receipt", "parents", "encoded", "output"):
        audit_.add_argument(name, type=Path)
    args = vars(parser.parse_args())
    globals()[args.pop("action")](**args)
