# SPDX-License-Identifier: Apache-2.0
"""Evaluate the frozen sound rules on GRID recordings with verified word clocks."""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.signal import resample_poly

from app.bench.video_word_content import CARRIERS, ROLES
from app.detectors.audio.decode import decode_audio, wav_bytes
from app.detectors.video.decode import probe_frames
from app.evidence.video_reference_timing import _landmarker
from app.evidence.video_viseme_timing import SCORE_UNITS
from scripts import grid_reserve_word_clocks as clock
from scripts import video_lip_articulation_transfer as official
from scripts.review_video_lipsync_local_delay_transfer import source_pcm
from scripts.video_bilabial_mouth_transfer import treatments
from scripts.video_lip_content_transfer import splice
from scripts.video_lip_delivery_transfer import compare, qualified_48k
from scripts.video_visual_speech_train import picture_patches
from scripts.video_word_audio_windows import load
from scripts.video_word_content_model import frame_logits, json_file, pin, predict
from scripts.video_word_content_transfer import lookup_weights
from scripts.video_word_soundtrack_transfer import commitments, observer, seal


def donor(root, reference):
    data = (root / reference["audio_file"]).read_bytes()
    if pin(data) != reference["audio_sha256"]:
        raise ValueError("Donor publisher WAV differs")
    native = clock.wave_pcm(BytesIO(data), clock.PUBLISHER_RATE)
    words = reference["words"]
    a, b = words[4][1:]
    lead = words[0][1] - clock.WAV_PADDING
    if len(native) != words[-1][2] - words[0][1] + 2 * clock.WAV_PADDING + 1:
        raise ValueError("Unexpected official WAV trim")
    pcm = np.rint(resample_poly(native.astype(np.float64), 16, 25)).clip(-32768, 32767).astype("<i2")
    return pcm, [round(Fraction((t - lead) * 16, 25)) for t in (a, b)]


def source_data(root, row, mp, task):
    path = root / row["video_file"]
    pin_video = pin(path.read_bytes())
    if pin_video != row["video_sha256"]:
        raise ValueError("Publisher picture stream differs")
    table = probe_frames(path, cap=120, timeout_s=12)
    original, origin, dropped = source_pcm(path, pin_video)
    if (
        table.truncated
        or table.interpolated
        or dropped
        or table.pts[0] * table.tb != origin
        or origin
        or len(table) != 75
        or len(original) != 47648
    ):
        raise ValueError("Source MPEG presentation falls outside the verified clock profile")
    target = row["recordings"]["target"]
    official_wave = (root / target["audio_file"]).read_bytes()
    if pin(official_wave) != target["audio_sha256"]:
        raise ValueError("Publisher target WAV differs")
    timing = clock.project(
        target["words"], clock.wave_pcm(BytesIO(official_wave), clock.PUBLISHER_RATE), original, origin, table
    )
    span = timing["words"][4]["source_pcm_samples"]
    varieties = {
        name: values
        for name, values in treatments(original, span).items()
        if name in ("original", "gain", "global_early", "global_late")
    }
    for name in ("same_words", "different_words"):
        audio, bounds = donor(root, row["recordings"][name])
        varieties[name] = splice(original, span, audio, bounds)
    patches, valid, samples, readback = picture_patches(path, mp, task)
    if readback.to_json() != table.to_json():
        raise ValueError("The measured picture clock changed during extraction")
    talking = official.parent.pictures(path, len(original), mp, task, with_shape=True)
    talking["pts"] = list(table.pts)
    talking["frames"] = [
        {key: value for key, value in frame.items() if key not in {"gap", "sample"}}
        | {"gap_units": None if frame["gap"] is None else round(frame["gap"] * SCORE_UNITS)}
        for frame in talking["frames"]
    ]
    return span, varieties, patches, valid, samples, timing, table, talking


def main(args):
    cv2.setNumThreads(1)
    _, stable = commitments(args.previous, args.word_weight)
    frozen = json_file(args.freeze)
    if any(frozen[key] != value for key, value in stable.items()):
        raise ValueError("The transfer inputs differ from the acoustic freeze")
    label_alias = dict(
        zip(
            ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
            "z123456789",
            strict=True,
        )
    )
    roster = args.grid / "with-audio.json"
    rows = json.loads(roster.read_bytes())
    records, skipped = [], []
    sys.path.insert(0, str((args.assets / "runtime").resolve()))
    phone_reader, _ = official.audio_reader(args.assets / "checkpoint")
    acoustic, visible, digits, _ = lookup_weights(args.old_dir, frozen["configuration"])
    secondary = observer(load(args.word_weight))
    with (
        _landmarker(args.assets / "face_landmarker.task") as (mp, task),
        TemporaryDirectory(prefix="grid-word-corroboration-") as directory,
    ):
        wav, aac = Path(directory) / "audio.wav", Path(directory) / "audio.m4a"
        for item in rows:
            try:
                span, varieties, patches, valid, samples, timing, table, talking = source_data(
                    args.grid, item, mp, task
                )
            except (OSError, ValueError, AssertionError) as error:
                skipped.append(
                    {"speaker": item["speaker"], "source_sha256": item["video_sha256"], "reason": str(error)}
                )
                print(item["speaker"], "unassessed", str(error)[:150], flush=True)
                continue
            completed = {}
            for enc in CARRIERS:
                measurements = {}
                for kind in ROLES:
                    track = varieties[kind]
                    encoded = None
                    if enc == "aac48":
                        track, mapping, meta = qualified_48k(track, aac)
                        encoded = meta["file_sha256"]
                    else:
                        wav.write_bytes(wav_bytes(track.astype(np.float32) / 32768))
                        submitted = decode_audio(wav, source_sha256=pin(wav.read_bytes()))
                        mapping = submitted.mapping
                        assert np.array_equal(track, np.rint(submitted.pcm * 32768).astype("<i2"))
                    assessment = compare(
                        talking,
                        phone_reader(track),
                        mapping,
                        {"width": table.width, "height": table.height},
                    )
                    reading = assessment["both" if enc == "pcm" else "articulation"]
                    baseline = reading["witness"]["audio_source_pts"] if reading["witness"] else None
                    frame_reading = frame_logits(acoustic, track)
                    voice = predict(*frame_reading) if frame_reading is not None else None
                    content = secondary(track, patches, valid, samples, voice, visible, digits, frame_reading)
                    if voice:
                        voice = {k: round(v, 6) if isinstance(v, float) else v for k, v in voice.items()}
                    measurements[kind] = content | {
                        "speech": voice,
                        "sha256": pin(track.tobytes()),
                        "encoded_sha256": encoded,
                        "baseline": baseline,
                        "baseline_status": reading["state"],
                    }
                completed[enc] = measurements
            source_record = {
                "speaker": "grid/" + item["speaker"],
                "condition": "other_capture_setup",
                "source_sha256": item["video_sha256"],
                "source_clock": {"pts": list(table.pts), "time_base": table.time_base},
                "samples": len(varieties["original"]),
                "truth": span,
                "expected_visual": label_alias[item["recordings"]["target"]["words"][4][0]],
                "expected_audible": label_alias[item["recordings"]["different_words"]["words"][4][0]],
                "cases": completed,
                "timing": timing,
            }
            records.append(source_record)
            print(item["speaker"], "assessed", flush=True)
    inputs = {
        "source": pin(roster.read_bytes()),
        "scheduled": len(rows),
        "unassessed": skipped,
        "grid_protocol": pin(Path(__file__).read_bytes()),
    }
    seal(args.output, records, stable, inputs, phase="grid", freeze_path=args.freeze)
    print({"qualified": len(records), "unassessed": len(skipped)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("grid", "assets", "old_dir", "previous", "word_weight", "freeze", "output"):
        parser.add_argument(name, type=Path)
    main(parser.parse_args())
