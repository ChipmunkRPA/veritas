# SPDX-License-Identifier: Apache-2.0
"""Compare independently read audible and visible digits at the measured word time."""

import argparse
import gzip
import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from app.bench.video_word_content import AUDIO_MARGIN, AUDIO_MIN, CARRIERS, ROLES, cutoffs, recount
from scripts.video_lip_delivery_transfer import qualified_48k
from scripts.video_lip_sequence_panel import make_soundtracks
from scripts.video_word_content_model import (
    AcousticDigits,
    WordSequence,
    frame_logits,
    json_file,
    pin,
    predict,
    word_tensor,
)

BOUNDARY_SHIFTS = (-320, 320)
DELAY_TEST_SAMPLES = (-3840, -3200, -2560, -1920, -1280, 1280, 1920, 2560, 3200, 3840)
VISUAL_MIN = 0.50


def lookup_weights(runner, config):
    eye = runner / f"word-epoch-{config['visual_epoch']:02d}.safetensors"
    ear = runner / f"acoustic-epoch-{config['audio_epoch']:02d}.safetensors"
    expected = {p.name: pin(p.read_bytes()) for p in (eye, ear)}
    record = config.get("weights")
    if record is not None and expected != record:
        raise ValueError("Model differs from the development freeze")
    classes = json.loads((runner / "word-fit.json").read_bytes())["classes"]
    indexes = [i for i, name in enumerate(classes) if name.startswith("4:")]
    digits = [classes[i].split(":")[1] for i in indexes]
    model = WordSequence(len(classes))
    model.load_state_dict(load_file(str(eye)))
    model.eval()
    acoustic = AcousticDigits()
    acoustic.load_state_dict(load_file(str(ear)))
    acoustic.eval()

    def assess(observations):
        with torch.inference_mode():
            scores = model(torch.from_numpy(np.stack(observations)))[:, indexes]
        return scores.softmax(-1).numpy()

    return acoustic, assess, digits, expected


def visual_read(patches, valid, samples, voice, assess, digits):
    if not voice or voice["posterior"] < AUDIO_MIN or voice["margin"] < AUDIO_MARGIN:
        return None
    return word_window(patches, valid, samples, voice["span"], voice["digit"], assess, digits)


def word_window(patches, valid, samples, span, audible_digit, assess, digits):
    """Measure the fixed visual check at an independently qualified sound interval."""
    begin, end = span
    shifts = (0, *BOUNDARY_SHIFTS, *DELAY_TEST_SAMPLES)
    observations = {}
    for shift in shifts:
        sample = word_tensor(patches, valid, samples, (begin + shift, end + shift))
        if sample is not None:
            observations[shift] = sample
    if 0 not in observations:
        return None
    predictions = dict(zip(observations, assess(list(observations.values())), strict=True))
    principal = predictions[0]
    named = digits.index(audible_digit)
    visible = int(np.argmax(principal))

    def score(row):
        return float(row[visible] - row[named]) if visible != named and row[visible] >= VISUAL_MIN else 0.0

    perturbations = [predictions.get(shift) for shift in BOUNDARY_SHIFTS]
    shared = all(row is not None and int(np.argmax(row)) == visible for row in perturbations)
    alignable = any(
        predictions[k][named] >= 0.65 and int(np.argmax(predictions[k])) == named
        for k in DELAY_TEST_SAMPLES
        if k in predictions
    )
    central = score(principal)
    corroborated = min([central, *[score(row) for row in perturbations]]) if shared and not alignable else 0.0
    return {
        "visible_digit": digits[visible],
        "visible_peak": round(float(principal[visible]), 6),
        "audible_digit_picture_response": round(float(principal[named]), 6),
        "shared_nearby_read": bool(shared),
        "matches_audible_at_other_offset": bool(alignable),
        "one_window": round(central, 6),
        "nearby_time_check": round(corroborated, 6),
    }


def collect(folder, source_dir, plan, visuals, baseline, freeze=None, phase="calibration", *, observe=None):
    config = {"visual_epoch": 15, "audio_epoch": 8} if freeze is None else freeze["configuration"]
    acoustic, visible, digits, weights = lookup_weights(folder, config)
    if baseline.suffix != ".gz":
        raise ValueError("Use a committed source-clocked timing receipt")
    baseline_rows = {row["picture_sha256"]: row for row in json_file(baseline)["rows"]}
    captures = {row["picture_sha256"]: row for row in json_file(visuals / "capture.json.gz")["rows"]}
    selected = [row for row in json_file(plan)["rows"] if row["split"] == phase]
    values = []
    with (
        ZipFile(source_dir / "lombardgrid_audio.zip") as voices,
        TemporaryDirectory(prefix="chipmunk-word-measure-") as directory,
    ):
        output = Path(directory) / "audio.m4a"
        for index, row in enumerate(selected, 1):
            source = row["target"]
            picture = source["front_sha256"]
            cap = captures[picture]
            old = baseline_rows[picture]
            assert (old["pts"], old["time_base"], old["truth"]) == (
                cap["pts"],
                cap["time_base"],
                source["span"],
            )
            crops = (visuals / cap["file"]).read_bytes()
            assert pin(crops) == cap["sha256"]
            with np.load(BytesIO(crops), allow_pickle=False) as packets:
                patches, valid, samples = (packets[k] for k in ("patches", "valid", "samples"))
            variant = make_soundtracks(voices, row)
            cases = {}
            for delivery in CARRIERS:
                readings = {}
                for role in ROLES:
                    pcm = variant[role]
                    encoded = None
                    if delivery == "aac48":
                        pcm, _, meta = qualified_48k(pcm, output)
                        encoded = meta["file_sha256"]
                    baseline_values = old["cases"][delivery][role]
                    pcm_pin = pin(pcm.tobytes())
                    if "source_pcm_sha256" in baseline_values:
                        assert pcm_pin == baseline_values["source_pcm_sha256"]
                        if encoded:
                            assert encoded == baseline_values["carrier_sha256"]
                        baseline_span = baseline_values["current_span"]
                        state = baseline_values["current_status"]
                    else:
                        assert pcm_pin == baseline_values["audio_sha256"]
                        if encoded:
                            assert encoded == baseline_values["carrier_sha256"]
                        baseline_span = baseline_values["baseline"]
                        state = baseline_values["baseline_status"]
                    frame_reading = frame_logits(acoustic, pcm)
                    voice = predict(*frame_reading) if frame_reading is not None else None
                    if observe is None:
                        reading = {"mouth": visual_read(patches, valid, samples, voice, visible, digits)}
                    else:
                        reading = observe(pcm, patches, valid, samples, voice, visible, digits, frame_reading)
                    if voice:
                        voice = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in voice.items()}
                    readings[role] = {
                        **reading,
                        "speech": voice,
                        "sha256": pcm_pin,
                        "baseline": baseline_span,
                        "baseline_status": state,
                        "encoded_sha256": encoded,
                    }
                cases[delivery] = readings
            values.append(
                {
                    "speaker": row["speaker"],
                    "condition": row["condition"],
                    "source_sha256": picture,
                    "source_clock": {"pts": cap["pts"], "time_base": cap["time_base"]},
                    "samples": source["samples"],
                    "expected_visual": source["id"][-2],
                    "expected_audible": row["different_words"]["id"][-2],
                    "truth": source["span"],
                    "cases": cases,
                }
            )
            print(f"Measured {phase} source {index}/{len(selected)}", flush=True)
    return values, config | {"weights": weights}


def write(path, payload):
    path.write_bytes(gzip.compress((json.dumps(payload, sort_keys=True, indent=1) + "\n").encode(), mtime=0))


def main(args):
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    control = json_file(args.freeze) if args.freeze else None
    rows, config = collect(
        args.folder, args.source_dir, args.plan, args.visuals, args.baseline, control, args.phase
    )
    points = control["points"] if control else cutoffs(rows)
    if control:
        denied = {row["speaker"] for row in control["rows"]}
        assert not denied.intersection(row["speaker"] for row in rows)
    outcomes = recount(rows, points)
    value = {
        "schema": "chipmunk.video-word-content/1",
        "configuration": config,
        "points": points,
        "source": pin(args.plan.read_bytes()),
        "raw_visual_input": pin((args.visuals / "capture.json.gz").read_bytes()),
        "baseline_pin": pin(args.baseline.read_bytes()),
        "freeze": pin(args.freeze.read_bytes()) if args.freeze else None,
        "rows": rows,
        "outcomes": outcomes,
    }
    write(args.output, value)
    print(json.dumps({"points": points, "outcomes": outcomes}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("folder", "source_dir", "plan", "visuals", "baseline", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--phase", default="calibration")
    parser.add_argument("--freeze", type=Path)
    main(parser.parse_args())
