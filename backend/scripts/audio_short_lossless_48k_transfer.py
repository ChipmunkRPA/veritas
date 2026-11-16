# SPDX-License-Identifier: MIT
"""Test the frozen short speech scan on matched 48 kHz lossless deliveries."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import wave
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from app.evidence import audio_short_review_binding as binding
from scripts import audio_short_edit_input_views as prior
from scripts.audio_background_lossless_rate_transfer import ffmpeg, require, sha

PARENT = prior.duration.previous.PANELS / "audio-short-edit-input-views-2026-11-15.json.gz"
SCHEMA = "chipmunk.audio-short-lossless-48k/1"
CARRIERS = ("wav", "flac")
METHOD = {
    "input": "Render the retained six-second 16 kHz PCM to 48 kHz without dither, encode the same native PCM in WAV and FLAC, and decode through the app. Pin the exact original clock and both carrier and decoded hashes.",
    "scoring": "Use the existing 16 kHz lossless speech-admission rule, schedule and point without refitting. Compare each delivery with its archived 16 kHz result on the same source and label.",
    "admission": "Keep all useful locations on panels with fewer than three earlier hits, and at least 80 percent otherwise. Allow up to two new genuine flags per panel and no new isolated-background flags. Check each carrier separately.",
}


def deliver(raw, folder):
    rendered = ffmpeg("-f", "s16le", "-ar", str(binding.RATE), "-ac", "1", "-i", "pipe:0",
                      "-af", "aresample=48000:dither_method=none", "-f", "s16le", "pipe:1", pcm=raw)
    require(len(rendered) == 2 * 6 * binding.LOSSLESS_RATE, "Native lossless render has the wrong extent")
    wav, flac = (folder / f"delivery.{ext}" for ext in CARRIERS)
    write_wav(wav, rendered)
    ffmpeg("-y", "-i", str(wav), "-c:a", "flac", str(flac))
    for file in (wav, flac):
        require(ffmpeg("-i", str(file), "-f", "s16le", "pipe:1") == rendered,
                "Lossless carrier changed native source samples")
    return sha(rendered), {"wav": wav, "flac": flac}


def write_wav(path, pcm, *, rate=binding.LOSSLESS_RATE, channels=1, depth=2):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((channels, depth, rate, 0, "NONE", "not compressed"))
        stream.writeframes(pcm)


def read_record(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def evaluate(parent, measured):
    require(set(measured) == set(prior.PANELS), "Held panel is incomplete")
    results = {}
    for panel, rows in measured.items():
        results[panel] = {carrier: Counter() for carrier in CARRIERS}
        cases = parent["plan"]["cases"][panel]
        baselines = parent["readings"][panel]
        for case, baseline, record in zip(cases, baselines, rows, strict=True):
            require(case["pcm_sha256"] == baseline["pcm_sha256"] == record["source_pcm_sha256"], "Source PCM differs")
            require(set(record["deliveries"]) == set(CARRIERS), "Missing lossless delivery")
            truth = case["truth"]
            category = "edited" if truth else "background" if case["kind"] == "background" else "genuine"
            old = baseline["current"]["range"]
            for carrier, result in record["deliveries"].items():
                windows = result["responses"]
                keys = [(item["start_sample"], item["end_sample"]) for item in windows]
                require(keys == sorted(set(keys)) and set(keys) <= set(binding.GRID)
                        and all(math.isfinite(item["logit"]) for item in windows), "Invalid short speech inputs")
                selected = binding.selected_range(windows)
                new = [selected["start_sample"], selected["end_sample"]] if selected else None
                old_useful = bool(truth and prior.duration.compatible(old, truth))
                new_useful = bool(truth and prior.duration.compatible(new, truth))
                counts = results[panel][carrier]
                counts.update({f"{category}_files": 1, f"{category}_old_assessed": bool(baseline["current"]["assessed"]),
                               f"{category}_assessed": bool(windows), f"{category}_old_flags": bool(old),
                               f"{category}_flags": bool(new), f"{category}_added_flags": bool(new and not old),
                               f"{category}_lost_flags": bool(old and not new), f"{category}_range_changes": old != new,
                               f"{category}_old_useful": old_useful, f"{category}_useful": new_useful,
                               f"{category}_lost_useful": old_useful and not new_useful})
    admission = {}
    for panel, carriers in results.items():
        admission[panel] = {}
        for carrier, counts in carriers.items():
            old = counts["edited_old_useful"]
            lost = counts["edited_lost_useful"]
            admission[panel][carrier] = (lost == 0 if old < 3 else 5 * (old - lost) >= 4 * old) and (
                counts["genuine_added_flags"] <= 2 and counts["background_added_flags"] == 0)
    return {"counts": results, "admission": admission}


def capture(args):
    from app.detectors.audio import short_voice_review
    from app.detectors.audio.decode import decode_audio
    from scripts.run_audio_specialists_v15 import load_voice

    class Deadline:
        def check(self, stage):
            pass

    parent = read_record(PARENT)
    roots = {"fresh": args.fresh, "diagnostic": args.diagnostic, "phoneme": args.room, "stock": args.room}
    assets = parent["plan"]["protocol"]["current_assets"]
    model = load_voice({"weights": args.weights, "assets": assets})
    activity = prior.ROOM.runtime(args.weights / "silero_vad.jit")
    measured = {}
    with TemporaryDirectory() as scratch:
        for panel in prior.PANELS:
            rows = measured[panel] = []
            for case in parent["plan"]["cases"][panel]:
                original = case["pcm_sha256"]
                raw = (roots[panel] / "pcm" / f"{original}.pcm").read_bytes()
                require(len(raw) == 2 * binding.SAMPLES and sha(raw) == original, "Held source waveform differs")
                native_pin, files = deliver(raw, Path(scratch))
                deliveries, scans = {}, {}
                for carrier, path in files.items():
                    carrier_pin = sha(path.read_bytes())
                    decoded = decode_audio(path, source_sha256=carrier_pin)
                    require(binding.supported_resampled_lossless(decoded.mapping), "Lossless delivery changed the verified presentation")
                    pin = decoded.mapping["pcm_sha256"]
                    if pin not in scans:
                        result = short_voice_review.scan(decoded.pcm, activity.spans(decoded.pcm), model, Deadline())
                        scans[pin] = result["responses"]
                    deliveries[carrier] = {"carrier_sha256": carrier_pin, "decoded_pcm_sha256": pin, "responses": scans[pin]}
                rows.append({"source_pcm_sha256": original, "native_pcm_sha256": native_pin, "deliveries": deliveries})
                if len(rows) % 10 == 0 or len(rows) == len(parent["plan"]["cases"][panel]):
                    print(f"Checked {len(rows)} matched {panel} recordings in both lossless formats.", flush=True)
    receipt = {"schema": SCHEMA, "method": METHOD, "parent_sha256": sha(PARENT.read_bytes()), "point": binding.REVIEW_POINT,
               "rows": measured, "result": evaluate(parent, measured)}
    with args.output.open("xb") as stream:
        stream.write(gzip.compress(json.dumps(receipt, sort_keys=True, allow_nan=False).encode() + b"\n", mtime=0))
    return receipt["result"]


def review(path):
    receipt = read_record(path)
    require(receipt["schema"] == SCHEMA and receipt["method"] == METHOD and receipt["point"] == binding.REVIEW_POINT
            and receipt["parent_sha256"] == sha(PARENT.read_bytes()), "The frozen timing protocol differs")
    result = evaluate(read_record(PARENT), receipt["rows"])
    require(receipt["result"] == result, "The result differs from numeric replay")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    native = phases.add_parser("capture")
    for name in ("fresh", "diagnostic", "room", "weights", "output"):
        native.add_argument(name, type=Path)
    replay = phases.add_parser("review")
    replay.add_argument("receipt", type=Path)
    args = parser.parse_args()
    result = capture(args) if args.phase == "capture" else review(args.receipt)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
