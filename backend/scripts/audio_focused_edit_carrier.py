# SPDX-License-Identifier: MIT
"""Replay the frozen six-second focused speech check through 48 kHz lossless uploads."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import wave
from collections import Counter
from pathlib import Path

from app.evidence import audio_focused_edit_binding as binding
from scripts.audio_focused_context_transfer import digest, read, save

SCHEMA = "chipmunk.audio-focused-edit-carrier/1"
RATE = 48000
CARRIERS = ("wav", "flac")


def render(raw, folder):
    source = folder / "native.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setparams((1, 2, binding.RATE, binding.SAMPLES, "NONE", "not compressed"))
        stream.writeframes(raw)
    wav = folder / "48k.wav"
    flac = folder / "48k.flac"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source), "-ar", str(RATE),
                    "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(wav), "-c:a", "flac", str(flac)], check=True)
    return wav, flac


def interval(reading):
    if reading is None or reading["score_units"] <= binding.REVIEW_POINT:
        return None
    return [reading["start_sample"], reading["end_sample"]]


def summarize(rows, carrier):
    counts = Counter(deliveries=len(rows), transport_pairs=len({row["input_pcm_sha256"] for row in rows}))
    for row in rows:
        earlier, measured = row["original"], row["carriers"][carrier]["reading"]
        for key, reading in (("original", earlier), ("resampled", measured)):
            if reading is None:
                counts[f"{key}_unassessed"] += 1
                continue
            assert binding.on_grid(reading["start_sample"], reading["end_sample"])
            assert type(reading["score_units"]) is int
            if interval(reading) is not None:
                counts[f"{key}_{row['kind'].split('/')[0]}_flagged"] += 1
        counts["matching_assessment"] += (earlier is None) == (measured is None)
        counts["matching_flagged_interval"] += interval(earlier) == interval(measured)
        counts["equal_best_interval"] += (earlier is not None and measured is not None
                                          and earlier["start_sample"] == measured["start_sample"]
                                          and earlier["end_sample"] == measured["end_sample"])
    return dict(counts)


def score(args):
    from app.detectors.audio import focused_edit_review
    from app.detectors.audio.decode import decode_audio
    from app.engine.boundary import Deadline
    from app.evidence.audio_short_review_binding import supported_resampled_lossless
    from scripts.audio_clean_teacher_transfer import runtime

    parent = read(args.study)
    activity = runtime(args.weights / "silero_vad.jit")
    model = focused_edit_review.load_model(focused_edit_review.MODEL_ID, "cpu", "float32")
    head, = [row for row in parent["heads"] if row["arm"] == "both"]
    assert head["sha256"] == model.revision
    cache = {}
    rows = []
    with tempfile.TemporaryDirectory(prefix="chipmunk-focused-carrier-") as temp:
        folder = Path(temp)
        for index, row in enumerate(parent["rows"], 1):
            pin = row["pcm_sha256"]
            if pin not in cache:
                raw = (args.media / "pcm" / f"{pin}.pcm").read_bytes()
                assert digest(raw) == pin and len(raw) == 2 * binding.SAMPLES
                observed, measured = {}, {}
                for path in render(raw, folder):
                    source_hash = digest(path.read_bytes())
                    original_bytes = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                                                             "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"])
                    assert len(original_bytes) == 6 * RATE * 2
                    decoded = decode_audio(path, source_sha256=source_hash, max_seconds=7)
                    assert supported_resampled_lossless(decoded.mapping)
                    pin_16k = decoded.mapping["pcm_sha256"]
                    if pin_16k not in measured:
                        measured[pin_16k] = focused_edit_review.scan(
                            decoded.pcm, activity.spans(decoded.pcm), model.obj, Deadline(None),
                        )
                    reading = measured[pin_16k]
                    observed[path.suffix[1:]] = {
                        "sha256": source_hash, "decoded_pcm_sha256": pin_16k,
                        "native_pcm_sha256": digest(original_bytes), "native_samples": decoded.mapping["source_duration_pts"],
                        "reading": reading["best"], "checked_windows": reading["checked_windows"]}
                assert len({fields["native_pcm_sha256"] for fields in observed.values()}) == 1
                cache[pin] = {"carriers": observed}
            original = row["candidates"]["both"]
            before = ({"start_sample": original["range"][0], "end_sample": original["range"][1],
                       "score_units": round(original["score"] * binding.SCORE_SCALE)} if original else None)
            rows.append({"panel": row["panel"], "group": row["group"], "kind": row["kind"], "truth": row["truth"],
                         "input_pcm_sha256": pin, "original": before} | cache[pin])
            if index % 34 == 0 or index == len(parent["rows"]):
                print(f"Verified lossless upload clocks and focused responses for {index}/{len(parent['rows'])} deliveries.", flush=True)
    receipt = {"schema": SCHEMA, "input_sha256": digest(args.study.read_bytes()), "head_sha256": model.revision,
               "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
               "rows": rows, "summary": {carrier: summarize(rows, carrier) for carrier in CARRIERS}}
    validate(receipt)
    save(args.output, receipt)


def validate(receipt):
    assert receipt["schema"] == SCHEMA
    rows = receipt["rows"]
    for row in rows:
        assert set(row["carriers"]) == set(CARRIERS)
        info = list(row["carriers"].values())
        assert all(item["native_samples"] == 6 * RATE for item in info)
        assert info[0]["native_pcm_sha256"] == info[1]["native_pcm_sha256"]
        for item in info:
            assert (item["checked_windows"] > 0) == (item["reading"] is not None)
            assert 0 <= item["checked_windows"] <= binding.WINDOWS
    counts = {carrier: summarize(rows, carrier) for carrier in CARRIERS}
    assert counts == receipt["summary"]
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser("score")
    for key in ("study", "media", "weights", "output"):
        run.add_argument(key, type=Path)
    actions.add_parser("review").add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.action == "score":
        score(args)
    else:
        print(validate(read(args.receipt)))
