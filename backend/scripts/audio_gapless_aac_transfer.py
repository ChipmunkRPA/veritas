# SPDX-License-Identifier: Apache-2.0
"""Verify gapless M4A clock and waveform evidence on the fixed AAC transfer panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from app.evidence import audio_reference_waveform
from scripts import audio_reference_waveform_aac_transfer as prior

FREEZE = prior.FREEZE.with_name("audio-gapless-aac-freeze-2026-11-18.json")
FREEZE_SHA = "16c673070128c5550831312ce5d2af660bf4034f6d7f455c1b409f49ac7b772d"
BASE = prior.FREEZE.with_name("audio-reference-waveform-aac-64-2026-11-18.json.gz")
BASE_SHA = "3a9891985da30873b7e00af5a6369a4c2bc69632237061977b10470cd017cc27"
SCHEMA = "chipmunk.gapless-aac-waveform-transfer/1"
RATES = (64, 96)
MIN_BIT_RATE = 60_000


def sha_input(samples):
    return hashlib.sha256(samples.tobytes()).hexdigest()


def prepare(output, wav, rate):
    pcm = output.with_suffix(".wav")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(wav), "-ar", "48000", "-ac", "1",
                    "-c:a", "pcm_s16le", "-map_metadata", "-1", str(pcm)], capture_output=True, check=True, timeout=20)
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", str(rate * 1000), str(pcm), str(output)],
                   capture_output=True, check=True, timeout=25)


def evaluate(case, *, scratch, rate):
    from app.evidence import audio_reference

    if audio_reference.WAVEFORM_AAC_MIN_BIT_RATE != MIN_BIT_RATE:
        raise ValueError("The app bitrate gate differs from the frozen evaluation")
    baseline, first, submitted = case
    if baseline["wav"]["sha256"] != [sha_input(first), sha_input(submitted)]:
        raise ValueError("A source pair differs from the frozen lossless inputs")
    with TemporaryDirectory(dir=scratch, prefix="pair-") as folder:
        folder = Path(folder)
        wav_earlier, earlier, wav, target = (folder / name for name in ("earlier.wav", "earlier.flac", "raw.wav", "later.m4a"))
        prior.prior.original.wav(wav_earlier, first)
        prior.flac(wav_earlier, earlier)
        prior.prior.original.wav(wav, submitted)
        prepare(target, wav, rate)
        verification = audio_reference._gapless_waveform_tracks(target, (prior.track(earlier), prior.track(target)))
        if verification is None:
            raise ValueError("The gapless table, packet clock and decoded clock did not all match")
        tracks, clock = verification
        measured = prior.probe.measurement(*(prior.pcm(track) for track in tracks))
        return {**{key: baseline[key] for key in ("panel", "id", "reader", "role", "scene", "truth", "wav")},
                "clock": clock, "aac": measured, "encoded_sha256": prior.sha(target),
                "encoder_input_sha256": prior.sha(target.with_suffix(".wav"))}


def verify_plan():
    if prior.sha(BASE) != BASE_SHA or prior.sha(FREEZE) != FREEZE_SHA:
        raise ValueError("The pinned gapless trial or preceding results differ")
    baseline = prior.read(BASE)
    prior.review(baseline)
    if baseline["original_receipt_sha256"] != prior.sha(prior.ORIGINAL):
        raise ValueError("The lossless input and implementation commitments differ")
    return baseline


def summarize(rows, baseline):
    controls = {(row["panel"], row["id"], row["scene"], row["role"]): row for row in baseline["rows"]}
    if len(rows) != len(controls):
        raise ValueError("A delivered case is missing")
    admitted = []
    for row in rows:
        prior_row = controls.pop((row["panel"], row["id"], row["scene"], row["role"]))
        clock, native = row["clock"], row["clock"]["native_clock"]
        if (any(row[key] != prior_row[key] for key in ("truth", "wav", "reader"))
                or tuple(clock[key] for key in ("side", "sample_rate", "presentation_start_pts", "presentation_end_exclusive_pts",
                                               "decoded_samples", "excluded_trailing_samples")) != ("submitted", 16000, 704, 96704, 96235, 235)
                or (native["method"], native["sample_rate"], native["presentation_start_pts"], native["presentation_end_exclusive_pts"],
                    native["priming_samples"], native["remainder_samples"], native["coded_samples"], native["packets"], native["channels"])
                != ("explicit_gapless_table", 48000, 2112, 290112, 2112, 704, 290816, 284, 1)
                or row["aac"]["sha256"][0] != prior_row["wav"]["sha256"][0] or row["aac"]["samples"] != [96000, 96000]):
            raise ValueError("A waveform measurement lacks the frozen clocks or source identity")
        candidate = row["aac"]
        if clock["bit_rate"] < MIN_BIT_RATE:
            candidate = candidate | {"status": "insufficient_matching_waveform", "selected": None}
        admitted.append(row | {"aac": candidate})
    rates = [row["clock"]["bit_rate"] for row in rows]
    refused = sum(rate < MIN_BIT_RATE for rate in rates)
    _, unconstrained = prior.summarize(rows)
    summary, qualified = prior.summarize(admitted)
    return {"cases_by_background": summary, "before_bitrate_gate": unconstrained, "eligible": qualified,
            "bitrate_exclusions": refused, "stream_bit_rate": {"min": min(rates), "max": max(rates)},
            "approval_criteria_met": qualified["approval_criteria_met"] and 100 * refused < 5 * len(rows)}


def run(archive, full, short, rate, baseline, scratch):
    sounds, noises = prior.background(archive)
    if sounds != baseline["backgrounds"]:
        raise ValueError("The background recordings differ from the prior panel")
    index = {(row["panel"], row["id"], row["scene"], row["role"]): row for row in baseline["rows"]}
    cases = []
    for panel, source, folder in zip(prior.prior.PANELS, prior.prior.sources(), (full, short), strict=True):
        for item, scene, role, _, first, second in prior.prior.cases(prior.prior.io.read(source), folder, noises):
            if scene != "clean":
                cases.append((index.pop((panel, item["id"], scene, role)), first, second))
    if index:
        raise ValueError("The inherited input cases differ")
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number, row in enumerate(pool.map(partial(evaluate, scratch=scratch, rate=rate), cases), 1):
            rows.append(row)
            if number % 60 == 0:
                print(f"Verified clock and decoded {number}/{len(cases)} clips at {rate} kbps", flush=True)
    return {"schema": SCHEMA, "freeze_sha256": FREEZE_SHA, "baseline_sha256": BASE_SHA, "rate_kbps": rate,
            "waveform_implementation_sha256": prior.sha(Path(audio_reference_waveform.__file__)),
            "rows": rows, "summary": summarize(rows, baseline)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    review = commands.add_parser("review")
    replay = commands.add_parser("run")
    for command in (review, replay):
        command.add_argument("receipt", type=Path)
    for field in ("archive", "full", "short"):
        replay.add_argument(field, type=Path)
    replay.add_argument("--kbps", type=int, choices=RATES, required=True)
    args = parser.parse_args()
    baseline = verify_plan()
    if args.mode == "run":
        if args.receipt.exists():
            parser.error("The evaluation needs a new output path")
        with TemporaryDirectory(prefix="chipmunk-gapless-aac-") as location:
            receipt = run(args.archive, args.full, args.short, args.kbps, baseline, Path(location))
        prior.save(args.receipt, receipt)
    else:
        receipt = prior.read(args.receipt)
    summary = summarize(receipt["rows"], baseline)
    if (receipt["schema"] != SCHEMA or receipt["freeze_sha256"] != FREEZE_SHA or receipt["baseline_sha256"] != BASE_SHA
            or receipt["rate_kbps"] not in RATES or summary != receipt["summary"]):
        raise ValueError("The replay receipt does not match the frozen trial and comparison")
    print(json.dumps(summary, sort_keys=True), flush=True)
