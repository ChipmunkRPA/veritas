# SPDX-License-Identifier: Apache-2.0
"""Compare standalone stereo repeat locations on matched AAC and Ogg Opus audio."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.audio_native_alignment_panel import digest, read, save

PANELS = Path(__file__).resolve().parents[2] / "docs" / "evaluation-panels"
PLAN = PANELS / "audio-stereo-opus-repetition-plan-2026-11-18.json"
BASELINE = PANELS / "audio-aac-repetition-held-2026-11-17.json.gz"
PLAN_SHA = "d79b49ce044f28a57e13702a1e31e450ef485428a07c3f9c500bc03abd8e01bd"
BASE_SHA = "a0273194e031422b5435808301c87b2d34d3599686082767e349c37633b2de9e"
DETECTOR_SHA = "3c6e948f47f4c39e9acd48f48984cc63792916b538b9c20a7b74753739237a13"
SCHEMA = "chipmunk.audio-stereo-opus-repetition/1"
SAMPLES = 180_000
PACKET_BYTES = 400
CLOCK = {"sample_rate": 48000, "channels": 2, "presentation_start_pts": 0, "presentation_end_exclusive_pts": SAMPLES,
         "end_granule": 180312, "pre_skip": 312, "end_trim": 168, "packets": 188, "packet_payload_bit_rate": 160427}
POSITIVES = ("repeat_10db", "repeat_18db")


def load_inputs():
    if (digest(PLAN), digest(BASELINE)) != (PLAN_SHA, BASE_SHA):
        raise ValueError("The frozen plan or AAC reference changed")
    rows = [row for row in read(BASELINE)["rows"] if row["bitrate_setting"] == 160]
    if len(rows) != 256 or Counter(row["role"] for row in rows) != dict.fromkeys(set(row["role"] for row in rows), 32):
        raise ValueError("The paired study requires all 256 previous 160 kb/s parents")
    return rows


def prepared_path(item):
    return f'{item["delivery"].removesuffix("-160.m4a")}.wav'


def useful(row, assessment):
    def overlaps(a, b):
        intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        return 2 * intersection >= max(a[1], b[1]) - min(a[0], b[0])

    return bool(row["truth"] and any(all(overlaps(a, b) for a, b in zip(pair, row["truth"], strict=True))
                                    for pair in assessment["spans"]))


def counts(rows, prior):
    index = {prepared_path(row): row for row in prior}
    totals = {}
    ordinary_sources, useful_scenes = set(), set()
    for row in rows:
        source = index.pop(row["wav_file"])
        old, new = source["responses"]["lower_band_stereo"], row["opus"]
        if (row["wav_sha256"] != source["parent_sha256"] or row["clock"] != CLOCK
                or row["decoded_samples"] != SAMPLES):
            raise ValueError("An exact input or presented audio clock changed")
        for reading in (old, new):
            if reading["assessed"] != (reading["reason"] is None) or reading["spans"] and not reading["assessed"]:
                raise ValueError("Invalid reading and coverage")
            for (a, b), (c, d) in reading["spans"]:
                if not 0 <= a < b <= c < d <= SAMPLES:
                    raise ValueError("An interval lies outside the native clock")
        left, right = useful(source, old), useful(source, new)
        role, group = source["role"], (source["scene"], source["start_seconds"])
        key = totals.setdefault(role, Counter())
        key.update(total=1, aac_assessed=old["assessed"], opus_assessed=new["assessed"], aac_useful=left,
                   opus_useful=right, kept=left and right, lost=left and not right, gained=right and not left,
                   aac_flags=bool(old["spans"]), opus_flags=bool(new["spans"]),
                   extra_opus_flag=bool(new["spans"] and not old["spans"]))
        if role not in POSITIVES and new["spans"] and not old["spans"]:
            ordinary_sources.add(group)
        if right:
            useful_scenes.add(source["scene"])
    if index:
        raise ValueError("The retained receipt omits announced test files")
    positive = [totals[role] for role in POSITIVES]
    kept, baseline = (sum(row[name] for row in positive) for name in ("kept", "aac_useful"))
    extra = sum(row["extra_opus_flag"] for role, row in totals.items() if role not in POSITIVES)
    gate = (baseline > 0 and 100 * kept >= 85 * baseline and all(row["lost"] <= 3 and row["opus_useful"] for row in positive)
            and len(useful_scenes) >= 2 and extra <= 2 and len(ordinary_sources) <= 2)
    return {"rows_by_role": totals, "added_ordinary_groups": len(ordinary_sources), "useful_scenes": len(useful_scenes),
            "kept": kept, "aac_useful": baseline, "extra_opus_ordinary": extra, "gate_passed": gate}


def encode(parent, output):
    args = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(parent), "-threads", "1", "-ar", "48000", "-ac", "2",
            "-c:a", "libopus", "-application", "audio", "-frame_duration", "20", "-b:a", "160k", "-vbr", "off",
            "-map_metadata", "-1", "-fflags", "+bitexact", str(output)]
    subprocess.run(args, check=True, capture_output=True, timeout=15)


def native_decoded(path):
    from app.detectors.audio.decode import _exact_zero_based_clock

    args = ["ffmpeg", "-nostdin", "-v", "info", "-protocol_whitelist", "file", "-copyts", "-i", str(path),
            "-map", "0:a:0", "-af", "ashowinfo", "-f", "s16le", "-c:a", "pcm_s16le", "pipe:1"]
    result = subprocess.run(args, check=True, capture_output=True, timeout=15)
    if len(result.stdout) != SAMPLES * 4 or not _exact_zero_based_clock(result.stderr.decode("utf-8"), SAMPLES):
        raise ValueError("The original decoded stereo frames disagree with the presented clock")
    return result.stdout


def measure(source, scratch, baseline):
    import hashlib

    import numpy as np

    from app.detectors.audio.ogg_opus_reference import native_clock
    from app.evidence import audio_aac_alignment as repeat

    if digest(Path(repeat.__file__)) != DETECTOR_SHA:
        raise ValueError("The existing measurement must be unchanged")
    output = scratch / "encode.opus"
    rows = []
    for number, item in enumerate(baseline, 1):
        parent = source / prepared_path(item)
        if digest(parent) != item["parent_sha256"]:
            raise ValueError("The repeat test needs the exact pinned stereo WAV parent")
        encode(parent, output)
        clock = native_clock(output, packet_bytes=PACKET_BYTES, presented=SAMPLES, channels=2)
        if clock != CLOCK:
            raise ValueError("Opus native packet timing differs from the frozen profile")
        pcm = native_decoded(output)
        alignment, reason, spans = repeat.search(np.frombuffer(pcm, dtype="<i2").reshape(-1, 2))
        rows.append({"wav_file": parent.name, "wav_sha256": item["parent_sha256"], "clock": clock,
                     "decoded_samples": len(pcm) // 4, "opus_sha256": digest(output), "native_pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                     "opus": {"assessed": reason is None, "reason": reason,
                              "spans": spans, "alignment": vars(alignment) if alignment else None}})
        if number % 32 == 0:
            print(f"Verified and measured {number}/256 exact native deliveries", flush=True)
    return rows


def diagnose(source, scratch, baseline, measurements):
    """Re-evaluate missed pairs after supplying the known lag, for diagnosis only."""
    from unittest.mock import patch

    import numpy as np

    from app.evidence import audio_aac_alignment as repeat

    by_file = {row["wav_file"]: row for row in measurements}
    output = scratch / "diagnose.opus"
    rows = []
    for item in baseline:
        parent = source / prepared_path(item)
        previous = by_file[parent.name]
        if not useful(item, item["responses"]["lower_band_stereo"]) or useful(item, previous["opus"]):
            continue
        if digest(parent) != item["parent_sha256"]:
            raise ValueError("The diagnostic requires the original stereo parent")
        encode(parent, output)
        if digest(output) != previous["opus_sha256"]:
            raise ValueError("The diagnostic must use the exact held Opus delivery")
        pcm = np.frombuffer(native_decoded(output), dtype="<i2").reshape(-1, 2)
        known_lag, remainder = divmod(item["truth"][1][0] - item["truth"][0][0], 3)
        if remainder:
            raise ValueError("The labeled repeat does not fit the original decimated lag grid")
        nominated = repeat._nominate_lags
        observed = []

        def only_known_lag(bands, nominated=nominated, observed=observed, known_lag=known_lag):
            observed.extend(nominated(bands))
            return [known_lag]

        with patch.object(repeat, "_nominate_lags", only_known_lag):
            alignment, reason, pairs = repeat.search(pcm)
        rows.append({"wav_file": parent.name, "role": item["role"], "known_lag_at_16k": known_lag,
                     "actual_alignment_offset": item["actual_offset"], "measured_alignment_offset": alignment.right_offset_samples,
                     "nominated_known_lag": known_lag in observed, "reason": reason, "with_known_lag": pairs,
                     "located_with_known_lag": useful(item, {"spans": pairs})})
    return {"comparison": "Post-holdout use of ground truth for diagnosis only", "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("measure", "review", "diagnose"):
        command = commands.add_parser(mode)
        command.add_argument("receipt", type=Path)
        if mode != "review":
            command.add_argument("source", type=Path)
    arguments = parser.parse_args()
    baseline = load_inputs()
    expected = {"schema": SCHEMA, "plan_sha256": PLAN_SHA, "aac_receipt_sha256": BASE_SHA, "detector_sha256": DETECTOR_SHA}
    if arguments.mode == "measure":
        if arguments.receipt.exists():
            parser.error("Use a new output path for the measurements")
        with TemporaryDirectory(prefix="chipmunk-stereo-opus-repeat-") as scratch:
            rows = measure(arguments.source, Path(scratch), baseline)
        summary = counts(rows, baseline)
        save(arguments.receipt, expected | {"rows": rows, "counts": summary})
    else:
        body = read(arguments.receipt)
        summary = counts(body["rows"], baseline)
        if summary != body["counts"] or any(body[key] != value for key, value in expected.items()):
            raise ValueError("The replay does not match the frozen result")
        if arguments.mode == "diagnose":
            with TemporaryDirectory(prefix="chipmunk-stereo-opus-oracle-") as scratch:
                ruling = diagnose(arguments.source, Path(scratch), baseline, body["rows"])
            print(json.dumps(ruling, sort_keys=True))
            return
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
