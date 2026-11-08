"""Separate model-free source, transformation, interval and codec replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from scripts import run_public_audioseal_speech_v2 as frozen
from scripts.review_public_audio_watermark_v1 import replay


def audit(trial, output):
    lock, _, inputs = frozen.check(trial)
    method = frozen.method
    result = method.read(trial / "result.json")
    rows = method.read(trial / "cases-private.json")["cases"]
    profile = method.read(trial / "profile/profile.json")
    method.require(
        profile["child_exit_code"] == 0 and profile["guard"] is None, "Completed native profile required"
    )
    method.require(
        len(rows) == 576 and result["generator_forwards"] == 32 and result["detector_forwards"] == 581,
        "Complete native study required",
    )
    output.mkdir(exist_ok=False)
    plan = method.read(inputs / "source-plan.json")
    speakers = {r["foreground"]["speaker"] for r in plan["trials"]}
    expected = {
        (speaker, state, condition)
        for speaker in speakers
        for state in frozen.STATES
        for condition in frozen.CONDITIONS
    }
    method.require(
        len(speakers) == 32 and {(r["speaker"], r["state"], r["condition"]) for r in rows} == expected,
        "Complete unique32x18 cohort required",
    )
    by_speaker = {r["foreground"]["speaker"]: r for r in plan["trials"]}
    base_cache = {}
    codec_count = 0
    max_mean_error = 0.0
    metric_rows = []
    for row in rows:
        speaker = row["speaker"]
        case = by_speaker[speaker]
        if speaker not in base_cache:
            method.require(
                method.digest(Path(row["base_path"])) == row["base_sha256"], "Retained generated mark changed"
            )
            with np.load(row["base_path"], allow_pickle=False) as z:
                base = {k: z[k] for k in z.files}
            fg = case["foreground"]
            path = inputs / fg["input_path"]
            method.require(method.digest(path) == fg["input_sha256"], "Source input checksum changed")
            s16 = np.load(path, allow_pickle=False)
            source = frozen.preparation.pcm((inputs / fg["source_path"]).read_bytes())
            method.require(
                method.digest(inputs / fg["source_path"]) == fg["source_sha256"]
                and np.array_equal(source[:128000], s16),
                "Original source/input binding differs",
            )
            unmarked = frozen.normalize(s16.astype(np.float32) / 32768.0, 0.1)
            method.require(np.array_equal(unmarked, base["unmarked"]), "Foreground normalization differs")
            white = (
                np.random.default_rng(int(frozen.preparation.stable(fg["id"])[:16], 16))
                .normal(size=128000)
                .astype(np.float32)
            )
            method.require(
                np.array_equal(base["white_0db"], frozen.normalize(white, frozen.rms(unmarked))),
                "White-noise recipe differs",
            )
            for kind, ref in case["backgrounds"].items():
                path = inputs / ref["input_path"]
                signal = np.load(path, allow_pickle=False)
                method.require(method.digest(path) == ref["input_sha256"], "Background input changed")
                original = frozen.preparation.production_pcm(ref)
                method.require(
                    np.array_equal(original[ref["start_sample"] : ref["start_sample"] + 128000], signal),
                    "Background source locator differs",
                )
                method.require(
                    np.array_equal(
                        base[kind + "_0db"],
                        frozen.normalize(signal.astype(np.float32) / 32768.0, frozen.rms(unmarked)),
                    ),
                    "Matched background gain differs",
                )
            mixtures, gain = frozen.mixtures(
                unmarked, unmarked + base["watermark"], {k: base[k] for k in base if k.endswith("_0db")}
            )
            base_cache[speaker] = (mixtures, gain)
        mixtures, gain = base_cache[speaker]
        method.require(gain == row["gain"], "Shared clipping-safe gain differs")
        path = Path(row["private_arrays_path"])
        method.require(method.digest(path) == row["private_arrays_sha256"], "Native case archive changed")
        with np.load(path, allow_pickle=False) as z:
            audio, values = z["pcm"], z["watermark_response"]
        method.require(
            audio.dtype == np.float32
            and audio.shape == (128000,)
            and np.isfinite(audio).all()
            and hashlib.sha256(audio.tobytes()).hexdigest() == row["pcm_sha256"],
            "Exact native input identity differs",
        )
        if row["condition"] == "mp3_64k":
            name = frozen.preparation.stable(row["foreground_id"])
            codec = trial / "private" / (name + "-" + row["state"] + ".mp3")
            decoded = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-i",
                    str(codec),
                    "-f",
                    "f32le",
                    "-acodec",
                    "pcm_f32le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "pipe:1",
                ],
                capture_output=True,
                check=True,
                timeout=30,
            ).stdout
            method.require(
                decoded == audio.astype("<f4").tobytes(),
                "Retained codec re-decode differs; no trimming permitted",
            )
            codec_count += 1
        else:
            method.require(
                np.array_equal(audio, mixtures[(row["state"], row["condition"])]),
                "Retained native mixture differs",
            )
        independent = replay(values)
        e = row["evidence"]
        method.require(
            independent["fraction"] == e["positive_sample_fraction"]
            and independent["intervals"] == e["intervals"]
            and len(e["bins"]) == 32,
            "Independent interval/readout replay differs",
        )
        for i, record in enumerate(e["bins"]):
            method.require(
                record["start_sample"] == i * 4000
                and record["end_sample"] == (i + 1) * 4000
                and record["positive_sample_fraction"] == independent["counts"][i] / 4000,
                "Bin source coordinates differ",
            )
            max_mean_error = max(
                max_mean_error, abs(record["mean_watermark_response"] - independent["means"][i])
            )
        method.require(
            e["provider_authenticated"] is e["used_in_confidence"] is e["early_stop_eligible"] is False,
            "Watermark response acquired unsupported AI meaning",
        )
        # Recompute overlap independently from interval intersections, without a sample mask.
        intervals = [(r["start_sample"], r["end_sample"]) for r in independent["intervals"]]
        truth = frozen.truth(row["state"])
        duration = sum(b - a for a, b in intervals)
        intersection = sum(max(0, min(b, d) - max(a, c)) for a, b in intervals for c, d in truth)
        truth_count = sum(b - a for a, b in truth)
        union = duration + truth_count - intersection
        metrics = {
            "any_selected": bool(intervals),
            "iou": intersection / union if truth_count else None,
            "mark_coverage": intersection / truth_count if truth_count else None,
            "outside_mark_s": (duration - intersection) / 16000,
        }
        method.require(
            row["expected_marked_spans"] == truth and metrics == row["metrics"],
            "Known mark geometry/overlap differs",
        )
        metric_rows.append({**row, "metrics": metrics})
    method.require(
        codec_count == 96 and max_mean_error <= 1e-12,
        "Complete codec replay and fixed arithmetic tolerance required",
    )
    aggregate = frozen.aggregate(metric_rows)
    method.require(aggregate == result["outcomes"], "Aggregate denominator/replay differs")
    receipt = {
        "schema": "chipmunk.public-speech-watermark-audit/2",
        "cases": len(rows),
        "speakers": len(speakers),
        "bins_replayed": len(rows) * 32,
        "codec_redecodes": codec_count,
        "max_mean_absolute_difference": max_mean_error,
        "source_plan_sha256": lock["input_plan_sha256"],
        "canonical_intervals_verified": True,
        "native_model_forwards": 0,
        "outside_human_reviewers": 0,
        "provider_requests": 0,
        "claim_scope": "Separate code/arithmetic replay; no outside human validation or AI-authorship accuracy claim.",
    }
    method.write(output / "audit.json", receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.trial, args.output)
