"""Separate waveform, readout, interval and codec replay; loads no watermark model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts import run_public_audio_watermark_v1 as frozen


def replay(values):
    frozen.method.require(
        values.dtype == np.float32
        and values.shape == (128000,)
        and np.isfinite(values).all()
        and np.all((values >= 0) & (values <= 1)),
        "Exact finite native sample responses required",
    )
    blocks = values.reshape(32, 4000)
    counts = np.count_nonzero(blocks > 0.5, axis=1)
    positive = counts > 2000
    edges = np.diff(np.r_[0, positive.astype(np.int8), 0])
    starts = np.flatnonzero(edges == 1) * 4000
    ends = np.flatnonzero(edges == -1) * 4000
    return {
        "fraction": int(counts.sum()) / 128000,
        "counts": counts.tolist(),
        "means": [math.fsum(map(float, block)) / 4000 for block in blocks],
        "intervals": [
            {"start_sample": int(a), "end_sample": int(b), "start_s": a / 16000, "end_s": b / 16000}
            for a, b in zip(starts, ends, strict=True)
        ],
    }


def audit(trial, output):
    frozen.check(trial)
    profile = frozen.method.read(trial / "profile/profile.json")
    result = frozen.method.read(trial / "result.json")
    rows = frozen.method.read(trial / "cases-private.json")["cases"]
    frozen.method.require(
        profile["child_exit_code"] == 0
        and profile["guard"] is None
        and len(rows) == 32
        and result["generator_forwards"] == 4
        and result["detector_forwards"] == 37,
        "Completed native contract required",
    )
    original = frozen.carriers()
    expected_keys = {(name, variant) for name in original for variant in frozen.POLICY["variants"]}
    frozen.method.require(
        {(r["carrier"], r["variant"]) for r in rows} == expected_keys,
        "Complete unique authored source/variant identities required",
    )
    loaded = {}
    groups = defaultdict(list)
    details = []
    maximum = 0.0
    codec_hashes = {}
    for row in rows:
        path = Path(row["private_arrays_path"])
        frozen.method.require(
            frozen.method.digest(path) == row["private_arrays_sha256"], "Private arrays changed"
        )
        with np.load(path, allow_pickle=False) as arrays:
            pcm, values = arrays["pcm"], arrays["watermark_response"]
        frozen.method.require(
            pcm.dtype == np.float32
            and pcm.shape == (128000,)
            and hashlib.sha256(pcm.tobytes()).hexdigest() == row["source_pcm_sha256"],
            "Native PCM binding changed",
        )
        loaded[(row["carrier"], row["variant"])] = pcm
        independent = replay(values)
        e = row["evidence"]
        frozen.method.require(
            independent["fraction"] == e["positive_sample_fraction"]
            and independent["intervals"] == e["intervals"]
            and len(e["bins"]) == 32
            and e["provider_authenticated"] is e["used_in_confidence"] is e["early_stop_eligible"] is False,
            "Independent readout/interval/scope replay differs",
        )
        for i, bin in enumerate(e["bins"]):
            frozen.method.require(
                bin["start_sample"] == i * 4000
                and bin["end_sample"] == (i + 1) * 4000
                and bin["positive_sample_fraction"] == independent["counts"][i] / 4000,
                "Quarter-second source coordinates differ",
            )
            maximum = max(maximum, abs(bin["mean_watermark_response"] - independent["means"][i]))
        truth = np.zeros(128000, bool)
        selected = np.zeros(128000, bool)
        for span in row["expected_marked_spans"]:
            truth[span["start_sample"] : span["end_sample"]] = True
        for span in independent["intervals"]:
            selected[span["start_sample"] : span["end_sample"]] = True
        tp = int((truth & selected).sum())
        fp = int((~truth & selected).sum())
        fn = int((truth & ~selected).sum())
        metrics = {
            "selected": bool(selected.any()),
            "marked": bool(truth.any()),
            "intersection_samples": tp,
            "false_positive_samples": fp,
            "false_negative_samples": fn,
            "interval_iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "outside_mark_selected_seconds": fp / 16000,
            "marked_seconds_missed": fn / 16000,
        }
        groups[row["variant"]].append(metrics)
        details.append({"carrier": row["carrier"], "variant": row["variant"], "metrics": metrics})
        if row["variant"].endswith("_mp3"):
            kind = "unmarked" if row["variant"] == "unmarked_mp3" else "partial"
            source = path.parent / (row["carrier"] + "-" + kind + ".mp3")
            decoded = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-i",
                    str(source),
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
            )
            frozen.method.require(
                decoded.stdout == pcm.astype("<f4").tobytes(),
                "Stored codec file no longer decodes to native input",
            )
            codec_hashes[source.name] = frozen.method.digest(source)
    for number, (name, pcm) in enumerate(original.items()):
        frozen.method.require(
            np.array_equal(loaded[name, "unmarked"], pcm), "Authored unmarked source changed"
        )
        partial = loaded[name, "partial"]
        marked = loaded[name, "marked"]
        frozen.method.require(
            np.array_equal(partial[:32000], pcm[:32000])
            and np.array_equal(partial[96000:], pcm[96000:])
            and np.array_equal(partial[32000:96000], marked[32000:96000]),
            "Exact partial mark placement differs",
        )
        noise = np.random.default_rng(100 + number).normal(size=len(pcm))
        noise = (
            noise
            / np.sqrt(np.mean(noise**2))
            * max(float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))), 0.001)
            / np.sqrt(10)
        ).astype(np.float32)
        for variant in ("unmarked", "marked", "partial"):
            frozen.method.require(
                np.array_equal(loaded[name, variant] + noise, loaded[name, variant + "_noise"]),
                "Matched strong-noise transform differs",
            )
    frozen.method.require(maximum <= 1e-12, "Independent exact sample-mean replay differs")
    aggregate = {
        name: {
            "cases": len(items),
            "cases_with_selected_interval": sum(r["selected"] for r in items),
            "marked_cases_with_no_interval": sum(r["marked"] and not r["selected"] for r in items),
            "mean_interval_iou_on_marked": math.fsum(r["interval_iou"] for r in items) / len(items)
            if items[0]["marked"]
            else None,
            "mean_outside_mark_selected_seconds": math.fsum(r["outside_mark_selected_seconds"] for r in items)
            / len(items),
            "mean_marked_seconds_missed": math.fsum(r["marked_seconds_missed"] for r in items) / len(items),
        }
        for name, items in groups.items()
    }
    frozen.method.write(
        output,
        {
            "schema": "chipmunk.public-audio-watermark-authored-replay/1",
            "freeze_sha256": frozen.method.digest(trial / "freeze.json"),
            "result_sha256": frozen.method.digest(trial / "result.json"),
            "audit_code_sha256": frozen.method.digest(Path(__file__)),
            "source_waveforms_replayed": 32,
            "independent_quarter_second_bins": 1024,
            "gapless_codec_inputs_replayed": len(codec_hashes),
            "maximum_response_mean_difference": maximum,
            "aggregate": aggregate,
            "authored_case_details": details,
            "codec_file_commitments": codec_hashes,
            "new_neural_forwards": 0,
            "outside_human_reviews": 0,
            "deployment_compatibility_verified": False,
            "serving_admitted": False,
        },
    )
    print(
        json.dumps(
            {
                "replayed_cases": 32,
                "quarter_second_bins": 1024,
                "gapless_codec_inputs": len(codec_hashes),
                "maximum_response_mean_difference": maximum,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.trial.resolve(), args.output.resolve())
