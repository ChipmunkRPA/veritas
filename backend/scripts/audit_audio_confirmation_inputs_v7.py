"""Replay every fresh mixture from committed source audio, without neural models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio, read_wav, wav_bytes
from scripts.audio_confirmation_v7 import CONDITIONS, PREPARED_SHA256
from scripts.prepare_audio_confirmation_v7 import digest, stable
from scripts.prepare_audio_guard_v7 import write


def audit(root, output):
    source_path, variant_path = root / "prepared-sources.json", root / "variants.json"
    if digest(source_path) != PREPARED_SHA256:
        raise ValueError("Prepared sources changed")
    source = json.loads(source_path.read_text())
    variants = json.loads(variant_path.read_text())["variants"]
    if len(variants) != 1080 or len({r["id"] for r in variants}) != 1080:
        raise ValueError("Every source/condition required")
    cache = {}
    for row in source["sources"] + source["backgrounds"]:
        path = root / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError("Original source bytes differ")
        decoded = decode_audio(path, source_sha256=row["sha256"])
        if row.get("production_pcm_sha256") and decoded.mapping["pcm_sha256"] != row["production_pcm_sha256"]:
            raise ValueError("Original production PCM differs")
        if (
            row.get("source_prefix_pcm_sha256")
            and hashlib.sha256(decoded.pcm[:64000].astype("<f4").tobytes()).hexdigest()
            != row["source_prefix_pcm_sha256"]
        ):
            raise ValueError("Original prefix differs")
        cache[row["sha256"]] = decoded.pcm
    pools = {
        k: sorted([r for r in source["backgrounds"] if r["kind"] == k], key=lambda r: stable(r["source_id"]))
        for k in ("noise", "music")
    }
    pools["speech"] = sorted(
        [{**r, "source_id": r["id"]} for r in source["sources"] if r["role"] == "background"],
        key=lambda r: stable(r["source_id"]),
    )
    original = sorted(
        [r for r in source["sources"] if r["role"] == "foreground"], key=lambda r: stable(r["id"])
    )
    counters, records, selected_decodes = Counter(), [], []
    for condition in CONDITIONS:
        for label in (0, 1):
            selected_decodes.append(
                min(r["id"] for r in variants if r["condition"] == condition and r["label"] == label)
            )
    records_by_source = {r["id"]: [v for v in variants if v["base_id"] == r["id"]] for r in original}
    max_quantization, max_snr_error = 0.0, 0.0
    for row in original:
        n = counters[row["label"]]
        counters[row["label"]] += 1
        base = cache[row["sha256"]][:64000].astype(np.float64)
        base = base * (0.1 / float(np.sqrt(np.mean(base**2))))
        components = {"clean": base}
        backgrounds = {}
        for kind in ("noise", "music", "speech"):
            b = pools[kind][n % len(pools[kind])]
            full = cache[b["sha256"]]
            start = int(stable(row["id"] + kind + ":offset")[:8], 16) % (len(full) - 64000 + 1)
            interval = full[start : start + 64000]
            level = float(np.sqrt(np.mean(interval.astype(float) ** 2)))
            for snr in (5, 15) if kind == "music" else (0, 5, 15):
                condition = f"{kind}_{snr}"
                components[condition] = base + interval.astype(float) * (0.1 / level / 10 ** (snr / 20))
                backgrounds[condition] = (
                    b,
                    start,
                    hashlib.sha256(interval.astype("<f4").tobytes()).hexdigest(),
                )
        gain = min(1.0, 0.98 / max(float(np.max(np.abs(a))) for a in components.values()))
        current = records_by_source[row["id"]]
        if [v["condition"] for v in current] != list(CONDITIONS):
            raise ValueError("Missing/reordered source condition")
        clean_decoded = None
        for variant in current:
            condition = variant["condition"]
            if (
                variant["source_sha256"] != row["sha256"]
                or variant["group"] != source["assignments"][row["id"]]
                or variant["label"] != row["label"]
            ):
                raise ValueError("Variant bound to wrong source/group/truth")
            if not math.isclose(gain, variant["common_peak_gain"], abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError("Common source gain differs")
            expected = (components[condition] * gain).astype(np.float32)
            if hashlib.sha256(expected.astype("<f4").tobytes()).hexdigest() != variant["payload_pcm_sha256"]:
                raise ValueError("Exact source/background mixture differs")
            path = root / variant["path"]
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != variant["sha256"] or raw != wav_bytes(expected):
                raise ValueError("Canonical sample payload/file bytes differ")
            actual = read_wav(raw)
            if (
                len(actual) != 64000
                or hashlib.sha256(actual.astype("<f4").tobytes()).hexdigest() != variant["pcm_sha256"]
            ):
                raise ValueError("Committed production PCM differs")
            max_quantization = max(max_quantization, float(np.max(np.abs(actual - expected))))
            if condition == "clean":
                clean_decoded = actual
            else:
                background, start, pcm_sha = backgrounds[condition]
                if (
                    variant["background_sha256"] != background["sha256"]
                    or variant["background_start_sample"] != start
                    or variant["background_interval_pcm_sha256"] != pcm_sha
                ):
                    raise ValueError("Background sample interval differs")
                measured = float(
                    20
                    * np.log10(
                        np.sqrt(np.mean(clean_decoded.astype(float) ** 2))
                        / np.sqrt(np.mean((actual.astype(float) - clean_decoded) ** 2))
                    )
                )
                if not math.isclose(measured, variant["measured_snr_db"], abs_tol=1e-9):
                    raise ValueError("Reported realized SNR differs")
                max_snr_error = max(max_snr_error, abs(measured - variant["snr_db"]))
            production_replay = variant["id"] in selected_decodes
            if production_replay:
                check = decode_audio(path, source_sha256=variant["sha256"])
                if (
                    not np.array_equal(check.pcm, actual)
                    or check.mapping["pcm_sha256"] != variant["pcm_sha256"]
                ):
                    raise ValueError("Sampled production FFmpeg decode differs")
            records.append(
                {"id": variant["id"], "passed": True, "sampled_production_decode": production_replay}
            )
    if max_quantization > 0.5 / 32768 + 1e-7:
        raise ValueError("Quantization exceeds canonical PCM tolerance")
    result = {
        "schema": "chipmunk.audio-fresh-source-mixture-audit/7",
        "prepared_sources_sha256": PREPARED_SHA256,
        "variants_sha256": digest(variant_path),
        "audit_code_sha256": digest(Path(__file__)),
        "passed": len(records) == 1080,
        "source_recordings_checked": len(cache),
        "mixture_inputs_checked": len(records),
        "mixture_samples_checked": 64000 * len(records),
        "sampled_production_decodes": len(selected_decodes),
        "max_quantization_error": max_quantization,
        "max_realized_snr_deviation_db": max_snr_error,
        "cases": records,
        "detector_inference_run": False,
        "scope": "All committed original PCM/prefix and separately reconstructed mixture recipes, canonical files/samples;18 sampled production FFmpeg decodes. Shared decoder/codec; no neural forwards, independent truth or outside reviewer.",
    }
    write(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.root, args.output)
