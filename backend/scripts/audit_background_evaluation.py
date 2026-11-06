"""Rebuild fresh controls and exact scored inputs from retained publisher audio."""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections import defaultdict

import numpy as np
import soundfile as sf
from evaluate_background_continuity import ROOT, save
from scipy.signal import resample_poly

from app.core.canonical import canonical_hash
from app.detectors.audio.decode import read_wav, wav_bytes


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_pcm(record, dtype="float64"):
    raw = (ROOT / record["source_path"]).read_bytes()
    assert digest(raw) == record["source_sha256"]
    pcm, rate = sf.read(io.BytesIO(raw), dtype=dtype, always_2d=True)
    pcm = pcm.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        pcm = resample_poly(pcm, 16000 // divisor, rate // divisor)
    return pcm


def main():
    first = json.loads((ROOT / "manifest.json").read_bytes())
    folder = ROOT / "fresh-followup"
    fresh = json.loads((folder / "manifest.json").read_bytes())
    environment = json.loads((ROOT / "environment-detector/manifest.json").read_bytes())
    protocol_bindings = []
    for directory, manifest in ((ROOT, first), (folder, fresh), (ROOT / "environment-detector", environment)):
        protocol = json.loads((directory / "protocol.json").read_bytes())
        assert canonical_hash(protocol) == manifest["protocol_sha256"]
        scores = json.loads((directory / "scores.json").read_bytes())
        assert scores["manifest_sha256"] == canonical_hash(manifest)
        protocol_bindings.append(manifest["protocol_sha256"])
    normalized = 0
    for record in [*first["records"], *fresh["records"]]:
        pcm = source_pcm(record)[:64000]
        gain = min(10 ** (-24 / 20) / np.sqrt(np.mean(pcm * pcm)), 0.8 / max(np.abs(pcm)))
        expected = wav_bytes(pcm * gain)
        assert digest(expected) == record["input_sha256"]
        assert expected == (ROOT / record["input_path"]).read_bytes()
        normalized += 1
    groups = defaultdict(list)
    for record in fresh["records"]:
        groups[record["pair_id"]].append(record)
    rebuilt = {}
    for pair_id, pair in groups.items():
        left, right = sorted(pair, key=lambda row: row["pair_side"])
        x, y = [read_wav((ROOT / row["input_path"]).read_bytes()).astype(np.float64) for row in (left, right)]
        gain = float(np.sqrt(np.mean(x[32000:] ** 2)) / max(np.sqrt(np.mean(y[32000:] ** 2)), 1e-9))
        hard = np.concatenate([x[:32000], gain * y[32000:]])
        fade = hard.copy()
        alpha = np.linspace(0, 1, 1600)
        fade[31200:32800] = (1 - alpha) * x[31200:32800] + alpha * gain * y[31200:32800]
        gain_only = x.copy()
        gain_only[32000:] *= 0.25
        for kind, pcm in (
            ("original_left", x),
            ("original_right", y),
            ("gain_only", gain_only),
            ("hard_splice", hard),
            ("crossfade_splice", fade),
        ):
            rebuilt[pair_id + ":" + kind] = wav_bytes(pcm * (0.8 / max(0.8, float(np.max(np.abs(pcm))))))
    for case in fresh["cases"]:
        assert digest(rebuilt[case["id"]]) == case["input_sha256"]
        assert rebuilt[case["id"]] == (ROOT / case["input_path"]).read_bytes()
        assert case["boundary_samples"] == (
            [32000] if case["case"] in {"hard_splice", "crossfade_splice"} else []
        )
    env_rows = json.loads((ROOT / "environment-detector/scores.json").read_bytes())["rows"]
    for record, scored in zip([*environment["calibration"], *environment["test"]], env_rows, strict=True):
        pcm = source_pcm(record, "float32")
        pcm = np.tile(pcm, int(np.ceil(64600 / len(pcm))))[:64600].astype("<f4")
        assert digest(pcm.tobytes()) == scored["input_sha256"]
        np.testing.assert_array_equal(
            pcm, np.load(ROOT / "environment-detector/inputs" / (scored["input_sha256"] + ".npy"))
        )
    assert not ({r["source_group"] for r in first["records"]} & {r["source_group"] for r in fresh["records"]})
    assert not (
        {r["source_sha256"] for r in first["records"]} & {r["source_sha256"] for r in fresh["records"]}
    )
    output = {
        "status": "passed",
        "protocol_bindings": protocol_bindings,
        "normalized_source_inputs_rebuilt_exactly": normalized,
        "fresh_derived_controls_rebuilt_exactly": len(rebuilt),
        "environment_classifier_inputs_rebuilt_exactly": len(env_rows),
        "fresh_vs_first_group_and_source_hash_overlap": 0,
        "scope": "Input and known-edit derivation audit; classifier logits are retained outputs, not recomputed by this audit.",
    }
    save(ROOT / "input-audit.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
