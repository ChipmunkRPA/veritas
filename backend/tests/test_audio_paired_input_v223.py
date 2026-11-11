"""Verify the consumed-role loader without accessing real audio or model runtimes."""

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from scripts import audio_paired_input_v223 as inputs
from scripts.audio_score_fusion_v83 import METHOD


def test_import_is_cold_and_confirmation_chunk_is_out_of_scope():
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; from scripts import audio_paired_input_v223; "
            "assert not {'torch', 'transformers', 'app.core.config'}.intersection(sys.modules)",
        ],
        check=True,
    )
    for bad in (-1, 36, 47, True):
        with pytest.raises(ValueError, match="three consumed"):
            inputs.chunk_paths(bad)


def test_role_sources_reject_known_voice_or_tape_aliases():
    fit = {"role": "fusion_fit", "raw_sha256": "a", "pcm_sha256": "p", "canonical_speaker": "v"}
    cal = {"role": "confidence_cal", "raw_sha256": "b", "pcm_sha256": "q", "canonical_speaker": "w"}
    assert inputs.role_sources([fit, cal]) == {"a": fit, "b": cal}
    for key in ("canonical_speaker", "pcm_sha256"):
        with pytest.raises(ValueError, match="crosses v223 roles"):
            inputs.role_sources([fit, cal | {key: fit[key]}])


def test_original_admission_preserves_exact_feature_and_unavailable_status():
    feature = np.linspace(-1, 1, inputs.WIDTH, dtype="<f4")
    row = {
        **{key: "bound" for key in inputs.META},
        "status": "ok",
        "raw": 0.4,
        "original_p": 0.2,
        "original_decision": "inconclusive",
        "candidates": {"complement": {"p": 0.7}},
        "wavlm": feature.tolist(),
        "wavlm_sha256": hashlib.sha256(feature.tobytes()).hexdigest(),
        "feature_frames": 199,
        "valid_feature_frames": 199,
        "input_pcm_sha256": "bound",
        "windows": [{"start_sample": 0, "end_sample": 64000}],
    }
    head = {
        "method": METHOD,
        "arm": "global",
        "branches": {
            "all": {"status": "calibrated", "coefficients": [0.0, 1.0, 0.0], "calibration": [1.0, 0.0]}
        },
    }
    projected = inputs.observation(row, row, None, head)
    assert projected["wavlm"] == row["wavlm"]
    assert projected["v83_global_p"] == pytest.approx(0.7)
    assert projected["background_sha256"] is None
    with pytest.raises(ValueError, match="sample clock"):
        inputs.observation(row | {"feature_frames": 198}, row, None, head)
    unassessed = {
        **row,
        **dict.fromkeys(
            (
                "raw",
                "original_p",
                "wavlm",
                "wavlm_sha256",
                "feature_frames",
                "valid_feature_frames",
                "input_pcm_sha256",
            )
        ),
        "status": "not_applicable",
        "candidates": {"complement": {"p": None}},
        "windows": [],
    }
    assert inputs.observation(unassessed, row, "tape", head)["v83_global_p"] is None
    with pytest.raises(ValueError, match="Unassessed"):
        inputs.observation(unassessed | {"raw": 0.0}, row, "tape", head)


def test_chunk_requires_exact_full_line_digest_and_plan_order(tmp_path, monkeypatch):
    short_path, full_path = tmp_path / "compact.jsonl", tmp_path / "full.jsonl"
    monkeypatch.setattr(inputs, "chunk_paths", lambda _: (short_path, full_path, tmp_path))
    monkeypatch.setattr(inputs, "CHUNK", 1)
    meta = {key: "value" for key in inputs.META} | {"condition": "music_5"}
    full_line = (json.dumps(meta) + "\n").encode()
    full_path.write_bytes(full_line)
    compact = meta | {"full_record_sha256": hashlib.sha256(full_line).hexdigest()}
    short_path.write_text(json.dumps(compact) + "\n")
    commitment = {
        "completed": 1,
        "compact_sha256": inputs.sha(short_path),
        "full_sha256": inputs.sha(full_path),
    }
    assert inputs.read_chunk(0, [meta], commitment, {}, {}, {}) == []
    with pytest.raises(ValueError, match="variant lineage"):
        inputs.read_chunk(0, [meta | {"id": "changed"}], commitment, {}, {}, {})
    full_path.write_bytes(full_line.rstrip() + b" \n")
    with pytest.raises(ValueError, match="variant lineage"):
        inputs.read_chunk(0, [meta], commitment, {}, {}, {})
