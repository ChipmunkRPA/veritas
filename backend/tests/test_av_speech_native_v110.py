import json

import numpy as np
import pytest

from scripts import av_speech_runtime_v110 as runtime
from scripts.av_speech_native_v110 import validate_pcm, validate_states


def fixture():
    base = np.arange(1500 * 384, dtype=np.float32).reshape(1, 1500, 384)
    values = [base + np.float32(i) for i in range(5)]
    return values, values[-1].copy(), [x.copy() for x in values[:4]]


def test_state_packing_keeps_native_time_then_layer_order():
    values, last, captured = fixture()
    actual = validate_states(values, last, captured)
    assert actual.shape == (1500, 5, 384) and actual.dtype == np.float32
    assert actual.flags.c_contiguous
    for i in range(5):
        assert np.array_equal(actual[:, i], values[i][0])


@pytest.mark.parametrize(
    "problem", ["count", "capture_count", "shape", "dtype", "nan", "order", "last", "capture"]
)
def test_incomplete_or_changed_native_readout_is_rejected(problem):
    values, last, captured = fixture()
    if problem == "count":
        values.pop()
    elif problem == "capture_count":
        captured.pop()
    elif problem == "shape":
        values[0] = values[0][:, :-1]
    elif problem == "dtype":
        values[0] = values[0].astype(np.float64)
    elif problem == "nan":
        values[0][0, 0, 0] = np.nan
    elif problem == "order":
        values[0], values[1] = values[1], values[0]
    elif problem == "last":
        last[0, 0, 0] += 1
    else:
        captured[0][0, 0, 0] += 1
    with pytest.raises(ValueError):
        validate_states(values, last, captured)


def test_source_pcm_contract_does_not_normalize_or_mutate_input():
    pcm = np.linspace(-1, 1, 32000, dtype=np.float32)
    before = pcm.copy()
    validate_pcm(pcm)
    assert np.array_equal(pcm, before)


@pytest.mark.parametrize("problem", ["dtype", "stereo", "short", "long", "nan", "range"])
def test_unqualified_pcm_is_rejected_before_encoder_execution(problem):
    pcm = np.ones(32000, np.float32) * 0.5
    if problem == "dtype":
        pcm = pcm.astype(np.float64)
    elif problem == "stereo":
        pcm = pcm.reshape(-1, 2)
    elif problem == "short":
        pcm = pcm[:639]
    elif problem == "long":
        pcm = np.ones(480001, np.float32)
    elif problem == "nan":
        pcm[0] = np.nan
    else:
        pcm[0] = 1.001
    with pytest.raises(ValueError):
        validate_pcm(pcm)


def test_existing_stage_is_preserved_without_a_child(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "TRIAL", tmp_path)
    monkeypatch.setattr(runtime, "verify", lambda: None)
    (tmp_path / "authored").mkdir()
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **k: pytest.fail("No retry"))
    with pytest.raises(ValueError, match="Preserve"):
        runtime.run("authored")


def test_neural_policy_is_the_unchanged_6gib_admission():
    row = {
        "elapsed_s": 0.0,
        "tree_rss_bytes": 0,
        "ram_available_bytes": int(5.9 * 1024**3),
        "disk_free_bytes": 3 * 1024**3,
    }
    assert "initial_ram_headroom" in runtime.violations(row, "authored", initial=True)
    row["ram_available_bytes"] = 6 * 1024**3
    assert runtime.violations(row, "authored", initial=True) == []


def test_short_ram_after_monitor_creates_no_neural_stage(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(runtime, "TRIAL", tmp_path)
    monkeypatch.setattr(runtime, "ARCHIVE", tmp_path)
    monkeypatch.setattr(runtime, "verify", lambda: None)
    monkeypatch.setattr(runtime, "sha", lambda path: "record")
    output = tmp_path / "monitor/output"
    output.mkdir(parents=True)
    (output / "result.json").write_text(
        json.dumps({"status": "complete", "owned_descendant_sampled": True, "sample_helper_sha256": "record"})
    )
    (output.parent / "profile.json").write_text(
        json.dumps(
            {"child_exit_code": 0, "guard": None, "sample_errors": 0, "worker_receipt_sha256": "record"}
        )
    )
    monkeypatch.setattr(
        runtime.limits,
        "observe",
        lambda *a: {
            "elapsed_s": 0.0,
            "tree_rss_bytes": 0,
            "ram_available_bytes": int(5.3 * 1024**3),
            "disk_free_bytes": 3 * 1024**3,
        },
    )
    monkeypatch.setattr(
        runtime.subprocess, "Popen", lambda *a, **k: pytest.fail("No inference before admission")
    )
    runtime.run("authored")
    assert "pending_headroom_no_stage_attempt" in capsys.readouterr().out
    assert not (tmp_path / "authored").exists()
