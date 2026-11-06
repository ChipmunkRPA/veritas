import copy
import hashlib
import json

import numpy as np
import pytest

from scripts import run_audio_layer_v12 as execution


def original():
    value = np.zeros(1024, np.float32)
    return {
        "id": "fixture",
        "role": "fit",
        "sha256": "source",
        "pcm_sha256": "pcm",
        "status": "ok",
        "raw": 0.5,
        "embedding": value.tolist(),
        "embedding_sha256": hashlib.sha256(value.astype("<f4").tobytes()).hexdigest(),
        "speech_spans_s": [[0, 4]],
        "windows": [{"start_sample": 0, "end_sample": 64000}],
    }


def layer_record():
    r = original()
    value = np.zeros((3, 1024), np.float32)
    r.update(
        layer_ids=[8, 16, 24],
        layer_embeddings=value.tolist(),
        layer_embedding_sha256=hashlib.sha256(value.astype("<f4").tobytes()).hexdigest(),
        attention_mask_sha256="a" * 64,
        feature_frames=199,
        valid_feature_frames=199,
    )
    return r


def reference():
    return {"head_weight": [1.0] + [0.0] * 1023, "head_bias": 0.0}


def test_new_layer_record_preserves_original_raw_vector_vad_and_window():
    record = layer_record()
    execution.validate_original(record, original(), reference())
    record["speech_spans_s"] = [[0.1, 4]]
    with pytest.raises(ValueError, match="changed original"):
        execution.validate_original(record, original(), reference())


def test_layer_payload_requires_exact_float32_not_only_the_same_quantized_hash():
    record = layer_record()
    record["layer_embeddings"][0][0] = float(np.float32(0.1))
    record["layer_embedding_sha256"] = hashlib.sha256(
        np.asarray(record["layer_embeddings"], "<f4").tobytes()
    ).hexdigest()
    execution.validate_layers(record)
    record["layer_embeddings"][0][0] = float(np.nextafter(record["layer_embeddings"][0][0], np.inf))
    with pytest.raises(ValueError, match="Exact finite float32"):
        execution.validate_layers(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("layer_ids", [24, 16, 8]),
        ("valid_feature_frames", 0),
        ("feature_frames", 198),
        ("attention_mask_sha256", "private-mask"),
    ],
)
def test_layer_order_and_original_frame_clock_cannot_drift(field, value):
    record = layer_record()
    record[field] = value
    with pytest.raises(ValueError):
        execution.validate_layers(record)


def test_unassessed_record_cannot_carry_measured_layer_vectors():
    r = layer_record()
    r["status"] = "not_applicable"
    with pytest.raises(ValueError, match="Unassessed speech"):
        execution.validate_layers(r)
    for k in execution.LAYER_FIELDS:
        if k != "layer_ids":
            r[k] = None
    execution.validate_layers(r)


def test_resume_refuses_a_failed_prefix_without_erasing_it_or_loading_models(tmp_path, monkeypatch):
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}")
    lock = {"weights": {}, "runtime": {}}
    source = original()
    broken = {**layer_record(), "status": "failed"}
    path = tmp_path / "features.jsonl"
    header = execution.representation_header(lock_path, lock)
    path.write_text(json.dumps(header) + "\n" + json.dumps(broken) + "\n")
    before = path.read_bytes()
    monkeypatch.setattr(execution, "check", lambda _: (lock, ({}, reference(), [source], {}, {})))
    monkeypatch.setattr(execution, "weights", lambda: {})
    monkeypatch.setattr(
        execution, "load_models", lambda _: pytest.fail("Model load must not occur for a failed prefix")
    )
    with pytest.raises(ValueError, match="Failed features"):
        execution.extract(lock_path, path)
    assert path.read_bytes() == before


def test_complete_feature_reader_rejects_a_different_header_schema(tmp_path, monkeypatch):
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}")
    lock = {"weights": {}, "runtime": {}}
    source = original()
    header = execution.representation_header(lock_path, lock)
    changed = copy.deepcopy(header)
    changed["schema"] = "another-profile"
    path = tmp_path / "features.jsonl"
    path.write_text(json.dumps(changed) + "\n" + json.dumps(layer_record()) + "\n")
    monkeypatch.setattr(execution, "check", lambda _: (lock, ({}, reference(), [source], {}, {})))
    with pytest.raises(ValueError, match="Every original layer feature"):
        execution.feature_rows(lock_path, path)
