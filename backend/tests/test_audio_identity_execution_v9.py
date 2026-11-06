import copy
import json

import numpy as np
import pytest
from scipy.special import expit

from scripts import run_audio_identity_v9 as runner
from scripts.audio_residual_head_v6 import digest_embedding
from scripts.run_audio_identity_v9 import (
    CONDITIONS,
    native_selection,
    reuse_record,
    validate,
    validate_feature,
)


def panel():
    sources, backgrounds, rows = [], [], []
    for role, count in (("fit", 214), ("cal", 97), ("development", 161)):
        for kind in ("noise", "music", "speech"):
            backgrounds.append({"sha256": f"{role}:{kind}", "kind": kind, "role": role, "dataset": "fixture"})
        for i in range(count):
            source = {
                "id": f"{role}:{i}",
                "sha256": f"source:{role}:{i}",
                "role": role,
                "group": f"group:{role}:{i}",
                "label": i % 2,
                "generator": "fixture-ai" if i % 2 else "human",
                "dataset": "fixture",
            }
            sources.append(source)
            for condition in CONDITIONS:
                kind, snr = (None, None) if condition == "clean" else condition.split("_")
                rows.append(
                    {
                        **{k: source[k] for k in ("role", "group", "label", "generator", "dataset")},
                        "id": f"{source['id']}:{condition}",
                        "base_id": source["id"],
                        "source_sha256": source["sha256"],
                        "condition": condition,
                        "samples": 64000,
                        "sample_rate": 16000,
                        "seconds": 4.0,
                        "source_interval_samples": [0, 64000],
                        "background_sha256": f"{role}:{kind}" if kind else None,
                        "background_role": role if kind else None,
                        "background_kind": kind,
                        "snr_db": int(snr) if snr is not None else None,
                    }
                )
    return {"panel": "development", "variants": rows}, {"sources": sources, "backgrounds": backgrounds}


def feature():
    embedding = np.zeros(1024, np.float32)
    embedding[0] = 2
    head = np.zeros(1024, np.float32)
    head[0] = 1
    return {
        "id": "old-id",
        "role": "fit",
        "sha256": "bytes",
        "pcm_sha256": "pcm",
        "status": "ok",
        "raw": float(expit(2)),
        "embedding": embedding.tolist(),
        "embedding_sha256": digest_embedding(embedding),
        "speech_spans_s": [[0.1, 3.9]],
        "windows": [{"start_sample": 0, "end_sample": 64000}],
    }, {"head_weight": head.tolist(), "head_bias": 0.0}


def test_complete_corrected_source_and_condition_denominators():
    body, plan = panel()
    validate(body, plan)
    assert len(body["variants"]) == 4248


def test_fresh_confirmation_cannot_enter_development():
    body, plan = panel()
    body["variants"][0]["role"] = "confirmation"
    with pytest.raises(ValueError, match="fresh confirmation"):
        validate(body, plan)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ({"background_role": "cal"}, "Backgrounds cannot cross"),
        ({"source_sha256": "another-source"}, "binding changed"),
        ({"label": 1}, "binding changed"),
        ({"sample_rate": 8000}, "source clock"),
        ({"source_interval_samples": [1, 64001]}, "source clock"),
        ({"snr_db": 15}, "actual background recipe"),
        ({"background_sha256": None}, "actual background recipe"),
    ],
)
def test_mislabeled_source_clock_or_background_is_rejected(mutation, reason):
    body, plan = panel()
    body["variants"][1].update(mutation)
    with pytest.raises(ValueError, match=reason):
        validate(body, plan)


def test_missing_condition_and_hidden_clean_background_are_rejected():
    body, plan = panel()
    altered = copy.deepcopy(body)
    altered["variants"].pop()
    with pytest.raises(ValueError, match="Every corrected"):
        validate(altered, plan)
    body["variants"][0]["background_kind"] = "noise"
    with pytest.raises(ValueError, match="hidden background"):
        validate(body, plan)


def test_cache_preserves_corrected_role_and_source_identity():
    old, head = feature()
    new = {"id": "corrected-id", "role": "cal", "sha256": "bytes", "pcm_sha256": "pcm"}
    actual = reuse_record(new, old)
    validate_feature(actual, head)
    assert actual["role"] == "cal" and actual["id"] == "corrected-id"
    assert actual["cached_prior_role"] == "fit" and actual["cached_row_id"] == "old-id"
    assert old["role"] == "fit"
    new["pcm_sha256"] = "changed"
    with pytest.raises(ValueError, match="exact input"):
        reuse_record(new, old)


def test_corrupt_vector_and_nonfinite_reference_cannot_pass_parity():
    old, head = feature()
    validate_feature(old, head)
    old["embedding"][1] = 1
    with pytest.raises(ValueError, match="representation changed"):
        validate_feature(old, head)
    old, head = feature()
    head["head_weight"][0] = float("nan")
    with pytest.raises(ValueError, match="Finite original checkpoint"):
        validate_feature(old, head)


def test_unassessed_remains_unassessed_and_failed_cache_is_refused():
    old, head = feature()
    old.update(status="not_applicable", raw=None, embedding=None, windows=[])
    validate_feature(old, head)
    new = {"id": "new", "role": "cal", "sha256": "bytes", "pcm_sha256": "pcm"}
    assert reuse_record(new, old)["status"] == "not_applicable"
    old["status"] = "failed"
    with pytest.raises(ValueError, match="Failed features"):
        reuse_record(new, old)


def test_native_replay_selection_uses_cells_and_status_not_response_strength():
    rows, cache = [], {}
    for label in (0, 1):
        for condition in CONDITIONS:
            for i in range(2):
                identifier = f"{label}:{condition}:{i}"
                rows.append(
                    {
                        "id": identifier,
                        "role": "fit",
                        "label": label,
                        "condition": condition,
                        "sha256": identifier,
                        "pcm_sha256": identifier,
                    }
                )
                cache[identifier] = {"pcm_sha256": identifier, "status": "ok", "raw": i / 2}
    cache["0:clean:1"]["status"] = "not_applicable"
    selected = native_selection(rows, cache)
    assert len(selected) == 19 and "0:clean:1" in selected
    for record in cache.values():
        record["raw"] = 0.999
    assert native_selection(list(reversed(rows)), cache) == selected


@pytest.mark.parametrize("changed", ["raw", "embedding_sha256", "speech_spans_s", "windows"])
def test_native_cache_disagreement_is_recorded_and_stops_execution(tmp_path, monkeypatch, changed):
    record, _ = feature()
    row = {**record, "label": 0, "condition": "clean"}
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"native_cache_ids": [row["id"]]}))
    actual = copy.deepcopy(record)
    actual[changed] = {"raw": 0.2, "embedding_sha256": "changed", "speech_spans_s": [], "windows": []}[
        changed
    ]
    monkeypatch.setattr(runner.prior, "input_pcm", lambda *_: None)
    monkeypatch.setattr(runner, "extract_one", lambda *_: actual)
    journal = tmp_path / "audits.jsonl"
    with pytest.raises(ValueError, match="never silently fall back"):
        runner.native_audit(lock, {"variants": [row]}, {row["sha256"]: record}, None, None, [], journal)
    audit = json.loads(journal.read_text())
    assert not audit["passed"] and not audit["cases"][0]["passed"]


def test_representation_header_cannot_replace_the_original_reference_head(tmp_path, monkeypatch):
    record, reference = feature()
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}")
    lock = {"weights": {}, "runtime": {}, "cache_reference_header": reference}
    body = {"variants": [record]}
    monkeypatch.setattr(runner, "check", lambda _: (lock, body))
    header = {
        "lock_sha256": runner.digest(lock_path),
        "weights": {},
        "runtime": {},
        "cache_sha256": runner.CACHE_SHA,
        **reference,
    }
    receipt = tmp_path / "features.jsonl"
    receipt.write_text(json.dumps(header) + "\n" + json.dumps(record) + "\n")
    assert len(runner.feature_rows(lock_path, receipt)[2]) == 1
    header["head_bias"] = 1
    receipt.write_text(json.dumps(header) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="bound terminal feature"):
        runner.feature_rows(lock_path, receipt)
