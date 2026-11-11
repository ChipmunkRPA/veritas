from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import audio_language_scope_v184 as study


def test_first_position_uses_all_languages_and_margin():
    ids = [50259, 50268, 50269]
    answer = study.summarise_logits([0.0, 1.0, 3.0], ids)
    assert answer["label"] == "other"
    assert 0 < answer["margin"] < 1
    assert study.summarise_logits([2.0, 0.0, -1.0], ids)["label"] == "english"
    assert study.summarise_logits([0.0, 2.0, -1.0], ids)["label"] == "turkish"
    assert study.summarise_logits([1.0, 1.0, 0.0], ids)["label"] == "english"
    assert study.summarise_logits([1.0, 1.0, 0.0], ids)["margin"] == 0


def test_context_survives_json_round_trip_without_freezing(monkeypatch):
    monkeypatch.setattr(study, "model_assets", lambda: {})
    monkeypatch.setattr(study, "turkish_rows", lambda: [])
    monkeypatch.setattr(study, "english_rows", lambda: [])
    context, rows = study.context()
    assert rows == [] and json.loads(json.dumps(context)) == context
    assert context["margin_bins"] == [0.0, 0.1, 0.25, 0.5, 1.0]


@pytest.mark.parametrize("logits,ids", [([0, float("nan"), 1], [50259, 50268, 50269]),
                                          ([0, 1, 2], [50259, 50259, 50268]), ([0, 1], [50259, 50268])])
def test_invalid_or_incomplete_language_logits_are_rejected(logits, ids):
    with pytest.raises(ValueError, match="language-token"):
        study.summarise_logits(logits, ids)


def test_summary_keeps_unavailable_and_dependent_english_groups():
    rows = [{"slot": f"tr-{i}", "group": f"tr-{i}", "language": "tr",
             "label": "unavailable" if i == 0 else "turkish", "margin": None if i == 0 else .1}
            for i in range(24)]
    rows += [{"slot": f"en-{i}", "group": f"voice-{i // 2}", "language": "en",
              "label": "english" if i % 2 == 0 else "other", "margin": .5} for i in range(14)]
    summary = study.summarise(rows)
    assert summary["tr"]["planned"] == 24 and summary["tr"]["labels"]["unavailable"] == 1
    assert summary["tr"]["margin_bins"][".1-.25"] == 23
    assert summary["en"]["groups"] == 7 and summary["en"]["groups_with_any_english"] == 7
    assert summary["en"]["groups_with_any_non_english"] == 7 and summary["en"]["margin_bins"][".5-1"] == 14
    with pytest.raises(ValueError, match="planned language"):
        study.summarise(rows[:-1])
    rows[0]["label"] = "invented"
    with pytest.raises(ValueError, match="known diagnostic outcomes"):
        study.summarise(rows)


def test_english_selection_reads_development_genuine_only(tmp_path, monkeypatch):
    root = tmp_path / "run"
    stage = root / "bind-00"
    (stage / "output").mkdir(parents=True)
    media = tmp_path / "media"
    folder = media / "development"
    folder.mkdir(parents=True)
    assert not (media / "holdout").exists()
    public_freeze = tmp_path / "old-freeze.json"
    public_freeze.write_text("{}")
    (stage / "resources.jsonl").write_text("{}\n")
    pcm = np.zeros(48000, dtype=np.float32)
    pcm_sha = study.turkish.sha256(pcm.astype("<f4").tobytes()).hexdigest()
    foregrounds = []
    for i in range(14):
        path = folder / f"sample-{i}.wav"
        path.write_bytes(b"private fixture")
        foregrounds.append({"slot": f"sample-{i}", "group_id": f"group-{i // 2}", "voice": f"librispeech:{i // 2}",
                            "candidate": f"libri:{i}", "lane": "genuine", "status": "qualified", "license": "CC-BY-4.0",
                            "file": str(path), "wav_sha256": study.sha(path), "prefix_pcm_sha256": pcm_sha,
                            "wav_range": study.turkish.source_range(48000)})
    foregrounds.append({"lane": "clone", "file": "must-not-read"})
    manifest = folder / "private-manifest.json"
    manifest.write_text(json.dumps({"foregrounds": foregrounds, "role": "development", "scored": False}))
    development = {"private_manifest_sha256": study.sha(manifest)}
    result = {"status": "complete", "holdout_scored": False, "new_neural_calls": 0,
              "freeze_sha256": study.sha(public_freeze), "roles": {"development": development}}
    result_path = stage / "output/result.json"
    result_path.write_text(json.dumps(result))
    profile = {"child_exit_code": 0, "guard": None, "sample_errors": 0, "new_neural_calls": 0,
               "worker_receipt_sha256": study.sha(result_path), "resource_journal_sha256": study.sha(stage / "resources.jsonl")}
    profile_path = stage / "profile.json"
    profile_path.write_text(json.dumps(profile))
    receipt = tmp_path / "public-result.json"
    receipt.write_text(json.dumps({"resource_profile_sha256": study.sha(profile_path), "roles": {"development": development}}))
    monkeypatch.setattr(study.english, "TRIAL", root)
    monkeypatch.setattr(study.english, "MEDIA", media)
    monkeypatch.setattr(study.english, "PUBLIC", public_freeze)
    monkeypatch.setattr(study, "ENGLISH_RECEIPT", receipt)
    monkeypatch.setattr(study, "read_wav", lambda _: pcm)
    rows = study.english_rows()
    assert len(rows) == 14 and len({row["group"] for row in rows}) == 7
    assert not (media / "holdout").exists()
    (folder / "sample-0.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Exact development WAV"):
        study.english_rows()
