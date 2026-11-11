import pytest

from scripts import audio_okur_turkish_followup_v245 as study


def test_exact_followup_text_regime_and_disjoint_seeds():
    rows = study.plan()
    prior = study.previous.plan()
    assert len(rows) == 24
    assert all(50 <= len(row["phrase"]) <= 56 and 7 <= len(row["phrase"].split()) <= 8 for row in rows)
    for key in ("phrase", "token_sha256", "seed"):
        values = {row[key] for row in rows}
        assert len(values) == 24 and values.isdisjoint(row[key] for row in prior)


def test_prior_text_cannot_enter_followup(monkeypatch):
    monkeypatch.setattr(study, "PHRASES", (study.previous.PHRASES[0], *study.PHRASES[1:]))
    with pytest.raises(ValueError, match="separate from v240"):
        study.plan()


@pytest.mark.parametrize("key", study.FINGERPRINTS)
def test_each_prior_waveform_commitment_is_excluded(key):
    old = {"status": "qualified", **{name: f"prior-{name}" for name in study.FINGERPRINTS}}
    new = {"status": "qualified", **{name: f"follow-up-{name}" for name in study.FINGERPRINTS}}
    study.reject_previous_waveforms([new, {"status": "unavailable"}], [old])
    with pytest.raises(ValueError, match="matches a v240 fingerprint"):
        study.reject_previous_waveforms([{**new, key: old[key]}], [old])


def test_worker_keeps_prior_asset_verification_and_uses_new_receipt_paths():
    scope = study.worker.__globals__
    assert scope["verified_assets"] is study.previous.verified_assets
    assert scope["STAGE"] == study.STAGE != study.previous.STAGE
    assert scope["OUTPUT"] == study.OUTPUT != study.previous.OUTPUT
    assert scope["PUBLIC"] == study.PUBLIC != study.previous.PUBLIC
    assert scope["prepare_case"].__globals__["OUTPUT"] == study.OUTPUT
