from hashlib import sha1, sha256

import pytest

from scripts import grid_blind_transcript_v163 as study


def authored_roster():
    matched = {"words": [[word, 0, 1] for word in ["put", "blue", "at", "a", "four", "now"]]}
    old = {"blind_exact_words": False, "word_edits": 5, "words": 6}
    unmatched = {"unavailable": study.UNMATCHED}
    return [matched, unmatched] * 8, [old, unmatched] * 8


def test_publisher_asset_requires_exact_blob_and_sha(tmp_path):
    body = b'{"example":true}'
    path = tmp_path / "publisher.json"
    path.write_bytes(body)
    blob = sha1(f"blob {len(body)}\0".encode() + body, usedforsecurity=False).hexdigest()
    assert study.file_bytes(path, len(body), blob, sha256(body).hexdigest()) == body
    with pytest.raises(ValueError, match="asset bytes"):
        study.file_bytes(path, len(body), blob, "0" * 64)


def test_controls_and_truncated_sequences_cannot_be_read_as_words():
    prefix = list(study.PREFIX)
    assert study.generated_content([*prefix, 19, 32, study.END]) == [19, 32]
    assert study.generated_content([*prefix, study.END]) == []
    with pytest.raises(ValueError, match="did not finish"):
        study.generated_content([*prefix, 19, 32])
    with pytest.raises(ValueError, match="control or timestamp"):
        study.generated_content([*prefix, 50364, study.END])
    with pytest.raises(ValueError, match="prefix"):
        study.generated_content([*prefix[:-1], 19, study.END])


def test_blind_input_only_and_all_roster_rows_are_preserved(monkeypatch):
    inputs, prior = authored_roster()
    waveforms = iter(range(8))
    published, seen = [], []
    monkeypatch.setattr(study, "waveform", lambda _row: next(waveforms))
    monkeypatch.setattr(study.native, "fingerprint", str)

    def blind(waveform):
        seen.append(waveform)
        if waveform == 0:
            raise ValueError("private failure detail")
        return "Put blue at A 4 now."

    rows = study.observe(inputs, prior, blind, published.append)
    assert rows == published
    assert [r["roster_ordinal"] for r in rows] == list(range(16))
    assert seen == list(range(8))
    assert rows[0] == {"roster_ordinal": 0, "unavailable": "ValueError"}
    result = study.summary(rows)
    assert result["publisher_word_reference_unmatched"] == 8
    assert result["other_unavailable"] == 1
    assert result["blind_prompts_assessed"] == 7
    assert result["blind_reference_words"] == 42
    assert result["blind_exact_prompts"] == 0
    assert result["blind_word_edits"] == 7
    assert result["digit_equivalent_exact_prompts"] == 7
    assert result["paired_v162_word_edits"] == 35
    assert result["paired_fewer_edits"] == 7
    assert "hypothesis" not in result


def test_empty_completed_hypothesis_counts_deletions():
    inputs, prior = authored_roster()
    scored = study.score(inputs[0], prior[0], "")
    assert scored["edits"] == scored["digit_equivalent_edits"] == 6
    assert not scored["exact"]


def test_previous_baseline_must_reconcile_exact_denominator_and_order():
    inputs, prior = authored_roster()
    previous_result = {
        "planned_originals": 16,
        "blind_prompts_assessed": 8,
        "publisher_word_reference_unmatched": 8,
        "other_unavailable": 0,
        "blind_exact_prompts": 0,
        "blind_word_edits": 40,
        "blind_reference_words": 48,
    }
    study.check_roster(inputs, prior, previous_result)
    with pytest.raises(ValueError, match="reconcile"):
        study.check_roster(inputs, prior, {**previous_result, "blind_word_edits": 38})
    with pytest.raises(ValueError, match="Unmatched"):
        study.check_roster(inputs, [prior[0]] * 16, previous_result)
