from scripts.plan_audio_identity_v9 import repair


def human(identifier, speaker, role):
    return {
        "id": identifier,
        "sha256": identifier,
        "label": 0,
        "generator": "human",
        "dataset": "openslr/librispeech_asr",
        "speaker": speaker,
        "role": role,
        "group": identifier,
    }


def clone(identifier, speaker, role):
    return {
        "id": identifier,
        "sha256": identifier,
        "label": 1,
        "generator": "Chatterbox",
        "dataset": "DuplexGen/duplexgen-spoken",
        "prompt_speaker_id": speaker,
        "role": role,
        "group": identifier,
    }


def test_consumed_clone_is_retired_instead_of_replacing_fresh_genuine():
    rows = [
        human("fresh", "101", "confirmation"),
        clone("old", "101", "fit"),
        human("other", "102", "development"),
    ]
    selected, _, retired = repair(rows, [])
    assert {r["id"] for r in selected if r["role"] == "confirmation"} == {"fresh"}
    assert [r["id"] for r in retired] == ["old"]
    assert len(selected) + len(retired) == len(rows)


def test_genuine_and_cloned_confirmation_recordings_share_one_resampling_group():
    rows = [
        human("fresh-real", "101", "confirmation"),
        clone("fresh-ai", "101", "confirmation"),
        human("old", "102", "fit"),
    ]
    selected, groups, retired = repair(rows, [])
    fresh = [r for r in selected if r["role"] == "confirmation"]
    assert len(fresh) == 2 and len({r["group"] for r in fresh}) == 1
    assert not retired
    assert sum(g["role"] == "confirmation" for g in groups) == 1


def test_known_shared_consumed_voice_is_kept_together_without_dropping_cases():
    rows = [
        human("fresh", "999", "confirmation"),
        human("old-real", "101", "fit"),
        clone("old-ai", "101", "cal"),
    ]
    selected, _, retired = repair(rows, [])
    old = [r for r in selected if r["id"].startswith("old")]
    assert len({r["role"] for r in old}) == 1 and len({r["group"] for r in old}) == 1
    assert not retired
    assert {r["id"] for r in selected} == {r["id"] for r in rows}
