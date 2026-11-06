import io
import tarfile

import pytest

from scripts.prepare_audio_confirmation_v7 import choose_duplex, tar_bytes, text_keys, user_turns


def test_file_indices_follow_synthesized_turns_not_longer_original_dialogue():
    meta = {
        "speakers": ["assistant", "user"],
        "utterances": [{"speaker": "assistant"}] * 28,
        "speech_meta": [
            {"speaker": 0, "tts_text": "assistant words"},
            {"speaker": 1, "tts_text": "actual cloned user speech"},
            {"speaker": 0, "tts_text": "more assistant words"},
        ],
    }
    assert user_turns(meta) == [(1, "actual cloned user speech")]


def test_known_openings_connect_fluent_and_disfluent_scripts():
    a = text_keys("The flow of genetic information in cells from DNA to RNA.")
    b = text_keys("The flow of genetic information in cells from DNA, um, to RNA.")
    assert a & b
    assert text_keys("  HELLO, world! ") == text_keys("hello world")
    assert text_keys("") == set()


def test_duplex_selection_keeps_distinct_dialogues_voices_and_prefixes():
    rows = [
        {
            "id": str(i),
            "source_group": "dialogue:" + str(i),
            "prompt_speaker_id": str(i),
            "member": f"utterances/{i:02d}.wav",
            "seconds": 4.0,
            "transcript": f"unique{i} user transcript",
        }
        for i in range(32)
    ]
    rows.append({**rows[0], "member": "duplicate.wav"})
    rows.append({**rows[1], "source_group": "other-dialogue", "member": "other.wav"})
    selected = choose_duplex(rows, text_keys("unique31 user transcript"))
    assert len(selected) == 30
    for key in ("source_group", "prompt_speaker_id", "transcript"):
        assert len({r[key] for r in selected}) == 30
    assert all(r["id"] != "31" for r in selected)


def test_archive_member_traversal_is_refused_without_extraction():
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w") as tar:
        member = tarfile.TarInfo("../outside.wav")
        member.size = 3
        tar.addfile(member, io.BytesIO(b"wav"))
    blob.seek(0)
    with tarfile.open(fileobj=blob) as tar, pytest.raises(ValueError, match="Unsafe"):
        tar_bytes(tar, tar.getmembers()[0])
