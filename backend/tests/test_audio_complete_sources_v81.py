import io

import numpy as np
import pytest
import soundfile as sf

from scripts.audio_complete_sources_v81 import connected_roles, float_wav


def source(voice, group, role="fit", **extras):
    return {"canonical_speaker": voice, "study_group": group, "role": role, "foreground": True, **extras}


def test_float_wav_retains_values_outside_pcm16_and_native_clock():
    values = np.array([0.0, -0.0, -1.2, 1.2, 0.12345679], dtype=np.float32)
    raw = float_wav(values)
    restored, rate = sf.read(io.BytesIO(raw), dtype="float32")
    assert rate == 24000
    assert sf.info(io.BytesIO(raw)).subtype == "FLOAT"
    assert restored.tobytes() == values.tobytes()


def test_actual_shared_source_and_paired_text_stay_in_one_declared_role():
    rows = [source("human", "paired", transcript="The boat crossed the quiet lake before dawn."),
            source("stock", "paired", transcript="The boat crossed the quiet lake before dawn."),
            source("other", "separate", role="confirmation")]
    assert connected_roles(rows) == 2
    rows[1]["role"] = "confirmation"
    with pytest.raises(ValueError, match="crosses roles"):
        connected_roles(rows)


def test_new_pcm_alias_cannot_silently_merge_separate_roles_or_groups():
    rows = [source("first", "one", pcm_sha256="same-pcm"),
            source("second", "two", pcm_sha256="same-pcm")]
    with pytest.raises(ValueError, match="merged previously separate"):
        connected_roles(rows)


def test_disconnected_declared_groups_and_foreground_background_reuse_fail():
    with pytest.raises(ValueError, match="disconnected"):
        connected_roles([source("first", "one"), source("second", "one")])
    with pytest.raises(ValueError, match="Foreground and background"):
        connected_roles([source("first", "one"), source("first", "one", foreground=False)])
