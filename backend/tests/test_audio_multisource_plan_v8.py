from collections import Counter

import pytest

from scripts.plan_audio_multisource_v8 import connected_sources, roles


def source(identifier, voice, label=0, text="", family="human", prompt=None):
    return {
        "id": identifier,
        "voice_key": voice,
        "source_key": identifier,
        "sha256": identifier,
        "label": label,
        "generator": family,
        "transcript": text,
        "dataset": "isjwdu/DFADD",
        "prompt_index": prompt,
    }


def test_paired_voice_and_known_prompt_connections_are_not_split():
    rows = [
        source("genuine", "voice-a"),
        source("cloned", "voice-a", 1, family="engine-a", prompt=42),
        source("same-prompt", "voice-b", 1, family="engine-b", prompt=42),
    ]
    groups = connected_sources(rows)
    assert len(groups) == 1
    assert set(groups[0]["sources"]) == {"genuine", "cloned", "same-prompt"}
    assert groups[0]["class_counts"] == {"0": 1, "1": 2}
    assert list(roles(groups).values()) == ["development"]


def test_shared_spoken_opening_connects_different_profile_ids():
    rows = [
        source("a", "voice-a", 1, "Today I will be presenting our work on astronomy", "engine"),
        source("b", "voice-b", 1, "Today I will be presenting our work on biology", "engine"),
    ]
    assert len(connected_sources(rows)) == 1


def test_role_assignment_is_fixed_under_inventory_order_changes():
    rows = [
        source(str(i), str(i), i // 8, text=f"different{i}", family="human" if i < 8 else "engine")
        for i in range(16)
    ]
    groups = connected_sources(rows)
    assigned = roles(groups)
    assert assigned == roles(list(reversed(connected_sources(list(reversed(rows))))))
    assert Counter(assigned.values()) == {"fit": 8, "cal": 4, "development": 4}


def test_repeated_original_source_identity_is_refused():
    with pytest.raises(ValueError, match="Duplicate"):
        connected_sources([source("same", "a"), source("same", "b")])
