import pytest

from scripts.prepare_audio_multisource_v8 import connected_roles


def row(identifier, role="fit", **changes):
    return {
        "id": identifier,
        "role": role,
        "voice_key": identifier,
        "source_key": identifier,
        "sha256": identifier,
        "production_pcm_sha256": identifier,
        "prefix_pcm_sha256": identifier,
        "dataset": "fixture",
        "label": 0,
        **changes,
    }


def test_actual_prefix_connection_cannot_cross_roles():
    with pytest.raises(ValueError, match="crosses frozen roles"):
        connected_roles([row("a"), row("b", "cal", prefix_pcm_sha256="a")])


def test_cloned_and_genuine_voice_share_one_group():
    assignments, groups = connected_roles([row("a"), row("b", voice_key="a", label=1)])
    assert assignments["a"] == assignments["b"]
    assert len(groups) == 1


def test_background_prompt_cannot_connect_to_foreground_even_in_same_role():
    text = "one two three four five six seven eight nine ten"
    with pytest.raises(ValueError, match="Background speaker/prompt"):
        connected_roles([row("a", transcript=text), row("b", kind="speech", transcript=text)])


def test_existing_planned_connections_are_preserved():
    with pytest.raises(ValueError, match="crosses frozen roles"):
        connected_roles([row("a", group="old"), row("b", "development", group="old")])


def test_shared_prefix_groups_are_deterministic():
    rows = [row("a"), row("b", prefix_pcm_sha256="a"), row("c")]
    assert connected_roles(rows) == connected_roles(list(reversed(rows)))


def test_duplicate_identity_is_refused():
    with pytest.raises(ValueError, match="Duplicate"):
        connected_roles([row("a"), row("a")])
