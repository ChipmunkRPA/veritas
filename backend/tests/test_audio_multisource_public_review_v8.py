import pytest

from scripts.review_audio_multisource_plan_v8 import connections


def row(identifier, role="fit", group=None, **extra):
    return {
        "sha256": identifier,
        "production_pcm_sha256": identifier,
        "prefix_pcm_sha256": identifier,
        "text_connection_keys": [],
        "voice_key_sha256": identifier,
        "source_key_sha256": identifier,
        "role": role,
        "group": group or identifier,
        **extra,
    }


def test_public_connections_catch_hashed_script_leakage_across_roles():
    a, b = row("a"), row("b", "cal")
    a["text_connection_keys"] = b["text_connection_keys"] = ["same-hashed-prefix"]
    with pytest.raises(ValueError, match="crosses role/group"):
        connections([a, b], [])


def test_public_connections_allow_genuine_and_cloned_truth_in_one_role_group():
    a, b = row("a", group="shared", label=0), row("b", group="shared", label=1)
    b["voice_key_sha256"] = a["voice_key_sha256"]
    assert connections([a, b], [])["fit"] == 1


def test_public_connections_refuse_same_pcm_with_different_file_bytes():
    a, b = row("a"), row("b")
    b["production_pcm_sha256"] = a["production_pcm_sha256"]
    with pytest.raises(ValueError, match="full production audio"):
        connections([a, b], [])
