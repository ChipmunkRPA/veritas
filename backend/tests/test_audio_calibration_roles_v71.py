import pytest

from scripts.audio_calibration_roles_v71 import (
    ROLES,
    assert_voice_roles,
    choose_scripts,
    round_robin,
    stock_assignments,
)


def test_role_recipe_is_deterministic_and_voice_disjoint():
    voices = [f"voice-{n}" for n in range(60)]
    first = round_robin(voices, 14)
    assert first == round_robin(list(reversed(voices)), 14)
    assert {role: list(first.values()).count(role) for role in ROLES} == dict.fromkeys(ROLES, 14)
    assert len(first) == 56


def test_parent_matched_sources_must_share_role():
    rows = [{"canonical_speaker": "real", "role": "fusion_fit"},
            {"canonical_speaker": "stock", "role": "confidence_cal", "matched_genuine_voice": "real"}]
    with pytest.raises(ValueError, match="crosses"):
        assert_voice_roles(rows)


def test_declared_scripts_cannot_be_reused_under_different_voices():
    pool = [{"id": "first", "text_identity": ["text:shared", "prefix8:shared"]}]
    with pytest.raises(ValueError, match="distinct"):
        choose_scripts(pool, 1, {"prefix8:shared"})


def test_stock_profiles_balance_roles_and_keep_british_support_in_each():
    entries = [{"canonical_speaker": f"{language}-{n}", "language": language}
               for language, count in [("a", 11), ("b", 5)] for n in range(count)]
    result = stock_assignments(entries)
    assert len(result) == 16
    assert all(sum(row["role"] == role for row in result) == 4 for role in ROLES)
    assert all(any(row["language"] == "b" and row["role"] == role for row in result) for role in ROLES)
