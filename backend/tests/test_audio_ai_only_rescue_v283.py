import pytest

from scripts.audio_ai_only_rescue_v283 import DIRECTIONS, apply_ai_only


@pytest.mark.parametrize("mono", DIRECTIONS)
@pytest.mark.parametrize("selected", DIRECTIONS)
def test_only_ai_can_resolve_a_mono_abstention(mono, selected):
    expected = "ai" if selected == "ai" and mono in ("inconclusive", "not_assessed") else mono
    assert apply_ai_only(mono, selected) == expected


def test_requires_a_known_decision_even_when_mono_has_direction():
    with pytest.raises(ValueError, match="Unknown speech decision"):
        apply_ai_only("original", "failed")
