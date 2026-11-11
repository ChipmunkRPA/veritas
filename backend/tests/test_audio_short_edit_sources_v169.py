import pytest

from scripts import audio_short_edit_sources_v169 as plan


def row(lane, identity, voice, text, **fields):
    return plan.candidate(lane, identity, voice, text, {}, "test license", **fields)


def test_clone_donor_dialogue_and_prefix_connect_before_roles():
    first = row("clone", "a", "speaker-a", "put this first item in the blue basket now", donor="donor-a")
    second = row("clone", "b", "donor-a", "please bring that book before you go", context="dialogue-a")
    third = row("clone", "c", "speaker-c", "there is a small glass on the table", context="dialogue-a")
    fourth = row("clone", "d", "speaker-d", "put this first item in the blue basket tomorrow")
    separated = row("clone", "e", "speaker-e", "another statement will follow")
    groups = plan.connected([first, second, third, fourth, separated])
    assert {frozenset(record["id"] for record in group) for group in groups} == {
        frozenset("abcd"),
        frozenset("e"),
    }


def test_history_filters_exact_bytes_and_text_even_for_a_seen_stock_voice():
    source = row("stock", "generated", "provider-voice", "fixed script", raw_sha256="abc")
    assert not plan.eligible(source, {"voice:provider-voice"}, set(), set())
    assert plan.eligible(source, {"voice:provider-voice"}, set(), set(), seen_voice=True)
    assert not plan.eligible(source, {"bytes:abc"}, set(), set(), seen_voice=True)
    assert not plan.eligible(source, {"text:fixed script"}, set(), set(), seen_voice=True)


def test_cross_lane_text_conflict_is_removed_before_fixed_assignment():
    genuine = [
        row("genuine", "genuine-one", "first", "the book was on the table"),
        row("genuine", "genuine-two", "second", "my blue jacket is hanging outside"),
    ]
    stock = [
        row("stock", "stock-one", "stock-a", "the book was on the table"),
        row("stock", "stock-two", "stock-b", "it was the seventh window"),
    ]
    lanes = {"genuine": genuine, "stock": stock, "clone": [], "expressive": []}
    groups = plan.freeze_groups(lanes)
    reversed_groups = plan.freeze_groups({lane: values[::-1] for lane, values in lanes.items()})
    assert reversed_groups == groups
    assert {row["id"] for g in groups for row in g["ordered_candidates"]} == {
        "genuine-one",
        "genuine-two",
        "stock-two",
    }
    assert {g["role"] for g in groups if g["lane"] == "genuine"} == {"development", "holdout"}
    plan.validate_roles(groups, [])


def test_wrong_role_on_backup_or_noise_is_rejected():
    source = row("clone", "selected", "same speaker", "the same script")
    groups = [
        {"role": "development", "ordered_candidates": [source]},
        {"role": "holdout", "ordered_candidates": [{**source, "id": "backup"}]},
    ]
    with pytest.raises(ValueError, match="Cross-role source identity"):
        plan.validate_roles(groups, [])
    with pytest.raises(ValueError, match="Cross-role noise identity"):
        plan.validate_roles(
            [],
            [
                {"role": role, "raw_sha256": role, "recorded_pcm_sha256": "same sample identity"}
                for role in plan.ROLES
            ],
        )
