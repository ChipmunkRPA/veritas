from scripts import audio_short_edit_sources_v169 as prior
from scripts import audio_short_edit_sources_v169b as plan


def test_shortlists_keep_cross_voice_text_out_of_both_fallback_sets():
    first = prior.candidate("stock", "shared-a", "a", "a long shared opening", {}, "test")
    second = prior.candidate("stock", "shared-b", "b", "a long shared opening", {}, "test")
    exclusive_a = prior.candidate("stock", "a-only", "a", "the first speaker's text", {}, "test")
    exclusive_b = prior.candidate("stock", "b-only", "b", "the second speaker's text", {}, "test")
    excluded = prior.candidate("stock", "claimed", "a", "already in earlier role", {}, "test")
    candidates = [first, second, exclusive_a, exclusive_b, excluded]
    groups = plan.stock_shortlists(candidates, set(excluded["texts"]))
    assert groups == plan.stock_shortlists(candidates[::-1], set(excluded["texts"]))
    assert len(groups) == 2
    assert sum(len(g["ordered_candidates"]) for g in groups) == 3
    assert {g["role"] for g in groups} == set(prior.ROLES)
    assert all(len({row["voice"] for row in g["ordered_candidates"]}) == 1 for g in groups)
    prior.validate_roles(groups, [])
