import pytest

from scripts.video_landmark_rerank_v259 import quality, rerank, summarize


def window(response, state, start, *, flagged=False):
    return flagged, {
        "response": response,
        "start_pts": start,
        "track_id": "face-track",
        "frames": (start, start + 5),
        "quality": state,
    }


def test_exact_sixty_percent_of_mean_flanks_is_included():
    assert quality(0.4, 0.3, 0.6) == "supported"
    assert quality(0.4, 0.299, 0.6) == "below_flank_fraction"
    assert quality(0.4, None, 0.6) == "geometry_unavailable"


def test_replacement_keeps_saved_rank_and_measures_known_overlap():
    rows = [
        window(0.9, "below_flank_fraction", 15, flagged=True),
        window(0.8, "geometry_unavailable", 20),
        window(0.7, "supported", 12),
    ]
    result = summarize([rerank(rows)], target=set(range(15, 21)))
    assert result["outcomes"] == {"lower_rank": 1}
    assert result["reranked_replacement_rank_drops"] == {"2": 1}
    assert result["reranked_flags_with_unavailable_geometry_at_higher_rank"] == 1
    assert result["known_edit_overlap"] == {
        "baseline_flags_overlapping_edit": 1,
        "reranked_flags_overlapping_edit": 1,
        "baseline_edited_frames_covered_across_renders": 6,
        "reranked_edited_frames_covered_across_renders": 3,
        "renders_with_same_edit_overlap": 0,
    }


@pytest.mark.parametrize(
    ("best_quality", "state"),
    [("below_flank_fraction", "no_supported_flag"), ("geometry_unavailable", "undetermined_geometry")],
)
def test_unavailable_is_distinct_and_point_remains_strict(best_quality, state):
    rows = [window(0.9, best_quality, 15, flagged=True), window(0.5, "supported", 16)]
    assert rerank(rows, point=0.5)["state"] == state


def test_rejects_different_retained_order():
    rows = [window(0.9, "supported", 16, flagged=True), window(0.9, "supported", 15)]
    with pytest.raises(ValueError, match="Exact saved appearance rank required"):
        rerank(rows)
