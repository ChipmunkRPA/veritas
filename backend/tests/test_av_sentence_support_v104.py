from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction

import pytest

from scripts.av_sentence_support_v104 import RESERVED, assign_roles, clock_support


def records():
    rows = []
    for sentence, n in [("H00", 2), *[(f"S{i:02d}", n) for i, n in enumerate((8, 6, 4, 4, 3, 2, 2, 1))]]:
        for i in range(n):
            key = f"{sentence}-{i}"
            rows.append(
                {
                    "id": key,
                    "actor": "actor-" + key,
                    "ancestry": "ancestry-" + key,
                    "sentence": sentence,
                    "prior_role": "prior",
                }
            )
    return rows


def test_sentence_components_are_balanced_without_responses_and_order_invariant():
    rows = records()
    before = deepcopy(rows)
    result = assign_roles(rows, {"H00"})
    assert rows == before
    by_sentence = defaultdict(set)
    for row in result:
        by_sentence[row["sentence"]].add(row["study_role"])
        if row["sentence"] == "H00":
            assert row["study_role"] == "fusion_fit"
    assert all(len(v) == 1 for v in by_sentence.values())
    counts = Counter(r["study_role"] for r in result)
    assert [counts[role] for role in RESERVED] == [10, 10, 10]
    assert all(sum(role in v for v in by_sentence.values()) >= 2 for role in RESERVED)
    assert {r["id"]: r["study_role"] for r in assign_roles(rows[::-1], {"H00"})} == {
        r["id"]: r["study_role"] for r in result
    }


@pytest.mark.parametrize("key", ["actor", "ancestry"])
def test_actor_or_ancestry_link_joins_sentences_and_keeps_historical_connection_in_fitting(key):
    rows = records()
    source = next(r for r in rows if r["sentence"] == "H00")
    next(r for r in rows if r["sentence"] == "S00")[key] = source[key]
    result = assign_roles(rows, {"H00"})
    assert {r["study_role"] for r in result if r["sentence"] in {"H00", "S00"}} == {"fusion_fit"}
    for field in ("actor", "ancestry", "sentence"):
        groups = defaultdict(set)
        for row in result:
            groups[row[field]].add(row["study_role"])
        assert all(len(v) == 1 for v in groups.values())


@pytest.mark.parametrize("problem", ["duplicate", "unknown", "no_fitting", "too_few_contexts"])
def test_missing_or_insufficient_lineage_is_refused(problem):
    rows = records()
    history = {"H00"}
    if problem == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif problem == "unknown":
        rows[0]["actor"] = ""
    elif problem == "no_fitting":
        history = set()
    else:
        rows = [r for r in rows if r["sentence"] not in {"S00", "S01", "S02"}]
    with pytest.raises(ValueError):
        assign_roles(rows, history)


def clocks(n=30, origin=27):
    video = {
        "truncated": False,
        "interpolated": False,
        "time_base": "1/1000",
        "pts": [origin + 40 * i for i in range(n)],
    }
    audio = {
        "clock_qualified": True,
        "contiguous": True,
        "time_base": "1/1000",
        "sample_rate": 1000,
        "frames": [{"pts": 0, "start": "0", "end": "3/2", "samples": 1500}],
    }
    return video, audio


def test_exact_stream_offset_and_native_edit_context_preserved_without_zeroing_audio():
    video, audio = clocks()
    before = deepcopy((video, audio))
    result = clock_support(video, audio)
    assert (video, audio) == before
    assert result["audio_minus_video_origin_seconds"] == "-27/1000"
    assert result["audio_frame_mappings"][0]["audio_relative_seconds"] == "27/1000"
    assert result["native20_edit"]["frames_half_open"] == [5, 25]
    assert result["native20_edit"]["representatives"] == {"before": 4, "during": 14, "after": 25}
    assert result["native20_edit"]["audio_relative_first_edit_seconds"] == "227/1000"
    assert result["audio_conditioning_ready"] is False
    assert result["native_audio_samples_declared"] == 1500


def test_short_sources_stay_explicitly_unavailable_without_padding():
    video, audio = clocks(n=23)
    result = clock_support(video, audio)
    assert result["native20_edit_eligible"] is False and result["native20_edit"] is None
    assert len(result["common_frame_indices"]) == 23
    assert "Fewer than 24" in result["native20_unavailable_reason"]


def test_quantized_audio_frame_overlap_retained_without_false_sample_exact_phase():
    video, audio = clocks(origin=100)
    audio.update(
        sample_rate=10000,
        frames=[
            {"pts": 0, "start": "0", "end": "101/1000", "samples": 1010},
            {"pts": 100, "start": "1/10", "end": "1499/1000", "samples": 13990},
        ],
    )
    result = clock_support(video, audio)
    assert len(result["audio_frame_mappings"][0]["audio_frame_candidates"]) == 2
    assert result["missing_or_ambiguous_audio_frame_mappings"] >= 1
    assert result["maximum_cumulative_audio_frame_deviation_seconds"] == "1/1000"
    assert result["native20_edit_eligible"] and result["audio_conditioning_ready"] is False


def test_common_support_cannot_extend_past_conditional_native_sample_end():
    video, audio = clocks(n=2, origin=900)
    video["pts"] = [900, 1000]
    audio.update(
        sample_rate=10000,
        frames=[
            {"pts": 0, "start": "0", "end": "1/10", "samples": 1000},
            {"pts": 101, "start": "101/1000", "end": "1001/1000", "samples": 9000},
        ],
    )
    result = clock_support(video, audio)
    assert result["common_frame_indices"] == [0]
    assert result["conditional_sample_lattice_end_seconds"] == "1"


@pytest.mark.parametrize(
    "problem", ["missing_audio", "truncated", "unordered", "sample_mismatch", "clock_drift", "bad_rate"]
)
def test_invalid_or_unqualified_clocks_cannot_supply_support(problem):
    video, audio = clocks()
    if problem == "missing_audio":
        audio = None
    elif problem == "truncated":
        video["truncated"] = True
    elif problem == "unordered":
        video["pts"][2] = video["pts"][1]
    elif problem == "sample_mismatch":
        audio["frames"][0]["samples"] = 1499
    elif problem == "clock_drift":
        audio["frames"] = [
            {"pts": 0, "start": "0", "end": "1/10", "samples": 100},
            {"pts": 200, "start": "1/5", "end": "6/5", "samples": 1000},
        ]
    else:
        audio["sample_rate"] = True
    with pytest.raises(ValueError):
        clock_support(video, audio)


def test_every_conditioning_time_keeps_the_rational_source_relationship():
    video, audio = clocks()
    result = clock_support(video, audio)
    for row in result["audio_frame_mappings"]:
        assert Fraction(row["source_time_seconds"]) == Fraction(row["video_pts"], 1000)
        assert (
            Fraction(row["uniform_sample_lattice_offset_fraction"])
            == Fraction(row["audio_relative_seconds"]) * 1000
        )
