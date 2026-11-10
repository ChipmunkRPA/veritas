import numpy as np
import pytest

from scripts import mouth_articulation_core_v122 as core
from scripts.av_conditioning_clock_v109 import frame_plan


def test_native_unequal_steps_use_actual_middle_time_not_index_average():
    a = np.zeros(768)
    c = np.ones(768) * 3
    b = np.ones(768)
    actual = core.temporal_visual(a, b, c, ["0", "1/50", "3/50"])
    assert np.array_equal(actual[:768], b)
    assert np.max(actual[768:1536]) < 1e-12
    assert np.allclose(actual[1536:], 2)


@pytest.mark.parametrize("times", [[0, "1/25", "1/4"], [0, 0, "1/25"], [0, 0.04, 0.08]])
def test_unknown_or_gapped_context_cannot_be_filled(times):
    with pytest.raises(ValueError):
        core.temporal_visual(np.zeros(768), np.zeros(768), np.zeros(768), times)


def test_speech_summary_keeps_layer_order_and_full_support():
    hidden = np.zeros((1500, 5, 384), np.float32)
    for layer in range(5):
        hidden[:, layer] = layer + np.arange(1500)[:, None]
    plan = frame_plan(25, "1", "0", "0", 32000, "0")
    actual = core.speech_summary(hidden, plan)
    assert actual.shape == (3840,)
    assert np.all(actual[:384] == 50.5) and np.all(actual[384:768] == 51.5)
    assert np.allclose(actual[1920:], np.std(np.arange(10)))
    with pytest.raises(ValueError):
        core.speech_summary(hidden, frame_plan(0, "0", "0", "0", 32000, "0"))


def test_projection_is_weighted_centered_and_sign_canonical():
    x = np.array([[0.0, 0.0, 0.0], [1, 2, 0], [2, 0, 1], [4, 3, 2]])
    w = np.array([1.0, 1.0, 2.0, 4.0])
    spec = core.projection(x, w, rank=2)
    output = core.project(x, spec)
    assert np.allclose(np.average(output, axis=0, weights=w), 0, atol=1e-12)
    assert np.allclose((output * (w / w.sum())[:, None]).T @ output, np.eye(2), atol=1e-12)
    assert all(row[np.argmax(np.abs(row))] >= 0 for row in np.array(spec["components"]))


@pytest.mark.parametrize(
    "role,condition",
    [
        ("confidence_cal", "original"),
        ("selection", "original"),
        ("confirmation", "original"),
        ("fusion_fit", "ai_aligned"),
        ("fusion_fit", "blur2"),
    ],
)
def test_reference_fit_cannot_learn_from_reserved_or_edited_rows(role, condition):
    with pytest.raises(ValueError, match="untouched fitting"):
        core.fit_reference([{"study_role": role, "condition": condition}])


def test_interactions_use_shared_coordinates_and_have_no_probability_output():
    # Authored identity projections make the cross-coordinate algebra explicit.
    rank = core.RANK
    spec = {
        "method": core.METHOD,
        "rank": rank,
        "ridge": core.RIDGE,
        "response_kind": "features_only_not_probability",
        "visual_projection": {
            "center": [0.0] * rank,
            "components": np.eye(rank).tolist(),
            "scale": [1.0] * rank,
        },
        "audio_projection": {
            "center": [0.0] * rank,
            "components": np.eye(rank).tolist(),
            "scale": [1.0] * rank,
        },
        "audio_to_visual": np.eye(rank).tolist(),
    }
    v = np.ones((1, rank)) * 2
    a = np.ones((1, rank)) * 3
    result = core.representations(v, a, spec)
    assert set(result) == {"visual_only", "concatenation", "interaction"}
    assert np.all(result["interaction"][0, 2 * rank : 3 * rank] == 6)
    assert np.all(result["interaction"][0, 3 * rank :] == 1)


def test_genuine_only_reference_recovers_authored_shared_latent_relation():
    rng = np.random.default_rng(122)
    latent = rng.normal(size=(32, core.RANK))
    visual = latent @ rng.normal(size=(core.RANK, core.VISUAL_DIM))
    audio = latent @ rng.normal(size=(core.RANK, core.AUDIO_DIM))
    rows = [
        {
            "study_role": "fusion_fit",
            "condition": "original",
            "source_group": f"g{i // 16}",
            "visual": visual[i],
            "audio": audio[i],
        }
        for i in range(32)
    ]
    reference = core.fit_reference(rows)
    result = core.representations(visual, audio, reference)
    v, aligned = np.split(result["concatenation"], 2, axis=1)
    # Full-rank shared whitened coordinates give the closed-form ridge shrinkage.
    assert np.linalg.norm(v - aligned) / np.linalg.norm(v) == pytest.approx(
        core.RIDGE / (1 + core.RIDGE), abs=1e-10
    )
    assert reference["fit_sources"] == ["g0", "g1"] and reference["deployment_eligible"] is False
