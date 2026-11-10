import numpy as np
import pytest

from scripts import review_av_source_encoder_v115 as review
from scripts.av_conditioning_clock_v109 import frame_plan


def fixture(tmp_path):
    hidden = np.zeros((1500, 5, 384), np.float32)
    file = tmp_path / "state.npy"
    np.save(file, hidden, allow_pickle=False)
    plan = {
        "asset_index": 1,
        "source_group": "test",
        "study_role": "fusion_fit",
        "mono_wav_sha256": "wav",
        "frames": [frame_plan(0, "0", "0", "0", 1280, "0")],
    }
    row = {
        **{k: v for k, v in plan.items() if k != "frames"},
        "encoder_routing_qualified": True,
        "generator_conditioning_qualified": False,
        "hidden_file": str(file),
        "hidden_file_sha256": review.sha(file),
        "hidden_shape": [1500, 5, 384],
        "proof": {
            "layer_readout_matches": True,
            "full_hidden_sha256": review.source.speech.fingerprint(hidden),
        },
        **review.source.audit_chunks(hidden, plan["frames"]),
    }
    return row, plan, file


def test_saved_source_review_passes_without_inference(tmp_path):
    row, plan, file = fixture(tmp_path)
    assert review.review_source(row, plan, file) == {"routed_frames": 1, "explicit_zero_tick_slots": 6}


@pytest.mark.parametrize(
    "key,value",
    [
        ("study_role", "confirmation"),
        ("generator_conditioning_qualified", True),
        ("encoder_routing_qualified", False),
        ("hidden_file_sha256", "changed"),
        ("routed_frames", 2),
    ],
)
def test_changed_source_or_unsupported_claim_is_rejected(tmp_path, key, value):
    row, plan, file = fixture(tmp_path)
    row[key] = value
    with pytest.raises(ValueError):
        review.review_source(row, plan, file)


def test_changed_saved_array_is_rejected_even_with_updated_file_hash(tmp_path):
    row, plan, file = fixture(tmp_path)
    value = np.load(file, allow_pickle=False)
    value[0, 0, 0] = 1
    np.save(file, value, allow_pickle=False)
    row["hidden_file_sha256"] = review.sha(file)
    with pytest.raises(ValueError, match="saved native states"):
        review.review_source(row, plan, file)
