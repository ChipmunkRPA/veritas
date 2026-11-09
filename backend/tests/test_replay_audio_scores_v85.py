import copy
import hashlib
import math

import numpy as np
import pytest

from scripts.replay_audio_scores_v85 import reconstruct, recorded_vector


def authored():
    embedding = np.zeros(1024, np.float32)
    embedding[0] = 0.4
    wavlm = np.zeros(4608, np.float32)
    x = float(embedding[0])
    def sigmoid(z):
        return 1 / (1 + math.exp(-z))

    row = {"embedding": embedding.tolist(), "wavlm": wavlm.tolist(), "raw": sigmoid(x),
           "embedding_sha256": hashlib.sha256(embedding.astype("<f4").tobytes()).hexdigest(),
           "wavlm_sha256": hashlib.sha256(wavlm.astype("<f4").tobytes()).hexdigest(),
           "original_p": sigmoid(0.5 * x - 0.4), "candidate_p": sigmoid(x - 0.1),
           "candidates": {"standalone": {"p": sigmoid(-0.7)}, "complement": {"p": sigmoid(0.7 * x + 0.2)}}}
    direction = np.zeros(1024)
    direction[0] = 1
    old = {"transform": {"unit_head": direction.tolist(), "center": np.zeros(1024).tolist(),
                         "directions": np.eye(1024)[1:17].tolist(), "scale": np.ones(16).tolist()},
           "coefficients": [1.0, *([0.0] * 16), -0.1]}
    transform = {"center": np.zeros(4608).tolist(), "scale": np.ones(4608).tolist(), "variance_factor": 0.5}
    newer = {"standalone": {"transform": transform, "coefficients": [*([0.0] * 4608), -0.7]},
             "complement": {"transform": transform, "coefficients": [0.7, *([0.0] * 4608), 0.2]}}
    reference = {"head_weight": direction.tolist(), "head_bias": 0.0}
    return row, reference, old, newer, {"slope": 0.5, "intercept": -0.4}


def test_distinct_head_scores_are_reconstructed_from_authored_representations():
    values = authored()
    errors = reconstruct(*values)
    assert max(errors.values()) < 1e-12


def test_altered_scalar_probability_is_rejected_even_with_unchanged_features():
    row, *parameters = authored()
    for field in ("raw", "original_p", "candidate_p"):
        altered = copy.deepcopy(row)
        altered[field] = 0.9
        with pytest.raises(ValueError, match="differs from separate"):
            reconstruct(altered, *parameters)
    altered = copy.deepcopy(row)
    altered["candidates"]["complement"]["p"] = None
    with pytest.raises(ValueError, match="Actual finite recorded"):
        reconstruct(altered, *parameters)


def test_changed_feature_hash_and_non_float32_payload_cannot_pass_replay():
    row, *_ = authored()
    changed = copy.deepcopy(row)
    changed["embedding"][0] = float(np.float32(0.5))
    with pytest.raises(ValueError, match="hash differs"):
        recorded_vector(changed, "embedding", 1024)
    changed = copy.deepcopy(row)
    changed["embedding"][0] = 0.123456789012345
    with pytest.raises(ValueError, match="lossless float32"):
        recorded_vector(changed, "embedding", 1024)
