import copy

import pytest

from scripts.publish_audio_full_results_v11 import expand_probe, review
from scripts.review_audio_full_representation_v11 import reconstruct


def cached_probe():
    return {
        "raw": 0.25,
        "embedding": [0.1, 0.2],
        "embedding_sha256": "e" * 64,
        "filtered_pcm_sha256": "f" * 64,
        "p": 0.75,
    }


def test_probe_reference_reuses_vector_but_keeps_its_own_calibrated_response():
    cached = cached_probe()
    reference = {k: v for k, v in cached.items() if k != "embedding"}
    reference.update(status="computed", reused_exact_probe=True, p=0.1)
    result = expand_probe(reference, cached)
    assert result["embedding"] == cached["embedding"]
    assert result["p"] == 0.1 and cached["p"] == 0.75
    assert "embedding" not in reference


@pytest.mark.parametrize("changed", ["raw", "filtered_pcm_sha256", "embedding_sha256", "embedding"])
def test_cached_reference_cannot_substitute_a_different_measurement(changed):
    cached = cached_probe()
    reference = {k: v for k, v in cached.items() if k != "embedding"}
    reference.update(status="computed", reused_exact_probe=True)
    reference[changed] = 0.5 if changed == "raw" else [0.2, 0.3] if changed == "embedding" else "a" * 64
    with pytest.raises(ValueError, match="exact original cache"):
        expand_probe(reference, cached)


def test_uncomputed_probe_cannot_smuggle_a_measured_response():
    empty = {"status": "not_required", "raw": None, "embedding": None, "p": None}
    assert expand_probe(empty, None) == empty
    changed = copy.deepcopy(empty)
    changed["p"] = 0.1
    with pytest.raises(ValueError, match="uncomputed probe"):
        expand_probe(changed, None)


def test_private_provider_field_is_refused_before_reference_loading(tmp_path):
    with pytest.raises(ValueError, match="Private source fields"):
        review({"nested": [{"provider_id": "raw-saved-identity"}]}, tmp_path)


def test_numerical_review_cannot_drop_an_unassessed_development_case():
    with pytest.raises(ValueError, match="Every original development outcome"):
        reconstruct({}, [{"id": "unassessed", "role": "development"}], {}, {}, {}, [])
