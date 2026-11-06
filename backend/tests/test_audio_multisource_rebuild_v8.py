import hashlib

import numpy as np
import pytest

from scripts.rebuild_audio_multisource_inputs_v8 import cached_source, pcm_hash, source_variants
from scripts.review_audio_multisource_plan_v8 import CONDITIONS


def test_offline_cache_refuses_path_traversal_and_changed_original(tmp_path):
    with pytest.raises(ValueError, match="basenames only"):
        cached_source(tmp_path, {"sha256": "../../outside"})
    original = b"known original reference"
    sha = hashlib.sha256(original).hexdigest()
    (tmp_path / (sha + ".audio")).write_bytes(b"changed recording")
    with pytest.raises(ValueError, match="original source bytes differ"):
        cached_source(tmp_path, {"sha256": sha})


def test_public_recipes_preserve_actual_signal_to_noise_levels():
    t = np.arange(64000) / 16000
    foreground = (0.2 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    background = (0.1 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    rows = [
        {
            "condition": condition,
            "common_peak_gain": 1.0,
            "background_sha256": "fixed",
            "background_start_sample": 0,
            "background_interval_pcm_sha256": pcm_hash(background),
            "snr_db": None if condition == "clean" else int(condition.rsplit("_", 1)[1]),
        }
        for condition in CONDITIONS
    ]
    values = source_variants(foreground, rows, {"fixed": background})
    signal = values["clean"].astype(float)
    assert np.sqrt(np.mean(signal**2)) == pytest.approx(0.1, abs=1e-8)
    for row in rows[1:]:
        noise = values[row["condition"]].astype(float) - signal
        measured = 20 * np.log10(np.sqrt(np.mean(signal**2)) / np.sqrt(np.mean(noise**2)))
        assert measured == pytest.approx(row["snr_db"], abs=1e-5)
    rows[0]["common_peak_gain"] = 0.5
    with pytest.raises(ValueError, match="common peak gain differs"):
        source_variants(foreground, rows, {"fixed": background})
