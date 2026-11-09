import numpy as np
import pytest

from scripts.audio_matched_generation_v80 import native_save


def test_native_container_preserves_signed_zero_and_float32_samples(tmp_path):
    wave = np.array([0.0, -0.0, 0.12345679, -0.99999994, 1.1], dtype=np.float32)
    path = tmp_path / "source.npy"
    restored = native_save(path, wave)
    assert restored.tobytes() == wave.tobytes()
    assert np.signbit(restored[1])
    with pytest.raises(FileExistsError):
        native_save(path, wave)


@pytest.mark.parametrize("wave", [np.zeros(4, dtype=np.float64), np.zeros((2, 2), dtype=np.float32),
                                  np.array([np.nan], dtype=np.float32), np.array([np.inf], dtype=np.float32)])
def test_invalid_native_output_is_rejected_before_writing(tmp_path, wave):
    path = tmp_path / "source.npy"
    with pytest.raises(ValueError, match="finite native"):
        native_save(path, wave)
    assert not path.exists()
