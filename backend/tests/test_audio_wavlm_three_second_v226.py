"""Verify sample/feature geometry with authored data and no checkpoint load."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import audio_wavlm_three_second_v226 as study

KERNELS = (10, 3, 3, 3, 3, 2, 2)
STRIDES = (5, 2, 2, 2, 2, 2, 2)


def test_model_free_import_and_convolution_clock():
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from scripts import audio_wavlm_three_second_v226 as v; v.PILOT.require_model_free()",
        ],
        check=True,
    )
    assert study.convolution_clock(48000, KERNELS, STRIDES) == {
        "frames": 149,
        "stride_samples": 320,
        "receptive_field_samples": 400,
        "last_receptive_range": [47360, 47760],
    }
    assert study.convolution_clock(64000, KERNELS, STRIDES)["frames"] == 199


def test_selects_first_fully_admitted_triplet_in_original_order():
    prepared, observed = [], []
    for triplet in ("triplet-00", "triplet-01", "triplet-02"):
        for i, arm in enumerate(study.ARMS):
            case = {
                "case": f"{triplet}-{arm}",
                "triplet": triplet,
                "arm": arm,
                "condition": study.CONDITION,
                "status": "prepared",
                "samples": study.SAMPLES,
                "pcm_sha256": f"pcm-{i}",
                "noise_residual_sha256": f"noise-{triplet}",
            }
            assessed = (triplet, arm) != ("triplet-00", "alania")
            prepared.append(case)
            observed.append(
                {
                    **case,
                    "route_status": "ok" if assessed else "not_applicable",
                    "windows": [[0, study.SAMPLES]] if assessed else [],
                }
            )
    assert study.choose(prepared, observed) == prepared[3:6]
    with pytest.raises(ValueError, match="three-second"):
        study.choose(prepared, [*observed[:3], observed[3] | {"windows": [[0, 47999]]}, *observed[4:]])


def test_extract_uses_unchanged_pooling_and_no_padding_with_authored_states():
    import torch

    values = np.linspace(-0.25, 0.25, study.SAMPLES, dtype=np.float32)
    frames = torch.arange(study.FRAMES, dtype=torch.float32).view(1, -1, 1)
    states = tuple((frames + i).expand(1, study.FRAMES, study.WIDTH) for i in range(13))

    def authored(tensor, *, attention_mask, output_hidden_states, return_dict):
        assert tensor.shape == attention_mask.shape == (1, study.SAMPLES)
        assert np.array_equal(tensor.numpy()[0], values)
        assert torch.all(attention_mask == 1)
        assert output_hidden_states is return_dict is True
        return SimpleNamespace(hidden_states=states, last_hidden_state=states[-1])

    result = study.extract(values, authored)
    assert result["feature_frames"] == result["valid_feature_frames"] == 149
    pooled = np.asarray(result["wavlm"]).reshape(3, 2, 768)
    assert pooled[:, 0, 0].tolist() == [78, 82, 86]
    assert pooled[0, 1, 0] == pytest.approx(np.arange(149).std())
    assert result["input_pcm_sha256"] == study.float_sha(values)
    assert len(set(result["pooled_state_sha256"])) == 3
    with pytest.raises(ValueError, match="three-second"):
        study.extract(values[:-1], authored)
    with pytest.raises(ValueError, match="native 149-frame"):
        study.extract(
            values,
            lambda *_args, **_kwargs: SimpleNamespace(
                hidden_states=states, last_hidden_state=states[-1][:, :-1]
            ),
        )


def test_repeat_requires_feature_and_source_identity():
    first = {
        "arm": study.REPEAT_ARM,
        "repeat": False,
        "wavlm_sha256": "feature",
        "pooled_state_sha256": ["a", "b", "c"],
        "input_pcm_sha256": "pcm",
    }
    repeated = first | {"repeat": True}
    study.check_repeat([first, repeated])
    with pytest.raises(ValueError, match="reproduce"):
        study.check_repeat([first, repeated | {"pooled_state_sha256": ["b", "b", "c"]}])
