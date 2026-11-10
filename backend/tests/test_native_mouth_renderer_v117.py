import copy

import numpy as np
import pytest

from scripts import native_mouth_renderer_v117 as renderer
from scripts.av_conditioning_clock_v109 import frame_plan


def test_earlier_audio_uses_exact_time_and_keeps_original_picture_index():
    frame = frame_plan(35, "7/6", "-13/500", "0", 64000, "1/500")
    before = copy.deepcopy(frame)
    earlier = renderer.earlier_plan(frame)
    assert earlier["encoder_anchor"] == frame["encoder_anchor"] - 16
    assert earlier["video_frame_index"] == 35
    assert earlier["source_audio_origin_seconds"] == frame["source_audio_origin_seconds"]
    assert earlier["anchor_residual_seconds"] == frame["anchor_residual_seconds"]
    assert frame == before


def test_insufficient_earlier_audio_is_not_padded_or_substituted():
    frame = frame_plan(1, "1/25", "0", "0", 64000, "0")
    with pytest.raises(ValueError, match="320-ms"):
        renderer.earlier_plan(frame)


def images():
    original = np.zeros((256, 256, 3), np.uint8)
    earlier = original.copy()
    earlier[10, 10, 0] = 1
    zero = original.copy()
    zero[20:22, 20:22, 2] = 1
    return [original, original.copy(), earlier, zero]


def test_response_counts_distinguish_seeded_repeat_and_conditioning_changes():
    assert renderer.response_counts(images()) == {"earlier_320ms": 1, "zero": 4}


@pytest.mark.parametrize(
    "problem", ["repeat_changed", "earlier_no_effect", "zero_no_effect", "missing", "dtype", "shape"]
)
def test_incomplete_or_nonresponsive_renderer_is_not_qualified(problem):
    values = images()
    if problem == "repeat_changed":
        values[1][0, 0, 0] = 1
    elif problem == "earlier_no_effect":
        values[2] = values[0].copy()
    elif problem == "zero_no_effect":
        values[3] = values[0].copy()
    elif problem == "missing":
        values.pop()
    elif problem == "dtype":
        values[2] = values[2].astype(np.float32)
    else:
        values[2] = values[2][:-1]
    with pytest.raises(ValueError):
        renderer.response_counts(values)


def test_worker_routing_accepts_only_fixed_qualification_command(monkeypatch):
    args = ["python", "-B", "-m", "scripts.video_appearance_native_v108", "authored", "private"]
    calls = []
    monkeypatch.setattr(renderer.subprocess, "Popen", lambda argv, **kw: calls.append((argv, kw)))
    renderer.routed_popen(args, cwd="backend")
    assert args[3] == "scripts.video_appearance_native_v108"
    assert calls == [([*args[:3], "scripts.native_mouth_renderer_v117", *args[4:]], {"cwd": "backend"})]
    with pytest.raises(ValueError):
        renderer.routed_popen([*args[:4], "extract", *args[5:]])
