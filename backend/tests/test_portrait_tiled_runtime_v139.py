import numpy as np
import pytest

from scripts import portrait_native_qualifier_v133 as original
from scripts import portrait_tiled_runtime_v139 as runtime


def test_partial_reference_tolerance_is_predeclared_not_invented_from_the_new_response():
    reference = np.zeros((512, 512, 3), np.uint8)
    same = runtime.compare_images(reference.copy(), reference)
    assert same["within_frozen_tolerance"]
    tiny = reference.copy()
    tiny[10, 10, 0] = 1
    assert runtime.compare_images(tiny, reference)["within_frozen_tolerance"]
    tiny[10, 10, 0] = 2
    assert not runtime.compare_images(tiny, reference)["within_frozen_tolerance"]
    assert not runtime.compare_images(np.ones_like(reference), reference)["within_frozen_tolerance"]
    with pytest.raises(ValueError):
        runtime.compare_images(reference[:-1], reference)


def test_new_context_preserves_original_run_code_and_dispatches_only_qualification(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda argv, **kwargs: calls.append(argv))
    scope = runtime.context()
    assert scope["run"].__code__ is original.run.__code__
    assert scope["TRIAL"] != original.TRIAL and scope["CACHE"] != original.CACHE
    scope["routed_popen"](["python", "-B", "-m", "scripts.video_appearance_native_v108", "authored", "out"])
    assert calls[0][3] == "scripts.portrait_tiled_runtime_v139"
    with pytest.raises(ValueError):
        scope["routed_popen"](
            ["python", "-B", "-m", "scripts.video_appearance_native_v108", "bind", "out", "0"]
        )
