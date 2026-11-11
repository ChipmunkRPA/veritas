from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from scripts import video_frame_full_app_v171 as study


def test_reconciliation_preserves_the_whole_owned_environment():
    initial = {"VERITAS_BILLING_MODE": "disabled", "VERITAS_OFFLINE": "true"}
    additions = {"TOKENIZERS_PARALLELISM": "false", "TORCHINDUCTOR_CACHE_DIR": "/owned/tmp/cache"}
    expected = initial | additions
    assert study.bound_environment(initial, additions, expected) == expected
    with pytest.raises(ValueError, match="TOKENIZERS_PARALLELISM"):
        study.bound_environment(initial, additions, expected | {"TOKENIZERS_PARALLELISM": "true"})
    with pytest.raises(ValueError, match="VERITAS_BILLING_MODE"):
        study.bound_environment(initial, additions, expected | {"VERITAS_BILLING_MODE": "stripe"})
    with pytest.raises(ValueError, match="VERITAS_OFFLINE"):
        study.bound_environment(
            initial, additions, {key: value for key, value in expected.items() if key != "VERITAS_OFFLINE"}
        )
    with pytest.raises(ValueError, match="unapproved"):
        study.bound_environment(initial, additions, expected | {"unapproved": "changed"})
    with pytest.raises(ValueError, match="absent before"):
        study.bound_environment(expected, additions, expected)


def test_exact_dependency_cache_must_stay_in_the_owned_temp_dir(monkeypatch, tmp_path):
    cache = tmp_path / "tmp/torchinductor_actor"
    module = SimpleNamespace(default_cache_dir=lambda: str(cache))
    monkeypatch.setitem(study.sys.modules, "torch._inductor.runtime.cache_dir_utils", module)
    loader = tmp_path / "environment/cv2"
    monkeypatch.setattr(
        study.importlib.metadata, "distribution", lambda _: SimpleNamespace(locate_file=lambda _: loader)
    )
    assert study.expected_runtime_additions(tmp_path) == {
        **study.BASE_DEFAULTS,
        "LD_LIBRARY_PATH": f"{loader}/../../lib:",
        "TORCHINDUCTOR_CACHE_DIR": str(cache),
    }
    module.default_cache_dir = lambda: str(tmp_path / "other/torchinductor_actor")
    with pytest.raises(ValueError, match="owned arm temporary directory"):
        study.expected_runtime_additions(tmp_path)


def test_second_boundary_reuses_original_settings_checks_with_exact_defaults(monkeypatch, tmp_path):
    current = {"VERITAS_OFFLINE": "true"}
    original = dict(current)
    additions = {"KMP_INIT_AT_FORK": "FALSE"}
    settings = object()
    checks = []
    monkeypatch.setattr(study.os, "environ", current)
    monkeypatch.setattr(study, "expected_runtime_additions", lambda _: additions)

    @dataclass(frozen=True)
    class OriginalContext:
        initialized_env: dict[str, str]

        def confirm(self):
            checks.append(self.initialized_env)
            assert self.initialized_env == current
            return settings

    context = study.ConfirmRuntimeEnvironment(OriginalContext(original), tmp_path)
    assert context.confirm() is settings
    current.update(additions)
    assert context.confirm() is settings
    assert checks == [original, original | additions]
    current["KMP_INIT_AT_FORK"] = "TRUE"
    with pytest.raises(ValueError, match="KMP_INIT_AT_FORK"):
        context.confirm()
