import pytest

from scripts import audio_common_prior_context_v87 as original
from scripts import audio_common_prior_study_v88 as successor
from scripts import local_probe_limits_v24 as neural
from scripts.audio_numeric_runtime_v88 import NumericBudget


def snapshot(ram=4 * 1024 ** 3, rss=600 * 1024 ** 2):
    return {"elapsed_s": 20.0, "tree_rss_bytes": rss, "ram_available_bytes": ram, "disk_free_bytes": 5 * 1024 ** 3}


def test_numeric_stage_uses_same_mathematics_and_leaves_neural_limits_unchanged():
    assert successor.calibrate is original.calibrate
    assert successor.predict is original.predict
    assert successor.METHOD is original.METHOD
    assert neural.START_RAM == neural.RSS_LIMIT == 6 * 1024 ** 3
    assert NumericBudget.START_RAM == 3 * 1024 ** 3
    assert NumericBudget.RSS_LIMIT == neural.RESERVE == 1536 * 1024 ** 2
    assert not NumericBudget.violations(snapshot(), initial=True)
    assert "initial_ram_headroom" in neural.violations(snapshot(), initial=True)


def test_numeric_budget_still_guards_rss_reserves_and_time():
    assert "initial_ram_headroom" in NumericBudget.violations(snapshot(ram=2 * 1024 ** 3), initial=True)
    assert "tree_rss_limit" in NumericBudget.violations(snapshot(rss=2 * 1024 ** 3))
    assert "ram_reserve" in NumericBudget.violations(snapshot(ram=1024 ** 3))
    data = snapshot()
    data["elapsed_s"] = 901.0
    assert "elapsed_time_limit" in NumericBudget.violations(data)
    data["disk_free_bytes"] = 1024 ** 3
    assert "disk_reserve" in NumericBudget.violations(data)


def test_invalid_observations_cannot_pass_new_numerical_budget():
    for field, value in (("elapsed_s", float("nan")), ("ram_available_bytes", -1), ("tree_rss_bytes", True)):
        data = snapshot()
        data[field] = value
        with pytest.raises(ValueError, match="Complete finite"):
            NumericBudget.violations(data)


def test_actual_study_source_and_protocol_bindings_all_resolve():
    bindings = successor.study_bindings()
    assert "docs/audio-common-reference-v87-protocol.md" in bindings
    assert "docs/audio-numeric-runtime-v88-protocol.md" in bindings
    assert {"backend/scripts/audio_common_prior_study_v88.py", "backend/scripts/audio_numeric_runtime_v88.py",
            "backend/scripts/audio_parallel_dispatch_v88.py", "backend/tests/test_audio_numeric_runtime_v88.py"} <= bindings.keys()
    assert all(len(value) == 64 for value in bindings.values())
