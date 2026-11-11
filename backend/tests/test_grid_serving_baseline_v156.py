import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import grid_serving_baseline_v156 as corrected


def child(script, folder):
    env = {
        key: os.environ[key] for key in ("HOME", "PATH", "LANG", "LC_ALL", "LC_CTYPE") if key in os.environ
    }
    env["PYTHONPATH"] = str(corrected.ROOT / "backend")
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(folder)],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_only_numba_disk_cache_changes_for_the_same_owned_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "inherited"))
    monkeypatch.setenv("LIBROSA_CACHE_DIR", str(tmp_path / "inherited"))
    args = ("cpu", tmp_path / "models", tmp_path / "hf", tmp_path / "data", tmp_path)
    old = corrected.baseline.controlled_environment(*args)
    new = corrected.controlled_environment(*args)
    assert new == {**old, "NUMBA_CACHE_DIR": str(tmp_path / "cache/numba")}
    assert "LIBROSA_CACHE_DIR" not in new


def test_auxiliary_cache_must_be_set_before_first_import(tmp_path):
    result = child(
        """
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from scripts import grid_serving_baseline_v156 as v
folder = Path(sys.argv[1])
sys.modules["numba"] = ModuleType("numba")
try:
    v.initialize(folder, "cpu")
except ValueError as exc:
    assert "Fresh auxiliary imports required" in str(exc)
else:
    raise AssertionError("Late cache override accepted")
del sys.modules["numba"]
context = v.initialize(folder, "cpu")
assert "numba" not in sys.modules and "librosa" not in sys.modules
assert context.use(v.controlled_environment("cpu", folder/"models", folder/"hf", folder/"data", folder)) is context.settings
os.environ["NUMBA_CACHE_DIR"] = str(folder / "wrong")
try:
    context.confirm()
except ValueError:
    pass
else:
    raise AssertionError("Unexpected external cache allowed")
print(json.dumps({"late_override_rejected": True}))
""",
        tmp_path,
    )
    assert result == {"late_override_rejected": True}


def test_real_sound_dependency_and_authored_numba_cache_under_private_guard(tmp_path):
    if sys.platform != "darwin":
        pytest.skip("System APFS is required for the owned auxiliary-cache smoke")
    proof = child(
        """
import json
import sys
from pathlib import Path
from scripts import grid_serving_baseline_v156 as v
folder = Path(sys.argv[1])
v.WORK, v.TRIAL = folder / "work", folder / "trial"
result = v.cache_smoke("cpu")
assert not list(v.WORK.iterdir())
print(json.dumps(result))
""",
        tmp_path,
    )
    assert proof["filesystem"] == "apfs" and proof["unlearned_filter_shape"] == [513, 64]
    assert proof["production_dependency_import_ok"] and proof["authored_cache_index_and_code_present"]
    assert proof["model_loads"] == proof["learned_forwards"] == 0
    assert proof["dependencies"] == corrected.dependency_identity()


def test_actual_partial_coverage_history_binds_prior_artifacts():
    prior = corrected.apfs.stage_folder(0)
    if not (prior / "output/result.json").is_file():
        pytest.skip("Actual v154 coverage receipts are local to the evaluation workspace")
    _, _, history = corrected.metadata("history")
    assert history["completed_analyses"] == history["sound_event_failures"] == 11
    assert history["v154_result_sha256"] == corrected.sha(prior / "output/result.json")
    assert history["v154_profile_sha256"] == corrected.sha(prior / "profile.json")
    assert history["v154_worker_log_sha256"] == corrected.sha(prior / "worker.log")


def test_worker_threads_the_same_cache_environment_into_base_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(corrected, "TRIAL", tmp_path / "trial")
    monkeypatch.setattr(corrected, "WORK", tmp_path / "work")
    monkeypatch.setattr(corrected, "PUBLIC", tmp_path / "freeze.json")
    monkeypatch.setattr(
        corrected.bootstrap, "header", lambda: {"pilot_clip_ordinals": [0], "configured_device": "cpu"}
    )
    corrected.PUBLIC.write_text("{}")
    stage = corrected.stage_folder(0)
    stage.mkdir(parents=True)
    (stage / "resources.jsonl").touch()
    (stage / "worker.log").touch()
    folder = corrected.WORK / "clip-00"
    calls = []

    def reuse(env):
        assert env["NUMBA_CACHE_DIR"] == str(folder / "cache/numba")
        calls.append("same environment")

    def initialize(active, device):
        assert active == folder and device == "cpu"
        calls.append("early settings")
        return SimpleNamespace(use=reuse)

    frozen = {
        "baseline_freeze_sha256": "baseline",
        "storage_smoke_sha256": "apfs",
        "numba_cache_smoke_sha256": "cache",
    }

    def verify(context):
        assert calls == ["early settings"]
        return frozen, {"configured_device": "cpu"}

    def authored_assess(ordinal, destination, saved):
        active = globals()["CACHE"] / f"clip-{ordinal:02d}"
        env = globals()["controlled_environment"](
            saved["configured_device"],
            active / "models",
            active / "hf",
            active / "data",
            active,
        )
        globals()["fresh_settings"](env)
        globals()["write"](
            destination / "result.json",
            {
                "schema": "chipmunk.grid-serving-baseline-result/152",
                "freeze_sha256": corrected.sha(corrected.PUBLIC),
            },
        )

    monkeypatch.setattr(corrected, "initialize", initialize)
    monkeypatch.setattr(corrected, "verify_initialized", verify)
    monkeypatch.setattr(corrected.baseline, "assess", authored_assess)
    corrected.worker_entry(0, stage / "output")
    result = corrected.read(stage / "output/result.json")
    assert result["schema"] == corrected.RESULT_SCHEMA
    assert result["numba_cache_smoke_sha256"] == "cache" and result["storage_smoke_sha256"] == "apfs"
    assert calls == ["early settings", "same environment"]
