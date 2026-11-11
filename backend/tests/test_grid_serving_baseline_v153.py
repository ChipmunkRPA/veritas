import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import grid_serving_baseline_v153 as corrected


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


def test_exact_owned_environment_and_settings_singleton_reuse(tmp_path):
    result = child(
        """
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from pydantic_settings import DotEnvSettingsSource
from scripts import grid_serving_baseline_v153 as v

folder = Path(sys.argv[1])
assert "app.core.config" not in sys.modules
os.environ["VERITAS_STRIPE_SECRET_KEY"] = "authored sentinel"
os.environ["VERITAS_DATABASE_URL"] = "authored sentinel"
with patch.object(DotEnvSettingsSource, "_static_read_env_file", side_effect=AssertionError("dotenv read")):
    context = v.initialize(folder, "cpu")
from app.core.config import settings
assert context.confirm() is settings
assert "VERITAS_STRIPE_SECRET_KEY" not in os.environ and "VERITAS_DATABASE_URL" not in os.environ
wrong = {**context.env, "VERITAS_DATA_DIR": str(folder / "wrong")}
for name, action in (
    ("changed_env", lambda: context.use(wrong)),
    ("second_bootstrap", lambda: v.initialize(folder / "wrong", "cpu")),
):
    try:
        action()
    except ValueError:
        pass
    else:
        raise AssertionError(f"{name} accepted")
os.environ["VERITAS_DATA_DIR"] = wrong["VERITAS_DATA_DIR"]
try:
    context.confirm()
except ValueError:
    pass
else:
    raise AssertionError("mutated ambient environment accepted")
print(json.dumps({"offline": settings.offline, "mode": settings.billing_mode}))
""",
        tmp_path,
    )
    assert result == {"offline": True, "mode": "disabled"}
    assert not (tmp_path / "data").exists()


def test_refuses_ambient_application_settings_even_if_already_offline(tmp_path):
    result = child(
        """
import json
import sys
from pathlib import Path

from scripts import grid_serving_baseline_v153 as v

folder = Path(sys.argv[1])
env = v.baseline.controlled_environment("cpu", folder / "models", folder / "hf", folder / "data", folder)
existing = v.baseline.fresh_settings(env)
assert existing.offline is True
try:
    v.initialize(folder, "cpu")
except ValueError as error:
    assert "Fresh settings process required" in str(error)
else:
    raise AssertionError("existing app settings accepted")
print(json.dumps({"refused": True}))
""",
        tmp_path,
    )
    assert result == {"refused": True}


def test_worker_initializes_owned_settings_before_recursive_admission(monkeypatch, tmp_path):
    monkeypatch.setattr(corrected, "TRIAL", tmp_path / "trial")
    monkeypatch.setattr(corrected, "CACHE", tmp_path / "archive/cache")
    monkeypatch.setattr(corrected, "ARCHIVE", tmp_path / "archive")
    monkeypatch.setattr(corrected, "header", lambda: {"pilot_clip_ordinals": [0], "configured_device": "cpu"})
    stage = corrected.stage_folder(0)
    stage.mkdir(parents=True)
    (stage / "resources.jsonl").touch()
    (stage / "worker.log").touch()
    output = stage / "output"
    context, frozen, prior, order = object(), object(), object(), []

    def initialize(folder, device):
        assert folder == corrected.CACHE / "clip-00" and device == "cpu"
        assert not output.exists()
        order.append("isolated settings")
        return context

    def verify(active):
        assert active is context and order == ["isolated settings"]
        order.append("recursive verification")
        return frozen, prior

    def worker(active, destination, contract):
        assert active is context and destination == output and contract is frozen

        def execute(ordinal, path, baseline_contract):
            assert ordinal == 0 and path == output and baseline_contract is prior
            order.append("worker")

        return execute

    monkeypatch.setattr(corrected, "initialize", initialize)
    monkeypatch.setattr(corrected, "verify_initialized", verify)
    monkeypatch.setattr(corrected, "corrected_worker", worker)
    corrected.worker_entry(0, output)
    assert order == ["isolated settings", "recursive verification", "worker"]


def test_scoped_worker_preserves_old_globals_and_writes_distinct_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(corrected, "TRIAL", tmp_path / "trial")
    monkeypatch.setattr(corrected, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(corrected, "PUBLIC", tmp_path / "v153-freeze.json")
    corrected.PUBLIC.write_text("{}")
    context = SimpleNamespace(use=lambda env: env)
    frozen = {"baseline_freeze_sha256": "4" * 64}
    output = tmp_path / "output"
    output.mkdir()
    previous = corrected.baseline.assess.__globals__
    worker = corrected.corrected_worker(context, output, frozen)
    assert worker.__globals__ is not previous
    assert previous["TRIAL"] == corrected.baseline.TRIAL
    assert previous["PUBLIC"] == corrected.baseline.PUBLIC
    assert previous["fresh_settings"] is corrected.baseline.fresh_settings
    result = {
        "schema": "chipmunk.grid-serving-baseline-result/152",
        "freeze_sha256": corrected.sha(corrected.PUBLIC),
    }
    worker.__globals__["write"](output / "result.json", result)
    saved = corrected.read(output / "result.json")
    assert saved["schema"] == corrected.RESULT_SCHEMA and saved["settings_initialized_before_metadata"]
    assert saved["baseline_freeze_sha256"] == frozen["baseline_freeze_sha256"]


def test_metadata_reader_uses_clean_child_without_changing_caller(monkeypatch):
    monkeypatch.setenv("VERITAS_STRIPE_SECRET_KEY", "authored sentinel")
    env_before = dict(os.environ)
    app_before = sys.modules.get("app.core.config")

    def run(args, **kwargs):
        assert args[-3:] == ["_metadata", "load", "0"]
        assert "VERITAS_STRIPE_SECRET_KEY" not in kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="[]")

    monkeypatch.setattr(corrected.subprocess, "run", run)
    assert corrected.load_completed(0) == []
    assert dict(os.environ) == env_before and sys.modules.get("app.core.config") is app_before


def test_actual_case_metadata_reuses_preconfigured_offline_settings(tmp_path):
    clip = corrected.baseline.clips.TRIAL / "clip-00/bind-00"
    if not all(
        path.is_file()
        for path in (
            corrected.baseline.PUBLIC,
            clip / "output/result.json",
            clip / "profile.json",
        )
    ):
        pytest.skip(
            "Actual completed v149 pilot metadata is available only in its local evaluation workspace"
        )
    observed = child(
        """
import json
import sys
from pathlib import Path
from unittest.mock import patch

from pydantic_settings import DotEnvSettingsSource
from scripts import grid_serving_baseline_v153 as v

folder = Path(sys.argv[1])
assert "app.core.config" not in sys.modules
with patch.object(DotEnvSettingsSource, "_static_read_env_file", side_effect=AssertionError("dotenv read")):
    context = v.initialize(folder, v.header()["configured_device"])
    from app.core.config import settings
    assert settings is context.settings
    saved = v.bound_baseline(context)
    rows, upstream = v.bind_case(context, 0)
    assert context.confirm() is settings
    assert settings.data_dir == folder / "data"
    assert "torch" not in sys.modules
assert len(rows) == saved["cases_per_clip"] == 11
assert all(row["status"] != "ready" or row["expected_serving_pcm_sha256"] for row in rows)
print(json.dumps({"rows": len(rows), "binding_fields": len(upstream), "offline": settings.offline}))
""",
        tmp_path,
    )
    assert observed == {"rows": 11, "binding_fields": 5, "offline": True}
    assert not (tmp_path / "data").exists()
