"""Initialize isolated app settings before admitting the frozen serving inputs."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts import grid_serving_baseline_v152 as baseline
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL, ARCHIVE = baseline.ROOT, baseline.BASE, baseline.PANEL, baseline.ARCHIVE
TRIAL = BASE / "grid-serving-baseline-v153"
CACHE = ARCHIVE / "grid-serving-baseline-v153"
PUBLIC = PANEL / "grid-serving-baseline-freeze-v153.json"
CODE = (
    "backend/scripts/grid_serving_baseline_v153.py",
    "backend/tests/test_grid_serving_baseline_v153.py",
    "docs/grid-serving-baseline-v153-method.md",
    "backend/scripts/grid_serving_baseline_v152.py",
)
FREEZE_SCHEMA = "chipmunk.grid-serving-baseline-freeze/153"
RESULT_SCHEMA = "chipmunk.grid-serving-baseline-result/153"
APP_ENV_DEFAULTS = {
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
}
read, write, sha, require = baseline.read, baseline.write, baseline.sha, baseline.require


@dataclass(frozen=True)
class IsolatedSettings:
    env: dict[str, str]
    initialized_env: dict[str, str]
    settings: Any

    def use(self, env):
        require(
            env == self.env
            and self.initialized_env == dict(os.environ)
            and getattr(sys.modules.get("app.core.config"), "settings", None) is self.settings
            and self.settings.offline is True
            and self.settings.run_worker is self.settings.webhooks_enabled is False
            and self.settings.billing_mode == "disabled"
            and self.settings.api_key is self.settings.database_url is None
            and self.settings.stripe_secret_key is self.settings.stripe_webhook_secret is None,
            "Serving metadata and pipeline must use the same owned offline settings",
        )
        require(
            self.settings.data_dir == Path(env["VERITAS_DATA_DIR"])
            and self.settings.models_dir == Path(env["VERITAS_MODELS_DIR"])
            and self.settings.hf_home == Path(env["VERITAS_HF_HOME"])
            and self.settings.device == env["VERITAS_DEVICE"],
            "Serving settings must retain the owned data, model and cache paths",
        )
        return self.settings

    def confirm(self):
        return self.use(self.env)


def initialize(folder, device):
    """Process entry only. Refuse an existing application settings singleton."""
    env = baseline.controlled_environment(device, folder / "models", folder / "hf", folder / "data", folder)
    settings = baseline.fresh_settings(env)
    initialized = dict(os.environ)
    require(
        initialized.keys() - env.keys() <= APP_ENV_DEFAULTS
        and all(initialized.get(key) == value for key, value in env.items()),
        "Only frozen application runtime defaults may extend the owned environment",
    )
    context = IsolatedSettings(env, initialized, settings)
    context.confirm()
    return context


def header():
    """Read the fixed nonsecret device and roster without traversing any producer context."""
    saved = read(baseline.PUBLIC)
    require(
        saved["schema"] == "chipmunk.grid-serving-baseline-freeze/152"
        and saved["configured_device"] in baseline.DEVICES
        and saved["profile"] == baseline.PROFILE,
        "Exact historical full-profile baseline context required",
    )
    return saved


def require_owned(folder):
    require(
        folder.parent == CACHE
        and folder.resolve().is_relative_to(CACHE.resolve())
        and CACHE.resolve().is_relative_to(ARCHIVE.resolve()),
        "Stage cache must be owned and remain on the external archive",
    )


def bound_baseline(context):
    context.confirm()
    saved = baseline.verify()
    context.confirm()
    require(not list(baseline.TRIAL.glob("clip-*")), "The frozen v152 baseline must remain unattempted")
    return saved


def identity(saved):
    return {
        "schema": FREEZE_SCHEMA,
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "baseline_freeze_sha256": sha(baseline.PUBLIC),
        "baseline_private_freeze_sha256": sha(baseline.TRIAL / "freeze.json"),
        "baseline_asset_inventory_sha256": saved["asset_inventory_sha256"],
        "configured_device": saved["configured_device"],
        "profile": saved["profile"],
        "pilot_clip_ordinals": saved["pilot_clip_ordinals"],
        "cases_per_clip": saved["cases_per_clip"],
        "initial_policy": saved["initial_policy"],
        "settings_bootstrap": "Initialize exclusive offline settings with dotenv disabled before recursive metadata.",
        "scope": "Same frozen serving pipeline, commercial models and resource limits as v152. Separate execution context.",
    }


def verify_initialized(context):
    saved = bound_baseline(context)
    frozen = identity(saved)
    require(frozen == read(PUBLIC) == read(TRIAL / "freeze.json"), "Exact v153 context changed")
    return frozen, saved


def bind_case(context, ordinal):
    context.confirm()
    result = baseline.case_binding(ordinal)
    context.confirm()
    return result


def stage_folder(ordinal):
    return TRIAL / f"clip-{ordinal:02d}/authored"


def read_cases(context, ordinal, frozen):
    require(type(ordinal) is int and ordinal in frozen["pilot_clip_ordinals"], "Fixed pilot ordinal required")
    cases, upstream = bind_case(context, ordinal)
    stage = stage_folder(ordinal)
    result = baseline.speech.qualification_profile(stage, None, {}, "grid_serving_baseline_v153.py")
    folder = CACHE / f"clip-{ordinal:02d}"
    require_owned(folder)
    artifacts = {
        "cases.jsonl": "private_cases_sha256",
        "planned-cases.json": "private_planned_cases_sha256",
        "runtime.json": "private_runtime_sha256",
        "upstream.json": "private_upstream_sha256",
    }
    require(
        result["schema"] == RESULT_SCHEMA
        and result["clip_ordinal"] == ordinal
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["baseline_freeze_sha256"] == frozen["baseline_freeze_sha256"]
        and result["settings_initialized_before_metadata"] is True
        and result["profile"] == frozen["profile"]
        and result["configured_device"] == frozen["configured_device"]
        and result["defaults_changed"]
        is result["candidate_added"]
        is result["billing_or_service_started"]
        is False
        and all(result[key] == sha(folder / filename) for filename, key in artifacts.items())
        and read(folder / "planned-cases.json") == baseline.planned_cases(cases)
        and read(folder / "upstream.json") == upstream
        and read(folder / "runtime.json")["resolved_device"] == result["resolved_device"],
        "Actual successful v153 receipt and exact private journals required",
    )
    rows = baseline.planning.reviewed.lines(folder / "cases.jsonl")
    registry = read(baseline.TRIAL / "private-inventory.json")["registry"]
    baseline.validate_case_rows(rows, cases, registry)
    require(
        all(result[key] == value for key, value in baseline.count_results(rows).items()),
        "Serving counts must reconcile with every matched case",
    )
    return rows


def metadata(action, ordinal=None):
    """Use a clean child so admission cannot import app settings in the caller."""
    args = [sys.executable, "-B", "-m", "scripts.grid_serving_baseline_v153", "_metadata", action]
    if ordinal is not None:
        args.append(str(ordinal))
    env = {
        key: os.environ[key] for key in ("HOME", "PATH", "LANG", "LC_ALL", "LC_CTYPE") if key in os.environ
    }
    env["PYTHONPATH"] = str(ROOT / "backend")
    response = subprocess.run(
        args, cwd=ROOT / "backend", env=env, capture_output=True, text=True, check=False
    )
    if response.returncode:
        raise RuntimeError(response.stderr.strip() or "Isolated serving metadata failed")
    return json.loads(response.stdout)


def load_completed(ordinal):
    """Return actual completed private serving observations in blind numeric order."""
    require(type(ordinal) is int, "Fixed pilot ordinal required")
    return metadata("load", ordinal)


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every prior context or attempt")
    frozen = metadata("identity")
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    print(json.dumps({"status": "frozen", "freeze_sha256": sha(PUBLIC)}), flush=True)


def prerequisites(ordinal, ordinals):
    preceding = ordinals.index(ordinal) - 1
    source = baseline.clips.TRIAL / f"clip-{ordinal:02d}/bind-00"
    stages = [source]
    if preceding >= 0:
        stages.append(stage_folder(ordinals[preceding]))
    if not all(
        (stage / name).is_file() for stage in stages for name in ("profile.json", "output/result.json")
    ):
        return False
    return metadata("prerequisite", ordinal)


def metadata_entry(action, ordinal):
    with contextlib.redirect_stdout(sys.stderr):
        folder = CACHE / "metadata-only"
        require_owned(folder)
        context = initialize(folder, header()["configured_device"])
        if action == "identity":
            return identity(bound_baseline(context))
        frozen, _ = verify_initialized(context)
        if action == "verify":
            return frozen
        ordinals = frozen["pilot_clip_ordinals"]
        require(type(ordinal) is int and ordinal in ordinals, "Fixed pilot ordinal required")
        if action == "load":
            return read_cases(context, ordinal, frozen)
        require(action == "prerequisite", "Exact metadata command required")
        before = ordinals.index(ordinal) - 1
        if before >= 0:
            read_cases(context, ordinals[before], frozen)
        bind_case(context, ordinal)
        return True


def corrected_worker(context, output, frozen):
    """Bind the prior worker's existing inference to the corrected process context."""
    inventory = TRIAL / "private-inventory.json"

    def read_bound(path):
        return read(baseline.TRIAL / path.name if path == inventory else path)

    def write_receipt(path, result):
        if path == output / "result.json":
            require(
                result["schema"] == "chipmunk.grid-serving-baseline-result/152"
                and result["freeze_sha256"] == sha(PUBLIC),
                "Expected result from the bound baseline worker",
            )
            result.update(
                schema=RESULT_SCHEMA,
                baseline_freeze_sha256=frozen["baseline_freeze_sha256"],
                settings_initialized_before_metadata=True,
            )
        write(path, result)

    return scoped(
        baseline.assess,
        {
            "TRIAL": TRIAL,
            "CACHE": CACHE,
            "PUBLIC": PUBLIC,
            "read": read_bound,
            "write": write_receipt,
            "case_binding": lambda ordinal: bind_case(context, ordinal),
            "fresh_settings": context.use,
        },
    )


def worker_entry(ordinal, output):
    saved = header()
    stage = stage_folder(ordinal)
    require(
        ordinal in saved["pilot_clip_ordinals"]
        and output.resolve() == (stage / "output").resolve()
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and not (stage / "profile.json").exists(),
        "Exclusive guarded v153 output required",
    )
    folder = CACHE / f"clip-{ordinal:02d}"
    require_owned(folder)
    context = initialize(folder, saved["configured_device"])
    frozen, prior = verify_initialized(context)
    output.mkdir(exist_ok=False)
    corrected_worker(context, output, frozen)(ordinal, output, prior)


def routed_popen(ordinal, argv, **kwargs):
    output = stage_folder(ordinal) / "output"
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored"]
        and argv[5:] == [str(output)],
        "Only the exclusive v153 full-profile worker is permitted",
    )
    return subprocess.Popen(
        [*argv[:3], "scripts.grid_serving_baseline_v153", "_worker", str(ordinal), str(output)], **kwargs
    )


def run(ordinal):
    from scripts import video_appearance_runtime_v108 as runtime

    frozen = metadata("verify")
    ordinals = frozen["pilot_clip_ordinals"]
    require(type(ordinal) is int and ordinal in ordinals, "Fixed pilot ordinal required")
    folder = stage_folder(ordinal).parent
    folder.mkdir(exist_ok=True)
    proxy = types.SimpleNamespace(
        Popen=lambda argv, **kwargs: routed_popen(ordinal, argv, **kwargs),
        STDOUT=subprocess.STDOUT,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    scoped(
        runtime.run,
        {
            "TRIAL": folder,
            "verify": lambda: metadata("verify"),
            "prerequisites": lambda _c, _b: prerequisites(ordinal, ordinals),
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("authored")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    commands.add_parser("run").add_argument("ordinal", type=int)
    worker = commands.add_parser("_worker")
    worker.add_argument("ordinal", type=int)
    worker.add_argument("output", type=Path)
    reader = commands.add_parser("_metadata")
    reader.add_argument("action", choices=("identity", "verify", "load", "prerequisite"))
    reader.add_argument("ordinal", type=int, nargs="?")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "run":
        run(args.ordinal)
    elif args.command == "_worker":
        worker_entry(args.ordinal, args.output)
    else:
        print(json.dumps(metadata_entry(args.action, args.ordinal), allow_nan=False))


if __name__ == "__main__":
    main()
