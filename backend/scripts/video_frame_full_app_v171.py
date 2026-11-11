"""Run the unchanged v168 arms with a verified dependency environment at inference entry."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import tempfile
import types
from dataclasses import replace
from pathlib import Path

from scripts import video_frame_full_app_v168 as prior
from scripts.video_appearance_corrected_v112 import scoped

TRIAL = prior.BASE / "video-frame-full-app-v171"
WORK = prior.BASE / "video-frame-full-app-work-v171"
MODULE = "scripts.video_frame_full_app_v171"
FILES = ("inputs.json", *(f"{arm}.yaml" for arm in prior.ARMS))
BASE_DEFAULTS = {
    "KMP_DUPLICATE_LIB_OK": "True",
    "KMP_INIT_AT_FORK": "FALSE",
    "TOKENIZERS_PARALLELISM": "false",
}
DEPENDENCY_SOURCES = {
    "opencv-python-headless": ("cv2/__init__.py", "cv2/config.py"),
    "torch": ("torch/_inductor/runtime/cache_dir_utils.py",),
    "scikit-learn": ("sklearn/__init__.py",),
    "open-clip-torch": ("open_clip/tokenizer.py",),
}
read, write, sha, require = prior.read, prior.write, prior.sha, prior.require
controlled_environment = prior.serving.controlled_environment


def dependency_identity():
    records = {}
    for name, files in DEPENDENCY_SOURCES.items():
        package = importlib.metadata.distribution(name)
        records[name] = {
            "version": package.version,
            "source_sha256": {file: sha(package.locate_file(file)) for file in files},
        }
    return records


def expected_runtime_additions(folder):
    cache_module = sys.modules.get("torch._inductor.runtime.cache_dir_utils")
    require(cache_module is not None, "The expected cache dependency must already have initialized")
    loader = importlib.metadata.distribution("opencv-python-headless").locate_file("cv2")
    cache = Path(cache_module.default_cache_dir())
    require(
        cache.parent == folder / "tmp" and cache.name.startswith("torchinductor_"),
        "The dependency cache must use the exact owned arm temporary directory",
    )
    return {**BASE_DEFAULTS, "LD_LIBRARY_PATH": f"{loader}/../../lib:", "TORCHINDUCTOR_CACHE_DIR": str(cache)}


def bound_environment(initial, additions, observed):
    require(initial.keys().isdisjoint(additions), "Dependency defaults must be absent before initialization")
    expected = initial | additions
    differing = sorted(
        key for key in expected.keys() | observed.keys() if expected.get(key) != observed.get(key)
    )
    require(not differing, f"Full-app inference environment differs at keys: {', '.join(differing)}")
    return expected


class ConfirmRuntimeEnvironment:
    """Keep the original first check and repeat it with exact observed defaults at inference entry."""

    def __init__(self, context, folder):
        self.context = context
        self.folder = folder
        self.initial_confirmed = False

    def confirm(self):
        if not self.initial_confirmed:
            settings = self.context.confirm()
            self.initial_confirmed = True
            return settings
        expected = bound_environment(
            self.context.initialized_env, expected_runtime_additions(self.folder), dict(os.environ)
        )
        return replace(self.context, initialized_env=expected).confirm()


def initialize(folder, device):
    context = scoped(prior.serving.initialize, {"controlled_environment": controlled_environment})(
        folder, device
    )
    return ConfirmRuntimeEnvironment(context, folder)


def original_failure():
    stage = prior.TRIAL / "reference/authored"
    profile_path = stage / "profile.json"
    profile = read(profile_path)
    log, journal = stage / "worker.log", stage / "resources.jsonl"
    observed = prior.WORK / "reference/runtime.json"
    trace = log.read_text()
    require(
        profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["command"] == "authored"
        and profile["child_started"] is True
        and profile["child_exit_code"] == 1
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and not profile["resource_limit_observations"]
        and profile["runtime_sha256"] == sha(Path(prior.__file__))
        and profile["resource_journal_sha256"] == sha(journal)
        and profile["worker_receipt_sha256"] is None
        and "receipt, registry, manager, context.confirm()" in trace
        and "Serving metadata and pipeline must use the same owned offline settings" in trace
        and observed.is_file()
        and not (stage / "output/result.json").exists()
        and not (prior.WORK / "reference/case.json").exists()
        and not (prior.TRIAL / "batch-one/authored").exists(),
        "Preserved v168 pre-inference failure required",
    )
    return {
        "profile_sha256": sha(profile_path),
        "worker_log_sha256": sha(log),
        "runtime_sha256": sha(observed),
    }


def identity(previous):
    files = (
        Path(__file__),
        prior.ROOT / "backend/tests/test_video_frame_full_app_v171.py",
        prior.ROOT / "docs/video-frame-full-app-v171-method.md",
    )
    return {
        **previous,
        "schema": "chipmunk.video-frame-full-app-freeze/171",
        "v168_freeze_sha256": sha(prior.TRIAL / "freeze.json"),
        "preserved_failure": original_failure(),
        "correction_code_sha256": {file.name: sha(file) for file in files},
        "dependency_sources": dependency_identity(),
        "environment_correction": "Compare every inherited key and the five exact dependency defaults before inference.",
    }


def freeze():
    require(not TRIAL.exists() and not WORK.exists(), "Preserve all prior or incomplete v171 attempts")
    previous, _, _ = prior.verify()
    frozen = identity(previous)
    TRIAL.mkdir(mode=0o700, exist_ok=False)
    for name in FILES:
        (TRIAL / name).write_bytes((prior.TRIAL / name).read_bytes())
    write(TRIAL / "freeze.json", frozen)
    print(
        json.dumps(
            {"status": "frozen", "freeze_sha256": sha(TRIAL / "freeze.json"), "arms": list(prior.ARMS)}
        )
    )


def verify():
    previous, file, receipt = prior.verify()
    frozen = identity(previous)
    require(
        read(TRIAL / "freeze.json") == frozen
        and all((TRIAL / name).read_bytes() == (prior.TRIAL / name).read_bytes() for name in FILES),
        "Exact corrected freeze and unchanged v168 arm inputs required",
    )
    return frozen, file, receipt


def preflight(arm):
    """Exercise both confirmations around the real registry and runtime without acquiring a model lease."""
    prior.verify()
    old = read(prior.baseline.PUBLIC)
    with tempfile.TemporaryDirectory(prefix=".frame-app-v171-preflight-", dir=prior.BASE) as directory:
        folder = Path(directory)
        (folder / "tmp").mkdir()
        manifest = folder / "models.yaml"
        manifest.write_bytes((prior.TRIAL / f"{arm}.yaml").read_bytes())

        def environment(device, models, hf, data, work):
            return controlled_environment(device, models, hf, data, work) | {
                "VERITAS_MODELS_MANIFEST": str(manifest)
            }

        context = scoped(initialize, {"controlled_environment": environment})(
            folder, old["configured_device"]
        )
        registry, manager, runtime = prior.baseline.runtime_identity(context.confirm(), old, folder)
        from app.engine.boundary import planned_allowance
        from app.storage.db import Analysis, get_session, shutdown

        try:
            allowance, plan = planned_allowance(registry.instance(prior.DETECTOR), None, manager)
            context.confirm()
            version, _, batch = prior.ARMS[arm]
            with get_session() as session:
                require(
                    session.query(Analysis).count() == 0 and not manager.snapshot(),
                    "A preflight must not start an analysis or acquire a model",
                )
            require(
                registry.adapter_spec(prior.DETECTOR).version == version
                and {use.model_id: use.max_items for use in plan} == dict.fromkeys(prior.MODELS, batch)
                and allowance == len(prior.MODELS) * batch * 400 * 1024**2,
                "Preflight selected a different adapter or activation plan",
            )
            return {
                "status": "model_free_inference_boundary_verified",
                "arm": arm,
                "device": runtime["resolved_device"],
                "planned_activation_bytes": allowance,
                "models_loaded": 0,
                "environment_additions": sorted(expected_runtime_additions(folder)),
            }
        finally:
            shutdown()


def worker(arm, output):
    serving = types.SimpleNamespace(
        apfs=prior.serving.apfs, controlled_environment=controlled_environment, initialize=initialize
    )
    scoped(prior.worker, {"TRIAL": TRIAL, "WORK": WORK, "verify": verify, "serving": serving})(arm, output)


def run(arm):
    scoped(prior.run, {"TRIAL": TRIAL, "MODULE": MODULE, "verify": verify, "__file__": __file__})(arm)


def readout():
    summary = scoped(prior.readout, {"TRIAL": TRIAL, "WORK": WORK, "verify": verify, "__file__": __file__})()
    return {**summary, "schema": "chipmunk.video-frame-full-app-readout/171"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("freeze")
    commands.add_parser("readout")
    for action in ("preflight", "run", "_worker"):
        command = commands.add_parser(action)
        command.add_argument("arm", choices=prior.ARMS)
        if action == "_worker":
            command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "readout":
        print(json.dumps(readout(), allow_nan=False))
    elif args.action == "preflight":
        print(json.dumps(preflight(args.arm), allow_nan=False))
    elif args.action == "run":
        run(args.arm)
    else:
        worker(args.arm, args.output)


if __name__ == "__main__":
    main()
