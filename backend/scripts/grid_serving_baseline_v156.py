"""Provide an owned Numba cache to the unchanged full serving pipeline."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

from scripts import grid_serving_baseline_v154 as apfs
from scripts.video_appearance_corrected_v112 import scoped

bootstrap, baseline = apfs.bootstrap, apfs.baseline
ROOT, BASE, PANEL = apfs.ROOT, apfs.BASE, apfs.PANEL
TRIAL = BASE / "grid-serving-baseline-v156"
WORK = BASE / "grid-serving-work-v156"
PUBLIC = PANEL / "grid-serving-baseline-freeze-v156.json"
CODE = (
    "backend/scripts/grid_serving_baseline_v156.py",
    "backend/tests/test_grid_serving_baseline_v156.py",
    "docs/grid-serving-baseline-v156-method.md",
    "backend/scripts/grid_serving_baseline_v154.py",
)
FREEZE_SCHEMA = "chipmunk.grid-serving-baseline-freeze/156"
RESULT_SCHEMA = "chipmunk.grid-serving-baseline-result/156"
DEPENDENCIES = {
    "numba": ("numba/core/caching.py",),
    "llvmlite": ("llvmlite/binding/ffi.py",),
    "librosa": ("librosa/core/notation.py", "librosa/filters.py"),
    "torchlibrosa": ("torchlibrosa/stft.py",),
    "joblib": ("joblib/memory.py",),
}
read, write, sha, require = apfs.read, apfs.write, apfs.sha, apfs.require


def controlled_environment(device, models, hf, data, work):
    env = baseline.controlled_environment(device, models, hf, data, work)
    env["NUMBA_CACHE_DIR"] = str(work / "cache/numba")
    return env


def initialize(folder, device):
    require("numba" not in sys.modules and "librosa" not in sys.modules, "Fresh auxiliary imports required")
    return scoped(
        bootstrap.initialize,
        {
            "baseline": types.SimpleNamespace(
                controlled_environment=controlled_environment,
                fresh_settings=baseline.fresh_settings,
            ),
        },
    )(folder, device)


def work_volume():
    return scoped(apfs.work_volume, {"WORK": WORK, "TRIAL": TRIAL})()


def dependency_identity():
    records = {}
    for package, paths in DEPENDENCIES.items():
        dist = importlib.metadata.distribution(package)
        records[package] = {
            "version": dist.version,
            "code_sha256": {path: sha(dist.locate_file(path)) for path in paths},
        }
    return records


def authored_cache_operation(folder, cache):
    fixture = folder / "authored_numba_v156.py"
    fixture.write_text(
        "from numba import njit\n@njit(cache=True)\ndef add_one(value):\n    return value + 1\n"
    )
    spec = importlib.util.spec_from_file_location(fixture.stem, fixture)
    require(spec is not None and spec.loader is not None, "Authored module must have a real source locator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(module.add_one(41) == 42, "Authored cache operation returned an unexpected value")
    outputs = [p for p in cache.rglob(f"{fixture.stem}.*") if p.is_file()]
    require(
        {p.suffix for p in outputs} == {".nbi", ".nbc"}
        and all(p.resolve().is_relative_to(cache.resolve()) for p in outputs),
        "Authored Numba index and machine-code cache must stay inside the owned directory",
    )


def cache_smoke(device):
    volume = work_volume()
    require(shutil.disk_usage(WORK.parent).free >= 3 * 1024**3, "Unchanged initial disk reserve required")
    WORK.mkdir(mode=0o700, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".numba-smoke-", dir=WORK) as directory:
        folder = Path(directory)
        context = initialize(folder, device)
        (folder / "tmp").mkdir()
        cache = Path(context.env["NUMBA_CACHE_DIR"])
        baseline.isolate_filesystem(folder, folder, set())

        import numpy as np

        from app.detectors.audio.upgraded import SoundEventsModel
        from app.third_party.panns.cnn10 import Cnn10, LogmelFilterBank

        # The same non-learned filter bank constructor used by Cnn10.
        filterbank = LogmelFilterBank(
            sr=32000,
            n_fft=1024,
            n_mels=64,
            fmin=50,
            fmax=14000,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )
        matrix = filterbank.melW.detach().cpu().numpy()
        require(
            matrix.shape == (513, 64) and np.isfinite(matrix).all() and np.all(matrix >= 0),
            "Production mel-filter dependency returned an invalid authored matrix",
        )

        import librosa.core.notation
        import librosa.filters
        import numba
        import torchlibrosa.stft

        require(
            Path(numba.config.CACHE_DIR) == cache
            and SoundEventsModel.__module__ == "app.detectors.audio.upgraded"
            and Cnn10.__module__ == "app.third_party.panns.cnn10"
            and LogmelFilterBank is torchlibrosa.stft.LogmelFilterBank,
            "Exact production sound-event dependency must import with the owned cache",
        )
        authored_cache_operation(folder, cache)
        require(
            cache.stat().st_dev == volume["device"]
            and all(path.resolve().is_relative_to(cache.resolve()) for path in cache.rglob("*")),
            "Auxiliary generated files must remain on the verified work volume",
        )
        dependencies = dependency_identity()
        imported = {
            "librosa": {
                "librosa/core/notation.py": librosa.core.notation,
                "librosa/filters.py": librosa.filters,
            },
            "torchlibrosa": {"torchlibrosa/stft.py": torchlibrosa.stft},
        }
        require(
            all(
                sha(module.__file__) == dependencies[pkg]["code_sha256"][path]
                for pkg, modules in imported.items()
                for path, module in modules.items()
            ),
            "Imported sound dependency differs from its bound installed files",
        )
        return {
            "schema": "chipmunk.grid-serving-numba-cache-smoke/156",
            "filesystem": volume["filesystem"],
            "device": volume["device"],
            "dependencies": dependencies,
            "production_dependency_import_ok": True,
            "unlearned_filter_shape": list(matrix.shape),
            "authored_cache_index_and_code_present": True,
            "model_loads": 0,
            "learned_forwards": 0,
        }


def stage_folder(ordinal):
    return TRIAL / f"clip-{ordinal:02d}/authored"


def failure_history(context):
    prior, saved = apfs.verify_initialized(context)
    rows = apfs.read_cases(context, 0, prior)
    stage = apfs.stage_folder(0)
    journal = apfs.WORK / "clip-00/cases.jsonl"
    log = stage / "worker.log"
    logged = log.read_text()
    sounds = [[r for r in row["detector_runs"] if r["detector_id"] == "aud.sound_events"] for row in rows]
    require(
        len(rows) == 11
        and all(row["status"] == "done" for row in rows)
        and all(len(runs) == 1 and runs[0]["status"] == "failed" for runs in sounds)
        and "cannot cache function '__o_fold': no locator available" in logged,
        "Exact v154 partial-coverage history required",
    )
    return (
        prior,
        saved,
        {
            "v154_freeze_sha256": sha(apfs.PUBLIC),
            "v154_result_sha256": sha(stage / "output/result.json"),
            "v154_profile_sha256": sha(stage / "profile.json"),
            "v154_resource_journal_sha256": sha(stage / "resources.jsonl"),
            "v154_worker_log_sha256": sha(log),
            "v154_case_journal_sha256": sha(journal),
            "completed_analyses": len(rows),
            "sound_event_failures": len(sounds),
            "cause": "Numba could not use a permitted disk cache when the sound-event dependency loaded",
        },
    )


def identity(prior, history, smoke):
    return {
        "schema": FREEZE_SCHEMA,
        "code_sha256": {path: sha(ROOT / path) for path in CODE},
        "baseline_freeze_sha256": prior["baseline_freeze_sha256"],
        "storage_smoke_sha256": prior["storage_smoke_sha256"],
        "configured_device": prior["configured_device"],
        "profile": prior["profile"],
        "pilot_clip_ordinals": prior["pilot_clip_ordinals"],
        "cases_per_clip": prior["cases_per_clip"],
        "historical_v154_coverage": history,
        "numba_cache_smoke_sha256": sha(TRIAL / "numba-cache-smoke.json"),
        "dependency_versions": {name: row["version"] for name, row in smoke["dependencies"].items()},
        "scope": "Exact v154 serving pipeline and resource limits. Set NUMBA_CACHE_DIR on owned APFS before auxiliary imports.",
    }


def verify_initialized(context):
    prior, saved, history = failure_history(context)
    smoke = read(TRIAL / "numba-cache-smoke.json")
    volume = work_volume()
    require(
        smoke["schema"] == "chipmunk.grid-serving-numba-cache-smoke/156"
        and smoke["device"] == volume["device"]
        and smoke["filesystem"] == volume["filesystem"]
        and smoke["dependencies"] == dependency_identity()
        and smoke["production_dependency_import_ok"] is smoke["authored_cache_index_and_code_present"] is True
        and smoke["unlearned_filter_shape"] == [513, 64]
        and smoke["model_loads"] == smoke["learned_forwards"] == 0,
        "Successful auxiliary dependency and cache smoke on the same APFS volume required",
    )
    frozen = identity(prior, history, smoke)
    require(frozen == read(PUBLIC) == read(TRIAL / "freeze.json"), "Unchanged v156 context required")
    return frozen, saved


def read_cases(context, ordinal, frozen):
    # Reuse the v154 complete case validator. Bind the actual v156 caller and
    # cache proof before delegating its unchanged roster/journal checks.
    stage = stage_folder(ordinal)
    receipt = baseline.speech.qualification_profile(stage, None, {}, "grid_serving_baseline_v156.py")
    require(
        receipt["numba_cache_smoke_sha256"] == frozen["numba_cache_smoke_sha256"],
        "Actual serving receipt must bind its admitted Numba cache proof",
    )

    def qualification(folder, expected, bindings, caller):
        require(
            folder == stage
            and expected is None
            and not bindings
            and caller == "grid_serving_baseline_v154.py",
            "Expected unchanged base case validation protocol",
        )
        return receipt

    return scoped(
        apfs.read_cases,
        {
            "stage_folder": stage_folder,
            "WORK": WORK,
            "PUBLIC": PUBLIC,
            "RESULT_SCHEMA": RESULT_SCHEMA,
            "baseline": types.SimpleNamespace(
                speech=types.SimpleNamespace(qualification_profile=qualification),
                planned_cases=baseline.planned_cases,
                planning=baseline.planning,
                validate_case_rows=baseline.validate_case_rows,
                count_results=baseline.count_results,
                TRIAL=baseline.TRIAL,
            ),
        },
    )(context, ordinal, frozen)


def metadata_entry(action, ordinal):
    with contextlib.redirect_stdout(sys.stderr):
        device = bootstrap.header()["configured_device"]
        if action == "smoke":
            return cache_smoke(device)
        context = initialize(WORK / "metadata-only", device)
        if action == "history":
            return failure_history(context)
        frozen, _ = verify_initialized(context)
        if action == "verify":
            return frozen
        ordinals = frozen["pilot_clip_ordinals"]
        require(type(ordinal) is int and ordinal in ordinals, "Fixed pilot ordinal required")
        if action == "load":
            return read_cases(context, ordinal, frozen)
        require(action == "prerequisite", "Exact metadata command required")
        previous = ordinals.index(ordinal) - 1
        if previous >= 0:
            read_cases(context, ordinals[previous], frozen)
        bootstrap.bind_case(context, ordinal)
        return True


def metadata(action, ordinal=None):
    args = [sys.executable, "-B", "-m", "scripts.grid_serving_baseline_v156", "_metadata", action]
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
        raise RuntimeError(response.stderr.strip() or "Isolated v156 metadata failed")
    return json.loads(response.stdout)


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every v156 attempt and freeze")
    prior, _, history = metadata("history")
    smoke = metadata("smoke")
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "numba-cache-smoke.json", smoke)
    frozen = identity(prior, history, smoke)
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    print(json.dumps({"status": "frozen", "freeze_sha256": sha(PUBLIC)}), flush=True)


def worker_entry(ordinal, output):
    saved = bootstrap.header()
    stage = stage_folder(ordinal)
    require(
        ordinal in saved["pilot_clip_ordinals"]
        and output.resolve() == (stage / "output").resolve()
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and not (stage / "profile.json").exists(),
        "Exclusive guarded v156 output required",
    )
    folder = WORK / f"clip-{ordinal:02d}"
    require(folder.resolve() == folder and not folder.exists(), "Exact unused APFS working path required")
    context = initialize(folder, saved["configured_device"])
    frozen, prior = verify_initialized(context)
    output.mkdir(exist_ok=False)

    def write_bound(path, data):
        if path == output / "result.json":
            data.update(
                storage_smoke_sha256=frozen["storage_smoke_sha256"],
                numba_cache_smoke_sha256=frozen["numba_cache_smoke_sha256"],
            )
        write(path, data)

    worker = scoped(
        bootstrap.corrected_worker,
        {
            "TRIAL": TRIAL,
            "CACHE": WORK,
            "PUBLIC": PUBLIC,
            "RESULT_SCHEMA": RESULT_SCHEMA,
            "write": write_bound,
        },
    )(context, output, frozen)
    scoped(
        worker,
        {
            "analyze_case": apfs.analyze_case,
            "controlled_environment": controlled_environment,
        },
    )(ordinal, output, prior)


def prerequisites(ordinal, ordinals):
    return scoped(apfs.prerequisites, {"stage_folder": stage_folder, "metadata": metadata})(ordinal, ordinals)


def run(ordinal):
    from scripts import video_appearance_runtime_v108 as runtime

    ordinals = metadata("verify")["pilot_clip_ordinals"]
    require(type(ordinal) is int and ordinal in ordinals, "Fixed pilot ordinal required")
    stage = stage_folder(ordinal)
    stage.parent.mkdir(exist_ok=True)

    def popen(argv, **kwargs):
        output = stage / "output"
        require(
            argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored"]
            and argv[5:] == [str(output)],
            "Only the exclusive v156 full-profile worker is permitted",
        )
        return subprocess.Popen(
            [*argv[:3], "scripts.grid_serving_baseline_v156", "_worker", str(ordinal), str(output)], **kwargs
        )

    scoped(
        runtime.run,
        {
            "TRIAL": stage.parent,
            "verify": lambda: metadata("verify"),
            "prerequisites": lambda _c, _b: prerequisites(ordinal, ordinals),
            "subprocess": types.SimpleNamespace(
                Popen=popen,
                STDOUT=subprocess.STDOUT,
                TimeoutExpired=subprocess.TimeoutExpired,
            ),
            "__file__": str(Path(__file__)),
        },
    )("authored")


def load_completed(ordinal):
    require(type(ordinal) is int, "Fixed pilot ordinal required")
    return metadata("load", ordinal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    commands.add_parser("run").add_argument("ordinal", type=int)
    worker = commands.add_parser("_worker")
    worker.add_argument("ordinal", type=int)
    worker.add_argument("output", type=Path)
    reader = commands.add_parser("_metadata")
    reader.add_argument("action", choices=("history", "smoke", "verify", "prerequisite", "load"))
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
