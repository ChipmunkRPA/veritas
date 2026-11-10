"""Isolated corrected execution: resolve the saved cutoff record before inference.

Frozen v108/v111 source, code objects, model math and failed artifacts remain
unchanged. New globals supply canonical planning data and exclusive paths.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import types
from pathlib import Path

from scripts import video_appearance_native_v108 as native
from scripts import video_appearance_readout_v111 as readout
from scripts import video_appearance_runtime_v108 as runtime

ROOT, BASE, PANEL = native.ROOT, native.BASE, native.PANEL
TRIAL = BASE / "video-appearance-corrected-v112"
PREP = TRIAL / "prepared"
PUBLIC = PANEL / "video-appearance-corrected-freeze-v112.json"
READOUT_PUBLIC = PANEL / "video-appearance-corrected-readout-freeze-v112.json"
RESULT_PANEL = PANEL / "video-appearance-corrected-v112"
CODE = (
    "backend/scripts/video_appearance_corrected_v112.py",
    "backend/tests/test_video_appearance_corrected_v112.py",
    "docs/video-appearance-corrected-v112-method.md",
)


def point_value(record):
    native.require(
        type(record) is dict
        and set(record) == {"threshold", "common_genuine_groups", "budget", "groups_flagged", "comparison"},
        "Versioned v106 cutoff record required",
    )
    value = record["threshold"]
    native.require(
        type(value) is float
        and math.isfinite(value)
        and 0 < value < 1
        and record["comparison"] == "strict_greater_than",
        "Finite unchanged strict research point required",
    )
    native.require(
        all(type(record[k]) is int for k in ("common_genuine_groups", "budget", "groups_flagged"))
        and 0 <= record["groups_flagged"] <= record["budget"] < record["common_genuine_groups"],
        "Complete cutoff support metadata required",
    )
    return value


def canonical_plan(original):
    result = copy.deepcopy(original)
    result["candidate_point"] = point_value(original["candidate_point"])
    return result


def scoped(function, overrides):
    context = {**function.__globals__, **overrides}
    result = types.FunctionType(
        function.__code__, context, function.__name__, function.__defaults__, function.__closure__
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def extractor():
    return scoped(native.extract, {"TRIAL": TRIAL, "PREP": PREP})


def identity():
    native.verify()
    old = native.read(native.PREP / "plan.json")
    record = old["candidate_point"]
    native.require(
        record
        == native.read(native.FITTED)["points"]["appearance_only"]
        == native.read(PANEL / "video-joint-scalar-result-v106.json")["new_points"]["appearance_only"],
        "Exactly the sealed selected record required",
    )
    native.require(
        native.read(PREP / "plan.json") == canonical_plan(old),
        "Only cutoff-record representation may change in derived plan",
    )
    failed = native.TRIAL / "extract-00"
    profile = native.read(failed / "profile.json")
    native.require(
        profile["child_exit_code"] == 1
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and (failed / "output/responses.jsonl").read_bytes() == b"",
        "Preserved original schema-failure evidence required",
    )
    old_lock = native.read(native.TRIAL / "freeze.json")
    code = {**old_lock["code_sha256"], **{p: native.sha(ROOT / p) for p in readout.CODE + CODE}}
    for name in ["authored"] + [f"bind-{i:02d}" for i in range(8)]:
        native.require(
            (TRIAL / name).is_symlink() and (TRIAL / name).resolve() == (native.TRIAL / name).resolve(),
            "Exact read-only qualified-artifact aliases required",
        )
    return {
        "schema": "chipmunk.video-appearance-corrected-freeze/112",
        "code_sha256": code,
        "original_native_freeze_sha256": native.sha(native.TRIAL / "freeze.json"),
        "original_support_plan_sha256": native.sha(native.PREP / "plan.json"),
        "derived_plan_sha256": native.sha(PREP / "plan.json"),
        "cutoff_record": record,
        "candidate_point": point_value(record),
        "raw_point": old_lock["raw_point"],
        "fitted_models_sha256": old_lock["fitted_models_sha256"],
        "local_heads_sha256": old_lock["local_heads_sha256"],
        "preserved_failure_profile_sha256": native.sha(failed / "profile.json"),
        "preserved_failure_log_sha256": native.sha(failed / "worker.log"),
        "preserved_empty_response_journal_sha256": native.sha(failed / "output/responses.jsonl"),
        "qualified_authored_result_sha256": native.sha(native.TRIAL / "authored/output/result.json"),
        "qualified_authored_profile_sha256": native.sha(native.TRIAL / "authored/profile.json"),
        "planned_cases": 138,
        "planned_batches": 8,
        "scope": "New isolated execution fixing only a saved-record/scalar handoff. Existing frozen code objects, source cases, model math, points, resource limits and failed artifacts unchanged. No fitting or serving promotion.",
    }


def freeze():
    TRIAL.mkdir(exist_ok=False)
    PREP.mkdir(exist_ok=False)
    RESULT_PANEL.mkdir(exist_ok=False)
    native.write(PREP / "plan.json", canonical_plan(native.read(native.PREP / "plan.json")))
    for name in ["authored"] + [f"bind-{i:02d}" for i in range(8)]:
        (TRIAL / name).symlink_to((native.TRIAL / name).resolve(), target_is_directory=True)
    body = identity()
    native.write(TRIAL / "freeze.json", body)
    native.write(PUBLIC, body)
    print(
        json.dumps(
            {
                "status": "frozen",
                "candidate_point": body["candidate_point"],
                "planned_cases": 138,
                "new_neural_calls": 0,
            }
        ),
        flush=True,
    )


def verify():
    native.require(
        native.read(TRIAL / "freeze.json") == native.read(PUBLIC) == identity(),
        "Unchanged published corrected context required",
    )


def routed_popen(argv, **kwargs):
    native.require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "extract"],
        "Only the original extract invocation can be routed",
    )
    routed = list(argv)
    routed[3] = "scripts.video_appearance_corrected_v112"
    return subprocess.Popen(routed, **kwargs)


def run(batch):
    native.require(type(batch) is int and 0 <= batch < 8, "One fixed source batch required")
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    runner = scoped(
        runtime.run, {"TRIAL": TRIAL, "verify": verify, "subprocess": proxy, "__file__": str(Path(__file__))}
    )
    runner("extract", batch)


def readout_context():
    context = {
        **vars(readout),
        "NATIVE": TRIAL,
        "PREP": PREP,
        "TRIAL": TRIAL / "readout",
        "PUBLIC": READOUT_PUBLIC,
        "PANEL": RESULT_PANEL,
        "CODE": readout.CODE + CODE,
    }
    for name in ("identity", "freeze", "evaluate"):
        fn = getattr(readout, name)
        clone = types.FunctionType(fn.__code__, context, fn.__name__, fn.__defaults__, fn.__closure__)
        clone.__kwdefaults__ = fn.__kwdefaults__
        context[name] = clone
    return context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "extract", "readout-freeze", "readout"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "freeze":
        native.require(not args.arguments, "Freeze has no arguments")
        freeze()
    elif args.command == "run":
        native.require(len(args.arguments) == 1, "One batch argument required")
        run(int(args.arguments[0]))
    elif args.command == "extract":
        verify()
        native.require(len(args.arguments) == 2, "Exact runtime output/batch required")
        output = Path(args.arguments[0])
        native.require(output.resolve().is_relative_to(TRIAL.resolve()), "Private exclusive output required")
        output.mkdir(exist_ok=False)
        extractor()(int(args.arguments[1]), output)
    else:
        verify()
        native.require(not args.arguments, "Readout has no arguments")
        readout_context()["freeze" if args.command == "readout-freeze" else "evaluate"]()
