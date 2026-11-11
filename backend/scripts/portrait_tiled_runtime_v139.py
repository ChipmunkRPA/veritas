"""New CPU context with bounded spatial convolutions; v133 failure untouched."""

from __future__ import annotations

import argparse
import json
import subprocess
import types
from pathlib import Path

from scripts import portrait_native_qualifier_v133 as original
from scripts import spatial_conv_cpu_v139 as tiling

ROOT, BASE, PANEL = original.ROOT, original.BASE, original.PANEL
TRIAL = BASE / "portrait-tiled-runtime-v139"
CACHE = original.CACHE.parent / "portrait-tiled-runtime-v139"
PUBLIC = PANEL / "portrait-tiled-runtime-freeze-v139.json"
CODE = original.CODE + (
    "backend/scripts/spatial_conv_cpu_v139.py",
    "backend/tests/test_spatial_conv_cpu_v139.py",
    "backend/scripts/portrait_tiled_runtime_v139.py",
    "backend/tests/test_portrait_tiled_runtime_v139.py",
    "docs/portrait-tiled-runtime-v139-method.md",
)
read, write, sha, require = original.read, original.write, original.sha, original.require


def partial_reference():
    original.verify()
    folder = original.TRIAL / "authored"
    profile = read(folder / "profile.json")
    require(
        profile["child_exit_code"] == -15
        and profile["guard"] == "resource_limit"
        and profile["sample_errors"] == 0
        and profile["worker_receipt_sha256"] is None
        and any("tree_rss_limit" in r["violations"] for r in profile["resource_limit_observations"]),
        "Preserved original CPU memory-limit failure required",
    )
    files = []
    for suffix in ("crop", "native", "mask"):
        path = original.CACHE / f"same_aperture-{suffix}.png"
        require(path.is_file(), "Original first complete partial render support required")
        files.append({"kind": suffix, "file": str(path), "bytes": path.stat().st_size, "sha256": sha(path)})
    return files


def compare_images(actual, reference):
    import numpy as np

    require(
        actual.shape == reference.shape == (512, 512, 3) and actual.dtype == reference.dtype == np.uint8,
        "Comparable original/full-resolution crop bytes required",
    )
    difference = np.abs(actual.astype(np.int16) - reference.astype(np.int16))
    metrics = {
        "maximum_abs_rgb_byte_difference": int(difference.max()),
        "mean_abs_rgb_byte_difference": float(difference.mean()),
        "changed_channel_values": int((difference != 0).sum()),
    }
    metrics["within_frozen_tolerance"] = (
        metrics["maximum_abs_rgb_byte_difference"] <= 1 and metrics["mean_abs_rgb_byte_difference"] <= 0.01
    )
    return metrics


def context():
    scope = {
        **vars(original),
        "TRIAL": TRIAL,
        "CACHE": CACHE,
        "PUBLIC": PUBLIC,
        "CODE": CODE,
        "__file__": str(Path(__file__)),
    }
    for name in ("identity", "verify", "run", "qualify"):
        fn = getattr(original, name)
        scope[name] = types.FunctionType(fn.__code__, scope, fn.__name__, fn.__defaults__, fn.__closure__)
    base_identity, base_qualify = scope["identity"], scope["qualify"]

    def identity():
        require(
            read(TRIAL / "partial-reference.json") == partial_reference(),
            "Original partial output bytes must remain unchanged",
        )
        body = base_identity()
        body.update(
            execution_context_schema="chipmunk.portrait-tiled-cpu-context/139",
            convolution_workspace_estimate_budget_bytes=tiling.WORKSPACE,
            partial_reference_manifest_sha256=sha(TRIAL / "partial-reference.json"),
            preserved_failure_profile_sha256=sha(original.TRIAL / "authored/profile.json"),
            preserved_failure_log_sha256=sha(original.TRIAL / "authored/worker.log"),
            preserved_failure_resource_journal_sha256=sha(original.TRIAL / "authored/resources.jsonl"),
            partial_reference_tolerance={
                "maximum_abs_rgb_byte_difference": 1,
                "mean_abs_rgb_byte_difference": 0.01,
            },
            context_change="Spatially tile large CPU Conv2d/Conv3d outputs with exact halos before full-output normalization/nonlinearity. Same sources, weights, CPU float32, full 256/512 dimensions, four conditions and neural resource limits. No detector or calibration changes.",
        )
        return body

    def qualify(output):
        import cv2

        from scripts import liveportrait_mouth_benchmark as portrait

        underlying_load = portrait.load
        patches, handles, events = [], [], []
        counts = {
            "spatial_convolutions": 0,
            "functional_tile_calls": 0,
            "maximum_estimated_tile_columns_bytes": 0,
        }
        root_completed = {}
        with (output / "forward-progress.jsonl").open("x") as journal:

            def observe(event):
                journal.write(json.dumps(event, allow_nan=False) + "\n")
                journal.flush()
                if event["event"] == "spatial_conv_begin":
                    counts["spatial_convolutions"] += 1
                    counts["maximum_estimated_tile_columns_bytes"] = max(
                        counts["maximum_estimated_tile_columns_bytes"], event["estimated_tile_columns_bytes"]
                    )
                elif event["event"] == "spatial_conv_end":
                    counts["functional_tile_calls"] += event["tiles"]
                events.append(event["event"])

            def before(name):
                return lambda module, args: observe({"event": "top_module_begin", "module": name})

            def after(name):
                def hook(module, args, result):
                    root_completed[name] = root_completed.get(name, 0) + 1
                    observe({"event": "top_module_end", "module": name})

                return hook

            def load(*args, **kwargs):
                observe({"event": "loader_begin_after_original_geometry_checks"})
                renderer = underlying_load(*args, **kwargs)
                observe({"event": "strict_loader_returned"})
                patches.extend(tiling.install(renderer.models, observe))
                for name, module in renderer.models.items():
                    handles.extend(
                        [
                            module.register_forward_pre_hook(before(name)),
                            module.register_forward_hook(after(name)),
                        ]
                    )
                return renderer

            def observed_write(path, value):
                if Path(path) == output / "result.json":
                    require(
                        root_completed == value["top_level_module_forwards"]
                        and counts["spatial_convolutions"] > 0
                        and counts["maximum_estimated_tile_columns_bytes"] <= tiling.WORKSPACE,
                        "Complete original module counts and bounded tile estimates required",
                    )
                    actual = cv2.imread(str(CACHE / "same_aperture-crop.png"))
                    reference = cv2.imread(str(original.CACHE / "same_aperture-crop.png"))
                    require(
                        actual is not None and reference is not None,
                        "Actual partial-reference crop comparison required",
                    )
                    fidelity = compare_images(actual, reference)
                    write(output / "partial-reference-fidelity.json", fidelity)
                    require(
                        fidelity["within_frozen_tolerance"],
                        "Full-model partial-reference fidelity failed; preserve all outputs",
                    )
                    value.update(
                        execution_context_schema="chipmunk.portrait-tiled-cpu-result/139",
                        bounded_convolutions=counts.copy(),
                        partial_reference_fidelity=fidelity,
                        private_forward_progress_sha256=sha(output / "forward-progress.jsonl"),
                        preserved_original_failure_profile_sha256=sha(
                            original.TRIAL / "authored/profile.json"
                        ),
                        original_failure_remains_failed=True,
                    )
                write(path, value)

            scope["write"] = observed_write
            portrait.load = load
            try:
                base_qualify(output)
            finally:
                portrait.load = underlying_load
                scope["write"] = write
                for handle in handles:
                    handle.remove()
                tiling.restore(patches)

    def routed_popen(argv, **kwargs):
        require(
            argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored"],
            "Only fixed tiled CPU qualification permitted",
        )
        routed = list(argv)
        routed[3] = "scripts.portrait_tiled_runtime_v139"
        return subprocess.Popen(routed, **kwargs)

    scope.update(identity=identity, qualify=qualify, routed_popen=routed_popen)
    return scope


def freeze():
    partial = partial_reference()
    selected = original.selection()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "partial-reference.json", partial)
    write(TRIAL / "selection.json", selected)
    body = context()["identity"]()
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print('{"status":"frozen","device":"cpu","precision":"float32","conditions":4}', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "authored"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        require(args.output is None, "No output for freeze")
        freeze()
    elif args.command == "run":
        require(args.output is None, "No output for run")
        context()["run"]()
    else:
        scope = context()
        scope["verify"]()
        require(
            args.output is not None and args.output.resolve() == (TRIAL / "authored/output").resolve(),
            "Exact exclusive guarded output required",
        )
        args.output.mkdir(exist_ok=False)
        scope["qualify"](args.output)
