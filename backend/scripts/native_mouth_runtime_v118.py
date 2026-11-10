"""Execute the v117 contract with the retained isolated generator runtime."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = ROOT / "backend/data/independent-evaluation/lipsync-v7/runtime"
sys.path.insert(0, str(OVERLAY))

from scripts import native_mouth_renderer_v117 as core  # noqa: E402

TRIAL = core.BASE / "native-mouth-renderer-v118"
CACHE = core.CACHE.parent / "native-mouth-renderer-v118"
PUBLIC = core.PANEL / "native-mouth-renderer-freeze-v118.json"
CODE = core.CODE + (
    "backend/scripts/native_mouth_runtime_v118.py",
    "backend/tests/test_native_mouth_runtime_v118.py",
    "docs/native-mouth-runtime-v118-method.md",
)


def overlay_identity():
    rows = []
    expected = core.read(core.PANEL / "mouth-v7-runtime.json")["packages"]
    for row in expected:
        dist = importlib.metadata.distribution(row["name"])
        core.require(dist.version == row["version"], "Exact retained generator runtime version required")
        files = []
        for file in dist.files:
            if file.suffix != ".py":
                continue
            path = Path(dist.locate_file(file)).resolve()
            core.require(
                path.is_relative_to(OVERLAY.resolve())
                and file.hash is not None
                and file.hash.mode == "sha256",
                "Recorded isolated Python source required",
            )
            digest = hashlib.sha256(path.read_bytes()).digest()
            actual = base64.urlsafe_b64encode(digest).decode().rstrip("=")
            core.require(actual == file.hash.value, "Installed runtime source differs from its RECORD")
            files.append((str(file), digest.hex()))
        core.require(files, "Complete retained Python runtime sources required")
        rows.append(
            {
                "name": row["name"],
                "version": dist.version,
                "source_files": len(files),
                "source_manifest_sha256": hashlib.sha256(
                    json.dumps(sorted(files), separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    return rows


def context():
    scope = {
        **vars(core),
        "TRIAL": TRIAL,
        "CACHE": CACHE,
        "PUBLIC": PUBLIC,
        "CODE": CODE,
        "__file__": str(Path(__file__)),
    }
    for name in ("identity", "freeze", "verify", "run", "qualify"):
        fn = getattr(core, name)
        scope[name] = types.FunctionType(fn.__code__, scope, fn.__name__, fn.__defaults__, fn.__closure__)
    measurement_identity = scope["identity"]

    def identity():
        body = measurement_identity()
        body["execution_context_schema"] = "chipmunk.native-mouth-isolated-context/118"
        body["retained_runtime"] = overlay_identity()
        body["runtime_plan_sha256"] = core.sha(core.PANEL / "mouth-v7-runtime.json")
        body["runtime_lock_sha256"] = core.sha(core.PANEL / "mouth-v7-runtime.lock")
        body["preserved_prefreeze_failure_sha256"] = core.sha(core.TRIAL / "preparation-failure.json")
        core.require(
            core.read(core.TRIAL / "preparation-failure.json")["new_neural_calls"] == 0
            and core.read(core.TRIAL / "selection.json") == core.read(TRIAL / "selection.json"),
            "Unchanged source after preserved preparation failure required",
        )
        return body

    def routed_popen(argv, **kwargs):
        core.require(
            argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored"],
            "Only fixed native qualification can be routed",
        )
        routed = list(argv)
        routed[3] = "scripts.native_mouth_runtime_v118"
        return subprocess.Popen(routed, **kwargs)

    scope["identity"], scope["routed_popen"] = identity, routed_popen
    return scope


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "authored"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    scope = context()
    if args.command == "freeze":
        core.require(args.output is None, "Freeze has no output argument")
        scope["freeze"]()
    elif args.command == "run":
        core.require(args.output is None, "Run has no output argument")
        scope["run"]()
    else:
        scope["verify"]()
        core.require(
            args.output is not None and args.output.resolve().is_relative_to(TRIAL.resolve()),
            "Private exclusive guarded output required",
        )
        args.output.mkdir(exist_ok=False)
        scope["qualify"](args.output)
