"""Explicit data-only continuation; preserve the stopped v131 attempt unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import types
from pathlib import Path
from urllib.request import Request, urlopen

from scripts import grid_media_preflight_v134 as preflight
from scripts.video_appearance_corrected_v112 import scoped

old = preflight.acquisition
core = old.original
ROOT, BASE, PANEL = core.ROOT, core.BASE, core.PANEL
TRIAL = BASE / "grid-acquisition-continuation-v135"
CACHE = core.runtime.ARCHIVE / "grid-acquisition-continuation-v135"
PUBLIC = PANEL / "grid-acquisition-continuation-freeze-v135.json"
CODE = (
    "backend/scripts/grid_acquisition_continuation_v135.py",
    "backend/tests/test_grid_acquisition_continuation_v135.py",
    "docs/grid-acquisition-continuation-v135-method.md",
)
read, write, sha, require = core.read, core.write, core.sha, core.require


def preserved():
    preflight.verify()
    source = read(preflight.TRIAL / "plan.json")["archive"]
    stage = preflight.TRIAL / "bind-00"
    result, profile = read(stage / "output/result.json"), read(stage / "profile.json")
    require(
        result["status"] == "complete"
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["worker_receipt_sha256"] == sha(stage / "output/result.json"),
        "Actual complete independent first archive preflight required",
    )
    return source


def identity():
    first = preserved()
    plan = read(TRIAL / "plan.json")
    original = read(old.TRIAL / "plan.json")
    partial = old.CACHE / "batch-00/source-01.zip"
    require(
        plan["selected"] == original["selected"]
        and len(plan["selected"]) == 16
        and plan["first_complete"] == first
        and plan["preserved_prefix"]["file"] == str(partial)
        and plan["preserved_prefix"]["bytes"] == partial.stat().st_size == 337641472
        and plan["preserved_prefix"]["sha256"] == sha(partial),
        "All original sources and original stopped prefix must remain unchanged",
    )
    return {
        "schema": "chipmunk.grid-acquisition-continuation-freeze/135",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "private_plan_sha256": sha(TRIAL / "plan.json"),
        "original_acquisition_freeze_sha256": sha(old.PUBLIC),
        "original_stopped_profile_sha256": sha(old.TRIAL / "bind-00/profile.json"),
        "original_archive_journal_sha256": sha(old.TRIAL / "bind-00/output/archives.jsonl"),
        "independent_first_source_preflight_sha256": sha(preflight.TRIAL / "bind-00/output/result.json"),
        "original_source_archives": 16,
        "retained_complete_archives": 1,
        "new_stages": 15,
        "assets_per_new_stage": 1,
        "preserved_prefix_bytes": partial.stat().st_size,
        "missing_tail_bytes": original["selected"][1]["bytes"] - partial.stat().st_size,
        "license": "CC-BY-4.0",
        "transport_change": "First new stage requests only the missing publisher byte range, then copies the frozen prefix and new tail into a NEW archive for full size/MD5 verification. Later unattempted sources use one full archive per new bounded stage, rather than two. Original failed stage, prefix and first complete archive remain read-only; no source substitution or neural result selection.",
        "limits": "Per data stage unchanged: 900s, 1.5GiB sampled descendant RSS, 3GiB initial RAM/project/archive disk, 1.5GiB running reserves. More acquisition stages are explicitly planned; this is not a claim of the original eight-batch procedure passing.",
        "new_neural_calls": 0,
    }


def freeze():
    first = preserved()
    source = read(old.TRIAL / "plan.json")
    partial = old.CACHE / "batch-00/source-01.zip"
    plan = {
        "selected": source["selected"],
        "first_complete": first,
        "preserved_prefix": {"file": str(partial), "bytes": partial.stat().st_size, "sha256": sha(partial)},
    }
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "plan.json", plan)
    body = identity()
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print('{"status":"frozen","original_sources":16,"new_stages":15,"new_neural_calls":0}', flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(), "Unchanged explicit continuation required"
    )


def download_tail(row, start, target):
    require(type(start) is int and 0 < start < row["bytes"], "Exact nonempty missing tail required")
    total, expected = 0, row["bytes"] - start
    request = Request(
        row["url"],
        headers={
            "User-Agent": "Chipmunk-Verify-transfer-evaluation",
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{row['bytes'] - 1}",
        },
    )
    with target.open("xb") as output, urlopen(request, timeout=30) as response:
        require(
            response.status == 206
            and response.headers.get("Content-Range") == f"bytes {start}-{row['bytes'] - 1}/{row['bytes']}"
            and response.headers.get("Content-Encoding", "identity") == "identity",
            "Exact publisher missing-range response required; never fall back to full redownload",
        )
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            require(total <= expected, "Publisher range exceeds requested tail")
            output.write(chunk)
    require(total == expected, "Publisher range is incomplete")
    return total


def assemble(prefix, tail, target, expected):
    require(
        Path(prefix["file"]).stat().st_size == prefix["bytes"]
        and sha(Path(prefix["file"])) == prefix["sha256"],
        "Preserved prefix bytes changed",
    )
    total, md5, digest = 0, hashlib.md5(), hashlib.sha256()
    with target.open("xb") as output:
        for path in (Path(prefix["file"]), tail):
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    total += len(chunk)
                    require(total <= expected["bytes"], "Reconstructed archive exceeds publisher size")
                    output.write(chunk)
                    md5.update(chunk)
                    digest.update(chunk)
    require(
        total == expected["bytes"] and md5.hexdigest() == expected["md5"],
        "Full reconstructed publisher archive checksum/size mismatch",
    )
    return {"file": str(target), "bytes": total, "md5": md5.hexdigest(), "sha256": digest.hexdigest()}


def acquire(ordinal, output):
    require(type(ordinal) is int and 1 <= ordinal <= 15, "One original non-first archive required")
    plan = read(TRIAL / "plan.json")
    row = plan["selected"][ordinal]
    folder = CACHE / f"source-{ordinal:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    target = folder / "archive.zip"
    if ordinal == 1:
        tail = folder / "missing-tail.bin"
        prefix = plan["preserved_prefix"]
        transferred = download_tail(row, prefix["bytes"], tail)
        observed = assemble(prefix, tail, target, row)
    else:
        observed = core.download(row, target)
        transferred = observed["bytes"]
    media = core.inspect_archive(target)
    record = {
        **row,
        **observed,
        "status": "publisher_archive_verified",
        "real_video_members": media,
        "continued_prefix": ordinal == 1,
    }
    with (output / "archives.jsonl").open("x") as journal:
        journal.write(json.dumps(record, allow_nan=False) + "\n")
    result = {
        "schema": "chipmunk.grid-acquisition-continuation-result/135",
        "status": "complete",
        "archives": 1,
        "verified_archive_bytes": observed["bytes"],
        "new_network_bytes": transferred,
        "used_preserved_prefix": ordinal == 1,
        "real_video_members_inventoried": len(media),
        "private_archive_journal_sha256": sha(output / "archives.jsonl"),
        "freeze_sha256": sha(PUBLIC),
        "new_neural_calls": 0,
        "defaults_changed": False,
        "scope": "One original selected source archive; complete size/publisher MD5, observed SHA-256 and safe directory. No media decoding, detector response, completed-cohort or original failed-stage success claim. Raw assets and individual fingerprints stay private.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def routed_popen(ordinal, argv, **kwargs):
    expected = TRIAL / f"source-{ordinal:02d}/bind-00/output"
    require(
        type(ordinal) is int
        and 1 <= ordinal <= 15
        and argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"]
        and argv[5:] == [str(expected), "0"],
        "Only exact explicit acquisition stage permitted",
    )
    routed = [*argv[:3], "scripts.grid_acquisition_continuation_v135", "acquire", str(expected), str(ordinal)]
    return subprocess.Popen(routed, **kwargs)


def run(ordinal):
    require(type(ordinal) is int and 1 <= ordinal <= 15, "One explicit original archive ordinal required")
    stage = TRIAL / f"source-{ordinal:02d}"
    stage.mkdir(exist_ok=True)
    proxy = types.SimpleNamespace(
        Popen=lambda argv, **kw: routed_popen(ordinal, argv, **kw),
        STDOUT=subprocess.STDOUT,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    scoped(
        core.runtime.run,
        {
            "TRIAL": stage,
            "verify": verify,
            "prerequisites": lambda c, b: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("bind", 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "acquire"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "No extra freeze arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One original ordinal required")
        run(int(args.arguments[0]))
    else:
        verify()
        require(
            len(args.arguments) == 2 and args.arguments[1] in {str(i) for i in range(1, 16)},
            "Exact source and output required",
        )
        ordinal = int(args.arguments[1])
        output = Path(args.arguments[0])
        require(
            output.resolve() == (TRIAL / f"source-{ordinal:02d}/bind-00/output").resolve(),
            "Exact guarded output required",
        )
        output.mkdir(exist_ok=False)
        acquire(ordinal, output)
