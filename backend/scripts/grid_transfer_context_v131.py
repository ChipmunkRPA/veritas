"""Record-bound publisher URLs in a new, isolated GRID acquisition context."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import types
from pathlib import Path
from urllib.parse import urlparse

from scripts import acquire_grid_transfer_v130 as original

TRIAL = original.BASE / "grid-transfer-acquisition-v131"
CACHE = original.runtime.ARCHIVE / "grid-transfer-sources-v131"
PUBLIC = original.PANEL / "grid-transfer-acquisition-freeze-v131.json"
CODE = original.CODE + (
    "backend/scripts/grid_transfer_context_v131.py",
    "backend/tests/test_grid_transfer_context_v131.py",
    "docs/grid-transfer-context-v131-method.md",
)


def select(record):
    original.require(
        record["id"] == 3625687
        and record["metadata"]["license"]["id"] == "cc-by-4.0"
        and record["metadata"]["access_right"] == "open",
        "Exact open CC-BY-4.0 publisher record required",
    )

    def asset(item, maximum):
        parsed = urlparse(item["links"]["self"])
        original.require(
            parsed.scheme == "https"
            and parsed.netloc == "zenodo.org"
            and parsed.path == f"/api/records/3625687/files/{item['key']}/content"
            and not parsed.query
            and not parsed.fragment,
            "Exact record/file-bound publisher content URL required",
        )
        original.require(
            type(item["size"]) is int
            and 0 < item["size"] <= maximum
            and re.fullmatch(r"md5:[0-9a-f]{32}", item["checksum"]) is not None,
            "Bounded publisher asset size and checksum required",
        )
        return {
            "publisher_name": item["key"],
            "url": item["links"]["self"],
            "bytes": item["size"],
            "md5": item["checksum"][4:],
        }

    candidates = [
        asset(item, 600 * 1024**2)
        for item in record["files"]
        if re.fullmatch(r"s(?:[1-9]|[12][0-9]|3[0-4])\.zip", item["key"])
    ]
    original.require(
        len(candidates) == 33 and len({r["publisher_name"] for r in candidates}) == 33,
        "All 33 available source archives required",
    )
    ordered = sorted(
        candidates,
        key=lambda r: hashlib.sha256(
            ("chipmunk.grid-transfer/130:" + r["publisher_name"]).encode()
        ).hexdigest(),
    )
    papers = [item for item in record["files"] if item["key"] == "jasagrid.pdf"]
    original.require(len(papers) == 1, "One original publisher documentation asset required")
    return {
        "selected": [dict(row, ordinal=i) for i, row in enumerate(ordered[:16])],
        "paper": asset(papers[0], 256 * 1024),
        "selection": "Fixed SHA-256 archive order; sixteen source groups; no media/face/detector response chooses sources.",
        "future_clip_recipe": "Two real .mpg entries per source, excluding resource forks; fixed SHA order with globally unique utterance stems, frozen before any model use. Unsupported clips remain unavailable, never substituted.",
        "source_license": "CC-BY-4.0",
        "purpose": "Frozen-head external transfer research, not serving-model training or customer uploads.",
    }


def context():
    scope = {
        **vars(original),
        "TRIAL": TRIAL,
        "CACHE": CACHE,
        "PUBLIC": PUBLIC,
        "CODE": CODE,
        "select": select,
        "__file__": str(Path(__file__)),
    }
    for name in ("identity", "verify", "run", "acquire"):
        fn = getattr(original, name)
        scope[name] = types.FunctionType(fn.__code__, scope, fn.__name__, fn.__defaults__, fn.__closure__)
    base_identity = scope["identity"]

    def identity():
        failure = original.TRIAL / "prefreeze-failure"
        receipt = original.read(failure / "result.json")
        original.require(
            receipt["status"] == "prefreeze_preparation_failed"
            and receipt["new_neural_calls"] == 0
            and not receipt["media_downloaded"]
            and not receipt["guarded_stage_attempted"]
            and receipt["record_sha256"] == original.RECORD_SHA,
            "Preserved preparation failure before any download/model stage required",
        )
        for relative, expected in receipt["original_code_sha256"].items():
            original.require(
                original.sha(original.ROOT / relative)
                == original.sha(failure / Path(relative).name)
                == expected,
                "Original preparation code and evidence must remain unchanged",
            )
        body = base_identity()
        body["execution_context_schema"] = "chipmunk.grid-transfer-record-url-context/131"
        body["preserved_prefreeze_failure_sha256"] = original.sha(failure / "result.json")
        body["context_change"] = (
            "Exact record/file content URL namespace only; same source selection, checksums, counts, monitor and limits."
        )
        return body

    def routed_popen(argv, **kwargs):
        original.require(
            argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"],
            "Only fixed data acquisition permitted",
        )
        routed = list(argv)
        routed[3] = "scripts.grid_transfer_context_v131"
        return subprocess.Popen(routed, **kwargs)

    scope["identity"], scope["routed_popen"] = identity, routed_popen
    return scope


def freeze():
    original.require(
        original.sha(original.METADATA / "grid-record.json") == original.RECORD_SHA,
        "Pinned open publisher metadata required",
    )
    plan = select(original.read(original.METADATA / "grid-record.json"))
    original.monitor_verified()
    TRIAL.mkdir(exist_ok=False)
    original.write(TRIAL / "plan.json", plan)
    body = context()["identity"]()
    original.write(TRIAL / "freeze.json", body)
    original.write(PUBLIC, body)
    print('{"status":"frozen","source_archives":16,"new_neural_calls":0}', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "bind"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "freeze":
        original.require(not args.arguments, "Freeze takes no arguments")
        freeze()
    elif args.command == "run":
        original.require(len(args.arguments) == 1, "One fixed batch required")
        context()["run"](int(args.arguments[0]))
    else:
        scope = context()
        scope["verify"]()
        original.require(
            len(args.arguments) == 2 and args.arguments[1] in {str(i) for i in range(8)},
            "Fixed guarded output and batch required",
        )
        batch = int(args.arguments[1])
        output = Path(args.arguments[0])
        original.require(
            output.resolve() == (TRIAL / f"bind-{batch:02d}/output").resolve(),
            "Exact exclusive guarded output required",
        )
        output.mkdir(exist_ok=False)
        scope["acquire"](batch, output)
