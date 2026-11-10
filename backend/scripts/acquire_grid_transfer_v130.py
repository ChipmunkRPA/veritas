"""Fixed, checksum-verified fresh GRID source archives; no model/decoder calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import types
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT = runtime.BACKEND.parent
BASE = ROOT / "backend/data/independent-evaluation"
PANEL = ROOT / "docs/evaluation-panels"
METADATA = BASE / "fresh-av-source-v130-metadata"
TRIAL = BASE / "grid-transfer-acquisition-v130"
CACHE = runtime.ARCHIVE / "grid-transfer-sources-v130"
PUBLIC = PANEL / "grid-transfer-acquisition-freeze-v130.json"
RECORD_SHA = "3581a251fdc5f4237e4b9357eaa615122ed720542d5dd5e5e56b99408ff06709"
CODE = (
    "backend/scripts/acquire_grid_transfer_v130.py",
    "backend/tests/test_acquire_grid_transfer_v130.py",
    "docs/grid-transfer-acquisition-v130-method.md",
)
read, write, sha, require = runtime.read, runtime.write, runtime.sha, runtime.require


def select(record):
    require(
        record["id"] == 3625687
        and record["metadata"]["license"]["id"] == "cc-by-4.0"
        and record["metadata"]["access_right"] == "open",
        "Exact open CC-BY-4.0 publisher record required",
    )
    candidates = []
    for item in record["files"]:
        if re.fullmatch(r"s(?:[1-9]|[12][0-9]|3[0-4])\.zip", item["key"]):
            require(
                type(item["size"]) is int
                and 0 < item["size"] <= 600 * 1024**2
                and re.fullmatch(r"md5:[0-9a-f]{32}", item["checksum"]) is not None,
                "Bounded archive size and publisher checksum required",
            )
            url = item["links"]["self"]
            parsed = urlparse(url)
            require(
                parsed.scheme == "https"
                and parsed.hostname == "zenodo.org"
                and parsed.path.startswith("/api/files/"),
                "Publisher archive link required",
            )
            candidates.append(
                {
                    "publisher_name": item["key"],
                    "url": url,
                    "bytes": item["size"],
                    "md5": item["checksum"][4:],
                }
            )
    require(
        len(candidates) == 33 and len({r["publisher_name"] for r in candidates}) == 33,
        "All 33 available source archives required",
    )
    ordered = sorted(
        candidates,
        key=lambda r: hashlib.sha256(
            ("chipmunk.grid-transfer/130:" + r["publisher_name"]).encode()
        ).hexdigest(),
    )
    paper = next(item for item in record["files"] if item["key"] == "jasagrid.pdf")
    require(
        0 < paper["size"] <= 256 * 1024 and re.fullmatch(r"md5:[0-9a-f]{32}", paper["checksum"]) is not None,
        "Original publisher documentation identity required",
    )
    return {
        "selected": [dict(row, ordinal=i) for i, row in enumerate(ordered[:16])],
        "paper": {
            "publisher_name": paper["key"],
            "url": paper["links"]["self"],
            "bytes": paper["size"],
            "md5": paper["checksum"][4:],
        },
        "selection": "Fixed SHA-256 archive order; sixteen source groups; no media/face/detector response chooses sources.",
        "future_clip_recipe": "Two real .mpg entries per source, excluding resource forks; fixed SHA order with globally unique utterance stems, frozen before any model use. Unsupported clips remain unavailable, never substituted.",
        "source_license": "CC-BY-4.0",
        "purpose": "Frozen-head external transfer research, not serving-model training or customer uploads.",
    }


def inspect_archive(path):
    with zipfile.ZipFile(path) as archive:
        all_entries = archive.infolist()
        require(
            len(all_entries) <= 5000 and len({r.filename for r in all_entries}) == len(all_entries),
            "Bounded unique publisher archive directory required",
        )
        media = []
        for entry in all_entries:
            parts = PurePosixPath(entry.filename).parts
            require(
                not PurePosixPath(entry.filename).is_absolute()
                and ".." not in parts
                and "\\" not in entry.filename,
                "Unsafe archive member path",
            )
            if "__MACOSX" in parts or PurePosixPath(entry.filename).name.startswith("._"):
                continue
            if entry.filename.lower().endswith(".mpg"):
                require(
                    not entry.flag_bits & 1
                    and entry.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    and 1000 <= entry.file_size <= 2 * 1024**2,
                    "Bounded real video member required",
                )
                media.append({"name": entry.filename, "bytes": entry.file_size, "crc32": entry.CRC})
        require(len(media) == 1000, "All one thousand real videos required; ignore Apple resource forks")
    return media


def monitor_verified():
    folder = runtime.TRIAL / "monitor"
    r, p = read(folder / "output/result.json"), read(folder / "profile.json")
    require(
        r["status"] == "complete"
        and r["owned_descendant_sampled"]
        and r["sample_helper_sha256"] == sha(ROOT / "backend/scripts/audio_tree_sampler_v64.py")
        and p["child_exit_code"] == 0
        and p["guard"] is None
        and p["sample_errors"] == 0
        and p["worker_receipt_sha256"] == sha(folder / "output/result.json"),
        "Successful exact owned-descendant preflight required",
    )


def identity():
    monitor_verified()
    require(sha(METADATA / "grid-record.json") == RECORD_SHA, "Exact publisher metadata snapshot required")
    body = select(read(METADATA / "grid-record.json"))
    require(read(TRIAL / "plan.json") == body, "Unchanged fixed source acquisition plan required")
    return {
        "schema": "chipmunk.grid-transfer-acquisition-freeze/130",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "publisher_record_url": "https://zenodo.org/records/3625687",
        "publisher_record_sha256": RECORD_SHA,
        "private_plan_sha256": sha(TRIAL / "plan.json"),
        "license": "CC-BY-4.0",
        "available_source_archives": 33,
        "selected_source_archives": 16,
        "planned_archive_bytes": sum(r["bytes"] for r in body["selected"]),
        "batch_count": 8,
        "archives_per_batch": 2,
        "scope": "One archive at a time; full publisher byte-count/MD5 and observed SHA-256, safe directory inventory only. No clip selected by content, decoding, inference, fitting or serving contribution. Individual names/checksums/media stay private.",
    }


def freeze():
    require(sha(METADATA / "grid-record.json") == RECORD_SHA, "Pinned open publisher metadata required")
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "plan.json", select(read(METADATA / "grid-record.json")))
    body = identity()
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "source_archives": 16, "new_neural_calls": 0}), flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Unchanged source archive procedure required",
    )


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"],
        "Only fixed data acquisition permitted",
    )
    args = list(argv)
    args[3] = "scripts.acquire_grid_transfer_v130"
    return subprocess.Popen(args, **kwargs)


def run(batch):
    require(type(batch) is int and 0 <= batch < 8, "One fixed source archive batch required")
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda c, b: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("bind", batch)


def download(row, path):
    total = 0
    md5 = hashlib.md5()
    digest = hashlib.sha256()
    with path.open("xb") as output:
        with urlopen(
            Request(
                row["url"],
                headers={"User-Agent": "Chipmunk-Verify-transfer-evaluation", "Accept-Encoding": "identity"},
            ),
            timeout=30,
        ) as response:
            require(
                response.status == 200,
                "Complete publisher asset response required; no partial/range substitution",
            )
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                require(total <= row["bytes"], "Publisher asset exceeds pinned size")
                output.write(chunk)
                md5.update(chunk)
                digest.update(chunk)
    require(
        total == row["bytes"] and md5.hexdigest() == row["md5"],
        "Full publisher archive checksum/size mismatch",
    )
    return {"file": str(path), "bytes": total, "md5": md5.hexdigest(), "sha256": digest.hexdigest()}


def acquire(batch, output):
    require(type(batch) is int and 0 <= batch < 8, "One fixed source archive batch required")
    plan = read(TRIAL / "plan.json")
    folder = CACHE / f"batch-{batch:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    if batch == 0:
        permissions = CACHE / "permissions"
        permissions.mkdir(exist_ok=False)
        paper = download(plan["paper"], permissions / "jasagrid.pdf")
        with (permissions / "publisher-record.json").open("xb") as stream:
            stream.write((METADATA / "grid-record.json").read_bytes())
        write(
            permissions / "receipt.json",
            {"paper": paper, "publisher_record_sha256": RECORD_SHA, "license": "CC-BY-4.0"},
        )
    records = []
    with (output / "archives.jsonl").open("x") as journal:
        for row in plan["selected"][batch * 2 : (batch + 1) * 2]:
            observed = download(row, folder / f"source-{row['ordinal']:02d}.zip")
            media = inspect_archive(Path(observed["file"]))
            record = {**row, **observed, "status": "publisher_archive_verified", "real_video_members": media}
            records.append(record)
            journal.write(json.dumps(record, allow_nan=False) + "\n")
            journal.flush()
            print(
                json.dumps({"archive_completed_ordinal": row["ordinal"], "real_video_members": len(media)}),
                flush=True,
            )
    result = {
        "schema": "chipmunk.grid-transfer-acquisition-result/130",
        "status": "complete",
        "batch": batch,
        "archives": len(records),
        "verified_bytes": sum(r["bytes"] for r in records),
        "real_video_members_inventoried": sum(len(r["real_video_members"]) for r in records),
        "private_archive_journal_sha256": sha(output / "archives.jsonl"),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "new_neural_calls": 0,
        "defaults_changed": False,
        "scope": "Full expected publisher MD5/size, observed SHA-256 and safe archive inventory; not decoding, genuine/AI detection, source-identity proof or validation accuracy. No raw media or individual fingerprints are published.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze", "run", "bind"))
    p.add_argument("arguments", nargs="*")
    args = p.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "Freeze takes no arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One batch required")
        run(int(args.arguments[0]))
    else:
        verify()
        require(
            len(args.arguments) == 2 and args.arguments[1] in {str(i) for i in range(8)},
            "Fixed output and batch required",
        )
        output = Path(args.arguments[0])
        require(output.resolve().is_relative_to(TRIAL.resolve()), "Exclusive private output required")
        output.mkdir(exist_ok=False)
        acquire(int(args.arguments[1]), output)
