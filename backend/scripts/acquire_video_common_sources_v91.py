"""Bounded, digest-verified acquisition of the fixed v89 video source plan.

Data-only: no decoder, model, generator, customer input, or remote Python code.
Per-read observations are not continuous or guaranteed peak measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from urllib.request import Request, urlopen

import psutil

from scripts.video_common_sources_v89 import CREMA_REV, RTFS_REV, read, require, sha, write

BACKEND = Path(__file__).resolve().parents[1]
SOURCE = BACKEND / "data/independent-evaluation/video-common-sources-v89"
TRIAL = BACKEND / "data/independent-evaluation/video-common-acquisition-v91"
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")
MEDIA = ARCHIVE / "video-common-sources-v91"
PLAN_SHA = "c48ffa31a547df984b57f0253c60ae55a612d6081387e25c63ae0af8d40381ff"
BATCH = 48
MAX_ASSET = 80 * 1024**2
METADATA_LIMIT = 8 * 1024**2
RESERVE = 1536 * 1024**2


class Budget:
    def __init__(self, folder):
        self.folder = folder
        self.started = time.monotonic()
        self.observations = []
        self.stream = None

    def check(self, initial=False):
        row = {"elapsed_s": time.monotonic() - self.started, "rss_bytes": psutil.Process().memory_info().rss,
               "available_ram_bytes": psutil.virtual_memory().available,
               "project_free_disk_bytes": shutil.disk_usage(BACKEND).free,
               "archive_free_disk_bytes": shutil.disk_usage(ARCHIVE).free}
        self.observations.append(row)
        if self.stream is not None:
            self.stream.write(json.dumps(row) + "\n")
            self.stream.flush()
        require(Path("/Volumes/SEAGATE").is_mount(), "Required archive volume unavailable")
        require(row["elapsed_s"] <= 900 and row["rss_bytes"] <= 256 * 1024**2
                and row["available_ram_bytes"] >= (2 * 1024**3 if initial else RESERVE)
                and min(row["project_free_disk_bytes"], row["archive_free_disk_bytes"]) >= (3 * 1024**3 if initial else RESERVE),
                "Data-only resource budget exceeded")


def fetch(url, budget, limit=METADATA_LIMIT):
    budget.check()
    with urlopen(Request(url, headers={"User-Agent": "Chipmunk-Verify-source-acquisition"}), timeout=30) as response:
        data = response.read(limit + 1)
    require(0 < len(data) <= limit, "Unexpected metadata payload length")
    budget.check()
    return data


def lfs_pointer(data):
    text = data.decode("ascii")
    require(text.startswith("version https://git-lfs.github.com/spec/v1\n"), "Expected immutable Git LFS pointer")
    digest, length = re.search(r"^oid sha256:([0-9a-f]{64})$", text, re.M), re.search(r"^size ([0-9]+)$", text, re.M)
    require(digest is not None and length is not None, "Complete LFS object identity required")
    size = int(length[1])
    require(0 < size <= MAX_ASSET, "Source asset outside acquisition size limit")
    return digest[1], size


def hf_object(row):
    lfs = row.get("lfs") or {}
    size, digest = row.get("size"), lfs.get("sha256")
    require(type(size) is int and 0 < size <= MAX_ASSET and lfs.get("size") == size
            and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "Publisher LFS size/digest missing or inconsistent")
    return digest, size


def prepare(budget):
    require(sha(SOURCE / "plan.json") == PLAN_SHA == read(SOURCE / "summary.json")["private_plan_sha256"],
            "Exact private v89 source plan required")
    source = read(SOURCE / "plan.json")
    require(sha(BACKEND / "scripts/video_common_sources_v89.py") == source["script_sha256"], "Source planner changed")
    require(not (TRIAL / "asset-plan.json").exists() and not MEDIA.exists(), "Preserve existing/partial acquisitions")
    info_bytes = fetch(f"https://huggingface.co/api/datasets/godmodes/rtfs-10k/revision/{RTFS_REV}?blobs=true", budget)
    info = json.loads(info_bytes)
    require(info["sha"] == RTFS_REV, "Pinned dataset revision changed")
    siblings = {row["rfilename"]: row for row in info["siblings"]}
    require(len(siblings) == len(info["siblings"]), "Unique publisher file descriptors required")
    with (TRIAL / "publisher-file-info.json").open("xb") as stream:
        stream.write(info_bytes)
    pointer_folder = TRIAL / "pointers"
    pointer_folder.mkdir()
    assets = []
    for group in source["groups"]:
        for asset in group["assets"]:
            index = len(assets)
            if group["source_family"] == "paired_face_swap":
                require(asset["source_path"] in siblings and f"/resolve/{RTFS_REV}/" in asset["url"], "Pinned source file missing")
                digest, size = hf_object(siblings[asset["source_path"]])
                suffix = ".mp4"
            else:
                require(group["source_family"] == "recorded_audiovisual_speech" and CREMA_REV in asset["url"]
                        and CREMA_REV in asset["pointer_url"], "Pinned audiovisual source required")
                pointer = fetch(asset["pointer_url"], budget, 4096)
                digest, size = lfs_pointer(pointer)
                with (pointer_folder / f"asset-{index:03d}.txt").open("xb") as stream:
                    stream.write(pointer)
                suffix = ".flv"
            assets.append({"index": index, "group": group["group"], "provisional_role": group["provisional_role"],
                "source_family": group["source_family"], "kind": asset["kind"], "url": asset["url"],
                "source_path": asset["source_path"], "file": f"asset-{index:03d}" + suffix, "sha256": digest, "bytes": size})
    require(len(assets) == 336 and len({row["file"] for row in assets}) == 336, "All planned source assets required")
    budget.check()
    MEDIA.mkdir()
    body = {"schema": "chipmunk.video-common-asset-plan/91", "private_source_plan_sha256": PLAN_SHA,
            "publisher_file_info_sha256": sha(TRIAL / "publisher-file-info.json"), "assets": assets,
            "implementation_sha256": sha(Path(__file__)), "media_root": str(MEDIA), "batch_size": BATCH,
            "planned_assets": 336, "planned_bytes": sum(row["bytes"] for row in assets),
            "neural_calls": 0, "roles_provisional": True}
    write(TRIAL / "asset-plan.json", body)
    write(TRIAL / "preparation-summary.json", {key: value for key, value in body.items() if key not in {"assets", "media_root"}} |
          {"asset_plan_sha256": sha(TRIAL / "asset-plan.json")})
    print(json.dumps({"publisher_digest_records_prepared": len(assets), "planned_bytes": body["planned_bytes"], "neural_calls": 0}), flush=True)


def download(asset, destination, budget):
    require(destination.name == asset["file"] and destination.suffix in {".mp4", ".flv"}, "Bound destination required")
    budget.check()
    digest, total = hashlib.sha256(), 0
    # Exclusive creation: a partial/failed file is retained and never silently reused.
    with destination.open("xb") as output:
        with urlopen(Request(asset["url"], headers={"User-Agent": "Chipmunk-Verify-source-acquisition"}), timeout=30) as response:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                require(total <= asset["bytes"], "Source exceeds publisher byte count")
                digest.update(chunk)
                output.write(chunk)
                budget.check()
    require(total == asset["bytes"] and digest.hexdigest() == asset["sha256"], "Publisher source digest/size mismatch")
    budget.check()
    return {"index": asset["index"], "bytes": total, "sha256": digest.hexdigest(), "status": "byte_verified"}


def acquire(index, budget):
    require(type(index) is int and 0 <= index < 7, "One explicit fixed acquisition batch required")
    summary, plan = read(TRIAL / "preparation-summary.json"), read(TRIAL / "asset-plan.json")
    require(sha(TRIAL / "asset-plan.json") == summary["asset_plan_sha256"]
            and plan["implementation_sha256"] == sha(Path(__file__)) and plan["private_source_plan_sha256"] == PLAN_SHA
            and plan["media_root"] == str(MEDIA), "Fixed acquisition method and source identities required")
    selected = plan["assets"][index * BATCH:(index + 1) * BATCH]
    require(len(selected) == BATCH, "Complete fixed batch required")
    records = TRIAL / f"batch-{index:02d}-assets.jsonl"
    with records.open("x") as journal:
        for asset in selected:
            result = download(asset, MEDIA / asset["file"], budget)
            journal.write(json.dumps(result) + "\n")
            journal.flush()
            print(json.dumps({"batch": index, "asset_completed": asset["index"]}), flush=True)
    write(TRIAL / f"batch-{index:02d}-complete.json", {"batch": index, "completed": BATCH,
        "bytes": sum(row["bytes"] for row in selected), "asset_journal_sha256": sha(records),
        "scope": "Publisher byte-size/SHA-256 verification only; media decoding, source-role qualification and detection remain pending."})


def run(command, index=None):
    require((command == "prepare" and index is None) or (command == "acquire" and type(index) is int and 0 <= index < 7),
            "One explicit data-only phase required")
    folder = TRIAL / ("prepare-run" if command == "prepare" else f"batch-{index:02d}-run")
    folder.mkdir(parents=True, exist_ok=False)
    budget, status, error = Budget(folder), "failed", None
    try:
        with (folder / "resources.jsonl").open("x") as budget.stream:
            budget.check(initial=True)
            prepare(budget) if command == "prepare" else acquire(index, budget)
            budget.check()
            status = "complete"
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        rows = budget.observations
        write(folder / "profile.json", {"status": status, "error_type": error, "duration_s": time.monotonic() - budget.started,
            "maximum_observed_process_rss_bytes": max((row["rss_bytes"] for row in rows), default=0),
            "minimum_observed_available_ram_bytes": min((row["available_ram_bytes"] for row in rows), default=None),
            "observations": len(rows), "implementation_sha256": sha(Path(__file__)), "new_neural_calls": 0,
            "limits": {"rss_bytes": 256 * 1024**2, "initial_available_ram_bytes": 2 * 1024**3,
                       "ram_and_disk_reserve_bytes": RESERVE, "initial_free_disk_bytes": 3 * 1024**3, "checked_seconds": 900},
            "scope": "Single-process data-only per-request/per-read observations with 30-second network timeouts; not continuous sampling, a peak guarantee or a hard preemption guarantee. No subprocess, decoder, neural call, retry or overwrite. Primary neural budgets unchanged."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "acquire"))
    parser.add_argument("index", nargs="?", type=int)
    args = parser.parse_args()
    run(args.command, args.index)
