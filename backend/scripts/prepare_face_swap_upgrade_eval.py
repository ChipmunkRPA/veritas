"""Lock paired RTFS-10K face-swap cases with shared original targets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prepare_video_upgrade_eval import download, prepare, sha

REVISION = "bca865e675852666a426368b2f5614abf1c1926a"
REPO = "https://huggingface.co/datasets/godmodes/rtfs-10k"
CARD = f"{REPO}/blob/{REVISION}/README.md"
POLICY = "chipmunk-video-face-swap/1"
FAMILIES = ("inswapper", "uniface")
METADATA = ("videos.csv", *[f"{family}.csv" for family in FAMILIES])


def read_rows(source: Path, name: str) -> list[dict]:
    with (source / name).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def order(value: str) -> bytes:
    return hashlib.sha256(f"{POLICY}:{value}".encode()).digest()


def plan(source: Path, groups: int = 24, prior_items: tuple[dict, ...] = ()) -> list[dict]:
    original = {row["filename"]: row for row in read_rows(source, "videos.csv")}
    generated: dict[str, dict[str, list[dict]]] = {family: defaultdict(list) for family in FAMILIES}
    for family in FAMILIES:
        for row in read_rows(source, f"{family}.csv"):
            if (
                row["success"] == "True"
                and not row["additional_features"].strip()
                and "child" not in row["photo_tags"].casefold()
                and "child" not in row["video_tags"].casefold()
            ):
                generated[family][row["video"]].append(row)
    seen_photos = {row["donor_filename"] for row in prior_items if row["donor_filename"]}
    seen_youtube = {row["group"].removeprefix("RTFS:") for row in prior_items}
    chosen = 0
    rows = []
    for target in sorted(original, key=order):
        record = original[target]
        if record["id"] in seen_youtube or "child" in record["tags"].casefold():
            continue
        selected = []
        for family in FAMILIES:
            matches = sorted(generated[family][target], key=lambda row: order(row["filename"]))
            available = [row for row in matches if row["photo"] not in seen_photos]
            if not available:
                break
            selected.append(available[0])
        if len(selected) != len(FAMILIES) or len({row["photo"] for row in selected}) != len(FAMILIES):
            continue
        seen_photos.update(row["photo"] for row in selected)
        seen_youtube.add(record["id"])
        chosen += 1
        split = "calibration" if not prior_items and chosen <= groups // 2 else "test"
        for family, source_file, donor in (
            ("original_videos", target, None),
            *((row["generator"], row["filename"], row["photo"]) for row in selected),
        ):
            if Path(source_file).name != source_file:
                raise ValueError(f"invalid source filename: {source_file}")
            source_path = f"{family}/{source_file}"
            rows.append(
                {
                    "id": f"rtfs-{len(seen_youtube):02}-{family}",
                    "label": int(donor is not None),
                    "family": family,
                    "split": split,
                    "group": f"RTFS:{record['id']}",
                    "source_path": source_path,
                    "source_url": CARD,
                    "source_video_url": record["original_link"],
                    "target_start_seconds": float(record["start_time"]),
                    "target_end_seconds": float(record["end_time"]),
                    "donor_filename": donor,
                    "download_url": f"{REPO}/resolve/{REVISION}/{source_path}",
                    "license": "Dataset publisher CC BY-SA 4.0",
                }
            )
        if chosen == groups:
            return rows
    raise ValueError("insufficient eligible paired groups")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--groups", type=int, default=24)
    parser.add_argument("--exclude-manifest", type=Path, help="Exclude these original sources and donors; reserve new cases for test")
    args = parser.parse_args()
    if args.groups < 2:
        parser.error("at least two source groups are required")
    prior_items = tuple(json.loads(args.exclude_manifest.read_text())["items"]) if args.exclude_manifest else ()
    for subfolder in ("metadata", "originals", "matched"):
        (args.output / subfolder).mkdir(parents=True, exist_ok=True)
    lock = args.output / "manifest.json"
    if lock.exists():
        parser.error("a manifest already exists, choose a new output directory")
    source = args.output / "metadata"
    for filename in METADATA:
        download(f"{REPO}/resolve/{REVISION}/{filename}", source / filename)
    with ThreadPoolExecutor(max_workers=4) as executor:
        items = []
        seen: dict[str, str] = {}
        for record in executor.map(lambda row: prepare(row, args.output), plan(source, args.groups, prior_items)):
            if "original" in record:
                digest = record["original"]["sha256"]
                prior = seen.setdefault(digest, record["id"])
                if prior != record["id"]:
                    record.update(status="duplicate", reason=f"same source bytes as {prior}")
            items.append(record)
            print(f"{record['id']}: {record['status']}", flush=True)
    body = {
        "policy": POLICY,
        "revision": REVISION,
        "excluded_manifest_sha256": sha(args.exclude_manifest) if args.exclude_manifest else None,
        "acquisition_script_sha256": sha(Path(__file__)),
        "preparation_sha256": sha(Path(__file__).with_name("prepare_video_upgrade_eval.py")),
        "metadata_sha256": {filename: sha(source / filename) for filename in METADATA},
        "ffmpeg_version": subprocess.check_output(
            ["ffmpeg", "-version"], text=True, timeout=10
        ).splitlines()[0],
        "items": items,
    }
    with lock.open("x") as file:
        file.write(json.dumps(body, indent=2) + "\n")
    print(f"manifest SHA-256 {sha(lock)}", flush=True)


if __name__ == "__main__":
    main()
