"""Lock an independent publisher's generated-video benchmark subset."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile

from prepare_video_upgrade_eval import MAX_BYTES, prepare, sha

POLICY = "chipmunk-video-genbuster/1"
REVISION = "7f184bef604882227bf2a585608d3e47e4f0b60b"
ARCHIVE_HASH = "b203126f81715d4338ab21f784791eda359a955acb32c4f1c9f2c353ba553c71"
CARD = f"https://huggingface.co/datasets/l8cv/GenBuster-Bench-plusplus/tree/{REVISION}"
ARCHIVE = f"https://huggingface.co/datasets/l8cv/GenBuster-Bench-plusplus/resolve/{REVISION}/GenBuster-Bench-plusplus.zip"
ENTRY = re.compile(r"video/(fake|real)/([0-9a-f]{64})\.mp4")


def acquire(path: Path, output: Path) -> None:
    if sha(path) != ARCHIVE_HASH:
        raise ValueError("unexpected GenBuster archive SHA-256")
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("originals", "matched"):
        (output / folder).mkdir(exist_ok=True)
    lock = output / "manifest.json"
    if lock.exists():
        raise ValueError("a manifest already exists, choose a new output directory")
    items = []
    with ZipFile(path) as archive:
        videos = [info for info in archive.infolist() if info.filename.startswith("video/") and info.filename.endswith(".mp4")]
        indexed = [info for info in videos if ENTRY.fullmatch(info.filename)]
        excluded = [info.filename for info in videos if not ENTRY.fullmatch(info.filename)]
        if len(videos) != 2000 or len(indexed) < 72:
            raise ValueError(f"unexpected archive inventory: {len(videos)} videos, {len(indexed)} hash-named")
        for label, name in ((0, "real"), (1, "fake")):
            selected = sorted(
                [info for info in indexed if info.filename.startswith(f"video/{name}/")],
                key=lambda info: hashlib.sha256(f"{POLICY}:{info.filename}".encode()).digest(),
            )[:36]
            for index, info in enumerate(selected):
                identity = Path(info.filename).stem
                item_id = f"genbuster-{name}-{index:02}"
                original = output / "originals" / f"{item_id}.mp4"
                if info.file_size > MAX_BYTES:
                    raise ValueError(f"archive member exceeds the source size cap: {info.filename}")
                with archive.open(info) as source, original.open("xb") as target:
                    shutil.copyfileobj(source, target)
                if sha(original) != identity:
                    raise ValueError(f"archive member does not match the source commitment: {info.filename}")
                row = prepare({
                    "id": item_id,
                    "label": label,
                    "family": "publisher_generated" if label else "publisher_real",
                    "split": "calibration" if index < 12 else "test",
                    "group": f"GenBuster:{identity}",
                    "source_path": info.filename,
                    "source_url": CARD,
                    "download_url": ARCHIVE,
                    "license": "Publisher repository MIT tag",
                }, output)
                items.append(row)
                print(f"{item_id}: {row['status']}", flush=True)
    if len({row["group"] for row in items}) != len(items):
        raise ValueError("duplicated source across the fixed selection")
    body = {
        "policy": POLICY,
        "revision": REVISION,
        "archive_sha256": ARCHIVE_HASH,
        "acquisition_script_sha256": sha(Path(__file__)),
        "preparation_sha256": sha(Path(__file__).with_name("prepare_video_upgrade_eval.py")),
        "ffmpeg_version": subprocess.check_output(
            ["ffmpeg", "-version"], text=True, timeout=10
        ).splitlines()[0],
        "source_group_scope": "Exact assets only. The release does not expose per-file generators or parent scenes.",
        "excluded_without_filename_hash": excluded,
        "items": items,
    }
    with lock.open("x") as file:
        file.write(json.dumps(body, indent=2) + "\n")
    print(f"manifest SHA-256 {sha(lock)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    acquire(args.archive, args.output)
