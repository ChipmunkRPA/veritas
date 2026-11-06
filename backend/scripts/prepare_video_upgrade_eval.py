"""Acquire and lock fresh DeepAction pairs for the offline video comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REVISION = "476932c3369424ca59dc1f8fb758cb3c13fe63fe"
SOURCE = f"https://huggingface.co/datasets/faridlab/deepaction_v1/resolve/{REVISION}"
CARD = f"https://huggingface.co/datasets/faridlab/deepaction_v1/blob/{REVISION}/README.md"
POLICY = "chipmunk-video-deepaction/1"
PRIOR_ACTIONS = {29, 40, 54, 85}
CAL_GENERATORS = ("CogVideoX5B", "RunwayML", "Veo")
TEST_GENERATORS = ("BDAnimateDiffLightning", "StableDiffusion", "VideoPoet")
MAX_BYTES = 160 * 1024 * 1024


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plan() -> list[dict]:
    actions = sorted(
        set(range(1, 101)) - PRIOR_ACTIONS,
        key=lambda action: hashlib.sha256(f"{POLICY}:{action}".encode()).digest(),
    )
    rows = []
    for split, groups, generators in (
        ("calibration", actions[:12], CAL_GENERATORS),
        ("test", actions[12:36], TEST_GENERATORS),
    ):
        for i, action in enumerate(groups):
            for label, family in ((0, "Pexels"), (1, generators[i % len(generators)])):
                source_path = f"{family}/{action}/a.mp4"
                rows.append(
                    dict(
                        id=f"deepaction-{action:03}-{family}",
                        label=label,
                        family=family,
                        split=split,
                        group=f"DeepAction:action-{action}",
                        source_path=source_path,
                        source_url=CARD,
                        download_url=f"{SOURCE}/{source_path}",
                        license="CC BY 4.0" if label else "Pexels License",
                    )
                )
    return rows


def download(url: str, path: Path) -> None:
    if path.exists():
        return
    partial = path.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=45) as response, partial.open("wb") as file:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError(f"download exceeds {MAX_BYTES} bytes")
                file.write(chunk)
            if total == 0:
                raise ValueError("empty download")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def media_info(path: Path) -> dict:
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "stream=codec_type,codec_name,width,height,avg_frame_rate:format=duration",
                "-of", "json", str(path),
            ],
            timeout=30,
        )
    )


def prepare(row: dict, output: Path) -> dict:
    original = output / "originals" / f"{row['id']}.mp4"
    target = output / "matched" / original.name
    # Re-encode both classes with identical rules to expose codec shortcuts.
    transcode = [
        "-map", "0:v:0", "-map_metadata", "-1", "-an", "-sn", "-dn", "-t", "6",
        "-vf", "scale=640:480:force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "30",
        "-threads", "2",
    ]
    try:
        download(row["download_url"], original)
        row = {
            **row,
            "original": {
                "path": str(original.relative_to(output)),
                "sha256": sha(original),
                "bytes": original.stat().st_size,
                "media": media_info(original),
            },
        }
        if not target.exists():
            partial = target.with_suffix(".part.mp4")
            try:
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                     "-i", str(original), *transcode, str(partial)],
                    check=True, timeout=180, capture_output=True,
                )
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
        return {
            **row,
            "status": "ready",
            "matched": {
                "path": str(target.relative_to(output)),
                "sha256": sha(target),
                "bytes": target.stat().st_size,
                "media": media_info(target),
                "ffmpeg_args": transcode,
            },
        }
    except (OSError, ValueError, subprocess.SubprocessError, urllib.error.URLError) as exc:
        return {**row, "status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("originals", "matched"):
        (args.output / name).mkdir(exist_ok=True)
    lock = args.output / "manifest.json"
    if lock.exists():
        parser.error("a manifest already exists, choose a new output directory")
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = []
        for record in executor.map(lambda row: prepare(row, args.output), plan()):
            records.append(record)
            print(f"{record['id']}: {record['status']}", flush=True)
    hashes: dict[str, str] = {}
    for record in records:
        if "original" in record:
            digest = record["original"]["sha256"]
            prior = hashes.setdefault(digest, record["id"])
            if prior != record["id"]:
                record.update(status="duplicate", reason=f"same source bytes as {prior}")
    body = {
        "policy": POLICY,
        "revision": REVISION,
        "acquisition_script_sha256": sha(Path(__file__)),
        "prior_actions_excluded": sorted(PRIOR_ACTIONS),
        "ffmpeg_version": subprocess.check_output(
            ["ffmpeg", "-version"], text=True, timeout=10
        ).splitlines()[0],
        "items": records,
    }
    with lock.open("x") as file:
        file.write(json.dumps(body, indent=2) + "\n")
    print(f"manifest SHA-256 {sha(lock)}", flush=True)


if __name__ == "__main__":
    main()
