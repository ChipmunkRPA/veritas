"""Prepare a small audiovisual set that separates timing errors from synthesis."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from prepare_video_upgrade_eval import download, media_info, sha

REVISION = "395b1fa4d494e7474ccaccd638d9ed2df7701d59"
REPO = "https://huggingface.co/datasets/cambrain/DigiFakeAV"
POLICY = "chipmunk-video-av-controls/1"
SAMPLES = (
    "real_videos_13", "real_videos_29",
    "FVRA_7", "FVRA_8", "FVRA_15164", "FVRA_15165",
    "FVFA_9508", "FVFA_20805", "FVFA_20806", "FVFA_21632",
)


def prepare(output: Path) -> dict:
    source = output / "originals"
    controls = output / "controls"
    source.mkdir(parents=True, exist_ok=True)
    controls.mkdir(exist_ok=True)
    rows = []
    for name in SAMPLES:
        path = source / f"{name}.mp4"
        url = f"{REPO}/resolve/{REVISION}/few_examples/{path.name}"
        download(url, path)
        rows.append({
            "id": name,
            "video_ai": name.startswith("FV"),
            "voice_ai": name.startswith("FVFA"),
            "category": "publisher_example",
            "path": str(path.relative_to(output)),
            "source_url": url,
            "sha256": sha(path),
            "media": media_info(path),
        })
    real = [row for row in rows if not row["video_ai"]]
    for picture, alternate in (real, real[::-1]):
        for treatment in ("natural", "late-320ms", "human-voice-over"):
            name = f"{picture['id']}-{treatment}"
            target = controls / f"{name}.mp4"
            sound = alternate if treatment == "human-voice-over" else picture
            command = [
                "-i", str(output / picture["path"]), "-i", str(output / sound["path"]),
                "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "-1",
                "-t", "3.8", "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "96k",
            ]
            if treatment == "late-320ms":
                command.extend(["-af", "adelay=320:all=1"])
            partial = target.with_suffix(".part.mp4")
            try:
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *command, str(partial)],
                    check=True, timeout=45, capture_output=True,
                )
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
            rows.append({
                "id": name,
                "video_ai": False,
                "voice_ai": False,
                "category": treatment,
                "picture_sha256": picture["sha256"],
                "sound_sha256": sound["sha256"],
                "path": str(target.relative_to(output)),
                "sha256": sha(target),
                "media": media_info(target),
                "ffmpeg_args": command,
            })
    return {
        "policy": POLICY,
        "source_card": f"{REPO}/blob/{REVISION}/README.md",
        "source_license": "Publisher dataset card MIT tag",
        "script_sha256": sha(Path(__file__)),
        "purpose": "Engineering controls only; too few independent originals for calibration",
        "items": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    lock = args.output / "manifest.json"
    if lock.exists():
        parser.error("a manifest already exists, choose a new output directory")
    body = prepare(args.output)
    with lock.open("x") as file:
        file.write(json.dumps(body, indent=2) + "\n")
    print(f"{len(body['items'])} examples and controls, manifest SHA-256 {sha(lock)}")
