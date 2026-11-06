"""Freeze original speech clips, exact sample shifts and published lip-redrawing examples."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from prepare_video_upgrade_eval import download, media_info, sha

CREMA = "1658cd342dff90010aa843eaeebd53610a08b1dc"
LATENT = "a229c3948406bc2cf6eaf4873e662e70c6a04746"
CREMA_REPO = "CheyneyComputerScience/CREMA-D"
LATENT_REPO = "bytedance/LatentSync"
POLICY = "chipmunk-video-timing/1"
RATE = 16000
SHIFT_SAMPLES = 5120
FF = ["ffmpeg", "-v", "error", "-nostdin", "-y"]


def order(value: str) -> bytes:
    return hashlib.sha256(f"{POLICY}:{value}".encode()).digest()


def matched_group(root: Path, group: str, files: list[str], split: str) -> list[dict]:
    for source in sorted(files, key=order):
        url = f"https://media.githubusercontent.com/media/{CREMA_REPO}/{CREMA}/{source}"
        original = root / "sources" / Path(source).name
        download(url, original)
        if float(media_info(original)["format"]["duration"]) >= 1.95:
            break
    else:
        raise ValueError(f"no eligible speech clip for {group}")
    pcm = np.frombuffer(subprocess.check_output(
        [*FF, "-i", str(original), "-map", "0:a:0", "-af", "aresample=16000:async=1:first_pts=0",
         "-ac", "1", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"], timeout=30), dtype="<i2")
    rows = []
    for variant, shift in (("original", 0), ("audio-later", SHIFT_SAMPLES), ("audio-earlier", -SHIFT_SAMPLES)):
        shifted = np.zeros_like(pcm)
        if shift > 0:
            shifted[shift:] = pcm[:-shift]
        elif shift < 0:
            shifted[:shift] = pcm[-shift:]
        else:
            shifted[:] = pcm
        case = f"crema-{group}-{variant}"
        target = root / "matched" / f"{case}.mkv"
        subprocess.run([*FF, "-i", str(original), "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", "pipe:0",
                        "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "-1", "-vf", "fps=25",
                        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20", "-threads", "2",
                        "-c:a", "pcm_s16le", "-shortest", str(target)],
                       input=shifted.tobytes(), timeout=40, check=True, capture_output=True)
        rows.append({"id": case, "group": f"CREMA-D:{group}", "category": variant, "split": split,
                     "shift_samples": shift, "sample_rate": RATE, "path": str(target.relative_to(root)),
                     "sha256": sha(target), "original_sha256": sha(original), "download_url": url,
                     "source_license": "ODbL 1.0 and Database Contents License 1.0"})
    return rows


def published_examples(root: Path) -> list[dict]:
    readme = root / "sources" / "LatentSync.md"
    source = f"https://raw.githubusercontent.com/{LATENT_REPO}/{LATENT}/README.md"
    download(source, readme)
    demo = readme.read_text().split("## 🎬 Demo", 1)[1].split("</table>", 1)[0]
    urls = re.findall(r"<video src=(https://github\.com/user-attachments/assets/[a-f0-9-]+)", demo)
    if len(urls) < 4 or len(urls) % 2:
        raise ValueError("publisher demo pairs changed")
    rows = []
    # The first two pairs show filmed performers. The later examples include animation.
    for j, url in enumerate(urls[:4]):
        variant = "publisher-lip-redrawn" if j % 2 else "publisher-original"
        case = f"latentsync-{j // 2 + 1}-{variant}"
        target = root / "matched" / f"{case}.mp4"
        download(url, target)
        rows.append({"id": case, "group": f"LatentSync:demo-{j // 2 + 1}", "category": variant,
                     "split": "publisher-example", "shift_samples": None, "path": str(target.relative_to(root)),
                     "sha256": sha(target), "download_url": url, "source_url": source,
                     "source_license": "Publisher demonstration media used for local evaluation only"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    lock = root / "manifest.json"
    if lock.exists():
        parser.error("manifest already frozen")
    for name in ("sources", "matched"):
        (root / name).mkdir(parents=True, exist_ok=True)
    tree = root / "crema-tree.json"
    download(f"https://api.github.com/repos/{CREMA_REPO}/git/trees/{CREMA}?recursive=1", tree)
    groups: dict[str, list[str]] = {}
    for row in json.loads(tree.read_text())["tree"]:
        if m := re.fullmatch(r"VideoFlash/(\d{4})_(?:MTI|ITS|IWL|DFA)_NEU_XX\.flv", row["path"]):
            groups.setdefault(m[1], []).append(row["path"])
    selected = sorted(groups, key=order)[:28]
    if len(selected) != 28:
        raise ValueError("missing source groups")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(matched_group, root, group, groups[group], "development" if i < 10 else "test")
                   for i, group in enumerate(selected)]
        items = []
        for group, future in zip(selected, futures, strict=True):
            items.extend(future.result())
            print(group, "ready", flush=True)
    items.extend(published_examples(root))
    with lock.open("x") as file:
        file.write(json.dumps({"policy": POLICY, "tree_sha256": sha(tree), "items": items}, indent=2) + "\n")
    print("manifest SHA-256", sha(lock), flush=True)


if __name__ == "__main__":
    main()
