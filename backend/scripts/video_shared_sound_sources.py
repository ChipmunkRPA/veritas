# SPDX-License-Identifier: Apache-2.0
"""Reserve previously unused recorded speech for stereo soundtrack localization."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from scripts.acquire_video_common_sources_v91 import lfs_pointer
from scripts.video_lip_multishot_panel import RATE, digest, read
from scripts.video_silent_mouth_clock_panel import (
    CREMA,
    CREMA_REV,
    FRAME_PERIOD,
    LICENSE_SHA,
    SOURCE_INDEX_SHA,
    fetch,
    fit_sound,
    sha,
    source_sound,
)

SCHEMA = "chipmunk.video-shared-sound-sources/1"


def acquire(index, license_file, earlier, second, output):
    from app.detectors.audio.decode import decode_audio
    from app.detectors.video.decode import probe_frames

    if digest(index) != SOURCE_INDEX_SHA or digest(license_file) != LICENSE_SHA:
        raise ValueError("The public CREMA-D index or license differs from its pinned release")
    first = {row["speaker"] for row in read(earlier)["rows"]}
    subsequent = {row["speaker"]: row["source_sha256"] for row in read(second)["clocks"] if "speaker" in row}
    known = first | subsequent.keys()
    clips = defaultdict(list)
    with index.open(newline="") as stream:
        for row in csv.DictReader(stream):
            label = row["Filename"]
            if label.endswith("_NEU_XX"):
                actor = label.split("_")[0]
                clips[sha(f"CREMA-D:{actor}".encode())].append(label + ".flv")

    def order(value):
        return sha(f"{SCHEMA}/{value}".encode())

    dev = sorted(subsequent, key=order)[:10]
    held = sorted(clips.keys() - known, key=order)
    if len(dev) != 10 or len(held) < 12:
        raise ValueError("The planned new-speaker confirmation is unavailable")
    entries = [{"split": split, "speaker": speaker, "alternatives": sorted(clips[speaker], key=order)}
               for split, speakers in (("development", dev), ("held", held)) for speaker in speakers]
    output.mkdir(mode=0o700)
    plan = {"schema": SCHEMA, "publisher_revision": CREMA_REV, "source_index_sha256": SOURCE_INDEX_SHA,
            "license_sha256": LICENSE_SHA, "earlier_receipts": [digest(earlier), digest(second)], "entries": entries}
    (output / "plan.json").write_text(json.dumps(plan, sort_keys=True) + "\n")

    def reserve(row):
        exclusions = []
        for name in row["alternatives"]:
            signature, size = lfs_pointer(fetch(f"{CREMA}/VideoFlash/{name}", 1024))
            if signature == subsequent.get(row["speaker"]):
                continue
            if size > 2 * 1024**2:
                raise ValueError("Unexpected size for the pinned public clip")
            root = f"https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/{CREMA_REV}/VideoFlash"
            body = fetch(f"{root}/{name}", size)
            if len(body) != size or sha(body) != signature:
                raise ValueError("The downloaded clip differs from the public Git LFS pointer")
            clip = output / name
            clip.write_bytes(body)
            table = probe_frames(clip)
            if (not 40 <= len(table) <= 120 or table.rotation or table.truncated or table.interpolated
                    or any(abs((pts - table.pts[0]) * table.tb - i * FRAME_PERIOD) > Fraction(3, 2000)
                           for i, pts in enumerate(table.pts))):
                exclusions.append({"name": name, "sha256": signature, "pictures": len(table), "reason": "picture_scope"})
                clip.unlink()
                continue
            try:
                pcm, _ = source_sound(decode_audio(clip, source_sha256=signature), table)
                fit_sound(pcm, round(len(table) * RATE / 30))
            except ValueError:
                exclusions.append({"name": name, "sha256": signature, "pictures": len(table), "reason": "sound_clock"})
                clip.unlink()
                continue
            return {key: row[key] for key in ("split", "speaker")} | {
                "name": name, "sha256": signature, "bytes": size, "scope_exclusions": exclusions,
            }
        raise ValueError("Unable to reserve a different clip for the requested speaker")

    with ThreadPoolExecutor(max_workers=5) as pool:
        media = list(pool.map(reserve, entries))
    (output / "sources.json").write_text(json.dumps({**plan, "entries": media}, sort_keys=True) + "\n")
    print(f"Verified {len(dev)} development recordings and {len(held)} held speakers against their source digests.", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "license_file", "earlier", "second", "output"):
        cli.add_argument(name, type=Path)
    acquire(**vars(cli.parse_args()))
