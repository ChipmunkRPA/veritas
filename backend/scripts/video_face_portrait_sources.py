# SPDX-License-Identifier: Apache-2.0
"""Reserve additional paired public face videos with pinned source exclusions."""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

from scripts import video_common_sources_v89 as io
from scripts import video_face_temporal_panel as existing
from scripts.acquire_video_common_sources_v91 import hf_object

V17 = "a826e166b0542ad3bf5fcf1de00c3f9958976a62351cb3c5fa55906220598da6"
FINGERPRINT = re.compile(rb'"group"\s*:\s*"([0-9a-f]{64})"')
COUNTS = {"development": 10, "held": 18}


def plan(base, panels, output):
    known = base / "video-common-sources-v89"
    permissions = known / "permissions"
    receipt = io.read(permissions / "receipt.json")
    historic = io.read(known / "plan.json")
    if receipt["rtfs_revision"] != io.RTFS_REV or io.sha(permissions / "receipt.json") != historic["metadata_receipt_sha256"]:
        raise ValueError("Historical source metadata commitment differs")
    metadata = ("rtfs-videos.csv", "rtfs-photos.csv", "rtfs-inswapper.csv", "rtfs-uniface.csv")
    if any(io.sha(permissions / name) != receipt["files"][name]["sha256"] for name in metadata):
        raise ValueError("Pinned publisher metadata differs")
    prior, manifests = [], {}
    for folder in io.PARENTS:
        name = base / folder / "manifest.json"
        pin = historic["historical_acquisition_sha256"][folder]
        if folder == "video-v17-sface/sources":
            # The source-plan items are public and reproduce the URL and donor exclusions
            # even if the archived acquisition manifest isn't mounted.
            name = panels / "video-v17-sface-source-plan.json"
            pin = V17
        if io.sha(name) != pin:
            raise ValueError("Historical source list differs from its pinned copy")
        manifests[folder] = pin
        prior.extend(io.read(name)["items"])
    for group in historic["groups"]:
        if group["source_family"] == "paired_face_swap":
            prior.extend({**group, "donor_filename": donor} for donor in group["donor_filenames"])
    for group in io.read(panels / "video-face-shot-continuity-sources-2026-11-17.json")["groups"]:
        for donor in group["donors"]:
            prefix = "rtfs-photo-filename:"
            prior.append(group | {"donor_filename": donor.removeprefix(prefix) if donor.startswith(prefix) else None})
    exclusions = set()
    for file in sorted(panels.glob("video*.json*")):
        content = gzip.decompress(file.read_bytes()) if file.suffix == ".gz" else file.read_bytes()
        exclusions.update(match.decode() for match in FINGERPRINT.findall(content))
    archive = base / "video-common-acquisition-v91"
    info = archive / "publisher-file-info.json"
    publisher = io.read(info)
    if (io.sha(info) != io.read(archive / "preparation-summary.json")["publisher_file_info_sha256"]
            or publisher["sha"] != io.RTFS_REV):
        raise ValueError("Publisher media digests differ from the pinned revision")
    siblings = {file["rfilename"]: file for file in publisher["siblings"]}
    candidates = io.paired_targets(
        io.table(permissions / "rtfs-videos.csv"),
        {family: io.table(permissions / f"rtfs-{family}.csv") for family in ("inswapper", "uniface")},
        io.table(permissions / "rtfs-photos.csv"), prior,
    )
    groups = []
    for candidate in candidates:
        target = candidate["target"]
        group = existing.digest(existing.RULE + "/" + target["id"])
        if group in exclusions or float(target["end_time"]) - float(target["start_time"]) < 5:
            continue
        paths = {"ordinary": "original_videos/" + target["filename"]} | {
            item["generator"]: item["generator"] + "/" + item["filename"] for item in candidate["swaps"]}
        files = {}
        for kind, name in paths.items():
            pin, size = hf_object(siblings[name])
            files[kind] = {"path": name, "sha256": pin, "size": size}
        if all(0 < item["size"] <= existing.LIMIT for item in files.values()):
            groups.append({"group": group, "donors": candidate["donor_source_keys"], "files": files,
                           "source_video_url": target["original_link"]})
    selected = sorted(groups, key=lambda group: group["group"])[:sum(COUNTS.values())]
    if (len(selected) != sum(COUNTS.values())
            or len({asset["sha256"] for group in selected for asset in group["files"].values()}) != len(selected) * 3):
        raise ValueError(f"Insufficient distinct publisher media after exclusions ({len(groups)} eligible groups)")
    for index, group in enumerate(selected):
        group.update(index=index, role="development" if index < COUNTS["development"] else "held")
    value = {"recipe": existing.RECIPE, "rules": existing.RULE, "revision": io.RTFS_REV, "groups": selected,
             "selected_groups": COUNTS, "historical_source_sha256": manifests, "published_excluded_groups": len(exclusions),
             "publisher_file_info_sha256": io.sha(info),
             "metadata_sha256": {name: receipt["files"][name]["sha256"] for name in metadata}}
    io.write(output, value)
    print("Reserved", len(selected), "distinct source groups and",
          sum(asset["size"] for group in selected for asset in group["files"].values()), "bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("panels", type=Path)
    parser.add_argument("output", type=Path)
    plan(**vars(parser.parse_args()))
