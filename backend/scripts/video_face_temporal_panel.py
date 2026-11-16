# SPDX-License-Identifier: MIT
"""Select and acquire fresh paired sources for the whole-clip temporal study."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from scripts import video_common_sources_v89 as sources
from scripts.acquire_video_common_sources_v91 import hf_object

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/independent-evaluation"
COUNTS = {"development": 14, "held": 22}
LIMIT = 6 * 1024**2
RULE = "whole-clip-face-temporal/1"
RECIPE = {"duration": 4, "size": [640, 360], "fps": 25, "crf": [23, 36],
          "source_sample_start_seconds": ".40", "faces": "largest unambiguous tracked face"}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def plan(base, output, *, counts=None, max_source_seconds=10, excluded_groups=()):
    counts = COUNTS if counts is None else counts
    known = base / "video-common-sources-v89"
    permissions = known / "permissions"
    receipt = sources.read(permissions / "receipt.json")
    assert receipt["rtfs_revision"] == sources.RTFS_REV
    metadata = ("rtfs-videos.csv", "rtfs-photos.csv", "rtfs-inswapper.csv", "rtfs-uniface.csv")
    for name in metadata:
        assert sources.sha(permissions / name) == receipt["files"][name]["sha256"], name
    historical_plan = sources.read(known / "plan.json")
    prior = []
    for folder in sources.PARENTS:
        manifest = base / folder / "manifest.json"
        assert sources.sha(manifest) == historical_plan["historical_acquisition_sha256"][folder]
        prior.extend(sources.read(manifest)["items"])
    for group in historical_plan["groups"]:
        if group["source_family"] == "paired_face_swap":
            prior.extend({**group, "donor_filename": donor} for donor in group["donor_filenames"])
    archive = base / "video-common-acquisition-v91"
    info = archive / "publisher-file-info.json"
    assert sources.sha(info) == sources.read(archive / "preparation-summary.json")["publisher_file_info_sha256"]
    assert (publisher := sources.read(info))["sha"] == sources.RTFS_REV
    siblings = {file["rfilename"]: file for file in publisher["siblings"]}
    candidates = sources.paired_targets(
        sources.table(permissions / "rtfs-videos.csv"),
        {family: sources.table(permissions / f"rtfs-{family}.csv") for family in ("inswapper", "uniface")},
        sources.table(permissions / "rtfs-photos.csv"), prior,
    )
    eligible = []
    for case in candidates:
        target = case["target"]
        group_key = digest(RULE + "/" + target["id"])
        if group_key in excluded_groups or not 5 <= float(target["end_time"]) - float(target["start_time"]) <= max_source_seconds:
            continue
        paths = {"ordinary": "original_videos/" + target["filename"]} | {
            item["generator"]: item["generator"] + "/" + item["filename"] for item in case["swaps"]}
        files = {}
        for family, name in paths.items():
            assert name in siblings
            sha, size = hf_object(siblings[name])
            files[family] = {"path": name, "sha256": sha, "size": size}
        if any(row["size"] > LIMIT for row in files.values()):
            continue
        eligible.append({"group": group_key, "source_video_url": target["original_link"],
                         "donors": case["donor_source_keys"], "files": files})
    eligible.sort(key=lambda row: row["group"])
    chosen = eligible[:sum(counts.values())]
    assert len(chosen) == sum(counts.values())
    for i, group in enumerate(chosen):
        group.update(role="development" if i < counts["development"] else "held", index=i)
    assert len({media["sha256"] for item in chosen for media in item["files"].values()}) == len(chosen) * 3
    assert len({donor for row in chosen for donor in row["donors"]}) == len(chosen) * 2
    value = {"recipe": RECIPE, "rules": RULE, "revision": sources.RTFS_REV, "selected_groups": counts,
             "groups": chosen, "historical_groups_excluded": len({row["group"] for row in prior}),
             "metadata_sha256": {name: receipt["files"][name]["sha256"] for name in metadata},
             "publisher_file_info_sha256": sources.sha(info)}
    sources.write(output, value)
    print("Planned", len(chosen), "source groups,", sum(f["size"] for row in chosen for f in row["files"].values()),
          "total bytes", flush=True)


def acquire(plan_path, output):
    data = sources.read(plan_path)
    assert data["rules"] == RULE and data["revision"] == sources.RTFS_REV and data["recipe"] == RECIPE
    output.mkdir(exist_ok=True, mode=0o700)

    def download(entry):
        index, kind, asset = entry
        assert "/" in asset["path"] and asset["path"].endswith(".mp4") and 0 < asset["size"] <= LIMIT
        path = output / f"{index:02d}-{kind}.mp4"
        if path.exists():
            assert path.stat().st_size == asset["size"] and sources.sha(path) == asset["sha256"]
            return
        url = f"{sources.RTFS}/resolve/{sources.RTFS_REV}/{quote(asset['path'])}"
        req = Request(url, headers={"User-Agent": "Chipmunk-Verify-video-study"})
        with urlopen(req, timeout=45) as response:
            content = response.read(asset["size"] + 1)
        if len(content) != asset["size"] or hashlib.sha256(content).hexdigest() != asset["sha256"]:
            raise ValueError(f"Publisher asset bytes differ for group {index}, {kind}")
        with path.open("xb") as file:
            file.write(content)

    pending = [(row["index"], arm, asset) for row in data["groups"] for arm, asset in row["files"].items()]
    with ThreadPoolExecutor(max_workers=6) as pool:
        errors = []
        for entry, future in zip(pending, [pool.submit(download, item) for item in pending], strict=True):
            try:
                future.result()
            except (OSError, ValueError) as error:
                errors.append((entry[0], entry[1], type(error).__name__))
        if errors:
            print(json.dumps(errors), flush=True)
            raise RuntimeError("Some fixed source assets did not arrive. Retry the same roster.")
    print("Verified publisher lengths and SHA-256 for", len(pending), "files.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "acquire"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("directory", type=Path, nargs="?")
    args = parser.parse_args()
    if args.action == "plan":
        plan(args.directory or BASE, args.manifest)
    else:
        if args.directory is None:
            parser.error("acquire needs a destination directory")
        acquire(args.manifest, args.directory)


if __name__ == "__main__":
    main()
