"""Reserve unseen RTFS target and donor keys for a face-residual study."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts import video_common_sources_v89 as sources
from scripts import video_face_temporal_panel as previous
from scripts.acquire_video_common_sources_v91 import hf_object

HELD_GROUPS = 24
FAMILIES = ("inswapper", "uniface")
SALT = "chipmunk-video-face-residuals-2026-11-16/"


def plan(base, output, retained):
    common = base / "video-common-sources-v89"
    permissions = common / "permissions"
    public_files = base / "video-common-acquisition-v91"
    file_index = public_files / "publisher-file-info.json"
    original = sources.read(common / "plan.json")
    receipt = sources.read(permissions / "receipt.json")
    publisher = sources.read(file_index)
    if not (receipt["rtfs_revision"] == publisher["sha"] == sources.RTFS_REV
            and sources.sha(file_index) == sources.read(public_files / "preparation-summary.json")["publisher_file_info_sha256"]):
        raise ValueError("RTFS metadata revision changed")
    filenames = ("rtfs-videos.csv", "rtfs-photos.csv", *(f"rtfs-{family}.csv" for family in FAMILIES))
    pins = {name: receipt["files"][name]["sha256"] for name in filenames}
    if any(sources.sha(permissions / name) != sha for name, sha in pins.items()):
        raise ValueError("One or more pinned publisher source tables changed")

    history = []
    for name in sources.PARENTS:
        manifest = base / name / "manifest.json"
        if sources.sha(manifest) != original["historical_acquisition_sha256"][name]:
            raise ValueError("Historical acquisition differs")
        history.extend(sources.read(manifest)["items"])
    for group in original["groups"]:
        if group["source_family"] == "paired_face_swap":
            history.extend(group | {"donor_filename": donor} for donor in group["donor_filenames"])
    old_media = set()
    used_groups = set()
    for path in retained:
        archive = sources.read(path)
        if archive["revision"] != sources.RTFS_REV:
            raise ValueError("Unknown historical paired source revision")
        for group in archive["groups"]:
            used_groups.add(group["group"])
            old_media.update(item["sha256"] for item in group["files"].values())
            history.extend({"group": group["group"], "source_video_url": group["source_video_url"],
                            "donor_filename": donor.removeprefix("rtfs-photo-filename:")}
                           for donor in group["donors"] if donor.startswith("rtfs-photo-filename:"))
            history.append({"group": group["group"], "source_video_url": group["source_video_url"], "donor_filename": None})

    photos = sources.table(permissions / "rtfs-photos.csv")
    file_info = {row["rfilename"]: row for row in publisher["siblings"]}

    def qualified(path):
        sha, size = hf_object(file_info[path])
        return 0 < size <= previous.LIMIT and sha not in old_media

    def key(video):
        return previous.digest(previous.RULE + "/" + video["id"])

    videos = [row for row in sources.table(permissions / "rtfs-videos.csv") if key(row) not in used_groups
              and 5 <= float(row["end_time"]) - float(row["start_time"]) and qualified("original_videos/" + row["filename"])]
    targets = {row["filename"] for row in videos}
    swaps = {family: [row for row in sources.table(permissions / f"rtfs-{family}.csv")
                      if row["video"] in targets and row["success"] == "True" and qualified(family + "/" + row["filename"])]
             for family in FAMILIES}
    candidates = sources.paired_targets(videos, swaps, photos, history)
    candidates.sort(key=lambda row: previous.digest(SALT + key(row["target"])))
    chosen = []
    for index, row in enumerate(candidates[:HELD_GROUPS]):
        video = row["target"]
        names = {"ordinary": "original_videos/" + video["filename"]} | {
            item["generator"]: item["generator"] + "/" + item["filename"] for item in row["swaps"]}
        assets = {family: {"path": path, "sha256": sha, "size": size}
                  for family, path in names.items() for sha, size in (hf_object(file_info[path]),)}
        chosen.append({"group": key(video), "role": "held", "source_video_url": video["original_link"],
                       "index": index, "donors": row["donor_source_keys"], "files": assets})
    if (len(chosen) != HELD_GROUPS or len({key for row in chosen for key in row["donors"]}) != 2 * HELD_GROUPS
            or len({asset["sha256"] for row in chosen for asset in row["files"].values()}) != 3 * HELD_GROUPS):
        raise ValueError(f"Incomplete selection or overlapping publisher identities among {len(candidates)} candidates")
    sources.write(output, {"recipe": previous.RECIPE, "rules": previous.RULE, "revision": sources.RTFS_REV,
                           "selected_groups": {"development": 0, "held": HELD_GROUPS}, "groups": chosen,
                           "selection_salt": SALT, "retained_plan_sha256": [sources.sha(path) for path in retained],
                           "historical_groups_excluded": len({row["group"] for row in history}),
                           "metadata_sha256": pins, "publisher_file_info_sha256": sources.sha(file_index)})
    print(f"Froze {len(chosen)} fresh targets and {len(chosen) * 2} separate recorded donor keys.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("retained", nargs="+", type=Path)
    args = parser.parse_args()
    plan(args.base, args.output, args.retained)
