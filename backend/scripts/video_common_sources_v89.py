"""Metadata-only source preparation for a common face/mouth video comparison.

No detector, generator, media decoder or customer storage is imported. Private
source identities stay in the preparation directory; the summary is aggregate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit
from urllib.request import Request, urlopen

import psutil

RTFS_REV = "bca865e675852666a426368b2f5614abf1c1926a"
CREMA_REV = "1658cd342dff90010aa843eaeebd53610a08b1dc"
RTFS = "https://huggingface.co/datasets/godmodes/rtfs-10k"
CREMA = "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/" + CREMA_REV
PARENTS = ("video-v2/face-swap", "video-v3/face-swap", "video-v4/face-sources",
           "video-v5/sources", "video-v6/sources", "video-v17-sface/sources")
ROLES = ("fusion_fit", "confidence_cal", "selection", "confirmation")
HISTORY_GLOBS = ("lipsync*/**/manifest.json", "lipsync*/**/*plan.json", "video*/**/manifest.json",
                 "multimodal/*video*manifest*.json", "rule80-v1/video-manifest.json")
ACTOR = re.compile(r"crema(?:[-_ ]?d)?[-:/_ ]+(10\d{2})|(?<!\d)(10\d{2})(?=_[A-Z]{3}_[A-Z]{3}_)", re.I)
CREMA_FILE = re.compile(r"(10\d{2})_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}(?:\.flv)?", re.I)
LIMIT = 8 * 1024**2


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def order(value):
    return hashlib.sha256(("chipmunk-common-video-sources/89\0" + value).encode()).digest()


def source_key(url):
    """Conservative aliases: unify protocols/www, YouTube forms and non-video queries.

    Query removal can merge distinct assets; it cannot prove distinct people.
    These keys are only grouping/exclusion metadata, never network destinations.
    """
    value = urlsplit(url)
    host = (value.hostname or "").lower().removeprefix("www.")
    require(value.scheme in {"http", "https"} and bool(host), "Source URL missing or malformed")
    path = unquote(value.path).rstrip("/")
    if host in {"youtube.com", "m.youtube.com", "youtu.be"}:
        identifier = (path.lstrip("/") if host == "youtu.be" else
                      parse_qs(value.query).get("v", [""])[0])
        if not identifier and path.startswith(("/shorts/", "/embed/", "/live/", "/v/")):
            identifier = path.split("/")[2]
        require(re.fullmatch(r"[\w-]{11}", identifier, flags=re.ASCII), "Unrecognized YouTube source identity")
        return "youtube:" + identifier
    return host + path


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def field(row, *names):
    values = {row[name].strip() for name in names if row.get(name, "").strip()}
    require(len(values) == 1, "Missing or conflicting metadata field")
    return values.pop()


def filename(value, suffix):
    require(Path(value).name == value and ".." not in value and value.endswith(suffix), "Unsafe source filename")
    return value


def fetch_metadata(destination):
    require(not destination.exists(), "Preserve every existing acquisition, including partial failures")
    require(psutil.virtual_memory().available >= 3 * 1024**3
            and shutil.disk_usage(destination.parent).free >= 3 * 1024**3, "Metadata preparation headroom unavailable")
    destination.mkdir()
    urls = {"rtfs-" + name: f"{RTFS}/resolve/{RTFS_REV}/{name}"
            for name in ("README.md", "videos.csv", "photos.csv", "inswapper.csv", "uniface.csv")}
    urls.update({"crema-" + name: CREMA + "/" + name
                 for name in ("LICENSE.txt", "README.md", "VideoDemographics.csv", "SentenceFilenames.csv")})
    started, files, observations = time.monotonic(), {}, []
    status, error = "failed", None
    try:
        for name, url in urls.items():
            with urlopen(Request(url, headers={"User-Agent": "Chipmunk-Verify-source-review"}), timeout=30) as response:
                data = response.read(LIMIT + 1)
            require(0 < len(data) <= LIMIT and not data.lstrip().startswith(b"<!DOCTYPE"), "Unexpected metadata payload")
            with (destination / name).open("xb") as stream:
                stream.write(data)
            files[name] = {"url": url, "bytes": len(data), "sha256": sha(destination / name)}
            observation = {"elapsed_s": time.monotonic() - started,
                           "rss_bytes": psutil.Process().memory_info().rss,
                           "available_ram_bytes": psutil.virtual_memory().available,
                           "free_disk_bytes": shutil.disk_usage(destination).free}
            observations.append(observation)
            require(observation["elapsed_s"] < 900 and observation["rss_bytes"] < 1536 * 1024**2
                    and min(observation["available_ram_bytes"], observation["free_disk_bytes"]) >= 1536 * 1024**2,
                    "Metadata preparation resource limit reached")
        require(files["rtfs-README.md"]["sha256"] == "72f7a92dab5b5ec9c7a82b05fb2095513a6d7f7650088588ae1984132b4bb6ae",
                "Reviewed immutable dataset card changed")
        license_text = (destination / "crema-LICENSE.txt").read_text()
        require("licenses/odbl/1.0/" in license_text and "licenses/dbcl/1.0/" in license_text,
                "Expected publisher database/content grants missing")
        status = "complete"
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:400]}
        raise
    finally:
        write(destination / "receipt.json", {"status": status, "files": files, "error": error,
            "rtfs_revision": RTFS_REV, "crema_revision": CREMA_REV,
            "script_sha256": sha(__file__), "recorded_utc": datetime.now(UTC).isoformat(),
            "observations": observations, "duration_s": time.monotonic() - started,
            "scope": "Per-download data-only observations, not continuous/peak monitoring. No raw media, neural call or role assignment."})


def historical_actors(base):
    paths = sorted({p for pattern in HISTORY_GLOBS for p in base.glob(pattern)})
    actors = set()
    for path in paths:
        actors.update(int(match[1] or match[2]) for match in ACTOR.finditer(path.read_text()))
    return actors, {str(path.relative_to(base)): sha(path) for path in paths}


def paired_targets(originals, generated, photos, historical):
    old_groups = {row["group"] for row in historical}
    old_urls = {source_key(row["source_video_url"]) for row in historical}
    donors = {row["donor_filename"] for row in historical if row["donor_filename"]}
    # The pinned publisher CSV currently supplies tags/filenames, despite the
    # card describing source links. Explicitly retain that identity/provenance
    # gap; filename equality is not a claim of distinct portraits or people.
    photo_map = {field(row, "filename"): (source_key(field(row, "link", "original_link"))
                 if row.get("link") or row.get("original_link") else "rtfs-photo-filename:" + field(row, "filename"))
                 for row in photos}
    require(len(photo_map) == len(photos) and donors <= photo_map.keys(), "Complete historical donor source map required")
    used_donor_urls = {photo_map[name] for name in donors}
    pools = {}
    for family, rows in generated.items():
        pool = defaultdict(list)
        for row in rows:
            require(row["generator"] == family, "Unexpected generator family in source table")
            if row["success"] == "True" and not row["additional_features"].strip() and "child" not in (row["photo_tags"] + " " + row["video_tags"]).lower():
                require(row["photo"] in photo_map, "Generated donor has no source URL")
                pool[row["video"]].append(row)
        pools[family] = pool
    selected = []
    for original in sorted(originals, key=lambda row: order(row["filename"])):
        source = source_key(original["original_link"])
        if "RTFS:" + original["id"] in old_groups or source in old_urls or "child" in original["tags"].lower():
            continue
        choices = []
        for family in ("inswapper", "uniface"):
            candidates = [row for row in pools[family][original["filename"]]
                          if row["photo"] not in donors and photo_map[row["photo"]] not in used_donor_urls]
            if not candidates:
                break
            choices.append(min(candidates, key=lambda row: order(row["filename"])))
        if len(choices) != 2 or len({photo_map[row["photo"]] for row in choices}) != 2:
            continue
        old_urls.add(source)
        donors.update(row["photo"] for row in choices)
        used_donor_urls.update(photo_map[row["photo"]] for row in choices)
        selected.append({"target": original, "swaps": choices,
                         "donor_source_keys": [photo_map[row["photo"]] for row in choices]})
    return selected


def plan(base, destination, rtfs_per_role=24, crema_per_role=12):
    require(type(rtfs_per_role) is int and rtfs_per_role > 0 and type(crema_per_role) is int
            and crema_per_role > 0, "Positive explicit source counts required")
    permissions = destination / "permissions"
    receipt = read(permissions / "receipt.json")
    require(receipt["status"] == "complete" and receipt["rtfs_revision"] == RTFS_REV
            and receipt["crema_revision"] == CREMA_REV, "Complete pinned metadata receipt required")
    require(all(sha(permissions / name) == row["sha256"] for name, row in receipt["files"].items()), "Metadata bytes changed")
    parents, parent_bindings = [], {}
    for name in PARENTS:
        path = base / name / "manifest.json"
        body = read(path)
        require(body["revision"] == RTFS_REV, "Historical dataset revision changed")
        for filename_, digest in body["metadata_sha256"].items():
            require(sha(permissions / ("rtfs-" + filename_)) == digest, "Historical and current metadata differ")
        parents.extend(body["items"])
        parent_bindings[name] = sha(path)
    candidates = paired_targets(table(permissions / "rtfs-videos.csv"),
        {name: table(permissions / ("rtfs-" + name + ".csv")) for name in ("inswapper", "uniface")},
        table(permissions / "rtfs-photos.csv"), parents)
    old_actors, history = historical_actors(base)
    published_actors = {int(field(row, "ActorID")) for row in table(permissions / "crema-VideoDemographics.csv")}
    require(len(published_actors) == 91, "Complete publisher actor inventory required")
    actor_clips = defaultdict(set)
    for row in table(permissions / "crema-SentenceFilenames.csv"):
        for value in row.values():
            if value and (match := CREMA_FILE.fullmatch(value.strip())):
                actor_clips[int(match[1])].add(value.strip() if value.strip().lower().endswith(".flv") else value.strip() + ".flv")
    require(set(actor_clips) == published_actors and old_actors <= published_actors,
            "Complete source-file actor mapping and historical actor exclusion required")
    eligible_actors = sorted(set(actor_clips) - old_actors, key=lambda actor: order(str(actor)))
    require(len(candidates) >= 4 * rtfs_per_role and len(eligible_actors) >= 4 * crema_per_role,
            "Insufficient paired/source-history-separated metadata; no response-driven substitution")
    groups = []
    for index, item in enumerate(candidates[:4 * rtfs_per_role]):
        target = item["target"]
        files = [("original_videos", filename(target["filename"], ".mp4")),
                 *[(row["generator"], filename(row["filename"], ".mp4")) for row in item["swaps"]]]
        require([name for name, _ in files] == ["original_videos", "inswapper", "uniface"], "Expected two pinned generator families")
        groups.append({"source_family": "paired_face_swap", "group": "RTFS:" + target["id"],
            "provisional_role": ROLES[index % 4], "source_video_url": target["original_link"],
            "donor_source_keys": item["donor_source_keys"], "donor_filenames": [row["photo"] for row in item["swaps"]],
            "assets": [{"kind": family, "source_path": family + "/" + name,
                        "url": f"{RTFS}/resolve/{RTFS_REV}/{family}/{quote(name)}"} for family, name in files]})
    for index, actor in enumerate(eligible_actors[:4 * crema_per_role]):
        name = min(actor_clips[actor], key=order)
        groups.append({"source_family": "recorded_audiovisual_speech", "group": f"CREMA-D:{actor}",
            "provisional_role": ROLES[index % 4], "assets": [{"kind": "genuine_av", "source_path": "VideoFlash/" + name,
                "pointer_url": CREMA + "/VideoFlash/" + name,
                "url": f"https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/{CREMA_REV}/VideoFlash/{name}"}]})
    summary = {"schema": "chipmunk.video-common-source-preparation/89", "source_groups": len(groups),
        "source_assets_planned": sum(len(row["assets"]) for row in groups),
        "families": dict(Counter(row["source_family"] for row in groups)),
        "roles": {role: dict(Counter(row["source_family"] for row in groups if row["provisional_role"] == role)) for role in ROLES},
        "historical_rtfs_groups_excluded": len({row["group"] for row in parents}),
        "historical_crema_actors_excluded": len(old_actors), "rtfs_metadata_pool_after_known_alias_exclusions": len(candidates),
        "remaining_crema_actor_codes": len(eligible_actors), "new_media_downloads": 0, "neural_calls": 0,
        "portrait_source_urls_supplied": sum(bool(row.get("link") or row.get("original_link"))
                                             for row in table(permissions / "rtfs-photos.csv")),
        "role_assignment_provisional": True, "defaults_changed": False,
        "permission_scope": "RTFS publisher CC-BY-SA-4.0; CREMA-D ODbL1.0/DbCL1.0. Publisher provenance/rights are not independently guaranteed. No restricted generator weights are downloaded.",
        "limits": "Metadata source plan only. Verify media, native clocks, decoded aliases, known edit boundaries and all selected weight-specific grants before any detector study. Pinned portrait CSV lacks the source URLs described by its card; donor grouping is filename-only when links are absent, with incomplete portrait attribution/identity linkage. No raw donor portraits are acquired. Actor/video/source-URL keys do not prove distinct people or unknown pretraining independence. Role assignments must be finalized after byte/decoded-identity checks. Silent clips cannot validate lip sync. No external reviewer participation."}
    body = {**summary, "groups": groups, "metadata_receipt_sha256": sha(permissions / "receipt.json"),
            "historical_acquisition_sha256": parent_bindings, "crema_history_sha256": history,
            "script_sha256": sha(__file__)}
    require(not (destination / "plan.json").exists() and not (destination / "summary.json").exists(), "Never overwrite a source preparation")
    write(destination / "plan.json", body)
    summary["private_plan_sha256"] = sha(destination / "plan.json")
    write(destination / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("metadata", "plan"))
    parser.add_argument("destination", type=Path)
    parser.add_argument("--base", type=Path, default=Path("data/independent-evaluation"))
    args = parser.parse_args()
    if args.command == "metadata":
        args.destination.mkdir(exist_ok=False)
        fetch_metadata(args.destination / "permissions")
    else:
        plan(args.base, args.destination)
