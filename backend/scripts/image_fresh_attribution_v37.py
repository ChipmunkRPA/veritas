"""Prospective, model-free expansion of the attributed image source pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import requests

from scripts.image_attribution_v20 import HEADERS, inspect_metadata, short_id, static_identity
from scripts.image_source_groups_v22 import author_key
from scripts.image_source_rights_v19 import ROOT, license_status, require, sha, write

ARCHIVE = ROOT / "backend/data/independent-evaluation/image-source-rights-v19/source/annotations_trainval2017.zip"
GRAPH = ROOT / "backend/data/independent-evaluation/image-owned-source-feasibility-v35/private-groups.json"
QUEUE = ROOT / "backend/data/independent-evaluation/image-source-rights-v19/review-queue/private-review-queue.json"
PINS = {
    str(ARCHIVE.relative_to(ROOT)): "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
    str(GRAPH.relative_to(ROOT)): "380fa4c3685339c66b6b135edf032326532bdcd0fec5c1727b862cab80cd28e8",
}
CODE = (
    "backend/scripts/image_fresh_attribution_v37.py",
    "backend/scripts/image_fresh_attribution_runtime_v37.py",
    "backend/tests/test_image_fresh_attribution_v37.py",
    "backend/scripts/image_attribution_v20.py",
    "backend/scripts/image_source_rights_v19.py",
    "backend/scripts/image_source_groups_v22.py",
    "backend/scripts/local_probe_limits_v24.py",
    "backend/scripts/data_resource_guard_v21.py",
)
COUNT, CHUNK = 256, 32


def rank(source):
    return hashlib.sha256(f"fresh-image-attribution-v37/20261108/{source['source_id']}".encode()).hexdigest()


def choose_sources(images, licenses, excluded, count=COUNT):
    """Selection uses historical licence and identifiers, never detector outcomes."""
    require(type(count) is int and count > 0, "Positive fixed source count required")
    require(len(images) == len({r["id"] for r in images}), "Unique COCO IDs required")
    candidates, invalid_urls = [], 0
    for image in images:
        require(type(image["id"]) is int and image["id"] > 0, "Positive integer source ID required")
        if image["id"] in excluded or license_status(image.get("license"), licenses) not in {"metadata_by", "metadata_by_sa"}:
            continue
        try:
            identity = static_identity(image["flickr_url"])
        except (KeyError, ValueError):
            invalid_urls += 1
            continue
        candidates.append({"source_id": image["id"], "publisher_role": "fresh-unassigned", "coco_image": image,
                           "recorded_license": licenses[image["license"]], "photo_id": identity["photo_id"]})
    unique, seen = [], set()
    for source in sorted(candidates, key=rank):
        if source["photo_id"] not in seen:
            unique.append(source)
            seen.add(source["photo_id"])
    require(len(unique) >= count, "Insufficient eligible unique historical photo IDs")
    return unique[:count], {"eligible_rows": len(candidates), "eligible_photo_ids": len(unique), "invalid_static_urls": invalid_urls}


def plan(folder):
    folder.mkdir(parents=True, exist_ok=False)
    for name, expected in PINS.items():
        require(sha(ROOT / name) == expected, "Pinned source inventory changed")
    graph = json.loads(GRAPH.read_text())
    require(len(graph["sources"]) == 705 and graph["roles_assigned"] is False, "Exact existing705-source inventory required")
    queue = json.loads(QUEUE.read_text())["sources"]
    require(len(queue) == 523, "Complete historical review queue required")
    excluded = {r["source_id"] for r in graph["sources"]} | {r["source_id"] for r in queue}
    images, licenses, members = [], None, {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        for split in ("train", "val"):
            name = f"annotations/captions_{split}2017.json"
            require(archive.namelist().count(name) == 1, "Unique exact annotation member required")
            info = archive.getinfo(name)
            require(0 < info.file_size < 300_000_000 and not info.flag_bits & 1, "Bounded unencrypted JSON member required")
            raw = archive.read(name)
            doc = json.loads(raw)
            current = {row["id"]: row for row in doc["licenses"]}
            require(len(current) == len(doc["licenses"]), "Unique license IDs required")
            require(licenses is None or current == licenses, "Train/validation licence tables differ")
            licenses = current
            members[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "crc32": info.CRC, "images": len(doc["images"])}
            images.extend({**image, "coco_split": split} for image in doc["images"])
            del doc, raw
    selected, selection = choose_sources(images, licenses, excluded)
    items = []
    for index, source in enumerate(selected):
        page = "https://flic.kr/p/" + short_id(source["photo_id"])
        url = "https://www.flickr.com/services/oembed/?url=" + quote(page, safe="") + "&format=json"
        items.append({"index": index, "source": source, "url": url})
    document = {
        "schema": "chipmunk.fresh-image-attribution-plan/37", "created_at": datetime.now(UTC).isoformat(),
        "input_sha256": {**PINS, str(QUEUE.relative_to(ROOT)): sha(QUEUE)},
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "caption_members": members, "selection": selection, "excluded_source_ids": len(excluded),
        "planned": COUNT, "chunk": CHUNK, "phases": COUNT // CHUNK, "items": items,
        "policy": "Fixed hash-ranked256 distinct photo IDs; exclude every705-source inventory ID and523-source prior review ID; historical BY/BY-SA2.0 only. Eight32-request phases, each once, no retries/replacements. At least0.25s between starts,100000-byte cap,10/20s timeouts. Require current exact photo/server/secret, owner and licence correspondence. No image download, neural call, source-role assignment, fit or evaluation admission. Unknown owner aliases/pretraining/source authenticity remain unresolved.",
    }
    write(folder / "plan.json", document)
    print(json.dumps({"planned": COUNT, "phases": COUNT // CHUNK, "selection": selection, "plan_sha256": sha(folder / "plan.json")}), flush=True)


def admission(folder):
    protocol = json.loads((folder / "plan.json").read_text())
    require(protocol["planned"] == COUNT and protocol["chunk"] == CHUNK and protocol["phases"] == 8, "Frozen complete source schedule required")
    require(all(sha(ROOT / name) == digest for name, digest in protocol["input_sha256"].items()), "Planned input changed")
    require(all(sha(ROOT / name) == digest for name, digest in protocol["code_sha256"].items()), "Planned source implementation changed")
    receipt = json.loads((folder / "execution-start.json").read_text())
    require(receipt.get("plan_sha256") == sha(folder / "plan.json"), "Execution plan commitment differs")
    require(receipt.get("remote_verified_before_first_request") is True, "Published metadata-only protocol must precede requests")
    require(isinstance(receipt.get("remote_freeze_blob_sha"), str) and len(receipt["remote_freeze_blob_sha"]) == 40, "Remote protocol receipt required")
    return protocol


def phase_items(protocol, number):
    require(type(number) is int and 0 <= number < 8, "One exact phase0..7 required")
    items = protocol["items"][number * CHUNK:(number + 1) * CHUNK]
    require(len(items) == CHUNK and [r["index"] for r in items] == list(range(number * CHUNK, (number + 1) * CHUNK)), "Complete ordered32-request phase required")
    return items


def acquire(folder, number):
    protocol = admission(folder)
    items = phase_items(protocol, number)
    for previous in range(number):
        verified_profile(folder, previous)
        review_phase(folder, protocol, previous)
        completed = json.loads((folder / f"phase-{previous:03d}/completed.json").read_text())
        require(completed["journal_sha256"] == sha(folder / f"phase-{previous:03d}/responses.jsonl") and completed["recorded"] == CHUNK, "Previous completed metadata denominator required")
    output = folder / f"phase-{number:03d}"
    output.mkdir(exist_ok=False)
    (output / "responses").mkdir()
    previous_start = 0.0
    with (output / "responses.jsonl").open("x") as journal:
        for item in items:
            time.sleep(max(0, 0.25 - (time.monotonic() - previous_start)))
            previous_start = time.monotonic()
            record = {"index": item["index"], "source_id": item["source"]["source_id"], "url": item["url"], "observed_at": datetime.now(UTC).isoformat()}
            body = bytearray()
            try:
                with requests.get(item["url"], headers=HEADERS, stream=True, timeout=(10, 20), allow_redirects=False) as response:
                    record.update({"http_status": response.status_code, "content_type": response.headers.get("Content-Type")})
                    for chunk in response.iter_content(16384):
                        body.extend(chunk[:100001 - len(body)])
                        require(len(body) <= 100000, "Attribution response exceeds size limit")
                    require(response.status_code == 200, "Attribution endpoint unavailable")
                    require("application/json" in response.headers.get("Content-Type", "").lower(), "JSON attribution response required")
                    record["attribution"] = inspect_metadata(item["source"], json.loads(body))
                    record["status"] = "matched_current_provider_metadata"
            except Exception as error:
                record["status"] = "unresolved"
                record["error"] = {"type": type(error).__name__, "message": str(error)}
            if body:
                name = f"responses/{item['index']:04d}.bin"
                (output / name).write_bytes(body)
                record.update({"body": name, "bytes": len(body), "sha256": sha(output / name), "body_may_be_partial": record["status"] != "matched_current_provider_metadata"})
            record["duration_s"] = time.monotonic() - previous_start
            journal.write(json.dumps(record, allow_nan=False) + "\n")
            journal.flush()
    records = review_phase(folder, protocol, number)
    result = {"phase": number, "planned": CHUNK, "recorded": len(records), "statuses": dict(Counter(r["status"] for r in records)), "journal_sha256": sha(output / "responses.jsonl"), "neural_calls": 0, "images_downloaded": 0}
    write(output / "completed.json", result)
    print(json.dumps(result), flush=True)


def review_phase(folder, protocol, number):
    items = phase_items(protocol, number)
    output = folder / f"phase-{number:03d}"
    records = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
    require(len(records) == CHUNK, "Complete denominator includes every unresolved request")
    for row, item in zip(records, items, strict=True):
        require((row["index"], row["source_id"], row["url"]) == (item["index"], item["source"]["source_id"], item["url"]), "Exact ordered response/source binding required")
        require(row["status"] in {"matched_current_provider_metadata", "unresolved"}, "Explicit metadata status required")
        if row.get("body"):
            require(row["body"] == f"responses/{row['index']:04d}.bin", "Exact response filename required")
            data = (output / row["body"]).read_bytes()
            require(len(data) == row["bytes"] <= 100001 and hashlib.sha256(data).hexdigest() == row["sha256"], "Response bytes changed")
        if row["status"] == "matched_current_provider_metadata":
            require(row.get("body") is not None, "Matched metadata requires retained complete response bytes")
            require(row.get("http_status") == 200 and "application/json" in row.get("content_type", "").lower() and not row.get("body_may_be_partial"), "Complete successful JSON response required")
            require(row["attribution"] == inspect_metadata(item["source"], json.loads(data)), "Reconstructed current metadata differs")
    return records


def verified_profile(folder, number):
    profile_path = folder / f"phase-{number:03d}-profile/profile.json"
    profile = json.loads(profile_path.read_text())
    require(profile["schema"] == "chipmunk.fresh-attribution-resource-profile/37", "Exact metadata profile schema required")
    require(profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0 and not profile["errors"] and not profile["resource_limit_observations"], "Every bounded metadata phase must have succeeded")
    require(profile["arguments"] == [str(folder), str(number)] and profile["command"] == "acquire", "Exact metadata worker profile required")
    require(profile["resource_journal_sha256"] == sha(profile_path.parent / "resource-samples.jsonl"), "Metadata resource journal changed")
    return profile_path, profile


def summarize(folder):
    protocol = admission(folder)
    records, profiles = [], []
    for number in range(8):
        profile_path, profile = verified_profile(folder, number)
        current = review_phase(folder, protocol, number)
        completed = json.loads((folder / f"phase-{number:03d}/completed.json").read_text())
        require(completed["journal_sha256"] == sha(folder / f"phase-{number:03d}/responses.jsonl") and completed["recorded"] == len(current) == CHUNK, "Completed metadata journal differs")
        records.extend(current)
        profiles.append({"phase": number, "profile_sha256": sha(profile_path), "sampled_tree_rss_lower_bound_bytes": profile["sampled_tree_rss_lower_bound_bytes"], "duration_s": profile["duration_s"]})
    graph = json.loads(GRAPH.read_text())
    previous_authors = {author_key(source["author"]) for source in graph["sources"] if source["author"]}
    matched = [row for row in records if row["status"] == "matched_current_provider_metadata"]
    linked = [row for row in matched if author_key(row["attribution"]["author_url"]) in previous_authors]
    unlinked = [row for row in matched if author_key(row["attribution"]["author_url"]) not in previous_authors]
    result = {
        "schema": "chipmunk.fresh-image-attribution-summary/37", "planned": COUNT, "recorded": len(records),
        "statuses": dict(Counter(row["status"] for row in records)),
        "http_statuses": dict(Counter(str(row.get("http_status", "network_error")) for row in records)),
        "unresolved_reasons": dict(Counter(row["error"]["message"] if row["error"]["type"] == "ValueError" else row["error"]["type"] for row in records if row["status"] == "unresolved")),
        "matched_photo_licences": dict(Counter(str(row["attribution"]["license_id"]) for row in matched)),
        "matched_linked_to_known_creator_keys": len(linked), "matched_without_known_creator_link": len(unlinked),
        "distinct_unlinked_creator_keys": len({author_key(row["attribution"]["author_url"]) for row in unlinked}),
        "prior_sources_with_unknown_creator": sum(not source["author"] for source in graph["sources"]),
        "plan_sha256": sha(folder / "plan.json"), "phase_profiles": profiles,
        "images_downloaded": 0, "neural_calls": 0, "roles_assigned": False, "evaluation_admitted": False,
        "limitations": "Current photo/server/secret/owner/CC-BY or BY-SA2.0 correspondence only. No source pixels, cross-pool exact/near duplicate grouping, creator-alias resolution, pretraining independence, camera authenticity or complete generation rights established. No external person contacted. Unresolved requests retained without replacement; no accuracy result.",
    }
    write(folder / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "acquire", "summarize"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("phase", type=int, nargs="?")
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(args.folder.resolve(), args.phase)
    else:
        globals()[args.command](args.folder.resolve())
