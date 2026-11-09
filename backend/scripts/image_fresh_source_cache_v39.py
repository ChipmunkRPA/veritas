"""Cache a frozen set of attributed JPEG references and compare source components."""

import argparse
import hashlib
import importlib.metadata
import json
import plistlib
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from scripts import image_fresh_attribution_v37 as attribution
from scripts.image_fresh_attribution_readout_v38 import known_creator_keys
from scripts.image_source_groups_v22 import author_key, components, sketch
from scripts.image_source_rights_v19 import ROOT, require, sha, write

VOLUME = Path("/Volumes/SEAGATE")
UUID = "0011BD4E-4AED-3A2E-B5FF-7731A071ABBD"
ARCHIVE_ROOT = VOLUME / "Codex-Development-Archive/2026-11-08-Chipmunk-Verify"
SUMMARY_SHA = "d52987951642ec2e96e554c0ab0db98c73a8633ee61c9fbd05483782d6f5dfb6"
CHUNK, COUNT, MAX_BYTES = 32, 163, 5_000_000
PACKAGES = ("numpy", "Pillow", "requests", "psutil")
CODE = (
    "backend/scripts/image_fresh_source_cache_v39.py", "backend/scripts/image_fresh_source_cache_runtime_v39.py",
    "backend/tests/test_image_fresh_source_cache_v39.py", "backend/scripts/image_fresh_attribution_v37.py",
    "backend/scripts/image_fresh_attribution_readout_v38.py", "backend/scripts/image_attribution_v20.py",
    "backend/scripts/image_source_groups_v22.py", "backend/scripts/image_source_rights_v19.py",
    "backend/scripts/local_probe_limits_v24.py", "backend/scripts/data_resource_guard_v21.py",
)


def volume_check():
    require(VOLUME.is_mount(), "Connected external volume required")
    result = subprocess.run(["diskutil", "info", "-plist", str(VOLUME)], check=True, capture_output=True, timeout=15)
    info = plistlib.loads(result.stdout)
    require(info["VolumeUUID"] == UUID and info["WritableVolume"] is True and info["MountPoint"] == str(VOLUME), "Exact writable Seagate volume required")


def prepare(catalog, trial, cache):
    volume_check()
    require(cache.parent.resolve() == ARCHIVE_ROOT.resolve() and not cache.exists(), "New private cache directly under authorized archive required")
    protocol = attribution.admission(catalog)
    require(sha(catalog / "summary-v38.json") == SUMMARY_SHA, "Exact completed corrected attribution summary required")
    profile = json.loads((catalog / "summary-v38-profile/profile.json").read_text())
    require(profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0, "Successful corrected metadata readout required")
    prior_keys, _ = known_creator_keys(json.loads(attribution.GRAPH.read_text())["sources"])
    selected, phase_hashes = [], {}
    for number in range(8):
        attribution.verified_profile(catalog, number)
        rows = attribution.review_phase(catalog, protocol, number)
        phase_hashes[str(number)] = sha(catalog / f"phase-{number:03d}/responses.jsonl")
        for row in rows:
            if row["status"] == "matched_current_provider_metadata" and author_key(row["attribution"]["author_url"]) not in prior_keys:
                selected.append({"metadata_index": row["index"], "source_id": row["source_id"], "attribution": row["attribution"], "metadata_body_sha256": row["sha256"]})
    require(len(selected) == len({row["source_id"] for row in selected}) == COUNT, "Every163 metadata-qualified unlinked source required")
    trial.mkdir(parents=True, exist_ok=False)
    cache.mkdir(exist_ok=False)
    (cache / "objects").mkdir()
    document = {
        "schema": "chipmunk.fresh-image-source-cache-plan/39", "catalog": str(catalog), "cache": str(cache),
        "catalog_plan_sha256": sha(catalog / "plan.json"), "catalog_summary_sha256": SUMMARY_SHA,
        "catalog_response_journal_sha256": phase_hashes, "prior_groups": str(attribution.GRAPH), "prior_groups_sha256": sha(attribution.GRAPH),
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "runtime_versions": {name: importlib.metadata.version(name) for name in PACKAGES},
        "volume_uuid": UUID, "planned": COUNT, "chunk": CHUNK, "phases": 6, "items": selected,
        "policy": "All163 current-photo/owner/licence matches without an identified creator-key link, in original metadata order. One HTTPS static JPEG request each, no redirects/retries/replacements,5MB bound,10/20s timeout,>=0.25s request starts. Keep failed and partial response bodies. JPEG only,dimensions1..2048,upright EXIF1 or absent; never silently rotate. Hash encoded bytes and exact native RGB. Apply frozen v22 source/creator/exact-RGB/informative256bit dHash rules to all705prior plus qualified new references, full transitive components. No neural call, generation, source-role assignment or evaluation admission.",
    }
    write(trial / "plan.json", document)
    print(json.dumps({"planned": COUNT, "phases": 6, "plan_sha256": sha(trial / "plan.json")}), flush=True)


def admission(trial):
    plan = json.loads((trial / "plan.json").read_text())
    require(plan["planned"] == COUNT and plan["chunk"] == CHUNK and plan["phases"] == 6, "Exact163-source schedule required")
    require(all(sha(ROOT / name) == value for name, value in plan["code_sha256"].items()), "Frozen cache implementation changed")
    require(plan["runtime_versions"] == {name: importlib.metadata.version(name) for name in PACKAGES}, "Frozen image decoder/runtime changed")
    require(sha(Path(plan["prior_groups"])) == plan["prior_groups_sha256"], "Existing source graph changed")
    catalog = Path(plan["catalog"])
    require(sha(catalog / "plan.json") == plan["catalog_plan_sha256"] and sha(catalog / "summary-v38.json") == plan["catalog_summary_sha256"], "Attribution commitments changed")
    require(all(sha(catalog / f"phase-{int(n):03d}/responses.jsonl") == digest for n, digest in plan["catalog_response_journal_sha256"].items()), "Attribution response journals changed")
    receipt = json.loads((trial / "execution-start.json").read_text())
    require(receipt.get("plan_sha256") == sha(trial / "plan.json") and receipt.get("remote_verified_before_first_image_request") is True, "Published cache commitment must precede image requests")
    require(isinstance(receipt.get("remote_freeze_blob_sha"), str) and len(receipt["remote_freeze_blob_sha"]) == 40, "Exact remote cache freeze receipt required")
    cache = Path(plan["cache"])
    require(cache.parent.resolve() == ARCHIVE_ROOT.resolve() and cache.is_dir(), "Private Seagate cache required")
    volume_check()
    return plan


def pixel_record(item, path, expected):
    require(0 < path.stat().st_size <= MAX_BYTES and sha(path) == expected, "Exact bounded encoded source bytes required")
    with Image.open(path) as image:
        require(image.format == "JPEG" and 0 < min(image.size) <= max(image.size) <= 2048, "Bounded native JPEG required")
        require(image.getexif().get(274, 1) == 1, "Unhandled source orientation remains unresolved")
        rgb = image.convert("RGB").copy()
    meta = item["attribution"]
    require(meta["license_id"] in {4, 5}, "Recorded BY/BY-SA source required")
    return {"key": f"fresh-jpeg/{item['source_id']}", "source_id": item["source_id"], "author": author_key(meta["author_url"]),
            "size": list(rgb.size), "rgb_sha256": hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(), "sketch": f"{sketch(rgb):064x}",
            "source_jpeg_file": str(path), "source_jpeg_sha256": expected, "source_jpeg_bytes": path.stat().st_size,
            "photo_page": meta["photo_page"], "author_profile": meta["author_url"], "license_id": meta["license_id"], "license_url": meta["license_url"],
            "metadata_body_sha256": item["metadata_body_sha256"], "roles_assigned": False, "neural_forwards": 0}


def phase_items(plan, number):
    require(type(number) is int and 0 <= number < 6, "One exact cache phase0..5 required")
    rows = plan["items"][number * CHUNK:(number + 1) * CHUNK]
    require(len(rows) == (3 if number == 5 else 32), "Complete fixed cache phase required")
    return rows


def verified_profile(trial, number):
    path = trial / f"fetch-{number:03d}-profile/profile.json"
    data = json.loads(path.read_text())
    require(data["schema"] == "chipmunk.fresh-image-cache-resource-profile/39" and data["command"] == "fetch" and data["arguments"] == [str(trial), str(number)], "Exact cache worker resource profile required")
    require(data["child_exit_code"] == 0 and data["guard"] is None and data["sample_errors"] == 0 and not data["resource_limit_observations"], "Successful bounded cache phase required")
    require(data["resource_journal_sha256"] == sha(path.parent / "resource-samples.jsonl"), "Cache resource journal changed")
    return path, data


def review_phase(trial, plan, number):
    folder = trial / f"fetch-{number:03d}"
    items = phase_items(plan, number)
    records = [json.loads(line) for line in (folder / "responses.jsonl").read_text().splitlines()]
    require(len(records) == len(items), "Complete cache denominator includes failed requests")
    cache = Path(plan["cache"])
    for row, item in zip(records, items, strict=True):
        require(row["source_id"] == item["source_id"] and row["metadata_index"] == item["metadata_index"] and row["url"] == item["attribution"]["expected_large_source_url"], "Exact cache source binding required")
        require(row["status"] in {"qualified_native_reference", "unresolved"}, "Explicit cache status required")
        if row.get("body"):
            require(row["body"] == f"objects/{row['sha256']}.bin", "Content-addressed cache object required")
            path = cache / row["body"]
            require(path.stat().st_size == row["bytes"] <= MAX_BYTES + 1 and sha(path) == row["sha256"], "Retained source bytes changed")
        if row["status"] == "qualified_native_reference":
            require(row.get("body") and row.get("http_status") == 200 and row["bytes"] <= MAX_BYTES, "Complete successful source bytes required")
            require(row["source"] == pixel_record(item, path, row["sha256"]), "Read-back source pixels/geometry/sketch differ")
    return records


def fetch(trial, number):
    plan = admission(trial)
    items = phase_items(plan, number)
    for previous in range(number):
        verified_profile(trial, previous)
        review_phase(trial, plan, previous)
    folder = trial / f"fetch-{number:03d}"
    folder.mkdir(exist_ok=False)
    cache, previous_start = Path(plan["cache"]), 0.0
    with (folder / "responses.jsonl").open("x") as journal:
        for item in items:
            time.sleep(max(0, 0.25 - (time.monotonic() - previous_start)))
            previous_start = time.monotonic()
            row = {"source_id": item["source_id"], "metadata_index": item["metadata_index"], "url": item["attribution"]["expected_large_source_url"]}
            body = bytearray()
            try:
                with requests.get(row["url"], headers=attribution.HEADERS, stream=True, timeout=(10, 20), allow_redirects=False) as response:
                    row.update({"http_status": response.status_code, "content_type": response.headers.get("Content-Type")})
                    for chunk in response.iter_content(65536):
                        body.extend(chunk[:MAX_BYTES + 1 - len(body)])
                        require(len(body) <= MAX_BYTES, "Source body exceeds5MB bound")
                    require(response.status_code == 200, "Static reference unavailable")
                require(body, "Nonempty source body required")
                row["status"] = "received_body"
            except Exception as error:
                row["status"] = "unresolved"
                row["error"] = {"type": type(error).__name__, "message": str(error)}
            if body:
                digest = hashlib.sha256(body).hexdigest()
                name = f"objects/{digest}.bin"
                path = cache / name
                if not path.exists():
                    with path.open("xb") as out:
                        out.write(body)
                require(sha(path) == digest, "Retained cache object differs")
                row.update({"body": name, "bytes": len(body), "sha256": digest})
            if row["status"] == "received_body":
                try:
                    row["source"] = pixel_record(item, path, digest)
                    row["status"] = "qualified_native_reference"
                except Exception as error:
                    row["status"] = "unresolved"
                    row["error"] = {"type": type(error).__name__, "message": str(error)}
            row["duration_s"] = time.monotonic() - previous_start
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
    records = review_phase(trial, plan, number)
    write(folder / "completed.json", {"phase": number, "planned": len(items), "recorded": len(records), "statuses": dict(Counter(r["status"] for r in records)), "journal_sha256": sha(folder / "responses.jsonl")})
    print(json.dumps({"phase": number, "recorded": len(records), "qualified": sum(row["status"] == "qualified_native_reference" for row in records)}), flush=True)


def seal(trial):
    plan = admission(trial)
    records, profiles = [], []
    for number in range(6):
        path, profile = verified_profile(trial, number)
        rows = review_phase(trial, plan, number)
        completed = json.loads((trial / f"fetch-{number:03d}/completed.json").read_text())
        require(completed["recorded"] == len(rows) and completed["journal_sha256"] == sha(trial / f"fetch-{number:03d}/responses.jsonl"), "Completed cache phase differs")
        records.extend(rows)
        profiles.append({"phase": number, "profile_sha256": sha(path), "duration_s": profile["duration_s"], "sampled_tree_rss_lower_bound_bytes": profile["sampled_tree_rss_lower_bound_bytes"]})
    require(len(records) == COUNT, "Every163 planned reference remains in denominator")
    new = [row["source"] for row in records if row["status"] == "qualified_native_reference"]
    prior = json.loads(Path(plan["prior_groups"]).read_text())["sources"]
    require(len(prior) == 705, "Complete prior705-source inventory required")
    rows = prior + new
    groups, links = components(rows)
    previous_keys, new_keys = {row["key"] for row in prior}, {row["key"] for row in new}
    new_groups = [group for group in groups if any(key in new_keys for key in group)]
    unlinked = [group for group in new_groups if not any(key in previous_keys for key in group)]
    linked = [group for group in new_groups if any(key in previous_keys for key in group)]
    by_key = {row["key"]: row for row in rows}
    write(trial / "private-groups.json", {"schema": "chipmunk.private-fresh-image-source-components/39", "sources": rows, "components": groups, "candidate_only_unlinked_components": unlinked, "roles_assigned": False})
    result = {
        "schema": "chipmunk.fresh-image-source-cache-summary/39", "metadata_selected": 256, "metadata_unresolved": 67, "metadata_matched_prior_creator_excluded": 26,
        "cache_planned": COUNT, "cache_recorded": len(records), "cache_statuses": dict(Counter(row["status"] for row in records)),
        "cache_unresolved_reasons": dict(Counter(row["error"]["message"] if row["error"]["type"] == "ValueError" else row["error"]["type"] for row in records if row["status"] == "unresolved")),
        "existing_inventory_sources": len(prior), "qualified_new_sources": len(new), "new_components": len(new_groups),
        "new_photos_in_prior_linked_components": sum(sum(key in new_keys for key in group) for group in linked),
        "metadata_and_pixel_unlinked_sources": sum(len(group) for group in unlinked), "metadata_and_pixel_unlinked_components": len(unlinked),
        "unlinked_licence_ids": dict(Counter(str(by_key[key]["license_id"]) for group in unlinked for key in group)),
        "pairwise_link_reasons_all_sources": links, "phase_profiles": profiles, "private_plan_sha256": sha(trial / "plan.json"), "private_groups_sha256": sha(trial / "private-groups.json"),
        "neural_forwards": 0, "generator_outputs": 0, "roles_assigned": False, "evaluation_admitted": False,
        "limitations": "Current JPEG reference pixels and conservative source/profile/exact-RGB/thumbnail components only. No camera-origin authentication, creator-alias resolution or assurance against transformed duplicates/pretraining overlap. Generator component rights and authored hardware/geometry probes remain necessary. No detector accuracy result, outside human review, workflow or outreach.",
    }
    write(trial / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fetch", "seal"))
    parser.add_argument("arguments", nargs="+")
    args = parser.parse_args()
    if args.command == "prepare" and len(args.arguments) == 3:
        prepare(*(Path(value).resolve() for value in args.arguments))
    elif args.command == "fetch" and len(args.arguments) == 2:
        fetch(Path(args.arguments[0]).resolve(), int(args.arguments[1]))
    elif args.command == "seal" and len(args.arguments) == 1:
        seal(Path(args.arguments[0]).resolve())
    else:
        parser.error("Explicit complete worker arguments required")
