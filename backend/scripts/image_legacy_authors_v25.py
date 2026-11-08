"""Legacy-photo ownership metadata only; never image acquisition or training permission."""

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import requests

from scripts.image_attribution_v20 import profile_owner, short_id, static_identity
from scripts.image_source_groups_v22 import components
from scripts.image_source_rights_v19 import ROOT, coco_images, require, sha, write

BASE = ROOT / "backend/data/independent-evaluation/image-regional-v22-index-successor/source-groups"
INVENTORY = ROOT / "backend/data/independent-evaluation/image-source-rights-v19/audit/private-sources.json"
CODE = ("backend/scripts/image_legacy_authors_v25.py", "backend/scripts/image_legacy_authors_runtime_v25.py", "backend/tests/test_image_legacy_authors_v25.py")
PROTECTED = ("backend/scripts/image_attribution_v20.py", "backend/scripts/image_source_rights_v19.py", "backend/scripts/image_source_groups_v22.py", "backend/scripts/relayformer_learning_runtime_v22.py", "backend/scripts/data_resource_guard_v21.py", "backend/app/detectors/audio/calibrations/local-speech-v3.json")


def read(path):
    return json.loads(path.read_text())


def inspect(expected, metadata):
    require(isinstance(metadata, dict), "Photo metadata object required")
    require(metadata.get("type") == metadata.get("flickr_type") == "photo" and metadata.get("provider_name") == "Flickr" and metadata.get("version") == "1.0", "Canonical Flickr photo response required")
    current = static_identity(metadata.get("url", ""))
    require(all(current[k] == expected[k] for k in ("photo_id", "server", "secret")), "Current photo identity differs from the historical source")
    owner = profile_owner(metadata.get("author_url", ""))
    require(profile_owner(metadata.get("web_page", ""), photo_id=expected["photo_id"]) == owner, "Photo and author owner differ")
    require(isinstance(metadata.get("author_name"), str) and 0 < len(metadata["author_name"].strip()) <= 1000, "Bounded named profile required")
    return {"owner": owner.casefold(), "profile_url": metadata["author_url"], "photo_id": expected["photo_id"], "license_or_camera_authentication_assessed": False}


def plan(folder):
    folder.mkdir(parents=True, exist_ok=False)
    result = read(ROOT / "docs/evaluation-panels/image-region-data-and-training-probes-v23.json")
    require(sha(BASE / "private-groups.json") == result["source_groups"]["private_groups_sha256"], "Published complete source graph required")
    rights = read(ROOT / "docs/evaluation-panels/image-source-rights-v19.json")
    bindings = read(INVENTORY.parent / "bindings.json")
    require(sha(INVENTORY) == bindings["private_sources_sha256"] and sha(INVENTORY.parent / "summary.json") == bindings["summary_sha256"] == rights["separate_local_full_review"]["summary_sha256"], "Previously reviewed source-inventory binding required")
    legacy = [r for r in read(INVENTORY)["sources"] if r["consumed_local_evaluation"]]
    rows = read(BASE / "private-groups.json")["sources"]
    require(len(legacy) == 240 and {r["source_id"] for r in legacy} == {r["source_id"] for r in rows if r["previously_consumed"]}, "All240 prior sources must remain in the denominator")
    archive = ROOT / "backend/data/independent-evaluation/image-source-rights-v19/source/annotations_trainval2017.zip"
    archive_sha = sha(archive)
    require(archive_sha == rights["summary"]["input_sha256"][archive.name], "Primary COCO metadata archive changed")
    primary_images, _licenses, _members = coco_images(archive, {r["source_id"] for r in legacy})
    require(all(row["coco_image"] == primary_images[row["source_id"]] for row in legacy), "Every historical photo reference must replay from primary COCO metadata")
    items = []
    for index, row in enumerate(sorted(legacy, key=lambda r: r["source_id"])):
        identity = static_identity(row["coco_image"]["flickr_url"])
        page = "https://flic.kr/p/" + short_id(identity["photo_id"])
        items.append({"index": index, "source_id": row["source_id"], "expected": identity, "url": "https://www.flickr.com/services/oembed/?url=" + quote(page, safe="") + "&format=json"})
    write(folder / "plan.json", {"schema": "chipmunk.legacy-photo-owner-plan/25", "items": items, "source_graph_sha256": sha(BASE / "private-groups.json"), "inventory_sha256": sha(INVENTORY), "primary_metadata_archive_sha256": archive_sha, "all240_historical_photo_references_replayed": True, "requests_version": requests.__version__, "code_sha256": {p: sha(ROOT / p) for p in CODE}, "protected_sha256": {p: sha(ROOT / p) for p in PROTECTED}, "rules": "All240 sources in fixed ascending order. Once-only GET, at least0.5s between starts, no credentials/redirects/retries,100000-byte maximum accepted body,10/20s timeouts. HTTP429 stops further requests and remaining cases stay explicitly unrequested. Exact photo/server/secret and canonical owner agreement. No image request, licence grant, training admission or source-neural inference."})
    print(json.dumps({"planned": 240, "private_plan_sha256": sha(folder / "plan.json")}), flush=True)


def preflight(folder):
    protocol = read(folder / "plan.json")
    public = read(ROOT / "docs/evaluation-panels/image-legacy-authors-v25-freeze.json")
    require(sha(folder / "plan.json") == public["private_plan_sha256"], "Published legacy-owner source plan required")
    require(all(sha(ROOT / p) == h for p, h in {**protocol["code_sha256"], **protocol["protected_sha256"]}.items()), "Frozen owner-audit or protected source differs")
    require(sha(BASE / "private-groups.json") == protocol["source_graph_sha256"] and sha(INVENTORY) == protocol["inventory_sha256"], "Committed legacy source inventory differs")
    require(requests.__version__ == protocol["requests_version"], "Planned HTTP client version differs")
    return protocol


def acquire(folder):
    protocol = preflight(folder)
    (folder / "responses").mkdir(exist_ok=False)
    previous, limited = 0.0, False
    session = requests.Session()
    session.auth = lambda request: request  # Preserve configured TLS/proxies, bypass netrc.
    try:
        with (folder / "responses.jsonl").open("x") as journal:
            for item in protocol["items"]:
                row = {"index": item["index"], "source_id": item["source_id"], "url": item["url"], "status": "unresolved"}
                body = bytearray()
                if limited:
                    row["status"] = "not_requested_after_rate_limit"
                else:
                    time.sleep(max(0.0, 0.5 - (time.monotonic() - previous)))
                    previous = time.monotonic()
                    row["observed_at"] = datetime.now(UTC).isoformat()
                    try:
                        with session.get(item["url"], headers={"User-Agent": "ChipmunkVerify-Research/1.0 (public photo-owner audit)"}, timeout=(10, 20), allow_redirects=False, stream=True) as response:
                            row.update({"http_status": response.status_code, "content_type": response.headers.get("Content-Type", "")})
                            limited = response.status_code == 429
                            for chunk in response.iter_content(16384):
                                body.extend(chunk[:100001 - len(body)])
                                require(len(body) <= 100000, "Owner response exceeds bounded metadata size")
                        path = folder / "responses" / f"{item['index']:04d}.bin"
                        with path.open("xb") as stream:
                            stream.write(body)
                        row.update({"body": str(path.relative_to(folder)), "bytes": len(body), "body_sha256": sha(path)})
                        require(row["http_status"] == 200 and "application/json" in row["content_type"].lower(), "Public photo-owner metadata unavailable")
                        row["owner"] = inspect(item["expected"], json.loads(body))
                        row["status"] = "matched_current_photo_owner"
                    except Exception as error:
                        if body and "body" not in row:
                            path = folder / "responses" / f"{item['index']:04d}.partial.bin"
                            with path.open("xb") as stream:
                                stream.write(body)
                            row.update({"body": str(path.relative_to(folder)), "bytes": len(body), "body_sha256": sha(path), "partial": True})
                        row["error"] = {"type": type(error).__name__, "message": str(error)}
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
                if (item["index"] + 1) % 40 == 0:
                    print(json.dumps({"completed": item["index"] + 1, "planned": 240}), flush=True)
    finally:
        session.close()
    summarize(folder)


def summarize(folder):
    protocol = preflight(folder)
    records = [json.loads(line) for line in (folder / "responses.jsonl").read_text().splitlines()]
    require(len(records) == len(protocol["items"]) == 240, "Complete240-source denominator required; partial run stays incomplete")
    owners = {}
    for item, row in zip(protocol["items"], records, strict=True):
        require(all(row[k] == item[k] for k in ("index", "source_id", "url")), "Legacy source correspondence differs")
        if "body" in row:
            require(sha(folder / row["body"]) == row["body_sha256"], "Retained provider body differs")
        if row["status"] == "matched_current_photo_owner":
            require(row["http_status"] == 200 and not row.get("partial", False), "Only complete successful provider bodies may supply a profile")
            owner = inspect(item["expected"], read(folder / row["body"]))
            require(owner == row["owner"], "Full provider owner replay differs")
            owners[row["source_id"]] = owner["owner"]
    rows = read(BASE / "private-groups.json")["sources"]
    enriched = [{**row, "author": owners.get(row["source_id"]) if row["previously_consumed"] else row["author"]} for row in rows]
    groups, links = components(enriched)
    lookup = {r["key"]: r for r in enriched}
    with_new = [g for g in groups if any(not lookup[k]["previously_consumed"] for k in g)]
    prior_link = [g for g in with_new if any(lookup[k]["previously_consumed"] for k in g)]
    candidates = [g for g in with_new if all(not lookup[k]["previously_consumed"] for k in g) and any(lookup[k]["publisher_role"] in {"validation", "testing"} for k in g)]
    write(folder / "private-groups.json", {"schema": "chipmunk.owner-enriched-source-groups/25", "sources": enriched, "components": groups, "roles_assigned": False})
    summary = {"schema": "chipmunk.legacy-photo-owner-summary/25", "planned": 240, "completed": 240, "statuses": dict(Counter(r["status"] for r in records)), "matched_current_owner_sources": len(owners), "legacy_sources_with_owner_unresolved": 240 - len(owners), "all_source_records": len(enriched), "all_components": len(groups), "components_with_new_sources": len(with_new), "new_components_linked_to_prior_sources": len(prior_link), "new_sources_linked_to_prior_sources": sum(not lookup[k]["previously_consumed"] for g in prior_link for k in g), "candidate_nontraining_components_without_detected_prior_link": len(candidates), "link_counts": links, "responses_sha256": sha(folder / "responses.jsonl"), "private_groups_sha256": sha(folder / "private-groups.json"), "source_plan_sha256": sha(folder / "plan.json"), "neural_forwards": 0, "fitting_steps": 0, "roles_assigned": False, "outside_human_review": False, "scope": "Current provider photo-owner correspondence, not verified individual photographer identity or rights clearance. Unresolved owners, aliases, bursts/crops and external pretraining remain unassessed. No fresh-confirmation or training admission follows."}
    write(folder / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "acquire", "summarize"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    {"plan": plan, "acquire": acquire, "summarize": summarize}[args.command](args.folder.resolve())
