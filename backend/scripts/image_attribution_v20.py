"""Source-bound public attribution audit; never permission or AI-origin proof."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlparse

import requests

from scripts.image_source_rights_v19 import ROOT, require, sha, write

QUEUE = ROOT / "backend/data/independent-evaluation/image-source-rights-v19/review-queue/private-review-queue.json"
ALPHABET = "123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"
CODE = ("backend/scripts/image_attribution_v20.py", "backend/scripts/image_attribution_runtime_v20.py", "backend/tests/test_image_attribution_v20.py")
HEADERS = {"User-Agent": "ChipmunkVerify-Research/1.0 (public attribution audit; no account credentials)"}


def short_id(number):
    require(type(number) is int and number > 0, "Positive integer Flickr photo ID required")
    letters = []
    while number:
        number, remainder = divmod(number, 58)
        letters.append(ALPHABET[remainder])
    return "".join(reversed(letters))


def static_identity(url):
    parsed = urlparse(url)
    require(parsed.scheme in {"http", "https"} and not parsed.query and not parsed.fragment and not parsed.params, "Unambiguous static photo URL required")
    require(parsed.netloc == "live.staticflickr.com" or re.fullmatch(r"farm[0-9]+\.staticflickr\.com", parsed.netloc), "Official static photo host required")
    match = re.fullmatch(r"/([0-9]+)/([0-9]+)_([a-f0-9]+)(?:_([a-z0-9]+))?\.(jpg|jpeg)", parsed.path)
    require(match is not None, "Recognized static JPEG reference required")
    server, identifier, secret, size, _ = match.groups()
    return {"server": server, "photo_id": int(identifier), "secret": secret, "size": size}


def profile_owner(url, *, photo_id=None):
    parsed = urlparse(url)
    require(parsed.scheme == "https" and parsed.netloc == "www.flickr.com" and not parsed.params and not parsed.query and not parsed.fragment, "Canonical HTTPS Flickr attribution URL required")
    pattern = r"/photos/([^/]+)/?" if photo_id is None else rf"/photos/([^/]+)/{photo_id}/?"
    match = re.fullmatch(pattern, parsed.path)
    require(match is not None, "Photo/author URL identity differs")
    return match[1]


def inspect_metadata(source, metadata):
    require(isinstance(metadata, dict), "Photo metadata object required")
    require(metadata.get("type") == "photo" and metadata.get("flickr_type") == "photo" and metadata.get("provider_name") == "Flickr" and metadata.get("version") == "1.0", "Flickr photo response required")
    original = static_identity(source["coco_image"]["flickr_url"])
    current = static_identity(metadata.get("url", ""))
    require(all(current[k] == original[k] for k in ("photo_id", "secret", "server")), "Current static source identity differs from COCO record")
    author = metadata.get("author_name")
    require(isinstance(author, str) and 0 < len(author.strip()) <= 1000, "Named public photographer/profile required")
    owner = profile_owner(metadata.get("author_url", ""))
    require(profile_owner(metadata.get("web_page", ""), photo_id=original["photo_id"]) == owner, "Author/photo owner references differ")
    expected = source["recorded_license"]
    require(expected["id"] in {4, 5}, "Preliminary commercial/adaptation license required")
    require(metadata.get("license_id") in {str(expected["id"]), expected["id"]}, "Current and recorded license identifiers differ")
    expected_slug = "by" if expected["id"] == 4 else "by-sa"
    allowed = {f"https://creativecommons.org/licenses/{expected_slug}/2.0/", f"http://creativecommons.org/licenses/{expected_slug}/2.0/"}
    require(metadata.get("license_url") in allowed and expected["url"] in allowed, "Current and recorded license URLs differ")
    require(all(type(metadata.get(k)) is int and 0 < metadata[k] <= 20000 for k in ("width", "height")), "Bounded photo dimensions required")
    title = metadata.get("title", "")
    require(isinstance(title, str) and len(title) <= 20000, "Bounded photo title required")
    return {
        "source_id": source["source_id"], "publisher_role": source["publisher_role"],
        "photo_id": original["photo_id"], "author_name": author, "author_url": metadata["author_url"],
        "title": title, "photo_page": metadata["web_page"], "license_id": expected["id"], "license_url": metadata["license_url"],
        "source_identity": original,
        "expected_large_source_url": f"https://live.staticflickr.com/{original['server']}/{original['photo_id']}_{original['secret']}_b.jpg",
        "provider_size": [metadata["width"], metadata["height"]],
        "provider_attribution_and_license_match": True, "pixel_lineage_verified": False, "training_admitted": False,
        "scope": "Current provider metadata matched to historical COCO photo identity; not independent photographer identity, raw-camera authentication or complete rights clearance.",
    }


def plan(folder):
    folder.mkdir(parents=True, exist_ok=False)
    public = json.loads((ROOT / "docs/evaluation-panels/image-source-rights-v19.json").read_text())
    require(sha(QUEUE) == public["rights_review_queue_summary"]["private_queue_sha256"], "Published v19 private queue commitment required")
    queue = json.loads(QUEUE.read_text())["sources"]
    require(len(queue) == len({r["source_id"] for r in queue}) == 523, "Complete523-photo review queue required")
    require(all(not r["ready_for_training"] for r in queue), "No implicit training admission")
    output = []
    for index, source in enumerate(queue):
        identity = static_identity(source["coco_image"]["flickr_url"])
        page = "https://flic.kr/p/" + short_id(identity["photo_id"])
        url = "https://www.flickr.com/services/oembed/?url=" + quote(page, safe="") + "&format=json"
        output.append({"index": index, "source": source, "url": url})
    doc = {
        "schema": "chipmunk.image-attribution-plan/20", "created_at": datetime.now(UTC).isoformat(),
        "input_queue_sha256": sha(QUEUE), "code_sha256": {p: sha(ROOT / p) for p in CODE}, "items": output,
        "policy": "All523 queued sources once in fixed order; no retries or replacements; >=0.25seconds between request starts; response limit100000bytes;10/20s network timeouts. Retain errors; require exact photo/server/secret, owner and current license correspondence. No model calls, photos, account credentials or training admission.",
    }
    write(folder / "plan.json", doc)
    print(json.dumps({"planned_sources": len(output), "plan_sha256": sha(folder / "plan.json")}), flush=True)


def acquire(folder):
    protocol = json.loads((folder / "plan.json").read_text())
    require(protocol["input_queue_sha256"] == sha(QUEUE), "Review queue changed")
    require(all(sha(ROOT / name) == value for name, value in protocol["code_sha256"].items()), "Planned execution code changed")
    bodies = folder / "responses"
    bodies.mkdir(exist_ok=False)
    previous = 0.0
    with (folder / "responses.jsonl").open("x") as journal:
        for item in protocol["items"]:
            time.sleep(max(0.0, 0.25 - (time.monotonic() - previous)))
            previous = time.monotonic()
            entry = {"index": item["index"], "source_id": item["source"]["source_id"], "url": item["url"], "observed_at": datetime.now(UTC).isoformat()}
            body = bytearray()
            try:
                with requests.get(item["url"], headers=HEADERS, stream=True, timeout=(10, 20), allow_redirects=False) as response:
                    entry.update({"http_status": response.status_code, "content_type": response.headers.get("Content-Type")})
                    for chunk in response.iter_content(16384):
                        body.extend(chunk[:100001 - len(body)])
                        require(len(body) <= 100000, "Attribution response exceeds size limit")
                    name = f"responses/{item['index']:04d}.bin"
                    (folder / name).write_bytes(body)
                    entry.update({"http_status": response.status_code, "content_type": response.headers.get("Content-Type"), "body": name, "bytes": len(body), "sha256": sha(folder / name)})
                    require(response.status_code == 200, "Attribution endpoint unavailable or non-photo result")
                    require("application/json" in response.headers.get("Content-Type", "").lower(), "JSON attribution response required")
                    entry["attribution"] = inspect_metadata(item["source"], json.loads(body))
                    entry["status"] = "matched_current_provider_metadata"
            except Exception as error:
                if body and "body" not in entry:
                    name = f"responses/{item['index']:04d}.partial.bin"
                    (folder / name).write_bytes(body)
                    entry.update({"body": name, "bytes": len(body), "sha256": sha(folder / name), "partial_body": True})
                entry["status"] = "unresolved"
                entry["error"] = {"type": type(error).__name__, "message": str(error)}
            entry["duration_s"] = time.monotonic() - previous
            journal.write(json.dumps(entry, allow_nan=False) + "\n")
            journal.flush()
            if (item["index"] + 1) % 50 == 0:
                print(json.dumps({"completed": item["index"] + 1, "total": len(protocol["items"])}), flush=True)
    summarize(folder)


def summarize(folder):
    protocol = json.loads((folder / "plan.json").read_text())
    records = [json.loads(line) for line in (folder / "responses.jsonl").read_text().splitlines()]
    require(len(records) == len(protocol["items"]) == 523, "Complete denominator required; partial run retained")
    require([r["index"] for r in records] == list(range(523)), "Ordered complete source records required")
    for record, item in zip(records, protocol["items"], strict=True):
        require(record["source_id"] == item["source"]["source_id"] and record["url"] == item["url"], "Source/response identity changed")
        if record.get("body"):
            require(sha(folder / record["body"]) == record["sha256"], "Attribution body changed")
    def counts(rows):
        return dict(sorted(Counter(r["status"] for r in rows).items()))

    photo_roles = {}
    for item in protocol["items"]:
        identifier = static_identity(item["source"]["coco_image"]["flickr_url"])["photo_id"]
        photo_roles.setdefault(identifier, set()).add(item["source"]["publisher_role"])
    public = {
        "schema": "chipmunk.image-attribution-summary/20", "planned": 523, "completed": len(records),
        "statuses": counts(records), "http_statuses": dict(sorted(Counter(str(r.get("http_status", "network_error")) for r in records).items())),
        "unresolved_reasons": dict(sorted(Counter(r["error"]["message"] if r["error"]["type"] == "ValueError" else r["error"]["type"] for r in records if r["status"] == "unresolved").items())),
        "publisher_roles": {role: counts([r for r, item in zip(records, protocol["items"], strict=True) if item["source"]["publisher_role"] == role]) for role in ("training", "validation", "testing")},
        "plan_sha256": sha(folder / "plan.json"), "journal_sha256": sha(folder / "responses.jsonl"),
        "unique_flickr_photo_ids": len(photo_roles), "flickr_photo_ids_in_multiple_roles": sum(len(roles) > 1 for roles in photo_roles.values()),
        "new_photos_downloaded": 0, "neural_forwards": 0, "training_admitted": 0,
        "limitations": "Provider author/profile/license correspondence only; source pixels, crops, other rights and independent camera authenticity remain unchecked. No outside human review.",
    }
    write(folder / "summary.json", public)
    print(json.dumps(public), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "acquire"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    (plan if args.command == "plan" else acquire)(args.folder)
