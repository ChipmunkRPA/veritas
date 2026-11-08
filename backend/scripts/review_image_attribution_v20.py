"""Independently recount retained provider metadata, without network or models."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


def demand(ok, message):
    if not ok:
        raise ValueError(message)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(url):
    value = urlsplit(url)
    demand(value.scheme in ("https", "http") and not value.query and not value.fragment, "Ambiguous static reference")
    demand(value.netloc == "live.staticflickr.com" or re.fullmatch("farm[0-9]+[.]staticflickr[.]com", value.netloc), "Unknown static host")
    parts = value.path.split("/")
    demand(len(parts) == 3 and parts[1].isdigit(), "Unknown static path")
    match = re.fullmatch(r"([0-9]+)_([a-f0-9]+)(?:_([a-z0-9]+))?[.](?:jpg|jpeg)", parts[2])
    demand(match is not None, "Unknown static image name")
    identifier, secret, size = match.groups()
    return parts[1], int(identifier), secret, size


def owner(url, photo=None):
    value = urlsplit(url)
    demand(value.scheme == "https" and value.netloc == "www.flickr.com" and not value.query and not value.fragment, "Unknown attribution URL")
    parts = value.path.rstrip("/").split("/")
    demand(len(parts) == (3 if photo is None else 4) and parts[:2] == ["", "photos"] and parts[2], "Unknown profile path")
    if photo is not None:
        demand(parts[3] == str(photo), "Wrong attribution photo")
    return parts[2]


def reconstruct(source, doc):
    demand(type(doc) is dict and doc.get("type") == doc.get("flickr_type") == "photo", "Not a photo")
    demand(doc.get("provider_name") == "Flickr" and doc.get("version") == "1.0", "Wrong provider/version")
    server, photo, secret, size = identity(source["coco_image"]["flickr_url"])
    demand(identity(doc.get("url", ""))[:3] == (server, photo, secret), "Source identity mismatch")
    author = doc.get("author_name")
    demand(type(author) is str and 0 < len(author.strip()) <= 1000, "Missing bounded author name")
    demand(owner(doc.get("author_url", "")) == owner(doc.get("web_page", ""), photo), "Wrong owner")
    license_id = source["recorded_license"]["id"]
    demand(license_id in (4, 5), "Unsupported historical license")
    demand(doc.get("license_id") in (license_id, str(license_id)), "Current license mismatch")
    path = "/licenses/" + ("by" if license_id == 4 else "by-sa") + "/2.0/"
    for url in (doc.get("license_url", ""), source["recorded_license"]["url"]):
        demand(url in {"https://creativecommons.org" + path, "http://creativecommons.org" + path}, "License URL mismatch")
    demand(all(type(doc.get(k)) is int and 1 <= doc[k] <= 20000 for k in ("width", "height")), "Invalid dimensions")
    demand(type(doc.get("title", "")) is str and len(doc.get("title", "")) <= 20000, "Invalid title")
    return {
        "source_id": source["source_id"], "publisher_role": source["publisher_role"], "photo_id": photo,
        "author_name": author, "author_url": doc["author_url"], "title": doc.get("title", ""),
        "photo_page": doc["web_page"], "license_id": license_id, "license_url": doc["license_url"],
        "source_identity": {"server": server, "photo_id": photo, "secret": secret, "size": size},
        "expected_large_source_url": "https://live.staticflickr.com/" + server + "/" + str(photo) + "_" + secret + "_b.jpg",
        "provider_size": [doc["width"], doc["height"]],
        "provider_attribution_and_license_match": True, "pixel_lineage_verified": False, "training_admitted": False,
    }


def review(folder, output):
    plan = json.loads((folder / "plan.json").read_text())
    journal = [json.loads(line) for line in (folder / "responses.jsonl").read_text().splitlines()]
    summary = json.loads((folder / "summary.json").read_text())
    demand(len(plan["items"]) == len(journal) == 523, "Complete523 input denominator required")
    demand(summary["planned"] == summary["completed"] == 523, "Public denominator differs")
    root = Path(__file__).resolve().parents[2]
    demand(all(file_sha(root / name) == expected for name, expected in plan["code_sha256"].items()), "Planned execution code changed")
    demand([r["index"] for r in journal] == list(range(523)), "Complete ordered journal required")
    demand(len({r["source_id"] for r in journal}) == 523, "Unique COCO source groups required")
    queue = json.loads((folder.parents[0] / "image-source-rights-v19/review-queue/private-review-queue.json").read_text())
    queue_path = folder.parents[0] / "image-source-rights-v19/review-queue/private-review-queue.json"
    demand(file_sha(queue_path) == plan["input_queue_sha256"], "Private queue commitment differs")
    demand([r["source"] for r in plan["items"]] == queue["sources"], "Full source plan differs")
    roles, all_status, http, reasons, photo_roles = {}, [], [], [], {}
    body_count, matched = 0, 0
    for item, observed in zip(plan["items"], journal, strict=True):
        source = item["source"]
        demand(item["index"] == observed["index"] and source["source_id"] == observed["source_id"] and item["url"] == observed["url"], "Source/request correspondence differs")
        _, photo, _, _ = identity(source["coco_image"]["flickr_url"])
        endpoint = urlsplit(item["url"])
        demand(endpoint.scheme == "https" and endpoint.netloc == "www.flickr.com" and endpoint.path == "/services/oembed/" and not endpoint.fragment, "Unknown metadata endpoint")
        query = parse_qs(endpoint.query)
        demand(set(query) == {"url", "format"} and query["format"] == ["json"] and len(query["url"]) == 1, "Ambiguous query")
        short = urlsplit(query["url"][0])
        demand(short.scheme == "https" and short.netloc == "flic.kr" and short.path.startswith("/p/") and not short.query and not short.fragment, "Unknown short URL")
        decoded = 0
        for char in short.path[3:]:
            decoded = decoded * 58 + "123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ".index(char)
        demand(decoded == photo, "Short URL targets a different photo")
        photo_roles.setdefault(photo, set()).add(source["publisher_role"])
        should_match, parsed = False, None
        if "body" in observed:
            demand(re.fullmatch(rf"responses/{item['index']:04d}(?:[.]partial)?[.]bin", observed["body"]), "Unexpected response filename")
            raw = (folder / observed["body"]).read_bytes()
            demand(len(raw) == observed["bytes"] and hashlib.sha256(raw).hexdigest() == observed["sha256"], "Retained body differs")
            body_count += 1
            if observed.get("http_status") == 200 and "application/json" in observed.get("content_type", "").lower() and not observed.get("partial_body"):
                try:
                    parsed = reconstruct(source, json.loads(raw))
                    should_match = True
                except (ValueError, TypeError, KeyError):
                    should_match = False
        demand((observed["status"] == "matched_current_provider_metadata") == should_match, "Independent eligibility reconstruction differs")
        if should_match:
            demand(all(observed["attribution"].get(k) == value for k, value in parsed.items()), "Full attribution/source field reconstruction differs")
            matched += 1
        else:
            demand(observed["status"] == "unresolved" and "error" in observed, "Unresolved source lacks retained error")
            reasons.append(observed["error"]["message"] if observed["error"]["type"] == "ValueError" else observed["error"]["type"])
        all_status.append(observed["status"])
        http.append(str(observed.get("http_status", "network_error")))
        roles.setdefault(source["publisher_role"], []).append(observed["status"])
    def tally(values):
        return dict(sorted(Counter(values).items()))
    demand(summary["statuses"] == tally(all_status) and summary["http_statuses"] == tally(http) and summary["unresolved_reasons"] == tally(reasons), "Public status counts differ")
    demand(summary["publisher_roles"] == {role: tally(values) for role, values in roles.items()}, "Role denominators differ")
    demand(summary["unique_flickr_photo_ids"] == len(photo_roles) and summary["flickr_photo_ids_in_multiple_roles"] == sum(len(v) > 1 for v in photo_roles.values()), "Underlying photo groups differ")
    demand(summary["plan_sha256"] == file_sha(folder / "plan.json") and summary["journal_sha256"] == file_sha(folder / "responses.jsonl"), "Summary source binding differs")
    demand(summary["training_admitted"] == summary["neural_forwards"] == summary["new_photos_downloaded"] == 0, "Metadata is not training or model validation")
    receipt = {
        "schema": "chipmunk.image-attribution-review/20", "source_requests_reviewed": 523,
        "response_bodies_reviewed": body_count, "matched_source_attributions_reconstructed": matched,
        "complete_counts_and_source_correspondence_match": True, "network_requeried": False,
        "pixel_lineage_or_camera_authenticity_verified": False, "training_admitted": 0, "outside_human_review": False,
        "summary_sha256": file_sha(folder / "summary.json"), "program_sha256": file_sha(__file__),
    }
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    review(args.folder, args.output)
