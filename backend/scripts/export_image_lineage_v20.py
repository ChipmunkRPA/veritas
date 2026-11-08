"""Publish aggregate lineage findings; keep photographs and attribution pairs private."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from scripts.image_source_rights_v19 import ROOT, require, sha, write

CODE = (
    "backend/scripts/image_attribution_v20.py", "backend/scripts/image_attribution_runtime_v20.py",
    "backend/scripts/review_image_attribution_v20.py", "backend/scripts/image_pixel_lineage_v20.py",
    "backend/scripts/review_image_pixel_lineage_v20.py", "backend/scripts/image_lineage_review_runtime_v20.py",
    "backend/scripts/export_image_lineage_v20.py", "backend/tests/test_image_attribution_v20.py",
    "backend/tests/test_image_pixel_lineage_v20.py", "backend/tests/test_image_lineage_review_v20.py",
)


def read(path):
    return json.loads(Path(path).read_text())


def profile(path):
    result = read(path)
    require(result["child_exit_code"] == 0 and result["guard"] is None and result["sample_errors"] == 0, "Every source/review phase must finish successfully")
    return {k: result[k] for k in ("child_exit_code", "guard", "sampled_tree_rss_lower_bound_bytes", "sample_errors", "duration_s", "seconds_limit")}


def export(attribution, lineage, output):
    attribution, lineage = attribution.resolve(), lineage.resolve()
    meta, meta_review, protocol = read(attribution / "summary.json"), read(attribution / "review.json"), read(lineage / "plan.json")
    require(meta_review["complete_counts_and_source_correspondence_match"] and meta_review["source_requests_reviewed"] == meta["completed"] == 523, "Complete523 attribution audit required")
    require(meta_review["summary_sha256"] == sha(attribution / "summary.json") and meta_review["program_sha256"] == sha(ROOT / CODE[2]), "Unchanged attribution review required")
    require(protocol["attribution_journal_sha256"] == sha(attribution / "responses.jsonl") and protocol["attribution_review_sha256"] == sha(attribution / "review.json"), "Lineage attribution commitments differ")
    require(all(sha(ROOT / p) == value for p, value in protocol["code_sha256"].items()), "Frozen source-pixel code changed")
    observations = {r["index"]: r for r in [json.loads(line) for line in (attribution / "responses.jsonl").read_text().splitlines()]}
    plan = read(attribution / "plan.json")
    require(all(sha(ROOT / p) == value for p, value in plan["code_sha256"].items()), "Attribution acquisition code changed")
    changed, nonphoto = Counter(), Counter()
    for row in observations.values():
        if row.get("http_status") != 200:
            continue
        doc = read(attribution / row["body"])
        if row.get("error", {}).get("message") == "Current and recorded license identifiers differ":
            old = plan["items"][row["index"]]["source"]["recorded_license"]["id"]
            changed[(old, str(doc.get("license_id")), doc.get("license_url") or "")] += 1
        if row.get("error", {}).get("message") == "Flickr photo response required":
            nonphoto[(str(doc.get("type")), str(doc.get("flickr_type")))] += 1
    suites = ET.parse(lineage / "authored-tests.xml").getroot().findall("testsuite")
    require(suites and sum(int(s.attrib["tests"]) for s in suites) == 49 and not any(int(s.attrib[k]) for s in suites for k in ("errors", "failures", "skipped")), "All49 authored checks required")
    summaries, reviews, resources, source_index = {}, {}, {}, []
    for role in ("training", "validation", "testing"):
        summary, review = read(lineage / role / "summary.json"), read(lineage / f"{role}-review.json")
        require(review["passed"] and review["failed_case_count"] == 0 and review["summary_sha256"] == sha(lineage / role / "summary.json") and review["program_sha256"] == sha(ROOT / CODE[4]), "Complete unchanged pixel replay required")
        require(review["cases_reviewed"] == summary["planned"] == summary["completed"], "Complete case denominator required")
        rows = [json.loads(line) for line in (lineage / role / "comparisons.jsonl").read_text().splitlines()]
        require(summary["comparisons_sha256"] == sha(lineage / role / "comparisons.jsonl"), "Pixel journal changed")
        summaries[role], reviews[role] = summary, {k: v for k, v in review.items() if k != "errors"}
        for phase in (role + "-profile", role + "-review-profile"):
            resources[phase] = profile(lineage / phase / "profile.json")
        for row in rows:
            provider = observations[row["index"]]["attribution"]
            for item in [row[key] for key in ("original", "jpeg") if key in row]:
                require(sha(lineage / role / item["file"]) == item["sha256"], "Private photo bytes changed")
            source_index.append({
                "source_id": row["source_id"], "publisher_role": role, "source_index": row["index"],
                "flickr_photo_id": provider["photo_id"], "author_name": provider["author_name"],
                "title": provider["title"], "author_url": provider["author_url"], "photo_page": provider["photo_page"],
                "license_id": provider["license_id"], "license_url": provider["license_url"],
                "original": {**row["original"], "root": str((lineage / role).relative_to(ROOT))} if "original" in row else None,
                "source_jpeg": {**row["jpeg"], "root": str((lineage / role).relative_to(ROOT))} if "jpeg" in row else None,
                "pixel_lineage_status": row["status"], "rgb_sha256": row.get("published_rgb_sha256"),
                "training_admitted": False,
                "remaining_requirement": "Separate dataset/generator-output training-use review, exact positive edit/control pairs and prospective source-separated learning protocol. Source controls are not independently authenticated camera originals.",
            })
    require(len(source_index) == 465 and len({r["flickr_photo_id"] for r in source_index}) == 465, "465 unique underlying photos required")
    private_path = lineage / "private-source-index.json"
    write(private_path, {"schema": "chipmunk.image-source-index/20", "sources": source_index, "attribution_unresolved_sources": 58, "training_admitted": 0})
    require(sha(ROOT / "backend/app/detectors/audio/calibrations/local-speech-v3.json") == "912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf", "Serving voice calibration must remain unchanged")
    resources["attribution-acquisition"] = profile(attribution / "acquire-profile/profile.json")
    resources["attribution-review"] = profile(attribution / "review-profile/profile.json")
    totals = Counter()
    for summary in summaries.values():
        totals.update(summary["statuses"])
    packet = {
        "schema": "chipmunk.image-attribution-lineage-public-package/20", "attribution_summary": meta, "attribution_local_review": meta_review,
        "license_change_diagnostics": [{"historical_license_id": old, "current_license_id": current, "current_license_url": url or None, "count": n} for (old, current, url), n in sorted(changed.items())],
        "nonphoto_diagnostics": [{"type": kind, "flickr_type": subkind, "count": n} for (kind, subkind), n in sorted(nonphoto.items())],
        "pixel_lineage_plan_sha256": sha(lineage / "plan.json"), "pixel_lineage": summaries, "pixel_local_reviews": reviews,
        "total_pixel_statuses": dict(sorted(totals.items())), "native_dual_decoder_comparisons": sum(r["native_pillow_opencv_decode_comparisons"] for r in reviews.values()),
        "private_source_index_sha256": sha(private_path), "local_resource_profiles": resources,
        "authored_tests_passed": 49, "junit_sha256": sha(lineage / "authored-tests.xml"), "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "primary_terms_snapshots": read(ROOT / "backend/data/independent-evaluation/image-source-terms-v20/receipt.json"),
        "retained_preparation_failure": read(ROOT / "backend/data/independent-evaluation/image-pixel-lineage-v20/failed-plan.json"),
        "training_admitted": 0, "new_neural_forwards": 0, "serving_or_confidence_changed": False, "outside_human_reviewers": 0,
        "limitations": [
            "Exact current source RGB correspondence is not detection accuracy or independent camera authentication.",
            "Changed license records include several still-permissive schemas; the strict matching audit did not reclassify them as cleared or uniformly restricted.",
            "Mismatched RGB can reflect image edits, server reencoding or historical decoding; it is not proof of AI manipulation or loss of rights.",
            "Review replays retained bytes with Pillow/OpenCV and separate integer geometry; it does not reacquire remote archives independently.",
            "No photographs, new photo IDs/pairings, individual photographer records, neural maps or learned weights are in this public packet.",
        ],
    }
    write(output, packet)
    print(json.dumps({"public_sha256": sha(output), "pixel_statuses": dict(totals), "tests": 49, "training_admitted": 0}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attribution", type=Path)
    parser.add_argument("lineage", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.attribution, args.lineage, args.output)
