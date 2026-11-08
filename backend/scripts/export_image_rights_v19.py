"""Export only reviewed aggregate metadata; keep photo pairings private."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.image_source_rights_v19 import ROOT, sha, write

CODE = (
    "backend/scripts/image_source_rights_v19.py", "backend/scripts/image_rights_runtime_v19.py",
    "backend/scripts/review_image_rights_v19.py", "backend/scripts/prepare_image_training_review_v19.py",
    "backend/scripts/export_image_rights_v19.py", "backend/tests/test_image_source_rights_v19.py",
    "backend/tests/test_image_training_review_v19.py", "backend/scripts/specialist_runtime_v4.py",
)


def export(folder, output):
    def read(name):
        return json.loads((folder / name).read_text())
    summary, review, public_review, queue = (read(n) for n in ("audit/summary.json", "review.json", "public-review.json", "review-queue/summary.json"))
    bindings, acquired = read("audit/bindings.json"), read("source/acquisition.json")
    conditions = [
        acquired["completed"] is True,
        acquired["program_sha256"] == bindings["program_sha256"] == sha(ROOT / CODE[0]),
        sha(folder / "source/acquisition.json") == bindings["acquisition_sha256"],
        sha(folder / "audit/private-sources.json") == bindings["private_sources_sha256"],
        review["private_source_rows_match"] is True and review["private_consumed_inventory_assessed"] is True,
        public_review["public_metadata_counts_match"] is True and public_review["private_consumed_inventory_assessed"] is False,
        sha(folder / "audit/summary.json") == review["summary_sha256"] == public_review["summary_sha256"] == queue["audited_summary_sha256"],
        sha(ROOT / CODE[2]) == review["review_program_sha256"] == public_review["review_program_sha256"],
        sha(ROOT / CODE[3]) == queue["program_sha256"],
        sha(folder / "review.json") == queue["source_review_sha256"],
        sha(folder / "review-queue/private-review-queue.json") == queue["private_queue_sha256"],
        summary["training_eligible_after_full_review"] == queue["admitted_to_training"] == 0,
        sha(ROOT / "backend/app/detectors/audio/calibrations/local-speech-v3.json") == "912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf",
    ]
    if not all(conditions):
        raise ValueError("Complete unchanged reviewed metadata and preserved voice calibration required")
    suites = ET.parse(folder / "authored-tests.xml").getroot().findall("testsuite")
    if not suites or sum(int(s.attrib["tests"]) for s in suites) != 52 or any(int(s.attrib[k]) for s in suites for k in ("errors", "failures", "skipped")):
        raise ValueError("All52 authored checks must pass without omissions")
    profiles = {}
    for name in ("acquire-profile", "audit-profile", "review-profile", "public-review-profile"):
        profile = read(name + "/profile.json")
        if profile["child_exit_code"] != 0 or profile["guard"] is not None or profile["sample_errors"] != 0:
            raise ValueError("Retain and resolve any failed resource/review phase")
        profiles[name] = {k: profile[k] for k in ("child_exit_code", "guard", "sampled_tree_rss_lower_bound_bytes", "sample_errors", "duration_s", "seconds_limit")}
    packet = {
        "schema": "chipmunk.image-source-rights-public-package/19", "summary": summary,
        "separate_local_full_review": review, "public_only_local_review": public_review,
        "rights_review_queue_summary": queue, "guarded_phases": profiles,
        "verification": {"authored_tests_passed": 52, "authored_tests_skipped": 0, "junit_sha256": sha(folder / "authored-tests.xml"), "code_sha256": {name: sha(ROOT / name) for name in CODE}},
        "publication": "Aggregate counts, public license tables and hashes only. No photographs, new source IDs/pairings, full private inventories, neural maps or private weights.",
        "release_decision": "No training admission, model promotion, new inference, overall-confidence change, cloud deployment or outside human review.",
    }
    write(output, packet)
    print(json.dumps({"public_package_sha256": sha(output), "all_sources": summary["all_sources"]["sources"], "rights_review_queue": queue["queued_source_photos"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.folder, args.output)
