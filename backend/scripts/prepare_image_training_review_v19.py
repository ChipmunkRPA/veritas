"""Queue source-rights checks, not training: group photos and exclude used tests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scripts.image_source_rights_v19 import ROOT, sha, write


def select(rows):
    if len({r["source_id"] for r in rows}) != len(rows):
        raise ValueError("One record per underlying source photo required")
    queue = []
    for row in rows:
        if row["license_status"] not in {"metadata_by", "metadata_by_sa"}:
            continue
        roles = row["publisher_roles"]
        if len(roles["original"]) != 1:
            raise ValueError("Ambiguous base source role")
        role = roles["original"][0]
        if role not in {"training", "validation", "testing"}:
            raise ValueError("Unknown publisher role")
        if any(len(roles[f]) > 1 for f in ("firefly", "sd2", "flux-fill")):
            raise ValueError("Source leakage across a generator's roles")
        if any(roles[f] and roles[f] != [role] for f in ("firefly", "sd2", "flux-fill")):
            raise ValueError("Generator/base source roles disagree")
        if not all(roles[f] == [role] for f in ("firefly", "sd2", "flux-fill")):
            continue
        if row["consumed_local_evaluation"]:
            continue
        if row["training_admitted"] or row["attribution_verified"] or row["source_bytes_lineage_verified"]:
            raise ValueError("New rights evidence requires a separately versioned review")
        queue.append({
            "source_id": row["source_id"], "publisher_role": role,
            "recorded_license": row["recorded_license"], "coco_image": row["coco_image"],
            "ready_for_training": False,
            "required_checks": ["photographer attribution", "higher-resolution source-byte lineage", "crop/orientation reconstruction", "data/generator/source terms and ShareAlike obligations"],
        })
    return sorted(queue, key=lambda r: hashlib.sha256(f"chipmunk-photo-rights-review-v19/{r['source_id']}".encode()).hexdigest())


def prepare(folder, output):
    output.mkdir(parents=True, exist_ok=False)
    review = json.loads((folder / "review.json").read_text())
    summary_path = folder / "audit/summary.json"
    private_path = folder / "audit/private-sources.json"
    bindings = json.loads((folder / "audit/bindings.json").read_text())
    if not (review["public_metadata_counts_match"] and review["private_source_rows_match"]
            and review["private_consumed_inventory_assessed"]
            and review["review_program_sha256"] == sha(ROOT / "backend/scripts/review_image_rights_v19.py")
            and review["summary_sha256"] == bindings["summary_sha256"] == sha(summary_path)
            and bindings["private_sources_sha256"] == sha(private_path)):
        raise ValueError("Complete source review and unchanged private/source bindings required")
    for name, expected in json.loads(summary_path.read_text())["input_sha256"].items():
        if sha(folder / "source" / name) != expected:
            raise ValueError("Reviewed primary-source metadata changed")
    rows = json.loads(private_path.read_text())["sources"]
    queue = select(rows)
    diagnostics, roles = {}, {}
    for role in ("training", "validation", "testing"):
        with (folder / f"source/metadata/orig_{role}.csv").open(newline="") as stream:
            records = list(csv.DictReader(stream))
        pairs = {(r["image_id"], r["category"]) for r in records}
        diagnostics[role] = {"rows": len(records), "distinct_source_category_pairs": len(pairs), "distinct_photos": len({r["image_id"] for r in records})}
        compatible = [r for r in rows if role in r["publisher_roles"]["original"] and r["license_status"] in {"metadata_by", "metadata_by_sa"}]
        common = [r for r in compatible if all(r["publisher_roles"][f] == [role] for f in ("firefly", "sd2", "flux-fill"))]
        roles[role] = {
            "preliminary_metadata_compatible": len(compatible), "all_three_generators": len(common),
            "all_three_already_locally_consumed": sum(r["consumed_local_evaluation"] for r in common),
            "not_locally_consumed_review_queue": sum(r["publisher_role"] == role for r in queue),
            "admitted_to_training": 0,
        }
    write(output / "private-review-queue.json", {"schema": "chipmunk.image-photo-review-queue/19", "sources": queue})
    public = {
        "schema": "chipmunk.image-photo-review-queue-summary/19",
        "audited_summary_sha256": sha(summary_path), "source_review_sha256": sha(folder / "review.json"),
        "private_queue_sha256": sha(output / "private-review-queue.json"),
        "program_sha256": sha(__file__), "source_category_diagnostics": diagnostics, "publisher_roles": roles,
        "queued_source_photos": len(queue), "admitted_to_training": 0, "neural_forwards": 0,
        "purpose": "Prioritized private rights/lineage review only; these are not frozen fit/calibration/confirmation assignments.",
        "limitations": [
            "Not locally consumed does not establish freshness, permission, source authenticity or absence of external pretraining overlap.",
            "The FLUX original CSV filenames are not used to assign roles; actual original/generation source IDs determine the queue.",
            "All category/generator/format variants of a photograph must stay one statistical source group.",
            "No photo is admitted until the required rights and lineage checks are completed in a separate version.",
        ],
    }
    write(output / "summary.json", public)
    print(json.dumps({"queued_source_photos": len(queue), "publisher_roles": roles, "admitted_to_training": 0}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.folder, args.output)
