"""Compare two independent runs of the same frozen evaluation method."""

from __future__ import annotations

from collections import Counter, defaultdict

from app.bench import independent

SCHEMA = "chipmunk.independent-replication/1"


def replicate(manifest, reference, replication, lock):
    """Count numerical and decision differences without disclosing case rows."""
    for receipt in (reference, replication):
        independent.summarize(manifest, receipt, lock, require_freeze_commitment=True)
    first = {row["id"]: row for row in independent.audit_receipt(manifest, reference, "confirmation")}
    second = independent.audit_receipt(manifest, replication, "confirmation")
    totals = defaultdict(Counter)
    differences = defaultdict(set)
    for row in second:
        before = first[row["id"]]
        key = row["modality"], row["condition"], row["label"]
        counts = totals[key]
        counts["planned"] += 1
        same_status = before["status"] == row["status"]
        assessed = before["status"] == row["status"] == "ok"
        score_changed = assessed and before["response"] != row["response"]
        point = lock["thresholds"][row["modality"]]
        before_flag = before["status"] == "ok" and before["response"] >= point
        after_flag = row["status"] == "ok" and row["response"] >= point
        location_before, location_after = before.get("localization"), row.get("localization")
        location_planned = location_before is not None
        location_status_changed = location_planned and location_before["status"] != location_after["status"]
        locations_measured = location_planned and location_before["status"] == location_after["status"] == "ok"
        counts_changed = locations_measured and any(location_before[field] != location_after[field]
                                                   for field in ("reported_units", "overlap_units"))
        counts.update(both_scored=assessed, changed_status=not same_status, changed_score=score_changed,
                      lost_flags=before_flag and not after_flag, added_flags=after_flag and not before_flag,
                      planned_locations=location_planned, both_locations_measured=locations_measured,
                      changed_location_status=location_status_changed, changed_overlap_counts=counts_changed)
        if not same_status or score_changed or location_status_changed or counts_changed:
            differences[key].add(row["group"])
    return {
        "schema": SCHEMA, "manifest_sha256": independent.digest(manifest), "freeze_sha256": independent.digest(lock),
        "reference_sha256": independent.digest(reference), "replication_sha256": independent.digest(replication),
        "strata": [{"modality": modality, "condition": condition, "label": label,
                    "groups_with_differences": len(differences[modality, condition, label]), **counts}
                   for (modality, condition, label), counts in sorted(totals.items())],
        "scope": "Exact numeric responses and overlap counts on the same declared files. Matching counts can cover different locations.",
    }
