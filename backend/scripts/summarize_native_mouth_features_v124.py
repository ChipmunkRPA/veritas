"""Publish privacy-safe aggregates after all eight feature/replay batches pass."""

from __future__ import annotations

import json
from fractions import Fraction

from scripts import review_native_mouth_features_v124 as review

f = review.features


def summarize():
    review.verify()
    profiles, receipts, audits, commitments, rows = [], [], [], [], []
    for batch in range(8):
        folder, result = review.completed_batch(batch)
        stage = review.TRIAL / f"bind-{batch:02d}"
        audit, profile = review.read(stage / "output/result.json"), review.read(stage / "profile.json")
        review.require(
            audit["status"] == "complete"
            and audit["batch"] == batch
            and profile["child_exit_code"] == 0
            and profile["guard"] is None
            and profile["sample_errors"] == 0
            and profile["worker_receipt_sha256"] == review.sha(stage / "output/result.json")
            and audit["private_review_journal_sha256"] == review.sha(stage / "output/reviewed-cases.jsonl")
            and audit["extraction_receipt_sha256"] == review.sha(folder / "output/result.json")
            and audit["freeze_sha256"] == review.sha(review.TRIAL / "freeze.json"),
            "Successful exact complete feature audit required",
        )
        rows.extend(
            json.loads(line) for line in (folder / "output/feature-cases.jsonl").read_text().splitlines()
        )
        profiles.append(review.read(folder / "profile.json"))
        receipts.append(result)
        audits.append(audit)
        commitments.append(
            {
                "batch": batch,
                "feature_result_sha256": review.sha(folder / "output/result.json"),
                "feature_profile_sha256": review.sha(folder / "profile.json"),
                "feature_journal_sha256": result["feature_case_journal_sha256"],
                "audit_result_sha256": review.sha(stage / "output/result.json"),
                "audit_profile_sha256": review.sha(stage / "profile.json"),
                "audit_journal_sha256": audit["private_review_journal_sha256"],
            }
        )
    review.require(
        len(rows) == 288 and sum(r["status"] == "extracted" for r in rows) == 283,
        "Complete predeclared bank required",
    )
    counts = {}
    for condition in f.clips.CONDITIONS:
        selected = [r for r in rows if r["condition"] == condition]
        paired_centres = 0
        for row in selected:
            if row["status"] != "extracted":
                continue
            from app.detectors.video.decode import FrameTable

            table = FrameTable.from_json(row["table"])
            available = {r["frame_index"] for r in row["frames"] if r["status"] == "ok"}
            speech = set(row["speech"]["indices"])
            for i in range(1, len(table) - 1):
                if i in speech and all(j in available for j in (i - 1, i, i + 1)):
                    times = [Fraction(table.pts[j]) * table.tb for j in (i - 1, i, i + 1)]
                    if max(times[1] - times[0], times[2] - times[1]) <= Fraction(2, 25):
                        paired_centres += 1
        counts[condition] = {
            "planned": len(selected),
            "extracted": sum(r["status"] == "extracted" for r in selected),
            "unavailable": sum(r["status"] == "unavailable" for r in selected),
            "native_pictures": sum(len(r.get("frames", [])) for r in selected),
            "assessed_mouth_pictures": sum(r.get("shape", [0])[0] for r in selected),
            "eligible_common_image_speech_centres": paired_centres,
        }
    qualification = f.TRIAL / "authored"
    q, p = review.read(qualification / "output/result.json"), review.read(qualification / "profile.json")
    review.require(
        p["child_exit_code"] == 0
        and p["guard"] is None
        and p["sample_errors"] == 0
        and p["worker_receipt_sha256"] == review.sha(qualification / "output/result.json"),
        "Exact encoder qualification required",
    )
    body = {
        "schema": "chipmunk.native-mouth-features-complete/124",
        "status": "complete",
        "planned_cases": len(rows),
        "extracted_cases": 283,
        "unavailable_cases": 5,
        "native_pictures": sum(r["native_pictures"] for r in receipts),
        "assessed_mouth_pictures": sum(r["assessed_mouth_pictures"] for r in receipts),
        "extraction_new_neural_calls": sum(r["new_neural_calls"] for r in receipts),
        "qualification_new_neural_calls": q["new_neural_calls"],
        "audit_new_neural_calls": sum(r["new_neural_calls"] for r in audits),
        "face_inputs": sum(r["face_inputs"] for r in receipts),
        "image_inputs": sum(r["image_inputs"] for r in receipts),
        "exact_rgb_cache_hits": sum(r["exact_rgb_cache_hits"] for r in receipts),
        "unique_source_speech_positions": sum(r["unique_source_speech_positions"] for r in audits),
        "conditions": counts,
        "extraction_duration_s_sum": sum(r["duration_s"] for r in profiles),
        "extraction_max_sampled_descendant_rss_bytes": max(
            r["sampled_tree_rss_lower_bound_bytes"] for r in profiles
        ),
        "extraction_minimum_sampled_available_ram_bytes": min(
            r["minimum_available_ram_bytes"] for r in profiles
        ),
        "feature_freeze_sha256": review.sha(f.PUBLIC),
        "audit_freeze_sha256": review.sha(review.PUBLIC),
        "summarizer_sha256": review.sha(f.ROOT / "backend/scripts/summarize_native_mouth_features_v124.py"),
        "batches": commitments,
        "defaults_changed": False,
        "scope": "All native pictures and cached speech-time routes, replayed from actual clips and arrays. No classifier fitted, calibrated probability, detector accuracy or achieved phonetic-sync claim. All face/voice vectors and individual media fingerprints remain private. Missing scope does not establish originality.",
    }
    review.require(
        body["native_pictures"] == sum(r["native_pictures"] for r in audits)
        and body["assessed_mouth_pictures"] == sum(r["assessed_mouth_pictures"] for r in audits),
        "Extraction/replay aggregate mismatch",
    )
    review.write(f.PANEL / "native-mouth-features-complete-v124.json", body)
    print(json.dumps({k: v for k, v in body.items() if k not in {"batches", "conditions"}}), flush=True)


if __name__ == "__main__":
    summarize()
