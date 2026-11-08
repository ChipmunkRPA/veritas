"""Aggregate completed localizer outcomes without source media or response maps."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts import image_relayformer_v18 as method
from scripts import run_relayformer_v18 as run


def export(output):
    run.check()
    run.points()
    rows = run.journal("diagnostic")
    original = run.read(run.TRIAL / "result.json")
    method.require(original == method.aggregate(rows), "Exact frozen summary required")
    review = run.read(run.TRIAL / "review.json")
    profiles = {
        name: run.read(run.ROOT / f"output/evaluation/relayformer-v18-{name}-profile/profile.json")
        for name in ("freeze", "point", "diagnostic", "review")
    }
    quality = review["status"] == "complete" and all(
        p["child_exit_code"] == 0 and p["guard"] is None for p in profiles.values()
    )
    phases = {name: run.read(run.TRIAL / (name + ".json")) for name in ("point", "diagnostic")}
    for name, phase in phases.items():
        method.require(
            phase["status"] == "complete"
            and phase["journal_sha256"] == run.prior.sha(run.TRIAL / (name + ".jsonl")),
            "Complete bound phase required",
        )
    ungated = {}
    families = {}
    for fmt in run.prior.FORMATS:
        part = [r for r in rows if r["format"] == fmt]
        ungated[fmt] = {
            "scope": "Post-result descriptive gate diagnosis; no changed criterion or alternate advancement",
            "useful_edits": sum(r["kind"] == "ai" and method.useful(r["ungated"]["candidate"]) for r in part),
            "original_highlights": sum(
                r["kind"] == "original" and r["ungated"]["candidate"]["predicted_positive_pixels"] > 0
                for r in part
            ),
            "ordinary_highlights": sum(
                r["kind"] == "ordinary" and r["ungated"]["candidate"]["predicted_positive_pixels"] > 0
                for r in part
            ),
        }
        families[fmt] = {}
        for family in run.prior.FAMILIES:
            subset = [r for r in part if r["family"] == family]
            families[fmt][family] = {}
            for arm in ("current", "candidate"):
                families[fmt][family][arm] = {
                    "edited": 16,
                    "useful_locations": sum(
                        r["kind"] == "ai" and method.useful(r["metrics"][arm]) for r in subset
                    ),
                    "original_highlights": sum(
                        r["kind"] == "original" and r["metrics"][arm]["predicted_positive_pixels"] > 0
                        for r in subset
                    ),
                    "ordinary_highlights": sum(
                        r["kind"] == "ordinary" and r["metrics"][arm]["predicted_positive_pixels"] > 0
                        for r in subset
                    ),
                }
    body = {
        "schema": "chipmunk.relayformer-public-result/18",
        "freeze_sha256": run.prior.sha(run.TRIAL / "freeze.json"),
        "point_seal_sha256": run.prior.sha(run.TRIAL / "point-seal.json"),
        "result_sha256": run.prior.sha(run.TRIAL / "result.json"),
        "review_sha256": run.prior.sha(run.TRIAL / "review.json"),
        "phases": phases,
        "points": run.read(run.TRIAL / "points.json"),
        "summary": original,
        "local_review": review,
        "resource_profiles": profiles,
        "family_diagnostics": families,
        "ungated_descriptive_diagnosis": ungated,
        "candidate_advance_to_fresh_confirmation": original["advance_to_separate_fresh_confirmation_arm"]
        if quality
        else None,
        "complete_local_replay_passed": quality,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "canonical_path": "Unmerged CPUfloat32,complete source maps retained losslessly; exact integer component and point ranking",
        "reference": "Input-bound retained current-selector outputs; no new full SaaS or baseline neural run",
        "source": "TGIF/TGIF2 publisher CC BY-SA4.0; underlying photograph rights and training overlap not independently cleared; evaluation only, no redistribution",
        "serving_admitted": False,
        "used_in_confidence": False,
        "github_workflows_executed": 0,
        "outside_human_reviewers": 0,
        "raw_photos_or_maps_or_checkpoint_values_published": False,
        "limitations": [
            "Consumed project source groups,not fresh accuracy or independently verified pixel authorship.",
            "Raw sigmoid maps are not calibrated AI-origin probabilities.",
            "Model plus complete readout tested together; checkpoint/renderer effects not isolated.",
            "Local repeat and independent component arithmetic do not replace external human review.",
            "Historical failed IML audits remain failed; this new graph/review does not clear them.",
        ],
    }
    text = json.dumps(body, indent=2, allow_nan=False)
    method.require(
        "/Users/" not in text and '"values"' not in text, "Private paths/tensors cannot be published"
    )
    run.write(output, body)
    print(
        json.dumps(
            {
                "aggregate_result_sha256": run.prior.sha(output),
                "complete_local_replay_passed": quality,
                "advance_to_fresh_confirmation": body["candidate_advance_to_fresh_confirmation"],
                "serving_admitted": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    export(parser.parse_args().output)
