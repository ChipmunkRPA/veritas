"""Publish aggregate writing-span outcomes, never source text or private heads."""

import argparse
from pathlib import Path

from scripts.run_writing_spans_v7 import TRIAL, check, controls, journal, read, receipt, summarize, write
from scripts.writing_span_data_v7 import ROOT, sha
from scripts.writing_span_geometry_v7 import require


def export(output):
    check()
    controls()
    phases = {name: receipt(name) for name in ("fit", "cal", "point", "diagnostic")}
    summary = read(TRIAL / "summary.json")
    require(summary == summarize(journal("diagnostic")), "Exact frozen summary/gates required")
    review = read(TRIAL / "review.json")
    require(review["freeze_sha256"] == sha(TRIAL / "freeze.json"), "Exact local review required")
    profiles = {
        name: read(ROOT / f"output/evaluation/writing-spans-v7-{name}-profile/profile.json")
        for name in (*phases, "review")
    }
    profile_pass = all(p["child_exit_code"] == 0 and p["guard"] is None for p in profiles.values())
    native_pass = review["status"] == "complete" and profile_pass
    numerical_pass = summary["passes_numerical_development_gates"]
    fitted = {kind: read(TRIAL / (kind + "-head.json")) for kind in ("token", "interval")}
    lock = read(TRIAL / "freeze.json")
    body = {
        "schema": "chipmunk.writing-span-public-result/7",
        "source": {
            "citation": "Iqbal, Md Sadiq; Kashem, Mohammod Abul (2026), TXD-22: A Large-Scale Benchmark Dataset for Multi-Class AI-Generated and Mixed-Source Text Identification, Mendeley Data, V1, doi:10.17632/prcjcggtjf.1",
            "url": "https://data.mendeley.com/datasets/prcjcggtjf/1",
            "license": "CC BY 4.0",
            "modifications": "Exact source slices recomposed into controlled160-word documents with explicit source-character targets; source text remains private.",
            "source_sha256": lock["source_sha256"],
            "input_sha256": lock["input_sha256"],
            "independent_authorship_verified": False,
            "consumed_development": True,
        },
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "control_seal_sha256": sha(TRIAL / "control-seal.json"),
        "summary_sha256": sha(TRIAL / "summary.json"),
        "review_sha256": sha(TRIAL / "review.json"),
        "roles": {
            k: {
                "documents": v["documents"],
                "source_groups": 32,
                "actual_pretrained_forwards": v["actual_pretrained_forwards"],
                "receipt_sha256": sha(TRIAL / (k + ".json")),
            }
            for k, v in phases.items()
        },
        "head_parameters": {k: v["parameters"] for k, v in fitted.items()},
        "private_head_sha256": phases["fit"]["head_sha256"],
        "calibration_target": "Retained publisher-AI non-whitespace fraction inside proposed interval; not customer authorship probability.",
        "calibrators": read(TRIAL / "calibrators.json"),
        "genuine_only_research_points": read(TRIAL / "points.json"),
        "summary": summary,
        "local_review": review,
        "resource_profiles": profiles,
        "passes_complete_local_development_gates": numerical_pass and native_pass,
        "serving_admitted": False,
        "customer_confidence_changed": False,
        "github_workflows_executed": 0,
        "outside_human_reviewers": 0,
        "public_source_text_or_vectors_or_learned_weights": False,
        "earlier_authored_runtime_attempt": {
            "status": "failed_coverage_refusal_retained",
            "profile": read(ROOT / "output/evaluation/writing-span-v7-native-smoke-profile/profile.json"),
            "exact_forward_counter_available": False,
        },
        "corrected_authored_runtime_attempt": {
            "receipt": read(ROOT / "output/evaluation/writing-span-v7-native-smoke-v2.json"),
            "profile": read(ROOT / "output/evaluation/writing-span-v7-native-smoke-v2-profile/profile.json"),
        },
        "limitations": [
            "No fresh sources or author IDs; external pretrained-model training overlap is unknown.",
            "Controlled short ASCII source slices with publisher labels, not natural article/email or intellectual-authorship proof.",
            "Nine compositions share each group; rates are not independent-person or SaaS population estimates.",
            "Local execution/refit replay uses the same native libraries and is not outside human review.",
            "The explicit unsupported combining-character probe remains refused; no full Unicode coverage claim.",
            "No serving promotion is permitted by this development protocol, even if numerical gates pass.",
        ],
    }
    import json

    serialized = json.dumps(body, allow_nan=False)
    require(
        not any(s in serialized for s in ("/Users/", "raw_text", "source_parts", '"values"', '"features"')),
        "Private source/path/state fields must not be published",
    )
    write(output, body)
    print(
        json.dumps(
            {
                "aggregate_result_sha256": sha(output),
                "serving_admitted": False,
                "complete_local_development_gates_passed": numerical_pass and native_pass,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    export(parser.parse_args().output)
