"""Repeat the unchanged PNG check on new source records from the Commons intake."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from scripts import image_png_exact_crop_resize as source
from scripts import review_image_png_exact_crop_resize as auditor
from scripts.image_png_exact_crop_resize import panel

SCHEMA = "chipmunk.image-png-exact-crop-resize-prospective/1"
RULE = {"exact_untouched_minimum": 8, "useful_local_blur_minimum": 8, "useful_local_contrast_minimum": 8,
        "wrong_or_incompatible_maximum": 0, "nuisance_maximum": 0}


def freeze(args):
    earlier_lock = panel.read(args.previous / "freeze.json")
    previous_result = panel.read(args.previous / "result.json")
    earlier_audit = panel.read(args.previous / "audit.json")
    current_method = panel.sha(Path(source.__file__))
    if (earlier_lock["source_sha256"] != current_method
            or previous_result["freeze_sha256"] != panel.sha(args.previous / "freeze.json")
            or earlier_audit["result_sha256"] != panel.sha(args.previous / "result.json")
            or earlier_lock["excluded_sha256"] != panel.sha(args.calibration)):
        raise ValueError("Earlier run, separate calibration roster or exact candidate changed")
    excluded = [*earlier_lock["commons"], *panel.read(args.calibration)["calibration_sources"]]
    if len({row["slot"] for row in excluded}) != 24:
        raise ValueError("The two prior roles do not have distinct recorded slots")
    args.output.mkdir(mode=0o700)
    exclusions = args.output / "excluded.json"
    panel.write(exclusions, {"calibration_sources": [{"slot": row["slot"]} for row in excluded]})
    profile = {
        "schema": SCHEMA, "rule": RULE, "candidate_sha256": current_method, "runner_sha256": panel.sha(Path(__file__)),
        "commons_metadata": {filename: panel.sha(args.commons / filename) for filename in ("plan.json", "acquisition.json")},
        "previous": {filename: panel.sha(args.previous / filename) for filename in ("freeze.json", "result.json", "audit.json")},
        "previous_scope_qualifies": previous_result["scope_qualifies"],
        "exclusions_sha256": panel.sha(exclusions), "calibration_sha256": panel.sha(args.calibration),
        "sources": source.fresh_sources(args.commons, exclusions), "roles": list(source.FRESH_ROLES),
    }
    panel.write(args.output / "freeze.json", profile)


def verify_freeze(args):
    lock = panel.read(args.output / "freeze.json")
    excluded = args.output / "excluded.json"
    if (lock["candidate_sha256"] != panel.sha(Path(source.__file__)) or lock["runner_sha256"] != panel.sha(Path(__file__))
            or lock["rule"] != RULE or lock["exclusions_sha256"] != panel.sha(excluded)
            or lock["sources"] != source.fresh_sources(args.commons, excluded)
            or lock["calibration_sha256"] != panel.sha(args.calibration)
            or any(panel.sha(args.commons / file) != digest for file, digest in lock["commons_metadata"].items())
            or any(panel.sha(args.previous / file) != digest for file, digest in lock["previous"].items())):
        raise ValueError("A fixed source or method changed")
    prior_sources = panel.read(args.previous / "freeze.json")["commons"]
    calibration_sources = panel.read(args.calibration)["calibration_sources"]
    known = {row["sha256"] for row in (*prior_sources, *calibration_sources)}
    if any(row["sha256"] in known for row in lock["sources"]):
        raise ValueError("Prior exact source bytes reentered confirmation")
    return lock


def qualified(outcome):
    return (outcome["roles"]["untouched_crop"]["exact_crop"] >= RULE["exact_untouched_minimum"]
            and outcome["roles"]["local_blur"]["localized"] >= RULE["useful_local_blur_minimum"]
            and outcome["roles"]["local_contrast"]["localized"] >= RULE["useful_local_contrast_minimum"]
            and outcome["wrong_or_incompatible"] <= RULE["wrong_or_incompatible_maximum"]
            and outcome["nuisance"] <= RULE["nuisance_maximum"])


def run(args):
    lock = verify_freeze(args)
    started = time.monotonic()
    cv2.setNumThreads(1)
    rows = list(source.measure(source.fresh(args.commons, lock["sources"])))
    if len(rows) != 120:
        raise ValueError("The prospective panel is incomplete")
    journal = args.output / "rows.json"
    panel.write(journal, rows)
    outcome = source.summarize(rows)
    result = {"schema": SCHEMA, "freeze_sha256": panel.sha(args.output / "freeze.json"),
              "journal_sha256": panel.sha(journal), "outcome": outcome, "qualifies": qualified(outcome),
              "earlier_regression_rule_qualifies": lock["previous_scope_qualifies"],
              "elapsed_s": round(time.monotonic() - started, 3)}
    panel.write(args.output / "result.json", result)
    print(outcome, flush=True)


def audit(args):
    lock = verify_freeze(args)
    cv2.setNumThreads(1)
    journal, result = (args.output / file for file in ("rows.json", "result.json"))
    measured = panel.read(result)
    outcome, rows, proofs = auditor.replay(source.fresh(args.commons, lock["sources"]), panel.read(journal))
    if (rows != 120 or outcome != measured["outcome"] or qualified(outcome) != measured["qualifies"]
            or measured["earlier_regression_rule_qualifies"] != lock["previous_scope_qualifies"]
            or measured["journal_sha256"] != panel.sha(journal)
            or measured["freeze_sha256"] != panel.sha(args.output / "freeze.json")):
        raise ValueError("Prospective outcome differs from independent replay")
    panel.write(args.output / "audit.json", {"inputs": rows, "verified_exact_projections": proofs,
                                            "result_sha256": panel.sha(result), "qualifies": measured["qualifies"]})
    print(f"{rows} inputs and {proofs} projections verified.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "audit"))
    for field in ("previous", "commons", "calibration", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "audit": audit}[args.stage](args)


if __name__ == "__main__":
    main()
