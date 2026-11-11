"""Evaluate new writing sources with the exact published two-sided fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

from scripts import writing_two_sided_context as original
from scripts.writing_focus_v3 import digest, text_sha

ROOT = Path(__file__).resolve().parents[2]
PRIOR_REPORT = ROOT / "docs/evaluation-panels/writing-two-sided-context-2026-11-11.json"
FIT_SHA256 = "7ecab534f10c7701810a237828f00079e81861282ade9034909ea64477f84054"
METHOD_SHA256 = "67844425de4758e8901c2adb6de843e178a1a8f0d9b34d0219997787e1568a36"
SCHEMA = "chipmunk.writing-two-sided-transfer/1"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def fit() -> dict:
    data = read(PRIOR_REPORT)
    fitted = data["fitted"]
    if (text_sha(json.dumps(fitted, sort_keys=True, indent=2, allow_nan=False) + "\n") != FIT_SHA256
            or data["commitments"]["fit"] != FIT_SHA256 or digest(Path(original.__file__)) != METHOD_SHA256):
        raise ValueError("Published fit or original method differs from the frozen comparison")
    return fitted


def lock(directory: Path) -> dict:
    fit()
    sources = read(directory / "source.json")
    prior_authors = {item["author"] for item in read(PRIOR_REPORT)["source_roster"]}
    authors = [name for item in sources["groups"] for name in item["authors"]]
    if len(authors) != len(set(authors)) or prior_authors.intersection(authors):
        raise ValueError("The new publication-author groups must be disjoint")
    return {
        "schema": SCHEMA,
        "fit_sha256": FIT_SHA256,
        "prior_report_sha256": digest(PRIOR_REPORT),
        "method_sha256": METHOD_SHA256,
        "transfer_runner_sha256": digest(Path(__file__)),
        "holdout_sha256": digest(directory / "private-holdout.jsonl"),
        "source_sha256": digest(directory / "source.json"),
        "generation_sha256": digest(directory / "generation-freeze.json"),
        "admission_sha256": digest(directory / "admission.json"),
        "classifier_assets": original.BASELINE_ASSETS,
        "classification_code_sha256": {
            key: digest(original.ROOT / key) for key in ("app/detectors/text/extract.py",
                "app/detectors/text/segment.py", "app/detectors/text/classifier.py")
        },
    }


def score(directory: Path, snapshot: Path) -> None:
    if read(directory / "frozen.json") != lock(directory):
        raise ValueError("Holdout differs from the freeze")
    fitted = fit()
    cases = original.read_cases(directory / "private-holdout.jsonl")
    if {row["group"] for row in cases} != {row["group"] for row in read(directory / "source.json")["groups"]}:
        raise ValueError("Holdout groups differ from the frozen source roster")
    start = monotonic()
    classifier = original.open_classifier(snapshot)
    forwards = [0]
    hook = classifier.model.register_forward_pre_hook(lambda *_: forwards.__setitem__(0, forwards[0] + 1))
    try:
        measured = list(zip(cases, original.score_cases(classifier, cases, lambda: None), strict=True))
    finally:
        hook.remove()
    original.write(directory / "private-native.json", {case["case"]: rows for case, rows in measured})
    result = original.aggregate(measured, fitted)
    result.update({
        "schema": SCHEMA, "model_forward_calls": forwards[0], "fitted": fitted,
        "source_groups": len({row["group"] for row in cases}),
        "elapsed_seconds": round(monotonic() - start, 3),
        "commitments": {"freeze": digest(directory / "frozen.json"), "fit": FIT_SHA256,
                        "native": digest(directory / "private-native.json")},
    })
    original.write(directory / "aggregate.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        original.write(args.directory / "frozen.json", lock(args.directory))
    elif args.snapshot is None:
        parser.error("A local classifier snapshot is required")
    else:
        score(args.directory, args.snapshot)


if __name__ == "__main__":
    main()
