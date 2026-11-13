"""Check the unchanged writing position point against the frozen Qwen source panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path

_review = runpy.run_path(str(Path(__file__).with_name("review_writing_position_confirmation.py")))
POINT = _review["POINT"]
measure = _review["measure"]
require = _review["require"]

SCHEMA = "chipmunk.writing-position-qwen-confirmation/1"
ADMISSION = "writing-position-qwen-admission-2026-11-12.json"
PANEL = {"gates": 6, "python-peps": 3, "wellcome": 5}
QWEN = {"gates": 5, "python-peps": 3, "wellcome": 5}


def review(result, admission_bytes, prior_bytes):
    require(result["schema"] == SCHEMA and result["point"] == POINT, "Frozen position point changed")
    require(hashlib.sha256(admission_bytes).hexdigest() == result["admission_sha256"]
            and hashlib.sha256(prior_bytes).hexdigest() == result["prior_receipt_sha256"],
            "Input admission or previous study receipt differs")
    admission = json.loads(admission_bytes)
    prior = json.loads(prior_bytes)
    require(admission["frozen_position_point"] == POINT == prior["points"]["interior_position"]
            and admission["detector_runs"] == 0 and admission["minimum_admitted_qwen_works"] == 8
            and admission["minimum_met"] is True
            and admission["initial_score_receipt_sha256"] == result["prior_receipt_sha256"]
            and admission["prior_head_sha256"] == result["prior_head_sha256"],
            "The source freeze does not authorize this confirmation")
    groups = {(row["publisher"], row["group"]) for row in admission["sources"]}
    require(len(groups) == len(admission["sources"]) == 14
            and Counter(row["publisher"] for row in admission["sources"]) == PANEL, "Publisher roster differs")
    cases = result["cases"]
    require(Counter(row["role"] for row in cases) == {"human": 14, "ordinary": 14, "qwen": 13}
            and Counter(row["publisher"] for row in cases if row["role"] == "qwen") == QWEN,
            "Frozen cases differ")
    frozen = {(r["publisher"], r["group"], r["role"]): r for r in admission["cases"]}
    keys = {(r["publisher"], r["group"], r["role"]) for r in cases}
    require(keys == frozen.keys() and len(keys) == len(cases) and all(key[:2] in groups for key in keys),
            "Admission or result case identities differ")
    for row in cases:
        expected = frozen[row["publisher"], row["group"], row["role"]]
        require(expected["text_sha256"] == row["sha256"] and expected["positions"] == [unit[:2] for unit in row["units"]]
                and expected["truth"] == row["truth"] and expected.get("seed") == row.get("seed"), "Admitted source position differs")
        measure(row, generated_roles=("qwen",))
    summary = {}
    for publisher in PANEL:
        subset = [row for row in cases if row["publisher"] == publisher]
        ordinary = [row for row in subset if row["role"] != "qwen"]
        edited = [row for row in subset if row["role"] == "qwen"]
        base_groups = {r["group"] for r in ordinary if r["current"]}
        candidate_groups = {r["group"] for r in ordinary if r["contrast_selected"]}
        summary[publisher] = {
            "originals": PANEL[publisher],
            "ordinary_revisions": PANEL[publisher],
            "qwen_edits": len(edited),
            "incumbent_useful": sum(r["current_useful"] for r in edited),
            "candidate_useful": sum(r["contrast_useful"] for r in edited),
            "additional_useful": sum(r["contrast_useful"] and not r["current_useful"] for r in edited),
            "incumbent_original_flags": sum(bool(r["current"]) for r in ordinary if r["role"] == "human"),
            "incumbent_revision_flags": sum(bool(r["current"]) for r in ordinary if r["role"] == "ordinary"),
            "candidate_original_flags": sum(r["contrast_selected"] for r in ordinary if r["role"] == "human"),
            "candidate_revision_flags": sum(r["contrast_selected"] for r in ordinary if r["role"] == "ordinary"),
            "incumbent_ordinary_groups": len(base_groups),
            "candidate_ordinary_groups": len(candidate_groups),
            "additional_ordinary_groups": len(candidate_groups - base_groups),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--prior", type=Path)
    args = parser.parse_args()
    try:
        directory = args.receipt.parent
        result = review(json.loads(args.receipt.read_bytes()), (args.admission or directory / ADMISSION).read_bytes(),
                        (args.prior or directory / "writing-two-context-transfer-2026-11-12.json").read_bytes())
    except (OSError, KeyError, IndexError, TypeError, ValueError) as error:
        raise SystemExit(f"Review failed. {error}") from None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
