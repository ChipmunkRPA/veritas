# SPDX-License-Identifier: Apache-2.0
"""Replay sanitized short lip-timing coverage, edit localization and control counts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

POSITIVES = {"local_earlier", "local_later"}
CONTROLS = {
    "unaltered",
    "local_gain",
    "local_filter",
    "local_background",
    "global_earlier",
    "global_later",
    "paired_edit",
}


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, problem):
    if not ok:
        raise ValueError(problem)


def replay(rows, point):
    totals = {role: Counter() for role in sorted(POSITIVES | CONTROLS)}
    groups = set()
    for row in rows:
        role, expected, proposal = row["role"], row["truth"], row["proposal"]
        require((role in POSITIVES) == (expected is not None), "Truth and case role differ")
        measured = row["mouth_events"] >= 3 and row["acoustic_events"] >= 3
        require(row["measured"] == measured and (not proposal or measured), "Coverage or proposal disagrees")
        flag = bool(proposal and proposal["margin"] >= 0.12 and proposal["score"] > point)
        correct = bool(
            proposal
            and expected
            and max(abs(x - y) for x, y in zip(proposal["times_s"], expected, strict=True)) <= 0.08
        )
        require(correct == row["raw_localized"] and flag == row["flag"], "Recorded decision does not replay")
        counts = totals[role]
        counts["cases"] += 1
        counts["measured"] += measured
        counts["proposals"] += bool(proposal)
        counts["flags"] += flag
        counts["raw_localized"] += correct
        counts["localized"] += correct and flag
        if correct and flag:
            groups.add(row["group"])
    useful = sum(totals[k]["localized"] for k in POSITIVES)
    false = sum(totals[k]["flags"] for k in CONTROLS)
    assessed = sum(k["measured"] for k in totals.values())
    essentials = sum(
        totals[k]["flags"] for k in ("unaltered", "global_earlier", "global_later", "paired_edit")
    )
    return {
        "by_role": totals,
        "measured": assessed,
        "cases": len(rows),
        "correct_signed_localizations": useful,
        "correct_groups": len(groups),
        "ordinary_false_flags": false,
        "gate": {
            "useful": useful >= 8,
            "group_spread": len(groups) >= 4,
            "signs": all(totals[k]["localized"] >= 2 for k in POSITIVES),
            "coverage": 20 * assessed >= 7 * len(rows),
            "false_flags": false <= 1,
            "essential_controls": essentials == 0,
        },
    }


def assemble(run):
    method, development, held = (run / f"{k}.json" for k in ("method-freeze", "development", "held"))
    lock = read(method)
    point = lock["point"]
    held_doc = read(held)
    observed = run / "observations-held.json"
    native = read(observed)
    require(
        lock["development_sha256"] == sha(development)
        and held_doc["method_freeze_sha256"] == sha(method)
        and held_doc["observation_sha256"] == sha(observed),
        "Frozen study commitments differ",
    )
    result = {
        "schema": "chipmunk.short-lip-release-timing/1",
        "method_sha256": sha(method),
        "prior_development_sha256": sha(development),
        "held_run_sha256": sha(held),
        "held_source_receipt_sha256": native["source_receipt_sha256"],
        "protocol": lock["spec"],
        "candidate": lock["candidate"],
        "point": point,
        "splits": {},
    }
    for split, data in (("development", read(development)["cases"]), ("held", held_doc["cases"])):
        rows = []
        for case in data:
            truth, candidate = case["truth"], case["candidate"]
            rows.append(
                {
                    "clip": case["ordinal"],
                    "group": case["group"],
                    "style": case["style"],
                    "role": case["role"],
                    "mouth_events": len(case["video_events"]),
                    "acoustic_events": len(case["sound_events"]),
                    "measured": case["measured"],
                    "raw_localized": case["correct"],
                    "truth": [truth["mouth_s"], truth["audio_s"], truth["shift_s"]] if truth else None,
                    "proposal": {
                        "score": candidate["score"],
                        "margin": candidate["alternative_margin"],
                        "times_s": [*candidate["event_seconds"], candidate["shift_s"]],
                    }
                    if candidate
                    else None,
                    "flag": bool(
                        candidate and candidate["alternative_margin"] >= 0.12 and candidate["score"] > point
                    ),
                }
            )
            require(case.get("flag", rows[-1]["flag"]) == rows[-1]["flag"], "Held decision does not replay")
        result["splits"][split] = {"rows": rows, "summary": replay(rows, point)}
    result["held_source_clock"] = {
        "verified_native_picture_ticks": sum(len(row["pts"]) for row in native["rows"]),
        "verified_audio_samples": sum(row["audio_samples"] for row in native["rows"]),
        "signed_16_originals": sum(row["source_sample_bytes"] == 2 for row in native["rows"]),
        "float_32_originals": sum(row["source_sample_bytes"] == 4 for row in native["rows"]),
    }
    audit_path = run / "native-independent-audit.json"
    audit = read(audit_path)
    require(
        audit["observations_sha256"] == sha(observed)
        and audit["scored_sha256"] == sha(held)
        and all(
            audit[k] == result["held_source_clock"][k]
            for k in ("verified_native_picture_ticks", "verified_audio_samples")
        ),
        "Independent native audit differs from scored media",
    )
    result["independent_native_audit"] = {"ledger_sha256": sha(audit_path), **audit}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("--assemble", type=Path)
    a = p.parse_args()
    if a.assemble:
        doc = assemble(a.assemble)
        a.receipt.write_text(json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + "\n")
    doc = read(a.receipt)
    require(
        doc["schema"] == "chipmunk.short-lip-release-timing/1"
        and set(doc["splits"]) == {"development", "held"},
        "Unknown report or incomplete study",
    )
    for split, record in doc["splits"].items():
        rows = record["rows"]
        require(
            len({r["clip"] for r in rows}) == (32 if split == "development" else 48)
            and len({r["group"] for r in rows}) == (8 if split == "development" else 12),
            "Incomplete source group roster",
        )
        require(replay(rows, doc["point"]) == record["summary"], "Published totals differ")
        print(split, json.dumps(record["summary"], sort_keys=True))
    held = doc["splits"]["held"]["rows"]
    audit = doc["independent_native_audit"]
    require(
        audit["scored_sha256"] == doc["held_run_sha256"]
        and audit["raw_useful_proposals"] == sum(row["raw_localized"] for row in held)
        and audit["exact_earlier_source_witnesses"] == sum(row["role"] == "local_earlier" for row in held)
        and audit["exact_later_source_witnesses"] == sum(row["role"] == "local_later" for row in held),
        "Independent witness counts differ from frozen held roles",
    )


if __name__ == "__main__":
    main()
