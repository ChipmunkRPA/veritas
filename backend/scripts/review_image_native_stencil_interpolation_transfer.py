# SPDX-License-Identifier: Apache-2.0
"""Audit generated-edit localization and veto counts from published scalar rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROLES = ("untouched", "bilinear_horizontal", "bilinear_diagonal", "bicubic_diagonal", "rotated",
         "blur", "quantized_bilinear", "local_tone", "generated_sharp", "generated_feathered")
POSITIVES = ("generated_sharp", "generated_feathered")
POINT = 2.4346873354224834
VETO = .70


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def region(row):
    proposal = row["base"]
    if proposal is None:
        require(row["status"] == "no_repeatable_stencil" and row["parity_fit"] is None,
                "Unsupported row reported measurements")
        return False, False, False, False
    require(row["status"] == "assessed", "Inconsistent assessment")
    box = proposal["box"]
    native_area = 0
    if box:
        x0, y0, x1, y1 = box
        require(0 <= x0 < x1 <= 2048 and 0 <= y0 < y1 <= 1536 and all(i % 32 == 0 for i in box),
                "Selected box outside the native coordinate grid")
        native_area = (y1 - y0) * (x1 - x0)
        require(len(row["parity_fit"]) == 2 and all(len(axis) == 2 and all(0 <= r <= 1 for r in axis)
                                                   for axis in row["parity_fit"]), "Invalid parity response")
    else:
        require(row["parity_fit"] is None and proposal["score"] == 0, "Empty box has measurements")
    observed = proposal["pixels"]
    overlap, truth = observed["intersection"], row["truth_pixels"]
    require(observed["predicted_positive_pixels"] == native_area and observed["reference_positive_pixels"] == truth
            and 0 <= overlap <= min(truth, native_area), "Impossible exact-mask counts")
    union = native_area + truth - overlap
    require(observed["iou"] == (overlap / union if union else 1)
            and observed["recall"] == (overlap / truth if truth else None), "Overlap ratios differ")
    flag = proposal["score"] > POINT
    useful = flag and bool(truth and 10 * overlap >= 3 * union and 2 * overlap >= truth)
    vetoed = flag and any(all(score > VETO for score in axis) for axis in row["parity_fit"])
    return True, flag, useful, vetoed


def audit(ledger_path, roster, receipt, *, freeze=None):
    ledger, published, sources = map(load, (ledger_path, receipt, roster))
    require(ledger["freeze_sha256"] == published["freeze_sha256"]
            and sha(ledger_path) == published["ledger_sha256"] and sha(roster) == published["roster_sha256"],
            "Case and measurement commitments differ")
    require(all(row["license"] == "CC0-1.0" for row in sources), "Source license changed")
    if freeze is not None:
        lock = load(freeze)
        require(sha(freeze) == published["freeze_sha256"] and lock["design"] == published["design"]
                and lock["roster_sha256"] == published["roster_sha256"], "Native freeze differs")
    keys = {source["id"] for source in sources}
    require(len(keys) == published["source_groups"] == 13 and len(ledger["rows"]) == published["exact_native_replays"] == 130,
            "Incomplete source roster")
    expected = {(key, role) for key in keys for role in ROLES}
    counts = {role: Counter(files=0, assessed=0, base_flags=0, base_useful=0, vetoed=0, kept_flags=0, kept_useful=0)
              for role in ROLES}
    for row in ledger["rows"]:
        key = row["key"], row["role"]
        require(key in expected and (row["truth_pixels"] == 0) == (row["role"] == "untouched"), "Unexpected role or truth")
        expected.remove(key)
        checked, flagged, useful, vetoed = region(row)
        counts[row["role"]].update(files=1, assessed=int(checked), base_flags=int(flagged),
                                   base_useful=int(useful), vetoed=int(vetoed),
                                   kept_flags=int(flagged and not vetoed), kept_useful=int(useful and not vetoed))
    require(not expected, "Missing camera/role rows")
    old = sum(c["base_flags"] for role, c in counts.items() if role not in POSITIVES)
    kept = sum(c["kept_flags"] for role, c in counts.items() if role not in POSITIVES)
    retained = all(counts[p]["kept_useful"] >= 7 and 5 * counts[p]["kept_useful"] >= 4 * counts[p]["base_useful"]
                   for p in POSITIVES)
    require(published["measurements"] == {"by_role": counts, "ordinary_base_flags": old, "ordinary_kept_flags": kept,
                                          "gate_passed": retained and old > 0 and 2*kept <= old
                                          and counts["untouched"]["kept_flags"] <= 1}, "Decision arithmetic differs")
    print(f"Verified {len(ledger['rows'])} frozen scalar rows and the advancement gate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path, help="Published scalar ledger or native run directory")
    parser.add_argument("roster", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.measurements.is_dir():
        audit(args.measurements / "scored.json", args.roster, args.receipt, freeze=args.measurements / "freeze.json")
    else:
        audit(args.measurements, args.roster, args.receipt)
