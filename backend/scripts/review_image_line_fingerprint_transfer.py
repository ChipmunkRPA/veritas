# SPDX-License-Identifier: Apache-2.0
"""Verify saved native overlap and response arithmetic for the line-pattern study."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from scripts import image_line_fingerprint_transfer as trial
from scripts.review_image_chroma_interpolation_transfer import digest, exact_overlap


def native_readback(folder, rows, points):
    remaining = {(r["slot"], r["generator"], r["role"], r["format"]): r for r in rows}
    assert len(remaining) == len(rows)
    verified = Counter()
    for case, role, pixels, truth in trial.sources(folder):
        for fmt in trial.FORMATS:
            row = remaining.pop((case["slot"], case["generator"], role, fmt))
            payload, _ = trial.delivery(pixels, fmt)
            assert digest(payload) == row["input_sha256"] and digest(truth.tobytes()) == row["truth_sha256"]
            candidate, baseline = row["proposal"], row["baseline"]
            if candidate:
                flank, core = candidate["flank"], candidate["core"]
                assert all(trial.MIN_REFERENCE <= value <= 1.000000000001 for value in flank)
                assert all(-1.000000000001 <= value <= 1.000000000001 for value in core)
                assert max(0., min(a - b for a, b in zip(flank, core, strict=True))) == candidate["score"]
            selections = (
                ("candidate", candidate, candidate and candidate["score"] > points[fmt]["point"]),
                ("baseline", baseline, baseline and baseline["selected"]),
                ("ranked", candidate, candidate),
            )
            for arm, proposal, selected in selections:
                expected = exact_overlap(truth.shape, truth, proposal, selected)
                assert all(row["measurements"][arm][key] == value for key, value in expected.items())
                verified["native_overlaps"] += 1
            verified["encoded_views"] += 1
    assert not remaining
    return dict(verified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("earlier", "later", "run", "publication"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = trial.read(args.run / "result.json")
    fitted = trial.read(args.run / "points.json")
    frozen = trial.read(args.run / "freeze.json")
    assert trial.sha(args.run / "freeze.json") == result["freeze_sha256"] == fitted["freeze_sha256"]
    assert trial.sha(args.run / "points.json") == result["points_sha256"]
    for fit in fitted["points"].values():
        assert len(fit["source_maxima"]) == 8
        assert fit["point"] == max(trial.POINT_FLOOR, sorted(fit["source_maxima"])[-2])
    audits = {}
    for role in ("earlier", "later"):
        rows = result["rows"][role]
        audits[role] = native_readback(getattr(args, role), rows, fitted["points"])
        assert trial.summarize(rows) == result["summary"][role]
    output = {key: result[key] for key in ("freeze_sha256", "points_sha256", "summary", "elapsed_s")}
    output.update(freeze=frozen, points=fitted["points"], native_readback=audits, result_sha256=trial.sha(args.run / "result.json"))
    trial.write(args.publication, output)
    print(audits)


if __name__ == "__main__":
    main()
