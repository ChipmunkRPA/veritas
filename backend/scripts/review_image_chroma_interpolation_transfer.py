# SPDX-License-Identifier: Apache-2.0
"""Replay and verify the local chroma-interpolation study without rerunning its detector."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import image_chroma_interpolation_transfer as trial


def digest(value):
    return hashlib.sha256(value).hexdigest()


def exact_overlap(shape, truth, proposed, selected):
    expected = {"reported": 0, "target": int(np.count_nonzero(truth)), "overlap": 0}
    if selected:
        x0, y0, x1, y1 = proposed["box"]
        assert 0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]
        expected.update(reported=(x1 - x0) * (y1 - y0), overlap=int(np.count_nonzero(truth[y0:y1, x0:x1])))
    return expected


def verify_response(proposal):
    if not proposal:
        return
    gaps = []
    for axis in range(2):
        flanks = [row[axis] for row in proposal["flank_signed"]]
        ordered = sorted(flanks)
        direction = 1 if ordered[1] + ordered[2] >= 0 else -1
        assert direction == proposal["phase"][axis]
        supported = sorted(direction * item for item in flanks)[1]
        assert supported >= trial.MIN_FLANK
        gaps.append(supported - direction * proposal["interior_signed"][axis])
    assert max(0., min(gaps)) == proposal["score"]


def verify_rows(folder, rows, points):
    committed = {(r["slot"], r["generator"], r["role"], r["format"]): r for r in rows}
    assert len(committed) == len(rows)
    counts = Counter()
    for case, views, native_truth in trial.panel.paired.inputs(folder):
        box = tuple(case["box"])
        variants = trial.inputs(views["original"], {"processed": views["local_processing"], "edited": views["edited"]}, box)
        for role, image in variants.items():
            truth = (np.count_nonzero(image != variants["preserved"], axis=-1) > 0
                     if role in trial.POSITIVE else np.zeros(native_truth.shape, dtype=bool))
            target = int(np.count_nonzero(truth))
            if role in trial.POSITIVE:
                assert exact_overlap(truth.shape, truth, {"box": box}, True)["overlap"] == target > 0
            for fmt in trial.FORMATS:
                record = committed.pop((case["slot"], case["generator"], role, fmt))
                body, _ = trial.delivery(image, fmt)
                assert record["canvas"] == list(reversed(truth.shape))
                assert digest(body) == record["input_sha256"] and digest(truth.tobytes()) == record["truth_sha256"]
                candidate, baseline = record["candidate"], record["baseline"]
                verify_response(candidate)
                selections = (
                    ("candidate", candidate, candidate and candidate["score"] > points[fmt]["point"]),
                    ("baseline", baseline, baseline and baseline["selected"]),
                    ("candidate_rank", candidate, candidate),
                )
                for arm, proposal, selected in selections:
                    expected = exact_overlap(truth.shape, truth, proposal, selected)
                    assert all(record[arm + "_overlap"][key] == value for key, value in expected.items())
                    counts["native_overlaps"] += 1
                counts["encoded_views"] += 1
    assert not committed
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("earlier", "later", "run", "publication"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    result = trial.read(args.run / "result.json")
    frozen, fitted = (trial.read(args.run / f"{name}.json") for name in ("freeze", "points"))
    assert result["freeze"] == frozen and result["points"] == fitted["points"]
    assert result["freeze_sha256"] == fitted["freeze_sha256"] == trial.sha(args.run / "freeze.json")
    assert result["points_sha256"] == trial.sha(args.run / "points.json")
    for row in fitted["points"].values():
        assert row["point"] == max(trial.POINT_FLOOR, sorted(row["source_maxima"])[-2])
    proof = {}
    for role in ("earlier", "later"):
        rows = result["rows"][role]
        proof[role] = verify_rows(getattr(args, role), rows, fitted["points"])
        assert trial.summarize(rows) == result["summary"][role]
    publication = {key: result[key] for key in ("freeze", "freeze_sha256", "points", "points_sha256", "summary", "elapsed_s")}
    publication["native_readback"] = proof
    publication["result_sha256"] = trial.sha(args.run / "result.json")
    trial.write(args.publication, publication)
    print(proof)


if __name__ == "__main__":
    main()
