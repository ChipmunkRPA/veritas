# SPDX-License-Identifier: Apache-2.0
"""Independently trace dense witnesses after the whole-footprint audit stopped."""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from scripts import image_embedded_pair_canvas_transfer as source
from scripts import image_embedded_pair_dense_third as trial


def native_support(case, pair, proof):
    _, _, role, rgb, _, _, actual, _ = case
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    filtered = cv2.sepFilter2D(gray, -1, cv2.getGaussianKernel(9, 1), cv2.getGaussianKernel(9, 1))
    difference = (source_crop(rgb, pair["a"]) != source_crop(rgb, pair["b"])).any(axis=2)
    x0, y0, right, bottom = pair["a"]
    width, height = right-x0, bottom-y0
    a, b, c = proof["transform"][0]
    d, e, f = proof["transform"][1]
    rect = proof["third"]
    corners = [(a*x+b*y+c, d*x+e*y+f) for x in (x0, right) for y in (y0, bottom)]
    assert rect == [math.floor(min(p[0] for p in corners)), math.floor(min(p[1] for p in corners)),
                    math.ceil(max(p[0] for p in corners)), math.ceil(max(p[1] for p in corners))]
    for x, y, r, t in (pair["a"], pair["b"]):
        assert rect[2]+8 <= x or r+8 <= rect[0] or rect[3]+8 <= y or t+8 <= rect[1]
    fractions = None
    if role in trial.THIRDS:
        x, y, r, t = actual
        overlap = max(0, min(r, rect[2])-max(x, rect[0])) * max(0, min(t, rect[3])-max(y, rect[1]))
        fractions = overlap / ((rect[2]-rect[0])*(rect[3]-rect[1]))
    cells = set()
    centers_in_third = 0
    for cx, cy, reported in proof["witnesses"]:
        assert 9 <= cx < width-8 and 9 <= cy < height-8 and (cx-9) % 19 == (cy-9) % 19 == 0
        assert not difference[cy-8:cy+8, cx-8:cx+8].any()
        yy, xx = np.mgrid[y0+cy-8:y0+cy+8, x0+cx-8:x0+cx+8]
        mx, my = (a*xx+b*yy+c).astype(np.float32), (d*xx+e*yy+f).astype(np.float32)
        first = filtered[yy, xx].astype(np.float64)
        second = cv2.remap(filtered, mx, my, cv2.INTER_LINEAR).astype(np.float64)
        assert min(first.std(), second.std()) >= 8
        first -= first.mean()
        second -= second.mean()
        correlation = float(np.sum(first*second) / np.sqrt(np.sum(first**2)*np.sum(second**2)))
        assert correlation >= .968 and abs(correlation-reported) < .002
        if fractions is not None:
            centers_in_third += x <= mx[8, 8] < r and y <= my[8, 8] < t
        cells.add((3*cx//width, 3*cy//height))
    assert len(cells) >= 6
    assert len(proof["witnesses"]) >= max(16, .55*proof["source_candidates"])
    return fractions, int(centers_in_third)


def source_crop(rgb, rect):
    x, y, right, bottom = rect
    return rgb[y:bottom, x:right]


def audit(data, run):
    freeze = source.read(run / "freeze.json")
    recorded = source.read(run / "score.json")
    assert freeze == trial.contract(data)
    assert recorded["freeze_sha256"] == source.fs(run / "freeze.json")
    counters, public = Counter(), []
    for case, sealed, row in zip(trial.cases(data), freeze["cases"], recorded["rows"], strict=True):
        assert trial.case_id(case) == sealed
        assert (list(case[0]), case[1], case[2]) == (row["size"], row["slot"], row["role"])
        pair, proof = row["existing"], row["proof"]
        exact = False
        if pair and row["role"] in ("valid_pair", "unrelated_third"):
            size, slot, _, rgb, a, b, third, truth = case
            exact = source.audit_pair((size, slot, "ordinary_local", "ordinary_local", rgb, a, b, third, truth), pair)
            counters["base_exact_masks"] += exact
        fraction, centers = None, 0
        if proof:
            assert row["current"] == "local_difference" and pair
            fraction, centers = native_support(case, pair, proof)
            counters["source_projections"] += 1
            counters["native_patch_witnesses"] += len(proof["witnesses"])
            counters["witness_centers_inside_authored_third"] += centers
            if fraction is not None and fraction < .90:
                counters["failed_original_footprint_requirement"] += 1
        public.append({key: row[key] for key in ("size", "slot", "role", "current", "seconds")}
                      | {"extra_withheld": proof is not None, "exact_valid_mask": exact,
                         "native_witness_count": len(proof["witnesses"]) if proof else 0,
                         "whole_projection_in_authored_third": fraction,
                         "witness_centers_inside_authored_third": centers})
    result = {"schema": trial.SCHEMA, "freeze_sha256": recorded["freeze_sha256"], "rule": trial.RULE,
              "rows": public, "audit": dict(counters), "outcomes": trial.aggregate(public)}
    source.write(run / "supplemental-public.json", result)
    print(result["audit"], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run", type=Path)
    cli.add_argument("--data", type=Path, default=Path("backend/data/independent-evaluation"))
    args = cli.parse_args()
    cv2.setNumThreads(1)
    audit(args.data, args.run)
