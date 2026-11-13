# SPDX-License-Identifier: Apache-2.0
"""Evaluate a same-framing JPEG identity fallback with frozen source pairs."""

from __future__ import annotations

import argparse
import io
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import reference_compare
from scripts import image_patch_proposal_audit as panel
from scripts import image_reference_exact_jpeg_candidate as candidate

SCHEMA = "chipmunk.image-reference-exact-jpeg-transfer/1"
DESIGN = {
    "sources": "20 development and 12 source-disjoint confirmation photo groups; previously used in project studies. No rule tuned on either.",
    "profile": "Same dimensions, upright 8-bit RGB PNG earlier copy and standard baseline JPEG75 4:2:0 submitted. No profile conversion.",
    "identity": {"minimum_exact_block_fraction": candidate.MIN_EXACT_SHARE, "tile_px": candidate.TILE,
                 "minimum_unique_detail_blocks": candidate.MIN_ANCHORS, "minimum_supported_zones": candidate.MIN_ZONES,
                 "minimum_anchors_per_zone": candidate.MIN_ZONE_ANCHORS, "minimum_block_channel_range": candidate.MIN_DETAIL_RANGE},
    "selection": "Only fill cases the existing earlier-copy pipeline did not compare. Preserve all other incumbent results.",
    "difference": "Existing same-framing paired mask routine, full source coordinates. Changed pixel support does not assign authorship.",
    "controls": ["untouched", "matched ordinary variants", "different same-size photo", "2px shift", "16px shift",
                 "horizontal flip", "JPEG82 then JPEG75", "JPEG95 then JPEG75", "periodic identical and shifted tiles"],
    "confirmation_requirements": {"minimum_added_feathered_groups": 2, "maximum_bad_alignment_controls": 0,
                                 "maximum_new_unchanged_or_recompress_highlights": 0, "maximum_lost_incumbent": 0,
                                 "maximum_added_misplaced_edits": 0},
}
read, write, require, sha = panel.read, panel.write, panel.require, panel.sha
NEGATIVES = ("unrelated", "shift2", "shift16", "flip")
RECOMPRESS = ("recompress82", "recompress95")


def encode_png(rgb):
    with io.BytesIO() as stream:
        Image.fromarray(rgb).save(stream, "PNG")
        return stream.getvalue()


def evaluate(earlier, body, truth, *, audit_encoder=False):
    with Image.open(io.BytesIO(earlier)) as reference, Image.open(io.BytesIO(body)) as submitted:
        if audit_encoder:
            original = np.array(reference)
            expected = candidate.encoded_reference(original, submitted)
            incumbent, matched = reference_compare._codec(original, submitted)
            require(matched and np.array_equal(expected, incumbent), "Production JPEG reference decoder differs")
            if audit_encoder == "single_pass":
                require(np.array_equal(expected, np.array(submitted)), "Untouched delivery was not reproduced exactly")
        report, mask = candidate.compare(reference, submitted)
    if mask is None:
        mask = np.zeros(truth.shape, dtype=bool)
    require(mask.shape == truth.shape and mask.dtype == bool, "Native comparison mask has unexpected geometry")
    area, target, matched = (int(v.sum()) for v in (mask, truth, mask & truth))
    regions = report["regions"]
    for region in regions:
        coords = {key: region[key] for key in ("x", "y", "w", "h")}
        require(region["reference_region"] == coords, "Evidence crop is not bound to the shared source coordinates")
        x, y, w, h = (coords[key] for key in ("x", "y", "w", "h"))
        require(0 <= x < x+w <= truth.shape[1] and 0 <= y < y+h <= truth.shape[0], "Native evidence crop exceeds source")
    require(sum(region["selected_pixels"] for region in regions) == area, "Native evidence crop and mask totals disagree")
    proof = report.get("alignment")
    if proof is not None:
        require(proof["reference_minus_submitted"] == {"x": 0, "y": 0}, "Identity proof implies another transform")
        require(report["status"] == "comparable", "Incomparable result carried an alignment")
        if audit_encoder:
            require(proof["expected_decoded_rgb_sha256"] == sha(expected.tobytes()), "Returned expected grid differs")
    return {"status": report["status"], "selected": area, "target": target, "intersection": matched,
            "mask_sha256": sha(mask.tobytes()), "alignment": proof,
            "useful": target > 0 and 10*matched >= 3*(area + target - matched) and 2*matched >= target}


def control_views(rgb, unrelated):
    height, width = rgb.shape[:2]
    roll2 = np.roll(rgb, (2, 2), axis=(0, 1))
    roll16 = np.roll(rgb, (16, -16), axis=(0, 1))
    flip = rgb[:, ::-1]
    wrong = np.asarray(Image.fromarray(unrelated).resize((width, height), Image.Resampling.LANCZOS))
    return {
        "unrelated": wrong, "shift2": roll2, "shift16": roll16, "flip": flip,
        **{f"recompress{q}": panel.heads.base.decode(panel.heads.base.encode(rgb, q)) for q in (82, 95)},
    }


def constructed_periodic():
    rng = np.random.default_rng(5516)
    unit = rng.integers(24, 230, (16, 16, 3), dtype=np.uint8)
    period = np.tile(unit, (20, 20, 1))
    y, x = np.indices((320, 320))
    mirrored = np.repeat((80 + 30*np.sin(np.minimum(x, 319 - x) * np.pi / 17)
                          + 25*np.cos(y * np.pi / 9))[..., None].astype(np.uint8), 3, axis=2)
    return {
        "periodic_unchanged": (period, period), "periodic_shift16": (period, np.roll(period, 16, axis=1)),
        "mirrored_identical": (mirrored, mirrored[:, ::-1]),
        "periodic_with_local_change": (period, np.where(((x < 60) & (y < 60))[..., None], 127, period).astype(np.uint8)),
    }


def synthetic():
    results = {}
    for name, (source, view) in constructed_periodic().items():
        body = panel.heads.base.encode(view, 75)
        result = evaluate(encode_png(source), body, np.zeros(source.shape[:2], bool))
        require(result["alignment"] is None and not result["selected"], "Ambiguous synthetic texture was admitted")
        results[name] = result["status"]
    return results


def study(args, lock, phase):
    baseline_path = args.prior_audit / "rows.json"
    baseline_rows = read(baseline_path)
    old = {(row["panel"], row["source"], row["role"]): row for row in baseline_rows}
    rows = []
    for name, deliveries in panel.panel_data(args, lock):
        stage = "development" if name == "earlier" else "confirmation"
        if stage != phase:
            continue
        groups = list(deliveries)
        originals = {key["source"]: original for _, key, _, _, original in groups if key["role"] == "original"}
        ordered = sorted(originals)
        for index, identity in enumerate(ordered):
            earlier = encode_png(originals[identity])
            require(originals[identity].shape == next(row[3].shape + (3,) for row in groups if row[1]["source"] == identity),
                    "Original reference does not match native grid")
            ordinary = [row for row in groups if row[1]["source"] == identity]
            source = originals[identity]
            truth_empty = np.zeros(source.shape[:2], dtype=bool)
            family = ordinary[0][1]["family"]
            for _, key, body, exact, _ in ordinary:
                require(sha(body) == key["input_sha256"] and sha(exact.tobytes()) == key["truth_sha256"],
                        "Frozen questioned JPEG or change mask differs")
                previous = old[name, identity, key["role"]]
                require(previous["input_sha256"] == key["input_sha256"] and previous["truth_sha256"] == key["truth_sha256"],
                        "Frozen production baseline pairing differs")
                incumbent = previous["paired"]["earlier_png"]
                result = evaluate(earlier, body, exact, audit_encoder="single_pass" if key["role"] == "original" else False)
                fallback = incumbent["status"] != "comparable" and result["status"] == "comparable"
                rows.append(key | {"prior": incumbent, "candidate": result,
                                   "fallback_selected": fallback, "merged": result if fallback else incumbent})
            unrelated = originals[ordered[(index + 1) % len(ordered)]]
            for role, modified in control_views(source, unrelated).items():
                body = panel.heads.base.encode(modified, 75)
                result = evaluate(earlier, body, truth_empty, audit_encoder=role in RECOMPRESS)
                rows.append({"source": identity, "family": family, "role": role, "input_sha256": sha(body),
                             "truth_sha256": sha(truth_empty.tobytes()), "prior": None, "candidate": result,
                             "fallback_selected": False, "merged": None})
        print(phase, len(ordered), "groups", len(rows), "deliveries", flush=True)
    require(len(rows) == (20 if phase == "development" else 12) * 14, "Frozen source coverage differs")
    return rows


def summarize(rows):
    strata = defaultdict(Counter)
    for row in rows:
        role = row["role"]
        values = row["candidate"]
        prior = row["prior"]
        fallback = row["fallback_selected"]
        added = values["useful"] and fallback
        strata[f"{row['family']}/{role}"].update(
            files=1, candidate_comparable=values["status"] == "comparable", candidate_identity=values["alignment"] is not None,
            candidate_highlight=values["selected"] > 0, candidate_useful=values["useful"],
            prior_comparable=bool(prior and prior["status"] == "comparable"),
            fallback_compared=fallback, fallback_highlight=fallback and values["selected"] > 0, fallback_useful=added,
        )
    gate = {
        "added_feathered_groups": sum(row["role"] == "feathered" and row["fallback_selected"] and row["candidate"]["useful"] for row in rows),
        "bad_alignment_controls": sum(row["role"] in NEGATIVES and row["candidate"]["alignment"] is not None for row in rows),
        "unchanged_or_recompress_highlights": sum((row["role"] == "original" and row["fallback_selected"] or row["role"] in RECOMPRESS)
                                                   and row["candidate"]["selected"] > 0 for row in rows),
        "lost_incumbent": sum(row["prior"] is not None and row["prior"]["status"] == "comparable"
                              and row["merged"] != row["prior"] for row in rows),
        "added_misplaced_edits": sum(row["role"] in panel.heads.HELD_POSITIVE and row["fallback_selected"]
                                      and row["candidate"]["selected"] > 0 and not row["candidate"]["useful"] for row in rows),
    }
    bounds = DESIGN["confirmation_requirements"]
    passes = (gate["added_feathered_groups"] >= bounds["minimum_added_feathered_groups"]
              and gate["bad_alignment_controls"] <= bounds["maximum_bad_alignment_controls"]
              and gate["unchanged_or_recompress_highlights"] <= bounds["maximum_new_unchanged_or_recompress_highlights"]
              and gate["added_misplaced_edits"] <= bounds["maximum_added_misplaced_edits"]
              and gate["lost_incumbent"] <= bounds["maximum_lost_incumbent"])
    return {"strata": strata, "rule": gate, "passes": passes, "deliveries": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "development", "confirmation", "review"))
    for name in ("archive", "prior-probe", "residual", "early-f5", "later-f5", "checkpoint", "output-fresh", "prior-audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    args.exclude = []
    cv2.setNumThreads(1)
    lock, _, _ = panel.fresh.previous(args)
    old = args.prior_audit / "rows.json"
    public = Path("docs/evaluation-panels/image-patch-proposal-audit-2026-11-12.json")
    require(panel.fresh.roster.sha(old) == read(public)["private_rows_sha256"], "Production baseline journal differs")
    paths = (__file__, candidate.__file__, panel.__file__, reference_compare.__file__)
    freeze = {"schema": SCHEMA, "design": DESIGN,
              "code": {Path(path).name: panel.fresh.roster.sha(Path(path)) for path in paths},
              "baseline_journal_sha256": panel.fresh.roster.sha(old), "synthetic": synthetic()}
    plan = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(plan, freeze)
        return
    require(read(plan) == freeze, "Frozen identity method or controls changed")
    if args.stage == "review":
        require(args.publication is not None, "Aggregate output path required")
        publication = {}
        for role in ("development", "confirmation"):
            start = time.monotonic()
            rows = study(args, lock, role)
            journal = args.output / f"{role}-rows.json"
            require(rows == read(journal) and summarize(rows) == read(args.output / f"{role}.json"), "Independent replay differs")
            publication[role] = summarize(rows) | {"verified_replays": len(rows), "journal_sha256": panel.fresh.roster.sha(journal),
                                                  "replay_seconds": round(time.monotonic() - start, 2)}
        write(args.publication, freeze | {"results": publication})
        return
    if args.stage == "confirmation":
        require((args.output / "development.json").is_file(), "Development must finish before confirmation")
    start = time.monotonic()
    rows = study(args, lock, args.stage)
    result = summarize(rows)
    write(args.output / f"{args.stage}-rows.json", rows)
    write(args.output / f"{args.stage}.json", result)
    print(result["rule"], "seconds", round(time.monotonic() - start, 2), flush=True)


if __name__ == "__main__":
    main()
