"""Audit image reconstruction decisions and exact native overlap without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageChops

from scripts import image_edge_spread_transfer as sources

BASELINE = "format_reference"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def measure(box, truth):
    count = truth.histogram()[1]
    predicted = overlap = 0
    if box is not None:
        check(len(box) == 4 and all(type(v) is int for v in box), "Region bounds must be integral")
        x, y, right, bottom = box
        right, bottom = min(right, truth.width), min(bottom, truth.height)
        check(0 <= x < right and 0 <= y < bottom, "Selected region lies outside native pixels")
        predicted = (right - x) * (bottom - y)
        overlap = truth.crop((x, y, right, bottom)).histogram()[1]
    return {"reported": predicted, "target": count, "overlap": overlap,
            "iou": float(Fraction(overlap, predicted + count - overlap)) if count else None,
            "recall": float(Fraction(overlap, count)) if count else None}


def useful(geometry):
    return geometry["target"] > 0 and geometry["iou"] >= .30 and geometry["recall"] >= .50


def replay(folder, journal, points, arms):
    rows = {}
    for line in journal.read_text().splitlines():
        row = json.loads(line)
        key = row["slot"], row["role"], row["format"]
        check(key not in rows, "Confirmation rows contain a duplicate")
        rows[key] = row
    ranked = {fmt: {arm: 0 for arm in arms} for fmt in points}
    for case, views, _ in sources.paired.inputs(folder):
        photo = Image.fromarray(views["original"])
        for role, native in sources.native_variants(views, case["box"]).items():
            truth = Image.new("L", photo.size)
            if role in sources.POSITIVE:
                channels = ImageChops.difference(photo, Image.fromarray(native)).split()
                truth = ImageChops.lighter(ImageChops.lighter(*channels[:2]), channels[2]).point(lambda value: int(value != 0))
            for fmt, cutoffs in points.items():
                row = rows[(case["slot"], role, fmt)]
                body, _ = sources.paired.encoded(native, fmt)
                check(hashlib.sha256(body).hexdigest() == row["input_sha256"]
                      and hashlib.sha256(truth.tobytes()).hexdigest() == row["truth_sha256"], "Encoded input or truth was substituted")
                check(set(row["proposals"]) == set(row["measurements"]) == {*arms, BASELINE}, "A declared arm is missing")
                for arm in (*arms, BASELINE):
                    proposal = row["proposals"][arm]
                    if arm == BASELINE:
                        selected = proposal is not None and proposal["selected"] is True
                    else:
                        check(proposal is None or (math.isfinite(proposal["score"]) and proposal["core_tiles"] > 0
                                                   and proposal["ring_tiles"] > 0), "Invalid proposed measurement")
                        selected = proposal is not None and proposal["score"] > cutoffs[arm]["point"]
                        if proposal is not None and role in sources.POSITIVE:
                            ranked[fmt][arm] += useful(measure(proposal["box"], truth))
                    expected = measure(proposal["box"] if selected else None, truth)
                    check(row["measurements"][arm] == expected, "Native selection overlap does not recompute")
    check(len(rows) == 192 and len({key[0] for key in rows}) == 12, "A source panel is incomplete")
    return list(rows.values()), ranked


def counts(rows, arms, points):
    totals = {}
    for fmt in points:
        positive = [row for row in rows if row["format"] == fmt and row["role"] in sources.POSITIVE]
        ordinary = [row for row in rows if row["format"] == fmt and row["role"] in sources.CONTROLS]
        counts = {}
        for arm in (*arms, BASELINE):
            hits = [row for row in positive if useful(row["measurements"][arm])]
            controls = [row for row in ordinary if row["measurements"][arm]["reported"]]
            extra = [row for row in controls if not row["measurements"][BASELINE]["reported"]]
            counts[arm] = dict(sharp=sum(row["role"] == "edited" for row in hits),
                               feathered=sum(row["role"] == "feathered_edit" for row in hits),
                               ordinary_flags=len(controls), ordinary_groups=len({r["slot"] for r in controls}),
                               extra_ordinary_flags=len(extra), extra_ordinary_groups=len({r["slot"] for r in extra}),
                               extra_feathered=sum(r["role"] == "feathered_edit" and not useful(r["measurements"][BASELINE]) for r in hits),
                               unassessed_positive=sum(r["proposals"][arm] is None for r in positive),
                               unassessed_ordinary=sum(r["proposals"][arm] is None for r in ordinary))
        totals[fmt] = counts
    return totals


def audit(run, earlier, later):
    protocol, fit, summary = (read(run / name) for name in ("freeze.json", "points.json", "aggregate.json"))
    check(sha(run / "freeze.json") == fit["freeze_sha256"] == summary["freeze_sha256"]
          and sha(run / "points.json") == summary["points_sha256"], "Fit or results differ from the sealed plan")
    points, method = fit["points"], protocol["method"]
    arms = method["arms"]
    for cutoffs in points.values():
        for arm in arms:
            values = cutoffs[arm]["control_maxima"]
            check(len(values) == 8 and max(0, sorted(values)[6]) == cutoffs[arm]["point"], "A point differs from the frozen fit rule")
    ranked = {}
    for name, folder in (("earlier", earlier), ("later", later)):
        journal = run / f"{name}-rows.jsonl"
        check(sha(journal) == summary["journals_sha256"][name], "A journal differs from the sealed run")
        rows, ranked[name] = replay(folder, journal, points, arms)
        check(counts(rows, arms, points) == summary["results"][name], "Reported counts differ from source replay")
    advances = {}
    limits = method["advance"]
    for fmt in points:
        measures = [panel[fmt][method["primary"]] for panel in summary["results"].values()]
        advances[fmt] = all(row["extra_feathered"] >= limits["minimum_extra_feathered_per_panel"]
                            and row["extra_ordinary_flags"] <= limits["maximum_extra_ordinary_per_panel"]
                            and row["extra_ordinary_groups"] <= limits["maximum_extra_ordinary_groups_per_panel"] for row in measures)
    check(summary["advances"] == advances and summary["confirmed_views"] == 384
          and summary["model_round_trips"] == fit["development_forwards"] + 384 * len(method["phases_yx"]),
          "Decisions or model-forward accounting do not recompute")
    return {"verified_encoded_views": 384, "verified_measurements": 384 * (len(arms) + 1),
            "advances": advances, "useful_top_regions_before_cutoff": ranked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "earlier", "later"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.earlier, args.later), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
