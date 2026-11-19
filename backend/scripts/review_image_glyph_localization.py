"""Recompute the frozen screenshot glyph comparison from anonymous scalar results."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CARRIERS = ("png", "jpeg")
GROUPS = {"dev": 6, "held": 12}
ORDINARY = ("untouched", "column_emphasis", "color_emphasis", "isolated_legitimate_emphasis", "ordinary_local_blur")
EDITED = ("single_font_substitution", "single_anisotropic_substitution")


def review(document: dict) -> list[dict]:
    if document.get("schema") != "chipmunk.glyph-localization-pilot/1":
        raise ValueError("Invalid screenshot pilot receipt")
    rows = {}
    for row in document["cases"]:
        key = row["split"], row["group"], row["carrier"], row["role"]
        if key in rows or type(row["group"]) is not int or len(row["top"]) != 2:
            raise ValueError("Duplicate group delivery or invalid measurement")
        wanted = row["truth_pixels"]
        if type(wanted) is not int or wanted < 0 or bool(wanted) != (row["role"] in EDITED):
            raise ValueError("Unexpected intervention mask")
        for point in row["top"]:
            if point is None:
                continue
            score, area, hits = (point[k] for k in ("score", "area", "hits"))
            if (type(score) not in (int, float) or not math.isfinite(score) or score < 0 or type(area) is not int
                    or type(hits) is not int or area <= 0 or not 0 <= hits <= min(area, wanted)):
                raise ValueError("Invalid region measurements")
        rows[key] = row
    expected = {(split, index, carrier, role) for split, size in GROUPS.items() for index in range(size)
                for carrier in CARRIERS for role in (*ORDINARY, *EDITED)}
    if rows.keys() != expected:
        raise ValueError("Missing or unexpected planned delivery")
    def score(row, arm):
        reading = row["top"][arm]
        return reading["score"] if reading is not None else 0

    results = []
    for carrier in CARRIERS:
        for arm, name in enumerate(("ink_shape", "ink_shape_and_aspect")):
            maxima = [max(score(rows["dev", group, carrier, role], arm) for role in ORDINARY)
                      for group in range(GROUPS["dev"])]
            cutoff = sorted(maxima)[-2]
            if cutoff != document["points"][carrier][arm]:
                raise ValueError("The ordinary development point differs from the locked value")
            totals = defaultdict(Counter)
            ordinary_groups, useful_groups = set(), set()
            for group in range(GROUPS["held"]):
                for role in (*ORDINARY, *EDITED):
                    row = rows["held", group, carrier, role]
                    point = row["top"][arm]
                    totals[role]["planned"] += 1
                    if point is None:
                        totals[role]["unassessed"] += 1
                        continue
                    target, hits, area = row["truth_pixels"], point["hits"], point["area"]
                    useful = bool(target and 2 * hits >= target and 10 * hits >= 3 * (target + area - hits))
                    totals[role]["strongest_at_edit"] += useful
                    if point["score"] > cutoff:
                        totals[role]["flags"] += 1
                        totals[role]["useful"] += useful
                        if role in ORDINARY:
                            ordinary_groups.add(group)
                        elif useful:
                            useful_groups.add(group)
            results.append({"carrier": carrier, "measurement": name, "development_point": cutoff,
                            "held_cases": totals, "held_ordinary_groups_flagged": len(ordinary_groups),
                            "held_useful_groups": len(useful_groups),
                            "proceed_to_ai_edit_testing": len(useful_groups) >= 4 and len(ordinary_groups) <= 2})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    print(json.dumps(review(json.loads(parser.parse_args().receipt.read_text(encoding="utf-8"))), indent=2))
