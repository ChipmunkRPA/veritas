"""Read the text-free semantic study and recalculate its decisions."""

import gzip
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path


def overlap(p, t):
    shared = max(0, min(p[1], t[1]) - max(p[0], t[0]))
    return 2 * shared >= t[1] - t[0] and shared * 10 >= 3 * ((p[1] - p[0]) + (t[1] - t[0]) - shared)


def match(predictions, truth):
    masks = {0}
    for span in predictions:
        masks |= {
            mask | (1 << i)
            for i, ref in enumerate(truth)
            if overlap(span, ref)
            for mask in masks
            if not mask & (1 << i)
        }
    return max(mask.bit_count() for mask in masks)


def top(ranges):
    ordered = sorted(ranges, reverse=True)
    if not ordered or len(ordered) > 1 and ordered[0][0] == ordered[1][0]:
        return None
    return ordered[0]


def recount(packet):
    calibration, held = packet["calibration"], packet["held"]
    assert len(calibration) == 122 and len(held) == 165
    assert len({row["native_sha256"] for row in calibration + held}) == len(calibration) + len(held)
    assert not {row["group"] for row in calibration} & {row["group"] for row in held}
    assert packet["origin_overlap"]["equivalent_delivery_sha256"] not in {
        row["native_sha256"] for row in held
    }
    assert packet["origin_overlap"]["before_delivery_sha256"] not in {row["native_sha256"] for row in held}
    assert all(a < b for row in held + calibration for a, b in row["compact_bounds"])
    methods = {}
    for column, control in enumerate(packet["points"], 2):
        maxima = defaultdict(list)
        for row in calibration:
            maxima[row["group"]].extend(p[column] for p in row["readings"])
        values = sorted(max(group) for group in maxima.values() if group)
        cutoff = values[(95 * len(values) + 99) // 100 - 1]
        assert control["point"] == cutoff and control["groups"] == len(values)
        assert control["above_point_groups"] == sum(value > cutoff for value in values)
        groups = defaultdict(set)
        strata = defaultdict(Counter)
        expected = defaultdict(Fraction)
        for row in held:
            bounds = row["compact_bounds"]
            truth = row["truth_compact"]
            base = row["baseline"]
            cohort, kind = row["cohort"], row["role"]
            scores = [(p[column], p[0]) for p in row["readings"]]
            assert len(scores) == len({index for _, index in scores})
            assert all(0 <= index < len(bounds) for _, index in scores)
            candidates = [bounds[i] for _, i in scores]
            eligible = [
                (v, i)
                for v, i in scores
                if all(min(b[1], bounds[i][1]) <= max(b[0], bounds[i][0]) for b in base)
            ]
            top_all = top(scores)
            top_new = top(eligible)
            best = [bounds[top_all[1]]] if top_all else []
            extra = [bounds[top_new[1]]] if top_new and top_new[0] > cutoff else []
            standalone = best if top_all and top_all[0] > cutoff else []
            before = match(base, truth)
            after = match(base + extra, truth)
            assert row["decisions"][control["method"]] == {
                "top": best,
                "standalone": standalone,
                "added": extra,
                "before": before,
                "after": after,
            }
            for area in ("all", cohort):
                local = strata[area, kind]
                local.update(
                    files=1,
                    assessed=bool(scores),
                    labels=len(truth),
                    labels_with_usable_span=match(candidates, truth),
                    top_matches=match(best, truth),
                    baseline_useful=before,
                    supplemented_useful=after,
                    baseline_flag=bool(base),
                    supplemented_flag=bool(base + extra),
                    misplaced_on_ai_file=len(extra) - (after - before) if kind == "edited" else 0,
                    natural_top_focus=match(best, [row["editorial_focus"]])
                    if "editorial_focus" in row
                    else 0,
                )
                if candidates and truth:
                    expected[area, kind] += Fraction(
                        sum(match([c], truth) for c in candidates), len(candidates)
                    )
                if kind != "edited" and extra and not base:
                    groups[area, "new_ordinary_groups"].add(row["group"])
                if after > before:
                    groups[area, "new_localized_groups"].add(row["group"])
                    groups[area, "new_localized_generator_families"].add(row["generator"])
        methods[control["method"]] = {
            area: {
                kind: dict(strata[area, kind])
                | {"uniform_expected_locations": round(float(expected[area, kind]), 5)}
                for kind in ("untouched", "ordinary_edit", "natural_revision", "edited")
            }
            | {
                key: len(groups[area, key])
                for key in ("new_ordinary_groups", "new_localized_groups", "new_localized_generator_families")
            }
            for area in ("all", "science", "economic", "official", "turing")
        }
    return methods


if __name__ == "__main__":
    inp = Path(sys.argv[1])
    packet = json.loads(gzip.decompress(inp.read_bytes()))
    assert packet["schema"] == "chipmunk.writing-remote-semantic-public-review/1"
    results = recount(packet)
    print(json.dumps(results, indent=2, sort_keys=True))
