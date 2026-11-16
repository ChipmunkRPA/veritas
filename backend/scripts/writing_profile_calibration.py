"""Compare document-profile routing and ordinary-calibrated writing intervals."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from app.core.canonical import FLOAT_DP
from app.evidence import document_passage_intervals as serving
from app.evidence.passage_interval_selection import (
    combine_responses,
    interval_candidates,
    length_bucket,
    normalize,
    probability_log_odds,
    select_intervals,
)
from scripts import writing_multi_passage_panel as source
from scripts.writing_multi_passage_selection import _match, _overlap, _quantile

SCHEMA = "chipmunk.writing-profile-calibration/1"
GENRES = ("historical_prose", "economic_report")
VIEWS = ("isolated", "context")
PROFILES = ("shape", "combined")
TOLERANCES = {"wider": .75, "tighter": .90}
METHODS = ("current", "shape-wider", "shape-tighter", "combined-wider", "combined-tighter", "oracle-wider", "oracle-tighter")


def require(valid, message):
    if not valid:
        raise ValueError(message)


def recorded(variant):
    geometry = [[r[0], r[1], r[-1]] for r in variant["sentences"]]
    if any(value is None for view in VIEWS for value in variant[view]):
        return geometry, None
    return geometry, {view: [round(value, FLOAT_DP) for value in variant[view]] for view in VIEWS}


def word_buckets(sentences):
    words = [row[-1] for row in sentences]
    return {"isolated": [length_bucket(n) for n in words], "context": [length_bucket(sum(words[max(0, i-1):i+2]))
                                                                      for i in range(len(words))]}


def profile(sentences, readings, method):
    counts = [row[2] for row in sentences]
    middle = max(1, median(counts))
    vector = [math.log(len(counts)), math.log(middle), _quantile(counts, .10) / middle, _quantile(counts, .90) / middle,
              sum(row[1] - row[0] for row in sentences) / max(1, sum(counts))]
    if method == "combined":
        buckets = word_buckets(sentences)
        reference = {"isolated": serving.ISOLATED, "context": serving.CONTEXT}
        for view in VIEWS:
            adjusted = [normalize(probability_log_odds(response), *reference[view][bucket])
                        for response, bucket in zip(readings[view], buckets[view], strict=True)]
            first = _quantile(adjusted, .25)
            vector.extend((first, median(adjusted) - first))
    return vector


def distance(left, right, scale):
    return math.sqrt(math.fsum(((a - b) / step)**2 for a, b, step in zip(left, right, scale, strict=True)) / len(scale))


def nearby(vector, examples, scale):
    closest = sorted(distance(vector, row, scale) for row in examples)[:3]
    return math.fsum(closest) / len(closest)


def fit_profiles(ordinary):
    models = {}
    for feature in PROFILES:
        by_group = defaultdict(list)
        for work, variant in ordinary:
            geometry, readings = recorded(variant)
            if readings is not None:
                by_group[work["genre"], work["group_sha256"]].append(profile(geometry, readings, feature))
        examples = {genre: [list(map(median, zip(*values, strict=True))) for (label, _), values in sorted(by_group.items())
                            if label == genre] for genre in GENRES}
        require(min(map(len, examples.values())) >= 7, "Each profile requires multiple separate source groups")
        columns = zip(*(row for group in examples.values() for row in group), strict=True)
        scale = [max(.08, _quantile(col, .85) - _quantile(col, .15)) for col in columns]
        radius = {genre: _quantile([nearby(row, bank[:i] + bank[i+1:], scale) for i, row in enumerate(bank)], .90)
                  for genre, bank in examples.items()}
        models[feature] = {"examples": examples, "scale": scale, "radius": radius}
    return models


def route(sentences, readings, method, fitted):
    head = fitted[method]
    vector = profile(sentences, readings, method)
    ranked = sorted((nearby(vector, rows, head["scale"]), genre) for genre, rows in head["examples"].items())
    (first, genre), (other, _) = ranked
    return genre if first <= head["radius"][genre] and other >= 1.5 * first else None


def genre_intervals(sentences, readings, scales):
    buckets = word_buckets(sentences)
    views = {view: [normalize(probability_log_odds(value), *scales[view][bucket])
                    for value, bucket in zip(readings[view], buckets[view], strict=True)] for view in VIEWS}
    return interval_candidates(combine_responses(views["isolated"], views["context"]))


def fit_scales(ordinary):
    groups = defaultdict(list)
    for work, variant in ordinary:
        geometry, readings = recorded(variant)
        if readings is None:
            continue
        for view, tags in word_buckets(geometry).items():
            for value, bucket in zip(readings[view], tags, strict=True):
                groups[work["genre"], view, bucket, work["group_sha256"]].append(probability_log_odds(value))
    stats = defaultdict(list)
    for (genre, view, bucket, _), values in groups.items():
        stats[genre, view, bucket].append((median(values), _quantile(values, .90)))
    fitted = {genre: {} for genre in GENRES}
    pooled = {"isolated": serving.ISOLATED, "context": serving.CONTEXT}
    for genre in GENRES:
        for view in VIEWS:
            units = []
            for bucket in range(3):
                values = stats[genre, view, bucket]
                if len(values) < 3:
                    units.append(list(pooled[view][bucket]))
                    continue
                middle, high = map(median, zip(*values, strict=True))
                units.append([middle, max(.25, high - middle)])
            fitted[genre][view] = units
    return fitted


def fit(works):
    ordinary = [(work, variant) for work in works for variant in work["variants"] if variant["version"] == "ordinary"]
    features = fit_profiles(ordinary)
    scales = fit_scales(ordinary)
    maxima = defaultdict(dict)
    for work, variant in ordinary:
        key, group = work["genre"], work["group_sha256"]
        geometry, readings = recorded(variant)
        if readings is not None:
            score = max((value for _, _, value in genre_intervals(geometry, readings, scales[key])), default=0.)
            maxima[key][group] = max(score, maxima[key].get(group, 0.))
    points = {genre: {label: _quantile(bank.values(), fraction) for label, fraction in TOLERANCES.items()}
              for genre, bank in sorted(maxima.items())}
    return {"features": features, "scales": scales, "points": points}


def measure(works, fitted):
    output = []
    for work in works:
        choices = []
        for variant in work["variants"]:
            geometry, readings = recorded(variant)
            current = serving._choices(geometry, readings, document_baseline=True)
            route_outputs = {}
            if readings is not None:
                route_outputs = {key: route(geometry, readings, key, fitted["features"]) for key in PROFILES}
            selected = {"current": (None, current)}
            routes = route_outputs | {"oracle": work["genre"] if work["genre"] in GENRES and readings is not None else None}
            for check, routed in routes.items():
                intervals = genre_intervals(geometry, readings, fitted["scales"][routed]) if routed else None
                for label in TOLERANCES:
                    found = ([(a, b, value, "profile") for a, b, value in select_intervals(
                        intervals, fitted["points"][routed][label], len(geometry))]
                        if intervals is not None else current)
                    selected[f"{check}-{label}"] = (routed, found)
            if readings is None:
                selected.update({name: (None, ()) for name in METHODS if name not in selected})
            rows = {}
            for check, (routed, plans) in selected.items():
                loc = [variant["sentences"][a:b] for a, b, _, _ in plans]
                spans = [{"compact": [items[0][2], items[-1][3]], "native": [items[0][0], items[-1][1]],
                          "basis": plan[-1]} for items, plan in zip(loc, plans, strict=True)]
                rows[check] = {"profile": routed, "ranges": spans}
            choices.append({"layout": variant["layout"], "version": variant["version"], "choices": rows,
                            "truth": [item["compact"] for item in variant["truth"]], "matched_targets": variant.get("matched_targets", []),
                            "assessed": readings is not None})
        output.append({"document": work["document"], "group": work["group_sha256"], "genre": work["genre"],
                       "generator": work["generator_backend"], "versions": choices})
    return output


def summarize(measured):
    counters, groups = defaultdict(Counter), defaultdict(set)
    for work in measured:
        group, genre = work["group"], work["genre"]
        originals = {v["layout"]: v for v in work["versions"] if v["version"] == "ordinary"}
        for row in work["versions"]:
            for check, option in row["choices"].items():
                proposal, label = option["ranges"], option["profile"]
                key = f"{genre}/{row['layout']}/{check}"
                stats = counters[key]
                if row["version"] == "ordinary":
                    stats.update(ordinary=1, ordinary_flagged=bool(proposal), ordinary_routed=label is not None,
                                 ordinary_wrong_route=label is not None and label != genre, unassessed=not row["assessed"])
                    if proposal:
                        groups[key].add(group)
                    continue
                matches = _match(proposal, row["truth"])
                original = originals[row["layout"]]
                original_variant = original["choices"][check]
                confounded = {index for index, target in enumerate(original["matched_targets"])
                              if any(_overlap(r["compact"], target) for r in original_variant["ranges"])}
                stats.update(mixed=1, targets=len(row["truth"]), located=len(matches), unmatched=len(proposal) - len(matches),
                             unconfounded=len({hit for _, hit in matches} - confounded), changes_of_profile=label != original_variant["profile"],
                             mixed_routed=label is not None, unassessed=not row["assessed"])
    return {key: dict(row) | {"flagged_source_groups": len(groups[key])} for key, row in sorted(counters.items())}


def split_roles(retained):
    by_genre = defaultdict(set)
    for work in retained:
        if any(v["version"] == "mixed" for v in work["variants"]):
            by_genre[work["genre"]].add(work["group_sha256"])
    selected = {key for genre, groups in by_genre.items() for key in sorted(groups, key=lambda g: source.digest(
        (SCHEMA + genre + g).encode()))[:max(4, len(groups)//4)]}
    return ([work for work in retained if work["group_sha256"] not in selected],
            [work for work in retained if work["group_sha256"] in selected])


def select(summary):
    feasible = []
    for check in METHODS[1:5]:
        paired = [(summary[f"{genre}/{layout}/{arm}"] for arm in (check, "current"))
                  for genre in GENRES for layout in ("published", "neutral")]
        cells = [tuple(pair) for pair in paired]
        if any(after["flagged_source_groups"] > before["flagged_source_groups"] + 1 for after, before in cells):
            continue
        gain = sum(after["unconfounded"] - before["unconfounded"] for after, before in cells)
        false = sum(after["ordinary_flagged"] - before["ordinary_flagged"] for after, before in cells)
        good = sum(after["located"] - before["located"] for after, before in cells)
        if good >= 0 and (gain > 0 or false < 0):
            feasible.append((-gain, false, -good, METHODS.index(check), check))
    return min(feasible)[-1] if feasible else "current"


def calibration(calibration_file, fresh_file, out):
    retained = source.read(calibration_file)["works"] + source.read(fresh_file)["held"]
    training, selection = split_roles(retained)
    model = fit(training)
    replay = measure(selection, model)
    summary = summarize(replay)
    best = select(summary)
    audit = {"schema": SCHEMA, "fit": model, "selected": best, "selection": replay, "selection_summary": summary,
             "fit_groups": sorted({w["group_sha256"] for w in training}), "selection_groups": sorted({w["group_sha256"] for w in selection}),
             "history": [source.digest(item.read_bytes()) for item in (calibration_file, fresh_file)],
             "code_sha256": source.digest(Path(__file__).read_bytes()), "held_measurements_at_freeze": 0}
    source.save(out, audit)
    print(json.dumps({"candidate": best, "fit_groups": len(audit["fit_groups"]), "selection_groups": len(audit["selection_groups"])}, sort_keys=True))


def report(frozen, held, output):
    lock, readings = source.read(frozen), source.read(held)
    require(lock["code_sha256"] == source.digest(Path(__file__).read_bytes()), "Profile code changed after selection")
    identities = {work["group_sha256"] for work in readings["works"]}
    require(not identities.intersection(lock["fit_groups"] + lock["selection_groups"]), "A previously used profile group is present in holdout")
    selected = measure(readings["works"], lock["fit"])
    receipt = {"schema": SCHEMA, "freeze": lock, "freeze_sha256": source.digest(frozen.read_bytes()), "checks": selected,
               "summary": summarize(selected), "source": readings}
    verify(receipt)
    source.save(output, receipt)


def verify(receipt):
    lock = receipt["freeze"]
    require(receipt["schema"] == lock["schema"] == SCHEMA, "Unexpected writing profile schema")
    require(summarize(lock["selection"]) == lock["selection_summary"] and select(lock["selection_summary"]) == lock["selected"],
            "Writing profile selection differs from the frozen study")
    recomputed = measure(receipt["source"]["works"], lock["fit"])
    require(recomputed == receipt["checks"] and summarize(recomputed) == receipt["summary"], "Held writing selections differ")
    held = {row["group"] for row in recomputed}
    roles = [set(lock["fit_groups"]), set(lock["selection_groups"]), held]
    require(sum(map(len, roles)) == len(set.union(*roles)), "Writing source groups cross evaluation roles")
    return {"selected": lock["selected"], "tested_files": sum(len(row["versions"]) for row in recomputed),
            "tested_works": len(recomputed), "genres": dict(Counter(w["genre"] for w in recomputed))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="step", required=True)
    training, testing, reader = (commands.add_parser(name) for name in ("fit", "assess", "verify"))
    for arg in ("calibration", "fresh", "output"):
        training.add_argument(arg, type=Path)
    for arg in ("freeze", "held", "output"):
        testing.add_argument(arg, type=Path)
    reader.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.step == "fit":
        calibration(args.calibration, args.fresh, args.output)
    elif args.step == "assess":
        report(args.freeze, args.held, args.output)
    else:
        print(json.dumps(verify(source.read(args.receipt)), sort_keys=True))
