"""Compare agreement after checking both writing responses against ordinary files."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_context_pool_transfer as pool
from scripts import writing_focus_inside_context as sources
from scripts import writing_sibling_paragraph_transfer as parent
from scripts.writing_editorial_worker_audit import covers
from scripts.writing_interval_transfer import fingerprints
from scripts.writing_missed_passage_transfer import POINT
from scripts.writing_sentence_probe import SHA256, write

SCHEMA = "chipmunk.writing-joint-tail-transfer/1"
MATCHED_SCHEMA = "chipmunk.writing-joint-tail-matched/1"
DEVELOPMENT = pool.PANEL_DIR / "writing-other-paragraphs-2026-11-14.json"
FAMILIES = ("gemini-2.5-flash", "gpt-5.4-nano")
DOMAINS = ("news", "abstracts")
TARGET = 15
EDGES = (24, 42)
FIT_CENTERS = (1, 2, 3, 6, 7, 8, 11, 12, 13)
HELD_CENTERS = (1, 2, 3)
RULES = ("stronger", "both", "average")
MATCHED_ROLES = ("fit", "point", "held")
CONTROL_GROUPS = {"fit": 48, "point": 36}
METHOD = {
    "fit": "Use first-cohort government calibration works only. For each exact ratio measurement, group by scored "
           "sentence token length (up to 24, up to 42, or longer). Record one maximum per measure, length bin and work.",
    "ranks": "Use the fraction of those work maxima strictly below the same-location reading, divided by n+1. "
             "Use classifier response in its native adjacent context and writing-pattern response on the exact sentence. "
             "Require at least 12 reference works in a length bin. Never substitute a missing or expanded ratio.",
    "candidate": "Compare the stronger rank, smaller rank and arithmetic mean of the two ranks. "
                 "Each calibrated policy requires one unique strongest above-point location per submitted document.",
    "calibration": "Set cutoffs from the document maxima of the separate ordinary government control works. "
                   "Use strict exceedance and allow at most 10% or 25% of all those files at the frozen point.",
    "held": "Use source-disjoint English news and scientific-abstract works. Retain the current union and raw "
            "same-center agreement as controls. Freeze all choices before held inference.",
    "utility": "Measure source-linked overlap, wrong positions and ordinary flags. Comparison includes every admitted file.",
}
MATCHED_METHOD = METHOD | {
    "fit": "For each genre use 48 separate five-sentence ordinary source excerpts to set the three token-length "
           "references. Record one maximum per measure, length bin and work, using the same three interior positions as the held files.",
    "calibration": "Use 36 separate ordinary source works per genre with the same input lengths as the held files. "
                   "Freeze per-genre cutoffs with strict exceedance for both prior operating rules.",
    "held": "Select new source works for fitting, calibration and held measurement. Exclude the government-transfer panel "
            "and all its earlier research exclusions. The frozen scores and rank rules from the first comparison remain in effect.",
}


def shard_pins():
    return {f"{domain}/{family}": pool.SHARDS[domain][family] for domain in DOMAINS for family in FAMILIES}


def prior_pins():
    return {file.name: pool.file_hash(file) for file in sorted(pool.PANEL_DIR.glob("writing-*-2026-11-14.json"))
            if not file.name.startswith("writing-joint-tail-")}


def ordinary_controls(domain, tables, excluded, used):
    eligible = {source_id: (family, versions["v0"]) for family, table in tables.items()
                for source_id, versions in table.items() if "v0" in versions}
    counts = Counter()
    for source_id in sorted(eligible, key=lambda key: pool.sha(f"{MATCHED_SCHEMA}/ordinary/{domain}/{key}")):
        group = pool.sha(source_id)
        if group in excluded:
            continue
        family, source = eligible[source_id]
        try:
            raw, labels = pool.truth_text(source)
        except ValueError:
            continue
        if labels or not 420 <= len(raw) <= 65000:
            continue
        phrases = fingerprints(raw)
        if phrases & used:
            continue
        for window in pool.source_windows(raw):
            try:
                text, spans = pool.normalize(raw, window)
            except ValueError:
                continue
            role = "fit" if counts["fit"] < CONTROL_GROUPS["fit"] else "point"
            yield {"role": role, "group": group, "domain": domain, "generator": family, "position": 2,
                   "origin": None, "sha256": pool.sha(text), "text": text, "spans": spans, "truth": None,
                   "spaces": [i for i, char in enumerate(text) if char.isspace()]}
            counts[role] += 1
            used.update(phrases)
            excluded.add(group)
            break
        if counts == CONTROL_GROUPS:
            return
    raise ValueError(f"Insufficient matching ordinary controls in {domain}")


def freeze(root, output, prior, previous=None):
    excluded = parent.source_exclusions(prior)
    schema = MATCHED_SCHEMA if previous is not None else SCHEMA
    if previous is not None:
        first = sources.read(previous)
        validate(first)
        if first["schema"] != SCHEMA or first["earlier_panels"] != prior:
            raise ValueError("The earlier transfer must use the same source exclusions")
        excluded.update(row["group"] for row in first["rows"])
    pins = shard_pins()
    used, rows, counts = set(), [], {}
    for domain in DOMAINS:
        tables = {}
        for family in FAMILIES:
            file = root / domain / f"{domain}_{family}.csv"
            table = sources.shard(file, domain, family, pins[f"{domain}/{family}"])
            if previous:
                tables[family] = table
            for key, versions in table.items():
                if pool.sha(key) in excluded and "v0" in versions:
                    used.update(fingerprints(versions["v0"]["text"]))
            fills = Counter()
            for source_id in sorted(table, key=lambda value: pool.sha(f"{schema}/{domain}/{family}/{value}")):
                group = pool.sha(source_id)
                versions = table[source_id]
                if group in excluded or "v0" not in versions:
                    continue
                phrases = fingerprints(versions["v0"]["text"])
                if phrases & used:
                    continue
                try:
                    options = sources.excerpt_options(source_id, versions)
                except ValueError:
                    continue
                if not options:
                    continue
                position = min(options, key=lambda index: (fills[index], index))
                for text, spans, variant in options[position]:
                    rows.append({"role": "held", "group": group, "domain": domain, "generator": family,
                                 "position": position, "origin": variant, "sha256": pool.sha(text), "text": text,
                                 "spans": spans, "truth": spans[position] if variant else None,
                                 "spaces": [i for i, char in enumerate(text) if char.isspace()]})
                fills[position] += 1
                excluded.add(group)
                used.update(phrases)
                if fills.total() == TARGET:
                    break
            if not fills:
                raise ValueError(f"No source works for {domain}/{family}")
            print("held", domain, family, dict(fills), flush=True)
            counts[f"{domain}/{family}"] = fills.total()
        if previous:
            rows.extend(ordinary_controls(domain, tables, excluded, used))
            counts.update({f"{role}/{domain}": amount for role, amount in CONTROL_GROUPS.items()})
            print("ordinary controls", domain, CONTROL_GROUPS, flush=True)
    record = {"schema": schema, "method": MATCHED_METHOD if previous else METHOD, "earlier_panels": prior,
              "shards": pins, "admissions": counts, "rows": rows}
    if previous:
        record["parent_sha256"] = pool.file_hash(previous)
    write(output, record)


def measurements(row, centers):
    for index in centers:
        ratio, tokens, expanded = row["ratio"][index]
        if ratio is not None and not expanded:
            yield index, sum(tokens > edge for edge in EDGES), row["classifier"][index], -ratio


def references(rows, centers):
    values = [[[], []] for _ in range(len(EDGES) + 1)]
    for row in rows:
        peaks = {}
        for _, key, classifier, ratio in measurements(row, centers):
            before = peaks.get(key, (0., -math.inf))
            peaks[key] = max(before[0], classifier), max(before[1], ratio)
        for key, pair in peaks.items():
            for lane, value in enumerate(pair):
                values[key][lane].append(value)
    return [[sorted(lane) for lane in bin_values] for bin_values in values]


def ranks(row, controls, centers):
    for index, key, classifier, ratio in measurements(row, centers):
        limits = controls[key]
        if min(map(len, limits)) < 12:
            continue
        pair = [sum(prior < value for prior in lane) / (len(lane) + 1)
                for lane, value in zip(limits, (classifier, ratio), strict=True)]
        yield index, {"stronger": max(pair), "both": min(pair), "average": sum(pair) / 2}


def make_points(fit_rows, ordinary, centers):
    fit = references(fit_rows, centers)
    values = {mode: [] for mode in RULES}
    for row in ordinary:
        candidates = [scores for _, scores in ranks(row, fit, centers)]
        for mode in RULES:
            values[mode].append(max((item[mode] for item in candidates), default=0.))
    points = {mode: {name: sorted(values[mode], reverse=True)[int(len(ordinary) * share)]
                     for name, share in (("tight", .10), ("broader", .25))}
              for mode in RULES}
    return {"reference_maxima": fit, "calibration_maxima": values, "points": points}


def calibrate(rows=None):
    if rows is not None:
        return {domain: make_points([row for row in rows if row["domain"] == domain and row["role"] == "fit"],
                                    [row for row in rows if row["domain"] == domain and row["role"] == "point"],
                                    HELD_CENTERS) for domain in DOMAINS}
    primary = sources.read(DEVELOPMENT)
    parent.review(primary)
    fit = [row for row in primary["rows"] if row["role"] == "calibration"]
    ordinary = [row for row in primary["rows"] if row["role"] == "held" and not row["truth"]]
    return {"development_sha256": pool.file_hash(DEVELOPMENT), **make_points(fit, ordinary, FIT_CENTERS)}


def policies(row, frozen):
    spans = row["spans"]
    original, overlap = {}, {}
    for index, (value, reading) in enumerate(zip(row["classifier"], row["ratio"], strict=True)):
        ratio, _, expanded = reading
        primary = value >= .5
        pattern = ratio is not None and ratio < POINT
        if primary or pattern:
            left, right = spans[max(index - 1, 0)][0], spans[min(index + 1, len(spans) - 1)][1]
            original[index] = spans[index] if pattern and not expanded else (left, right)
        if primary and pattern and not expanded:
            overlap[index] = spans[index]
    yield "current", original, bool(spans)
    yield "raw_agreement", overlap, any(True for _ in measurements(row, range(len(spans))))
    ranks_by_center = list(ranks(row, frozen["reference_maxima"], HELD_CENTERS))
    for mode in RULES:
        ordered = sorted(((scores[mode], index) for index, scores in ranks_by_center), reverse=True)
        unique = ordered and (len(ordered) == 1 or ordered[0][0] > ordered[1][0])
        for name in ("tight", "broader"):
            chosen = {}
            if unique and ordered[0][0] > frozen["points"][mode][name]:
                index = ordered[0][1]
                chosen[index] = spans[index]
            yield f"{mode}/{name}", chosen, bool(ordered)


def evaluate(rows, frozen, matched=False):
    totals = defaultdict(Counter)
    reference = calibrate() if matched else None
    for row in rows:
        if row["role"] != "held":
            continue
        truth, spaces = row["truth"], set(row["spaces"])
        dimensions = (row["domain"], f"{row['domain']}/{row['generator']}", "all")
        selections = list(policies(row, frozen[row["domain"]] if matched else frozen))
        baseline = selections[0][1]
        baseline_useful = bool(truth and covers(baseline.values(), truth, spaces))
        if reference is not None:
            selections += [(f"government/{mode}", shown, assessed) for mode, shown, assessed in policies(row, reference)
                           if mode not in ("current", "raw_agreement")]
        for mode, shown, assessed in selections:
            for domain in dimensions:
                stats = totals[f"{domain}/{mode}/{'rewrite' if truth else 'ordinary'}"]
                stats["files"] += 1
                stats["assessed"] += int(assessed)
                stats["flagged"] += int(bool(shown))
                if truth:
                    useful = covers(shown.values(), truth, spaces)
                    stats["useful"] += int(useful)
                    stats["exact_center"] += int(row["position"] in shown)
                    stats["wrong_only"] += int(bool(shown) and not useful)
                    if matched and mode != "current":
                        stats["additional_useful"] += int(useful and not baseline_useful)
                        stats["new_wrong_only_file"] += int(bool(shown) and not useful and not baseline)
                        stats["additional_wrong_focus"] += sum(
                            not covers([location], truth, spaces) for index, location in shown.items() if index not in baseline
                        )
                elif matched and mode != "current":
                    stats["additional_ordinary"] += int(bool(shown) and not baseline)
    return {key: dict(value) for key, value in totals.items()}


def validate(document, previous=None):
    matched = document["schema"] == MATCHED_SCHEMA
    if (document["schema"] not in (MATCHED_SCHEMA, SCHEMA)
            or document["method"] != (MATCHED_METHOD if matched else METHOD) or document["shards"] != shard_pins()):
        raise ValueError("The frozen study definition differs")
    excluded = parent.source_exclusions(document["earlier_panels"])
    roles = MATCHED_ROLES if matched else ("held",)
    if matched:
        if previous is None or pool.file_hash(previous) != document["parent_sha256"]:
            raise ValueError("The earlier held study is needed to replay the genre calibration")
        first = sources.read(previous)
        validate(first)
        if first["schema"] != SCHEMA or first["earlier_panels"] != document["earlier_panels"]:
            raise ValueError("The earlier transfer must use the same source exclusions")
        excluded.update(row["group"] for row in first["rows"])
    groups = defaultdict(list)
    for row in document["rows"]:
        spans = row["spans"]
        if (row["role"] not in roles or row["domain"] not in DOMAINS or row["generator"] not in FAMILIES
                or row["group"] in excluded or row["position"] not in HELD_CENTERS or len(spans) != 5
                or not all(SHA256.fullmatch(row[name]) for name in ("group", "sha256"))
                or not all(0 <= a < b for a, b in spans)
                or any(a[1] > b[0] for a, b in zip(spans, spans[1:]))
                or any(type(index) is not int or not 0 <= index < spans[-1][1] for index in row["spaces"])
                or row["origin"] not in (None, *pool.VARIANTS)
                or row["role"] != "held" and row["origin"]
                or row["truth"] != (spans[row["position"]] if row["origin"] else None)
                or len(row["classifier"]) != 5 or len(row["ratio"]) != 5
                or any(not math.isfinite(score) or not 0 <= score <= 1 for score in row["classifier"])
                or any(score is not None and (score <= 0 or not math.isfinite(score) or tokens <= 0)
                       for score, tokens, _ in row["ratio"])):
            raise ValueError("Invalid held source or reading")
        groups[row["group"]].append(row)
    counts = Counter()
    for pair in groups.values():
        row = pair[0]
        if row["role"] == "held" and (
            len(pair) != 2 or sum(entry["origin"] is None for entry in pair) != 1 or any(
                row[key] != pair[1][key] for key in ("role", "domain", "generator", "position"))
        ):
            raise ValueError("A matched ordinary/edited pair is incomplete")
        if row["role"] != "held" and len(pair) != 1:
            raise ValueError("A source work occurs twice across roles")
        key = (f"{row['domain']}/{row['generator']}" if row["role"] == "held" else f"{row['role']}/{row['domain']}")
        counts[key] += 1
    if (counts != document["admissions"]
            or len(counts) != len(DOMAINS) * (len(FAMILIES) + (len(CONTROL_GROUPS) if matched else 0))
            or matched and any(counts[f"{role}/{domain}"] != amount for role, amount in CONTROL_GROUPS.items()
                               for domain in DOMAINS)):
        raise ValueError("Held admissions changed")


def read_measurements(roster, parts, roles):
    data = sources.read(roster)
    reports = {(value["role"], value["phase"]): value for value in map(sources.read, parts)}
    pin = pool.file_hash(roster)
    phases = ("classifier", "ratio")
    if set(reports) != {(role, phase) for role in roles for phase in phases} or any(
        value["roster_sha256"] != pin for value in reports.values()
    ):
        raise ValueError("Reading inputs differ from the held cohort")
    rows = [{key: value for key, value in row.items() if key != "text"} |
            {phase: reports[row["role"], phase]["scores"][row["sha256"]] for phase in phases}
            for row in data["rows"] if row["role"] in roles]
    return data | {"rows": rows, "roster_sha256": pin}


def analyze(roster, parts, calibration, output, previous=None):
    matched = previous is not None
    data = read_measurements(roster, parts, MATCHED_ROLES if matched else ("held",))
    validate(data, previous)
    rows = data["rows"]
    freeze_point = sources.read(calibration)
    if freeze_point != calibrate(rows if matched else None):
        raise ValueError("The frozen control points differ")
    receipt = data | {"frozen": freeze_point, "outcomes": evaluate(rows, freeze_point, matched)}
    write(output, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "classifier", "ratio", "calibrate", "analyze", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--shards", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--previous-transfer", type=Path)
    parser.add_argument("--role", choices=MATCHED_ROLES)
    parser.add_argument("--parts", type=Path, nargs="+")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        exposure = args.previous_transfer or args.prior
        freeze(args.shards, args.path, sources.read(exposure)["earlier_panels"] if exposure else prior_pins(),
               args.previous_transfer)
    elif args.phase in ("classifier", "ratio"):
        parent.readings(args.path, args.role, args.phase, args.output)
    elif args.phase == "calibrate":
        rows = read_measurements(args.path, args.parts, ("fit", "point"))["rows"] if args.parts else None
        write(args.output or args.path, calibrate(rows))
    elif args.phase == "analyze":
        analyze(args.path, args.parts, args.calibration, args.output, args.previous_transfer)
    else:
        record = sources.read(args.path)
        validate(record, args.previous_transfer)
        matched = record["schema"] == MATCHED_SCHEMA
        fitted = calibrate(record["rows"] if matched else None)
        outcomes = evaluate(record["rows"], fitted, matched)
        if record["frozen"] != fitted or record["outcomes"] != outcomes:
            raise ValueError("The frozen calibration or final outcomes differ")
        print(json.dumps(outcomes, sort_keys=True))


if __name__ == "__main__":
    main()
