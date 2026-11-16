# SPDX-License-Identifier: MIT
"""Measure whether word ranges improve when compared at the same length."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_curvature_range_transfer as curvature
from scripts import writing_focus_inside_context as intake
from scripts import writing_word_localization as words

pool = intake.pool
SCHEMA = "chipmunk.writing-width-calibration/1"
ESSAYS = "chipmunk.writing-width-calibration-essays/1"
ESSAY_PINS = {
    "gemini-2.5-flash": "d4805b719907ea7e3e7215c0cbfd5d47e95f7bad394a51fcdc851d4bc029222e",
    "gpt-5.4-nano": "b1177c4cbda63ccd623dc903037aa8e3ac20b31fcb5e4fe8bcf93c32980f3045",
}
EXPOSURE = tuple(sorted({*pool.EXPOSURE, intake.PARENT.name,
                         "writing-focus-inside-context-2026-11-14.json"}))
COUNT = 36
CURVES = ("absolute", "two_flanks")
ARMS = (*CURVES, "width_absolute", "width_two_flanks", "width_either")
PROPOSALS = ("paired", "absolute", "width_absolute", "width_two_flanks", "width_either")
PANELS = pool.PANEL_DIR
BASE = PANELS / "writing-curvature-ranges-2026-11-15.json.gz"
SENTENCE_BASE = PANELS / "writing-independent-range-transfer-2026-11-15.json.gz"
METHOD = {
    "geometry": "Keep the prior word grid and evidence scores. Group maxima by exact proposed width and use a "
                "separate group for complete sentences whose length is outside that grid.",
    "standardization": "For each genre, score and width, subtract the ordinary-work median and divide by the "
                       "difference from the median to the 90th percentile, with a 0.25 floor and at least 12 works.",
    "multiple_checks": "Compare raw and width-adjusted selections. A joint arm selects the larger of the two "
                       "adjusted scores. Estimate both width profiles and file cutoffs on ordinary works only.",
    "points": "Use the maximum eligible range for each work. Withhold ties. Each ordinary file is ranked "
              "using width profiles that leave out its work. Use the two prior ordinary alert budgets.",
    "held": "Replay the prior report, news and journal panels without refitting on them. Use 36 ordinary "
            "calibration essays and separate held works with 18 matched original and rewrite pairs per generator.",
}
SENTENCE_METHOD = (
    "The previous fixed independent sentence score decides whether and where to flag. Compare only word proposals "
    "inside that sentence, using the same ordinary work points. On the essays verify the same pinned classifier "
    "and compute its own genre points. Look for wrong sentences, missed word overlap, and ordinary flags."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exposure_pins():
    return {name: pool.file_hash(pool.PANEL_DIR / name) for name in EXPOSURE}


def assay_text(gemini, nano, *, extra_exclusions=(), exclude_phrases=(), seed=ESSAYS,
               held_per_family=COUNT // 2, held_first=False, held_position_limit=None, shard_pins=ESSAY_PINS):
    paths = (gemini,) if nano is None else (gemini, nano)
    sources = [
        intake.shard(path, "essays", family, pin)
        for path, (family, pin) in zip(paths, shard_pins.items(), strict=True)
    ]
    exposure = exposure_pins()
    known = {digest for name in exposure
             for digest in re.findall(r'"([0-9a-f]{64})"', (pool.PANEL_DIR / name).read_text())}
    known.update(extra_exclusions)
    prior_phrases = {phrase for table in sources for key in table if pool.sha(key) in known
                     for phrase in intake.fingerprints(table[key]["v0"]["text"])}
    prior_phrases.update(exclude_phrases)
    rows = []
    calibration = ("original", {key: value for table in sources for key, value in table.items()}, COUNT)
    held = tuple((family, table, held_per_family) for family, table in zip(shard_pins, sources, strict=True))
    for family, table, limit in ((*held, calibration) if held_first else (calibration, *held)):
        chosen = Counter()
        role = "calibration" if family == "original" else "held"
        position_limit = held_position_limit if role == "held" and held_position_limit is not None else limit // 3
        order = sorted(table, key=lambda key: pool.sha(f"{seed}/{family}/{key}"))
        for key in order:
            raw, labels = pool.truth_text(table[key]["v0"])
            group = pool.sha(key)
            if labels or group in known:
                continue
            phrases = intake.fingerprints(raw)
            if phrases & prior_phrases:
                continue
            options = {}
            try:
                if role == "calibration":
                    for window in pool.source_windows(raw):
                        try:
                            excerpt, bounds = pool.normalize(raw, window)
                        except ValueError:
                            continue
                        options = {position: ((excerpt, bounds, None),) for position in curvature.parent.CENTERS}
                        break
                else:
                    options = intake.excerpt_options(key, table[key])
            except (KeyError, ValueError):
                continue
            available = [position for position in options if chosen[position] < position_limit]
            if not available:
                continue
            target = min(available, key=lambda position: (chosen[position], position))
            families = ("original", family) if role == "held" else ("original",)
            pair = [{"group": group, "role": role, "family": family, "position": target,
                     "sha256": pool.sha(text), "text": text, "spans": spans, "version": version}
                    for family, (text, spans, version) in zip(families, options[target], strict=True)]
            for row in pair:
                row["words"] = [list(item[:3]) for item in words.items(row)]
                row["changed"] = sorted(words.edit_words(pair[0], row)) if row["version"] else []
            if any(row["version"] and not row["changed"] for row in pair):
                continue
            rows.extend(pair)
            prior_phrases.update(phrases)
            known.add(group)
            chosen[target] += 1
            if sum(chosen.values()) == limit:
                break
        require(sum(chosen.values()) == limit, f"The source lacks the required distinct {family} works: {dict(chosen)}")
    return {"schema": ESSAYS, "sources": shard_pins, "exclusions": exposure, "rows": rows}


def save(output, data):
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with output.open("xb") as out:
        out.write(gzip.compress(raw, mtime=0))


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def capture_essays(path, plan, model):
    require(plan["schema"] == ESSAYS and plan["sources"] == ESSAY_PINS and plan["exclusions"] == exposure_pins(),
            "The essay input plan differs")
    measure = curvature.causal_reader(model)
    readings = {}
    for index, row in enumerate(plan["rows"], start=1):
        require(pool.sha(row["text"]) == row["sha256"], "An essay's source bytes changed")
        readings[row["sha256"]] = measure(row["text"], row["words"])
        if index % 12 == 0:
            print(f"Scored {index} excerpts", flush=True)
    public_rows = [{key: value for key, value in row.items() if key != "text"} for row in plan["rows"]]
    save(path, {**plan, "rows": public_rows, "readings": readings, "model": curvature.REVISION,
                "runtime": curvature.RUNTIME, "method": METHOD})


def capture_sentences(path, plan, assets):
    import numpy as np

    reader = curvature.parent
    model, tokenizer, runtime = reader.load_classifier(assets)
    require(runtime == "1.24.3", "The sentence scorer needs the pinned ONNX Runtime")
    readings = {}
    for row in plan["rows"]:
        text = row["text"]
        require(pool.sha(text) == row["sha256"], "An essay's source bytes changed")
        scores = []
        for position in reader.CENTERS:
            a, b = row["spans"][position]
            ids = tokenizer.encode(text[a:b]).ids
            require(2 < len(ids) <= 512, "The essay sentence exceeds the trained tokenizer extent")
            inputs = np.array([ids], dtype=np.int64)
            [logits] = model.run(["logits"], {"input_ids": inputs, "attention_mask": np.ones_like(inputs)})
            scores.append(round(float(logits[0, 1]) - float(logits[0, 0]), 7))
        readings[row["sha256"]] = scores
    save(path, {"reader": reader.RELEASE, "assets": reader.ASSETS, "runtime": runtime, "method": SENTENCE_METHOD,
                "readings": readings})


def range_buckets(row, profile):
    """A monotone width adjustment can only select one of each bucket's maxima."""
    buckets = defaultdict(list)
    for position, curves in curvature.range_scores(row, profile).items():
        for arm in CURVES:
            for score, _idx, start, stop in curves.get(arm, []):
                width = stop - start
                band = str(width) if width in curvature.WIDTHS else "whole_sentence"
                buckets[arm, band, position].append((score, position, start, stop))
    return {key: sorted(values, reverse=True)[:2] for key, values in buckets.items()}


def percentile(values, fraction):
    left, remainder = divmod((len(values) - 1) * fraction, 1)
    x = int(left)
    return values[x] + remainder * (values[min(x + 1, len(values) - 1)] - values[x])


def fit(rows, scored, *, exclude=None):
    distributions = defaultdict(list)
    for row in rows:
        if row["group"] == exclude:
            continue
        maxima = {}
        for (arm, width, _idx), candidates in scored[row["sha256"]].items():
            key = arm, width
            maxima[key] = max(candidates[0][0], maxima.get(key, -math.inf))
        for key, score in maxima.items():
            distributions[key].append(score)
    result = {}
    for key, values in distributions.items():
        if len(values) >= 12:
            values.sort()
            midpoint = percentile(values, .5)
            result[key] = midpoint, max(.25, percentile(values, .9) - midpoint), len(values)
    return result


def candidates(scored, fit_, *, focus=None):
    values = defaultdict(set)
    for (arm, width, position), maxima in scored.items():
        if focus is not None and focus != position:
            continue
        values[arm].update(maxima)
        if (profile := fit_.get((arm, width))) is None:
            continue
        center, scale, _count = profile
        for raw, i, a, b in maxima:
            row = round((raw - center) / scale, 7), i, a, b
            values[f"width_{arm}"].add(row)
            values["width_either"].add(row)
    selected = {}
    for arm in ARMS:
        ranked = sorted(values[arm], reverse=True)
        if not ranked or len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
            continue
        selected[arm] = ranked[0]
    return selected


def cutoffs(rows, scored):
    require(len({row["group"] for row in rows}) == len(rows), "Ordinary calibration reuses a source work")
    controls = [candidates(scored[row["sha256"]], fit(rows, scored, exclude=row["group"])) for row in rows]
    cuts = {}
    for arm in ARMS:
        strongest = sorted((max(0., v[arm][0]) if arm in v else 0. for v in controls), reverse=True)
        cuts[arm] = {level: strongest[len(strongest) // divisor] for level, divisor in curvature.parent.previous.LEVELS.items()}
    return cuts


def useful(span, change):
    return bool(span and words.useful_words(set(range(span[2], span[3])), change))


def evaluate_panel(rows, scored, profiles, points):
    selected = {row["sha256"]: candidates(scored[row["sha256"]], profiles) for row in rows}
    matched_original = {row["group"]: selected[row["sha256"]] for row in rows if row["family"] in {"original", "before"}}
    out = defaultdict(Counter)
    ordinary = curvature.parent.previous.ORDINARY
    for row in rows:
        true = set(row["changed"])
        guesses = selected[row["sha256"]]
        target = candidates(scored[row["sha256"]], profiles, focus=row["position"]) if true else {}
        for arm in ARMS:
            baseline = arm.removeprefix("width_") if arm != "width_either" else "absolute"
            raw = guesses.get(baseline)
            pick = guesses.get(arm)
            paired = matched_original.get(row["group"], {}).get(arm)
            for level, threshold in points[arm].items():
                flagged = bool(pick and pick[0] > threshold)
                hit = flagged and useful(pick, true)
                initial = bool(raw and raw[0] > points[baseline][level] and useful(raw, true))
                same = bool(flagged and paired and paired[0] > threshold and paired[1] == pick[1])
                key = f"{arm}/{level}/{row['family']}"
                out[key].update(files=1, assessed=bool(pick), flagged=flagged,
                                exact_sentence=row["family"] not in ordinary and flagged and pick[1] == row["position"],
                                useful=hit, gained=hit and not initial, lost=initial and not hit,
                                matched_ordinary_same_sentence=row["family"] not in ordinary and same,
                                useful_given_sentence=useful(target.get(arm), true))
    return dict(out)


def essay_rows(receipt):
    require(receipt["schema"] == ESSAYS and receipt["sources"] == ESSAY_PINS and receipt["exclusions"] == exposure_pins()
            and receipt["model"] == curvature.REVISION and receipt["runtime"] == curvature.RUNTIME and receipt["method"] == METHOD,
            "The bound essay source or reader changed")
    point, held = [], []
    grouped = defaultdict(list)
    for row in receipt["rows"]:
        digest = row["sha256"]
        reading = receipt["readings"][digest]
        measures = reading["words"]
        spans = row["spans"]
        require("text" not in row and row["position"] in curvature.parent.CENTERS
                and len(spans) == 5 and all(0 <= a < b for a, b in spans)
                and all(x[1] <= y[0] for x, y in zip(spans, spans[1:]))
                and all(0 <= i < len(spans) and spans[i][0] <= a < b <= spans[i][1] for a, b, i in row["words"])
                and all(x[1] <= y[0] for x, y in zip(row["words"], row["words"][1:]))
                and (measures is None) == (not 2 <= reading["token_count"] <= curvature.TOKEN_LIMIT)
                and (measures is None or len(measures) == len(row["words"])), "Essay source geometry changed")
        require(measures is None or all(vals is None or (len(vals) == 2 and all(math.isfinite(value) for value in vals)
                                                        and vals[1] > 0.) for vals in measures), "Invalid numeric token reading")
        require(row["changed"] == sorted(set(row["changed"]))
                and all(0 <= i < len(row["words"]) and row["words"][i][2] == row["position"] for i in row["changed"]),
                "Changed words leave the target sentence")
        grouped[row["group"]].append(row)
        (point if row["role"] == "calibration" else held).append(row)
    require(len(point) == COUNT and len(held) == 2 * COUNT and len(grouped) == 2 * COUNT
            and len({row["sha256"] for row in receipt["rows"]}) == len(receipt["readings"]) == 3 * COUNT,
            "Essay roster size differs")
    for group in grouped.values():
        roles = {r["role"] for r in group}
        families = {r["family"] for r in group}
        require((len(group) == 1 and roles == {"calibration"} and families == {"original"})
                or (len(group) == 2 and roles == {"held"} and "original" in families and len(families & ESSAY_PINS.keys()) == 1),
                "An essay work was reused")
        require(len({r["position"] for r in group}) == 1
                and all(bool(r["changed"]) == (r["family"] != "original") for r in group), "Matched essay labels differ")
    for family in ESSAY_PINS:
        require(Counter(row["position"] for row in held if row["family"] == family)
                == dict.fromkeys(curvature.parent.CENTERS, COUNT // 6), "The held generator roster differs")
    return point, held


def sentence_selection(row, references, readings, *, exclude=None):
    candidates = []
    for position, reading in zip(curvature.parent.CENTERS, readings[row["sha256"]], strict=True):
        ordinary = sorted(value for group, value in references[position].items() if group != exclude)
        rank = 10_000 * bisect_left(ordinary, reading) // (len(ordinary) + 1)
        candidates.append((rank, position))
    chosen, next_, _ = sorted(candidates, reverse=True)
    return chosen if chosen[0] != next_[0] else (0, None)


def nominated_ranges(rows, references, readings, scored, profiles, points):
    out = defaultdict(Counter)
    ordinary = curvature.parent.previous.ORDINARY
    original = {row["group"]: sentence_selection(row, references, readings)
                for row in rows if row["family"] in {"original", "before"}}
    for row in rows:
        rank, focus = sentence_selection(row, references, readings)
        truth = set(row["changed"])
        options = candidates(scored[row["sha256"]], profiles, focus=focus) if focus is not None else {}
        raw = useful(options.get("absolute"), truth)
        known = next((p["words"] for p in row.get("by_sentence", []) if p["sentence"] == focus), [])
        paired = bool(known and words.useful_words(set(known), truth))
        for arm in PROPOSALS:
            if arm == "paired" and "by_sentence" not in row:
                continue
            alternative = set(range(options[arm][2], options[arm][3])) if arm in options else set()
            for level, point in points.items():
                flagged = focus is not None and rank > point
                hit = flagged and (paired if arm == "paired" else words.useful_words(alternative, truth))
                previous_rank, previous_focus = original.get(row["group"], (0, None))
                out[f"{arm}/{level}/{row['family']}"].update(
                    files=1, sentence_flagged=flagged, word_range=flagged and bool(known if arm == "paired" else alternative),
                    right_sentence=flagged and row["family"] not in ordinary and focus == row["position"],
                    useful_with_matched_ordinary_flag=hit and previous_rank > point and previous_focus == focus,
                    useful=hit, gained=hit and not (flagged and paired), lost=flagged and paired and not hit,
                    gained_over_raw=hit and not (flagged and raw), lost_over_raw=flagged and raw and not hit,
                )
    return dict(out)


def sentence_comparison(calibration, panels, baseline, essays, sentence, scored, profiles):
    reader = curvature.parent
    prior = read(SENTENCE_BASE)
    require(all(prior[key] == value for key, value in reader.evaluate(prior).items()), "Prior sentence readings differ")
    extra = sentence["readings"]
    require(sentence["reader"] == reader.RELEASE and sentence["assets"] == reader.ASSETS
            and sentence["runtime"] == "1.24.3" and sentence["method"] == SENTENCE_METHOD
            and set(extra) == set(essays["readings"])
            and all(len(scores) == 3 and all(math.isfinite(v) for v in scores) for scores in extra.values()),
            "The essay sentence receipt differs")
    readings = prior["readings"] | extra
    points, refs = {}, {}
    for genre, rows in calibration.items():
        ref = refs[genre] = {position: {row["group"]: readings[row["sha256"]][i] for row in rows}
                            for i, position in enumerate(reader.CENTERS)}
        control_ranks = sorted((sentence_selection(row, ref, readings, exclude=row["group"])[0] for row in rows), reverse=True)
        points[genre] = {level: control_ranks[len(control_ranks) // divisor]
                         for level, divisor in curvature.parent.previous.LEVELS.items()}
        if genre in baseline:
            require(points[genre] == prior["points"][genre]["independent"], "A frozen sentence point changed")
    return {"method": SENTENCE_METHOD, "points": points,
            "outcomes": {name: nominated_ranges(rows, refs[genre], readings, scored, profiles[genre], points[genre])
                         for name, (genre, rows) in panels.items()}}


def evaluate(essays, sentence):
    calibration, panels, baseline, sources = curvature.parent.reference()
    historic = read(BASE)
    curvature.compute((calibration, panels, baseline, sources), historic)
    ordinary, focused = essay_rows(essays)
    calibration["essays"] = ordinary
    panels["essays"] = "essays", focused
    readings = historic["readings"] | essays["readings"]
    scored = {row["sha256"]: range_buckets(row, readings[row["sha256"]])
              for rows in (*calibration.values(), *(rows for _, rows in panels.values())) for row in rows}
    profiles = {genre: fit(rows, scored) for genre, rows in calibration.items()}
    points = {genre: cutoffs(rows, scored) for genre, rows in calibration.items()}
    result = {panel: evaluate_panel(rows, scored, profiles[genre], points[genre]) for panel, (genre, rows) in panels.items()}
    return {"points": points, "width_profiles": {genre: {f"{arm}/{width}": list(profile) for (arm, width), profile in parts.items()}
                                               for genre, parts in profiles.items()}, "outcomes": result,
            "sentence_then_words": sentence_comparison(calibration, panels, baseline, essays, sentence, scored, profiles)}


def assemble(essay_path, sentence_path):
    essays, sentences = read(essay_path), read(sentence_path)
    return {"schema": SCHEMA, "base_sha256": pool.file_hash(BASE), "sentence_base_sha256": pool.file_hash(SENTENCE_BASE),
            "essays": essays, "essay_sentence_readings": sentences, "report": evaluate(essays, sentences)}


def review(path):
    data = read(path)
    require(data["schema"] == SCHEMA and data["base_sha256"] == pool.file_hash(BASE)
            and data["sentence_base_sha256"] == pool.file_hash(SENTENCE_BASE), "The previous word or sentence readings differ")
    report = evaluate(data["essays"], data["essay_sentence_readings"])
    require(data["report"] == report, "The published localizations do not reproduce")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="phase", required=True)
    freeze = subs.add_parser("freeze-essays")
    freeze.add_argument("output", type=Path)
    freeze.add_argument("gemini", type=Path)
    freeze.add_argument("nano", type=Path)
    capture = subs.add_parser("capture-essays")
    capture.add_argument("output", type=Path)
    capture.add_argument("plan", type=Path)
    capture.add_argument("model", type=Path)
    sentence = subs.add_parser("capture-sentences")
    sentence.add_argument("output", type=Path)
    sentence.add_argument("plan", type=Path)
    sentence.add_argument("assets", type=Path)
    summary = subs.add_parser("assemble")
    summary.add_argument("output", type=Path)
    summary.add_argument("essays", type=Path)
    summary.add_argument("--sentences", type=Path, required=True)
    check = subs.add_parser("review")
    check.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze-essays":
        save(args.output, assay_text(args.gemini, args.nano))
    elif args.phase == "capture-essays":
        capture_essays(args.output, read(args.plan), args.model)
    elif args.phase == "capture-sentences":
        capture_sentences(args.output, read(args.plan), args.assets)
    elif args.phase == "assemble":
        save(args.output, assemble(args.essays, args.sentences))
    else:
        print(json.dumps(review(args.receipt), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
