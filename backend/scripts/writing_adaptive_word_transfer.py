"""Compare fixed and adaptive smoothing of exact word scores on retained writing panels."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from scripts import writing_phrase_context_stability as trial
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_focus_v3 import digest

SCHEMA = "chipmunk.writing-adaptive-word-transfer/1"
PANELS = {
    "kubernetes": "writing-phrase-context-stability-2026-11-12-sentence-fragments",
    "pep": "writing-pep-phrase-replication-2026-11-12",
}
WIDTHS = (1, 3, 7)
ARMS = ("unsmoothed", "fixed_3", "adaptive")
METHOD = {
    "smoothing": "Triangular widths 1, 3, and 7 within the focus sentence. Truncate and renormalize at its ends.",
    "compatibility": "Half the 75th percentile of the paired ordinary word-score gap per width sets a radius. "
    "For each context and word, use the widest estimate compatible with all narrower estimates.",
    "score": "Use the minimum of two contexts after subtracting the stronger four-word flank from each, "
    "with identical eight-word phrase geometry and the existing four-word core guard.",
    "point": "Strictly exceed the second-largest ordinary selection work maximum, with a unique top region.",
    "gate": "Within each held cohort, gain useful locations on at least two works across both generators, "
    "lose no unsmoothed useful location, and add at most one false ordinary work group.",
}


def smooth(scores, width):
    radius = width // 2
    output = []
    for position in range(len(scores)):
        offsets = range(max(-radius, -position), min(radius + 1, len(scores) - position))
        weights = [radius + 1 - abs(delta) for delta in offsets]
        output.append(math.fsum(scores[position + delta] * weight for delta, weight in zip(offsets, weights, strict=True))
                      / sum(weights))
    return output


def compatible(vectors, radii):
    output, counts = [], Counter()
    for values in zip(*vectors, strict=True):
        choice = max(i for i in range(len(WIDTHS)) if all(
            abs(values[i] - values[j]) <= radii[i] + radii[j] for j in range(i)))
        output.append(values[choice])
        counts[str(WIDTHS[choice])] += 1
    return output, counts


def focus_scores(row, responses, used):
    text = row["text"]
    require(trial.text_hash(text) == row["sha256"], "Original phrase fixture text differs")
    for focus in trial.bindings(text):
        views = focus["views"]
        for view in views.values():
            source = view["text"]
            used.add(source)
            response = responses[source]
            require(response["input_sha256"] == trial.text_hash(source), "Native model input hash differs")
        if len(views) != 3:
            continue
        start, stop = focus["span"]
        words = list(trial.WORDS.finditer(text[start:stop]))
        contexts = []
        for name in ("left", "right"):
            view = views[name]
            shift = view["offset"]
            require(view["text"][shift:shift + stop - start] == text[start:stop], "Focus text does not align")
            tokens = responses[view["text"]]["offsets"]
            require(all(0 <= a < b <= len(view["text"]) and math.isfinite(value) for a, b, value in tokens),
                    "Invalid native token score or position")
            scores = [trial.reading(tokens, shift + word.start(), shift + word.end()) for word in words]
            contexts.append([smooth(scores, width) for width in WIDTHS])
        candidates = []
        for i in range(trial.FLANK, len(words) - trial.WIDTH - trial.FLANK + 1):
            j = i + trial.WIDTH
            span = [start + words[i].start(), start + words[j - 1].end()]
            if trial.clean_phrase(text[slice(*span)]):
                candidates.append((i, j, span))
        require(candidates, "Native phrase plan has no candidate")
        yield contexts, candidates


def load(panel, role):
    root = trial.BASE / PANELS[panel]
    response = read(root / f"{role}-responses.json")
    require(response["fixture_sha256"] == digest(root / "fixtures.json"), "Frozen native response binding differs")
    rows = [row for row in read(root / "fixtures.json")["fixtures"] if row["role"] == role]
    used = set()
    inputs = [(row, list(focus_scores(row, response["responses"], used))) for row in rows]
    require(used == set(response["responses"]) and len(used) == response["inputs"], "Native input roster differs")
    return inputs, len(used)


def contrast(scores, start, end):
    core = min(mean(scores[start:end]), mean(scores[start + 2:end - 2]))
    return core - max(mean(scores[start - 4:start]), mean(scores[end:end + 4]))


def winners(focuses, radii):
    candidates, counts = [], Counter()
    for (left, right), positions in focuses:
        left_adaptive, left_counts = compatible(left, radii)
        right_adaptive, right_counts = compatible(right, radii)
        counts.update(left_counts)
        counts.update(right_counts)
        for start, end, span in positions:
            scores = [min(contrast(a, start, end), contrast(b, start, end))
                      for a, b in ((left[0], right[0]), (left[1], right[1]), (left_adaptive, right_adaptive))]
            candidates.append((span, scores))
    result = {}
    for index, name in enumerate(ARMS):
        ordered = sorted(candidates, key=lambda item: item[1][index], reverse=True)
        if not ordered:
            result[name] = None
        else:
            span, scores = ordered[0]
            result[name] = {"span": span, "score": scores[index],
                            "unique": len(ordered) == 1 or scores[index] > ordered[1][1][index]}
    return result, counts, len(candidates)


def selection():
    inputs, count = load("kubernetes", "calibration")
    differences = [[] for _ in WIDTHS]
    groups = defaultdict(list)
    for row, focuses in inputs:
        require(row["kind"] in {"original", "ordinary"}, "Selection received a generated case")
        for (left, right), _ in focuses:
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                differences[index].extend(abs(x - y) for x, y in zip(a, b, strict=True))
    radii = [sorted(values)[math.ceil(.75 * len(values)) - 1] / 2 for values in differences]
    for row, focuses in inputs:
        selected, _, _ = winners(focuses, radii)
        require(all(selected.values()), "Selection missing a phrase winner")
        groups[row["group"]].append(selected)
    maxima = [{name: max(row[name]["score"] for row in rows) for name in ARMS} for rows in groups.values()]
    require(len(maxima) == trial.CAL_GROUPS and all(len(rows) == 2 for rows in groups.values()),
            "Selection group roster differs")
    points = {name: sorted((group[name] for group in maxima), reverse=True)[1] for name in ARMS}
    return {"radii": radii, "points": points, "group_maxima": maxima, "model_inputs": count,
            "ordinary_context_gaps": [len(values) for values in differences]}


def geometry(text, target, region):
    if target is None or region is None:
        return None
    active = {i for i in range(*region) if not text[i].isspace()}
    truth = {i for i in range(*target) if not text[i].isspace()}
    return [len(active & truth), len(active | truth), len(truth)]


def held(panel, points, radii):
    inputs, count = load(panel, "holdout")
    selection_rows = read(trial.BASE / PANELS["kubernetes"] / "fixtures.json")["fixtures"]
    require({row["group"] for row, _ in inputs}.isdisjoint(
        row["group"] for row in selection_rows if row["role"] == "calibration"), "Selection and held source groups overlap")
    groups = {}
    width_use, cases = Counter(), []
    for row, focuses in inputs:
        group = groups.setdefault(row["group"], len(groups) + 1)
        proposed, counts, total = winners(focuses, radii)
        width_use.update(counts)
        generated = row["kind"] in trial.SEEDS
        target = row["span"] if generated else None
        cases.append({"group": group, "kind": row["kind"], "candidate_count": total, "target": target, "methods": {
            name: proposal | {"overlap": geometry(row["text"], target, proposal["span"]),
                              "flag": proposal["unique"] and proposal["score"] > points[name]}
            if proposal else None for name, proposal in proposed.items()}})
    return {"model_inputs": count, "selected_widths": dict(width_use), "cases": cases}


def lock():
    paths = {"kubernetes": ("fixtures.json", "calibration-responses.json", "holdout-responses.json"),
             "pep": ("fixtures.json", "holdout-responses.json")}
    return {"schema": SCHEMA, "method": METHOD, "widths": list(WIDTHS), "runner_sha256": digest(Path(__file__)),
            "sources": {panel: {name: digest(trial.BASE / PANELS[panel] / name) for name in names}
                        for panel, names in paths.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "held"))
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    frozen = lock()
    path = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir()
        write(path, frozen | {"selection": selection()})
        return
    fitted = selection()
    require(read(path) == frozen | {"selection": fitted}, "Source or frozen method differs")
    output = frozen | {"selection": fitted,
                       "held": {panel: held(panel, fitted["points"], fitted["radii"]) for panel in PANELS}}
    write(args.run / "result.json", output)


if __name__ == "__main__":
    main()
