"""Measure ordered token-surprisal features at exact edited sentence positions."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from statistics import fmean, median, pstdev

from scripts import writing_revision_trajectory_transfer as source
from scripts import writing_temperature_transfer as short

SCHEMA = "chipmunk.writing-surprisal-dynamics/1"
PRIOR = short.PANELS / "writing-word-edit-transfer-2026-11-13.json"
PRIOR_SHA = "86837d924e3c5256f1bd731ffb56fcf0363090df48f3883bce3ac8a9da4367ec"
ROSTER_SHA = "217eb06cc145d060cfdb1242d28219d1a0a9817a3278744bb57bdcc45a739ee9"
COUNTS = {"fit": 72, "calibration": 42, "report": 60, "forum": 16, "uswds": 12, "elife": 36}
MODELS = ("static", "ordered", "permuted")
MIN_TOKENS, MAX_TOKENS = 12, 512
FEATURES = ("mean", "standard_deviation", "skewness", "excess_kurtosis",
            "mean_absolute_first_difference", "second_difference_standard_deviation", "lag_one_correlation")
METHOD = {
    "source": "Fit on 24 prior matched GAO works, and set the point on a separate 21 genuine journal works. "
              "Hold out 20 new GAO works selected by the frozen matched-polish intake after excluding both older banks. "
              "Report previously exposed eLife, forum and USWDS controls separately.",
    "measurement": "Read the exact submitted excerpt once with a pinned base causal model. Compare "
                   "distributional features, those plus ordered dynamics, and the same dynamics after a fixed "
                   "within-sentence permutation. Subtract the per-excerpt feature median as an extra input for each feature.",
    "decision": "Fit class- and work-balanced logistic heads with fixed ridge penalty. Select one top native sentence "
                "strictly above the second-highest separate ordinary work maximum. Compare with the current primary "
                "proxy's first six native three-sentence contexts at the original research point.",
    "advance": "The ordered check must add four strict report locations across both generators, including at "
               "least two gains beyond each same-model control. Allow at most one extra flagged genuine work per "
               "held cohort and require 90% held edited-report coverage. No authorship confidence is produced.",
}


def native_row(row):
    checked = short.record(row["text"], row["cohort"], row["group"], row["family"], row["truth"])
    source.require(all(row[key] == checked[key] for key in checked if key != "id"),
                   "Retained native source geometry differs")
    return row


def case_id(row):
    cohort = "development" if row["cohort"] == "calibration" else row["cohort"]
    return source.sha(f"{cohort}/{row['group']}/{row['family']}")


def intake(prior_roster, cache):
    from scripts.writing_word_edit_transfer import held_reports

    source.require(source.digest(PRIOR) == PRIOR_SHA and source.digest(prior_roster) == ROSTER_SHA,
                   "Prior report or native roster differs")
    old = list(map(native_row, source.read(prior_roster)))
    old_public = {row["id"]: row for row in source.read(PRIOR)["rows"]}
    for row in old:
        if (parent := old_public.get(row["id"])) is not None:
            keys = ("group", "family", "cohort", "text_sha256", "truth", "spans", "spaces")
            source.require(all(row[key] == parent[key] for key in keys), "Prior public and native sources differ")
    exclusions = [dict(row, cohort="report") for row in old if row["cohort"] in ("fit", "report")]
    held, counts = held_reports(cache, exclusions)
    rows = [dict(row, id=source.sha(f"fit/{row['group']}/{row['family']}")) if row["cohort"] == "fit" else row
            for row in old if row["cohort"] != "report"] + held
    sets = {cohort: {row["group"] for row in rows if row["cohort"] == cohort} for cohort in COUNTS}
    source.require(Counter(row["cohort"] for row in rows) == COUNTS
                   and all(row["id"] == case_id(row) for row in rows)
                   and len({row["id"] for row in rows}) == len(rows)
                   and sum(map(len, sets.values())) == len(set().union(*sets.values())),
                   "Incomplete or overlapping source cohorts")
    return rows, counts


def freeze(run, prior_roster, cache):
    from scripts.writing_focus_v3 import BASELINE_ASSETS, BASELINE_REVISION
    from scripts.writing_token_curvature import MODEL_ASSETS, REVISION

    rows, counts = intake(prior_roster, cache)
    source.seal(run / "roster.json", rows)
    lock = {
        "schema": SCHEMA, "method": METHOD, "features": FEATURES,
        "runner_sha256": source.digest(Path(__file__)), "roster_sha256": source.digest(run / "roster.json"),
        "previous_receipt_sha256": PRIOR_SHA, "previous_roster_sha256": ROSTER_SHA, "intake": counts,
        "causal_assets": MODEL_ASSETS, "causal_revision": REVISION,
        "primary_assets": BASELINE_ASSETS, "primary_revision": BASELINE_REVISION,
        "versions": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "transformers", "torch", "pysbd")},
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    }
    source.seal(run / "freeze.json", lock)
    print("frozen", counts, flush=True)


def locked(run):
    lock = source.read(run / "freeze.json")
    source.require(lock["runner_sha256"] == source.digest(Path(__file__))
                   and lock["roster_sha256"] == source.digest(run / "roster.json"), "Frozen runner or source differs")
    return lock, source.read(run / "roster.json")


def check_offline(lock):
    source.require(all(os.getenv(name) == "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
                   and all(importlib.metadata.version(key) == value for key, value in lock["versions"].items()),
                   "Use the frozen offline runtime")


def dynamic(values):
    first = [b - a for a, b in zip(values, values[1:])]
    second = [b - a for a, b in zip(first, first[1:])]
    left, right = values[:-1], values[1:]
    m1, m2, s1, s2 = fmean(left), fmean(right), pstdev(left), pstdev(right)
    correlation = fmean((x - m1) * (y - m2) for x, y in zip(left, right, strict=True)) / (s1 * s2) if s1 * s2 else 0.0
    return [fmean(abs(x) for x in first), pstdev(second), correlation]


def moments(values):
    avg, sd = fmean(values), pstdev(values)
    z = [(value - avg) / sd for value in values] if sd else [0.0] * len(values)
    static = [avg, sd, fmean(value**3 for value in z), fmean(value**4 for value in z) - 3]
    indices = list(range(len(values)))
    Random(6202 + len(values)).shuffle(indices)
    return [round(x, 7) for x in static + dynamic(values) + dynamic([values[i] for i in indices])]


def causal(run, snapshot, stage):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from transformers import AutoTokenizer

    from scripts.writing_token_curvature import target_token_indices, verified_model

    lock, rows = locked(run)
    check_offline(lock)
    points_hash = None
    if stage == "held":
        saved = run / "points.json"
        source.require(source.read(saved)["freeze_sha256"] == source.digest(run / "freeze.json"), "Missing pre-held points")
        points_hash = source.digest(saved)
    verified_model(snapshot)
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(96 * 1024 * 1024)
    model, _ = load(str(snapshot), tokenizer_config={"trust_remote_code": False})
    nn.quantize(model, **lock["quantization"])
    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    head = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    outputs = {}
    for row in rows:
        if (row["cohort"] in {"fit", "calibration"}) != (stage == "development"):
            continue
        text = row["text"]
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens, offsets = encoded["input_ids"], encoded["offset_mapping"]
        foci = []
        for span in row["spans"]:
            try:
                indices = target_token_indices(text, offsets, span)
            except ValueError:
                indices = []
            foci.append(indices if len(indices) >= MIN_TOKENS and len(tokens) <= MAX_TOKENS else [])
        selected = sorted({i for indices in foci for i in indices})
        measured = {}
        if selected:
            hidden = model.model(mx.array([tokens]))[0]
            mx.eval(hidden)
            for start in range(0, len(selected), 16):
                batch = selected[start:start + 16]
                logits = head(hidden[mx.array([i - 1 for i in batch])]).astype(mx.float32)
                observed = logits[mx.arange(len(batch)), mx.array([tokens[i] for i in batch])]
                scores = (mx.logsumexp(logits, axis=-1) - observed).tolist()
                source.require(all(math.isfinite(value) and value >= 0 for value in scores), "Invalid token surprisal")
                measured.update(zip(batch, scores, strict=True))
        outputs[row["id"]] = [None if not indices else {"tokens": len(indices),
                              "token_indices_sha256": source.sha(source.serialize(indices)),
                              "features": moments([measured[i] for i in indices])} for indices in foci]
        if len(outputs) % 20 == 0:
            print(stage, len(outputs), flush=True)
    source.seal(run / f"{stage}.json", {"freeze_sha256": source.digest(run / "freeze.json"),
                "points_sha256": points_hash, "rows": outputs})


def vectors(readings, model):
    indices = {"static": [0, 1, 2, 3], "ordered": list(range(7)), "permuted": [0, 1, 2, 3, 7, 8, 9]}[model]
    candidates = [[item["features"][i] for i in indices] if item else None for item in readings]
    available = [item for item in candidates if item is not None]
    if len(available) < 3:
        return [None] * len(candidates)
    centers = [median(values) for values in zip(*available, strict=True)]
    return [item + [x - y for x, y in zip(item, centers, strict=True)] if item is not None else None
            for item in candidates]


def summarize_scores(row, values, models):
    scores = {}
    for name, parameters in models.items():
        means, scales, coefs = (parameters[key] for key in ("means", "scales", "coefs"))
        scores[name] = [None if vector is None else round(coefs[0] + math.fsum(
            c * (x - m) / s for c, x, m, s in zip(coefs[1:], vector, means, scales, strict=True)), 7)
            for vector in vectors(values[row["id"]], name)]
    return scores


def fit_heads(rows, values):
    import numpy as np

    heads = {}
    for name in MODELS:
        groups = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if row["cohort"] != "fit":
                continue
            for span, vector in zip(row["spans"], vectors(values[row["id"]], name), strict=True):
                if vector is not None:
                    groups[row["group"]][int(span == row["truth"])].append(vector)
        source.require(len(groups) >= 20 and all(set(group) == {0, 1} for group in groups.values()),
                       "Insufficient positive or ordinary sentence coverage in fitting groups")
        items = [(vector, label, 1 / (2 * len(groups) * len(items)))
                 for group in groups.values() for label, items in group.items() for vector in items]
        raw = np.array([vector for vector, _, _ in items], dtype=np.float64)
        y = np.array([label for _, label, _ in items], dtype=np.float64)
        weights = np.array([weight for _, _, weight in items], dtype=np.float64)
        means = weights @ raw
        scales = np.maximum(.05, np.sqrt(weights @ (raw - means)**2))
        design = np.column_stack((np.ones(len(raw)), (raw - means) / scales))
        coefs = np.zeros(design.shape[1], dtype=np.float64)
        for _ in range(180):
            probability = 1 / (1 + np.exp(-np.clip(design @ coefs, -40., 40.)))
            regularization = .10 * coefs
            regularization[0] = 0
            coefs -= .25 * (design.T @ (weights * (probability - y)) + regularization)
        heads[name] = {"means": means.tolist(), "scales": scales.tolist(), "coefs": coefs.tolist(),
                       "labels": dict(Counter(str(int(value)) for value in y)), "groups": len(groups)}
    return heads


def best(items):
    eligible = [(value, -i) for i, value in enumerate(items) if value is not None]
    return max(eligible, default=None)


def calibration(rows, values, heads):
    maxima = {name: {} for name in MODELS}
    for row in rows:
        if row["cohort"] != "calibration":
            continue
        for name, items in summarize_scores(row, values, heads).items():
            if (entry := best(items)) is not None:
                by_group = maxima[name]
                group = row["group"]
                by_group[group] = max(by_group.get(group, entry[0]), entry[0])
    source.require(all(len(entries) >= 18 for entries in maxima.values()), "Insufficient genuine calibration groups")
    return maxima, {name: sorted(entries.values())[-2] for name, entries in maxima.items()}


def fit(run):
    _, rows = locked(run)
    data = source.read(run / "development.json")
    binding = source.digest(run / "freeze.json")
    source.require(data["freeze_sha256"] == binding, "Development measurement differs")
    heads = fit_heads(rows, data["rows"])
    maxima, points = calibration(rows, data["rows"], heads)
    source.seal(run / "points.json", {"freeze_sha256": binding, "heads": heads, "maxima": maxima, "points": points})
    print("sealed points", points, flush=True)


def baseline(run, snapshot):
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier
    from scripts.writing_word_edit_transfer import model_assets

    lock, rows = locked(run)
    check_offline(lock)
    model_assets(snapshot)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    fresh = [row for row in rows if row["cohort"] == "report"]
    texts = sorted({text for row in fresh for text in row["context"]}, key=source.sha)
    clf = FourClassTextClassifier.from_snapshot(snapshot, "cpu")
    scores = clf.predict(texts, batch=6, check=lambda _: None)
    by_text = dict(zip(texts, scores, strict=True))
    prior = {row["id"]: row for row in source.read(PRIOR)["rows"]}
    result = {row["id"]: [by_text[text] for text in row["context"]] if row["cohort"] == "report" else prior[row["id"]]["primary"]
              for row in rows if row["cohort"] not in {"fit", "calibration"}}
    source.seal(run / "primary.json", {"freeze_sha256": source.digest(run / "freeze.json"), "rows": result,
                "unique_new_contexts": len(texts)})
    print("primary contexts", len(texts), flush=True)


def evaluate(rows, readings, points, primary):
    tally = defaultdict(lambda: defaultdict(Counter))
    added = {name: set() for name in MODELS}
    alarms = {name: defaultdict(set) for name in MODELS}
    for row in rows:
        key = f"{row['cohort']}/{row['family']}"
        spans = row["spans"]
        p = primary[row["id"]]
        source.require(len(p) == len(spans) and all(math.isfinite(x) and 0 <= x <= 1 for x in p), "Invalid primary measurements")
        choices = {"primary": [[spans[max(0, i - 1)][0], spans[min(len(spans) - 1, i + 1)][1]]
                               for i, score in enumerate(p) if score >= .50][:6]}
        coverage = {"primary": True}
        ranked = {}
        for name, values in summarize_scores(row, readings, points["heads"]).items():
            winner = best(values)
            coverage[name] = winner is not None
            ranked[name] = None if winner is None else spans[-winner[1]]
            choices[name] = [ranked[name]] if winner is not None and winner[0] > points["points"][name] else []
            choices[f"plus_{name}"] = choices["primary"] + choices[name]
            coverage[f"plus_{name}"] = True
        truth, spaces = row["truth"], set(row["spaces"])
        found = {name: any(short.useful(span, truth, spaces) for span in spans) for name, spans in choices.items()}
        for name, spans in choices.items():
            marked = set().union(*(range(*span) for span in spans)) - spaces
            tally[key][name].update(files=1, assessed=coverage[name], useful=found[name], flags=bool(spans),
                wrong=bool(spans) and truth is not None and not found[name], ordinary_characters=len(marked) if not truth else 0,
                useful_before_point=ranked.get(name) is not None and short.useful(ranked[name], truth, spaces))
        for name in MODELS:
            if found[name] and not found["primary"] and row["cohort"] == "report":
                added[name].add((row["group"], row["family"]))
            if not truth and choices[name] and not choices["primary"]:
                alarms[name][row["cohort"]].add(row["group"])
    ordered = added["ordered"]
    passed = (len(ordered) >= 4 and len(ordered) >= max(len(added["static"]), len(added["permuted"])) + 2
              and {family for _, family in ordered} == set(short.FAMILIES)
              and all(len(groups) <= 1 for groups in alarms["ordered"].values())
              and sum(tally[f"report/{family}"]["ordered"]["assessed"] for family in short.FAMILIES) >= 36)
    return {"counts": tally, "added_report_locations": {name: sorted([list(pair) for pair in items]) for name, items in added.items()},
            "new_ordinary_groups": {name: {key: sorted(groups) for key, groups in cells.items()} for name, cells in alarms.items()},
            "passes": passed}


def portable_rows(rows):
    return [{key: row[key] for key in ("id", "group", "cohort", "family", "length", "spans", "spaces", "truth", "text_sha256")}
            for row in rows]


def review(receipt, native_rows=None):
    source.require(receipt["schema"] == SCHEMA and receipt["freeze"]["method"] == METHOD
                   and receipt["freeze"]["previous_receipt_sha256"] == PRIOR_SHA
                   and receipt["freeze"]["previous_roster_sha256"] == ROSTER_SHA, "Published method differs")
    rows = receipt["rows"]
    source.require(Counter(row["cohort"] for row in rows) == COUNTS and len({row["id"] for row in rows}) == len(rows),
                   "Incomplete portable roster")
    if native_rows is not None:
        source.require(rows == portable_rows(native_rows), "Native source commitments differ")
    for row in rows:
        held = row["cohort"] not in {"fit", "calibration"}
        table = receipt["held"] if held else receipt["development"]
        spans = row["spans"]
        source.require(row["id"] == case_id(row)
                       and spans == sorted(spans) and all(0 <= a < b <= row["length"] for a, b in spans)
                       and (row["truth"] is None or row["truth"] in spans)
                       and len(table[row["id"]]) == len(spans), "Invalid published source geometry")
        for reading in table[row["id"]]:
            source.require(reading is None or (MIN_TOKENS <= reading["tokens"] <= MAX_TOKENS
                           and len(reading["token_indices_sha256"]) == 64 and len(reading["features"]) == 10
                           and all(math.isfinite(x) for x in reading["features"])), "Invalid published feature measurements")
    points = receipt["points"]
    maxima, operating = calibration(rows, receipt["development"], points["heads"])
    source.require(maxima == points["maxima"] and operating == points["points"], "Operating points differ")
    if native_rows is not None:
        heads = fit_heads(rows, receipt["development"])
        source.require(heads == points["heads"], "Development refit differs")
    scored = [row for row in rows if row["cohort"] not in {"fit", "calibration"}]
    actual = evaluate(scored, receipt["held"], points, receipt["primary"])
    source.require(actual == receipt["summary"], "Recomputed outcomes differ")
    return actual


def publish(run, output):
    lock, rows = locked(run)
    development, held, points, primary = (source.read(run / f"{name}.json")
                                          for name in ("development", "held", "points", "primary"))
    source.require(all(data["freeze_sha256"] == source.digest(run / "freeze.json") for data in (development, held, points, primary))
                   and held["points_sha256"] == source.digest(run / "points.json"), "Evidence came from different runs")
    receipt = {"schema": SCHEMA, "freeze": lock, "rows": portable_rows(rows), "points": points,
               "development": development["rows"], "held": held["rows"], "primary": primary["rows"],
               "primary_unique_new_contexts": primary["unique_new_contexts"],
               "summary": evaluate([row for row in rows if row["cohort"] not in {"fit", "calibration"}],
                                   held["rows"], points, primary["rows"])}
    review(receipt, rows)
    source.seal(output, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "fit", "held", "baseline", "publish", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--prior-roster", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.path, args.prior_roster, args.cache)
    elif args.phase in {"development", "held"}:
        causal(args.path, args.snapshot, args.phase)
    elif args.phase == "fit":
        fit(args.path)
    elif args.phase == "baseline":
        baseline(args.path, args.snapshot)
    elif args.phase == "publish":
        publish(args.path, args.output)
    else:
        native = intake(args.prior_roster, args.cache)[0] if args.prior_roster and args.cache else None
        print(source.serialize(review(source.read(args.path), native)), end="")


if __name__ == "__main__":
    main()
