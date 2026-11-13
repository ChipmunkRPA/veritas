"""Evaluate a temperature-response cue on native sentences in short mixed writing."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from time import monotonic

from scripts import writing_interval_transfer as interval
from scripts import writing_revision_trajectory_transfer as source

PANELS = interval.PANELS
SCHEMA = "chipmunk.writing-temperature-transfer/1"
EARLIER = ("writing-revision-trajectory", "writing-interval-transfer", "writing-gao-archive-transfer")
ARM_NAMES = ("log_probability", "temperature")
LOW, HIGH = 0.7, 1.4
MIN_TOKENS, MAX_TOKENS = 12, 512
FAMILIES = tuple(interval.FAMILIES)
GROUPS = 24
METAL_CACHE_LIMIT = 96 * 1024 * 1024
PROTOCOL = {
    "method": "Absolute mean target-token log probability difference at temperatures .7 and 1.4, minus "
    "mean expected difference under unscaled logits, divided by mean conditional standard deviation.",
    "native": "One pass over the unchanged decoded short file. Exact app-native sentence token coverage and "
    "at least twelve predicted tokens. No leading unpredicted token or text truncation.",
    "source": "First twenty-four eligible work IDs in a fixed hash order, with distinct publisher originals "
    "and common Gemini and Nano v1 polish on the same exact original sentence. Keep exact flanks; "
    "prefer one extra unchanged sentence on a hash-assigned side when supported.",
    "development": "Forty-two dated historical genuine excerpts across twenty-one work groups. "
    "Strictly exceed the second-highest assessed group maximum separately for each novel score.",
    "comparison": "First six native primary three-sentence contexts at response .50; also compare "
    "same-base mean log probability with its own genuine-work point.",
    "hit": "One location covers at least half the known edit and reaches .30 non-whitespace intersection over union.",
    "advance": "At least four additional strict locations across both generators, at least two more than "
    "mean log probability adds, no more than one newly flagged genuine work in each held cohort, "
    "and at least 90% candidate coverage of edited files. This research produces no authorship confidence.",
}


def known_sources():
    return [PANELS / f"{name}-2026-11-13.json" for name in EARLIER]


def native(text):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    doc = from_string(text)
    source.require(doc.text == text, "Native extraction changed the submitted text")
    return segment(doc)


def record(text, cohort, group, family, truth=None, **meta):
    from app.detectors.text.pipeline import windows

    units = native(text)
    source.require(3 <= len(units) <= 12, "Expected a short native sentence sequence")
    spans = [[s.start, s.end] for s in units]
    source.require(truth is None or truth in spans, "Native geometry changed the edit boundary")
    return {
        "id": source.sha(f"{cohort}/{group}/{family}"), "cohort": cohort, "group": group, "family": family,
        "text": text, "text_sha256": source.sha(text), "truth": truth, "spans": spans,
        "spaces": [i for i, c in enumerate(text) if c.isspace()], "length": len(text),
        "context": windows(units, 3), **meta,
    }


def annotated(row):
    text, gold = source.publisher_case(row)
    return text, gold, native(text)


def align(edited, base, base_sentences, sides):
    text, gold, units = edited
    left, right = sides
    candidates = {}
    truth = set(map(tuple, gold))
    for index in range(left, len(units) - right):
        target = units[index]
        if (target.start, target.end) not in truth or not 14 <= target.n_words <= 72:
            continue
        excerpt = units[index - left:index + right + 1]
        if excerpt[0].paragraph != excerpt[-1].paragraph:
            continue
        if any(a < excerpt[-1].end and b > excerpt[0].start and (a, b) != (target.start, target.end) for a, b in truth):
            continue
        start, end = excerpt[0].start, excerpt[-1].end
        signature = (text[start:target.start], text[target.end:end])
        hits = []
        for j in range(left, len(base_sentences) - right):
            first, center, last = base_sentences[j - left], base_sentences[j], base_sentences[j + right]
            if (first.paragraph == last.paragraph and center.text != target.text
                    and (base[first.start:center.start], base[center.end:last.end]) == signature):
                hits.append(j)
        if len(hits) != 1:
            continue
        j = hits[0]
        if j in candidates:
            candidates[j] = None
        else:
            candidates[j] = (text[start:end], [target.start - start, target.end - start], [start, end])
    return {key: value for key, value in candidates.items() if value is not None}


def select(cache):
    records = {f: interval.read_shard(cache, f, pin)
               for f, pin in (interval.FAMILIES | interval.EXCLUDED_SHARD).items()}
    gem, nano, qwen = records.values()
    previous = {r["group"] for ledger in known_sources() for r in source.read(ledger)["rows"]}
    allowed = (gem.keys() & nano.keys()) - qwen.keys()
    historical_phrases = set()
    for work in (gem.keys() | nano.keys() | qwen.keys()):
        if work in qwen or source.sha(work) in previous:
            for table in records.values():
                historical_phrases.update(*(interval.fingerprints(r["text"]) for r in table.get(work, {}).values()))
    order = sorted((i for i in allowed if source.sha(i) not in previous),
                   key=lambda value: source.sha(f"chipmunk/short-temperature/2026-11-13/{value}"))
    rows, intake = [], Counter(eligible_source_ids=len(order))
    for work in order:
        if intake["admitted"] == GROUPS:
            break
        intake["examined"] += 1
        group = source.sha(work)
        if gem[work]["v0"]["text"] != nano[work]["v0"]["text"]:
            intake["different_original"] += 1
            continue
        if any(records[family][work]["v1"]["edit_operation"] != "polish" for family in FAMILIES):
            raise ValueError("Unexpected publisher edit operation")
        try:
            base, gold, original = annotated(gem[work]["v0"])
            source.require(not gold, "Original has publisher edit annotations")
            variants = {f: annotated(records[f][work]["v1"]) for f in FAMILIES}
        except ValueError:
            intake["unsupported_publisher_geometry"] += 1
            continue
        if historical_phrases & interval.fingerprints(base):
            intake["previous_or_duplicate_source_phrase"] += 1
            continue
        preferred = (2, 3) if int(group[0], 16) % 2 else (3, 2)
        chosen = None
        for left, right in (preferred, (2, 2)):
            matching = {f: align(v, base, original, (left, right)) for f, v in variants.items()}
            common = set.intersection(*(set(m) for m in matching.values()))
            if common:
                j = min(common, key=lambda i: source.sha(f"{group}/{i}"))
                chosen = left, right, j, matching
                break
        if chosen is None:
            intake["no_common_isolated_edit"] += 1
            continue
        left, right, j, matching = chosen
        a, b = original[j - left].start, original[j + right].end
        options = [("original", base[a:b], None, [a, b], gem[work]["v0"])]
        options.extend((f, *matching[f][j], records[f][work]["v1"]) for f in FAMILIES)
        candidate = []
        try:
            for family, text, truth, bounds, row in options:
                current = record(text, "report", group, family, truth, source=source.normalize_title_source(work),
                                 publisher_record=row["record_id"], source_range=bounds, target_index=left)
                source.require(len(current["spans"]) == left + right + 1
                               and (truth is None or current["spans"][left] == truth),
                               "Clipping changed native source geometry")
                candidate.append(current)
        except ValueError:
            intake["clipping_changed_geometry"] += 1
            continue
        rows.extend(candidate)
        historical_phrases.update(interval.fingerprints(base))
        intake["admitted"] += 1
        intake[f"sentences_{left + right + 1}"] += 1
    source.require(intake["admitted"] == GROUPS, "Incomplete fresh held roster")
    return rows, dict(intake)


def roster(run, cache, historical):
    from scripts.writing_entropy_conditioned_transfer import RUNTIME
    from scripts.writing_focus_v3 import BASELINE_ASSETS
    from scripts.writing_token_curvature import MODEL_ASSETS, REVISION

    source.require(all(source.digest(cache / name) == pin for name, pin in source.METADATA.items()),
                   "Publisher metadata differs")
    cropped, intake = select(cache)
    rows = []
    for row in source.read(historical):
        if row["cohort"] == "selection":
            rows.append(record(row["text"], "development", row["group"], row["family"]))
        elif row["cohort"] in {"forum", "uswds"} and row["family"] in {"before", "after"}:
            rows.append(record(row["text"], row["cohort"], row["group"], row["family"]))
    source.require(Counter(row["cohort"] for row in rows) == {"development": 42, "forum": 16, "uswds": 12},
                   "Historical genuine cohorts differ")
    rows.extend(cropped)
    lock = {"schema": SCHEMA, "protocol": PROTOCOL, "script_sha256": source.digest(Path(__file__)),
            "historical_sha256": source.digest(historical), "publisher_revision": source.REVISION,
            "publisher_metadata": source.METADATA, "publisher_shards": interval.FAMILIES | interval.EXCLUDED_SHARD,
            "older_sha256": {p.name: source.digest(p) for p in known_sources()}, "runtime": RUNTIME,
            "causal_assets": MODEL_ASSETS, "causal_revision": REVISION, "primary_assets": BASELINE_ASSETS,
            "temperatures": [LOW, HIGH], "minimum_tokens": MIN_TOKENS, "maximum_tokens": MAX_TOKENS,
            "intake": intake, "roster": [{k: v for k, v in r.items() if k not in {"text", "context"}} for r in rows]}
    source.seal(run / "roster.json", rows)
    lock["roster_sha256"] = source.digest(run / "roster.json")
    source.seal(run / "freeze.json", lock)


def read_locked(run):
    lock = source.read(run / "freeze.json")
    source.require(lock["script_sha256"] == source.digest(Path(__file__))
                   and lock["roster_sha256"] == source.digest(run / "roster.json"), "Frozen inputs differ")
    return lock, source.read(run / "roster.json")


def moments(mx, logits, labels):
    shifted = logits - mx.max(logits, axis=-1, keepdims=True)
    log_normalizers = [mx.logsumexp(shifted / temp, axis=-1, keepdims=True) for temp in (LOW, HIGH, 1.0)]
    low, high, default = [shifted / temp - z for temp, z in zip((LOW, HIGH, 1.0), log_normalizers, strict=True)]
    observed = default[mx.arange(len(labels)), mx.array(labels)]
    delta = low - high
    weights = mx.exp(default)
    expected = mx.sum(weights * delta, axis=-1)
    sd = mx.sqrt(mx.sum(weights * mx.square(delta - expected[:, None]), axis=-1))
    return mx.stack([observed, delta[mx.arange(len(labels)), mx.array(labels)], expected, sd], axis=-1).tolist()


def score(run, snapshot, cohort):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from transformers import AutoTokenizer

    from scripts.writing_token_curvature import target_token_indices, verified_model

    lock, rows = read_locked(run)
    source.require(all(os.getenv(k) == "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
                   and all(importlib.metadata.version(k) == v for k, v in lock["runtime"].items()),
                   "Use the locked offline runtime")
    verified_model(snapshot)
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(METAL_CACHE_LIMIT)
    model, _ = load(str(snapshot), tokenizer_config={"trust_remote_code": False})
    nn.quantize(model, mode="affine", group_size=64, bits=4)
    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    head = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    outputs = {}
    t0 = monotonic()
    for row in rows:
        if (row["cohort"] == "development") != (cohort == "fit"):
            continue
        text = row["text"]
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens, offsets = encoded["input_ids"], encoded["offset_mapping"]
        focus = []
        for span in row["spans"]:
            try:
                indices = target_token_indices(text, offsets, span)
            except ValueError:
                indices = []
            focus.append(indices if len(indices) >= MIN_TOKENS and len(tokens) <= MAX_TOKENS else [])
        selected = sorted({i for indices in focus for i in indices})
        measured = {}
        if selected:
            hidden = model.model(mx.array([tokens]))[0]
            mx.eval(hidden)
            for start in range(0, len(selected), 16):
                batch = selected[start:start + 16]
                logits = head(hidden[mx.array([i - 1 for i in batch])]).astype(mx.float32)
                measured.update(zip(batch, moments(mx, logits, [tokens[i] for i in batch]), strict=True))
        values = []
        for indices in focus:
            if not indices:
                values.append(None)
                continue
            means = [fmean(items) for items in zip(*(measured[i] for i in indices), strict=True)]
            source.require(all(map(math.isfinite, means)) and means[3] > 0, "Nonfinite or degenerate token moments")
            values.append([len(indices), *means])
        outputs[row["id"]] = values
        if len(outputs) % 10 == 0:
            print(cohort, len(outputs), round(monotonic() - t0, 1), "s", flush=True)
    source.seal(run / f"{cohort}.json", {"freeze_sha256": source.digest(run / "freeze.json"), "rows": outputs})


def primary(run, snapshot):
    import torch

    from scripts.score_text_upgrade import sequence_predictor

    lock, rows = read_locked(run)
    source.require(all(source.digest(snapshot / k) == v for k, v in lock["primary_assets"].items())
                   and all(os.getenv(k) == "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")),
                   "Use the locked offline primary snapshot")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    texts = sorted({text for row in rows if row["cohort"] != "development" for text in row["context"]}, key=source.sha)
    predict = sequence_predictor("tropa", snapshot, "cpu", 4, max_len=512)
    responses = {}
    for start in range(0, len(texts), 80):
        batch = texts[start:start + 80]
        responses.update(zip(batch, predict(batch), strict=True))
        print("primary", len(responses), len(texts), flush=True)
    source.seal(run / "primary.json", {"freeze_sha256": source.digest(run / "freeze.json"), "inputs": len(texts),
                "runtime": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "pysbd")},
                "rows": {r["id"]: [[source.sha(t), responses[t]] for t in r["context"]] for r in rows
                         if r["cohort"] != "development"}})


def responses(values):
    out = {arm: [] for arm in ARM_NAMES}
    for i, row in enumerate(values):
        if row is None:
            continue
        count, logp, observed, expected, sd = row
        source.require(type(count) is int and count >= MIN_TOKENS and sd > 0 and all(map(math.isfinite, row)),
                       "Malformed causal sentence values")
        out["log_probability"].append((logp, i))
        out["temperature"].append(((abs(observed) - expected) / sd, i))
    return {arm: max(items, default=None) for arm, items in out.items()}


def decision_points(rows, values):
    maxima = {arm: defaultdict(list) for arm in ARM_NAMES}
    for row in rows:
        for arm, response in responses(values[row["id"]]).items():
            if response is not None:
                maxima[arm][row["group"]].append(response[0])
    by_arm = {arm: {g: max(v) for g, v in groups.items()} for arm, groups in maxima.items()}
    source.require(all(len(groups) >= 18 for groups in by_arm.values()), "Too few assessed genuine fit groups")
    return {"maxima": by_arm, "points": {arm: sorted(groups.values())[-2] for arm, groups in by_arm.items()}}


def fit(run):
    lock, rows = read_locked(run)
    measured = source.read(run / "fit.json")
    source.require(measured["freeze_sha256"] == source.digest(run / "freeze.json"), "Fit readings changed")
    result = decision_points([row for row in rows if row["cohort"] == "development"], measured["rows"])
    source.seal(run / "points.json", {"freeze_sha256": source.digest(run / "freeze.json"),
                                     "fit_sha256": source.digest(run / "fit.json"), **result})


def useful(span, truth, spaces):
    if not truth:
        return False
    gold = set(range(*truth)) - spaces
    guess = set(range(*span)) - spaces
    hit = len(gold & guess)
    return 2 * hit >= len(gold) and 10 * hit >= 3 * len(gold | guess)


def evaluate(rows, held, primary_values, points):
    totals = defaultdict(lambda: defaultdict(Counter))
    additional = {arm: set() for arm in ARM_NAMES}
    extra_genuine = defaultdict(set)
    cases = []
    for row in rows:
        geometry = row["spans"]
        current = row["id"]
        source.require(len(geometry) == len(held[current]) == len(primary_values[current]), "Incomplete native positions")
        b = [[geometry[max(0, i - 1)][0], geometry[min(len(geometry) - 1, i + 1)][1]]
             for i, (_, p) in enumerate(primary_values[current]) if p >= .50][:6]
        options = {"primary": b}
        rank = responses(held[current])
        for arm, item in rank.items():
            assessed = item is not None
            options[arm] = [geometry[item[1]]] if assessed and item[0] > points[arm] else []
            options[f"primary_plus_{arm}"] = b + options[arm]
        spaces = set(row["spaces"])
        found = {arm: any(useful(span, row["truth"], spaces) for span in chosen) for arm, chosen in options.items()}
        label = f"{row['cohort']}/{row['family']}"
        for arm, chosen in options.items():
            counts = totals[label][arm]
            counts["files"] += 1
            counts["assessed"] += arm not in ARM_NAMES or rank[arm] is not None
            counts["flagged"] += bool(chosen)
            counts["useful"] += found[arm]
        for arm in ARM_NAMES:
            if row["truth"] and found[arm] and not found["primary"]:
                additional[arm].add((row["family"], row["group"]))
            if not row["truth"] and options[arm] and not b:
                extra_genuine[arm, row["cohort"]].add(row["group"])
        cases.append({"id": current, "locations": options, "useful": found})
    stats = {label: {arm: dict(count) for arm, count in methods.items()} for label, methods in totals.items()}
    return {"totals": stats, "new_locations": {arm: sorted([list(pair) for pair in pairs])
                                               for arm, pairs in additional.items()},
            "new_genuine_groups": {arm: {cohort: sorted(extra_genuine[arm, cohort]) for cohort in ("report", "forum", "uswds")}
                                   for arm in ARM_NAMES}, "cases": cases}


def publish(run, output):
    _, rows = read_locked(run)
    lock = source.read(run / "freeze.json")
    inputs = {phase: source.read(run / f"{phase}.json") for phase in ("fit", "held", "primary", "points")}
    require_bound = all(r["freeze_sha256"] == source.digest(run / "freeze.json") for r in inputs.values())
    source.require(require_bound and inputs["points"]["fit_sha256"] == source.digest(run / "fit.json"), "Run changed")
    held = [r for r in rows if r["cohort"] != "development"]
    result = evaluate(held, inputs["held"]["rows"], inputs["primary"]["rows"], inputs["points"]["points"])
    source.seal(output, {"schema": SCHEMA, "freeze": lock, "freeze_sha256": source.digest(run / "freeze.json"),
                         "measurements": inputs, "results": result})


def review(path):
    ledger = source.read(path)
    lock, values = ledger["freeze"], ledger["measurements"]
    source.require(ledger["schema"] == lock["schema"] == SCHEMA and lock["protocol"] == PROTOCOL
                   and source.sha(source.serialize(lock)) == ledger["freeze_sha256"]
                   and all(r["freeze_sha256"] == ledger["freeze_sha256"] for r in values.values())
                   and source.sha(source.serialize(values["fit"])) == values["points"]["fit_sha256"], "Run changed")
    rows = lock["roster"]
    fit_rows = [r for r in rows if r["cohort"] == "development"]
    held_rows = [r for r in rows if r["cohort"] != "development"]
    source.require(len({r["id"] for r in rows}) == len(rows)
                   and {r["id"] for r in fit_rows} == set(values["fit"]["rows"])
                   and {r["id"] for r in held_rows} == set(values["held"]["rows"]) == set(values["primary"]["rows"])
                   and not {r["group"] for r in fit_rows} & {r["group"] for r in held_rows}, "Bad review partitions")
    points = decision_points(fit_rows, values["fit"]["rows"])
    source.require(all(points[key] == values["points"][key] for key in points), "Decision points differ")
    result = evaluate(held_rows, values["held"]["rows"], values["primary"]["rows"], points["points"])
    source.require(result == ledger["results"], "Aggregate or native positions differ")
    print(source.serialize({k: v for k, v in result.items() if k != "cases"}), end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("roster", "score-fit", "fit", "score-held", "primary", "publish", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--publisher", type=Path)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "roster":
        roster(args.path, args.publisher, args.historical)
    elif args.phase in {"score-fit", "score-held"}:
        if args.phase == "score-held":
            source.require(source.read(args.path / "points.json")["freeze_sha256"] == source.digest(args.path / "freeze.json"),
                           "Freeze decision points before held scoring")
        score(args.path, args.snapshot, args.phase.removeprefix("score-"))
    elif args.phase == "fit":
        fit(args.path)
    elif args.phase == "primary":
        primary(args.path, args.snapshot)
    elif args.phase == "publish":
        publish(args.path, args.output)
    else:
        review(args.path)


if __name__ == "__main__":
    main()
