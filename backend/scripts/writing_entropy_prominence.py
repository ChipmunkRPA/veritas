# SPDX-License-Identifier: Apache-2.0
"""Compare next-token entropy peaks with the app's writing interval check."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
from collections import defaultdict
from pathlib import Path

from app.bench import writing_entropy_prominence as bench
from scripts import writing_multi_passage_panel as sources

SCHEMA = "chipmunk.writing-entropy-prominence/1"
MAX_TOKENS = 2048
RUNTIME = {"mlx": "0.32.4", "mlx-lm": "0.31.3", "transformers": "5.15.1"}
OPERATING_POINTS = {"strict": .95, "wider": .90}
METHOD = {
    "context": "One prediction pass on the exact extracted document with a leading end-of-text token. "
               "Use each boundary-delimited native span of one to three sentences, with 24–192 tokens. "
               "Report coverage when the exact tokenizer positions cannot be verified or the document exceeds 2048 tokens.",
    "prominence": "Measure the sum of strict entropy peak heights over their local valleys, divided by total entropy "
                  "within the same source interval. Compare independently with the document's nearby sentences and "
                  "with a second reading of observed-token predictability.",
    "selection": "Choose a scoring method and operating point on the retained study before scoring the fresh sources. "
                 "Each method adds at most one distinct source interval to the app's existing passage ranges.",
    "utility": "Two per additional correctly located edit, minus three per additional ordinary file flagged and one "
               "per additional misplaced candidate. Break ties with fewer additional ordinary flags and stricter point.",
}


def causal_reader(snapshot):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from transformers import AutoTokenizer

    from scripts.writing_token_curvature import target_token_indices, verified_model

    if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("Use the pinned model in offline mode")
    if {name: importlib.metadata.version(name) for name in RUNTIME} != RUNTIME:
        raise ValueError("Unexpected causal inference runtime")
    verified_model(snapshot)
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(96 * 1024 * 1024)
    model, _ = load(str(snapshot), tokenizer_config={"trust_remote_code": False})
    nn.quantize(model, mode="affine", group_size=64, bits=4)
    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    head = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head

    def measure(text, sentences):
        encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        ids = [tokenizer.eos_token_id, *encoded["input_ids"]]
        offsets = [(0, 0), *encoded["offset_mapping"]]
        if len(ids) > MAX_TOKENS + 1:
            return [], len(ids) - 1, "length"
        if len(ids) != len(offsets) or any(not 0 <= a <= b <= len(text) for a, b in offsets):
            raise ValueError("Causal token positions differ from the extracted document")
        located = []
        for a, b, *_ in sentences:
            try:
                tokens = target_token_indices(text, offsets, (a, b))
            except ValueError:
                located.append(None)
            else:
                located.append((tokens[0], tokens[-1] + 1))
        bounds = {}
        for first, beginning in enumerate(located):
            if not beginning:
                continue
            for width in bench.WIDTHS:
                stop = first + width
                if stop > len(located) or any(item is None for item in located[first:stop]):
                    break
                a, b = beginning[0], located[stop - 1][1]
                if bench.TOKEN_BOUNDS[0] <= b - a < bench.TOKEN_BOUNDS[-1]:
                    bounds[first, stop] = (a, b)
        if not bounds:
            return [], len(ids) - 1, "positions_or_length"
        selected = sorted({i for a, b in bounds.values() for i in range(a, b)})
        hidden = model.model(mx.array([ids]))[0]
        mx.eval(hidden)
        moments = {}
        for offset in range(0, len(selected), 24):
            batch = selected[offset:offset + 24]
            logits = head(hidden[mx.array([i - 1 for i in batch])]).astype(mx.float32)
            shifted = logits - mx.max(logits, axis=-1, keepdims=True)
            logp = shifted - mx.logsumexp(shifted, axis=-1, keepdims=True)
            prob = mx.exp(logp)
            mean = mx.sum(prob * logp, axis=-1)
            variance = mx.sum(prob * mx.square(logp - mean[:, None]), axis=-1)
            observed = logp[mx.arange(len(batch)), mx.array([ids[i] for i in batch])]
            trace = mx.stack([-mean, observed - mean, variance], axis=-1).tolist()
            if any(not all(math.isfinite(value) for value in row) or row[0] < 0 or row[2] < 0 for row in trace):
                raise ValueError("Nonfinite moments in the conditional vocabulary distribution")
            moments.update((i, [round(value * bench.UNITS) for value in row])
                           for i, row in zip(batch, trace, strict=True))
        result = []
        for (first, stop), (a, b) in bounds.items():
            channels = zip(*(moments[i] for i in range(a, b)), strict=True)
            if reading := bench.reduce_window(*channels):
                result.append([first, stop, *reading])
        return result, len(ids) - 1, "assessed" if result else "positions_or_length"

    return measure


def baselines(files):
    result = {}
    for file in files:
        for work in sources.read(file)["works"]:
            for version in work["variants"]:
                if version["layout"] == "published":
                    result[work["document"], version["version"]] = version
    return result


def existing(variant, reading):
    from app.evidence.document_passage_intervals import _reviewed_choices

    if reading["native_text_sha256"] != variant["native_text_sha256"] or reading["sentences"] != variant["sentences"]:
        raise ValueError("Existing detector readings do not belong to these source positions")
    views = {name: [round(value, 6) if value is not None else None for value in reading[name]]
             for name in ("isolated", "context")}
    if any(value is None for view in views.values() for value in view):
        return [], False
    geometry = [[row[0], row[1], row[-1]] for row in variant["sentences"]]
    choices, _ = _reviewed_choices(geometry, views, document_baseline=True, report_review=True)
    return [[first, stop] for first, stop, *_ in choices], True


def capture(plan_file, snapshot, prior, output, lock=None):
    plan = sources.read(plan_file)
    if plan["cohort"] == "fresh":
        frozen = sources.read(lock) if lock else {}
        if frozen.get("held_plan_sha256") != sources.digest(plan_file.read_bytes()) or frozen.get("implementation") != implementation():
            raise ValueError("Freeze the scoring code, choice and final input roster before held inference")
    prepared = baselines(prior)
    measure = causal_reader(snapshot)
    cases = []
    for index, work in enumerate(plan["works"], start=1):
        for variant in work["variants"]:
            role = variant["version"]
            baseline, covered = existing(variant, prepared[work["document"], role])
            text = variant["text"]
            if sources.digest(text.encode()) != variant["native_text_sha256"]:
                raise ValueError("Source excerpt differs from the verified native plan")
            values, count, status = measure(text, variant["sentences"])
            cases.append({"group": work["group_sha256"], "document": work["document"], "genre": work["genre"],
                          "split": work["split"], "role": role, "native_sha256": variant["native_text_sha256"],
                          "generator": work.get("generator_backend"), "token_count": count, "status": status,
                          "evaluation_panel": work.get("evaluation_panel", "older"),
                          "sentences": variant["sentences"], "truth": [row["compact"] for row in variant["truth"]],
                          "incumbent_assessed": covered, "existing": baseline, "measurements": values})
        if index % 10 == 0 or index == len(plan["works"]):
            print(plan["cohort"], index, len(plan["works"]), flush=True)
    sources.save(output, {"schema": SCHEMA, "cohort": plan["cohort"], "plan_sha256": sources.digest(plan_file.read_bytes()),
                          "lock_sha256": sources.digest(lock.read_bytes()) if lock else None, "cases": cases})


def reading_summary(cases, scales, points):
    results = defaultdict(lambda: defaultdict(int))
    for case in cases:
        old_locations = [bench.compact_range(case["sentences"], row) for row in case["existing"]]
        truth = case["truth"]
        old = bench.matched(old_locations, truth)
        options = bench.selections(case, scales, points)
        for arm, (option, eligible) in options.items():
            for genre in (case["genre"], "all"):
                out = results[arm, genre]
                out["assessed"] += eligible
                out["files"] += 1
                if case["role"] == "ordinary":
                    out["ordinary"] += 1
                    out["existing_flags"] += bool(old_locations)
                    out["extra_flags"] += option is not None and not old_locations
                else:
                    out["truth"] += len(truth)
                    out["existing_hits"] += len(old)
                    if option is not None:
                        match = bench.matched([*old_locations, bench.compact_range(case["sentences"], option)], truth)
                        out["extra_hits"] += len(match) - len(old)
                        out["unmatched_additions"] += len(match) == len(old)
    return {arm: {genre: dict(value) for (name, genre), value in results.items() if name == arm} for arm in bench.ARMS}


def implementation():
    return {"capture_and_replay": sources.digest(Path(__file__).read_bytes()),
            "selector": sources.digest(Path(bench.__file__).read_bytes())}


def freeze(older_file, held_plan, output):
    older = sources.read(older_file)["cases"]
    calibration = [case for case in older if case["split"] == "validation" and case["role"] == "ordinary"]
    selection = [case for case in older if case["split"] == "test"]
    scales = bench.fit_scales(calibration)
    points = {key: bench.fit_points(calibration, scales, fraction=fraction) for key, fraction in OPERATING_POINTS.items()}
    evidence = {key: reading_summary(selection, scales, gate) for key, gate in points.items()}

    def rank(option):
        point, arm = option
        summary = evidence[point][arm]["all"]
        flags, hits, missed = (summary.get(key, 0) for key in ("extra_flags", "extra_hits", "unmatched_additions"))
        return (2 * hits - 3 * flags - missed, -flags, point == "strict", arm)

    chosen = max(((label, arm) for label in points for arm in bench.ARMS), key=rank)
    print("frozen", chosen, rank(chosen)[:3], flush=True)
    sources.save(output, {"schema": SCHEMA, "method": METHOD, "selection": list(chosen), "summary": evidence,
                          "scales": scales, "points": points, "older_sha256": sources.digest(older_file.read_bytes()),
                          "implementation": implementation(), "runtime": RUNTIME,
                          "held_plan_sha256": sources.digest(held_plan.read_bytes())})


def confirm(held_file, lock_file, output):
    held, lock = (sources.read(path) for path in (held_file, lock_file))
    if held["lock_sha256"] != sources.digest(lock_file.read_bytes()) or held["plan_sha256"] != lock["held_plan_sha256"]:
        raise ValueError("Held inference does not bind to the frozen rule and panel")
    comparisons = {key: reading_summary(held["cases"], lock["scales"], gate) for key, gate in lock["points"].items()}
    choice, arm = lock["selection"]
    print(comparisons[choice][arm], flush=True)
    sources.save(output, {"schema": SCHEMA, "frozen_sha256": held["lock_sha256"], "selection": lock["selection"],
                          "held_sha256": sources.digest(held_file.read_bytes()), "outcomes": comparisons})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    measuring = actions.add_parser("capture")
    fitting = actions.add_parser("freeze")
    checking = actions.add_parser("confirm")
    for stage, method, paths in ((measuring, capture, ("plan_file", "snapshot", "output")),
                                 (fitting, freeze, ("older_file", "held_plan", "output")),
                                 (checking, confirm, ("held_file", "lock_file", "output"))):
        stage.set_defaults(method=method)
        for path in paths:
            stage.add_argument(path, type=Path)
    measuring.add_argument("--prior", nargs="+", type=Path, required=True)
    measuring.add_argument("--lock", type=Path)
    args = vars(parser.parse_args())
    args.pop("command")
    args.pop("method")(**args)
