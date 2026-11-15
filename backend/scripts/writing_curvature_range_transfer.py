# SPDX-License-Identifier: MIT
"""Compare native word ranges using a causal model's observed and expected token readings."""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import math
import os
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_independent_range_transfer as parent

SCHEMA = "chipmunk.writing-curvature-range/1"
ARMS = ("absolute", "two_flanks", "agreement")
WIDTHS = (3, 5, 8, 12, 18, 24)
TOKEN_LIMIT = 512
RUNTIME = {"mlx": "0.32.4", "mlx-lm": "0.31.3", "transformers": "5.22.1"}
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
ASSETS_SHA256 = "135aad18251722edc31a55de9287e27a0c52b5ef6a180f6178684a85906d41ed"
INPUT_FIELDS = ("reports", "matched", "additional", "dated", "elife", "fresh", "gemini", "gpt")
METHOD = {
    "signal": "Read the excerpt with exact tokenizer positions. Record the sum of observed minus expected "
              "log probability and the sum of conditional variances for the completely covered words.",
    "geometry": "Compare every contiguous range of 3, 5, 8, 12, 18 or 24 words within each of three "
                "interior sentences, plus complete sentences of 3 to 36 words.",
    "comparison": "Compare standardized absolute curvature, the smaller standardized rise over the two "
                  "immediate flanks, and the smaller of those signals. The flank width is at least 3 "
                  "and at most 12 words. Equal maxima withhold the location.",
    "points": "Set each check's maximum-per-file points on separate ordinary works in the matching genre, "
              "with the same two ordinary-work budgets as the prior word-range check.",
    "scope": "Retain exact changed-word overlap, ordinary alerts, and the difference between finding a "
             "sentence and finding useful words. Scores have no authorship-confidence interpretation.",
}


def range_scores(row, profile):
    """Calculate all candidate scores from prefix sums."""
    words = row["words"]
    reading = profile["words"]
    if reading is None:
        return {}
    effects, variance, missing = [0.], [0.], [0]
    for item in reading:
        effects.append(effects[-1] + (item[0] if item else 0.))
        variance.append(variance[-1] + (item[1] if item else 0.))
        missing.append(missing[-1] + (item is None))

    def totals(a, b):
        if not 0 <= a < b <= len(words) or missing[b] > missing[a]:
            return None
        effect, spread = effects[b] - effects[a], variance[b] - variance[a]
        return (effect, spread) if spread > 0.000001 else None

    by_sentence = {}
    for idx in parent.CENTERS:
        indices = [i for i, word in enumerate(words) if word[2] == idx]
        if not indices:
            continue
        sentence_scores = defaultdict(list)
        length = len(indices)
        widths = set(WIDTHS) | ({length} if 3 <= length <= 36 else set())
        for width in sorted(widths):
            for a in range(indices[0], indices[-1] + 2 - width):
                b = a + width
                inner = totals(a, b)
                if inner is None:
                    continue
                effect, spread = inner
                absolute = effect / math.sqrt(spread)
                sentence_scores["absolute"].append((round(absolute, 7), idx, a, b))
                pad = min(12, max(3, width))
                flanks = (totals(a - pad, a), totals(b, b + pad))
                if any(side is None for side in flanks):
                    continue
                ratio = width / pad
                contrast = min((effect - ratio * other) / math.sqrt(spread + ratio * ratio * noise)
                               for other, noise in flanks)
                sentence_scores["two_flanks"].append((round(contrast, 7), idx, a, b))
                sentence_scores["agreement"].append((round(min(absolute, contrast), 7), idx, a, b))
        by_sentence[idx] = sentence_scores
    return by_sentence


def ranked_ranges(row, profile):
    def maximum(values):
        if not values:
            return None
        values.sort(reverse=True)
        if len(values) > 1 and values[0][0] == values[1][0]:
            return None
        score, idx, a, b = values[0]
        return {"score": score, "sentence": idx, "words": list(range(a, b))}

    all_scores = defaultdict(list)
    peaks = {}
    coverage = Counter()
    for idx, sentence_scores in range_scores(row, profile).items():
        for arm, values in sentence_scores.items():
            all_scores[arm].extend(values)
            coverage[arm] += len(values)
        peaks[idx] = {arm: maximum(values) for arm, values in sentence_scores.items()}
    return {arm: maximum(values) for arm, values in all_scores.items()}, peaks, coverage


def calibration_points(controls):
    out = {}
    for arm in ARMS:
        maxima = sorted((max(0., v[arm]["score"]) if v.get(arm) else 0. for v in controls), reverse=True)
        out[arm] = {label: maxima[len(maxima) // divisor] for label, divisor in parent.previous.LEVELS.items()}
    return out


def compute(source, document):
    calibration, panels, baseline_points, sources = source
    readings = document["readings"]
    inputs = {row["sha256"]: row for batch in [*calibration.values(), *(rows for _, rows in panels.values())] for row in batch}
    if (document["schema"] != SCHEMA or document["method"] != METHOD or document["sources"] != sources
            or document["model"] != REVISION or document["runtime"] != RUNTIME or set(readings) != set(inputs)):
        raise ValueError("Source roster, runtime or method differs")
    if parent.previous.pool.sha(json.dumps(document["assets"], sort_keys=True)) != ASSETS_SHA256:
        raise ValueError("Pinned causal model files differ")
    results = {}
    for digest, row in inputs.items():
        profile = readings[digest]
        values = profile["words"]
        count = profile["token_count"]
        malformed = values is not None and any(
            item is not None and (len(item) != 2 or not all(math.isfinite(v) for v in item) or item[1] <= 0.)
            for item in values
        )
        if (type(count) is not int or count < 0
                or (values is None) != (not 2 <= count <= TOKEN_LIMIT)
                or (values is not None and len(values) != len(row["words"])) or malformed):
            raise ValueError("Invalid recorded variance, positions or token coverage")
        results[digest] = ranked_ranges(row, profile)
    points, outcomes, diagnostics = {}, {}, {}
    for genre, rows in calibration.items():
        points[genre] = {"paired": baseline_points[genre]} | calibration_points(
            [results[row["sha256"]][0] for row in rows])
    if points != document["points"]:
        raise ValueError("Ordinary point selection no longer matches its frozen values")
    for panel, (genre, rows) in panels.items():
        selections = []
        partial = defaultdict(Counter)
        for row in rows:
            proposed, targets, eligible = results[row["sha256"]]
            before = row["best"]["range"]
            picked = {"paired": before} | {arm: proposed.get(arm) for arm in ARMS}
            selections.append(row | {"reports": picked})
            ordinary = row["family"] in parent.previous.ORDINARY
            target = {arm: targets.get(row["position"], {}).get(arm) for arm in ARMS}
            for arm, proposal in picked.items():
                for level, cutoff in points[genre][arm].items():
                    current_use = proposal is not None and proposal["score"] > cutoff
                    baseline_use = before is not None and before["score"] > points[genre]["paired"][level]
                    known = target.get(arm) if arm != "paired" else next(
                        (p for p in row["by_sentence"] if p["sentence"] == row["position"]), None)
                    correct = current_use and parent.previous.words.useful_words(set(proposal["words"]), set(row["changed"]))
                    both = baseline_use and parent.previous.words.useful_words(set(before["words"]), set(row["changed"]))
                    partial[f"{arm}/{level}/{row['family']}"].update(
                        assessed=bool(eligible[arm]) if arm != "paired" else row["assessed"],
                        exact_sentence=not ordinary and current_use and proposal["sentence"] == row["position"],
                        target_range=not ordinary and bool(known),
                        useful_given_target=not ordinary and bool(known and parent.previous.words.useful_words(
                            set(known["words"]), set(row["changed"]))),
                        union_useful=not ordinary and (correct or both),
                        union_ordinary=ordinary and (current_use or baseline_use),
                    )
        outcomes[panel] = parent.previous.count(selections, points[genre])
        diagnostics[panel] = {key: dict(count) for key, count in partial.items()}
    return {"points": points, "outcomes": outcomes, "diagnostics": diagnostics}


def word_tokens(words, offsets):
    """Keep words only when every character belongs to predicted pieces for that word."""
    word_ends = [item[1] for item in words]
    assigned = [[] for _ in words]
    blocked = set()
    for index, (start, stop) in enumerate(offsets):
        if start == stop:
            continue
        cursor = bisect_right(word_ends, start)
        positions = []
        while cursor < len(words) and words[cursor][0] < stop:
            positions.append(cursor)
            cursor += 1
        if index == 0 or len(positions) > 1:
            blocked.update(positions)
        elif positions:
            assigned[positions[0]].append(index)
    output = []
    for i, (a, b, _sentence) in enumerate(words):
        covered = {position for index in assigned[i] for position in
                   range(max(a, offsets[index][0]), min(b, offsets[index][1]))}
        output.append(assigned[i] if i not in blocked and len(covered) == b - a else ())
    return output


def causal_reader(snapshot):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from transformers import AutoTokenizer

    from scripts.writing_token_curvature import verified_model

    if any(os.getenv(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("Use offline model settings")
    if {name: importlib.metadata.version(name) for name in RUNTIME} != RUNTIME:
        raise ValueError("The causal scoring runtime differs")
    verified_model(snapshot)
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(256 * 1024 * 1024)
    model, _ = load(str(snapshot), tokenizer_config={"trust_remote_code": False})
    nn.quantize(model, mode="affine", group_size=64, bits=4)
    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    head = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head

    def measure(text, words):
        encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        tokens, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if len(tokens) != len(offsets) or any(not 0 <= a <= b <= len(text) for a, b in offsets):
            raise ValueError("Tokenizer returned inconsistent source positions")
        count = len(tokens)
        if not 2 <= count <= TOKEN_LIMIT:
            return {"token_count": count, "words": None}
        ids = word_tokens(words, offsets)
        selected = sorted({index for span in ids for index in span})
        result = {}
        if selected:
            hidden = model.model(mx.array([tokens]))[0]
            mx.eval(hidden)
            for start in range(0, len(selected), 16):
                batch = selected[start:start + 16]
                logits = head(hidden[mx.array([i - 1 for i in batch])]).astype(mx.float32)
                shifted = logits - mx.max(logits, axis=-1, keepdims=True)
                logp = shifted - mx.logsumexp(shifted, axis=-1, keepdims=True)
                probability = mx.exp(logp)
                expected = mx.sum(probability * logp, axis=-1)
                variance = mx.sum(probability * mx.square(logp - expected[:, None]), axis=-1)
                observed = logp[mx.arange(len(batch)), mx.array([tokens[i] for i in batch])]
                result.update(zip(batch, mx.stack([observed - expected, variance], axis=-1).tolist(), strict=True))
        output = []
        for span in ids:
            if not span:
                output.append(None)
                continue
            effect, variance = (round(math.fsum(result[i][col] for i in span), 7) for col in (0, 1))
            output.append([effect, variance] if variance > 0.000001 else None)
        return {"token_count": count, "words": output}

    return measure


def capture(args, source):
    from scripts.writing_token_curvature import MODEL_ASSETS

    calibration, panels, baseline, sources = source
    native = parent.native(args)
    controls = {r["sha256"]: r for rows in calibration.values() for r in rows}
    held = {r["sha256"]: r for _, rows in panels.values() for r in rows}
    if controls.keys() & held.keys():
        raise ValueError("A held excerpt appears in point selection")
    measure = causal_reader(args.model)
    readings, points = {}, {}
    with args.output.with_suffix(".progress.jsonl").open("x", buffering=1) as journal:
        for index, (digest, row) in enumerate((controls | held).items(), 1):
            body = native[digest]
            words = [list(item[:3]) for item in parent.previous.words.items(body)]
            if body["spans"] != row["spans"] or parent.previous.pool.sha(body["text"]) != digest or words != row["words"]:
                raise ValueError("Native excerpt or word coordinates do not match the held commitment")
            reading = measure(body["text"], words)
            readings[digest] = reading
            journal.write(json.dumps({"source": digest, "profile": reading}, sort_keys=True, allow_nan=False) + "\n")
            if index == len(controls):
                for genre, rows in calibration.items():
                    maxima = [ranked_ranges(item, readings[item["sha256"]])[0] for item in rows]
                    points[genre] = {"paired": baseline[genre]} | calibration_points(maxima)
                print(f"Set points on {len(controls)} ordinary works before held inference.", flush=True)
            if index % 40 == 0:
                print(f"Measured {index}/{len(controls) + len(held)} excerpts.", flush=True)
    return {"schema": SCHEMA, "method": METHOD, "model": REVISION, "assets": MODEL_ASSETS, "sources": sources,
            "runtime": RUNTIME, "points": points, "readings": readings}


def repeat_samples(args, source):
    document = parent.previous.read(args.receipt)
    if document["replay"] != compute(source, document):
        raise ValueError("The held receipt does not match its replay")
    calibration, panels, _, _ = source
    samples = [(f"point/{genre}", min(rows, key=lambda row: row["sha256"])) for genre, rows in calibration.items()]
    for panel, (_, rows) in panels.items():
        by_source = sorted(rows, key=lambda row: row["sha256"])
        for ordinary in (True, False):
            match = next((row for row in by_source if (row["family"] in parent.previous.ORDINARY) == ordinary), None)
            if match is not None:
                samples.append((panel, match))
    native = parent.native(args)
    measure = causal_reader(args.model)
    audited = []
    for panel, row in samples:
        digest = row["sha256"]
        source_text = native[digest]["text"]
        if parent.previous.pool.sha(source_text) != digest:
            raise ValueError("The audited source text changed")
        actual = measure(source_text, row["words"])
        if actual != document["readings"][digest]:
            raise ValueError(f"The independent causal pass differed in panel {panel}")
        audited.append({"panel": panel, "source": digest, "word_values": sum(v is not None for v in actual["words"] or [])})
    return {"receipt_sha256": parent.previous.digest(args.receipt), "runtime": RUNTIME, "audited": audited}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    measure = phases.add_parser("capture")
    measure.add_argument("output", type=Path)
    audit = phases.add_parser("audit")
    audit.add_argument("receipt", type=Path)
    audit.add_argument("output", type=Path)
    for phase in (measure, audit):
        for field in ("model", *INPUT_FIELDS):
            phase.add_argument(f"--{field}", type=Path, required=True)
    phases.add_parser("review").add_argument("receipt", type=Path)
    args = parser.parse_args()
    source = parent.reference()
    if args.phase == "audit":
        audit_result = repeat_samples(args, source)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(audit_result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"Reproduced {len(audit_result['audited'])} native excerpts independently.")
        return
    if args.phase == "capture":
        document = capture(args, source)
        document["replay"] = compute(source, document)
        raw = json.dumps(document, separators=(",", ":"), sort_keys=True, allow_nan=False).encode() + b"\n"
        with args.output.open("xb") as handle:
            handle.write(gzip.compress(raw, mtime=0))
    else:
        document = parent.previous.read(args.receipt)
        if document["replay"] != compute(source, document):
            raise ValueError("Stored results differ from portable replay")
    print(json.dumps(document["replay"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
