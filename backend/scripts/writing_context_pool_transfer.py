"""Compare sentence-local token pooling in short and full writing contexts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median

from scripts.writing_focus_v3 import BASELINE_ASSETS, BASELINE_REVISION
from scripts.writing_interval_transfer import fingerprints
from scripts.writing_local_context_transfer import (
    ESSAYS,
    GENERATORS,
    ROOT,
    SHARDS,
    VARIANTS,
    file_hash,
    sha,
    shard,
    truth_text,
    useful,
)
from scripts.writing_sentence_anchor_transfer import source_spans

SCHEMA = "chipmunk.writing-context-pool-transfer/1"
MODES = ("short", "combined")
DOMAINS = ("news", "abstracts", "essays")
PANEL_DIR = ROOT / "docs/evaluation-panels"
EXPOSURE = (
    "writing-paired-scale-news-2026-11-14.json",
    "writing-paired-scale-abstracts-2026-11-14.json",
    "writing-independent-anchor-news-2026-11-14.json",
    "writing-independent-anchor-abstracts-2026-11-14.json",
    "writing-local-context-transfer-2026-11-14.json",
)
ROLES = {"news": (18, 12, 18), "abstracts": (18, 12, 18), "essays": (0, 0, 30)}
SNAPSHOT = ROOT / "models/hf/hub/models--wasitaigeneratedcom--ai-text-detector-small/snapshots" / BASELINE_REVISION
METHOD = {
    "admission": "Use pinned test shards and held publisher character truth. Admit one five-sentence, "
    "one-paragraph excerpt and two native single-sentence rewrites per work. Rotate the edited "
    "position between the three interior sentences within each role and domain before inference. "
    "Exclude prior work hashes and normalized 12-word phrase overlap against prior and admitted works.",
    "input": "Both experimental arms use every interior sentence. Pool exactly its contextual content "
    "tokens using the incumbent linear mean-pooling head. Measure each focus in its native three "
    "sentences and, separately, the entire five-sentence excerpt.",
    "features": "For each view use the AI-like log odds, log odds relative to the median interior "
    "sentence, and the edit-like versus human margin. The joint arm also uses the lower of the "
    "two relative odds. Compare to the unchanged proxy of first six native contexts at .50.",
    "fit": "Use only news and abstracts. In each arm fit a standardized ridge-logistic ranker on "
    "the focus labels, equally weighting source works and positive and negative examples per work. "
    "Use .15 L2, a .1 feature scale floor, and 120 steps of .3. Admit at most one unique maximum.",
    "calibration": "Freeze two separate points per arm from genuine calibration only. Use strict "
    "excess over the second-highest work maximum, and separately use the observed incumbent count "
    "of highlighted genuine calibration works to match its alarm budget.",
    "advance": "A complement needs four useful new essay locations spanning both generators, "
    "at most two new genuine flags in the union, and at most two wrong-location additions. "
    "A substitute would need more useful locations than the proxy with no more ordinary flags. "
    "Both also require assessment of at least 90 percent of held files. No authorship score weight.",
}


def source_windows(text):
    units = source_spans(text)
    for i in range(len(units) - 4):
        window = units[i:i + 5]
        if (window[0][0].paragraph == window[-1][0].paragraph
                and all(8 <= unit.n_words <= 80 for unit, _, _ in window)):
            yield [(a, b) for _, a, b in window]


def signature(text, spans, index):
    return tuple(text[a:b] for j, (a, b) in enumerate(spans) if j != index)


def normalize(raw, spans):
    from app.detectors.text.extract import _normalise, from_string
    from app.detectors.text.segment import segment

    text = raw[spans[0][0]:spans[-1][1]]
    doc = from_string(text)
    units = segment(doc)
    if (len(units) != 5 or len(doc.paragraphs) != 1 or not 60 <= doc.meta["words"] <= 240
            or [u.text for u in units] != [_normalise(raw[a:b])[0] for a, b in spans]):
        raise ValueError("Unsupported excerpt geometry")
    return doc.text, [[u.start, u.end] for u in units]


def candidates(rows, original_spans, raw, index):
    found = {}
    for variant in VARIANTS:
        text, gold = truth_text(rows[variant])
        for spans in source_windows(text):
            focus = spans[index]
            if focus not in gold:
                continue
            key = signature(text, spans, index)
            if key not in original_spans or key in found:
                continue
            start, end = spans[0][0], spans[-1][1]
            if any(a < end and b > start and (a, b) != focus for a, b in gold):
                continue
            if raw[slice(*original_spans[key][index])] != text[slice(*focus)]:
                found[key] = (text, spans, variant)
    return found


def prepare(folders):
    paths = {name: PANEL_DIR / name for name in EXPOSURE}
    excluded = {value for path in paths.values()
                for value in re.findall(r'"([a-f0-9]{64})"', path.read_text())}
    sources = {domain: [shard(folders[domain], domain, family) for family in GENERATORS] for domain in DOMAINS}
    prior_words = set()
    for first, second in sources.values():
        for key in first.keys() & second.keys():
            if sha(key) in excluded:
                prior_words.update(fingerprints(first[key]["v0"]["text"]))

    cases, admission = [], {}
    for domain, (first, second) in sources.items():
        fit_count, calibration_count, held_count = ROLES[domain]
        wanted = fit_count + calibration_count + held_count
        counts = Counter()
        for key in sorted(first.keys() & second.keys(), key=lambda k: sha(f"{SCHEMA}/{domain}/{k}")):
            n = counts["used"]
            if n == wanted:
                break
            if sha(key) in excluded:
                counts["prior_work"] += 1
                continue
            role = "fit" if n < fit_count else "calibration" if n < fit_count + calibration_count else "held"
            index = n % 3 + 1
            raw, gold = truth_text(first[key]["v0"])
            if gold or raw != second[key]["v0"]["text"]:
                counts["seed_disagreement"] += 1
                continue
            phrases = fingerprints(raw)
            if phrases & prior_words:
                counts["prior_words"] += 1
                continue
            try:
                options = [(signature(raw, spans, index), spans) for spans in source_windows(raw)]
                duplicates = Counter(k for k, _ in options)
                available = {k: spans for k, spans in options if duplicates[k] == 1}
                hits = [candidates(source[key], available, raw, index) for source in (first, second)]
                common = hits[0].keys() & hits[1].keys()
                if not common:
                    counts["no_matching_revision"] += 1
                    continue
                pick = min(common, key=lambda k: sha(key + "/".join(k)))
                versions = [(raw, available[pick], None), *(item[pick] for item in hits)]
                normalized = [(normalize(text, spans), variant) for text, spans, variant in versions]
            except ValueError:
                counts["invalid_geometry"] += 1
                continue
            if any(len({text[a:b] for a, b in spans}) != 5 for (text, spans), _ in normalized):
                counts["repeated_sentence"] += 1
                continue
            for family, ((text, spans), variant) in zip(("original", *GENERATORS), normalized, strict=True):
                cases.append({"group": sha(key), "domain": domain, "role": role, "family": family,
                              "variant": variant, "sha256": sha(text), "spans": spans, "text": text,
                              "truth": None if variant is None else spans[index],
                              "spaces": [i for i, c in enumerate(text) if c.isspace()]})
            counts["used"] += 1
            prior_words.update(phrases)
        if counts["used"] != wanted:
            raise ValueError(f"Insufficient position-balanced worked examples for {domain}: {counts}")
        admission[domain] = dict(counts)
        print(domain, admission[domain], flush=True)
    return cases, admission, {name: file_hash(path) for name, path in paths.items()}


def inputs(row):
    text = row["text"]
    spans = row["spans"]
    for i in range(5):
        lo, hi = max(0, i - 1), min(5, i + 2)
        fragments = [text[a:b] for a, b in spans[lo:hi]]
        prefix = sum(len(f) + 1 for f in fragments[:i - lo])
        yield " ".join(fragments), [(prefix, prefix + len(fragments[i - lo]))] if i in (1, 2, 3) else []
    yield text, spans[1:4]


def score_batch(items, clf):
    import torch

    texts = [item[0] for item in items]
    batch = clf.tok(texts, padding=True, return_tensors="pt", return_offsets_mapping=True,
                    return_special_tokens_mask=True, truncation=False)
    offsets = batch.pop("offset_mapping").tolist()
    specials = batch.pop("special_tokens_mask").tolist()
    if batch["input_ids"].shape[1] > 512:
        raise ValueError("Unexpected token overflow")
    encoded = {k: v.to(clf.device) for k, v in batch.items() if k in {"input_ids", "attention_mask"}}
    with torch.inference_mode():
        vec = clf.model.model(**encoded).last_hidden_state
        attended = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (vec * attended).sum(1) / attended.sum(1)
        baseline = (1 - clf.model.classifier(pooled).float().softmax(-1)[:, 0]).cpu().tolist()
        token_logits = clf.model.classifier(vec).float().cpu()
    results = []
    for (text, foci), mask, tokens, valid, total, logits in zip(
        items, specials, offsets, batch["attention_mask"].tolist(), baseline, token_logits, strict=True,
    ):
        focuses = []
        for start, end in foci:
            kept = []
            covered = set()
            for i, ((a, b), special, active) in enumerate(zip(tokens, mask, valid, strict=True)):
                if special or not active or b <= a:
                    continue
                # A tokenizer can attach a leading space to a word. Admit that space but no text from another sentence.
                relevant = {k for k in range(a, b) if not text[k].isspace()}
                if not any(start <= k < end for k in relevant):
                    continue
                if any(k < start or k >= end for k in relevant):
                    kept = []
                    break
                kept.append(i)
                covered.update(relevant)
            expected = {i for i in range(start, end) if not text[i].isspace()}
            if not kept or covered != expected:
                focuses.append(None)
                continue
            human, generated, edited, humanized = logits[kept].mean(0).tolist()
            equal_to_human = (generated - human, edited - human, humanized - human)
            best = max(equal_to_human)
            nonhuman = best + math.log(math.fsum(math.exp(x - best) for x in equal_to_human))
            edit_best = max(equal_to_human[1:])
            edit_like = edit_best + math.log(math.fsum(math.exp(x - edit_best) for x in equal_to_human[1:]))
            focuses.append([nonhuman, edit_like])
        results.append([total, focuses])
    return results


def measurements(rows, clf):
    cache = {}
    for row in rows:
        request = [(text, tuple(tuple(span) for span in spans)) for text, spans in inputs(row)]
        fresh = [key for key in dict.fromkeys(request) if key not in cache]
        if fresh:
            cache.update(zip(fresh, score_batch(fresh, clf), strict=True))
        values = [cache[key] for key in request]
        yield ({k: v for k, v in row.items() if k != "text"}
               | {"input_sha256": [sha(text) for text, _ in request],
                  "baseline": [v[0] for v in values[:5]], "short": [v[1][0] for v in values[1:4]],
                  "full": values[5][1]})


def features(row, mode):
    views = (row["short"],) if mode == "short" else (row["short"], row["full"])
    if any(None in view for view in views):
        return None
    values = []
    for position in range(3):
        f = []
        for view in views:
            center = view[position]
            f.extend((center[0], center[0] - median(x[0] for x in view), center[1]))
        if mode == "combined":
            f.append(min(f[1], f[4]))
        values.append(f)
    return values


def fit(rows, mode):
    groups = group_rows(row for row in rows if row["role"] == "fit")
    complete = [rs for rs in groups.values() if all(features(r, mode) is not None for r in rs)]
    if len(complete) < 24:
        raise ValueError("Insufficient complete fitting works")
    data = []
    for versions in complete:
        by_class = {False: [], True: []}
        for row in versions:
            for f, span in zip(features(row, mode), row["spans"][1:4], strict=True):
                by_class[span == row["truth"]].append(f)
        for label, examples in by_class.items():
            data.extend((f, int(label), 1 / (2 * len(complete) * len(examples))) for f in examples)
    size = len(data[0][0])
    mean = [math.fsum(f[i] * w for f, _, w in data) for i in range(size)]
    scale = [max(.1, math.sqrt(math.fsum((f[i] - mean[i])**2 * w for f, _, w in data))) for i in range(size)]
    scaled = [([1, *[(v - m) / s for v, m, s in zip(f, mean, scale, strict=True)]], y, w) for f, y, w in data]
    coefficients = [0.] * (size + 1)
    for _ in range(120):
        errors = []
        for f, label, weight in scaled:
            logit = math.fsum(x * b for x, b in zip(f, coefficients, strict=True))
            prediction = 1 / (1 + math.exp(-min(40, max(-40, logit))))
            errors.append((f, (prediction - label) * weight))
        coefficients = [b - .3 * (math.fsum(f[i] * e for f, e in errors) + (.15 * b if i else 0))
                        for i, b in enumerate(coefficients)]
    return {"mean": mean, "scale": scale, "coefficients": coefficients, "fit_groups": len(complete)}


def rank(row, mode, head):
    vectors = features(row, mode)
    if vectors is None:
        return None
    scores = [head["coefficients"][0] + math.fsum(
        b * (x - m) / s for b, x, m, s in zip(
            head["coefficients"][1:], f, head["mean"], head["scale"], strict=True,
        )
    ) for f in vectors]
    order = sorted(range(3), key=scores.__getitem__, reverse=True)
    if scores[order[0]] == scores[order[1]]:
        return None
    return scores[order[0]], order[0] + 1


def group_rows(rows):
    result = {}
    for row in rows:
        result.setdefault(row["group"], []).append(row)
    return result


def baseline(row):
    spans = row["spans"]
    return [[spans[max(0, i - 1)][0], spans[min(4, i + 1)][1]]
            for i, value in enumerate(row["baseline"]) if value >= .50]


def points(dev):
    if any(row["role"] == "held" for row in dev):
        raise ValueError("Held data cannot set an operating point")
    controls = [row for row in dev if row["role"] == "calibration" and row["family"] == "original"]
    result = {}
    for mode in MODES:
        head = fit(dev, mode)
        ranked = sorted((winner[0] for row in controls if (winner := rank(row, mode, head)) is not None),
                        reverse=True)
        budget = sum(bool(baseline(row)) for row in controls)
        if len(ranked) < 18 or budget >= len(ranked):
            raise ValueError("Insufficient genuine calibration coverage")
        result[mode] = head | {"strict": ranked[1], "matched": ranked[budget],
                               "genuine_budget": budget, "calibration_assessed": len(ranked)}
    return result


def summary(rows, parameters):
    result = {}
    for domain in DOMAINS:
        stats, gains = Counter(), {}
        for row in (r for r in rows if r["domain"] == domain):
            original = row["family"] == "original"
            base = baseline(row)
            located = any(useful(row, span) for span in base)
            stats["genuine" if original else "edited"] += 1
            stats["baseline_useful"] += located
            stats["baseline_genuine_flags"] += original and bool(base)
            stats["baseline_wrong_only"] += not original and bool(base) and not located
            for mode, head in parameters.items():
                winner = rank(row, mode, head)
                stats[f"{mode}/assessed"] += winner is not None
                for point in ("strict", "matched"):
                    key = f"{mode}/{point}"
                    flags = winner is not None and winner[0] > head[point]
                    hit = flags and useful(row, row["spans"][winner[1]])
                    stats[f"{key}/useful"] += hit
                    stats[f"{key}/retained"] += hit and located
                    stats[f"{key}/genuine_flags"] += flags and original
                    stats[f"{key}/new_genuine_flags"] += flags and original and not base
                    stats[f"{key}/wrong_only"] += flags and not original and not hit
                    stats[f"{key}/new_wrong_only"] += flags and not original and not hit and not base
                    stats[f"{key}/new_useful"] += hit and not located
                    if hit and not located:
                        gains.setdefault(key, {}).setdefault(row["family"], set()).add(row["group"])
        for key, by_family in gains.items():
            for family, groups in by_family.items():
                stats[f"{key}/gain_works/{family}"] = len(groups)
        result[domain] = dict(stats)
    return result


def validate_rows(dev, held):
    groups = group_rows(dev + held)
    expected = Counter()
    for rows in groups.values():
        identities = {(r["domain"], r["role"]) for r in rows}
        if len(identities) != 1 or len(rows) != 3 or {r["family"] for r in rows} != {"original", *GENERATORS}:
            raise ValueError("Work overlaps roles or lacks a delivery")
        domain, role = next(iter(identities))
        positions = {r["spans"].index(r["truth"]) for r in rows if r["family"] != "original"}
        if len(positions) != 1 or not positions <= {1, 2, 3}:
            raise ValueError("Labels disagree about native position")
        for r in rows:
            if (len(r["spans"]) != 5 or len(r["baseline"]) != 5 or len(r["input_sha256"]) != 6
                    or r["input_sha256"][-1] != r["sha256"] or "text" in r):
                raise ValueError("Invalid numeric delivery")
            for view in ("short", "full"):
                if len(r[view]) != 3 or any(
                    f is not None and (len(f) != 2 or not all(math.isfinite(v) for v in f)) for f in r[view]
                ):
                    raise ValueError("Invalid contextual pooling responses")
        expected[domain, role, next(iter(positions))] += 1
    targets = {(d, role, i): n // 3 for d in DOMAINS for role, n in zip(
        ("fit", "calibration", "held"), ROLES[d], strict=True,
    ) for i in (1, 2, 3) if n}
    if expected != targets or any(r["role"] != "held" for r in held):
        raise ValueError("Unexpected held roles or position balance")


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "develop", "held", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output", type=Path)
    for domain in DOMAINS:
        parser.add_argument(f"--{domain}", type=Path)
    args = parser.parse_args()
    runner = file_hash(Path(__file__))
    if args.mode == "review":
        result = json.loads(args.path.read_text())
        dev, held = result["development"], result["held"]
        validate_rows(dev, held)
        if (result["freeze"]["method"] != METHOD or result["freeze"]["runner_sha256"] != runner
                or result["parameters"] != points(dev) or result["counts"] != summary(held, result["parameters"])):
            raise ValueError("The numeric study does not replay")
        print(json.dumps(result["counts"], indent=2))
        return
    if args.mode == "freeze":
        rows, admission, exposure = prepare({d: getattr(args, d) for d in DOMAINS})
        freeze = {"schema": SCHEMA, "method": METHOD, "runner_sha256": runner, "admission": admission,
                  "exposure": exposure, "weights": BASELINE_ASSETS, "revision": BASELINE_REVISION,
                  "shards": SHARDS | {"essays": ESSAYS}, "runtime": {pkg: importlib.metadata.version(pkg)
                  for pkg in ("torch", "transformers", "tokenizers", "pysbd")}}
        write(args.path, {"freeze": freeze, "rows": rows})
        return
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier

    if args.output is None:
        parser.error("--output is required")
    task = json.loads(args.path.read_text())
    freeze = task["freeze"]
    if (runner != freeze["runner_sha256"] or freeze["method"] != METHOD
            or any(file_hash(args.snapshot / name) != expected for name, expected in BASELINE_ASSETS.items())):
        raise ValueError("The frozen procedure or pinned weights changed")
    digest = file_hash(args.path)
    if args.mode == "held":
        dev = json.loads(args.path.with_suffix(".development.json").read_text())
        if dev["input_roster_sha256"] != digest or points(dev["development"]) != dev["parameters"]:
            raise ValueError("Development record does not match")
    torch.set_num_threads(3)
    clf = FourClassTextClassifier.from_snapshot(args.snapshot, "cpu")
    rows = []
    for row in measurements((r for r in task["rows"] if (r["role"] == "held") == (args.mode == "held")), clf):
        rows.append(row)
        if len(rows) % 36 == 0:
            print(f"{args.mode}: {len(rows)} scored", flush=True)
    if args.mode == "develop":
        write(args.output, {"freeze": freeze, "input_roster_sha256": digest, "development": rows,
                            "parameters": points(rows)})
    else:
        validate_rows(dev["development"], rows)
        counts = summary(rows, dev["parameters"])
        write(args.output, dev | {"held": rows, "counts": counts})
        print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
