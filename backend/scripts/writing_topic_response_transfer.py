"""Compare matched-topic predictability with the existing formal-writing passage rule."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median

from scripts import writing_context_pool_transfer as pool
from scripts import writing_editorial_worker_audit as matched
from scripts import writing_focus_inside_context as intake
from scripts import writing_missed_passage_transfer as causal
from scripts import writing_sentence_probe as prior
from scripts.writing_interval_transfer import fingerprints

SCHEMA = "chipmunk.writing-topic-response-transfer/1"
PARENT = pool.PANEL_DIR / "writing-sentence-probe-2026-11-14.json"
WORKER = pool.PANEL_DIR / "writing-sentence-probe-worker-2026-11-14.json"
STUDY = {
    "sources": "Reuse the 48 known government-report work groups for development and calibration. Use 24 new works per held generator. Exclude recorded works and overlapping normalized 12-word phrases.",
    "windows": "Read each exact interior sentence with a six-term cue built from the other four sentences. A neutral cue has exactly the same token count. Measure loss changes on identical non-cue target tokens with the existing base model.",
    "features": "Measure mean, median, spread and positive share of loss reduction with the topic cue. Center the mean against the other interior sentences. One head uses these changes. A second adds baseline ratio, observed loss, reference cross-entropy, and native context classifier response.",
    "screen": "Keep every natively supporting classifier window. Admit new ratio-only locations only where the ratio already supports the exact sentence and a learned response exceeds a frozen genuine-only point. Edges and incomplete response pairs supply no independent ratio-only location.",
    "points": "Use each genuine calibration document's highest eligible ratio-only response. Freeze a point allowing half of eligible calibration documents and one allowing one third. No public probability.",
}
GENERATORS = ("gemini-2.5-flash", "gpt-5.4-nano")
PINS = (
    "e4a52b8cd3a56ac9d748d101f9d1ea75e6ae13b6eb6c38dc95e41f64488c45b5",
    "912d90f2f22b2ea97bdf69930f6891584112381f5b811aedcac390bbdf1da064",
)
HEADS = ("topic", "raw", "joint")
STOP = frozenset("about after again against being could during every first found given having other report should since their there these thing those through under using where which while within would years".split())
WORDS = re.compile(r"[A-Za-z]{5,}")


def contract():
    return {"schema": SCHEMA, "method": STUDY, "sources": {p.name: pool.file_hash(p) for p in (PARENT, WORKER)},
            "held_pins": dict(zip(GENERATORS, PINS, strict=True)), "base_model": causal.MODELS[0][1]}


def prepare(old, shard_dir, path):
    from collections import defaultdict

    parent = intake.read(PARENT)
    if pool.file_hash(old) != parent["roster_sha256"]:
        raise ValueError("The earlier study roster differs")
    native = intake.read(old)["rows"]
    development = [r for r in native if r["role"] == "held" and r["domain"] == "reports"]
    groups = defaultdict(list)
    for row in development:
        groups[row["pair_family"]].append(row["group"])
    fit_groups = {key for values in groups.values() for key in sorted(set(values), key=lambda k: pool.sha(SCHEMA + k))[:16]}
    first = [r | {"role": "fit" if r["group"] in fit_groups else "calibration"} for r in development]

    prior_ids = {r["group"] for r in native}
    for receipt in pool.PANEL_DIR.glob("writing-*-2026-11-14.json"):
        if "topic-response" not in receipt.name:
            prior_ids.update(prior.SHA256.findall(receipt.read_text()))
    used = {word for r in native for word in fingerprints(r["text"])}
    held = []
    # Screen full genuine sources on both held-generator shards against the complete prior pool.
    tables = [intake.shard(shard_dir / f"reports_{family}.csv", "reports", family, pin)
              for family, pin in zip(GENERATORS, PINS, strict=True)]
    for table in tables:
        for work, versions in table.items():
            if pool.sha(work) in prior_ids and "v0" in versions:
                used.update(fingerprints(versions["v0"]["text"]))
    for family, table in zip(GENERATORS, tables, strict=True):
        positions = Counter()
        for key in sorted(table, key=lambda k: pool.sha(SCHEMA + family + k)):
            if sum(positions.values()) == 24:
                break
            group = pool.sha(key)
            versions = table[key]
            if group in prior_ids or "v0" not in versions:
                continue
            phrases = fingerprints(versions["v0"]["text"])
            if phrases & used:
                continue
            try:
                options = intake.excerpt_options(key, versions)
            except (KeyError, ValueError):
                continue
            if not options:
                continue
            pos = min(options, key=lambda i: (positions[i], i))
            for (text, spans, variant), label in zip(options[pos], ("original", family), strict=True):
                held.append({"domain": "reports", "role": "held", "family": label, "pair_family": family, "group": group,
                             "sha256": pool.sha(text), "spans": spans, "text": text, "variant": variant,
                             "truth": spans[pos] if variant else None, "spaces": [i for i, c in enumerate(text) if c.isspace()]})
            positions[pos] += 1
            prior_ids.add(group)
            used.update(phrases)
        if sum(positions.values()) != 24:
            raise ValueError(f"Insufficient unused held works for {family}")
        print(f"{family}: {dict(positions)}", flush=True)
    prior.write(path, contract() | {"rows": first + held})


def verify_model():
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer

    name, revision, digest = causal.MODELS[0]
    path = pool.ROOT / f"models/hf/hub/models--Qwen--{name}/snapshots/{revision}"
    if pool.file_hash(path / "model.safetensors") != digest or pool.file_hash(path / "tokenizer.json") != causal.TOKENIZER:
        raise ValueError("The frozen causal model differs")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True, dtype=dtype).eval().to(device)
    return tokenizer, model, device


def terms(parts, index):
    words = [w.lower() for j, part in enumerate(parts) if j != index for w in WORDS.findall(part) if w.lower() not in STOP]
    count = Counter(words)
    return sorted(count, key=lambda w: (-count[w], -len(w), w))[:6]


def feature_inputs(row, tokenizer):
    parts = [row["text"][a:b] for a, b in row["spans"]]
    stem = tokenizer.encode("Topic cues:", add_special_tokens=False)
    suffix = tokenizer.encode("\nSentence:", add_special_tokens=False)
    neutral = tokenizer.encode(" detail", add_special_tokens=False)
    if len(neutral) != 1:
        raise ValueError("Expected one neutral token")
    for c in range(1, 4):
        keywords = terms(parts, c)
        if len(keywords) < 6:
            yield None
            continue
        cue = tokenizer.encode(" " + ", ".join(keywords), add_special_tokens=False)
        target = " " + parts[c]
        tokenized = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
        token_ids = tokenized["input_ids"]
        ignore = [(m.start(), m.end()) for m in WORDS.finditer(target) if m[0].lower() in keywords]
        selected = [j for j, (a, b) in enumerate(tokenized["offset_mapping"])
                    if b > a and target[a:b].strip() and not any(u < b and a < v for u, v in ignore)]
        if len(selected) < 12 or len(token_ids) + len(stem) + len(cue) + len(suffix) > 256:
            yield None
            continue
        rest = suffix + token_ids
        prefix_length = len(stem) + len(cue) + len(suffix)
        yield stem + cue + rest, stem + neutral * len(cue) + rest, prefix_length, selected


def score_group(item, model, device):
    import numpy as np
    import torch

    if item is None:
        return None
    cue, neutral, begin, keep = item
    ids = torch.tensor([cue, neutral], device=device)
    selected = torch.tensor([begin + offset for offset in keep], device=device)
    with torch.inference_mode():
        logits = model(ids, use_cache=False).logits[:, selected - 1].float()
        loss = (logits.logsumexp(-1) - logits.gather(-1, ids[:, selected, None]).squeeze(-1)).cpu().numpy()
    delta = loss[1] - loss[0]
    return [round(float(value), 7) for value in (delta.mean(), np.median(delta), delta.std(), np.mean(delta > 0))]


def compact(row):
    return {key: value for key, value in row.items() if key != "text"}


def contextual_phase(path, output):
    data = intake.read(path)
    tokenizer, model, device = verify_model()
    results = []
    for index, row in enumerate(data["rows"], 1):
        values = [score_group(item, model, device) for item in feature_inputs(row, tokenizer)]
        results.append(compact(row) | {"topic": values})
        if index % 24 == 0:
            print(f"{device}: measured cues for {index}/{len(data['rows'])}", flush=True)
    prior.write(output, {"roster_sha256": pool.file_hash(path), "rows": results})


def read_prior_windows(path):
    import sqlite3

    study = intake.read(PARENT)
    parent = {r["sha256"]: r for r in study["held"] if r["domain"] == "reports"}
    with sqlite3.connect((path / "veritas.db").resolve().as_uri() + "?immutable=1", uri=True) as conn:
        pins = dict(conn.execute("select sha256, result_hash from analyses where status='done'"))
    if set(pins) != set(parent):
        raise ValueError("Prior paid report sources differ")
    output = {}
    for sha, pin in pins.items():
        source = path / "objects" / "sha256" / pin[:2] / pin[2:4] / pin
        if pool.file_hash(source) != pin:
            raise ValueError("A stored result hash differs")
        signal = {s["detector_id"]: s["details"]["text_evidence"] for s in intake.read(source)["signals"]
                  if s.get("details", {}).get("text_evidence", {}).get("windows")}
        source_focuses = parent[sha]["spans"]
        scores = []
        for key in ("txt.four_class", "txt.binoculars"):
            windows = signal[key]["windows"]
            if [[w["focus"]["start"], w["focus"]["end"]] for w in windows] != source_focuses:
                raise ValueError("Stored detector locations differ")
            scores.append(windows)
        output[sha] = ([entry["score"] for entry in scores[0]],
                       [[float.fromhex(r[f"{name}_exact_hex"]) for name in
                         ("score", "mean_negative_log_likelihood", "mean_cross_entropy")]
                        + [r["neighbor_expansion"]] for r in scores[1]])
    return output


def held_classifier(path, output):
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier

    if any(pool.file_hash(pool.SNAPSHOT / filename) != digest for filename, digest in pool.BASELINE_ASSETS.items()):
        raise ValueError("The serving classifier bytes differ")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    clf = FourClassTextClassifier.from_snapshot(pool.SNAPSHOT, device)
    rows = [r for r in intake.read(path)["rows"] if r["role"] == "held"]
    texts = [text for r in rows for text, _ in list(pool.inputs(r))[:5]]
    prob = clf.predict(texts, batch=8)
    prior.write(output, {"roster_sha256": pool.file_hash(path),
                         "scores": {r["sha256"]: prob[i*5:i*5+5] for i, r in enumerate(rows)}})


def held_ratios(path, output):
    from app.detectors.text.binoculars import sentence_score_records
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    reader, alt = causal.load_pair()
    ratios = {}
    for index, row in enumerate((r for r in intake.read(path)["rows"] if r["role"] == "held"), 1):
        text = row["text"]
        doc = from_string(text)
        sentences = segment(doc)
        if [[s.start, s.end] for s in sentences] != row["spans"] or doc.text != text:
            raise ValueError("Native sentence positions differ")
        scores = reader.token_scores(text)
        mirror = alt(text, return_offsets_mapping=True, add_special_tokens=False)
        if mirror["input_ids"] != scores.source_token_ids or list(map(tuple, mirror["offset_mapping"])) != scores.offsets:
            raise ValueError("The ratio tokenizations differ on this excerpt")
        ratios[row["sha256"]] = [
            None if r["score"] is None else
            [r["score"], r["mean_negative_log_likelihood"], r["mean_cross_entropy"], r["neighbor_expansion"]]
            for r in sentence_score_records(scores, sentences)
        ]
        if index % 24 == 0:
            print(f"Measured {index} ratio excerpts", flush=True)
    prior.write(output, {"roster_sha256": pool.file_hash(path), "scores": ratios})


def eligible(row, index):
    ratio = row["ratio"][index]
    return (row["classifier"][index] < .5 and ratio is not None and ratio[0] < causal.POINT
            and not ratio[-1] and 0 < index < 4)


def features(row, index, name):
    ratio = row["ratio"][index]
    raw = None if ratio is None or ratio[-1] else ratio[:3] + [row["classifier"][index]]
    if name == "raw":
        return raw
    values = row["topic"][index - 1]
    other = [v[0] for j, v in enumerate(row["topic"], 1) if j != index and v is not None]
    if values is None or not other:
        return None
    record = [*values, values[0] - median(other)]
    if name == "topic":
        return record
    return None if raw is None else record + raw


def fit(rows, name):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    items, labels = [], []
    for row in rows:
        if row["role"] != "fit":
            continue
        for i in range(1, 4):
            f = features(row, i, name)
            if f is not None:
                items.append(f)
                labels.append(row["spans"][i] == row["truth"])
    if sum(labels) < 24:
        raise ValueError("Insufficient fit edits")
    x = np.asarray(items)
    mean = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), .05)
    reg = LogisticRegression(C=.10, class_weight="balanced", max_iter=200).fit((x - mean)/scale, labels)
    return {"coef": reg.coef_[0].tolist(), "bias": float(reg.intercept_[0]), "mean": mean.tolist(), "scale": scale.tolist()}


def readout(row, model):
    output = {}
    for name, fit in model.items():
        readings = []
        for i in range(1, 4):
            f = features(row, i, name)
            readings.append(None if f is None else round(fit["bias"] + math.fsum(
                (v - m) * w / s for v, m, w, s in zip(f, fit["mean"], fit["coef"], fit["scale"], strict=True)), 7))
        output[name] = readings
    return output


def calibration(rows):
    cutoffs = {}
    for name in HEADS:
        maxima = []
        for row in rows:
            if row["role"] == "calibration" and row["truth"] is None:
                scores = [v for i, v in enumerate(row["scores"][name], 1) if v is not None and eligible(row, i)]
                if scores:
                    maxima.append(max(scores))
        if len(maxima) < 8:
            raise ValueError("Insufficient ordinary cases with a measurable ratio-only response")
        ordered = sorted(maxima, reverse=True)
        cutoffs[name] = {policy: ordered[len(ordered) // n] for policy, n in (("half", 2), ("third", 3))}
    return cutoffs


def native_range(row, selected):
    """Mirror the native five-sentence report precedence of exact ratio and classifier context."""
    output = []
    spans = row["spans"]
    for i, (prob, ratio) in enumerate(zip(row["classifier"], row["ratio"], strict=True)):
        support = ratio is not None and ratio[0] < causal.POINT
        ratio_selected = support and (prob >= .5 or selected is None or i in selected)
        local = spans[i]
        context = [spans[max(0, i - 1)][0], spans[min(4, i + 1)][1]]
        if ratio_selected and not ratio[-1]:
            output.append(local)
        elif prob >= .5 or ratio_selected:
            output.append(context)
    return output


def summarize(rows, points):
    partitions = {}
    for row in rows:
        label = ("ordinary" if row["truth"] is None else "rewritten")
        group = partitions.setdefault("/".join((row["pair_family"], label)), Counter())
        success = bool if row["truth"] is None else lambda ranges: matched.covers(ranges, row["truth"], set(row["spaces"]))
        group.update(total=1, current=success(native_range(row, None)), classifier_only=success(native_range(row, set())))
        for name, policy in points.items():
            for key, cutoff in policy.items():
                retained = {i for i, v in enumerate(row["scores"][name], 1) if v is not None and v > cutoff and eligible(row, i)}
                group[f"{name}/{key}"] += success(native_range(row, retained))
    return partitions


def analyze(roster, cue, classifier, ratio, prior_db, output):
    data = intake.read(roster)
    checks = [intake.read(f) for f in (cue, classifier, ratio)]
    if any(item["roster_sha256"] != pool.file_hash(roster) for item in checks):
        raise ValueError("Scoring stages used different frozen inputs")
    historic = read_prior_windows(prior_db)
    rows = []
    for binding, reading in zip(data["rows"], checks[0]["rows"], strict=True):
        sha = binding["sha256"]
        if compact(binding) != {key: value for key, value in reading.items() if key != "topic"}:
            raise ValueError("Conditioned-response record differs from its source")
        clf, rew = ((checks[1]["scores"][sha], checks[2]["scores"][sha]) if binding["role"] == "held" else historic[sha])
        rows.append(reading | {"classifier": clf, "ratio": rew})
    model = {name: fit(rows, name) for name in HEADS}
    report = [r | {"scores": readout(r, model)} for r in rows]
    fit_data = [r for r in report if r["role"] == "fit"]
    cal = [r for r in report if r["role"] == "calibration"]
    held = [r for r in report if r["role"] == "held"]
    points = calibration(cal)
    result = contract() | {"roster_sha256": pool.file_hash(roster), "fit": fit_data, "calibration": cal, "held": held,
                           "heads": model, "points": points, "results": summarize(held, points)}
    review(result)
    prior.write(output, result)
    print(json.dumps(result["results"], sort_keys=True), flush=True)


def review(receipt):
    if any(receipt[k] != v for k, v in contract().items()):
        raise ValueError("Probe method or source commitments differ")
    if set(receipt["heads"]) != set(HEADS):
        raise ValueError("A conditioned or raw response head is missing")
    groups = [set(r["group"] for r in receipt[key]) for key in ("fit", "calibration", "held")]
    if list(map(len, groups)) != [32, 16, 48] or len(set.union(*groups)) != 96:
        raise ValueError("Inconsistent source split")
    for key in ("fit", "calibration", "held"):
        rows = receipt[key]
        if (len(rows) != 2 * len({r["group"] for r in rows})
                or any(r["role"] != key or r["scores"] != readout(r, receipt["heads"]) for r in rows)):
            raise ValueError("A reading or source role differs")
        pairs = {}
        for row in rows:
            spans = row["spans"]
            if (not prior.SHA256.fullmatch(row["sha256"]) or len(spans) != 5
                    or any(not 0 <= a < b for a, b in spans)
                    or any(a[1] > b[0] for a, b in zip(spans, spans[1:]))
                    or row["truth"] is not None and row["truth"] not in spans[1:4]
                    or len(row["classifier"]) != 5 or any(not 0 <= value <= 1 for value in row["classifier"])):
                raise ValueError("A source position or response is invalid")
            pairs.setdefault(row["group"], []).append(row)
        if any(len(pair) != 2 or sum(row["truth"] is None for row in pair) != 1
               or len({row["pair_family"] for row in pair}) != 1 for pair in pairs.values()):
            raise ValueError("A work is missing its matched control or rewrite")
    cutoffs = calibration(receipt["calibration"])
    counts = summarize(receipt["held"], cutoffs)
    if counts != receipt["results"] or cutoffs != receipt["points"]:
        raise ValueError("The reported points or outcomes differ")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "topic", "classifier", "ratio", "analyze", "review"))
    parser.add_argument("path", type=Path)
    for arg in ("old", "shards", "output", "cue", "classifier", "ratio", "prior-db"):
        parser.add_argument(f"--{arg}", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.old, args.shards, args.path)
    elif args.mode == "review":
        print(json.dumps(review(intake.read(args.path)), sort_keys=True))
    elif args.mode == "analyze":
        analyze(args.path, args.cue, args.classifier, args.ratio, args.prior_db, args.output)
    else:
        phase = {"topic": contextual_phase, "classifier": held_classifier, "ratio": held_ratios}[args.mode]
        phase(args.path, args.output)


if __name__ == "__main__":
    main()
