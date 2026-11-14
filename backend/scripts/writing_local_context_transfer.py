"""Evaluate source-disjoint single-upload sentence localization with one or two checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from scripts.writing_paired_scale_transfer import GENERATORS, SHARDS, clamped_odds
from scripts.writing_revision_trajectory_transfer import ranges_from_tags

SCHEMA = "chipmunk.writing-local-context-transfer/1"
ESSAYS = {
    "gemini-2.5-flash": "d4805b719907ea7e3e7215c0cbfd5d47e95f7bad394a51fcdc851d4bc029222e",
    "gpt-5.4-nano": "b1177c4cbda63ccd623dc903037aa8e3ac20b31fcb5e4fe8bcf93c32980f3045",
}
METHOD = {
    "roles": "Use source-disjoint news and abstracts for fit (first 24 works per genre) and calibration "
             "(next 16 each). Hold out essays entirely. Use the next 20 news and abstract works each as "
             "separate same-genre held groups if available. Use up to 48 unseen essay works.",
    "selection": "Select in fixed SHA-256 order. Exclude works in the previous paired news and abstract cohorts "
                 "and any normalized 12-word overlap between admitted seeds or those excluded works. Each work "
                 "must have one source-native five-sentence excerpt with exactly one annotated center from each "
                 "of two generators. Use the unchanged source excerpt as a control. Use character offsets and "
                 "inline tags for truth. Reject reviews with uncertain native offset mapping.",
    "arms": "Fit two source-balanced logistic heads. The first uses the primary classifier in isolation and in "
            "native three-sentence context. The second combines the same measurements from both checkpoints. "
            "For each checkpoint use isolated center, context, their difference, and each relative to the "
            "median across the interior sentences. Only the submitted excerpt enters inference.",
    "fit": "Weight each source equally and balance positive and negative truth per source. L2 coefficient .10, "
           "standard deviations floored at .1, 100 deterministic gradient steps at .3. Use separate points at "
           "the second-highest genuine calibration maximum per arm. Select at most one unique interior sentence.",
    "baseline": "Use the primary checkpoint's first six windows at .50, with its native edge windows. "
                "A useful range covers at least half of the labeled nonspace characters and reaches .30 "
                "intersection over union. Also count wrong candidate locations and ordinary false flags.",
    "decision": "An optional new confirmation hint needs at least four additional useful locations across both "
                "generators and at most two more flagged ordinary works than the incumbent on the essay holdout. "
                "A narrower presentation policy needs at least 75 percent of incumbent useful essay locations "
                "and at least three fewer ordinary works flagged. Neither can alter authorship confidence.",
}
VARIANTS = ("v1", "v2", "v4")
ROOT = Path(__file__).resolve().parents[2]
SOURCE = "docs/evaluation-panels/writing-paired-scale-{}-2026-11-14.json"
HISTORIC = "docs/evaluation-panels/writing-paired-center-transfer-2026-11-12.json"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def truth_text(row):
    raw, tagged = ranges_from_tags(row["text_tagged"])
    if raw != row["text"] or tagged != json.loads(row["ai_spans_char"]):
        raise ValueError("The published character labels disagree with the inline tags")
    return raw, {tuple(span) for span in tagged}


def shard(folder, domain, family):
    path = folder / f"{domain}_{family}.csv"
    limits = ESSAYS if domain == "essays" else SHARDS[domain]
    if file_hash(path) != limits[family]:
        raise ValueError("Benchmark bytes differ")
    csv.field_size_limit(1024 * 1024)
    work = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, strict=True):
            if (row["domain"], row["generator"], row["split"], row["subset"]) != (domain, family, "test", "main"):
                raise ValueError("Unexpected benchmark provenance")
            variant = row["version"]
            if variant in ("v0", *VARIANTS):
                if variant in work[row["id"]]:
                    raise ValueError("Duplicate benchmark work/version")
                work[row["id"]][variant] = row
    return work


def passage_windows(raw):
    from scripts.writing_sentence_anchor_transfer import source_spans

    units = source_spans(raw)
    for i in range(1, len(units) - 3):
        chosen = units[i - 1:i + 4]
        # Include two sentences on each side of position two in every excerpt.
        if chosen[0][0].paragraph != chosen[-1][0].paragraph or any(row[0].n_words < 6 for row in chosen):
            continue
        start, stop = chosen[0][1], chosen[-1][2]
        focus = chosen[2][1:]
        signature = tuple(raw[a:b] for j, (_, a, b) in enumerate(chosen) if j != 2)
        yield signature, (start, stop, focus)


def variants(rows, lookup, raw):
    output = {}
    for version in VARIANTS:
        try:
            text, labels = truth_text(rows[version])
            for signature, (start, stop, focus) in passage_windows(text):
                old = lookup.get(signature)
                if old is None or focus not in labels or raw[slice(*old[2])] == text[slice(*focus)]:
                    continue
                if any(lo < stop and hi > start and (lo, hi) != focus for lo, hi in labels):
                    continue
                output.setdefault(signature, (text, start, stop, focus, version))
        except ValueError:
            continue
    return output


def normalize_excerpt(text, start, end, focus=None):
    from app.detectors.text.extract import _normalise, from_string
    from app.detectors.text.segment import segment

    excerpt = text[start:end]
    doc = from_string(excerpt)
    units = segment(doc)
    if len(units) != 5 or len(doc.paragraphs) != 1 or not 60 <= doc.meta["words"] <= 280:
        raise ValueError("Excerpt no longer has five bounded source sentences")
    spans = [[u.start, u.end] for u in units]
    if focus is not None:
        left, right = focus
        if _normalise(text[left:right])[0] != units[2].text:
            raise ValueError("Publisher truth does not match the normalized center")
        focus = spans[2]
    return doc.text, spans, focus


def prepare(folders):
    from scripts.writing_interval_transfer import fingerprints

    result, admission, exclusion = [], {}, set()
    for domain in ("news", "abstracts"):
        path = ROOT / SOURCE.format(domain)
        exclusion.update(label["group"] for row in json.loads(path.read_text())["rows"]
                         for label in row["truth"] if label["family"] != "ordinary")
    global_phrases = set()
    sources = {domain: [shard(folders[domain], domain, family) for family in GENERATORS] for domain in folders}
    for first, second in sources.values():
        for name in first.keys() & second.keys():
            if sha(name) in exclusion:
                global_phrases.update(fingerprints(first[name]["v0"]["text"]))
    for domain, (first, second) in sources.items():
        count = Counter()
        limit = 48 if domain == "essays" else 60
        for name in sorted(first.keys() & second.keys(), key=lambda key: sha(f"{SCHEMA}/{domain}/{key}")):
            if count["used"] == limit:
                break
            g = sha(name)
            if g in exclusion:
                count["excluded_prior"] += 1
                continue
            try:
                raw, gold = truth_text(first[name]["v0"])
                if raw != second[name]["v0"]["text"] or gold:
                    count["seed_differs"] += 1
                    continue
                seqs = fingerprints(raw)
                if seqs & global_phrases:
                    count["phrase_overlap"] += 1
                    continue
                original = list(passage_windows(raw))
                totals = Counter(key for key, _ in original)
                lookup = {key: value for key, value in original if totals[key] == 1}
                hits = [variants(table[name], lookup, raw) for table in (first, second)]
                common = hits[0].keys() & hits[1].keys()
                if not common:
                    count["no_matched_center"] += 1
                    continue
                key = min(common, key=lambda value: sha(g + "/" + "/".join(value)))
                start, stop, _ = lookup[key]
                bands = (("original", normalize_excerpt(raw, start, stop), None),
                         *((f, normalize_excerpt(*rows[key][:-1]), rows[key][-1])
                           for f, rows in zip(GENERATORS, hits, strict=True)))
                if any(len(values[0].split()) < 60 for _, values, _ in bands):
                    count["short"] += 1
                    continue
            except ValueError:
                count["invalid_geometry"] += 1
                continue
            role = "held" if domain == "essays" or count["used"] >= 40 else "calibration" if count["used"] >= 24 else "fit"
            for family, (text, spans, truth), version in bands:
                result.append({"genre": domain, "role": role, "group": g, "family": family, "variant": version,
                               "sha256": sha(text), "spans": spans, "truth": truth,
                               "spaces": [i for i, c in enumerate(text) if c.isspace()], "text": text})
            count["used"] += 1
            global_phrases.update(seqs)
        admission[domain] = dict(count)
    print("Source admission", admission, flush=True)
    roles = Counter((r["genre"], r["role"]) for r in result if r["family"] == "original")
    if roles["essays", "held"] < 20 or any(roles[d, r] < 16 for d in ("news", "abstracts") for r in ("fit", "calibration")):
        raise ValueError(f"Insufficient disjoint work groups {roles}")
    return result, admission


def input_windows(row):
    text = row["text"]
    parts = [text[a:b] for a, b in row["spans"]]
    contexts = [" ".join(parts[max(0, i - 1):i + 2]) for i in range(5)]
    return contexts + parts[1:-1]


def measure(rows, cached=None):
    from app.evidence.document_revision_writing import _score

    cached = {} if cached is None else cached
    for row in rows:
        inputs = input_windows(row)
        fresh = [s for s in dict.fromkeys(inputs) if s not in cached]
        if fresh:
            scores = _score(fresh, allow_partial=True)
            if scores is None:
                raise ValueError("Pinned local writing checkpoints unavailable")
            cached.update(zip(fresh, zip(*scores, strict=True), strict=True))
        yield {k: v for k, v in row.items() if k != "text"} | {
            "input_sha256": list(map(sha, inputs)), "responses": [list(cached[s]) for s in inputs],
        }


def features(record, streams, center):
    values = []
    responses = record["responses"]
    for j in range(streams):
        ctx, alone = ([responses[i][j] for i in span] for span in ((1, 2, 3), (5, 6, 7)))
        if any(value is None for value in ctx + alone):
            return None
        a, b = (list(map(lambda v: math.log(clamped_odds(v)), scores)) for scores in (ctx, alone))
        c, d = a[center], b[center]
        values.extend((c, d, c - d, c - median(a), d - median(b)))
    return values


def sigmoid(x):
    return 1 / (1 + math.exp(-min(40, max(-40, x))))


def fit(rows, streams):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["role"] != "fit":
            continue
        for i in range(3):
            f = features(row, streams, i)
            if f is not None:
                groups[row["group"]][bool(row["truth"] and i == 1)].append(f)
    complete = [r for r in groups.values() if set(r) == {False, True}]
    if len(complete) < 16:
        raise ValueError("Insufficient complete fit inputs")
    data = [(f, int(label), 1 / (len(complete) * 2 * len(values)))
            for group in complete for label, values in sorted(group.items()) for f in values]
    size = 5 * streams
    means = [math.fsum(f[i] * w for f, _, w in data) for i in range(size)]
    scales = [max(.1, math.sqrt(math.fsum((f[i]-means[i])**2*w for f, _, w in data))) for i in range(size)]
    inputs = [([1., *[(v - m)/s for v, m, s in zip(f, means, scales, strict=True)]], y, w) for f, y, w in data]
    beta = [0.] * (size + 1)
    for _ in range(100):
        errors = [(f, (sigmoid(math.fsum(x*y for x, y in zip(f, beta, strict=True))) - label) * w)
                  for f, label, w in inputs]
        beta = [b - .3 * (math.fsum(f[i]*e for f,e in errors) + (.1*b if i else 0)) for i,b in enumerate(beta)]
    return {"means": means, "scales": scales, "coefficients": beta}


def ranked(row, streams, params):
    scores = []
    coef = params["coefficients"]
    for i in range(3):
        v = features(row, streams, i)
        if v is None:
            return None
        scores.append(coef[0] + math.fsum(c * (x - m) / s for c, x, m, s in zip(
            coef[1:], v, params["means"], params["scales"], strict=True)))
    ordered = sorted(range(3), key=lambda i: scores[i], reverse=True)
    return (scores[ordered[0]], ordered[0] + 1) if scores[ordered[0]] != scores[ordered[1]] else None


def cutoffs(development):
    if any(row["role"] == "held" for row in development):
        raise ValueError("Held data is excluded from fitting and calibration")
    arms = {}
    for name, count in (("primary", 1), ("combined", 2)):
        head = fit(development, count)
        controls = [row for row in development if row["role"] == "calibration" and row["family"] == "original"]
        maxima = sorted(value[0] for row in controls if (value := ranked(row, count, head)) is not None)
        if len(maxima) < 24:
            raise ValueError("Insufficient assessable calibration works")
        arms[name] = head | {"point": maxima[-2], "calibration_groups": len(maxima)}
    return arms


def useful(row, proposed):
    if row["truth"] is None:
        return False
    ignore = set(row["spaces"])
    p = set(range(*proposed)) - ignore
    truth = set(range(*row["truth"])) - ignore
    overlap = len(p & truth)
    return 2*overlap >= len(truth) and 10*overlap >= 3*len(p|truth)


def report(rows, arms):
    summary = {}
    for genre in sorted({r["genre"] for r in rows if r["role"] == "held"}):
        counts = Counter()
        for r in rows:
            if r["role"] != "held" or r["genre"] != genre:
                continue
            base = [i for i, (score, _) in enumerate(r["responses"][:5]) if score is not None and score >= .50][:6]
            base_useful = any(useful(r, (r["spans"][max(0, i-1)][0], r["spans"][min(4, i+1)][1])) for i in base)
            ordinary = r["family"] == "original"
            counts["ordinary" if ordinary else "generated"] += 1
            counts["incumbent_ordinary"] += ordinary and bool(base)
            counts["incumbent_useful"] += base_useful
            for arm, streams in (("primary", 1), ("combined", 2)):
                winner = ranked(r, streams, arms[arm])
                added = winner is not None and winner[0] > arms[arm]["point"]
                accurate = added and useful(r, r["spans"][winner[1]])
                counts[f"{arm}/assessed"] += winner is not None
                counts[f"{arm}/ordinary"] += ordinary and added
                counts[f"{arm}/additional_ordinary"] += ordinary and added and not base
                counts[f"{arm}/accurate"] += accurate
                counts[f"{arm}/wrong"] += not ordinary and added and not accurate
                counts[f"{arm}/additional"] += accurate and not base_useful
                counts[f"{arm}/retained"] += accurate and base_useful
                if accurate and not base_useful:
                    counts[f"{arm}/gain/{r['family']}"] += 1
        summary[genre] = dict(counts)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "develop", "held", "review"))
    parser.add_argument("path", type=Path)
    for name in ("news", "abstracts", "essays"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "review":
        packet = json.loads(args.path.read_text())
        expected = cutoffs(packet["development"])
        if (packet["method"] != METHOD or packet["schema"] != SCHEMA or packet["parameters"] != expected
                or packet["result"] != report(packet["held"], expected)):
            raise ValueError("Saved study arithmetic does not replay")
        print(json.dumps(packet["result"], indent=2))
        return
    if args.mode == "freeze":
        cases, counts = prepare({name:getattr(args, name) for name in ("news", "abstracts", "essays")})
        marker = {"schema": SCHEMA, "method": METHOD, "admission": counts, "code_sha256": file_hash(Path(__file__)),
                  "model_receipt_sha256": file_hash(ROOT/HISTORIC)}
        with args.path.open("x") as stream:
            json.dump({"metadata":marker, "cases":cases}, stream, separators=(",", ":"))
        return
    import torch

    torch.set_num_threads(2)
    roster = json.loads(args.path.read_text())
    marker = roster["metadata"]
    if marker["code_sha256"] != file_hash(Path(__file__)) or marker["method"] != METHOD:
        raise ValueError("Frozen runner differs")
    if args.output is None:
        parser.error("provide a new output path")
    if args.mode == "develop":
        dev = list(measure(row for row in roster["cases"] if row["role"] != "held"))
        data = marker | {"freeze_sha256": file_hash(args.path), "development": dev, "parameters": cutoffs(dev)}
    else:
        dev_path = args.path.with_suffix(".development.json")
        dev = json.loads(dev_path.read_text())
        if dev["freeze_sha256"] != file_hash(args.path) or cutoffs(dev["development"]) != dev["parameters"]:
            raise ValueError("The frozen cutoff and inputs do not match")
        inputs = (row for row in roster["cases"] if row["role"] == "held")
        held = list(measure(inputs))
        data = dev | {"held": held, "result": report(held, dev["parameters"])}
        print(json.dumps(data["result"], indent=2), flush=True)
    with args.output.open("x") as stream:
        json.dump(data, stream, separators=(",", ":"))
        stream.write("\n")


if __name__ == "__main__":
    main()
