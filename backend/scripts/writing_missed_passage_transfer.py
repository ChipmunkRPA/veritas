"""Test exact-sentence likelihood as a second reading for missed writing edits."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from statistics import median

from scripts import writing_context_pool_transfer as pool

SCHEMA = "chipmunk.writing-missed-passage-transfer/1"
CONTEXT = pool.PANEL_DIR / "writing-context-pool-transfer-2026-11-14.json"
TRANSFER = pool.PANEL_DIR / "writing-focus-inside-context-2026-11-14.json"
FITS = ("pooled", "ratio", "joint")
POINT = 0.9505
MODELS = (
    ("Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323",
     "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796"),
    ("Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
     "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"),
)
TOKENIZER = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
METHOD = {
    "windows": "Read the unchanged complete five-sentence excerpt with the existing causal model pair on Apple "
    "GPU at serving precision. Reduce the observed loss and cross-entropy over tokens wholly attributable to "
    "each exact sentence. Require all its nonspace characters and at least 12 tokens. No neighbor expansion.",
    "comparison": "Measure the existing primary contexts at 0.50 and the existing ratio direction at 0.9505 "
    "on each eligible exact sentence as separate and union proxies. Compare only added hints in files with no "
    "native proxy flags. This is not the whole paid app or its calibrated confidence.",
    "methods": "Keep the prior combined pooling head. Fit a separate ratio-only and a joint ridge-logistic "
    "head on the previously declared source work fit split. The ratio arm reads negative ratio, its "
    "difference from the median of the other interior responses and negative mean observed surprisal. "
    "The joint version also reads the frozen pooled sentence score and its median contrast.",
    "points": "Use only ordinary previously declared calibration works. Record the number with no native flags. "
    "For each candidate use a strict score excess over the largest complete ordinary calibration work maximum, "
    "and over the second largest as a separate point. The same points apply to the silent subset.",
    "advance": "For an optional complement need at least four more useful locations across both generators "
    "on the first held set, at most two new genuine flags, and no more wrong than useful locations. "
    "The second held set must recover at least one additional useful location with at most one genuine "
    "addition. Preserve all existing source locations and report decisions.",
}


def read(path):
    return json.loads(path.read_text())


def manifest():
    return {"schema": SCHEMA, "method": METHOD, "script_sha256": pool.file_hash(Path(__file__)),
            "documents": {p.name: pool.file_hash(p) for p in (CONTEXT, TRANSFER)},
            "model_manifest_sha256": pool.file_hash(pool.ROOT / "backend/app/models.yaml")}


def prepare(main, transfer):
    frozen = read(CONTEXT)
    fresh = read(TRANSFER)
    if pool.file_hash(main) != frozen["input_roster_sha256"] or pool.file_hash(transfer) != fresh["roster_sha256"]:
        raise ValueError("Private source roster differs")
    indexed = {(r["group"], r["family"]): r for r in read(main)["rows"] + read(transfer)["rows"]}
    labels = [(r, r["role"]) for r in frozen["development"] + frozen["held"]]
    labels.extend((r, "transfer") for r in fresh["rows"])
    for row, role in labels:
        source = indexed[row["group"], row["family"]]
        if (pool.sha(source["text"]) != row["sha256"]
                or any(source[key] != row[key] for key in ("spans", "spaces", "truth"))):
            raise ValueError("Private source differs from frozen positions")
        yield {"response": row, "role": role, "text": source["text"]}


def load_pair():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from app.detectors.text.binoculars import Binoculars

    if not torch.backends.mps.is_available():
        raise ValueError("Apple GPU is required for this pinned profile")
    torch.set_num_threads(2)
    snapshots = []
    for name, revision, weights in MODELS:
        path = pool.ROOT / f"models/hf/hub/models--Qwen--{name}/snapshots/{revision}"
        if pool.file_hash(path / "model.safetensors") != weights or pool.file_hash(path / "tokenizer.json") != TOKENIZER:
            raise ValueError("Pinned model or tokenizer bytes differ")
        snapshots.append(path)
    tokenizer = AutoTokenizer.from_pretrained(snapshots[0], local_files_only=True, trust_remote_code=False)
    alternate = AutoTokenizer.from_pretrained(snapshots[1], local_files_only=True, trust_remote_code=False)
    if tokenizer.get_vocab() != alternate.get_vocab():
        raise ValueError("Model vocabularies differ")
    models = [AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=torch.float16,
    ).to("mps").eval() for path in snapshots]
    return Binoculars.from_parts(tokenizer, *models, torch.device("mps")), alternate


def reduce_tokens(lane, alternate, text, spans):
    from app.core.canonical import canonical_hash

    scores = lane.token_scores(text)
    counterpart = alternate(text, return_offsets_mapping=True, add_special_tokens=False)
    if counterpart["input_ids"] != scores.source_token_ids or list(map(tuple, counterpart["offset_mapping"])) != scores.offsets:
        raise ValueError("The model pair reads different source tokens")
    masks = scores.valid_mask()
    reading = []
    for start, end in spans:
        target = set(i for i in range(start, end) if not text[i].isspace())
        selected, covered = [], set()
        for i, (a, b) in enumerate(scores.offsets):
            if not masks[i] or b <= start or a >= end:
                continue
            positions = {p for p in range(a, b) if not text[p].isspace()}
            if positions and positions <= target:
                selected.append(i)
                covered.update(positions)
        if covered != target or len(selected) < 12:
            reading.append(None)
            continue
        obs = float(scores.obs_nll[selected].mean())
        cross = float(scores.xent[selected].mean())
        if not math.isfinite(obs) or not math.isfinite(cross) or cross <= 0:
            raise ValueError("Invalid sentence losses")
        reading.append({"obs_hex": obs.hex(), "cross_hex": cross.hex(), "tokens": len(selected),
                        "offset_sha256": canonical_hash([scores.offsets[i] for i in selected])})
    return reading


def ratio_values(row):
    return [
        float.fromhex(item["obs_hex"]) / float.fromhex(item["cross_hex"]) if item else None
        for item in row["ratio"]
    ]


def measured_spans(row):
    ratios = ratio_values(row)
    original = pool.baseline(row)
    return original + [span for v, span in zip(ratios, row["spans"], strict=True) if v is not None and v < POINT]


def vectors(row, name, pooled):
    ratios = ratio_values(row)[1:4]
    if any(value is None for value in ratios):
        return None
    text_features = pool.features(row, "combined")
    if text_features is None:
        return None
    b = pooled["coefficients"]
    pool_scores = [b[0] + math.fsum(c * (x - m) / s for c, x, m, s in zip(
        b[1:], entry, pooled["mean"], pooled["scale"], strict=True,
    )) for entry in text_features]
    if name == "pooled":
        return [[q] for q in pool_scores]
    raw = [[-q, median(ratios) - q, -float.fromhex(item["obs_hex"])] for q, item in
           zip(ratios, row["ratio"][1:4], strict=True)]
    if name == "joint":
        center = median(pool_scores)
        return [values + [q, q - center] for values, q in zip(raw, pool_scores, strict=True)]
    return raw


def ranked(row, name, pooled, coefficients):
    features = vectors(row, name, pooled)
    if features is None:
        return None
    if name == "pooled":
        values = [v[0] for v in features]
    else:
        b = coefficients["beta"]
        values = [b[0] + math.fsum(c * (x - m) / s for c, x, m, s in zip(
            b[1:], f, coefficients["mean"], coefficients["sd"], strict=True,
        )) for f in features]
    order = sorted(range(3), key=values.__getitem__, reverse=True)
    return (values[order[0]], order[0] + 1) if values[order[0]] != values[order[1]] else None


def fit(rows, name, pooled):
    complete = [rs for rs in pool.group_rows(rows).values() if all(vectors(r, name, pooled) is not None for r in rs)]
    if len(complete) < 20:
        raise ValueError("Not enough source-balanced fit examples")
    data = []
    for group in complete:
        items = [(values, int(row["truth"] == span)) for row in group
                 for values, span in zip(vectors(row, name, pooled), row["spans"][1:4], strict=True)]
        classes = Counter(label for _, label in items)
        data.extend((v, label, 1 / (2 * len(complete) * classes[label])) for v, label in items)
    size = len(data[0][0])
    means = [math.fsum(v[i] * w for v, _, w in data) for i in range(size)]
    scales = [max(.1, math.sqrt(math.fsum((v[i] - means[i])**2 * w for v, _, w in data))) for i in range(size)]
    work = [([1, *[(x - m) / s for x, m, s in zip(entry, means, scales, strict=True)]], label, w)
            for entry, label, w in data]
    coefficients = [0.] * (size + 1)
    for _ in range(120):
        errors = []
        for vector, label, weight in work:
            logit = math.fsum(v * c for v, c in zip(vector, coefficients, strict=True))
            errors.append((vector, (1 / (1 + math.exp(-min(40, max(-40, logit)))) - label) * weight))
        coefficients = [c - .3 * (math.fsum(v[i] * e for v, e in errors) + (.15 * c if i else 0))
                        for i, c in enumerate(coefficients)]
    return {"mean": means, "sd": scales, "beta": coefficients, "fit_groups": len(complete)}


def combine(readings):
    """Join numeric causal results to existing text classifier scores."""
    primary, other = read(CONTEXT), read(TRANSFER)
    cases = {(r["role"], r["group"], r["family"]): r for r in primary["development"] + primary["held"]}
    cases.update({("transfer", r["group"], r["family"]): r for r in other["rows"]})
    for observed in readings:
        key = (observed["role"], observed["group"], observed["family"])
        row = cases[key]
        if observed["sha256"] != row["sha256"] or len(observed["ratio"]) != len(row["spans"]):
            raise ValueError("Invalid causal response commitment")
        yield row | observed


def parameters(rows, pooled):
    controls = [r for r in rows if r["role"] == "calibration" and not r["truth"]]
    fit_rows = [r for r in rows if r["role"] == "fit"]
    heads = {}
    for name in FITS:
        head = {} if name == "pooled" else fit(fit_rows, name, pooled)
        maxima = sorted(v[0] for r in controls if (v := ranked(r, name, pooled, head)) is not None)
        if len(maxima) < 8:
            raise ValueError("Insufficient complete calibration controls")
        heads[name] = head | {"strict": maxima[-1], "one_control": maxima[-2], "control_count": len(maxima),
                              "unflagged_control_count": sum(not measured_spans(r) for r in controls)}
    return heads


def summarize(rows, pooled, heads):
    report = {}
    for role in ("held", "transfer"):
        for domain in (*pool.DOMAINS, "all"):
            counts = Counter()
            for r in rows:
                if r["role"] != role or domain not in (r["domain"], "all"):
                    continue
                truth = r["truth"]
                base = measured_spans(r)
                counts["genuine" if truth is None else "edited"] += 1
                counts["base_genuine_flags"] += truth is None and bool(base)
                counts["base_useful"] += any(pool.useful(r, span) for span in base)
                counts["ratio_eligible"] += all(v is not None for v in ratio_values(r)[1:4])
                if base:
                    continue
                counts["silent_genuine" if truth is None else "silent_edited"] += 1
                for name, head in heads.items():
                    winner = ranked(r, name, pooled, head)
                    if winner is None:
                        continue
                    for point in ("strict", "one_control"):
                        if winner[0] <= head[point]:
                            continue
                        prefix = f"{name}/{point}"
                        result = "false" if truth is None else "useful" if r["spans"][winner[1]] == truth else "wrong"
                        counts[f"{prefix}/{result}"] += 1
                        if result == "useful":
                            counts[f"{prefix}/{r['family']}"] += 1
            report[f"{role}/{domain}"] = dict(counts)
    return report


def load_numeric(path):
    packet = read(path)
    if packet["definition"] != manifest():
        raise ValueError("Definition or benchmark changed")
    identity = {(r["role"], r["group"], r["family"]) for r in packet["readings"]}
    if len(identity) != len(packet["readings"]):
        raise ValueError("Duplicate causal readings")
    return packet, list(combine(packet["readings"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("develop", "held", "review"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--context-roster", type=Path)
    parser.add_argument("--transfer-roster", type=Path)
    parser.add_argument("--development", type=Path)
    args = parser.parse_args()
    definition = manifest()
    pooled = read(CONTEXT)["parameters"]["combined"]
    if args.mode == "review":
        packet, rows = load_numeric(args.output)
        heads = parameters(rows, pooled)
        if heads != packet["heads"] or summarize(rows, pooled, heads) != packet["result"]:
            raise ValueError("Published held arithmetic differs")
        print(json.dumps(packet["result"], indent=2))
        return
    if any(os.environ.get(v) != "1" for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        parser.error("Run offline")
    readings = []
    if args.mode == "held":
        dev, fit_rows = load_numeric(args.development)
        if dev["heads"] != parameters(fit_rows, pooled):
            raise ValueError("Frozen development reading differs")
        readings.extend(dev["readings"])
    selected = [row for row in prepare(args.context_roster, args.transfer_roster)
                if (row["role"] in ("fit", "calibration")) == (args.mode == "develop")]
    lane, alternate = load_pair()
    cache = {}
    for i, item in enumerate(selected, 1):
        row = item["response"]
        if row["sha256"] not in cache:
            cache[row["sha256"]] = reduce_tokens(lane, alternate, item["text"], row["spans"])
        readings.append({k: row[k] for k in ("family", "group", "sha256")}
                        | {"role": item["role"], "ratio": cache[row["sha256"]]})
        if i % 16 == 0 or i == len(selected):
            print(f"{args.mode} {i}/{len(selected)}", flush=True)
    rows = list(combine(readings))
    heads = parameters(rows, pooled)
    data = {"definition": definition, "readings": readings, "heads": heads}
    if args.mode == "held":
        data.update(development_sha256=pool.file_hash(args.development), result=summarize(rows, pooled, heads))
        print(json.dumps(data["result"], indent=2))
    pool.write(args.output, data)


if __name__ == "__main__":
    main()
