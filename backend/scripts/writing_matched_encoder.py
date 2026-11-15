"""Fine-tune a compact local encoder on exact sentence labels and matched rewrites."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_context_pool_transfer as pool
from scripts import writing_focus_inside_context as intake
from scripts import writing_sentence_probe as previous
from scripts import writing_topic_response_transfer as shipped
from scripts.writing_interval_transfer import fingerprints

SCHEMA = "chipmunk.writing-matched-encoder/1"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
SOURCES = (
    ("fit", "train", "gemini-2.5-flash", 480, "c4af2313124b86f5003f6635bf451960f6cbd015a849fb33cd35935c89cf778e"),
    ("fit", "train", "gpt-5.4-nano", 480, "c0b0189df99b6be49f4917dd3db32e268deddd366a641c337f3a0e07789f2d9e"),
    ("calibration", "dev", "gpt-5.4-nano", 72, "d62fb56d81ee6e4d0ce34a116770b86af910308fd7102bef5e79036dd87e9767"),
    ("held", "test", "qwen3-8b", 11, previous.REPORT_PINS[0]),
    ("held", "test", "gpt-5.4", 60, previous.REPORT_PINS[1]),
)
LOCAL_FAMILY = "local-qwen2.5-1.5b"
ENCODER_ASSETS = {
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "model.safetensors": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "tokenizer_config.json": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
PRIOR = (shipped.PARENT, shipped.WORKER, pool.PANEL_DIR / "writing-topic-response-transfer-2026-11-14.json")
MAX_TOKENS = 384
BATCH_PAIRS = 8
EPOCHS = 4
SEED = 131
METHOD = {
    "intake": "Use publisher train and dev government-report shards for fitting and setting the points. Hold out new "
              "test works and two previously unseen generators. Compare dated real journal revisions using the earlier full worker.",
    "focus": "One five-sentence source paragraph excerpt with exactly one annotated and materially changed interior "
             "sentence. Preserve four exact native neighbors and independent source groups. A source pair shares the edit position.",
    "model": "Fine-tune the pinned six-layer encoder and linear head for four epochs. Each interior vector combines its "
             "mean token representation and its difference from the mean of its two neighbors. Use complete source-character coverage.",
    "fit": "AdamW at 3e-5, eight matched source pairs per batch and seed 131. Combine five-times positive-weighted "
           "sentence cross-entropy and 0.4 times a softplus margin comparing the same sentence in its original and rewritten form.",
    "point": "Calibration uses maxima from genuine dev works only. The two points allow at most ten or twenty-five percent "
             "of these works to be flagged. The held file contributes only the submitted text. At most one exact sentence is selected.",
    "baseline": "Replay the existing passage selection rule using the two serving measurements. Report the matched journal "
                "worker output separately. A correct candidate range matches the exact changed sentence.",
}


def contract():
    return {"schema": SCHEMA, "method": METHOD, "revision": REVISION,
            "initial_assets": sorted([name, digest] for name, digest in ENCODER_ASSETS.items()),
            "prior": {p.name: pool.file_hash(p) for p in PRIOR},
            "source_sha256": {f"{split}-reports_{family}.csv": sha for _, split, family, _, sha in SOURCES}}


def load_shard(path, split, family, pin):
    if pool.file_hash(path) != pin:
        raise ValueError("Pinned dataset file differs")
    work = defaultdict(dict)
    csv.field_size_limit(1024 * 1024)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, strict=True):
            if (row["domain"], row["split"], row["generator"], row["subset"]) != ("reports", split, family, "main"):
                raise ValueError("Unexpected shard metadata")
            version = row["version"]
            if version not in ("v0", *pool.VARIANTS):
                continue
            versions = work[row["id"]]
            if version in versions:
                raise ValueError("Repeated work and version")
            versions[version] = {key: row[key] for key in ("text", "text_tagged", "ai_spans_char")}
    return work


def freeze(shards, parent_roster, topic_roster, uploads):
    inputs = (parent_roster, topic_roster)
    if tuple(map(pool.file_hash, inputs)) != tuple(intake.read(p)["roster_sha256"] for p in (PRIOR[0], PRIOR[2])):
        raise ValueError("The exclusion rosters differ from the published commitments")
    earlier = [row for path in inputs for row in intake.read(path)["rows"]]
    exposed = {row["group"] for row in earlier}
    for path in pool.PANEL_DIR.glob("writing-*-2026-11-14.json"):
        if not path.name.startswith("writing-matched-encoder-"):
            exposed.update(previous.SHA256.findall(path.read_text()))
    used = {piece for row in earlier for piece in fingerprints(row["text"])}
    rows = []
    admitted = {}
    # Reserve untouched test sources and their phrases before preparing train and calibration pairs.
    for role, split, family, target, pin in sorted(SOURCES, key=lambda item: {"held": 0, "calibration": 1, "fit": 2}[item[0]]):
        table = load_shard(shards / f"{split}-reports_{family}.csv", split, family, pin)
        for key, versions in table.items():
            if pool.sha(key) in exposed and "v0" in versions:
                used.update(fingerprints(versions["v0"]["text"]))
        coverage = Counter()
        operation = Counter()
        for key in sorted(table, key=lambda item: pool.sha(f"{SCHEMA}/{role}/{family}/{item}")):
            if sum(coverage.values()) == target:
                break
            versions = table[key]
            group = pool.sha(key)
            if group in exposed or "v0" not in versions:
                continue
            pieces = fingerprints(versions["v0"]["text"])
            if pieces & used:
                continue
            try:
                variants = intake.excerpt_options(key, versions)
            except (KeyError, ValueError):
                continue
            if not variants:
                continue
            position = min(variants, key=lambda i: (coverage[i], i))
            for (text, spans, variant), label in zip(variants[position], ("original", family), strict=True):
                if role == "calibration" and variant:
                    continue
                rows.append({"role": role, "domain": "reports", "family": label, "pair_family": family,
                             "group": group, "text": text, "sha256": pool.sha(text), "spans": spans, "variant": variant,
                             "spaces": [i for i, c in enumerate(text) if c.isspace()],
                             "truth": spans[position] if variant else None, "position": position})
                if variant:
                    operation[variant] += 1
            coverage[position] += 1
            exposed.add(group)
            used.update(pieces)
        admitted[f"{role}/{family}"] = {"groups": sum(coverage.values()), "positions": dict(coverage), "versions": dict(operation)}
        print(role, family, admitted[f"{role}/{family}"], flush=True)
    if sum(cell["groups"] for key, cell in admitted.items() if key.startswith("fit/")) < 600:
        raise ValueError("The study needs at least 600 distinct fit works")
    if admitted["held/gpt-5.4"]["groups"] < 40:
        raise ValueError("Too few new test works for the primary held generator")
    rows.extend(previous.editorial_rows(uploads))
    return contract() | {"admitted": admitted, "rows": rows}


def local_seeds(data, reports, past, needed=42):
    table = load_shard(reports, "test", "gpt-5.4", previous.REPORT_PINS[1])
    excluded = {r["group"] for r in data["rows"] + past}
    phrases = {piece for r in data["rows"] + past for piece in fingerprints(r["text"])}
    for key, versions in table.items():
        if "v0" in versions and pool.sha(key) in excluded:
            phrases.update(fingerprints(versions["v0"]["text"]))
    seeds = []
    for key in sorted(table, key=lambda item: pool.sha(f"{SCHEMA}/{LOCAL_FAMILY}/{item}")):
        if len(seeds) == needed:
            break
        group = pool.sha(key)
        if group in excluded or "v0" not in table[key]:
            continue
        raw, labels = pool.truth_text(table[key]["v0"])
        parts = fingerprints(raw)
        if labels or parts & phrases:
            continue
        position = len(seeds) % 3 + 1
        windows = sorted(pool.source_windows(raw), key=lambda s: pool.sha(f"{key}/{s[0][0]}"))
        for window in windows:
            focus = raw[slice(*window[position])]
            if not 14 <= len(focus.split()) <= 44:
                continue
            try:
                text, spans = pool.normalize(raw, window)
            except ValueError:
                continue
            seeds.append({"role": "held", "domain": "reports", "group": group, "pair_family": LOCAL_FAMILY,
                          "family": "original", "text": text, "position": position, "sha256": pool.sha(text),
                          "spans": spans, "spaces": [i for i, c in enumerate(text) if c.isspace()],
                          "variant": None, "truth": None})
            phrases.update(parts)
            break
    if len(seeds) != needed:
        raise ValueError("Insufficient independent seed works for the local generator")
    return seeds


def local_rewrites(seeds):
    import re

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    name, revision, weight_hash = shipped.causal.MODELS[1]
    folder = pool.ROOT / f"models/hf/hub/models--Qwen--{name}/snapshots/{revision}"
    if pool.file_hash(folder / "model.safetensors") != weight_hash or pool.file_hash(folder / "tokenizer.json") != shipped.causal.TOKENIZER:
        raise ValueError("The local rewrite checkpoint differs")
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True, dtype=dtype).eval().to(device)
    for offset in range(0, len(seeds), 4):
        batch = seeds[offset:offset + 4]
        targets = [r["text"][slice(*r["spans"][r["position"]])] for r in batch]
        actions = ["Lightly polish" if (offset + i) % 2 == 0 else "Paraphrase" for i in range(len(batch))]
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content":
            f"{action} the following sentence of a formal report. Preserve all facts, names, and numbers. "
            f"Output one complete sentence, with no labels or extra commentary.\n\n{text}"}],
            tokenize=False, add_generation_prompt=True) for action, text in zip(actions, targets, strict=True)]
        encoded = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=90, use_cache=True)
        completions = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        for row, action, target, completion in zip(batch, actions, targets, completions, strict=True):
            rewritten = completion.strip().strip('"“”')
            if (not rewritten or "\n" in rewritten or rewritten == target
                    or re.findall(r"\d[\d.,]*", rewritten) != re.findall(r"\d[\d.,]*", target)
                    or not .5 <= len(rewritten.split()) / len(target.split()) <= 1.8):
                continue
            a, b = row["spans"][row["position"]]
            changed = from_string(row["text"][:a] + rewritten + row["text"][b:])
            sentences = segment(changed)
            if len(sentences) != 5 or any(s.paragraph != 0 for s in sentences):
                continue
            spans = [[s.start, s.end] for s in sentences]
            same_neighbors = all(s.text == row["text"][slice(*row["spans"][i])]
                                 for i, s in enumerate(sentences) if i != row["position"])
            if not same_neighbors or sentences[row["position"]].text == target:
                continue
            yield row, row | {"family": LOCAL_FAMILY, "variant": action, "text": changed.text, "spans": spans,
                              "sha256": pool.sha(changed.text), "truth": spans[row["position"]],
                              "spaces": [i for i, c in enumerate(changed.text) if c.isspace()]}


def extend(data, reports, old, topic):
    paths = (old, topic)
    if tuple(map(pool.file_hash, paths)) != tuple(intake.read(p)["roster_sha256"] for p in (PRIOR[0], PRIOR[2])):
        raise ValueError("Local generation exclusion rosters differ")
    seeds = local_seeds(data, reports, [r for path in paths for r in intake.read(path)["rows"]])
    pairs = list(local_rewrites(seeds))
    if not pairs:
        raise ValueError("No generated cases met the fixed source checks")
    data["rows"].extend(row for pair in pairs for row in pair)
    data["admitted"][f"held/{LOCAL_FAMILY}"] = {"groups": len(pairs),
        "positions": dict(Counter(r["position"] for r, _ in pairs)),
        "versions": dict(Counter(r["variant"] for _, r in pairs))}
    print(f"Admitted {len(pairs)} of {len(seeds)} independent local rewrites", flush=True)
    return data | {"local_generator_revision": shipped.causal.MODELS[1][1], "local_seed_count": len(seeds)}


def encode(row, tokenizer):
    encoded = tokenizer(row["text"], return_offsets_mapping=True, truncation=False)
    ids = encoded["input_ids"]
    if len(ids) > MAX_TOKENS:
        return None
    groups = previous.token_groups(row["text"], row["spans"], encoded["offset_mapping"], encoded["attention_mask"])
    if groups is None:
        return None
    return ids, groups


def forward(model, head, batch, device):
    import torch

    width = max(len(ids) for ids, _ in batch)
    ids = torch.zeros((len(batch), width), dtype=torch.long)
    mask = torch.zeros_like(ids)
    spans = torch.zeros((len(batch), 5, width), dtype=torch.float32)
    for index, (tokens, groups) in enumerate(batch):
        ids[index, :len(tokens)] = torch.tensor(tokens)
        mask[index, :len(tokens)] = 1
        for position, group in enumerate(groups):
            spans[index, position, group] = 1 / len(group)
    result = model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
    means = spans.to(device).bmm(result)
    center = means[:, 1:4]
    relative = center - (means[:, :3] + means[:, 2:]) * .5
    return head(torch.cat((center, relative), dim=-1)).squeeze(-1)


def init_encoder(folder):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if any(pool.file_hash(folder / name) != digest for name, digest in ENCODER_ASSETS.items()):
        raise ValueError("The base encoder or tokenizer differs")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id != 0 or not tokenizer.is_fast:
        raise ValueError("The installed encoder tokenizer differs")
    model = AutoModel.from_pretrained(folder, local_files_only=True, trust_remote_code=False,
                                     use_safetensors=True, attn_implementation="eager").to(device)
    if model.config.num_hidden_layers != 6 or model.config.hidden_size != 384:
        raise ValueError("The encoder shape differs from the pinned revision")
    head = torch.nn.Linear(model.config.hidden_size * 2, 1).to(device)
    return tokenizer, model, head, device


def train(data, encoder, directory):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rng = random.Random(SEED)
    tokenizer, model, head, device = init_encoder(encoder)
    groups = defaultdict(list)
    for row in data["rows"]:
        if row["role"] == "fit":
            groups[row["group"]].append(row)
    examples = []
    for base, edit in groups.values():
        left, right = encode(base, tokenizer), encode(edit, tokenizer)
        if left and right:
            examples.append((left, right, edit["position"] - 1))
    if len(examples) < 600:
        raise ValueError("Fewer than 600 source pairs have exact token coverage")
    print(f"Training on {device} with {len(examples)} complete pairs", flush=True)
    optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=3e-5)
    model.train()
    history = []
    for epoch in range(EPOCHS):
        rng.shuffle(examples)
        current = 0.
        for start in range(0, len(examples), BATCH_PAIRS):
            items = examples[start:start + BATCH_PAIRS]
            lengths = len(items)
            requests = [case for before, after, _ in items for case in (before, after)]
            positions = torch.tensor([position for _, _, position in items], device=device)
            scores = forward(model, head, requests, device).reshape(lengths, 2, 3)
            labels = torch.zeros_like(scores)
            labels[torch.arange(lengths, device=device), 1, positions] = 1.
            target_delta = (scores[:, 1] - scores[:, 0]).gather(1, positions[:, None])
            loss = F.binary_cross_entropy_with_logits(scores, labels, pos_weight=torch.tensor(5., device=device))
            loss = loss + .4 * F.softplus(1. - target_delta).mean()
            if not torch.isfinite(loss):
                raise ValueError("The fitted objective is not finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*model.parameters(), *head.parameters()], 1.)
            optimizer.step()
            current += loss.item() * lengths
        history.append(round(current / len(examples), 6))
        print(f"Epoch {epoch + 1}/{EPOCHS} loss {history[-1]:.6f}", flush=True)
    save_file({f"encoder.{key}": value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
              | {f"head.{key}": value.detach().cpu().contiguous() for key, value in head.state_dict().items()},
              directory / "trained.safetensors")
    return tokenizer, model, head, device, len(examples), history


def predict(data, tokenizer, model, head, device):
    import torch

    model.eval()
    records = []
    rows = [row for row in data["rows"] if row["role"] != "fit"]
    for start in range(0, len(rows), 2 * BATCH_PAIRS):
        batch = rows[start:start + 2 * BATCH_PAIRS]
        encoded = [encode(row, tokenizer) for row in batch]
        qualified = [item for item in encoded if item is not None]
        with torch.inference_mode():
            measured = iter(forward(model, head, qualified, device).cpu().tolist() if qualified else [])
        for row, item in zip(batch, encoded, strict=True):
            scores = None if item is None else [round(score, 7) for score in next(measured)]
            records.append(shipped.compact(row) | {"scores": scores})
        if next(measured, None) is not None:
            raise ValueError("Response count differs from the admitted source rows")
    return records


def operating_points(rows):
    ordinary = sorted((max(r["scores"]) for r in rows if r["role"] == "calibration" and r["scores"] is not None), reverse=True)
    if len(ordinary) < 64:
        raise ValueError("A substantial independent genuine calibration set is required")
    return {label: ordinary[len(ordinary) // denominator] for label, denominator in (("10pct", 10), ("25pct", 4))}


def fit_and_measure(path, encoder, out):
    out.mkdir(mode=0o700, exist_ok=False)
    data = intake.read(path)
    if any(data[key] != val for key, val in contract().items()):
        raise ValueError("Frozen roster changed")
    tokenizer, model, head, device, count, history = train(data, encoder, out)
    records = predict(data, tokenizer, model, head, device)
    previous.write(out / "measurements.json", contract() | {"admitted": data["admitted"], "roster_sha256": pool.file_hash(path),
                   "local_generator_revision": data.get("local_generator_revision"), "local_seed_count": data.get("local_seed_count"),
                   "trained_sha256": pool.file_hash(out / "trained.safetensors"),
                   "complete_fit_groups": count, "loss": history, "device": device, "points": operating_points(records), "rows": records})


def outcomes(rows, points):
    journal = {(pool.sha(row["group"]), row["role"]): row for row in intake.read(previous.EDITORIAL)["rows"]}
    results = {}
    for row in rows:
        truth = row["truth"]
        genuine = truth is None
        category = (row["family"] if row["role"] == "editorial" else "ordinary" if genuine else "rewritten")
        scope = row.get("pair_family", "journal")
        table = results.setdefault(f"{scope}/{category}", Counter())
        if row["role"] == "editorial":
            past = journal[row["group"], row["family"]]
            if past["text_sha256"] != row["sha256"]:
                raise ValueError("The journal source binding differs")
            ranges = [span for card in past["cards"] for span in card["spans"]]
        else:
            ranges = shipped.native_range(row, None)
        baseline = bool(ranges) if genuine else previous.editorial.covers(ranges, truth, set(row["spaces"]))
        scores = row["scores"]
        index = scores.index(max(scores)) + 1 if scores and scores.count(max(scores)) == 1 else None
        table.update(total=1, assessed=int(scores is not None), baseline=int(baseline),
                     top_rank=int(index is not None and not genuine and row["spans"][index] == truth))
        for name, point in points.items():
            selected = index is not None and scores[index - 1] > point
            located = selected and (genuine or row["spans"][index] == truth)
            table[name] += located
            table[name + "_union"] += baseline or located
            table[name + "_wrong"] += selected and not genuine and not located
    return results


def readout(roster, measurements, classifier, ratio, output):
    observed = intake.read(measurements)
    clf, raw = (intake.read(p) for p in (classifier, ratio))
    pin = pool.file_hash(roster)
    if any(r["roster_sha256"] != pin for r in (observed, clf, raw)):
        raise ValueError("Evaluation phases used different inputs")
    result = observed | {"rows": [
        r | {"classifier": clf["scores"][r["sha256"]], "ratio": raw["scores"][r["sha256"]]} if r["role"] == "held" else r
        for r in observed["rows"]
    ]}
    result["results"] = outcomes([r for r in result["rows"] if r["role"] != "calibration"], observed["points"])
    audit(result)
    previous.write(output, result)


def audit(receipt):
    if any(receipt[key] != value for key, value in contract().items()):
        raise ValueError("Study commitments differ")
    rows = receipt["rows"]
    counts = Counter(r["role"] for r in rows)
    if counts["editorial"] != 74 or receipt["complete_fit_groups"] < 600:
        raise ValueError("Incomplete recorded fit or editorial validation")
    groups = defaultdict(list)
    for row in rows:
        spans = row["spans"]
        if (len(spans) != 5 or not previous.SHA256.fullmatch(row["sha256"])
                or any(not 0 <= a < b for a, b in spans)
                or any(a[1] > b[0] for a, b in zip(spans, spans[1:]))
                or row["truth"] is not None and row["truth"] not in spans[1:4]
                or row["scores"] is not None and (len(row["scores"]) != 3 or not all(map(math.isfinite, row["scores"])))):
            raise ValueError("Invalid source position or model response")
        if row["role"] != "editorial":
            groups[row["group"]].append(row)
    for deliveries in groups.values():
        if len({(r["role"], r["pair_family"]) for r in deliveries}) != 1 or sum(r["truth"] is None for r in deliveries) != 1:
            raise ValueError("Source groups overlap or have inconsistent truth")
        expected = 2 if deliveries[0]["role"] == "held" else 1
        if len(deliveries) != expected:
            raise ValueError("Missing or repeated source deliveries")
    group_counts = Counter(f"{group[0]['role']}/{group[0]['pair_family']}" for group in groups.values())
    if group_counts != {key: cell["groups"] for key, cell in receipt["admitted"].items() if not key.startswith("fit/") and cell["groups"]}:
        raise ValueError("Source group roster differs from the recorded intake")
    if receipt["points"] != operating_points(rows):
        raise ValueError("Frozen calibration changes under replay")
    result = outcomes([row for row in rows if row["role"] != "calibration"], receipt["points"])
    if result != receipt["results"]:
        raise ValueError("The recorded and recomputed outcomes differ")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "local", "train", "analyze", "review"))
    parser.add_argument("path", type=Path)
    for name in ("shards", "old", "topic", "uploads", "reports", "encoder", "output", "measurements", "classifier", "ratio"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    required = {"freeze": ("shards", "old", "topic", "uploads"), "local": ("reports", "old", "topic", "output"),
                "train": ("encoder", "output"), "analyze": ("measurements", "classifier", "ratio", "output")}
    missing = [name for name in required.get(args.phase, ()) if getattr(args, name) is None]
    if missing:
        parser.error("Missing required options " + ", ".join(f"--{name}" for name in missing))
    if args.phase == "freeze":
        previous.write(args.path, freeze(args.shards, args.old, args.topic, args.uploads))
    elif args.phase == "local":
        previous.write(args.output, extend(intake.read(args.path), args.reports, args.old, args.topic))
    elif args.phase == "train":
        fit_and_measure(args.path, args.encoder, args.output)
    elif args.phase == "analyze":
        readout(args.path, args.measurements, args.classifier, args.ratio, args.output)
    else:
        print(json.dumps(audit(intake.read(args.path)), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
