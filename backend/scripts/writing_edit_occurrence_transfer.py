# SPDX-License-Identifier: MIT
"""Test whether ordinary editorial changes help disambiguate AI rewrite locations."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_local_word_spans as spans
from scripts import writing_word_localization as prior

SCHEMA = "chipmunk.writing-edit-occurrence-transfer/1"
FAMILIES = ("gemini-2.5-flash", "gpt-5.4-nano")
ARMS = ("direct", "edit_and_origin")
FIT_PER_FAMILY = 160
SEED = 171
TRANSFER_SCHEMA = "chipmunk.writing-edit-occurrence-ordinary/1"
RULE = {"fit_pairs_per_generator": FIT_PER_FAMILY, "seed": SEED, "epochs": 12,
        "context": "Frozen prior encoder. Concatenate lexical-word state, its difference from its sentence, and the sentence's difference from its two neighbors.",
        "direct": "Single classifier for AI-replaced words. Known genuine edits are negative examples.",
        "edit_and_origin": "Fit one head for a lexical change and a second, on changed words, for generated versus ordinary revision. Multiply responses at inference.",
        "limits": "Five sentences, scored on three interior sentences. Score each submitted excerpt alone. Reuse the previous top-six sentence and top-60-percent word selection. The small genuine revision sample makes this a feasibility experiment."}


def fit_roster(rows):
    selected = set()
    for family in FAMILIES:
        groups = sorted((row["group"] for row in rows if row["role"] == "fit" and row["family"] == family),
                        key=lambda group: prior.parent.pool.sha(f"{SCHEMA}/{group}"))
        if len(groups) < FIT_PER_FAMILY:
            raise ValueError("The source family lacks the frozen fit group count")
        selected.update(groups[:FIT_PER_FAMILY])
    return [row for row in rows if row["role"] == "fit" and row["group"] in selected
            or row["role"] == "editorial" and row["family"] in {"before", "edited"}]


def feature_batches(rows, tokenizer, model, device):
    import torch

    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            batch = rows[start:start + 16]
            encoded = [prior.encode(row, tokenizer) for row in batch]
            width = max((len(item[0]) for item in encoded if item), default=0)
            ids = torch.zeros((len(batch), width), dtype=torch.long)
            attention = torch.zeros_like(ids)
            for i, item in enumerate(encoded):
                if item:
                    sequence = item[0]
                    ids[i, :len(sequence)] = torch.tensor(sequence)
                    attention[i, :len(sequence)] = 1
            if width:
                hidden = model(input_ids=ids.to(device), attention_mask=attention.to(device)).last_hidden_state.cpu().float()
            for i, (row, item) in enumerate(zip(batch, encoded, strict=True)):
                if not item:
                    yield row, None
                    continue
                _, word_ids, words = item
                subwords = [j for j, index in enumerate(word_ids) if index is not None]
                index = torch.tensor([word_ids[j] for j in subwords])
                count = torch.bincount(index).unsqueeze(1)
                states = hidden[i].new_zeros((len(words), hidden.shape[-1])).index_add_(0, index, hidden[i, subwords]) / count
                parts = torch.tensor([word[2] for word in words])
                centers = torch.stack([states[parts == part].mean(0) for part in range(5)])
                context = centers.clone()
                context[1:4] -= (centers[:3] + centers[2:]) / 2
                useful = [i for i, word in enumerate(words) if 1 <= word[2] <= 3]
                features = torch.cat((states[useful], states[useful] - centers[parts[useful]], context[parts[useful]]), dim=1)
                yield row, (words, useful, features)

def fit(examples, truth):
    import torch
    from torch import nn
    from torch.nn.functional import binary_cross_entropy_with_logits as bce

    groups = defaultdict(list)
    for row, example in examples:
        groups[row["role"], row["group"]].append((row, example))
    x, y, populations, revised_groups = [], [], Counter(), set()
    for pair in groups.values():
        if len(pair) != 2 or not all(example is not None for _, example in pair):
            continue
        for row, (_, useful, features) in pair:
            changed = truth.get(row["sha256"], set())
            kind = 2 if row["family"] == "edited" else 1
            labels = [kind if i in changed else 0 for i in useful]
            if 2 in labels:
                revised_groups.add(row["group"])
            x.append(features.clone())
            y.extend(labels)
            populations.update(labels)
    if not populations[1] or not revised_groups:
        raise ValueError("The fit requires generated and ordinary changed words")
    x, y = torch.cat(x), torch.tensor(y)
    torch.manual_seed(SEED)
    mean, scale = x.mean(0), x.std(0).clamp_min(.025)
    x = (x - mean) / scale
    weights = torch.tensor([.5, .25, .25])[y] * (len(y) / torch.bincount(y)[y])
    models, histories = {}, {}
    for arm in ARMS:
        torch.manual_seed(SEED)
        head = nn.Sequential(nn.Linear(x.shape[1], 32), nn.GELU(), nn.Linear(32, 2))
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
        histories[arm] = []
        for epoch in range(RULE["epochs"]):
            generator = torch.Generator().manual_seed(SEED + epoch)
            losses = []
            for indices in torch.randperm(len(y), generator=generator).split(512):
                features, labels, balance = x[indices], y[indices], weights[indices]
                values = head(features)
                if arm == "direct":
                    loss = bce(values[:, 0], (labels == 1).float(), weight=balance)
                else:
                    changed = labels != 0
                    origin = bce(values[:, 1], (labels == 1).float(), weight=balance * changed)
                    loss = bce(values[:, 0], changed.float(), weight=balance) + origin
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            histories[arm].append(round(sum(losses) / len(losses), 7))
        print(f"Fitted {arm} on {len(y)} native words. Final loss {histories[arm][-1]:.4f}.", flush=True)
        models[arm] = head.eval()
    return (mean, scale, models), {"fit_words": dict(sorted(populations.items())),
                                  "ordinary_revision_groups": len(revised_groups), "loss": histories}


def best(words, score):
    sections = defaultdict(list)
    for (i, (_, _, section, _)), value in zip(words, score, strict=True):
        sections[section].append((round(float(value), 7), i))
    ranked = []
    for sentence, entries in sorted(sections.items()):
        ordered = sorted(entries, key=lambda item: (-item[0], item[1]))
        strength = round(sum(value for value, _ in ordered[:6]) / 6, 7)
        ranked.append({"sentence": sentence, "score": strength,
                       "words": sorted(i for _, i in ordered[:math.ceil(.6 * len(ordered))])})
    ranked.sort(key=lambda item: -item["score"])
    return ranked[0] if ranked[0]["score"] != ranked[1]["score"] else None


def measure(rows, truth, tokenizer, backbone, device, trained):
    import torch

    mean, scale, heads = trained
    with torch.no_grad():
        for row, example in feature_batches(rows, tokenizer, backbone, device):
            record = {key: row[key] for key in ("role", "group", "family", "sha256", "spans", "position")}
            word_spans = [[a, b, sentence] for a, b, sentence, _ in prior.items(row)]
            record.update(words=word_spans, changed=sorted(truth.get(row["sha256"], ())),
                          assessed=example is not None, best={arm: None for arm in ARMS})
            if example:
                words, useful, features = example
                indexed = [(i, words[i]) for i in useful]
                for arm, head in heads.items():
                    scores = head((features - mean) / scale).sigmoid()
                    values = scores[:, 0] if arm == "direct" else scores.prod(1)
                    record["best"][arm] = best(indexed, values)
            yield record


def points(calibration):
    if len({row["group"] for row in calibration}) != len(calibration) or any(row["changed"] for row in calibration):
        raise ValueError("Operating points require separate ordinary works")
    return {arm: prior.points([{"truth": None, "family": "original", "best": row["best"][arm]} for row in calibration])
            for arm in ARMS}


def compare(rows, cuts):
    totals = defaultdict(Counter)
    for row in rows:
        geometry = row["words"]
        if any(not (0 <= a < b and 0 <= section < 5 and row["spans"][section][0] <= a < b <= row["spans"][section][1])
               for a, b, section in geometry) or any(b > a for (_, b, _), (a, _, _) in zip(geometry, geometry[1:])):
            raise ValueError("The receipt contains invalid word geometry")
        if any(not 0 <= i < len(geometry) or geometry[i][2] != row["position"] for i in row["changed"]):
            raise ValueError("The changed-word labels differ from the known edit sentence")
        for arm, by_level in cuts.items():
            reading = row["best"][arm]
            if reading is not None and (not row["assessed"] or not math.isfinite(reading["score"])
                    or reading["sentence"] not in {1, 2, 3}
                    or reading["words"] != sorted(set(reading["words"]))
                    or any(not 0 <= i < len(geometry) or geometry[i][2] != reading["sentence"] for i in reading["words"])):
                raise ValueError("Highlighted words differ from the submitted sentence")
            for level, point in by_level.items():
                counts = totals[(row["family"], arm, level)]
                counts.update(files=1, assessed=row["assessed"])
                if reading is None or reading["score"] <= point:
                    continue
                counts["flagged"] += 1
                if row["family"] in {"original", "before", "edited"}:
                    continue
                selected, changed = set(reading["words"]), set(row["changed"])
                common = len(selected & changed)
                counts["exact_sentence"] += reading["sentence"] == row["position"]
                counts["useful"] += bool(changed) and common * 2 >= len(changed) and common * 10 >= 3 * len(selected | changed)
    return {" / ".join(key): dict(value) for key, value in sorted(totals.items())}


def compare_prior(record):
    baseline = spans.read(spans.PRIMARY)
    known = {(row["group"], row["family"]): row for row in baseline["held"] if row["role"] == "held"}
    totals = defaultdict(Counter)
    for row in record["held"]:
        key = row["group"], row["family"]
        peer = known.pop(key)
        if (row["sha256"], row["words"], row["changed"]) != (peer["sha256"], peer["word_ranges"], peer["changed_words"]):
            raise ValueError("The held case differs from the prior native-word evaluation")

        def selected(reading, cut):
            if reading is None or reading["score"] <= cut:
                return False
            if row["family"] == "original":
                return True
            changed, words = set(row["changed"]), set(reading["words"])
            common = len(changed & words)
            return bool(changed) and common * 2 >= len(changed) and common * 10 >= 3 * len(changed | words)

        for arm, cuts in record["points"].items():
            for level, cut in cuts.items():
                old = selected(peer["best"], baseline["points"][level])
                new = selected(row["best"][arm], cut)
                values = totals[(row["family"], arm, level)]
                values.update(files=1, prior=old, candidate=new, added=new and not old, lost=old and not new)
    if known:
        raise ValueError("Previously evaluated report works are missing")
    return {" / ".join(key): dict(value) for key, value in sorted(totals.items())}


def run(args):
    import torch
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(4)
    rows, truth, _ = prior.input_plan(args.roster)
    pinned = spans.read(spans.PRIMARY)
    if prior.parent.pool.file_hash(args.encoder_weights) != pinned["weights_sha256"]:
        raise ValueError("The frozen prior word encoder differs from its evaluated weights")
    args.output.mkdir(mode=0o700)
    fit_rows = fit_roster(rows)
    freeze = {"schema": SCHEMA, "rule": RULE, "encoder_sha256": pinned["weights_sha256"],
              "fit": [[row[key] for key in ("role", "group", "family", "sha256")] for row in fit_rows]}
    prior.parent.previous.write(args.output / "freeze.json", freeze)
    tokenizer, backbone, _, device = prior.parent.init_encoder(args.encoder)
    frozen = load_file(args.encoder_weights)
    backbone.load_state_dict({key.removeprefix("encoder."): value for key, value in frozen.items()
                              if key.startswith("encoder.")}, strict=True)
    backbone.eval()
    trained, metrics = fit(feature_batches(fit_rows, tokenizer, backbone, device), truth)
    mean, scale, heads = trained
    filename = args.output / "head.safetensors"
    save_file({"mean": mean, "scale": scale} | {f"{arm}.{key}": value for arm, head in heads.items()
              for key, value in head.state_dict().items()}, filename)
    calibration = list(measure([row for row in rows if row["role"] == "calibration"], truth, tokenizer, backbone, device, trained))
    cuts = points(calibration)
    prior.parent.previous.write(args.output / "points.json", cuts)
    held = list(measure([row for row in rows if row["role"] == "held"], truth, tokenizer, backbone, device, trained))
    record = freeze | metrics | {"head_sha256": prior.parent.pool.file_hash(filename), "calibration": calibration,
                                "held": held, "points": cuts, "results": compare(held, cuts)}
    prior.parent.previous.write(args.output / "receipt.json", record)
    print(json.dumps(record["results"], sort_keys=True), flush=True)


def review(path):
    record = spans.read(path)
    if record["schema"] != SCHEMA or record["rule"] != RULE:
        raise ValueError("The frozen experiment differs")
    if {row[1] for row in record["fit"]} & {row["group"] for row in record["calibration"] + record["held"]}:
        raise ValueError("An evaluated work entered the fit")
    cuts = points(record["calibration"])
    result = compare(record["held"], cuts)
    if cuts != record["points"] or result != record["results"]:
        raise ValueError("The measured operating points or results differ")
    print(json.dumps({"scores": result, "paired_prior": compare_prior(record)}, sort_keys=True, indent=2))


def ordinary_inputs(folder):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    sources = folder / "originals.json"
    source_receipt = spans.read(folder / "source-receipt.json")
    if spans.parent.pool.file_hash(sources) != source_receipt["originals_sha256"]:
        raise ValueError("The admitted public journal sources changed")
    texts = []
    for article in spans.read(sources):
        for side, family in (("before", "before"), ("after", "edited")):
            excerpt = article[side]
            document = from_string(excerpt["text"])
            sentences = segment(document)
            ranges = [[sentence.start, sentence.end] for sentence in sentences]
            if document.text != excerpt["text"] or len(sentences) != 5 or ranges[2] != excerpt["focus"]:
                raise ValueError("The original journal passage geometry changed")
            texts.append({"role": "ordinary_transfer", "domain": "journals", "family": family, "group": article["group"],
                          "text": document.text, "spans": ranges, "position": 2, "truth": None,
                          "sha256": spans.parent.pool.sha(document.text)})
    changed = {new["sha256"]: prior.edit_words(old, new) for old, new in zip(texts[::2], texts[1::2], strict=True)}
    return texts, changed, source_receipt


def ordinary_transfer(args):
    import torch
    from safetensors.torch import load_file
    from torch import nn

    torch.set_num_threads(4)
    primary = spans.read(args.fitted / "receipt.json")
    if points(primary["calibration"]) != primary["points"] or primary["rule"] != RULE:
        raise ValueError("The fitted operating points differ")
    digest = spans.parent.pool.file_hash
    fitted = args.fitted / "head.safetensors"
    if digest(fitted) != primary["head_sha256"] or digest(args.encoder_weights) != primary["encoder_sha256"]:
        raise ValueError("The frozen transfer assets differ")
    rows, truth, source = ordinary_inputs(args.sources)
    tokenizer, backbone, old_head, device = prior.parent.init_encoder(args.encoder)
    frozen = load_file(args.encoder_weights)
    for prefix, module in (("encoder", backbone), ("head", old_head)):
        module.load_state_dict({key.removeprefix(prefix + "."): value for key, value in frozen.items()
                               if key.startswith(prefix + ".")}, strict=True)
        module.eval()
    parameters = load_file(fitted)
    mean, scale = parameters["mean"], parameters["scale"]
    heads = {}
    for arm in ARMS:
        model = nn.Sequential(nn.Linear(len(mean), 32), nn.GELU(), nn.Linear(32, 2))
        model.load_state_dict({key.removeprefix(arm + "."): value for key, value in parameters.items()
                               if key.startswith(arm + ".")}, strict=True)
        heads[arm] = model.eval()
    measurements = list(measure(rows, truth, tokenizer, backbone, device, (mean, scale, heads)))
    older = prior.score(rows, truth, tokenizer, backbone, old_head, device)
    for row, old in zip(measurements, older, strict=True):
        if row["sha256"] != old["sha256"] or row["words"] != old["word_ranges"]:
            raise ValueError("The old and new checks measured different source words")
        row["best"]["prior"] = old["best"]
    record = {"schema": TRANSFER_SCHEMA, "source_receipt": source, "primary_sha256": digest(args.fitted / "receipt.json"),
              "points": primary["points"] | {"prior": spans.read(spans.PRIMARY)["points"]}, "rows": measurements}
    record["results"] = compare(measurements, record["points"])
    prior.parent.previous.write(args.output, record)
    print(json.dumps(record["results"], sort_keys=True), flush=True)


def review_ordinary(receipt, primary_path):
    record, primary = map(spans.read, (receipt, primary_path))
    primary_bytes = primary_path.read_bytes()
    if primary_path.suffix == ".gz":
        primary_bytes = gzip.decompress(primary_bytes)
    if (record["schema"] != TRANSFER_SCHEMA or record["primary_sha256"] != hashlib.sha256(primary_bytes).hexdigest()
            or record["points"] != points(primary["calibration"]) | {"prior": spans.read(spans.PRIMARY)["points"]}):
        raise ValueError("The independent ordinary transfer changed the frozen points")
    roles = defaultdict(set)
    for row in record["rows"]:
        roles[row["group"]].add(row["family"])
    if len(record["rows"]) != 2 * len(roles) or any(variants != {"before", "edited"} for variants in roles.values()):
        raise ValueError("Ordinary revision mates are missing")
    result = compare(record["rows"], record["points"])
    if result != record["results"]:
        raise ValueError("The ordinary transfer result differs")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="phase", required=True)
    train = actions.add_parser("run")
    for name in ("roster", "encoder", "encoder_weights", "output"):
        train.add_argument(name, type=Path)
    actions.add_parser("review").add_argument("receipt", type=Path)
    transfer = actions.add_parser("ordinary-transfer")
    for name in ("sources", "fitted", "encoder", "encoder_weights", "output"):
        transfer.add_argument(name, type=Path)
    portable = actions.add_parser("review-ordinary")
    portable.add_argument("receipt", type=Path)
    portable.add_argument("primary", type=Path)
    args = cli.parse_args()
    if args.phase == "run":
        run(args)
    elif args.phase == "review":
        review(args.receipt)
    elif args.phase == "ordinary-transfer":
        ordinary_transfer(args)
    else:
        review_ordinary(args.receipt, args.primary)
