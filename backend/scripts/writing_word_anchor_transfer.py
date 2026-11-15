# SPDX-License-Identifier: MIT
"""Fit exact-word anchors for short rewrite localization on single uploads."""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median

from scripts import writing_adjacent_range_transfer as adjacent
from scripts import writing_word_localization as old

PANEL = old.RECEIPT.with_name("writing-word-localization-2026-11-14.json")
SCHEMA = "chipmunk.writing-word-anchor-transfer/1"
EPOCHS = 5
SEED = 1849
BATCH = 16
HEADS = ("original", "plain", "anchors", "anchors_and_local")
WIDTHS = (2, 3, 5, 8, 12)
GAP_WIDTHS = (2, 3, 4)
ALL_WIDTHS = sorted(set(WIDTHS) | set(GAP_WIDTHS))
METHOD = {
    "fit": "Freeze the earlier six-layer word encoder. Use the same qualified 882 source pairs for all readouts. "
    "Refit three linear heads from the earlier word head. Keep word classification and the margin against "
    "the six highest original words in every fit. Add a paired consistency loss on exact monotone unchanged "
    "words, including inside the edited sentence. Test a further within-sentence margin against the six "
    "highest unchanged words there. Neither fitting arm uses the held excerpts or a new checkpoint.",
    "decoder": "Replay the earlier 60 percent word highlight on one sentence. Compare the strongest narrow "
    "contiguous range and two short fragments. Each fragment has two to four words and a gap of one to five "
    "words, within one sentence or adjacent interior sentences. Scale each shape using ordinary file maxima. "
    "Leave out each calibration work when measuring its own normalized maximum. Freeze two ordinary-only points.",
    "held": "Use fresh OpAI-Bench report and news works with one or two adjacent changed sentences. Count "
    "useful changed-word overlap, ordinary flags, matched original flags and real journal revision flags. "
    "Do not use the original version, genre labels for ranking, or change labels to produce held word scores. "
    "Use genre only for the ordinary calibration scale. A useful range covers at least half the changed words "
    "and achieves 0.30 intersection over union.",
}
SIGNALS = (("original", "broad"), ("original", "narrow"), ("original", "fragments"),
           ("plain", "narrow"), ("anchors", "narrow"), ("anchors_and_local", "narrow"),
           ("anchors_and_local", "fragments"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def archive():
    return old.parent.intake.read(PANEL)


def contract():
    prior = archive()
    return {"schema": SCHEMA, "method": METHOD, "checkpoint_sha256": prior["weights_sha256"],
            "word_receipt_sha256": old.parent.pool.file_hash(PANEL), "fit_sources": prior["freeze"]["source_roster_sha256"],
            "seed": SEED, "epochs": EPOCHS, "batch": BATCH}


def native_runtime(folder, checkpoint):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    require(old.parent.pool.file_hash(checkpoint) == contract()["checkpoint_sha256"], "Pinned word checkpoint differs")
    require(all(old.parent.pool.file_hash(folder / name) == digest for name, digest in old.parent.ENCODER_ASSETS.items()),
            "Encoder or tokenizer bytes differ")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    config = AutoConfig.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    config._attn_implementation = "eager"
    encoder = AutoModel.from_config(config, trust_remote_code=False).eval()
    head = torch.nn.Linear(config.hidden_size * 2, 1)
    state = load_file(str(checkpoint))
    encoder.load_state_dict({name.removeprefix("encoder."): value for name, value in state.items()
                             if name.startswith("encoder.")}, strict=True)
    head.load_state_dict({name.removeprefix("head."): value for name, value in state.items()
                          if name.startswith("head.")}, strict=True)
    require(tokenizer.is_fast and tokenizer.pad_token_id == 0 and config.hidden_size == 384,
            "Unexpected word coordinate or representation policy")
    return tokenizer, encoder, head


def features(items, encoder):
    import torch

    widest = max(len(ids) for ids, _, _ in items)
    tokens = torch.zeros((len(items), widest), dtype=torch.long)
    attention = torch.zeros_like(tokens)
    for index, (ids, _, _) in enumerate(items):
        tokens[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[index, :len(ids)] = 1
    with torch.no_grad():
        hidden = encoder(input_ids=tokens, attention_mask=attention).last_hidden_state
        output = []
        for index, (_, word_ids, words) in enumerate(items):
            positions, indices = zip(*((i, word) for i, word in enumerate(word_ids) if word is not None), strict=True)
            word_index = torch.tensor(indices, dtype=torch.long)
            source = hidden[index, list(positions)]
            representations = source.new_zeros((len(words), hidden.shape[-1])).index_add_(0, word_index, source)
            representations /= torch.bincount(word_index, minlength=len(words))[:, None]
            sentences = torch.tensor([word[2] for word in words], dtype=torch.long)
            count = int(sentences[-1]) + 1
            sentence_states = source.new_zeros((count, hidden.shape[-1])).index_add_(0, sentences, representations)
            sentence_states /= torch.bincount(sentences, minlength=count).clamp_min(1)[:, None]
            output.append(torch.cat((representations, representations-sentence_states[sentences]), dim=-1))
    return output


def word_alignment(before, after):
    original, edited = [], []
    for sentence in sorted({word[2] for word in before}):
        left = [(index, word[3]) for index, word in enumerate(before) if word[2] == sentence]
        right = [(index, word[3]) for index, word in enumerate(after) if word[2] == sentence]
        matched = SequenceMatcher(None, [w for _, w in left], [w for _, w in right], autojunk=False)
        for a, b, count in matched.get_matching_blocks():
            for i in range(count):
                original.append(left[a+i][0])
                edited.append(right[b+i][0])
    return original, edited


def qualified_pairs(roster, tokenizer, encoder):
    rows, truth, _ = old.input_plan(roster)
    groups = defaultdict(list)
    for row in rows:
        if row["role"] == "fit":
            groups[row["group"]].append(row)
    candidates = []
    for group, (original, edited) in sorted(groups.items()):
        before, after = old.encode(original, tokenizer), old.encode(edited, tokenizer)
        changed = sorted(truth[edited["sha256"]])
        if before is not None and after is not None and changed:
            candidates.append((group, before, after, changed, edited["position"]))
    require(len(candidates) == archive()["fit_groups"], "Eligible fit pairs differ from the word checkpoint")
    dataset = []
    for start in range(0, len(candidates), BATCH):
        part = candidates[start:start+BATCH]
        data = iter(features([encoded for item in part for encoded in item[1:3]], encoder))
        for group, before, after, changed, sentence in part:
            original, edited = word_alignment(before[2], after[2])
            require(set(changed).isdisjoint(edited), "An anchor was labeled as a changed training word")
            whole_sentence = [index for index, word in enumerate(before[2]) if word[2] == sentence]
            neighbors = [index for index in edited if after[2][index][2] == sentence]
            dataset.append((group, next(data), next(data), changed, original, edited, whole_sentence, neighbors))
        if len(dataset) % 128 == 0 or len(dataset) == len(candidates):
            print(f"Encoded {len(dataset)}/{len(candidates)} matched training pairs", flush=True)
    return dataset


def train(dataset, initial, arm):
    import torch
    from torch.nn import functional as F

    rng = random.Random(SEED)
    model = torch.nn.Linear(initial.in_features, 1)
    model.load_state_dict(initial.state_dict(), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    trace = []
    for _ in range(EPOCHS):
        order = list(range(len(dataset)))
        rng.shuffle(order)
        total = 0.
        for i in range(0, len(order), BATCH):
            pairs = [dataset[index] for index in order[i:i+BATCH]]
            lengths = [len(values) for pair in pairs for values in pair[1:3]]
            points = model(torch.cat([state for pair in pairs for state in pair[1:3]])).flatten()
            segments = iter(points.split(lengths))
            labels = torch.zeros_like(points)
            margins, stability, contrast = [], [], []
            start = 0
            for _, before, after, changed, source_words, target_words, controls, neighbors in pairs:
                left, right = next(segments), next(segments)
                start += len(before)
                labels[start + torch.tensor(changed)] = 1
                start += len(after)
                edited = right[changed].mean()
                margins.append(F.softplus(1. - edited + left[controls].topk(6).values.mean()))
                if arm != "plain" and source_words:
                    stability.append(F.smooth_l1_loss(left[source_words], right[target_words]))
                if arm == "anchors_and_local" and neighbors:
                    close = right[neighbors].topk(min(len(neighbors), 6)).values.mean()
                    contrast.append(F.softplus(1. - edited + close))
            loss = F.binary_cross_entropy_with_logits(points, labels, pos_weight=torch.tensor(8.))
            loss = loss + .4 * torch.stack(margins).mean()
            if stability:
                loss = loss + .35 * torch.stack(stability).mean()
            if contrast:
                loss = loss + .25 * torch.stack(contrast).mean()
            require(torch.isfinite(loss).item(), "Training objective is not finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * len(pairs)
        trace.append(round(total / len(dataset), 7))
    return model.eval(), trace


def fit(args):
    from safetensors.torch import save_file

    expected = contract()
    require(old.parent.pool.file_hash(args.roster) == expected["fit_sources"], "The fit source roster differs")
    tokenizer, encoder, initial = native_runtime(args.encoder, args.checkpoint)
    args.output.mkdir(mode=0o700)
    adjacent.save(args.output / "freeze.json", expected)
    dataset = qualified_pairs(args.roster, tokenizer, encoder)
    heads = {}
    for arm in HEADS[1:]:
        head, losses = train(dataset, initial, arm)
        path = args.output / f"{arm}.safetensors"
        save_file(head.state_dict(), str(path))
        heads[arm] = {"weights_sha256": old.parent.pool.file_hash(path), "losses": losses}
        print(f"Fitted {arm} on {len(dataset)} qualified source pairs", flush=True)
    adjacent.save(args.output / "fit.json", {"freeze": expected, "heads": heads, "groups": [d[0] for d in dataset]})


def broad(words, scores, size):
    sentences = defaultdict(list)
    for index, word in enumerate(words):
        if 0 < word[2] < size - 1:
            sentences[word[2]].append((scores[index], index))
    ranked = []
    for values in sentences.values():
        values.sort(key=lambda item: (-item[0], item[1]))
        strength = round(sum(value for value, _ in values[:6]) / 6, 7)
        chosen = sorted(index for _, index in values[:math.ceil(.60 * len(values))])
        ranked.append({"score": strength, "indices": chosen})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[0] if len(ranked) >= 2 and ranked[0]["score"] != ranked[1]["score"] else None


def best_by_shape(words, scores, count):
    sentence = [w[2] for w in words]
    totals = [0.]
    for value in scores:
        totals.append(totals[-1] + value)
    best = {}
    for start, first in enumerate(sentence):
        if not 0 < first < count - 1:
            continue
        for width in ALL_WIDTHS:
            stop = start + width
            if stop > len(scores) or sentence[stop-1] != first:
                continue
            value = totals[stop] - totals[start]
            if width in WIDTHS:
                key = f"one/{width}"
                candidate = {"score": round(value / width, 7), "indices": list(range(start, stop))}
                if key not in best or candidate["score"] > best[key]["score"]:
                    best[key] = candidate
            if width not in GAP_WIDTHS:
                continue
            for gap in range(1, 6):
                second, end = stop + gap, stop + gap + width
                if end > len(scores) or sentence[second] != sentence[end-1] or not first <= sentence[second] <= min(first+1, count-2):
                    continue
                both = value + totals[end] - totals[second]
                key = f"two/{width}"
                candidate = {"score": round(both / (2*width), 7), "indices": [*range(start, stop), *range(second, end)]}
                if key not in best or candidate["score"] > best[key]["score"]:
                    best[key] = candidate
    return best


def measures(rows, measurements):
    results = {}
    for row in rows:
        evidence = measurements[row["sha256"]]
        prepared = {}
        shape_cache = {}
        for head, decoder in SIGNALS:
            raw = evidence[head]
            if not raw:
                prepared[head, decoder] = {}
            elif decoder == "broad":
                selected = broad(row["words"], raw, len(row["spans"]))
                prepared[head, decoder] = {"broad": selected} if selected else {}
            else:
                if head not in shape_cache:
                    shape_cache[head] = best_by_shape(row["words"], raw, len(row["spans"]))
                shapes = shape_cache[head]
                prepared[head, decoder] = {shape: pick for shape, pick in shapes.items() if
                                           decoder != "narrow" or shape.startswith("one/")}
        results[row["sha256"]] = prepared
    return results


def profiles(rows, evaluated, approach, *, exclude=None):
    collected = defaultdict(list)
    for row in rows:
        if row["sha256"] == exclude:
            continue
        for shape, pick in evaluated[row["sha256"]][approach].items():
            collected[shape].append(pick["score"])
    norms = {}
    for shape, values in collected.items():
        if len(values) >= 24:
            values.sort()
            central = median(values)
            high = values[math.ceil(len(values) * .9)-1]
            norms[shape] = (central, max(.05, high - central)) if shape != "broad" else (0., 1.)
    return norms


def best(proposals, norms):
    options = []
    for shape, candidate in proposals.items():
        if shape in norms:
            middle, scale = norms[shape]
            value = round((candidate["score"] - middle) / scale, 7)
            options.append(candidate | {"score": value, "shape": shape})
    return min(options, key=lambda row: (-row["score"], len(row["indices"]), row["indices"]), default=None)


def calibrate(rows, evaluated):
    result = {}
    for genre in adjacent.PINS:
        genuine = [row for row in rows if row["genre"] == genre and row["family"] == "calibration"]
        require(len(genuine) == 36 and all(row["delivery"] == "original" for row in genuine), "Ordinary calibration roster differs")
        for approach in SIGNALS:
            candidate = []
            for row in genuine:
                norms = profiles(genuine, evaluated, approach, exclude=row["sha256"])
                chosen = best(evaluated[row["sha256"]][approach], norms)
                if chosen:
                    candidate.append(chosen["score"])
            require(len(candidate) >= 24, "Genuine calibration has insufficient word coverage")
            candidate.sort(reverse=True)
            shapes = {name: list(values) for name, values in profiles(genuine, evaluated, approach).items()}
            result["/".join((genre, *approach))] = {
                "shapes": shapes, "assessed": len(candidate),
                "points": {"10pct": candidate[len(candidate)//10], "25pct": candidate[len(candidate)//4]},
            }
    return result


def native_rows(rows, tokenizer, encoder, heads):
    import torch

    scored = {}
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start+BATCH]
        encoded = []
        for row in batch:
            text = old.encode(row, tokenizer)
            counts = Counter(word[2] for word in text[2]) if text else {}
            encoded.append(text if all(counts.get(index, 0) >= 6 for index in range(len(row["spans"]))) else None)
        ready = [value for value in encoded if value is not None]
        projected = iter(features(ready, encoder) if ready else [])
        with torch.no_grad():
            for row, signal in zip(batch, encoded, strict=True):
                vectors = next(projected) if signal is not None else None
                scored[row["sha256"]] = {name: ([] if vectors is None else
                                               [round(value, 7) for value in head(vectors).flatten().tolist()])
                                         for name, head in heads.items()}
    return scored


def check_fit(record):
    require(record["freeze"] == contract() and set(record["heads"]) == set(HEADS[1:]), "The fitted readout inventory differs")
    require(len(record["groups"]) == len(set(record["groups"])) == archive()["fit_groups"], "Fitting source groups differ")
    for stats in record["heads"].values():
        require(len(stats["losses"]) == EPOCHS and all(math.isfinite(value) for value in stats["losses"]),
                "An incomplete fitted readout was used")


def capture(args):
    import torch
    from safetensors.torch import load_file

    lock = adjacent.read(args.fit / "fit.json")
    check_fit(lock)
    require(old.parent.pool.file_hash(args.roster) == lock["freeze"]["fit_sources"], "The fit source roster differs")
    plan = adjacent.read(args.plan)
    require(plan["schema"] == adjacent.SCHEMA and plan["sources"] == adjacent.SOURCES, "Holdout source manifest differs")
    fresh = plan["rows"]
    editorials, truth, _ = old.input_plan(args.roster)
    editorials = [row for row in editorials if row["role"] == "editorial" and row["family"] in {"before", "edited"}]
    editorials = [{"group": r["group"], "genre": "reports", "family": "editorial", "delivery": r["family"],
                   "target": [r["position"]], "changed": sorted(truth.get(r["sha256"], ())),
                   **{name: r[name] for name in ("text", "spans", "sha256")}} for r in editorials]
    require(len(editorials) == 42, "The genuine revision bank differs")
    for row in [*fresh, *editorials]:
        require(old.parent.pool.sha(row["text"]) == row["sha256"], "Source text changed after the roster was frozen")
    tokenizer, encoder, original = native_runtime(args.encoder, args.checkpoint)
    heads = {"original": original}
    for name, stats in lock["heads"].items():
        path = args.fit / f"{name}.safetensors"
        require(old.parent.pool.file_hash(path) == stats["weights_sha256"], "Held word head differs from its fitted bytes")
        head = torch.nn.Linear(original.in_features, 1)
        head.load_state_dict(load_file(str(path)), strict=True)
        heads[name] = head.eval()
    genuine = [row for row in fresh if row["family"] == "calibration"]
    calibration = native_rows(genuine, tokenizer, encoder, heads)
    clean = list(map(adjacent.public, genuine))
    points = calibrate(clean, measures(clean, calibration))
    args.output.mkdir(mode=0o700)
    adjacent.save(args.output / "calibration.json", {"points": points, "fit": lock, "rows": clean})
    print("Sealed the 72 genuine calibration inputs before held word inference", flush=True)
    held = [row for row in fresh if row["family"] != "calibration"] + editorials
    readings = calibration | native_rows(held, tokenizer, encoder, heads)
    public = list(map(adjacent.public, fresh + editorials))
    sentence = adjacent.read(args.sentence)
    adjacent.review(sentence)
    require(sentence["rows"] == list(map(adjacent.public, fresh)), "The sentence comparison used different inputs")
    receipt = {"freeze": contract(), "fit": lock, "exposures": plan["exposures"], "rows": public, "readings": readings,
               "points": points, "anchors": held_anchors(fresh),
               "sentence": {key: sentence[key] for key in ("model", "readings", "points", "summary")}}
    receipt["result"] = report(receipt)
    receipt["diagnostic"] = diagnostic(receipt)
    adjacent.save(args.output / "receipt.json.gz", receipt)


def held_anchors(rows):
    original = {row["group"]: row for row in rows if row["delivery"] == "original"}
    matches = {}
    for revised in (row for row in rows if row["delivery"] == "edited"):
        source = original[revised["group"]]
        old_words, new_words = old.items(source), old.items(revised)
        pair = zip(*word_alignment(old_words, new_words), strict=True)
        changes = set(revised["changed"])
        matches[revised["group"]] = [[left, right] for left, right in pair
                                     if right not in changes and new_words[right][2] in revised["target"]]
    return matches


def diagnostic(record):
    ordinary = {row["group"]: row for row in record["rows"] if row["delivery"] == "original"}
    synth = [r for r in record["rows"] if r["delivery"] == "edited" and r["family"] != "editorial"]
    require(set(record["anchors"]) == {r["group"] for r in synth}, "Edit-clock comparisons differ")
    ratings = record["readings"]
    scales, ranking, drift = {}, defaultdict(list), defaultdict(list)
    for head in HEADS:
        reference = [value for row in ordinary.values() if row["family"] == "calibration"
                     for value in ratings[row["sha256"]][head]]
        middle = median(reference)
        scales[head] = max(.05, median(abs(value-middle) for value in reference))
    reach = Counter()
    for revised in synth:
        group = revised["group"]
        source = ordinary[group]
        local = record["anchors"][group]
        changed = set(revised["changed"])
        target_words = revised["words"]
        require(all(0 <= a < len(source["words"]) and 0 <= b < len(target_words) and b not in changed and
                    source["words"][a][2] == target_words[b][2] and target_words[b][2] in revised["target"]
                    for a, b in local) and all(a < c and b < d for (a, b), (c, d) in zip(local, local[1:])),
                "Unchanged word alignment violates original source coordinates")
        masks = [float(i in changed) for i in range(len(revised["words"]))]
        maxima = best_by_shape(revised["words"], masks, len(revised["spans"]))
        useful_shapes = {shape.split("/")[0] for shape, pick in maxima.items()
                         if old.useful_words(set(pick["indices"]), changed)}
        reach["edited"] += 1
        reach["one_span_possible"] += "one" in useful_shapes
        reach["two_spans_possible"] += "two" in useful_shapes
        reach["either_possible"] += bool(useful_shapes)
        comparisons = [i for i, (_, _, sentence) in enumerate(target_words)
                       if i not in changed and sentence in revised["target"]]
        for head in HEADS:
            before, later = ratings[source["sha256"]][head], ratings[revised["sha256"]][head]
            if later and comparisons:
                wins = sum(1.0 if later[i] > later[j] else .5 if later[i] == later[j] else 0.
                           for i in changed for j in comparisons)
                ranking[head].append(wins / (len(changed)*len(comparisons)))
            if before and later and local:
                drift[head].append(median(abs(before[i] - later[j]) / scales[head] for i, j in local))
    return {"geometry": dict(reach), "heads": {
        head: {"local_rank_assessed": len(ranking[head]), "median_edit_rank": round(median(ranking[head]), 7),
               "unchanged_pairs_assessed": len(drift[head]),
               "median_unchanged_logit_drift": round(median(drift[head]), 7)}
        for head in HEADS}}


def check_sources(record):
    rows = record["rows"]
    signals = record["readings"]
    require(len(rows) == len(signals) == 282 and {row["sha256"] for row in rows} == set(signals),
            "Held or ordinary readings are missing or duplicated")
    known, _ = adjacent.context.abstract_exclusions(record["exposures"])
    known.update(record["fit"]["groups"])
    editorial, fresh = [], []
    for row in rows:
        word_ranges = row["words"]
        for a, b, sentence in word_ranges:
            require(0 <= sentence < len(row["spans"]) and row["spans"][sentence][0] <= a < b <= row["spans"][sentence][1],
                    "A lexical coordinate falls outside its original sentence")
        require(all(b <= c for (_, b, _), (c, _, _) in zip(word_ranges, word_ranges[1:])) and
                all(0 <= index < len(word_ranges) and word_ranges[index][2] in row["target"] for index in row["changed"]),
                "The source word sequence or labeled changes differ")
        trace = signals[row["sha256"]]
        require(set(trace) == set(HEADS) and len({len(values) for values in trace.values()}) == 1 and
                all(not values or len(values) == len(word_ranges) and all(math.isfinite(value) for value in values)
                    for values in trace.values()), "The complete word-score grid differs")
        if row["family"] == "editorial":
            editorial.append(row)
        else:
            require(row["group"] not in known, "A fresh holdout or calibration group has prior exposure")
            fresh.append(row)
    require(len(fresh) == 240 and Counter(row["delivery"] for row in editorial) == {"before": 21, "edited": 21},
            "The held bank differs")
    editorial_pairs = defaultdict(list)
    for row in editorial:
        editorial_pairs[row["group"]].append(row["delivery"])
    require(len(editorial_pairs) == 21 and all(set(pair) == {"before", "edited"} for pair in editorial_pairs.values()),
            "A genuine revision is missing its earlier copy")
    sentence = {"schema": adjacent.SCHEMA, "method": adjacent.METHOD, "sources": adjacent.SOURCES,
                "source_revision": adjacent.SOURCE_REVISION, "exposures": record["exposures"], "rows": fresh,
                **record["sentence"]}
    adjacent.review(sentence)
    return fresh, editorial


def add_count(tally, row, chosen, ordinary=None):
    edited = row["delivery"] == "edited" and row["family"] != "editorial"
    role = "edited" if edited else "ordinary"
    tally[role] += 1
    if not chosen:
        return
    tally[f"{role}_flags"] += 1
    proposal = set(chosen["indices"])
    if edited:
        correct = old.useful_words(proposal, set(row["changed"]))
        tally["useful"] += correct
        tally["useful_without_original_alert"] += correct and ordinary is None
        tally["matched_sentences"] += sorted({row["words"][word][2] for word in proposal}) == row["target"]
    elif row["family"] == "editorial" and row["delivery"] == "edited":
        tally["genuine_revision_words_flagged"] += old.useful_words(proposal, set(row["changed"]))


def report(record):
    require(record["freeze"] == contract(), "The method or archived word checkpoint differs")
    check_fit(record["fit"])
    fresh, genuine_edits = check_sources(record)
    evaluated = measures(record["rows"], record["readings"])
    points = calibrate(fresh, evaluated)
    require(points == record["points"], "Operating points changed after calibration")
    groups = {row["group"]: row for row in fresh if row["delivery"] == "original"}
    counts = defaultdict(Counter)
    for approach in SIGNALS:
        name = "/".join(approach)
        for level in ("10pct", "25pct"):
            for genre in adjacent.PINS:
                point = points[f"{genre}/{name}"]
                rows = [row for row in fresh if row["genre"] == genre and row["family"] != "calibration"]
                if genre == "reports":
                    rows.extend(genuine_edits)
                selected = {}
                for row in rows:
                    pick = best(evaluated[row["sha256"]][approach], point["shapes"])
                    selected[row["sha256"]] = (pick, pick if pick is not None and pick["score"] > point["points"][level] else None)
                for row in rows:
                    pick, highlight = selected[row["sha256"]]
                    group = row["group"]
                    ordinary = selected[groups[group]["sha256"]][1] if row["family"] != "editorial" else None
                    scopes = (("editorial", row["delivery"]) if row["family"] == "editorial" else
                              ("all", genre, f"{genre}/{row['family']}", f"{genre}/sentences-{len(row['target'])}"))
                    for scope in scopes:
                        tally = counts[f"{name}/{level}/{scope}"]
                        tally["assessed"] += pick is not None
                        add_count(tally, row, highlight, ordinary)
    return {key: dict(item) for key, item in sorted(counts.items())}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    run = commands.add_parser("fit")
    for name in ("output", "roster", "encoder", "checkpoint"):
        run.add_argument(name, type=Path)
    measure = commands.add_parser("capture")
    for name in ("output", "fit", "plan", "sentence", "roster", "encoder", "checkpoint"):
        measure.add_argument(name, type=Path)
    commands.add_parser("review").add_argument("receipt", type=Path)
    args = cli.parse_args()
    if args.command == "fit":
        fit(args)
    elif args.command == "capture":
        capture(args)
    else:
        record = adjacent.read(args.receipt)
        outcome = report(record)
        require(outcome == record["result"], "Saved word results disagree with replay")
        require(diagnostic(record) == record["diagnostic"], "Saved word diagnostics disagree with replay")
        print(f"Replayed {len(record['rows'])} excerpts across {len(SIGNALS)} word checks", flush=True)
        for key, count in outcome.items():
            if key.endswith(("/all", "/editorial")):
                print(key, count)
        print("diagnostic", record["diagnostic"])


if __name__ == "__main__":
    main()
