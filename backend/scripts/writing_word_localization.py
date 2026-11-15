# SPDX-License-Identifier: MIT
"""Test word-level localization on matched native human and generated revisions."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from scripts import writing_context_order_transfer as transfer_sources
from scripts import writing_matched_encoder as parent

SCHEMA = "chipmunk.writing-word-localization/1"
RECEIPT = parent.pool.PANEL_DIR / "writing-matched-encoder-2026-11-14.json"
WORD = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*", re.UNICODE)
EPOCHS = 3
METHOD = {
    "training": "Start at the pinned Apache-2.0 encoder. Fit on original and generated report excerpts. Label only "
                "submitted words within the known changed sentence that are outside a monotonic exact-word alignment. "
                "Use inverse token-piece counts to give each word equal supervised weight.",
    "head": "Concatenate each token representation and its difference from the mean lexical representation "
            "within its sentence. Fit a linear head with token BCE plus a paired margin over the edited words "
            "and the six strongest corresponding original-center words.",
    "selection": "Use the six strongest averaged word scores to rank each interior sentence. A unique winning "
                 "sentence may be marked. Compare highlighting that sentence with retaining its top 60 percent "
                 "of scored words. Fit both alert points on the separate genuine report files, before held inference.",
    "evaluation": "Compare exact sentence and changed-word selection. Assess the separate developer-defined "
                  "report generators and dated ordinary journal revisions. Truth is reproducible changed wording, "
                  "not a label for authorship of individual words. Word success requires at least 50 percent recall "
                  "and 30 percent intersection over union of changed words. Report false and misplaced alerts.",
}


def receipt_digest(record):
    canonical = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return parent.pool.sha(canonical)


def fit_contract(baseline):
    return {"schema": SCHEMA, "method": METHOD, "source_roster_sha256": baseline["roster_sha256"],
            "baseline_sha256": parent.pool.file_hash(RECEIPT),
            "encoder_manifest_sha256": receipt_digest(parent.ENCODER_ASSETS)}


def items(row):
    return [(found.start() + a, found.end() + a, sentence, found.group().casefold())
            for sentence, (a, b) in enumerate(row["spans"])
            for found in WORD.finditer(row["text"][a:b])]


def edit_words(before, after):
    center = after["position"]
    left = [item[3] for item in items(before) if item[2] == center]
    right = [(i, item[3]) for i, item in enumerate(items(after)) if item[2] == center]
    matched = SequenceMatcher(None, left, [lexeme for _, lexeme in right], autojunk=False)
    return {right[j][0] for action, _, _, a, b in matched.get_opcodes() if action != "equal" for j in range(a, b)}


def input_plan(path):
    baseline = parent.intake.read(RECEIPT)
    data = parent.intake.read(path)
    if parent.pool.file_hash(path) != baseline["roster_sha256"]:
        raise ValueError("The source roster differs from the measured baseline")
    rows = [row | {"position": 2} if row["role"] == "editorial" else row for row in data["rows"]]
    bases = {(r["role"], r["group"]): r for r in rows if r["family"] in {"original", "before"}}
    truth = {}
    for row in rows:
        if row["truth"] is not None or row["family"] == "edited":
            truth[row["sha256"]] = edit_words(bases[row["role"], row["group"]], row)
    return rows, truth, baseline


def encode(row, tokenizer, *, tokenized=None):
    encoded = tokenizer(row["text"], return_offsets_mapping=True, truncation=False) if tokenized is None else tokenized
    ids = encoded["input_ids"]
    if len(ids) > parent.MAX_TOKENS:
        return None
    words = items(row)
    ends = [b for _, b, _, _ in words]
    covered = [set() for _ in words]
    word_ids = []
    for a, b in encoded["offset_mapping"]:
        index = bisect_right(ends, a)
        if b <= a or index == len(words) or b <= words[index][0]:
            word_ids.append(None)
            continue
        x, y, _, _ = words[index]
        if a < x or b > y:
            return None
        covered[index].update(range(a, b))
        word_ids.append(index)
    if any(len(points) != b-a for (a, b, _, _), points in zip(words, covered, strict=True)):
        return None
    sentences = Counter(sentence for _, _, sentence, _ in words)
    return (ids, word_ids, words) if all(sentences[i] >= 6 for i in range(5)) else None


def forward(model, head, batch, device):
    import torch

    width = max(len(ids) for ids, _, _ in batch)
    input_ids = torch.zeros((len(batch), width), dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    parts = torch.zeros((len(batch), 5, width))
    sentence_ids = torch.zeros_like(input_ids)
    for i, (ids, word_ids, words) in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        attention[i, :len(ids)] = 1
        counts = Counter(index for index in word_ids if index is not None)
        totals = Counter(sentence for _, _, sentence, _ in words)
        for j, index in enumerate(word_ids):
            if index is not None:
                sentence = words[index][2]
                parts[i, sentence, j] = 1 / (counts[index] * totals[sentence])
                sentence_ids[i, j] = sentence
    states = model(input_ids=input_ids.to(device), attention_mask=attention.to(device)).last_hidden_state
    means = parts.to(device).bmm(states)
    background = means.gather(1, sentence_ids.to(device).unsqueeze(-1).expand_as(states))
    return head(torch.cat((states, states-background), dim=-1)).squeeze(-1)


def fit(rows, truth, tokenizer, model, head, device):
    import torch
    from torch.nn import functional as F

    pairs = defaultdict(list)
    for row in rows:
        if row["role"] == "fit":
            pairs[row["group"]].append(row)
    examples = []
    for original, edited in pairs.values():
        old, new = encode(original, tokenizer), encode(edited, tokenizer)
        labels = truth[edited["sha256"]]
        if old and new and labels:
            examples.append((old, new, labels, edited["position"]))
    if len(examples) < 600:
        raise ValueError(f"Only {len(examples)} source pairs have exact word coverage")
    optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=3e-5)
    rng = random.Random(parent.SEED)
    model.train()
    losses = []
    for epoch in range(EPOCHS):
        rng.shuffle(examples)
        total = 0.
        for offset in range(0, len(examples), parent.BATCH_PAIRS):
            paired = examples[offset:offset+parent.BATCH_PAIRS]
            batch = [encoded for old, new, _, _ in paired for encoded in (old, new)]
            scores = forward(model, head, batch, device)
            labels = torch.zeros(scores.shape, dtype=torch.float32)
            weights = torch.zeros_like(labels)
            prior_masks, target_masks = [], torch.zeros((len(paired), scores.shape[1]))
            for i, (old, new, changed, center) in enumerate(paired):
                for side, (_ids, word_ids, _words) in enumerate((old, new)):
                    counts = Counter(index for index in word_ids if index is not None)
                    for j, index in enumerate(word_ids):
                        if index is not None:
                            weights[2*i + side, j] = 1 / counts[index]
                            labels[2*i + side, j] = side == 1 and index in changed
                candidates = {}
                for j, index in enumerate(old[1]):
                    if index is not None and old[2][index][2] == center:
                        candidates.setdefault(index, []).append(j)
                prior_masks.append(list(candidates.values()))
                target_masks[i] = labels[2*i+1] * weights[2*i+1]
            loss = F.binary_cross_entropy_with_logits(scores, labels.to(device), weight=weights.to(device),
                                                       pos_weight=torch.tensor(8., device=device), reduction="sum") / weights.sum().item()
            controls = torch.zeros((len(paired), max(map(len, prior_masks)), scores.shape[1]))
            for i, rows_of_tokens in enumerate(prior_masks):
                for j, tokens in enumerate(rows_of_tokens):
                    controls[i, j, tokens] = 1 / len(tokens)
            valid = controls.sum(dim=-1).bool().to(device)
            averages = controls.to(device).bmm(scores[::2].unsqueeze(-1)).squeeze(-1)
            prior = averages.masked_fill(~valid, -torch.inf).topk(6).values.mean(dim=-1)
            positive = target_masks.to(device)
            edited = (scores[1::2] * positive).sum(dim=-1) / positive.sum(dim=-1)
            loss = loss + .4 * F.softplus(1. - edited + prior).mean()
            if not torch.isfinite(loss):
                raise ValueError("The training objective is not finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*model.parameters(), *head.parameters()], 1.)
            optimizer.step()
            total += loss.item() * len(paired)
        losses.append(round(total / len(examples), 7))
        print(f"Epoch {epoch+1}/{EPOCHS} on {len(examples)} pairs: {losses[-1]}", flush=True)
    return len(examples), losses


def select(encoded, scores):
    _, word_ids, words = encoded
    totals = defaultdict(list)
    for index, score in zip(word_ids, scores[:len(word_ids)], strict=True):
        if index is not None:
            totals[index].append(score)
    by_sentence = defaultdict(list)
    for index, (_, _, sentence, _) in enumerate(words):
        points = totals[index]
        by_sentence[sentence].append((sum(points)/len(points), index))
    ranked = []
    for sentence in range(1, 4):
        order = sorted(by_sentence[sentence], key=lambda item: (-item[0], item[1]))
        strength = round(sum(value for value, _ in order[:6])/6, 7)
        ranked.append({"sentence": sentence, "score": strength,
                       "words": sorted(int(index) for _, index in order[:math.ceil(.6*len(order))])})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[0] if ranked[0]["score"] != ranked[1]["score"] else None


def score(rows, truth, tokenizer, model, head, device):
    import torch

    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            batch = rows[start:start+16]
            encoded = [encode(row, tokenizer) for row in batch]
            qualified = [item for item in encoded if item is not None]
            values = iter(forward(model, head, qualified, device).cpu().tolist() if qualified else [])
            for row, value in zip(batch, encoded, strict=True):
                answer = select(value, next(values)) if value else None
                record = {key: row[key] for key in ("role", "group", "family", "sha256", "spans", "truth", "position")}
                if row["domain"] in {"news", "abstracts"}:
                    record["domain"] = row["domain"]
                out.append(record | {"word_ranges": [[a, b, s] for a, b, s, _ in items(row)],
                              "changed_words": sorted(truth.get(row["sha256"], ())), "assessed": value is not None, "best": answer})
    return out


def points(calibration, *, minimum=64):
    ordinary = sorted((row["best"]["score"] for row in calibration if row["best"]), reverse=True)
    if len(ordinary) < minimum or any(row["truth"] is not None or row["family"] != "original" for row in calibration):
        raise ValueError("Genuine calibration coverage is incomplete")
    return {label: ordinary[len(ordinary) // denominator] for label, denominator in (("10pct", 10), ("25pct", 4))}


def counts(rows, operating, *, baseline=None):
    if baseline is None:
        original = parent.intake.read(RECEIPT)
        baseline = {(r["role"], r["group"], r["family"]): r for r in original["rows"]}
    journal = {(parent.pool.sha(row["group"]), row["role"]): row for row in parent.intake.read(parent.previous.EDITORIAL)["rows"]}
    output = defaultdict(Counter)
    for row in rows:
        if row["role"] == "calibration":
            continue
        key = row["role"], row["group"], row["family"]
        prior = baseline[key]
        if any(row[attr] != prior[attr] for attr in ("sha256", "truth", "spans")):
            raise ValueError("Candidate source differs from the baseline")
        ranges = row["word_ranges"]
        changed = set(row["changed_words"])
        if (not all(row["spans"][s][0] <= a < b <= row["spans"][s][1] for a, b, s in ranges)
                or not all(x[1] <= y[0] for x, y in zip(ranges, ranges[1:]))
                or not all(0 <= i < len(ranges) and ranges[i][2] == row["position"] for i in changed)):
            raise ValueError("Invalid word coordinates")
        if "generator" in prior:
            _, ranges_by_focus, _ = next(transfer_sources.prior.policies(prior, {}))
            selection = list(ranges_by_focus.values())
        elif row["role"] == "editorial":
            selection = [span for card in journal[row["group"], row["family"]]["cards"] for span in card["spans"]]
        else:
            selection = parent.shipped.native_range(prior, None)
        target = row["truth"]
        existing_cards = [{i for i, (a, b, _) in enumerate(ranges) if x <= a < b <= y} for x, y in selection]
        existing = set().union(*existing_cards)
        baseline_localizes = any(useful_words(card, changed) for card in existing_cards)
        cells = [("existing", existing)]
        best = row["best"]
        if best and (not row["assessed"] or not math.isfinite(best["score"]) or not 1 <= best["sentence"] <= 3
                     or best["words"] != sorted(set(best["words"]))
                     or len(best["words"]) != math.ceil(.6 * sum(s == best["sentence"] for _, _, s in ranges))
                     or not all(0 <= i < len(ranges) and ranges[i][2] == best["sentence"] for i in best["words"])):
            raise ValueError("Invalid selected writing span")
        for name, point in operating.items():
            active = best is not None and best["score"] > point
            cells.append((name + "/sentence", {i for i, (_, _, s) in enumerate(ranges) if active and s == best["sentence"]}))
            cells.append((name + "/words", set(best["words"]) if active else set()))
        category = row["family"] if row["role"] == "editorial" else "ordinary" if target is None else row["family"]
        for arm, selected in cells:
            overlap = len(selected & changed)
            cell = output[row["role"], category, arm]
            cell.update(files=1, flagged=bool(selected), assessed=row["assessed"] if arm != "existing" else 1,
                        changed_words=len(changed), selected_words=len(selected), matched_words=overlap)
            if target is not None:
                localized = baseline_localizes if arm == "existing" else useful_words(selected, changed)
                cell.update(word_locations=localized, missed_or_misplaced=not localized,
                            new_location=localized and not baseline_localizes,
                            wrong_sentence=bool(selected) and all(ranges[i][2] != row["position"] for i in selected))
            else:
                cell["new_ordinary_flag"] += bool(selected and not existing)
    return {"/".join(key): dict(val) for key, val in sorted(output.items())}


def useful_words(selected, changed):
    overlap = len(selected & changed)
    return bool(changed) and 2*overlap >= len(changed) and 10*overlap >= 3*len(selected | changed)


def transfer_plan(training_roster, matched, additional):
    from scripts.writing_interval_transfer import fingerprints

    training, _, _ = input_plan(training_roster)
    exposed_groups = {row["group"] for row in training}
    exposed_sequences = {piece for row in training for piece in fingerprints(row["text"])}
    truth, admitted, rejected = {}, [], Counter()
    for split, path in (("point", matched), ("held", matched), ("transfer", additional)):
        groups = defaultdict(list)
        for row in transfer_sources.roster_rows(path, split):
            groups[row["group"]].append(row)
        for group, rows in groups.items():
            require_size = 1 if split == "point" else 2
            if len(rows) != require_size or sum(row["truth"] is None for row in rows) != 1:
                raise ValueError("A transfer source group is incomplete")
            phrases = {piece for row in rows for piece in fingerprints(row["text"])}
            if group in exposed_groups or phrases & exposed_sequences:
                rejected[f"{split}/{rows[0]['domain']}"] += 1
                continue
            exposed_groups.add(group)
            exposed_sequences.update(phrases)
            original = next(row for row in rows if row["truth"] is None)
            for row in rows:
                kind = "original" if row["truth"] is None else row["generator"]
                role = "calibration" if split == "point" else split
                admitted.append(row | {"role": role, "family": kind})
                if row["truth"] is not None:
                    truth[row["sha256"]] = edit_words(original, row)
    return admitted, truth, dict(rejected)


def transfer_thresholds(calibration):
    return {genre: points([row for row in calibration if row["domain"] == genre], minimum=24)
            for genre in ("news", "abstracts")}


def transfer_counts(held, source_points, domain_points):
    baseline = {}
    for split in ("held", "transfer"):
        for row in transfer_sources.indexed(split):
            kind = "original" if row["truth"] is None else row["generator"]
            baseline[split, row["group"], kind] = row
    return {genre: {"report_calibration": counts(rows, source_points, baseline=baseline),
                    "genre_calibration": counts(rows, domain_points[genre], baseline=baseline)}
            for genre in domain_points if (rows := [row for row in held if row["domain"] == genre])}


def transfer(args):
    import torch
    from safetensors.torch import load_file

    candidate = parent.intake.read(args.fitted / "receipt.json")
    weights = args.fitted / "trained.safetensors"
    if parent.pool.file_hash(weights) != candidate["weights_sha256"] or candidate["freeze"]["method"] != METHOD:
        raise ValueError("The fitted word head differs")
    rows, truth, rejected = transfer_plan(args.roster, args.matched, args.additional)
    args.output.mkdir(mode=0o700)
    freeze = {"schema": SCHEMA, "method": "Compare report calibration with separately frozen news and abstract "
              "points on matching ordinary excerpts. Use the original fitted head. Exclude old work IDs and shared "
              "normalized 12-word stretches across all source roles before inference.",
              "parents": transfer_sources.committed(), "excluded_groups": rejected,
              "fit_receipt_sha256": receipt_digest(candidate),
              "planned": [row["sha256"] for row in rows]}
    parent.previous.write(args.output / "freeze.json", freeze)
    torch.set_num_threads(4)
    tokenizer, model, head, device = parent.init_encoder(args.encoder)
    params = load_file(weights)
    for prefix, module in (("encoder.", model), ("head.", head)):
        module.load_state_dict({k.removeprefix(prefix): v for k, v in params.items() if k.startswith(prefix)}, strict=True)
    cal = score([row for row in rows if row["role"] == "calibration"], truth, tokenizer, model, head, device)
    fitted_points = transfer_thresholds(cal)
    parent.previous.write(args.output / "points.json", {"report": candidate["points"], "by_genre": fitted_points, "calibration": cal})
    held = score([row for row in rows if row["role"] != "calibration"], truth, tokenizer, model, head, device)
    result = {"freeze": freeze, "calibration": cal, "held": held, "report_points": candidate["points"],
              "genre_points": fitted_points, "outcomes": transfer_counts(held, candidate["points"], fitted_points)}
    parent.previous.write(args.output / "receipt.json", result)
    print(f"Measured {len(cal)} genuine calibration excerpts and {len(held)} held files", flush=True)


def run(args):
    import torch
    from safetensors.torch import save_file

    args.output.mkdir(mode=0o700)
    rows, truth, baseline = input_plan(args.roster)
    freeze = fit_contract(baseline)
    parent.previous.write(args.output / "freeze.json", freeze)
    torch.manual_seed(parent.SEED)
    torch.set_num_threads(4)
    tokenizer, model, head, device = parent.init_encoder(args.encoder)
    groups, losses = fit(rows, truth, tokenizer, model, head, device)
    weights = args.output / "trained.safetensors"
    save_file({"encoder." + k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
              | {"head." + k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, weights)
    cal = score([row for row in rows if row["role"] == "calibration"], truth, tokenizer, model, head, device)
    operating = points(cal)
    parent.previous.write(args.output / "points.json", {"freeze": freeze, "weights_sha256": parent.pool.file_hash(weights),
                                                        "points": operating, "calibration": cal})
    held = score([row for row in rows if row["role"] in {"held", "editorial"}], truth, tokenizer, model, head, device)
    result = {"freeze": freeze, "device": device, "fit_groups": groups, "losses": losses, "weights_sha256": parent.pool.file_hash(weights),
              "calibration": cal, "points": operating, "held": held, "outcomes": counts(held, operating)}
    parent.previous.write(args.output / "receipt.json", result)


def verify_sources(record, rows, truth):
    native = [row for row in rows if row["role"] != "fit"]
    saved = record["calibration"] + record["held"]
    expected = {(row["role"], row["group"], row["family"]): row for row in native}
    actual = {(row["role"], row["group"], row["family"]): row for row in saved}
    if len(expected) != len(native) or len(actual) != len(saved) or set(expected) != set(actual):
        raise ValueError("Recorded word locations omit or duplicate a frozen source")
    for key, item in expected.items():
        row = actual[key]
        ranges = [[a, b, s] for a, b, s, _ in items(item)]
        if (any(row[field] != item[field] for field in ("sha256", "position", "spans", "truth"))
                or row.get("domain", item["domain"]) != item["domain"] or row["word_ranges"] != ranges
                or row["changed_words"] != sorted(truth.get(item["sha256"], ()))):
            raise ValueError("Recorded word locations differ from the native source wording")


def verify_primary(result):
    freeze = result["freeze"]
    baseline = parent.intake.read(RECEIPT)
    rows = result["calibration"] + result["held"]
    keys = ("role", "group", "family", "sha256")
    expected = {tuple(row[k] for k in keys) for row in baseline["rows"]}
    observed = {tuple(row[k] for k in keys) for row in rows}
    if (freeze != fit_contract(baseline)
            or len(observed) != len(rows) or expected != observed or result["fit_groups"] < 600
            or len(result["losses"]) != EPOCHS or not all(map(math.isfinite, result["losses"]))
            or points(result["calibration"]) != result["points"] or counts(result["held"], result["points"]) != result["outcomes"]):
        raise ValueError("The published results do not reproduce")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    fit_command = commands.add_parser("run")
    for path in ("roster", "encoder", "output"):
        fit_command.add_argument(path, type=Path)
    review_command = commands.add_parser("review")
    review_command.add_argument("receipt", type=Path)
    review_command.add_argument("--roster", type=Path)
    confirm = commands.add_parser("transfer")
    for field in ("roster", "encoder", "fitted", "matched", "additional", "output"):
        confirm.add_argument(field, type=Path)
    transfer_review = commands.add_parser("review-transfer")
    for field in ("receipt", "fit_receipt"):
        transfer_review.add_argument(field, type=Path)
    for field in ("roster", "matched", "additional"):
        transfer_review.add_argument(f"--{field}", type=Path)
    args = parser.parse_args()
    if args.phase == "run":
        run(args)
    elif args.phase == "transfer":
        transfer(args)
    elif args.phase == "review-transfer":
        result, fit_result = (parent.intake.read(path) for path in (args.receipt, args.fit_receipt))
        if (result["freeze"]["parents"] != transfer_sources.committed()
                or result["freeze"]["fit_receipt_sha256"] != receipt_digest(fit_result)
                or [row["sha256"] for row in result["calibration"] + result["held"]] != result["freeze"]["planned"]
                or result["report_points"] != fit_result["points"] or transfer_thresholds(result["calibration"]) != result["genre_points"]
                or transfer_counts(result["held"], result["report_points"], result["genre_points"]) != result["outcomes"]):
            raise ValueError("The separate recalibration or transfer outcomes differ")
        if any((args.roster, args.matched, args.additional)):
            if not all((args.roster, args.matched, args.additional)):
                parser.error("Source verification needs the training and both transfer rosters")
            rows, truth, rejected = transfer_plan(args.roster, args.matched, args.additional)
            verify_sources(result, rows, truth)
            if rejected != result["freeze"]["excluded_groups"]:
                raise ValueError("The transfer source exclusions differ")
        print("Verified independent genre calibration and held outcomes.")
    else:
        result = parent.intake.read(args.receipt)
        verify_primary(result)
        if args.roster:
            rows, truth, _ = input_plan(args.roster)
            verify_sources(result, rows, truth)
        print("Verified word-level localization, ordinary controls and source ranges.")
