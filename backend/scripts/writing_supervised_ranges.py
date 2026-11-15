# SPDX-License-Identifier: MIT
"""Train and evaluate a range-level writing check on aligned revision pairs."""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_left
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from scripts import writing_context_response_words as context
from scripts import writing_edit_occurrence_transfer as occurrence
from scripts import writing_local_word_spans as spans
from scripts import writing_word_localization as prior

SCHEMA = "chipmunk.writing-supervised-ranges/1"
REFERENCE = spans.PRIMARY.with_name("writing-context-response-fit-2026-11-15.json.gz")
REFERENCE_TRANSFER = spans.PRIMARY.with_name("writing-context-response-transfer-2026-11-15.json.gz")
WIDTHS = (3, 5, 8, 12, 18, 24)
SEED = 781
EPOCHS = 8
RULE = {
    "decision": "Fit native word ranges directly, scoring competing ranges across the three interior sentences.",
    "features": "Use the existing full and isolated sentence word features. Project words to 32 dimensions. "
                "Read the range mean, its difference from the sentence mean, and two edge differences.",
    "ranges": "Use 3, 5, 8, 12, 18 and 24 adjacent words, plus the whole sentence, in each interior sentence.",
    "labels": "An AI revision range is positive only when it covers half the changed words and has at least .30 word IoU. "
              "Every range in an ordinary excerpt or ordinary editorial revision is negative.",
    "training": "Use the same report and editorial fit roster as the prior heads. Per file, read all positives and "
                "the eight highest-scoring negatives. Fit eight epochs and calibrate only on separate ordinary works.",
    "seed": SEED,
    "measurement": "Replay geometry, separately fitted genre points, matched controls and paired gains and losses.",
}
KEYS = context.KEYS
RANGE_REPORT = spans.PRIMARY.with_name("writing-supervised-ranges-report-2026-11-15.json.gz")
RANGE_TRANSFER = spans.PRIMARY.with_name("writing-supervised-ranges-transfer-2026-11-15.json.gz")
PAIRED_RULE = RULE | {"training": RULE["training"] + " For each generated rewrite, add a unit-margin softplus loss "
    "between the mean of its useful ranges and the four strongest ranges in the same sentence of the matched ordinary file."}


def candidates(words):
    """Yield the source word ids and the bounds of their native sentence."""
    sections = defaultdict(list)
    for i, (_, _, section, *_) in enumerate(words):
        if 1 <= section <= 3:
            sections[section].append(i)
    for section, indices in sorted(sections.items()):
        first, stop = indices[0], indices[-1] + 1
        for width in sorted({*WIDTHS, stop - first}):
            for start in range(first, stop - width + 1):
                yield section, start, start + width, first, stop


def useful(start, stop, changed):
    overlap = sum(start <= i < stop for i in changed)
    return overlap > 0 and 2 * overlap >= len(changed) and 10 * overlap >= 3 * (stop - start + len(changed) - overlap)


def ready(row, example, truth):
    import torch

    if example is None:
        return None
    words, indices, features = example
    first = indices[0]
    proposals = list(candidates(words))
    geometry = torch.tensor([proposal[1:] for proposal in proposals]) - first
    changes = truth.get(row["sha256"], ()) if row["family"] not in {"original", "before", "edited"} else ()
    labels = torch.tensor([useful(start, stop, changes) for _, start, stop, *_ in proposals])
    return features, geometry, labels, proposals


def head():
    import torch
    from torch import nn

    class RangeHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.word = nn.Sequential(nn.Linear(1152, 32), nn.GELU())
            self.range = nn.Sequential(nn.Linear(32 * 4 + 2, 32), nn.GELU(), nn.Linear(32, 1))

        def forward(self, features, geometry):
            state = self.word(features)
            prefix = torch.cat((state.new_zeros((1, 32)), state.cumsum(0)), dim=0)
            start, stop, left, right = geometry.unbind(1)
            size = (stop - start).unsqueeze(1)
            sentence_size = (right - left).unsqueeze(1)
            middle = (prefix[stop] - prefix[start]) / size
            relative = middle - (prefix[right] - prefix[left]) / sentence_size
            enter = state[start] - state[(start - 1).maximum(left)]
            exit = state[stop - 1] - state[stop.minimum(right - 1)]
            return self.range(torch.cat((middle, relative, enter, exit, size / 24, size / sentence_size), dim=1)).squeeze(1)

    return RangeHead()


def fit(inputs, truth, tokenizer, encoder, device, *, paired_margin=False):
    import torch
    from torch.nn.functional import softplus

    by_group = defaultdict(list)
    for row, item in occurrence.feature_batches(inputs, tokenizer, encoder, device, context_response=True):
        by_group[row["role"], row["group"]].append((row, ready(row, item, truth)))
    admitted, counterparts = [], {}
    for pair in by_group.values():
        if len(pair) != 2 or any(item is None for _, item in pair):
            continue
        index = len(admitted)
        admitted.extend(item for _, item in pair)
        if paired_margin:
            for i, (row, item) in enumerate(pair):
                before, original = pair[1 - i]
                if before["family"] == "original" and item[2].any():
                    mask = torch.tensor([section == row["position"] for section, *_ in original[3]])
                    if row["position"] != before["position"] or not mask.any():
                        raise ValueError("Matched fit sentences have incompatible native locations")
                    counterparts[index + i] = (index + 1 - i, mask)
    words = torch.cat([example[0] for example in admitted])
    mean, scale = words.mean(0), words.std(0).clamp_min(.001)
    del words
    examples = [((features - mean) / scale, geometry.clone(), labels.clone()) for features, geometry, labels, _ in admitted]
    positive = sum(bool(labels.any()) for _, _, labels in examples)
    if positive < 100:
        raise ValueError("The fit lacks eligible generated ranges")
    del admitted, by_group
    random = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    model = head()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    for epoch in range(EPOCHS):
        order = torch.randperm(len(examples), generator=random).tolist()
        for offset in range(0, len(order), 16):
            losses = []
            for i in order[offset:offset + 16]:
                features, geometry, labels = examples[i]
                response = model(features, geometry)
                negative = response[~labels]
                loss = softplus(negative.topk(min(8, len(negative))).values).mean()
                if labels.any():
                    loss = loss + softplus(-response[labels]).mean()
                if i in counterparts:
                    j, mask = counterparts[i]
                    baseline = model(*examples[j][:2])[mask]
                    loss = loss + softplus(1 + baseline.topk(min(4, len(baseline))).values.mean() - response[labels].mean())
                losses.append(loss)
            optimizer.zero_grad(set_to_none=True)
            torch.stack(losses).mean().backward()
            optimizer.step()
        print(f"Completed range fit epoch {epoch + 1}/{EPOCHS}", flush=True)
    coverage = {"files": len(examples), "with_useful_training_ranges": positive}
    if paired_margin:
        assert len(counterparts) == positive
        coverage["matched_training_ranges"] = len(counterparts)
    return model.eval(), mean, scale, coverage


def selected(readings):
    ranked = sorted(readings, key=lambda item: -item["score"])
    return ranked[0] if len(ranked) > 1 and ranked[0]["score"] > ranked[1]["score"] else None


def measure(inputs, truth, tokenizer, encoder, device, fitted):
    import torch

    model, mean, scale = fitted
    with torch.inference_mode():
        for row, item in occurrence.feature_batches(inputs, tokenizer, encoder, device, context_response=True):
            ref = {key: row[key] for key in ("role", "group", "family", "sha256", "spans", "position")}
            if "domain" in row:
                ref["domain"] = row["domain"]
            words = [[a, b, section] for a, b, section, _ in prior.items(row)]
            evidence = []
            if item is not None:
                features, geometry, _, possibilities = ready(row, item, {})
                choices = {}
                scores = model((features - mean) / scale, geometry).tolist()
                for (section, start, stop, *_), score in zip(possibilities, scores, strict=True):
                    rank = (-round(score, 7), stop - start, start)
                    if section not in choices or rank < choices[section]:
                        choices[section] = rank
                for section, (negative, length, start) in sorted(choices.items()):
                    evidence.append({"sentence": section, "score": -negative, "words": list(range(start, start + length))})
            yield ref | {"words": words, "changed": sorted(truth.get(row["sha256"], ())),
                         "assessed": item is not None, "by_sentence": evidence, "best": {"range": selected(evidence)}}


def operating(rows, minimum=64):
    return {"range": prior.points([row | {"best": row["best"]["range"], "truth": None} for row in rows], minimum=minimum)}


def by_domain(calibration, report_points):
    return {"report": report_points, **{domain: operating([r for r in calibration if r["domain"] == domain], minimum=24)
                                       for domain in ("news", "abstracts")}}


def workflow_summary(rows, points, baseline, base_points):
    from scripts.writing_context_response_words import matched_sentence_alerts, paired_outcomes

    measured = occurrence.compare(rows, points)
    exact = defaultdict(Counter)
    for row in rows:
        truth = row["changed"]
        if not truth or row["family"] in {"before", "edited", "original"}:
            continue
        exact[row["family"]].update(files=1, assessed=row["assessed"])
        if item := next((item for item in row["by_sentence"] if item["sentence"] == row["position"]), None):
            section, start, stop = item["sentence"], item["words"][0], item["words"][-1] + 1
            possible = any(useful(a, b, truth) for part, a, b, *_ in candidates(row["words"]) if part == section)
            exact[row["family"]].update(geometry_supports=possible, target_sentence_useful=useful(start, stop, truth))
    by_source = {(r["role"], r["group"], r["family"]): r for r in baseline}
    paired = []
    for reading in rows:
        earlier = by_source.pop((reading["role"], reading["group"], reading["family"]))
        if any(reading[key] != earlier[key] for key in KEYS):
            raise ValueError("A candidate used different source word geometry")
        earlier_best = next((earlier["best"][key] for key in ("range", "context/edit_and_origin", "edit_and_origin")
                             if key in earlier["best"]), None)
        paired.append((earlier | {"best": {"range": earlier_best}}, reading))
    if by_source:
        raise ValueError("A prior comparison work is missing")
    return {"counts": measured, "paired": paired_outcomes(paired, {"range": base_points}, points),
            "matched_sentences": matched_sentence_alerts(rows, points), "target_sentence": {k: dict(v) for k, v in exact.items()}}


def review(path, report=None):
    return review_record(spans.read(path), report)


def validate_range_row(row):
    items = row["by_sentence"]
    if [item["sentence"] for item in items] != ([1, 2, 3] if row["assessed"] else []):
        raise ValueError("Missing sentence response")
    legal = {(part, a, b) for part, a, b, *_ in candidates(row["words"])}
    for item in items:
        words = item["words"]
        if (not words or words != list(range(words[0], words[-1] + 1))
                or (item["sentence"], words[0], words[-1] + 1) not in legal or not math.isfinite(item["score"])):
            raise ValueError("The proposed words are outside the native range")
    if row["best"] != {"range": selected(items)}:
        raise ValueError("The selected span does not match the sentence responses")


def review_record(record, report=None):
    baseline = spans.read(REFERENCE_TRANSFER if report else REFERENCE)
    expected_phase = "transfer" if report else "report"
    if (record["schema"] != SCHEMA or record["phase"] != expected_phase or record["rule"] not in (RULE, PAIRED_RULE)
            or record["baseline_sha256"] != prior.receipt_digest(baseline)):
        raise ValueError("The comparison inputs differ")
    for row in record["calibration"] + record["held"]:
        validate_range_row(row)
    if report:
        review(report)
        primary = spans.read(report)
        if record["head_sha256"] != primary["head_sha256"] or record["rule"] != primary["rule"]:
            raise ValueError("The transfer used different fitted weights")
        points = by_domain(record["calibration"], primary["points"])
        summary = {}
        for genre in ("news", "abstracts", "journals"):
            rows = [row for row in record["held"] if row["domain"] == genre]
            before = [row for row in baseline["held"] if row["domain"] == genre]
            use_genre = genre if genre in points else "report"
            summary[genre] = workflow_summary(rows, points[use_genre], before, baseline["points"][use_genre]["context/edit_and_origin"])
    else:
        if record["fit"] != baseline["fit"] or record["encoder_sha256"] != baseline["encoder_sha256"]:
            raise ValueError("The training roster or encoder changed")
        points = operating(record["calibration"])
        summary = workflow_summary(record["held"], points, baseline["held"], baseline["points"]["edit_and_origin"])
    if record["rule"] == PAIRED_RULE:
        earlier = spans.read(RANGE_TRANSFER if report else RANGE_REPORT)
        if record["range_reference_sha256"] != prior.receipt_digest(earlier):
            raise ValueError("The earlier range comparison differs")
        if report:
            paired = {}
            for genre in ("news", "abstracts", "journals"):
                domain = genre if genre in points else "report"
                paired[genre] = workflow_summary([r for r in record["held"] if r["domain"] == genre], points[domain],
                    [r for r in earlier["held"] if r["domain"] == genre], earlier["points"][domain]["range"])
        else:
            paired = workflow_summary(record["held"], points, earlier["held"], earlier["points"]["range"])
        summary = {"against_context_words": summary, "against_unpaired_ranges": paired}
    for now, past in zip(record["calibration"], baseline["calibration"], strict=True):
        if any(now[key] != past[key] for key in KEYS):
            raise ValueError("The genuine calibration work or its geometry changed")
    if points != record["points"] or "summary" in record and summary != record["summary"]:
        raise ValueError("The operating points or outcomes do not reproduce")
    return summary


def joint_comparison(candidate, reference, baseline_key, baseline_points):
    """Recount both checks at a single ordinary-work alert point."""
    def aligned(phase):
        before = {(r["role"], r["group"], r["family"]): r for r in reference[phase]}
        for row in candidate[phase]:
            old = before[row["role"], row["group"], row["family"]]["best"][baseline_key]
            yield row, (old, row["best"]["range"])

    def rank(reading, ordered):
        return Fraction(bisect_left(ordered, reading["score"]) + 1, len(ordered) + 1) if reading else Fraction(0)

    def banks(calibration):
        return [sorted(item["score"] if item else -math.inf for item in column)
                for column in zip(*calibration, strict=True)]

    calibration = [readings for _, readings in aligned("calibration")]
    # The ordinary score for each file is ranked against the other works.
    maxima = []
    for i, pair in enumerate(calibration):
        remaining = banks(calibration[:i] + calibration[i + 1:])
        maxima.append(max(rank(item, pool) for item, pool in zip(pair, remaining, strict=True)))
    maxima.sort(reverse=True)
    operations = {label: maxima[len(maxima) // divisor] for label, divisor in (("10pct", 10), ("25pct", 4))}
    reference_scores = banks(calibration)
    output = {}
    for level, point in operations.items():
        strata = defaultdict(Counter)
        file_flags = {}
        for row, (old, new) in aligned("held"):
            flagged = [item for item, pool in zip((old, new), reference_scores, strict=True) if rank(item, pool) > point]
            ordinary = row["family"] in {"before", "edited", "original"}
            prior_location = context.selected(row | {"best": {"range": old}}, "range", baseline_points[level])
            joint_location = bool(flagged) if ordinary else any(
                prior.useful_words(set(item["words"]), set(row["changed"])) for item in flagged)
            strata[row["family"]].update(files=1, prior=int(prior_location), combined=int(joint_location),
                                         gained=int(joint_location and not prior_location),
                                         lost=int(prior_location and not joint_location),
                                         alerts=int(bool(flagged)))
            file_flags[row["group"], row["family"]] = {item["sentence"] for item in flagged}
        matched = Counter()
        for row, _ in aligned("held"):
            if row["family"] in {"before", "edited", "original"}:
                continue
            if (row["position"] in file_flags[row["group"], row["family"]]
                    and row["position"] in file_flags.get((row["group"], "original"), set())):
                matched[row["family"]] += 1
        output[level] = {"point": str(point), "counts": {name: dict(values) for name, values in strata.items()},
                         "matched_sentence_alerts": dict(matched)}
    return output


def combined(path, report=None):
    review(path, report)
    candidate = spans.read(path)
    reference = spans.read(REFERENCE_TRANSFER if report else REFERENCE)
    if report is None:
        return joint_comparison(candidate, reference, "edit_and_origin", reference["points"]["edit_and_origin"])
    return {domain: joint_comparison(
        {phase: [r for r in candidate[phase] if r["domain"] == domain] for phase in ("calibration", "held")},
        {phase: [r for r in reference[phase] if r["domain"] == domain] for phase in ("calibration", "held")},
        "context/edit_and_origin", reference["points"][domain]["context/edit_and_origin"])
        for domain in ("news", "abstracts")}


def run(args):
    import torch
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(4)
    native, expected, _ = prior.input_plan(args.roster)
    baseline = spans.read(REFERENCE)
    if prior.parent.pool.file_hash(args.encoder_weights) != baseline["encoder_sha256"]:
        raise ValueError("The frozen word encoder differs")
    tokenizer, encoder, _, device = prior.parent.init_encoder(args.encoder)
    raw = load_file(args.encoder_weights)
    encoder.load_state_dict({k.removeprefix("encoder."): v for k, v in raw.items() if k.startswith("encoder.")}, strict=True)
    encoder.eval()
    del raw
    transfer = args.phase == "transfer"
    if transfer:
        reference = spans.read(REFERENCE_TRANSFER)
        primary = spans.read(args.fitted / "receipt.json")
        rows, truth, _ = prior.transfer_plan(args.roster, args.matched, args.additional)
        ordinary, genuine, _ = occurrence.ordinary_inputs(args.editorial)
        rows += ordinary
        truth.update(genuine)
        weights = args.fitted / "range.safetensors"
        if prior.parent.pool.file_hash(weights) != primary["head_sha256"]:
            raise ValueError("The previously fitted range weights differ")
        values = load_file(weights)
        model = head()
        model.load_state_dict({k.removeprefix("head."): v for k, v in values.items() if k.startswith("head.")}, strict=True)
        fitted = model.eval(), values["mean"], values["scale"]
        hold = [row for row in rows if row["role"] != "calibration"]
        extra = {}
    else:
        reference = baseline
        rows, truth = native, expected
        hold = [row for row in rows if row["role"] == "held"]
        fit_rows = occurrence.fit_roster(rows)
        identities = [[row[key] for key in ("role", "group", "family", "sha256")] for row in fit_rows]
        if identities != baseline["fit"]:
            raise ValueError("The fit did not use the original source works")
        model, mean, scale, coverage = fit(fit_rows, truth, tokenizer, encoder, device, paired_margin=args.paired_margin)
        fitted = model, mean, scale
        weights = args.output / "range.safetensors"
        save_file({"mean": mean, "scale": scale, **{f"head.{k}": v for k, v in model.state_dict().items()}}, weights)
        extra = {"fit": identities, "coverage": coverage}
    calibration = list(measure([row for row in rows if row["role"] == "calibration"], truth, tokenizer, encoder, device, fitted))
    points = by_domain(calibration, primary["points"]) if transfer else operating(calibration)
    prior.parent.previous.write(args.output / "points.json", points)
    held = list(measure(hold, truth, tokenizer, encoder, device, fitted))
    rule = primary["rule"] if transfer else PAIRED_RULE if args.paired_margin else RULE
    if rule == PAIRED_RULE:
        extra["range_reference_sha256"] = prior.receipt_digest(spans.read(RANGE_TRANSFER if transfer else RANGE_REPORT))
    result = {"schema": SCHEMA, "phase": args.phase, "rule": rule, "baseline_sha256": prior.receipt_digest(reference),
              "encoder_sha256": baseline["encoder_sha256"], "head_sha256": prior.parent.pool.file_hash(weights),
              "calibration": calibration, "held": held, "points": points, **extra}
    result["summary"] = review_record(result, args.fitted / "receipt.json" if transfer else None)
    prior.parent.previous.write(args.output / "receipt.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("report", "transfer", "review", "review-combined"))
    parser.add_argument("output", type=Path)
    for option in ("roster", "encoder", "encoder-weights", "fitted", "matched", "additional", "editorial", "report"):
        parser.add_argument(f"--{option}", type=Path)
    parser.add_argument("--paired-margin", action="store_true", help="Train against the corresponding ordinary sentence")
    args = parser.parse_args()
    if args.phase in {"review", "review-combined"}:
        action = review if args.phase == "review" else combined
        print(json.dumps(action(args.output, args.report), sort_keys=True, indent=2))
    else:
        needed = ["roster", "encoder", "encoder_weights"]
        if args.phase == "transfer":
            needed += ["fitted", "matched", "additional", "editorial"]
        missing = [name.replace("_", "-") for name in needed if getattr(args, name) is None]
        if missing:
            parser.error("Missing options: " + ", ".join(f"--{name}" for name in missing))
        args.output.mkdir(mode=0o700)
        run(args)
