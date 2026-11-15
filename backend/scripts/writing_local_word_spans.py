# SPDX-License-Identifier: MIT
"""Compare calibrated contiguous word ranges with the frozen word-level head."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from scripts import writing_word_localization as previous

parent = previous.parent
PANELS = parent.pool.PANEL_DIR
PRIMARY = PANELS / "writing-word-localization-2026-11-14.json"
TRANSFER = PANELS / "writing-word-localization-transfer-2026-11-14.json"
SCHEMA = "chipmunk.writing-local-word-spans/1"
SIZES = (3, 5, 8, 12, 18, 24)
ARMS = ("absolute", "relative_to_adjacent")
REFERENCE_LIMIT = 144
METHOD = {
    "intervals": list(SIZES),
    "selection": "Take the highest contiguous mean at each size across the three interior sentences. Test the "
                 "absolute mean and the difference from the strongest same-size range in either adjacent sentence. "
                 "When a neighbor is shorter, use its full mean. Pick a unique maximum after per-size robust "
                 "normalization. Repeat sizes or modes never imply independent evidence.",
    "reference": "A fixed digest order chooses 144 original-only report fit works. For each arm and size, estimate "
                 "the median and scaled median absolute deviation of file maxima. Use a minimum scale of .10. "
                 "The encoder and head are frozen. All held sources stay outside ranking and normalization.",
    "calibration": "Use the original word study's untouched report files for 10% and 25% operating points. "
                   "Separately calibrate on the existing ordinary news and abstract files then test both genres. "
                   "The widest selection still has a comparable maximum-based point.",
    "evaluation": "Measure precise changed-word support without supplying an earlier copy to the model. "
                  "Compare original word selection, retained and lost locations, ordinary and natural edit flags, "
                  "and matched ordinary flags landing on the same source sentence. Separately give the reader the "
                  "known edit sentence after inference to measure unthresholded word localization within it.",
}


def read(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return parent.intake.read(path)


def reference_rows(rows):
    eligible = [row for row in rows if row["role"] == "fit" and row["family"] == "original"]
    return sorted(eligible, key=lambda row: parent.pool.sha(f"{SCHEMA}/{row['sha256']}"))[:REFERENCE_LIMIT]


def native_word_scores(encoded, scores):
    _, ids, words = encoded
    grouped = defaultdict(list)
    for i, score in zip(ids, scores[:len(ids)], strict=True):
        if i is not None:
            grouped[i].append(score)
    sentences = defaultdict(list)
    for i, (_, _, sentence, _) in enumerate(words):
        values = grouped[i]
        sentences[sentence].append((i, sum(values) / len(values)))
    return sentences


def windows(words, length):
    if len(words) < length:
        return []
    prefix = [0.]
    for _, value in words:
        prefix.append(prefix[-1] + value)
    return [(round((prefix[i+length] - prefix[i]) / length, 7), words[i][0])
            for i in range(len(words)-length+1)]


def maxima(sentences):
    result = {}
    for length in SIZES:
        alternatives = [windows(sentences[i], length) for i in range(5)]
        adjacent = [max((value for value, _ in spans), default=sum(v for _, v in sentences[i]) / len(sentences[i]))
                    for i, spans in enumerate(alternatives)]
        arms = {name: [] for name in ARMS}
        for i in range(1, 4):
            if not alternatives[i]:
                continue
            flanks = max(adjacent[i-1], adjacent[i+1])
            for strength, start in alternatives[i]:
                arms["absolute"].append((strength, start, i))
                arms["relative_to_adjacent"].append((round(strength - flanks, 7), start, i))
        for name, proposals in arms.items():
            if proposals:
                proposals.sort(key=lambda candidate: (-candidate[0], candidate[1]))
                strength, start, sentence = proposals[0]
                if len(proposals) == 1 or strength != proposals[1][0]:
                    result.setdefault(name, {})[str(length)] = {"score": strength, "start": start, "sentence": sentence}
    return result


def target_probe(sentences, target):
    ranks = {sentence: sorted(sentences[sentence], key=lambda word: (-word[1], word[0])) for sentence in range(1, 4)}
    strengths = {sentence: round(sum(value for _, value in words[:6]) / 6, 7) for sentence, words in ranks.items()}
    ordered = ranks[target]
    return {"best_sentence": all(strengths[target] > value for sentence, value in strengths.items() if sentence != target),
            "words": sorted(index for index, _ in ordered[:math.ceil(.6 * len(ordered))])}


def signs(rows, tokenizer, model, head):
    import torch

    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(rows), 16):
            chunk = rows[offset:offset+16]
            encoded = [previous.encode(row, tokenizer) for row in chunk]
            admitted = [item for item in encoded if item is not None]
            values = iter(previous.forward(model, head, admitted, "cpu").tolist() if admitted else [])
            for row, item in zip(chunk, encoded, strict=True):
                raw = next(values) if item is not None else None
                words = native_word_scores(item, raw) if item is not None else None
                yield {"id": [row["role"], row["group"], row["family"], row["sha256"]],
                       "coverage": item is not None,
                       "maxima": maxima(words) if words is not None else {},
                       "target_probe": target_probe(words, row["position"]) if words is not None and row["truth"] else None}, (
                           previous.select(item, raw) if item is not None else None)


def reference(rows):
    scales = {}
    for arm in ARMS:
        scaled = {}
        for length in map(str, SIZES):
            peaks = [record["score"] for row in rows if (record := row["maxima"].get(arm, {}).get(length))]
            if len(peaks) < 24:
                continue
            center = median(peaks)
            spread = max(.1, median(abs(point-center) for point in peaks) * 1.4826)
            scaled[length] = {"median": round(center, 8), "scale": round(spread, 8), "ordinary_files": len(peaks)}
        scales[arm] = scaled
    return scales


def selected(row, scales):
    output = {}
    for arm in ARMS:
        ranked = []
        for length, measured in row["maxima"].get(arm, {}).items():
            if length in scales[arm]:
                norm = scales[arm][length]
                ranked.append((round((measured["score"] - norm["median"]) / norm["scale"], 8),
                               measured["start"], int(length), measured["sentence"]))
        ranked.sort(reverse=True)
        if ranked and (len(ranked) == 1 or ranked[0][0] != ranked[1][0]):
            score, start, length, sentence = ranked[0]
            output[arm] = {"score": score, "words": list(range(start, start+length)), "sentence": sentence}
    return output


def index(records):
    keys = ("role", "group", "family", "sha256")
    return {tuple(row[key] for key in keys): row for row in records}


def verify_rows(rows, originals):
    keys = [tuple(row["id"]) for row in rows]
    if set(keys) != set(originals) or len(keys) != len(set(keys)):
        raise ValueError("Scored sources differ from the frozen word receipt")
    for row in rows:
        known = originals[tuple(row["id"])]
        if row["coverage"] != known["assessed"] or (not row["coverage"] and row["maxima"]):
            raise ValueError("Tokenizer coverage differs from the frozen word model")
        probe = row["target_probe"]
        if (probe is not None) != (known["truth"] is not None and known["assessed"]):
            raise ValueError("Target-only diagnostic is missing or applied to an ordinary case")
        if probe is not None:
            word_indices = {i for i, (_, _, sentence) in enumerate(known["word_ranges"]) if sentence == known["position"]}
            if (type(probe["best_sentence"]) is not bool or len(probe["words"]) != math.ceil(.6 * len(word_indices))
                    or not set(probe["words"]) <= word_indices or probe["words"] != sorted(set(probe["words"]))
                    or probe["best_sentence"] != bool(known["best"] and known["best"]["sentence"] == known["position"])):
                raise ValueError("Target-only diagnostic differs from the known sentence or frozen selection")
        for arm, proposals in row["maxima"].items():
            if arm not in ARMS:
                raise ValueError("Unknown scoring arm")
            for size, proposal in proposals.items():
                start, sentence = proposal["start"], proposal["sentence"]
                if (size not in map(str, SIZES) or not math.isfinite(proposal["score"]) or sentence not in (1, 2, 3)
                        or not 0 <= start <= len(known["word_ranges"]) - int(size)
                        or not all(expected == sentence for _, _, expected in known["word_ranges"][start:start+int(size)])):
                    raise ValueError("Proposed words leave their source sentence")


def operating(calibration, scales, *, minimum):
    out = {}
    chosen = [selected(row, scales) for row in calibration]
    for arm in ARMS:
        values = sorted((record[arm]["score"] for record in chosen if arm in record), reverse=True)
        if len(values) < minimum:
            raise ValueError("Too few assessed calibration works")
        out[arm] = {name: values[len(values) // denominator] for name, denominator in (("10pct", 10), ("25pct", 4))}
    return out


def compare(rows, baseline, scales, cuts, baseline_points):
    verify_rows(rows, baseline)
    cells = defaultdict(Counter)
    calls, controls = {}, defaultdict(list)
    for row in rows:
        identity = tuple(row["id"])
        known = baseline[identity]
        prior = known["best"]
        output = selected(row, scales)
        for arm, thresholds in cuts.items():
            for name, cutoff in thresholds.items():
                proposed = output.get(arm)
                active = proposed if proposed and proposed["score"] > cutoff else None
                old = prior if prior and prior["score"] > baseline_points[name] else None
                calls[identity, arm, name] = (old, active)
                if active and known["truth"] is None:
                    controls[identity[:2], arm, name].append(active["sentence"])
    for (identity, arm, name), (old, active) in calls.items():
        known = baseline[identity]
        role, group, family, _ = identity
        category = "ordinary" if role != "editorial" and family == "original" else family
        cell = cells["/".join((role, category, arm, name))]
        truth = set(known["changed_words"])
        cell.update(files=1, assessed=known["assessed"], prior_flags=old is not None, flags=active is not None)
        if known["truth"] is None:
            cell.update(additional_ordinary=active is not None and old is None, removed_ordinary=old is not None and active is None)
            continue
        old_words = set(old["words"]) if old else set()
        words = set(active["words"]) if active else set()
        prior_good = previous.useful_words(old_words, truth)
        good = previous.useful_words(words, truth)
        cell.update(prior_useful=prior_good, useful=good, added=good and not prior_good, lost=prior_good and not good,
                    changed_words=len(truth), prior_selected_words=len(old_words), selected_words=len(words),
                    prior_matched_words=len(old_words & truth), matched_words=len(words & truth))
        if good:
            cell["useful_with_control_same_sentence"] += active["sentence"] in controls[(role, group), arm, name]
    return dict(cells)


def diagnose(rows, known):
    totals = defaultdict(Counter)
    for row in rows:
        role, _, family, _ = row["id"]
        baseline = known[tuple(row["id"])]
        if baseline["truth"] is None:
            continue
        probe = row["target_probe"]
        count = totals[f"{role}/{family}"]
        count.update(files=1, assessed=probe is not None)
        if probe is not None:
            changed = set(baseline["changed_words"])
            words = set(probe["words"])
            useful = previous.useful_words(words, changed)
            count.update(best_sentence=probe["best_sentence"], useful_given_sentence=useful,
                         useful_given_sentence_and_best=useful and probe["best_sentence"])
    return dict(totals)


def readout(document):
    method = document["method"]
    if method != METHOD or document["schema"] != SCHEMA:
        raise ValueError("Localization method differs")
    reports, transfer = read(PRIMARY), read(TRANSFER)
    expected = {path.name: parent.pool.file_hash(path) for path in (PRIMARY, TRANSFER)}
    if document["inputs"] != expected or document["weights_sha256"] != reports["weights_sha256"]:
        raise ValueError("Frozen comparison receipts differ")
    refs = [tuple(row["id"]) for row in document["reference"]]
    if len(refs) != REFERENCE_LIMIT or len({identity[1] for identity in refs}) != REFERENCE_LIMIT or any(
            identity[0] != "fit" or identity[2] != "original" for identity in refs):
        raise ValueError("Ordinary size references differ from the declared source roles")
    scales = reference(document["reference"])
    report = index(reports["calibration"])
    verify_rows(document["calibration"], report)
    points = {"reports": operating(document["calibration"], scales, minimum=64)}
    prior_held = index(reports["held"])
    outcomes = {"reports_and_journals": compare(document["held"], prior_held, scales, points["reports"], reports["points"])}
    diagnostics = {"reports_and_journals": diagnose(document["held"], prior_held)}
    expected_genre_calibration = index(transfer["calibration"])
    expected_genre_held = index(transfer["held"])
    if ({tuple(row["id"]) for row in document["genre_calibration"]} != set(expected_genre_calibration)
            or {tuple(row["id"]) for row in document["genre_held"]} != set(expected_genre_held)):
        raise ValueError("Genre score rows differ from the frozen comparison")
    for genre in ("news", "abstracts"):
        calibration = [row for row in transfer["calibration"] if row["domain"] == genre]
        known = index(calibration)
        measured = [row for row in document["genre_calibration"] if tuple(row["id"]) in known]
        verify_rows(measured, known)
        points[genre] = operating(measured, scales, minimum=24)
        held = index(row for row in transfer["held"] if row["domain"] == genre)
        candidate = [row for row in document["genre_held"] if tuple(row["id"]) in held]
        outcomes[genre] = compare(candidate, held, scales, points[genre], transfer["genre_points"][genre])
        diagnostics[genre] = diagnose(candidate, held)
    return {"reference_scales": scales, "points": points, "outcomes": outcomes, "diagnostics": diagnostics}


def run(args):
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(4)
    reports = read(PRIMARY)
    native, _, _ = previous.input_plan(args.roster)
    transferred, _, rejected = previous.transfer_plan(args.roster, args.matched, args.additional)
    transfer = read(TRANSFER)
    if rejected != transfer["freeze"]["excluded_groups"]:
        raise ValueError("Genre panel exclusions differ from the frozen transfer")
    if parent.pool.file_hash(args.weights) != reports["weights_sha256"]:
        raise ValueError("The previously evaluated model bytes differ")
    tokenizer, model, head, _ = parent.init_encoder(args.encoder)
    model, head = model.cpu(), head.cpu()
    params = load_file(args.weights)
    for prefix, module in (("encoder.", model), ("head.", head)):
        module.load_state_dict({key.removeprefix(prefix): value for key, value in params.items() if key.startswith(prefix)}, strict=True)
    args.output.mkdir(mode=0o700)
    document = {"schema": SCHEMA, "method": METHOD, "inputs": {path.name: parent.pool.file_hash(path) for path in (PRIMARY, TRANSFER)},
                "weights_sha256": reports["weights_sha256"]}

    def measure(name, rows, prior_rows):
        expected = index(prior_rows)
        records = []
        for row, original in signs(rows, tokenizer, model, head):
            if name != "reference":
                known = expected[tuple(row["id"])]
                # Native CPU differences can alter final floating-point digits.
                # The comparator uses its persisted results at the original point.
                older = known["best"]
                if (older is None) != (original is None) or (older is not None and
                        (older["sentence"], older["words"]) != (original["sentence"], original["words"])):
                    raise ValueError("The previously selected words did not reproduce")
            records.append(row)
        print(f"Read {len(records)} exact native excerpts for {name}.", flush=True)
        document[name] = records

    measure("reference", reference_rows(native), [])
    measure("calibration", [row for row in native if row["role"] == "calibration"], reports["calibration"])
    measure("genre_calibration", [row for row in transferred if row["role"] == "calibration"], transfer["calibration"])
    scales = reference(document["reference"])
    points = {"reports": operating(document["calibration"], scales, minimum=64)}
    genre_reference = index(transfer["calibration"])
    for domain in ("news", "abstracts"):
        samples = [row for row in document["genre_calibration"] if genre_reference[tuple(row["id"])]["domain"] == domain]
        points[domain] = operating(samples, scales, minimum=24)
    parent.previous.write(args.output / "points.json", {"reference": scales, "points": points})
    measure("held", [row for row in native if row["role"] in {"held", "editorial"}], reports["held"])
    measure("genre_held", [row for row in transferred if row["role"] != "calibration"], transfer["held"])
    report = readout(document)
    if report["points"] != points:
        raise ValueError("calibration changed after held scoring")
    parent.previous.write(args.output / "receipt.json", document | report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    measure = phases.add_parser("run")
    for name in ("roster", "matched", "additional", "encoder", "weights", "output"):
        measure.add_argument(name, type=Path)
    audit = phases.add_parser("review")
    audit.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "run":
        run(args)
    else:
        record = read(args.receipt)
        observed = readout(record)
        if any(observed[key] != record[key] for key in observed):
            raise ValueError("The published span counts differ")
        print(json.dumps(observed, sort_keys=True, indent=2))
