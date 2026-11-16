# SPDX-License-Identifier: Apache-2.0
"""Compare adjacent and nonadjacent word evidence for locating short rewrites."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from scripts import writing_editorial_negative_transfer as previous
from scripts.writing_editorial_direction_transfer import PANEL, exposure_paths, exposure_snapshot, require

SCHEMA = "chipmunk.writing-word-neighbors/1"
PARENT = PANEL / "writing-editorial-direction-2026-11-15.json.gz"
ABSTRACT_PINS = {
    "gemini-2.5-flash": {"split": "test", "sha256": "2c626b0597f6082908a0183b31c48884a45d310fea8bb1a84043ed2432568b30"},
    "gpt-5.4": {"split": "dev", "sha256": "ea88c02630e2bc02ccdf239be81172aec0c1d91956b9088073141c63a83f8191"},
}
ABSTRACT_LIMIT = 24
EXPOSURE_COMMIT = "ee2171f024f7d36c51a0a892fc101aa361dde2e3"
ARMS = ("ai_only", "nonadjacent_words", "adjacent_words", "mean_with_incumbent")
CAPTURED = ARMS[:3]
METHOD = {
    "input": "Use the frozen isolated sentence encoder, unchanged word offsets and exact changed-word labels.",
    "head": "Project each word to 24 features and combine left, center and right features with one 32-unit hidden layer. "
            "Use the center feature in place of a missing edge neighbor. The matched control rotates both neighbors "
            "by half the sentence before inference and training. Keep the incumbent fit and compare the per-word mean "
            "of the two adjacent-word and incumbent scores as a separate arm.",
    "fit": "Use the same original and edited sentences, 1:1 class weight, 10 epochs, 512-word batches and optimizer "
           "settings as the prior AI-only head. All values used to normalize features come from the fit role.",
    "ranking": "Rank by the mean of the six strongest responses in an interior sentence. Highlight the strongest "
               "60 percent of words there. Reuse the useful-overlap and ordinary-only calibration protocol, "
               "with separate 10 and 25 percent operating points per arm and genre.",
    "transfer": "Freeze up to 24 unused five-sentence abstract pairs per generator, up to eight per interior position, "
                "from pinned test and development shards. Select by work hash before scoring and then reserve 36 ordinary works "
                "for genre calibration. Exclude known source works and normalized 12-word matches.",
    "analysis": "Recount correct ranges, same-sentence alerts on the matched ordinary versions, correct sentence "
                "selection, useful ranges with the exact sentence supplied, and false alarms on historical revisions.",
}


def freeze_abstracts(folder, output, *, reference_receipt=None):
    from scripts import writing_focus_inside_context as intake

    if reference_receipt is None:
        paths = sorted(path for path in PANEL.glob("*.json*") if path.name.startswith(("writing-", "text-", "document-", "email-"))
                       and path.name.endswith((".json", ".json.gz")))
    else:
        paths = exposure_paths(previous.read(reference_receipt)["fresh"]["exposures"])
    history, known = exposure_snapshot(paths)
    require(PARENT.name in history, "The exposure inventory must include the retained source plans")
    sources = {family: intake.shard(folder / f"abstracts_{family}.csv", "abstracts", family, pin["sha256"], split=pin["split"])
               for family, pin in ABSTRACT_PINS.items()}
    used = {phrase for table in sources.values() for key, variants in table.items()
            if previous.sha(key) in known and "v0" in variants for phrase in intake.fingerprints(variants["v0"]["text"])}
    rows = []
    for family, table in (*sources.items(), ("calibration", {key: rows for source in sources.values() for key, rows in source.items()})):
        counts = Counter()
        is_control = family == "calibration"
        need = 36 if is_control else ABSTRACT_LIMIT
        for work in sorted(table, key=lambda key: previous.sha(f"{SCHEMA}/abstracts/{family}/{key}")):
            group = previous.sha(work)
            if group in known or "v0" not in table[work]:
                continue
            phrases = set(intake.fingerprints(table[work]["v0"]["text"]))
            if not phrases or phrases & used:
                continue
            try:
                if is_control:
                    raw, marks = intake.pool.truth_text(table[work]["v0"])
                    require(not marks, "The ordinary calibration text contains edited labels")
                    normalized = (intake.pool.normalize(raw, window) for window in intake.pool.source_windows(raw))
                    text, spans = next(normalized, (None, None))
                    selections = {0: ((text, spans, None),)} if text else {}
                else:
                    selections = intake.excerpt_options(work, table[work])
            except (KeyError, ValueError):
                continue
            slots = [slot for slot in selections if is_control or counts[slot] < ABSTRACT_LIMIT // 3]
            if not slots:
                continue
            position = min(slots, key=lambda slot: (counts[slot], slot))
            rowset = [{"group": group, "genre": "abstracts", "family": "original" if version is None else family,
                       "delivery": "original" if version is None else "edited", "position": position,
                       "sha256": previous.sha(text), "text": text, "spans": spans}
                      for text, spans, version in selections[position]]
            for row in rowset:
                row["changed"] = sorted(previous.words.edit_words(rowset[0], row)) if row["delivery"] == "edited" else []
            if any(row["delivery"] == "edited" and not row["changed"] for row in rowset):
                continue
            excerpts = {phrase for row in rowset for phrase in intake.fingerprints(row["text"])}
            if excerpts & used:
                continue
            rows.extend(previous.source_record(row) | {"role": "calibration" if is_control else "held"} for row in rowset)
            used.update(phrases | excerpts)
            known.add(group)
            counts[position] += 1
            if counts.total() == need:
                break
        print(f"Abstracts {family} frozen source groups: {dict(sorted(counts.items()))}", flush=True)
        require(counts.total() == need, "The pinned abstract sources lack sufficient eligible works")
    previous.write(output, {"schema": SCHEMA, "base": EXPOSURE_COMMIT, "sources": ABSTRACT_PINS, "exposures": history, "rows": rows})


def neighbor_positions(indices, *, adjacent):
    length = len(indices)
    shift = 0 if adjacent else length // 2
    return [[indices[(max(0, i - 1) + shift) % length], center,
             indices[(min(length - 1, i + 1) + shift) % length]] for i, center in enumerate(indices)]


def fit_neighbors(plan, store, mean, scale, *, adjacent):
    import torch
    from torch import nn
    from torch.nn.functional import binary_cross_entropy_with_logits as bce

    # Use only paired fit works with both tokenizer inputs covered.
    samples = []
    for first, second in zip(plan["fit"][::2], plan["fit"][1::2], strict=True):
        require(first["group"] == second["group"] and
                {first["delivery"], second["delivery"]} == {"original", "edited"}, "Fit pair order changed")
        pair = [list(previous.features(row, store)) for row in (first, second)]
        if all(pair):
            samples.extend((matched[0][1], matched[0][2], row["changed"])
                           for matched, row in zip(pair, (first, second), strict=True))

    features, labels, positions = [], [], []
    start = 0
    for indices, values, changed in samples:
        truth = set(changed)
        features.append((values - mean) / scale)
        labels.extend(float(index in truth) for index in indices)
        end = start + len(indices)
        positions.extend(neighbor_positions(range(start, end), adjacent=adjacent))
        start = end
    values, target, context = torch.cat(features), torch.tensor(labels), torch.tensor(positions)
    positive = target.bool()
    weight = torch.where(positive, len(target) / (2 * positive.sum()), len(target) / (2 * (~positive).sum()))
    torch.manual_seed(previous.SEED)
    head = nn.Sequential(nn.Linear(values.shape[1], 24), nn.GELU(), nn.Flatten(start_dim=1),
                         nn.Linear(72, 32), nn.GELU(), nn.Linear(32, 1))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    for epoch in range(previous.EPOCHS):
        order = torch.randperm(len(target), generator=torch.Generator().manual_seed(previous.SEED + epoch))
        for batch in order.split(512):
            predicted = head(values[context[batch]]).flatten()
            loss = bce(predicted, target[batch], weight=weight[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    print(f"Fit {'adjacent' if adjacent else 'nonadjacent'} context on {len(target)} words", flush=True)
    return head.eval(), {"ai": int(positive.sum()), "same_work": int((~positive).sum())}


def measure(rows, store, fits):
    import torch

    mean, scale, incumbent, nonadjacent, adjacent = fits
    output = {}
    with torch.inference_mode():
        for row in rows:
            scored = list(previous.features(row, store))
            signals = None
            if len(scored) == len(row["sentences"]):
                signals = {arm: [] for arm in CAPTURED}
                for _, indices, values in scored:
                    scaled = (values - mean) / scale
                    relative = range(len(indices))
                    for arm, head in zip(CAPTURED, (incumbent, nonadjacent, adjacent), strict=True):
                        inputs = scaled
                        if arm != "ai_only":
                            positions = torch.tensor(neighbor_positions(relative, adjacent=arm == "adjacent_words"))
                            inputs = scaled[positions]
                        signals[arm].extend(zip(indices, head(inputs).sigmoid().flatten().tolist(), strict=True))
            require(row["sha256"] not in output or output[row["sha256"]] == signals, "Identical input changed its word responses")
            output[row["sha256"]] = signals
    return output


def per_sentence(words, scores):
    groups = {}
    for index, score in scores:
        groups.setdefault(words[index][2], []).append((score, index))
    proposals = []
    for sentence, values in groups.items():
        ordered = sorted(values, key=lambda item: (-item[0], item[1]))
        length = math.ceil(len(values) * 0.6)
        proposals.append({"sentence": sentence, "score": round(sum(value for value, _ in ordered[:6]) / 6, 7),
                          "indices": sorted(index for _, index in ordered[:length])})
    return proposals


def reading(row, signals):
    require(row["complete"] == (signals is not None), "Exact word coverage changed")
    if signals is None:
        return dict.fromkeys(ARMS), dict.fromkeys(ARMS, False)
    require(set(signals) == set(CAPTURED), "A word comparison arm is missing")
    original = [index for index, _ in signals[ARMS[0]]]
    require(all(type(index) is int and 0 <= index < len(row["words"]) for index in original), "Invalid word position")
    sections = {row["words"][index][2] for index in original}
    require(sections and original == [i for i, word in enumerate(row["words"]) if word[2] in sections]
            and ("inspected" not in row or sorted(sections) == row["inspected"]),
            "Scored positions must cover each measured sentence without gaps")
    require(all([index for index, _ in values] == original and all(0 <= value <= 1 for _, value in values)
                for values in signals.values()), "Word responses disagree on source positions or fall outside their range")
    combined = [(index, (left + right) / 2) for (index, left), (_, right) in zip(signals[ARMS[0]], signals[ARMS[2]], strict=True)]
    candidates = {arm: per_sentence(row["words"], values) for arm, values in (signals | {ARMS[3]: combined}).items()}
    selected = {arm: min(proposals, key=lambda item: (-item["score"], item["sentence"]))
                for arm, proposals in candidates.items()}
    require("readings" not in row or selected[ARMS[0]] == row["readings"][ARMS[0]],
            "The incumbent response differs from the earlier native replay")
    oracle = {arm: any(previous.useful(proposal, row["changed"]) for proposal in proposals)
              for arm, proposals in candidates.items()}
    return selected, oracle


def evaluate(rows, cutoffs):
    def flag(row, selected, method, level):
        genre = "reports" if row["genre"] == "wiki" else row["genre"]
        reading = selected[method]
        return reading if reading and reading["score"] > cutoffs[genre][method][level] else None

    outputs = {}
    for arm in ARMS:
        for level in previous.LEVELS:
            controls = {row["group"]: flag(row, selected, arm, level)
                        for row, selected, _ in rows if row["delivery"] == "original"}
            total, by_family = Counter(), {}
            for row, selected, oracle in rows:
                kind = row["delivery"]
                choice = flag(row, selected, arm, level)
                incumbent = flag(row, selected, ARMS[0], level)
                total.update({kind: 1, f"{kind}_assessed": row["complete"], f"{kind}_flagged": bool(choice),
                              f"{kind}_gained_flags": bool(choice and not incumbent),
                              f"{kind}_lost_flags": bool(incumbent and not choice)})
                if kind == "edited":
                    useful = previous.useful(choice, row["changed"])
                    baseline = previous.useful(incumbent, row["changed"])
                    related = controls[row["group"]]
                    clear = useful and (not related or related["sentence"] != choice["sentence"])
                    target_sections = {row["words"][index][2] for index in row["changed"]}
                    counts = by_family.setdefault(row["family"], Counter())
                    counts.update(edits=1, assessed=row["complete"], useful=useful, clear=clear,
                                  gained=useful and not baseline, lost=baseline and not useful,
                                  useful_before_cutoff=previous.useful(selected[arm], row["changed"]),
                                  useful_with_target_sentence=oracle[arm],
                                  flagged_correct_sentence=bool(choice and choice["sentence"] in target_sections),
                                  useful_matching_ordinary_sentence=useful and not clear)
            outputs[f"{arm}/{level}"] = {"counts": dict(total), "by_family": {key: dict(value) for key, value in by_family.items()}}
    return outputs


def calculate(packet, reference):
    fresh = packet["fresh"]["rows"]
    groups = {"calibration": reference["calibration"] + [row for row in fresh if row["role"] == "calibration"],
              "abstracts": [row for row in fresh if row["role"] == "held"], **reference["panels"]}
    require(set(packet["responses"]) == {row["sha256"] for rows in groups.values() for row in rows},
            "Response rows and source role commitments differ")
    measured = {name: [(row, *reading(row, packet["responses"][row["sha256"]])) for row in rows]
                for name, rows in groups.items()}
    calibration = measured.pop("calibration")
    cutoffs = {}
    for genre in ("news", "reports", "essays", "abstracts"):
        controls = [selected for row, selected, _ in calibration if row["genre"] == genre and row["complete"]]
        require(len(controls) >= 28, "Too few assessed ordinary works")
        values = {arm: sorted((selected[arm]["score"] for selected in controls), reverse=True) for arm in ARMS}
        cutoffs[genre] = {arm: {level: values[arm][math.floor(36 * rate)] for level, rate in previous.LEVELS.items()}
                          for arm in ARMS}
    return cutoffs, {name: evaluate(rows, cutoffs) for name, rows in measured.items()}


def reference():
    data = PARENT.read_bytes()
    return previous.sha(data), previous.read(PARENT)


def run(source, abstract_plan, encoder, output):
    parent_sha, earlier = reference()
    require(previous.sha(source.read_bytes()) == earlier["source_plan_sha256"], "The frozen source plan changed")
    plan = previous.read(source)
    fresh = previous.read(abstract_plan)
    require(fresh["schema"] == SCHEMA and fresh["sources"] == ABSTRACT_PINS, "Abstract plan differs")
    plan["calibration"].extend(row for row in fresh["rows"] if row["role"] == "calibration")
    plan["held"]["abstracts"] = [row for row in fresh["rows"] if row["role"] == "held"]
    store = previous.embeddings(plan, encoder)
    populations, population = previous.training_sets(plan, store)
    require(population == earlier["fit"], "Fit population differs from the incumbent")
    mean, scale, heads = previous.fit_heads(populations, use_ordinary_augmentations=False)
    control, control_count = fit_neighbors(plan, store, mean, scale, adjacent=False)
    adjacent, adjacent_count = fit_neighbors(plan, store, mean, scale, adjacent=True)
    labels = {label: population["words"][label] for label in ("ai", "same_work")}
    require(control_count == adjacent_count == labels, "Fit arms use different word labels")
    rows = [*plan["calibration"], *(row for rows in plan["held"].values() for row in rows)]
    responses = measure(rows, store, (mean, scale, heads[ARMS[0]], control, adjacent))
    fresh["rows"] = [{key: value for key, value in row.items() if key != "sentences"} |
                     {"inspected": [s["index"] for s in row["sentences"]], "complete": responses[row["sha256"]] is not None}
                     for row in fresh["rows"]]
    packet = {"schema": SCHEMA, "method": METHOD, "parent": parent_sha, "labels": labels, "responses": responses, "fresh": fresh}
    packet["points"], packet["result"] = calculate(packet, earlier)
    review(packet)
    previous.write(output, packet)
    return packet["result"]


def review(packet):
    digest, earlier = reference()
    require(packet["schema"] == SCHEMA and packet["parent"] == digest and packet["method"] == METHOD
            and packet["labels"] == {label: earlier["fit"]["words"][label] for label in ("ai", "same_work")},
            "The comparison protocol or reference changed")
    fresh = packet["fresh"]
    require(fresh["schema"] == SCHEMA and fresh["sources"] == ABSTRACT_PINS and fresh["base"] == EXPOSURE_COMMIT,
            "The pinned abstract sources changed")
    history, exposed = exposure_snapshot(exposure_paths(fresh["exposures"]))
    require(history == fresh["exposures"] and history.get(PARENT.name) == digest, "The earlier source exposure inventory changed")
    groups = {}
    for row in fresh["rows"]:
        require("sentences" not in row and "text" not in row and row["group"] not in exposed and row["genre"] == "abstracts"
                and row["inspected"] == [1, 2, 3] and all(a >= 0 and b > a and 0 <= section < 5 for a, b, section in row["words"])
                and all(0 <= i < len(row["words"]) and row["words"][i][2] in row["inspected"] for i in row["changed"]),
                "Fresh word evidence is outside its frozen input role")
        groups.setdefault(row["group"], []).append(row)
    calibration, edits = 0, Counter()
    for rows in groups.values():
        if len(rows) == 1:
            row, = rows
            require(row["role"] == "calibration" and row["delivery"] == row["family"] == "original" and not row["changed"],
                    "A new assessment work is missing its matched original")
            calibration += 1
        else:
            original, edited = rows
            require(original["role"] == edited["role"] == "held" and original["delivery"] == original["family"] == "original"
                    and not original["changed"] and edited["delivery"] == "edited" and bool(edited["changed"]),
                    "The matched transfer sources changed roles")
            target, = {edited["words"][index][2] for index in edited["changed"]}
            edits[edited["family"], target] += 1
    require(calibration == 36 and edits == {(family, index): ABSTRACT_LIMIT // 3 for family in ABSTRACT_PINS for index in (1, 2, 3)},
            "Fresh abstract denominators changed")
    points, results = calculate(packet, earlier)
    require(packet["points"] == points and packet["result"] == results, "Reported comparison does not match replay")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    roster = phases.add_parser("freeze-abstracts")
    roster.add_argument("folder", type=Path)
    roster.add_argument("output", type=Path)
    roster.add_argument("--reference", type=Path)
    native = phases.add_parser("run")
    for name in ("source", "abstract_plan", "encoder", "output"):
        native.add_argument(name, type=Path)
    portable = phases.add_parser("review")
    portable.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze-abstracts":
        freeze_abstracts(args.folder, args.output, reference_receipt=args.reference)
        return
    result = run(args.source, args.abstract_plan, args.encoder, args.output) if args.phase == "run" else review(previous.read(args.receipt))
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
