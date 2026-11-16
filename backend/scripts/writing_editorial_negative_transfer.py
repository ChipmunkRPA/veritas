# SPDX-License-Identifier: Apache-2.0
"""Test whether genuine edited wording improves single-upload rewrite localization."""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_edit_occurrence_transfer as earlier
from scripts import writing_editorial_negative_source as editorial
from scripts import writing_word_localization as words
from scripts.writing_language_window_transfer import read, sha, write

SCHEMA = "chipmunk.writing-editorial-negative-transfer/1"
ARMS = ("ai_only", "with_ordinary_originals", "with_ordinary_revisions")
LEVELS = {"tight": 0.10, "wide": 0.25}
INPUTS = {
    "layer": "ac5491aa566418b9d41fa361e71c06dafa5148f425d118ce6252fe2bcce60e63",
    "ranges": "45b3065d5ea5242a3ce52f89467074e4b9b7ed30f519c341b9b468de98f06491",
}
SEED = 1841
EPOCHS = 10
LIMIT = 128
METHOD = {
    "inputs": "Read each candidate sentence alone. Pool pieces into exact words, and concatenate the word state "
              "and the difference from its sentence mean. Use the same frozen six-layer encoder in all three arms.",
    "fit": "Same AI edits and before-text in all arms. Give AI-changed words half the weight. For the two augmented "
           "arms give the matched Wikipedia source versions one quarter and other AI-case words one quarter. "
           "Use human before-text in one arm and the corresponding human-edited text in the other.",
    "score": "For a submitted excerpt, average the six strongest word responses to rank each interior sentence. "
             "Propose the highest 60% of words in the winning sentence. A useful range covers half the changed "
             "words with at least 0.30 intersection over union. No text or reference version is fetched at inference.",
    "decision": "Set independent 10% and 25% permitted flag counts on 36 English controls per genre. Use the "
                "report settings for Wikipedia as a separate transfer check. Keep previous measurements visible.",
}
BASELINE = Path("docs/evaluation-panels/writing-layer-transfer-2026-11-15.json.gz")
RANGE_BASELINE = BASELINE.with_name("writing-structured-range-transfer-2026-11-15.json.gz")


def source_record(row, *, target_only=False):
    indices = row["target"] if target_only else range(1, len(row["spans"]) - 1)
    return {"group": row["group"], "genre": row["genre"], "family": row["family"],
            "delivery": row["delivery"], "sha256": row["sha256"], "changed": row["changed"],
            "sentences": [{"index": index, "offset": row["spans"][index][0],
                           "text": row["text"][slice(*row["spans"][index])]} for index in indices],
            "words": [[a, b, section] for a, b, section, _ in words.items(row)]}


def wiki_records(rows):
    result = []
    for row in rows:
        for version in ("before", "after"):
            value = row[version]
            specs = [[match.start(), match.end(), 0] for match in words.WORD.finditer(value)]
            result.append({"group": row["page"], "genre": "wiki", "family": "historical_wiki",
                           "delivery": version, "sha256": sha(value), "changed": row["changed"] if version == "after" else [],
                           "sentences": [{"index": 0, "offset": 0, "text": value}], "words": specs})
    return result


def prepare(roster, layers, ranges, wiki, output):
    old_rows, targets, _ = words.input_plan(roster)
    ai = [row for row in earlier.fit_roster(old_rows) if row["role"] == "fit"]
    assert len(ai) == 640
    paths = {"layer": layers, "ranges": ranges}
    for name, path in paths.items():
        assert sha(path.read_bytes()) == INPUTS[name]
    data = {name: read(path) for name, path in paths.items()}
    held = {"layer": [source_record(row) for row in data["layer"]["held"]],
            "ranges": [source_record(row) for row in data["ranges"]["rows"]]}
    calibration = [source_record(row) for row in data["layer"]["calibration"]]
    reserved = data["layer"]["calibration"] + data["layer"]["held"] + data["ranges"]["rows"]
    reserved_marks = set().union(*(editorial.fingerprints(row["text"]) for row in reserved))
    excluded = {row["group"] for row in ai if editorial.fingerprints(row["text"]) & reserved_marks}
    ai = [row for row in ai if row["group"] not in excluded]
    fit = [source_record({
        **r, "genre": "reports", "target": [r["position"]],
        "delivery": "edited" if r["family"] != "original" else "original",
        "changed": sorted(targets.get(r["sha256"], ())),
    }, target_only=True) for r in ai]
    groups = [{row["group"] for row in rows} for rows in (fit, calibration, *held.values())]
    assert len(set.union(*groups)) == sum(len(group) for group in groups)
    english_marks = reserved_marks | set().union(*(editorial.fingerprints(r["text"]) for r in ai))
    source = read(wiki)
    assert source["schema"] == editorial.SCHEMA and source["archive_sha256"] == editorial.ARCHIVE_SHA256
    credits = []
    for split, rows in source["splits"].items():
        assert len(rows) == editorial.QUOTAS[split]
        for row in rows:
            assert row["changed"] == editorial.changed(row["before"], row["after"])
            assert row["timestamp"] < "2020"
            mark = editorial.fingerprints(row["before"]) | editorial.fingerprints(row["after"])
            assert not mark & english_marks
            credits.append({key: row[key] for key in ("revision", "page", "parent", "timestamp")} | {"split": split})
    assert len({credit["page"] for credit in credits}) == len(credits)
    plan = {"schema": SCHEMA, "inputs": INPUTS | {"wiki": sha(wiki.read_bytes()), "roster": sha(roster.read_bytes())},
            "fit": fit, "wiki_fit": wiki_records(source["splits"]["fit"]), "calibration": calibration,
            "held": held | {"wiki": wiki_records(source["splits"]["held"])}, "credits": credits,
            "removed_ai_fit_groups": sorted(excluded)}
    write(output, plan)


def tokenize(text, tokenizer):
    spans = [match.span() for match in words.WORD.finditer(text)]
    if len(spans) < 6:
        return None
    encoding = tokenizer(text, return_offsets_mapping=True, truncation=False, verbose=False)
    ids = encoding["input_ids"]
    if len(ids) > LIMIT:
        return None
    ends = [end for _, end in spans]
    pieces, covering = [], Counter()
    for index, (a, b) in enumerate(encoding["offset_mapping"]):
        if a >= b:
            continue
        word = bisect_right(ends, a)
        if word == len(spans) or b <= spans[word][0]:
            continue
        if not spans[word][0] <= a < b <= ends[word]:
            return None
        pieces.append((index, word))
        covering[word] += b - a
    if any(covering[index] != b - a for index, (a, b) in enumerate(spans)):
        return None
    return ids, pieces, spans


def embeddings(plan, encoder):
    import torch
    from transformers import AutoModel, AutoTokenizer

    for name, digest in words.parent.ENCODER_ASSETS.items():
        if sha((encoder / name).read_bytes()) != digest:
            raise ValueError("The pinned encoder differs")
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(encoder, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(encoder, local_files_only=True, use_safetensors=True,
                                     trust_remote_code=False, attn_implementation="eager").eval()
    all_rows = [*plan["fit"], *plan["wiki_fit"], *plan["calibration"]]
    all_rows.extend(row for rows in plan["held"].values() for row in rows)
    inputs = {sentence["text"] for row in all_rows for sentence in row["sentences"]}
    eligible = [(text, value) for text in sorted(inputs, key=sha) if (value := tokenize(text, tokenizer)) is not None]
    store = {}
    with torch.inference_mode():
        for start in range(0, len(eligible), 32):
            batch = eligible[start:start + 32]
            size = max(len(ids) for _, (ids, _, _) in batch)
            ids = torch.zeros((len(batch), size), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, (_, (sequence, _, _)) in enumerate(batch):
                ids[i, :len(sequence)] = torch.tensor(sequence)
                mask[i, :len(sequence)] = 1
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
            for i, (text, (_, pieces, positions)) in enumerate(batch):
                token, dest = map(torch.tensor, zip(*pieces, strict=True))
                states = hidden[i].new_zeros((len(positions), hidden.shape[-1])).index_add_(0, dest, hidden[i, token])
                states /= torch.bincount(dest).unsqueeze(1)
                store[text] = positions, torch.cat((states, states - states.mean(0)), dim=1).clone()
            if start % 512 == 0:
                print(f"Embedded {min(start + 32, len(eligible))}/{len(eligible)} exact sentence inputs", flush=True)
    return store


def features(row, store):
    indices = {(a, b, section): i for i, (a, b, section) in enumerate(row["words"])}
    for sentence in row["sentences"]:
        if (record := store.get(sentence["text"])) is not None:
            offset, section = sentence["offset"], sentence["index"]
            positions, values = record
            yield section, [indices[a + offset, b + offset, section] for a, b in positions], values


def training_sets(plan, store):
    import torch

    populations = defaultdict(list)
    required = defaultdict(set)
    pairs = defaultdict(list)
    for row in plan["fit"]:
        observed = list(features(row, store))
        pairs[row["group"]].append((row, observed[0] if observed else None))
    for group, pair in pairs.items():
        assert len(pair) == 2 and {row["delivery"] for row, _ in pair} == {"original", "edited"}
        if any(reading is None for _, reading in pair):
            continue
        for row, (_, indices, values) in pair:
            truth = set(row["changed"])
            mask = torch.tensor([index in truth for index in indices])
            populations["ai"].append(values[mask])
            populations["same_work"].append(values[~mask])
            if mask.any().item():
                required["ai"].add(group)
    wiki = defaultdict(dict)
    for row in plan["wiki_fit"]:
        observed = list(features(row, store))
        if observed:
            wiki[row["group"]][row["delivery"]] = observed[0][2]
    for group, pair in wiki.items():
        if set(pair) == {"before", "after"}:
            required["wiki"].add(group)
            for version, values in pair.items():
                populations[version].append(values)
    assert len(required["ai"]) >= 220 and len(required["wiki"]) >= 120
    return {name: torch.cat(values) for name, values in populations.items()}, {
        "groups": {name: len(values) for name, values in required.items()},
        "words": {name: sum(len(item) for item in rows) for name, rows in populations.items()},
        "removed_ai_fit_groups": plan["removed_ai_fit_groups"],
    }


def fit_heads(populations, *, use_ordinary_augmentations=True):
    import torch
    from torch import nn
    from torch.nn.functional import binary_cross_entropy_with_logits as bce

    reference = torch.cat([populations["ai"], populations["same_work"]])
    mean, scale = reference.mean(0), reference.std(0).clamp_min(0.025)
    del reference
    heads = {}
    choices = zip(ARMS, (None, "before", "after"), strict=True) if use_ordinary_augmentations else ((ARMS[0], None),)
    for arm, human in choices:
        torch.manual_seed(SEED)
        partitions = [("ai", 0.5), ("same_work", 0.25 if human else 0.5)]
        if human:
            partitions.append((human, 0.25))
        values = torch.cat([(populations[name] - mean) / scale for name, _ in partitions])
        labels = torch.cat([torch.full((len(populations[name]),), float(name == "ai")) for name, _ in partitions])
        weights = torch.cat([torch.full((len(populations[name]),), share * len(values) / len(populations[name]))
                             for name, share in partitions])
        head = nn.Sequential(nn.Linear(values.shape[1], 32), nn.GELU(), nn.Linear(32, 1))
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
        for epoch in range(EPOCHS):
            order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(SEED + epoch))
            for indices in order.split(512):
                logits = head(values[indices]).flatten()
                loss = bce(logits, labels[indices], weight=weights[indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        heads[arm] = head.eval()
        print(f"Fit {arm} on {len(values)} exact words", flush=True)
    return mean, scale, heads


def measure(rows, store, fitted):
    import torch

    mean, scale, heads = fitted
    output = []
    with torch.inference_mode():
        for row in rows:
            scored = list(features(row, store))
            complete = len(scored) == len(row["sentences"])
            readings = dict.fromkeys(heads)
            if complete:
                for arm, head in heads.items():
                    proposals = []
                    for section, indices, values in scored:
                        scores = head((values - mean) / scale).sigmoid().flatten().tolist()
                        order = sorted(zip(scores, indices, strict=True), key=lambda x: (-x[0], x[1]))
                        strength = round(sum(score for score, _ in order[:6]) / 6, 7)
                        highlight = sorted(index for _, index in order[:math.ceil(0.6 * len(order))])
                        proposals.append({"score": strength, "sentence": section, "indices": highlight})
                    readings[arm] = min(proposals, key=lambda p: (-p["score"], p["sentence"]))
            output.append({key: value for key, value in row.items() if key != "sentences"} | {
                "complete": complete, "readings": readings,
            })
    return output


def points(calibration):
    result = {}
    for genre in ("news", "reports"):
        rows = [row for row in calibration if row["genre"] == genre]
        assert len(rows) == 36
        result[genre] = {}
        for arm in ARMS:
            values = sorted((row["readings"][arm]["score"] for row in rows if row["complete"]), reverse=True)
            assert len(values) >= 28
            result[genre][arm] = {level: values[math.floor(len(rows) * rate)] for level, rate in LEVELS.items()}
    return result


def useful(proposal, changed):
    if proposal is None or not changed:
        return False
    target, marked = set(changed), set(proposal["indices"])
    overlap = len(target & marked)
    return overlap * 2 >= len(target) and overlap * 10 >= len(target | marked) * 3


def baseline(rows):
    history = read(BASELINE)
    matches = {row["sha256"]: row for row in history["rows"]}
    result = {}
    for row in rows:
        assert matches[row["sha256"]]["words"] == row["words"] and matches[row["sha256"]]["changed"] == row["changed"]
        bucket = "edited" if row["delivery"] == "edited" else "ordinary"
        result[row["sha256"]] = {level: history["outcomes"][row["group"]][point]["original"][bucket]
                                for level, point in (("tight", "10pct"), ("wide", "25pct"))}
    return result


def evaluate(panels, calibration, previous):
    result = {}
    comparator = {"before", "after"}
    for panel, rows in panels.items():
        for arm in ARMS:
            for level in LEVELS:
                stats = Counter()
                flags = {}
                for row in rows:
                    genre = "reports" if row["genre"] == "wiki" else row["genre"]
                    reading = row["readings"][arm]
                    flag = reading if reading and reading["score"] > calibration[genre][arm][level] else None
                    flags[row["group"], row["delivery"]] = flag
                    delivery = row["delivery"]
                    stats[delivery] += 1
                    stats[f"{delivery}_assessed"] += row["complete"]
                    stats[f"{delivery}_flagged"] += flag is not None
                    if delivery not in comparator:
                        located = useful(flag, row["changed"])
                        stats["useful"] += located
                        if panel == "layer":
                            prior = previous[row["sha256"]][level]
                            old = useful({"indices": prior}, row["changed"])
                            stats["prior_useful"] += old
                            stats["added_useful"] += located and not old
                            stats["lost_useful"] += old and not located
                            stats["added_ordinary_flags"] += delivery == "original" and flag is not None and not prior
                            stats["cleared_ordinary_flags"] += delivery == "original" and bool(prior) and flag is None
                            stats["prior_ordinary_flags"] += delivery == "original" and bool(prior)
                    elif delivery == "after":
                        stats["ordinary_revision_location"] += useful(flag, row["changed"])
                for row in rows:
                    if row["delivery"] == "edited":
                        flag = flags[row["group"], "edited"]
                        control = flags[row["group"], "original"]
                        stats["useful_same_sentence_ordinary"] += bool(
                            useful(flag, row["changed"]) and control and flag["sentence"] == control["sentence"])
                result[f"{panel}/{arm}/{level}"] = dict(stats)
    return result


def review(record):
    assert record["schema"] == SCHEMA and record["method"] == METHOD
    credits = record["credits"]
    assert len({row["page"] for row in credits}) == len(credits) == sum(editorial.QUOTAS.values())
    for rows in (record["calibration"], *record["panels"].values()):
        for row in rows:
            assert set(row["readings"]) == set(ARMS) and "text" not in row
            assert all(0 <= a < b and section >= 0 for a, b, section in row["words"])
            assert all(0 <= index < len(row["words"]) for index in row["changed"])
            for reading in row["readings"].values():
                assert bool(reading) == row["complete"]
                if reading:
                    assert 0 <= reading["score"] <= 1
                    assert reading["indices"] == sorted(set(reading["indices"]))
                    assert all(row["words"][i][2] == reading["sentence"] for i in reading["indices"])
    cuts = points(record["calibration"])
    assert cuts == record["points"]
    prior = baseline(record["panels"]["layer"])
    result = evaluate(record["panels"], cuts, prior)
    assert result == record["results"]
    return result


def comparisons(record):
    """Compare complementary locations and separate ranking from cutoff losses."""
    output = {}
    for panel in ("layer", "ranges"):
        for level in LEVELS:
            counts = Counter()
            for row in record["panels"][panel]:
                measures = row["readings"]
                active = {arm: measures[arm] if measures[arm] and measures[arm]["score"] >
                          record["points"][row["genre"]][arm][level] else None for arm in ARMS}
                if row["delivery"] == "original":
                    if not active["ai_only"]:
                        counts["ordinary_added_by_revisions"] += active["with_ordinary_revisions"] is not None
                    if not active["with_ordinary_revisions"]:
                        counts["ordinary_cleared_by_revisions"] += active["ai_only"] is not None
                    continue
                old = useful(active["ai_only"], row["changed"])
                revised = useful(active["with_ordinary_revisions"], row["changed"])
                counts["extra_locations_from_revision_fit"] += revised and not old
                counts["locations_lost_by_revision_fit"] += old and not revised
                counts["union_locations"] += revised or old
                for arm in ARMS:
                    counts[f"{arm}_useful_before_cutoff"] += useful(measures[arm], row["changed"])
            output[f"{panel}/{level}"] = dict(counts)
    return output


def range_unions(record):
    from scripts.writing_structured_range_transfer import proposals

    previous = read(RANGE_BASELINE)
    assert previous["plan_sha256"] == INPUTS["ranges"]
    lookup = {row["sha256"]: row for row in previous["rows"]}
    calibration = previous["freeze"]["calibration"]
    result = defaultdict(Counter)
    for row in record["panels"]["ranges"]:
        reference = lookup[row["sha256"]]
        assert (reference["words"], reference["changed"]) == (row["words"], row["changed"])
        key = row["sha256"]
        settings = calibration[f'{row["genre"]}/original']
        prior = proposals(reference, previous["readings"][key]["original"], settings["profile"])["two_ranges"]
        for level, published_key in (("tight", "10pct"), ("wide", "25pct")):
            baseline_flag = bool(prior and prior[0] > settings["points"]["two_ranges"][published_key])
            baseline_useful = baseline_flag and useful({"indices": prior[1]}, row["changed"])
            for arm in ARMS:
                counts = result[f"{arm}/{level}"]
                proposed = row["readings"][arm]
                point = record["points"][row["genre"]][arm][level]
                flag = bool(proposed and proposed["score"] > point)
                if row["delivery"] == "edited":
                    located = flag and useful(proposed, row["changed"])
                    counts["existing_useful"] += baseline_useful
                    counts["union_useful"] += baseline_useful or located
                else:
                    counts["existing_ordinary_flags"] += baseline_flag
                    counts["union_ordinary_flags"] += baseline_flag or flag
    return {key: dict(value) for key, value in result.items()}


def run(input_plan, encoder, output):
    plan = read(input_plan)
    assert plan["schema"] == SCHEMA
    store = embeddings(plan, encoder)
    sets, population = training_sets(plan, store)
    fit = fit_heads(sets)
    calibration = measure(plan["calibration"], store, fit)
    cuts = points(calibration)
    panels = {panel: measure(rows, store, fit) for panel, rows in plan["held"].items()}
    prior = baseline(panels["layer"])
    record = {"schema": SCHEMA, "method": METHOD, "plan_sha256": sha(input_plan.read_bytes()),
              "credits": plan["credits"], "fit": population, "calibration": calibration,
              "panels": panels, "points": cuts, "results": evaluate(panels, cuts, prior)}
    result = review(record)
    write(output, record)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    build = phases.add_parser("freeze")
    for name in ("roster", "layers", "ranges", "wiki", "output"):
        build.add_argument(name, type=Path)
    inference = phases.add_parser("run")
    for name in ("frozen", "encoder", "output"):
        inference.add_argument(name, type=Path)
    phases.add_parser("review").add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        prepare(args.roster, args.layers, args.ranges, args.wiki, args.output)
    elif args.phase == "run":
        run(args.frozen, args.encoder, args.output)
    else:
        record = read(args.receipt)
        print(json.dumps({"counts": review(record), "paired": comparisons(record), "two_range_unions": range_unions(record)},
                         indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
