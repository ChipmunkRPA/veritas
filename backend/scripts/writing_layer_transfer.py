# SPDX-License-Identifier: MIT
"""Compare word readouts from different layers of the same frozen encoder."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_ratio_fusion_transfer as prior

ranges, adjacent, pool = prior.ranges, prior.adjacent, prior.pool
anchor, read, save, require = ranges.anchor, prior.read, prior.save, prior.require
words = anchor.old
SCHEMA = "chipmunk.writing-layer-transfer/1"
BASELINE = "original"
LAYERS = ("final", "middle", "change")
ARMS = (BASELINE, *LAYERS)
QUOTA = 4
SHARDS = {
    ("news", "gpt-5.4"): (
        "train",
        198982785,
        "11d6c9b371c3f43aa25a108c2d557a92adaa2db75b9ae68cc813b2692bd570be",
    ),
    ("reports", "gpt-5.4"): (
        "train",
        248151536,
        "a205bd7102225997b9e2245589cdf416e42499677b31f76382e28bd1b0b9fa68",
    ),
    **{(genre, "gemini-2.5-flash"): families["gemini-2.5-flash"] for genre, families in prior.SHARDS.items()},
}
PARENT = prior.PUBLISHED_TEST.with_name("writing-ratio-fusion-transfer-2026-11-15.json.gz")
METHOD = {
    "representations": "Compare the published experimental head to three identically initialized linear heads. "
    "Use layer six, layer three, or layer six minus layer three from one frozen encoder forward. "
    "Mean-pool pieces into words; concatenate the absolute and sentence-centered vectors. "
    "Scale each block to the same L2 norm before scoring with a new head.",
    "fit": "Use only the previous 882 qualified matched report pairs, with the previous changed-word, "
    "unchanged-word stability, and within-sentence margin objective. Keep the same fit order for each head.",
    "decision": "Keep the same highest-six sentence rank and top-60-percent word selection across all arms. "
    "Freeze separate one-tenth and quarter ordinary calibration points per genre and head. "
    "A useful location requires half changed-word coverage and 0.30 word intersection over union.",
    "holdout": "Select unchanged-flank six-sentence excerpts from new captured works, hashed by work and checked "
    "for prior exposure and original twelve-word overlap. Use one or two adjacent labeled sentences. "
    "Measure useful locations, ordinary flags, missed coverage and matched ordinary collisions. "
    "Separately supply only the true sentence position after inference to diagnose word-ranking limits.",
}


def contract():
    return {
        "schema": SCHEMA,
        "method": METHOD,
        "quota": QUOTA,
        "publisher_revision": adjacent.SOURCE_REVISION,
        "script_sha256": pool.file_hash(Path(__file__)),
        "head_seed": anchor.SEED,
        "anchors": anchor.contract(),
        "anchor_code_sha256": pool.file_hash(Path(anchor.__file__)),
        "extraction_code_sha256": pool.file_hash(Path(prior.__file__)),
        "previous_panel_sha256": pool.file_hash(PARENT),
        "shards": {"/".join(key): list(pin) for key, pin in SHARDS.items()},
    }


def prepare(folder, plans, inventory=None):
    baseline = read(PARENT)
    require(
        [pool.file_hash(path) for path in plans][1:]
        == [baseline["freeze"]["earlier_test_sha256"], baseline["plan_sha256"]],
        "Earlier holdouts differ",
    )
    require(
        pool.file_hash(plans[0]) == baseline["freeze"]["development_sha256"], "Ordinary source plan differs"
    )
    development, earlier, recent = [read(path) for path in plans]
    reference = ranges.reference()
    calibration = [row for row in development["rows"] if row["family"] == "calibration"]
    require(
        list(map(adjacent.public, calibration)) == prior.ordinary_words()[0], "Ordinary source rows differ"
    )
    known, inventory = adjacent.context.abstract_exclusions(inventory)
    exposed_rows = development["rows"] + earlier["rows"] + recent["held"]
    known.update(row["group"] for row in exposed_rows)
    known.update(reference["fit"]["groups"])
    fingerprints = adjacent.intake.fingerprints
    used = {mark for row in exposed_rows for mark in fingerprints(row["text"])}
    for (genre, family), pin in SHARDS.items():
        table = prior.load_shard(folder, genre, family, pin)
        for key, versions in table.items():
            if pool.sha(key) in known and "v0" in versions:
                used.update(fingerprints(versions["v0"]["text"]))
        del table

    cases = []
    for (genre, family), pin in SHARDS.items():
        table = prior.load_shard(folder, genre, family, pin)
        remaining = dict.fromkeys(ranges.SLOTS, QUOTA)
        for key in sorted(table, key=lambda value: pool.sha(f"{SCHEMA}/{genre}/{family}/{value}")):
            group, versions = pool.sha(key), table[key]
            if group in known or "v0" not in versions:
                continue
            marks = fingerprints(versions["v0"]["text"])
            if not marks or marks & used:
                continue
            try:
                example = next(prior.local_examples(versions, remaining), None)
            except ValueError:
                continue
            if example is None:
                continue
            slot, stage, before, after = example
            labels = {"group": group, "genre": genre, "family": family, "stage": stage, "target": list(slot)}
            cases.extend(
                (
                    labels | before | {"delivery": "original", "changed": []},
                    labels | after | {"delivery": "edited"},
                )
            )
            known.add(group)
            used.update(marks)
            remaining[slot] -= 1
            if not remaining[slot]:
                del remaining[slot]
            if not remaining:
                break
        print(
            f"{genre}/{family} selected {QUOTA * len(ranges.SLOTS) - sum(remaining.values())} held pairs",
            flush=True,
        )
        del table
    return {
        "freeze": {"contract": contract(), "exposures": inventory},
        "calibration": calibration,
        "held": cases,
    }


def features(encoded, encoder):
    import torch

    tokens = torch.zeros((len(encoded), max(len(item[0]) for item in encoded)), dtype=torch.long)
    mask = torch.zeros_like(tokens)
    for row, (ids, _, _) in enumerate(encoded):
        tokens[row, : len(ids)] = torch.tensor(ids)
        mask[row, : len(ids)] = 1
    with torch.no_grad():
        states = encoder(input_ids=tokens, attention_mask=mask, output_hidden_states=True).hidden_states
    require(len(states) == 7, "Unexpected encoder depth")
    output = []
    for row, (_, word_ids, items) in enumerate(encoded):
        positions, indices = zip(
            *((i, word) for i, word in enumerate(word_ids) if word is not None), strict=True
        )
        index = torch.tensor(indices)
        sentence = torch.tensor([word[2] for word in items])
        pieces = torch.bincount(index, minlength=len(items))[:, None]
        sizes = torch.bincount(sentence).clamp_min(1)[:, None]
        vectors = []
        for state in (states[6], states[3]):
            selected = state[row, list(positions)]
            pooled = selected.new_zeros((len(items), state.shape[-1])).index_add_(0, index, selected) / pieces
            centers = pooled.new_zeros((len(sizes), pooled.shape[-1])).index_add_(0, sentence, pooled) / sizes
            vectors.append(torch.cat((pooled, pooled - centers[sentence]), dim=-1))
        output.append(tuple(vectors))
    return output


def scaled(values):
    from torch.nn.functional import normalize

    width = values.shape[-1] // 2
    return normalize(values.reshape(-1, 2, width), dim=-1).flatten(1) * math.sqrt(width)


def representation(pair, arm):
    final, middle = pair
    if arm == BASELINE:
        return final
    vectors = {"final": final, "middle": middle}
    return scaled(final - middle if arm == "change" else vectors[arm])


def fit(args):
    import torch
    from safetensors.torch import save_file

    frozen = contract()
    require(pool.file_hash(args.roster) == frozen["anchors"]["fit_sources"], "Frozen fitting roster differs")
    args.output.mkdir(mode=0o700)
    save(args.output / "freeze.json", frozen)
    tokenizer, encoder, initial = anchor.native_runtime(args.encoder, args.checkpoint)
    rows, truth, _ = words.input_plan(args.roster)
    groups = defaultdict(list)
    for row in rows:
        if row["role"] == "fit":
            groups[row["group"]].append(row)
    selected = []
    for group, (original, edited) in sorted(groups.items()):
        before, after = words.encode(original, tokenizer), words.encode(edited, tokenizer)
        changed = sorted(truth[edited["sha256"]])
        if before is not None and after is not None and changed:
            selected.append((group, before, after, changed, edited["position"]))
    require([row[0] for row in selected] == ranges.reference()["fit"]["groups"], "Fit group IDs differ")
    prepared = []
    for start in range(0, len(selected), anchor.BATCH):
        batch = selected[start : start + anchor.BATCH]
        encoded = iter(features([view for pair in batch for view in pair[1:3]], encoder))
        for group, before, after, changed, sentence in batch:
            source, target = anchor.word_alignment(before[2], after[2])
            controls = [i for i, word in enumerate(before[2]) if word[2] == sentence]
            neighbors = [i for i in target if after[2][i][2] == sentence]
            require(set(changed).isdisjoint(target), "An unchanged anchor was labeled edited")
            prepared.append(
                (group, next(encoded), next(encoded), changed, source, target, controls, neighbors)
            )
        if len(prepared) % 128 == 0 or len(prepared) == len(selected):
            print(f"Extracted both layers for {len(prepared)}/{len(selected)} pairs", flush=True)
    torch.manual_seed(anchor.SEED)
    common = torch.nn.Linear(initial.in_features, 1)
    weights = {}
    for arm in LAYERS:
        dataset = [
            (group, representation(before, arm), representation(after, arm), *labels)
            for group, before, after, *labels in prepared
        ]
        head, trace = anchor.train(dataset, common, "anchors_and_local")
        path = args.output / f"{arm}.safetensors"
        save_file(head.state_dict(), str(path))
        weights[arm] = {"weights_sha256": pool.file_hash(path), "loss": trace}
        del dataset
        print(f"Fitted {arm} on {len(prepared)} source pairs", flush=True)
    save(args.output / "fit.json", {"freeze": frozen, "weights": weights, "groups": len(prepared)})


def score(rows, tokenizer, encoder, heads):
    import torch

    readings = {}
    for start in range(0, len(rows), anchor.BATCH):
        batch = rows[start : start + anchor.BATCH]
        ready = []
        for row in batch:
            value = words.encode(row, tokenizer)
            counts = Counter(word[2] for word in value[2]) if value else {}
            ready.append(value if all(counts.get(index, 0) >= 6 for index in range(6)) else None)
        inputs = [value for value in ready if value is not None]
        vectors = iter(features(inputs, encoder) if inputs else [])
        with torch.no_grad():
            for row, value in zip(batch, ready, strict=True):
                pair = next(vectors) if value is not None else None
                readings[row["sha256"]] = {
                    name: [round(x, 7) for x in head(representation(pair, name)).flatten().tolist()]
                    if pair is not None
                    else []
                    for name, head in heads.items()
                }
    return readings


def proposals(rows, readings):
    require(
        len(rows) == len(readings) and set(readings) == {row["sha256"] for row in rows}, "Incomplete readouts"
    )
    selected = {}
    for row in rows:
        key, geometry = row["sha256"], row["words"]
        require(
            len(row["spans"]) == 6 and all(b <= c for (_, b, _), (c, _, _) in zip(geometry, geometry[1:])),
            "Invalid six-sentence word geometry",
        )
        require(
            all(0 <= i < 6 and row["spans"][i][0] <= a < b <= row["spans"][i][1] for a, b, i in geometry),
            "A word falls outside its recorded sentence",
        )
        require(set(readings[key]) == set(ARMS), "Readout arms differ")
        require(
            len({len(values) for values in readings[key].values()}) == 1, "Readout coverage differs by layer"
        )
        selected[key] = {}
        for name, values in readings[key].items():
            require(
                not values or len(values) == len(geometry) and all(map(math.isfinite, values)),
                "Invalid word scores",
            )
            selected[key][name] = anchor.broad(geometry, values, 6) if values else None
    return selected


def points(rows, readings):
    selected = proposals(rows, readings)
    result = {}
    for genre in ("news", "reports"):
        sources = [selected[row["sha256"]] for row in rows if row["genre"] == genre]
        require(len(sources) == 36, "Unexpected number of ordinary calibration works")
        result[genre] = {}
        for name in ARMS:
            scores = sorted((row[name]["score"] for row in sources if row[name]), reverse=True)
            require(len(scores) >= 24, "Insufficient ordinary calibration coverage")
            result[genre][name] = {
                "assessed": len(scores),
                **{level: scores[len(scores) // divisor] for level, divisor in ranges.POINTS.items()},
            }
    return result


def known_sentence_words(geometry, values, target):
    selected = set()
    for sentence in target:
        candidates = [index for index, item in enumerate(geometry) if item[2] == sentence]
        candidates.sort(key=lambda index: (-values[index], index))
        selected.update(candidates[: math.ceil(0.60 * len(candidates))])
    return selected


def evaluate(rows, readings, operating):
    selected = proposals(rows, readings)
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    results = {level: {name: defaultdict(Counter) for name in ARMS} for level in ranges.POINTS}
    outcomes = {}
    for before, after in groups.values():
        original, edited = before["sha256"], after["sha256"]
        target, changed = set(after["target"]), set(after["changed"])
        work_results = {}
        for level in ranges.POINTS:
            work_results[level] = {}
            for name in ARMS:
                clean_pick, edit_pick = selected[original][name], selected[edited][name]
                limit = operating[after["genre"]][name][level]
                clean = clean_pick is not None and clean_pick["score"] > limit
                flagged = edit_pick is not None and edit_pick["score"] > limit
                clean_indices = clean_pick["indices"] if clean else []
                edit_indices = edit_pick["indices"] if flagged else []
                clean_sentences = {before["words"][i][2] for i in clean_indices}
                edit_sentences = {after["words"][i][2] for i in edit_indices}
                useful = words.useful_words(set(edit_indices), changed)
                values = readings[edited][name]
                oracle = bool(
                    values
                    and words.useful_words(known_sentence_words(after["words"], values, target), changed)
                )
                work_results[level][name] = {
                    "ordinary": clean_indices,
                    "edited": edit_indices,
                    "useful": useful,
                }
                base = work_results[level][BASELINE]
                tally = results[level][name][f"{after['genre']}/{after['family']}"]
                tally.update(
                    pairs=1,
                    ordinary_flags=clean,
                    missed_ordinary=clean_pick is None,
                    ordinary_added=clean and not base["ordinary"],
                    ordinary_cleared=not clean and bool(base["ordinary"]),
                    edited_flags=flagged,
                    missed_edited=edit_pick is None,
                    wrong_sentence=flagged and not edit_sentences & target,
                    useful=useful,
                    gained=useful and not base["useful"],
                    lost=not useful and base["useful"],
                    matched_collision=useful and bool(clean_sentences & edit_sentences),
                    known_sentence_useful=oracle,
                )
        outcomes[after["group"]] = work_results
    result = {
        level: {name: {key: dict(tally) for key, tally in strata.items()} for name, strata in modes.items()}
        for level, modes in results.items()
    }
    return result, outcomes


def review(packet):
    freeze = packet["freeze"]
    require(freeze["contract"] == packet["fit"]["freeze"] == contract(), "Frozen inputs or procedure differ")
    require(
        packet["fit"]["groups"] == len(ranges.reference()["fit"]["groups"]), "Fitting source count differs"
    )
    require(set(packet["fit"]["weights"]) == set(LAYERS), "Fitted model arms differ")
    known, _ = adjacent.context.abstract_exclusions(freeze["exposures"])
    known.update(ranges.reference()["fit"]["groups"])
    require(packet["calibration_rows"] == prior.ordinary_words()[0], "Ordinary source rows differ")
    groups = defaultdict(list)
    for row in packet["rows"]:
        require(row["group"] not in known, "A previously exposed work entered the holdout")
        groups[row["group"]].append(row)
    sizes = Counter()
    for before, after in groups.values():
        stratum = (after["genre"], after["family"])
        slot = tuple(after["target"])
        require(
            stratum in SHARDS
            and slot in ranges.SLOTS
            and (before["delivery"], after["delivery"]) == ("original", "edited")
            and all(before[field] == after[field] for field in ("family", "genre", "stage", "target")),
            "Invalid held pair",
        )
        require(
            not before["changed"]
            and after["changed"] == sorted(set(after["changed"]))
            and all(0 <= i < len(after["words"]) for i in after["changed"])
            and {after["words"][i][2] for i in after["changed"]} == set(slot),
            "Invalid changed-word labels",
        )
        sizes[stratum, slot] += 1
    require(
        sizes and max(sizes.values()) <= QUOTA and {group for group, _ in sizes} == set(SHARDS),
        "Held stratum coverage differs",
    )
    operating = points(packet["calibration_rows"], packet["calibration_scores"])
    require(operating == packet["points"], "Ordinary points differ")
    result, outcomes = evaluate(packet["rows"], packet["scores"], operating)
    require(
        result == packet.get("results", result) and outcomes == packet.get("outcomes", outcomes),
        "Held outcomes differ",
    )
    return result, outcomes


def capture(args):
    import torch
    import transformers
    from safetensors.torch import load_file

    plan, fitted = read(args.plan), read(args.fit / "fit.json")
    require(plan["freeze"]["contract"] == fitted["freeze"] == contract(), "Plan and fit differ")
    tokenizer, encoder, original = anchor.native_runtime(args.encoder, args.checkpoint)
    heads = {BASELINE: original}
    for name in LAYERS:
        path = args.fit / f"{name}.safetensors"
        require(pool.file_hash(path) == fitted["weights"][name]["weights_sha256"], "A fitted head differs")
        heads[name] = torch.nn.Linear(original.in_features, 1).eval()
        heads[name].load_state_dict(load_file(str(path)), strict=True)
    ordinary = list(map(adjacent.public, plan["calibration"]))
    controls = score(plan["calibration"], tokenizer, encoder, heads)
    _, baseline = prior.ordinary_words()
    require(
        all(
            len(baseline[key]) == len(values[BASELINE])
            and all(abs(a - b) < 0.0001 for a, b in zip(baseline[key], values[BASELINE], strict=True))
            for key, values in controls.items()
        ),
        "Final layer changed the previous head's ordinary readouts",
    )
    operating = points(ordinary, controls)
    args.output.mkdir(mode=0o700)
    save(args.output / "points.json", operating)
    print(f"Froze ordinary points on {len(ordinary)} files before held inference", flush=True)
    held = score(plan["held"], tokenizer, encoder, heads)
    packet = {
        "freeze": plan["freeze"],
        "fit": fitted,
        "plan_sha256": pool.file_hash(args.plan),
        "calibration_rows": ordinary,
        "calibration_scores": controls,
        "points": operating,
        "rows": list(map(adjacent.public, plan["held"])),
        "scores": held,
        "runtime": {
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "device": "cpu",
        },
    }
    packet["results"], packet["outcomes"] = review(packet)
    save(args.output / "receipt.json.gz", packet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    intake = modes.add_parser("prepare")
    intake.add_argument("folder", type=Path)
    intake.add_argument("plans", nargs=3, type=Path)
    intake.add_argument("output", type=Path)
    intake.add_argument("--reference", type=Path)
    for mode, filenames in {"fit": ("roster",), "capture": ("plan", "fit")}.items():
        command = modes.add_parser(mode)
        for name in (*filenames, "encoder", "checkpoint", "output"):
            command.add_argument(name, type=Path)
    replay = modes.add_parser("review")
    replay.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        existing = read(args.reference) if args.reference else None
        plan = prepare(args.folder, args.plans, existing["freeze"]["exposures"] if existing else None)
        save(args.output, plan)
        if existing:
            require(
                pool.file_hash(args.output) == existing["plan_sha256"]
                and plan["freeze"] == existing["freeze"]
                and list(map(adjacent.public, plan["held"])) == existing["rows"],
                "Reconstructed held sources differ",
            )
    elif args.mode == "fit":
        fit(args)
    elif args.mode == "capture":
        capture(args)
    else:
        result, _ = review(read(args.receipt))
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
