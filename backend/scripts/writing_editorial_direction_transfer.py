# SPDX-License-Identifier: Apache-2.0
"""Test whether suppressing ordinary revision directions helps localize AI rewrites."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from scripts import writing_editorial_negative_transfer as prior
from scripts import writing_width_calibration_transfer as essays

SCHEMA = "chipmunk.writing-editorial-direction/1"
PANEL = prior.BASELINE.parent
PARENT = PANEL / "writing-historical-editorial-negatives-2026-11-15.json.gz"
EXPOSURE_PREFIXES = ("writing-", "text-", "document-", "email-")
ARMS = ("ai_only", "unrelated_directions_removed", "editorial_directions_removed")
RANK = 12
ESSAY_HOLDOUT = 12
ESSAY_PINS = {"gemini-2.5-flash": essays.ESSAY_PINS["gemini-2.5-flash"]}
METHOD = {
    "input": "Use the same exact word states, target words, and AI-only fit as the historical editorial comparison.",
    "directions": "For each qualified historical word replacement, subtract the average earlier word state from the average replacement word state. Compare with the same before and after states paired across unrelated articles by a one-step rotation. Give each unit vector equal weight.",
    "fit": "Remove the top twelve orthogonal directions of either uncentered covariance from both feature blocks. Fit the same head on the same AI pairs with separately projected word features. Keep an unprojected arm.",
    "test": "First freeze 12 matched Gemini essay pairs, with at most six per interior position. Then select 36 distinct ordinary essay calibration works. Exclude earlier reported works and normalized twelve-word overlap. Recheck retained news and report rewrites from two generators and dated Wikipedia pairs under the same single-excerpt measurement.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exposure_files(reference):
    if reference is not None:
        return exposure_paths(prior.read(reference)["inputs"]["exposures"])
    return sorted(path for path in PANEL.iterdir()
                  if path.name.startswith(EXPOSURE_PREFIXES) and not path.name.startswith("writing-editorial-direction-")
                  and (path.suffix == ".json" or path.name.endswith(".json.gz")))


def exposure_paths(names):
    require(all(Path(name).name == name and name.startswith(EXPOSURE_PREFIXES)
                and name.endswith((".json", ".json.gz")) for name in names), "Invalid exposure receipt path")
    return [PANEL / name for name in sorted(names)]


def exposure_snapshot(files):
    exposed, history = set(), {}
    for file in files:
        data = file.read_bytes()
        history[file.name] = prior.sha(data)
        if file.suffix == ".gz":
            data = gzip.decompress(data)
        exposed.update(match.decode() for match in re.findall(rb'"([a-f0-9]{64})"', data))
    return history, exposed


def freeze(base, gemini, output, reference):
    parent = prior.read(PARENT)
    require(prior.sha(base.read_bytes()) == parent["plan_sha256"], "The original editorial fit plan changed")
    history, exposed = exposure_snapshot(exposure_files(reference))
    if reference is not None:
        require(history == prior.read(reference)["inputs"]["exposures"], "The sealed exposure snapshot changed")
    plan = prior.read(base)
    english = [*plan["fit"], *plan["calibration"], *plan["held"]["ranges"], *plan["held"]["layer"]]
    fingerprints = set().union(*(prior.editorial.fingerprints(" ".join(s["text"] for s in row["sentences"]))
                                for row in english))
    known = {row["group"] for row in english} | exposed
    fresh = essays.assay_text(gemini, None, extra_exclusions=known, exclude_phrases=fingerprints, seed=SCHEMA,
                             held_per_family=ESSAY_HOLDOUT, held_first=True, held_position_limit=6, shard_pins=ESSAY_PINS)
    calibration, held = [], []
    for row in fresh["rows"]:
        delivery = "original" if row["family"] == "original" else "edited"
        item = prior.source_record(row | {"genre": "essays", "delivery": delivery, "target": [row["position"]]})
        (calibration if row["role"] == "calibration" else held).append(item)
    added = {row["group"] for row in calibration + held}
    require(len(added) == 36 + ESSAY_HOLDOUT and not added & known and len(calibration) == 36 and len(held) == 2 * ESSAY_HOLDOUT,
            "The fresh essay work split differs")
    plan["calibration"].extend(calibration)
    plan["held"]["essays"] = held
    plan["schema"] = SCHEMA
    plan["inputs"] = {"parent_plan": parent["plan_sha256"], "essay_shards": ESSAY_PINS, "exposures": history}
    prior.write(output, plan)
    print(f"Froze 36 ordinary essay calibration works and {ESSAY_HOLDOUT} separate matched pairs after excluding {len(history)} project receipts.", flush=True)


def replacement_pairs(plan, store):
    pairs = defaultdict(dict)
    for row in plan["wiki_fit"]:
        pairs[row["group"]][row["delivery"]] = row["sentences"][0]["text"]
    original, edited, included = [], [], []
    for key, pair in sorted(pairs.items(), key=lambda pair: prior.sha(str(pair[0]))):
        before, after = pair["before"], pair["after"]
        if before not in store or after not in store:
            continue
        left, right = ([match.group().casefold() for match in prior.words.WORD.finditer(text)] for text in (before, after))
        earlier, later = [], []
        for operation, a, b, c, d in SequenceMatcher(a=left, b=right, autojunk=False).get_opcodes():
            if operation == "replace":
                earlier.extend(range(a, b))
                later.extend(range(c, d))
        if not earlier:
            continue
        for text, indices, values in ((before, earlier, original), (after, later, edited)):
            _, states = store[text]
            values.append(states[indices, : states.shape[1] // 2].mean(0).double())
        included.append(key)
    require(len(included) >= 100, "Too few historical pairs with mutually replaced exact words")
    return original, edited, included


def directions(original, edited, *, shuffled):
    import torch

    left, right = torch.stack(original), torch.stack(edited)
    if shuffled:
        left = left.roll(1, dims=0)
    delta = right - left
    norms = delta.norm(dim=1, keepdim=True)
    require(bool((norms.flatten() > 0).all()), "An ordinary edit has no usable direction")
    unit = delta / norms
    _, singular, vectors = torch.linalg.svd(unit, full_matrices=False)
    basis = vectors[:RANK].float()
    return basis, {"rank": RANK, "variance_fraction": round(float(singular[:RANK].square().sum() / singular.square().sum()), 9),
                   "delta_pairs": len(unit)}


def project(store, basis):
    projected = {}
    for text, (positions, values) in store.items():
        matrix = values.reshape(len(values), 2, basis.shape[1])
        residual = matrix - (matrix @ basis.T) @ basis
        projected[text] = positions, residual.flatten(1)
    return projected


def fit_and_measure(plan, store, input_sets):
    fit = prior.fit_heads(input_sets, use_ordinary_augmentations=False)
    scored = {"calibration": prior.measure(plan["calibration"], store, fit)}
    scored.update({name: prior.measure(rows, store, fit) for name, rows in plan["held"].items()})
    return scored


def combine(measurements):
    state = measurements[ARMS[0]]
    output = {}
    for panel, anchor in state.items():
        rows = []
        for index, baseline in enumerate(anchor):
            geometry = {key: value for key, value in baseline.items() if key != "readings"}
            readings = {}
            for arm in ARMS:
                record = measurements[arm][panel][index]
                require(all(record[key] == value for key, value in geometry.items()), "A projected comparison changed source geometry")
                readings[arm] = record["readings"]["ai_only"]
            rows.append(geometry | {"readings": readings})
        output[panel] = rows
    return output


def points(rows):
    cutoffs = {}
    for genre in ("news", "reports", "essays"):
        controls = [row for row in rows if row["genre"] == genre]
        require(len(controls) == len({row["group"] for row in controls}) == 36, "The independent ordinary role changed")
        cutoffs[genre] = {}
        for arm in ARMS:
            values = sorted((row["readings"][arm]["score"] for row in controls if row["complete"]), reverse=True)
            require(len(values) >= 28, "Too few assessed ordinary works for this cutoff")
            cutoffs[genre][arm] = {name: values[math.floor(len(controls) * allowed)] for name, allowed in prior.LEVELS.items()}
    return cutoffs


def place(row, arm, level, cutoffs):
    genre = "reports" if row["genre"] == "wiki" else row["genre"]
    value = row["readings"][arm]
    return value if value and value["score"] > cutoffs[genre][arm][level] else None


def evaluate(proposals, cutoffs):
    results = {}
    for panel, rows in proposals.items():
        for arm in ARMS:
            for level in prior.LEVELS:
                counts = Counter()
                ordinary = {row["group"]: place(row, arm, level, cutoffs) for row in rows if row["delivery"] == "original"}
                by_family = defaultdict(Counter)
                for row in rows:
                    selected = place(row, arm, level, cutoffs)
                    original = place(row, "ai_only", level, cutoffs)
                    kind = row["delivery"]
                    counts.update({kind: 1, f"{kind}_assessed": row["complete"], f"{kind}_flagged": bool(selected),
                                   f"{kind}_added_flags": bool(selected and not original),
                                   f"{kind}_cleared_flags": bool(original and not selected)})
                    if kind in {"edited", "after"}:
                        useful = prior.useful(selected, row["changed"])
                        old_useful = prior.useful(original, row["changed"])
                        control = ordinary.get(row["group"])
                        clear = bool(useful and kind == "edited" and (control is None or control["sentence"] != selected["sentence"]))
                        counts.update(useful=useful, useful_clear=clear, gained=useful and not old_useful, lost=old_useful and not useful,
                                      useful_before_cutoff=prior.useful(row["readings"][arm], row["changed"]))
                        by_family[row["family"]].update(works=1, useful=useful, useful_clear=clear)
                results[f"{panel}/{arm}/{level}"] = {"counts": dict(counts), "by_family": dict(by_family)}
    return results


def run(input_plan, encoder, output):
    plan = prior.read(input_plan)
    require(plan["schema"] == SCHEMA, "The fitted sources differ from the planned comparison")
    store = prior.embeddings(plan, encoder)
    input_sets, population = prior.training_sets(plan, store)
    original, edited, included = replacement_pairs(plan, store)
    print(f"Measured word-state directions from {len(included)} dated revision pairs", flush=True)
    measured = {ARMS[0]: fit_and_measure(plan, store, input_sets)}
    bases = {}
    for arm, shuffled in zip(ARMS[1:], (True, False), strict=True):
        basis, bases[arm] = directions(original, edited, shuffled=shuffled)
        altered = project(store, basis)
        projected_sets, projected_population = prior.training_sets(plan, altered)
        require(projected_population == population, "Projection changed which words were used for fitting")
        measured[arm] = fit_and_measure(plan, altered, projected_sets)
        del altered, projected_sets
    panels = combine(measured)
    calibration = panels.pop("calibration")
    cutoffs = points(calibration)
    result = evaluate(panels, cutoffs)
    packet = {"schema": SCHEMA, "method": METHOD, "source_plan_sha256": prior.sha(input_plan.read_bytes()),
              "inputs": plan["inputs"], "projection": bases, "fit": population, "wiki_directions": included,
              "calibration": calibration, "panels": panels, "points": cutoffs, "result": result}
    review(packet)
    prior.write(output, packet)
    return result


def review(packet):
    parent = prior.read(PARENT)
    require(packet["schema"] == SCHEMA and packet["method"] == METHOD and packet["inputs"]["parent_plan"] == parent["plan_sha256"]
            and packet["inputs"]["essay_shards"] == ESSAY_PINS, "The recorded comparison protocol changed")
    history, exposed = exposure_snapshot(exposure_paths(packet["inputs"]["exposures"]))
    require(history == packet["inputs"]["exposures"], "The prior project evaluation receipts changed")
    frozen = {row["page"] for row in parent["credits"] if row["split"] == "fit"}
    subset = set(packet["wiki_directions"])
    require(len(subset) >= 100 and subset <= frozen and set(packet["projection"]) == set(ARMS[1:])
            and all(row["rank"] == RANK and row["delta_pairs"] == len(subset) for row in packet["projection"].values()),
            "The projection fit is outside its declared ordinary source role")
    schema = set(packet["panels"]) == {"ranges", "layer", "wiki", "essays"} and len(packet["calibration"]) == 108
    require(schema and packet["fit"] == parent["fit"], "The published fit or panel geometry changed")
    for rows in (packet["calibration"], *packet["panels"].values()):
        for row in rows:
            require("sentences" not in row and set(row["readings"]) == set(ARMS)
                    and all(0 <= a < b and section >= 0 for a, b, section in row["words"])
                    and all(0 <= index < len(row["words"]) for index in row["changed"]), "Invalid word evidence")
            for reading in row["readings"].values():
                require(bool(reading) == row["complete"], "A missing measurement changes the comparison scope")
                if reading:
                    require(0 <= reading["score"] <= 1 and reading["indices"] == sorted(set(reading["indices"]))
                            and all(0 <= i < len(row["words"]) and row["words"][i][2] == reading["sentence"] for i in reading["indices"]),
                            "Highlight is outside its measured sentence")
    old_groups = {row["group"] for rows in parent["panels"].values() for row in rows}
    essay_test = Counter(row["group"] for row in packet["panels"]["essays"])
    essay_points = {row["group"] for row in packet["calibration"] if row["genre"] == "essays"}
    panel = packet["panels"]["essays"]
    counts = Counter(row["family"] for row in panel)
    require(len(essay_test) == ESSAY_HOLDOUT and len(essay_points) == 36 and set(essay_test.values()) == {2}
            and not essay_points & essay_test.keys() and not (essay_points | essay_test.keys()) & (old_groups | exposed)
            and counts == {"original": ESSAY_HOLDOUT, **dict.fromkeys(ESSAY_PINS, ESSAY_HOLDOUT)}, "The new held source split differs")

    def geometry(row):
        return {key: value for key, value in row.items() if key != "readings"}

    for name in ("layer", "ranges", "wiki"):
        require([geometry(row) for row in packet["panels"][name]] == [geometry(row) for row in parent["panels"][name]],
                "A retained held panel differs from its source roles")
    cutoffs = points(packet["calibration"])
    results = evaluate(packet["panels"], cutoffs)
    require(cutoffs == packet["points"] and results == packet["result"], "Measured outcomes disagree with replay")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    source = phases.add_parser("freeze")
    for name in ("base", "gemini", "output"):
        source.add_argument(name, type=Path)
    source.add_argument("--reference", type=Path)
    analyze = phases.add_parser("run")
    for name in ("plan", "encoder", "output"):
        analyze.add_argument(name, type=Path)
    replay = phases.add_parser("review")
    replay.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.base, args.gemini, args.output, args.reference)
    else:
        result = run(args.plan, args.encoder, args.output) if args.phase == "run" else review(prior.read(args.receipt))
        print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
