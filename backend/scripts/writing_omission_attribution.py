"""Freeze, measure, and replay paired sentence-omission attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from scripts.review_writing_f1000_position_transfer import POINTS
from scripts.review_writing_long_position import baseline, select, useful
from scripts.writing_focus_v3 import BASELINE_ASSETS, BASELINE_REVISION

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/hf/hub/models--wasitaigeneratedcom--ai-text-detector-small/snapshots" / BASELINE_REVISION
SCHEMA = "chipmunk.writing-omission-attribution/1"
PANELS = {
    "point": ("calibration", 28, 14,
              "e2147c608e0924eff09ab6e0128112347075c0de0f7e7a9486c2a05085ef42ea",
              "ecf1788e22bc22f5eb3855cb29ea40968fa08fb00b96113c4178128006ff47f9"),
    "f1000": ("held", 36, 9,
              "9b07d5ee0afa19051f11063e5e1615a72222eeae4042661df3b653d42145e44c",
              "0f79223c1bf5e4634a79945a4eb722ed74bac92ebda3ea1545301946c0163752"),
    "elife": ("held", 45, 12,
              "597573e182a68e0edf5800a5c470fee23aa5bc79592b514f6a8c87811258a61c",
              "e25001843121ad27113666f72cc6ef8f0e0e585cb2aaf0cf05c9306c78684edd"),
}
ARMS = ("shared_drop", "shared_excess")
LONG_POINT = POINTS["long_document"]
METHOD = {
    "geometry": "Complete interior sentence and immediate neighbors in same mapped paragraph; ends in .?!; 12–65 center and 7–80 neighbor words; each full input at most 512 tokens",
    "score": "Natural log odds of 1 minus the human softmax response, clipped only to [1e-6, 1-1e-6]",
    "shared_drop": "Min of drop from left+center and center+right after removing center, divided by square root of its words over 24",
    "shared_excess": "Min of the two normalized drops and both margins over the normalized drop obtained by removing the adjacent sentence instead",
    "point": "Strict excess over max(0, second highest group maximum across both ordinary views in 14 distinct old F1000Research works)",
    "candidate": "At most one unique strongest measured focus per file; display exact source sentence range",
    "incumbents": "First six native contexts at 0.5 and optional longer passage point, as previously measured",
    "gate_per_held_panel": "At least three useful locations added over the union of incumbents in separate groups with both generators represented; at most one additional ordinary group and at most two wrong-only candidate files",
    "runtime": "CPU float32, 2 threads, batch 8, no added weights or online services",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, allow_nan=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


def read_panel(name, path):
    role, count, groups, source_pin, prior_pin = PANELS[name]
    source_path, prior_path = path / "cases.json", path / f"{role}.json"
    require(digest(source_path) == source_pin and digest(prior_path) == prior_pin, f"{name} source or prior response changed")
    source = [row for row in read(source_path) if row["role"] == role]
    prior = read(prior_path)["cases"]
    require(len(source) == len(prior) == count and len({case["group"] for case in source}) == groups, f"{name} panel count changed")
    for raw, old in zip(source, prior, strict=True):
        require(all(raw[key] == old[key] for key in ("group", "variant", "publisher", "truth"))
                and text_hash(raw["text"]) == old["text_sha256"], "The new inputs differ from the prior classifier inputs")
    return source, prior


def freeze(args):
    from importlib.metadata import version

    args.run.mkdir(exist_ok=False)
    seen = set()
    for name in PANELS:
        source, _ = read_panel(name, getattr(args, name))
        groups = {row["group"] for row in source}
        require(not seen & groups, "A work overlaps roles")
        seen |= groups
    require(all(digest(MODEL / key) == sha for key, sha in BASELINE_ASSETS.items()), "The pinned classifier differs")
    write(args.run / "freeze.json", {
        "schema": SCHEMA, "method": METHOD, "panels": PANELS, "code_sha256": digest(Path(__file__)),
        "incumbent_long_point": LONG_POINT, "checkpoint": BASELINE_REVISION, "assets": BASELINE_ASSETS,
        "runtime": {name: version(name) for name in ("torch", "transformers", "pysbd", "numpy")},
    })


def prepare_case(source, old):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    text = source["text"]
    doc = from_string(text)
    units = segment(doc)
    positions = [[u.start, u.end, u.paragraph, u.n_words] for u in units]
    require(doc.text == text and positions == old["positions"], "Native extracted sentence positions changed")
    prepared = []
    for center in range(1, len(units) - 1):
        prev, focus, following = units[center - 1:center + 2]
        if (prev.paragraph == focus.paragraph == following.paragraph and 12 <= focus.n_words <= 65
                and 7 <= prev.n_words <= 80 and 7 <= following.n_words <= 80
                and all(unit.text.endswith((".", "?", "!")) for unit in (prev, focus, following))):
            prepared.append((center, [f"{prev.text} {focus.text}", f"{focus.text} {following.text}",
                                      prev.text, focus.text, following.text]))
    base = baseline(old)
    best = select(old, LONG_POINT)
    control = {key: old[key] for key in ("group", "variant", "truth", "positions", "nonspace", "text_sha256")}
    control["incumbent"] = {"context_candidates": base, "context_useful": any(useful(old, item) for item in base),
                            "position_candidate": best, "position_useful": best is not None and positions[best][:2] == old["truth"]}
    return control, prepared


def measure(args):
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier

    path = args.run / "freeze.json"
    locked = read(path)
    require(locked["code_sha256"] == digest(Path(__file__)) and locked["method"] == METHOD, "The score freeze changed")
    require(os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1", "Run offline")
    require(all(digest(MODEL / key) == sha for key, sha in BASELINE_ASSETS.items()), "The pinned classifier differs")
    if args.panel != "point":
        require(read(args.run / "points.json")["freeze_sha256"] == digest(path), "Freeze the ordinary operating points before scoring held inputs")
    source, prior = read_panel(args.panel, getattr(args, args.panel))
    cases = [prepare_case(raw, old) for raw, old in zip(source, prior, strict=True)]
    unique = sorted({value for _, centers in cases for _, items in centers for value in items}, key=text_hash)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = FourClassTextClassifier.from_snapshot(MODEL, "cpu")
    token_counts = {text: len(model.tok(text, truncation=False, verbose=False)["input_ids"]) for text in unique}
    valid = [text for text in unique if token_counts[text] <= 512]
    scores = {}
    started = time.monotonic()
    for index in range(0, len(valid), 64):
        subset = sorted(valid[index:index + 64], key=token_counts.get)
        output = model.predict(subset, max_len=512, batch=8)
        require(len(output) == len(subset) and all(math.isfinite(x) and 0 <= x <= 1 for x in output), "Unusable classifier scores")
        scores.update(zip(subset, (float(x).hex() for x in output), strict=True))
        print(args.panel, min(index + 64, len(valid)), "/", len(valid), flush=True)
    write(args.run / f"{args.panel}.json", {
        "freeze_sha256": digest(path), "points_sha256": None if args.panel == "point" else digest(args.run / "points.json"),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "inputs": {text_hash(t): [token_counts[t], scores.get(t)] for t in unique},
        "cases": [{**case, "centers": [[i, *map(text_hash, strings)] for i, strings in centers]} for case, centers in cases],
    })


def log_odds(raw):
    bounded = min(1 - 1e-6, max(1e-6, float.fromhex(raw)))
    return math.log(bounded) - math.log1p(-bounded)


def responses(case, observations):
    result = {}
    for index, *keys in case["centers"]:
        parts = [observations[key] for key in keys]
        if any(score is None for _, score in parts):
            continue
        left_pair, right_pair, left, center, right = (log_odds(score) for _, score in parts)
        lwords, cwords, rwords = [case["positions"][i][3] for i in range(index - 1, index + 2)]
        left_drop = (left_pair - left) / math.sqrt(cwords / 24)
        right_drop = (right_pair - right) / math.sqrt(cwords / 24)
        left_alt = (left_pair - center) / math.sqrt(lwords / 24)
        right_alt = (right_pair - center) / math.sqrt(rwords / 24)
        result[index] = (min(left_drop, right_drop), min(left_drop, right_drop, left_drop - left_alt, right_drop - right_alt))
    return result


def verify_panel(name, data):
    _, count, n_groups, *_ = PANELS[name]
    require(len(data["cases"]) == count and len({case["group"] for case in data["cases"]}) == n_groups, "Panel count or groups changed")
    require(len({(c["group"], c["variant"]) for c in data["cases"]}) == count, "Duplicate case")
    inputs = data["inputs"]
    require(set(inputs) == {key for case in data["cases"] for _, *keys in case["centers"] for key in keys}, "An input commitment is missing or unreferenced")
    for key, (tokens, raw) in inputs.items():
        require(re.fullmatch("[a-f0-9]{64}", key) and type(tokens) is int and tokens >= 2
                and ((raw is None and tokens > 512) or (tokens <= 512 and 0 <= float.fromhex(raw) <= 1)), "Invalid input commitment")
    for case in data["cases"]:
        spans = case["positions"]
        seen = set()
        for index, *keys in case["centers"]:
            require(1 <= index < len(spans) - 1 and index not in seen and len(keys) == 5, "Invalid center list")
            seen.add(index)
            left, center, right = spans[index - 1:index + 2]
            require(left[2] == center[2] == right[2] and 7 <= left[3] <= 80 and 12 <= center[3] <= 65 and 7 <= right[3] <= 80,
                    "Omission focus contradicts the source paragraph or word limits")
        require(case["truth"] is None or any(row[:2] == case["truth"] for row in spans), "Known edit no longer maps to a source sentence")
        incumbent = case["incumbent"]
        positions = incumbent["context_candidates"]
        require(len(positions) <= 6 and positions == sorted(set(positions)) and all(0 <= j < len(spans) for j in positions), "Invalid incumbent range")
        require(incumbent["context_useful"] == any(useful(case, j) for j in positions), "Incumbent context overlap differs")
        idx = incumbent["position_candidate"]
        require(idx is None or 0 <= idx < len(spans), "Invalid longer-passage range")
        require(incumbent["position_useful"] == (idx is not None and spans[idx][:2] == case["truth"]), "Incumbent position overlap differs")


def points(data):
    verify_panel("point", data)
    maxima = defaultdict(list)
    for case in data["cases"]:
        require(case["truth"] is None, "Only historical ordinary revisions may set the point")
        maxima[case["group"]].extend(responses(case, data["inputs"]).values())
    require(all(maxima.values()), "Every point-selection work must be measured")
    return {name: max(0, sorted(max(items[i] for items in scores) for scores in maxima.values())[-2])
            for i, name in enumerate(ARMS)}


def decisions(data, cutoff):
    outcome = {}
    for i, arm in enumerate(ARMS):
        counts, added, false, inherited_false, introduced_false, families, locations = Counter(), set(), set(), set(), set(), set(), []
        for case in data["cases"]:
            scores = responses(case, data["inputs"])
            group, variant, truth, incumbent = (case[key] for key in ("group", "variant", "truth", "incumbent"))
            ranked = sorted(((values[i], index) for index, values in scores.items()), reverse=True)
            winner = ranked[0][1] if ranked and (len(ranked) == 1 or ranked[0][0] > ranked[1][0]) else None
            index = winner if winner is not None and ranked[0][0] > cutoff[arm] else None
            span = case["positions"][index][:2] if index is not None else None
            counts["assessed"] += bool(scores)
            if truth is None:
                counts[variant + "_controls"] += 1
                if incumbent["context_candidates"] or incumbent["position_candidate"] is not None:
                    inherited_false.add(group)
                if span is not None:
                    counts[variant + "_flags"] += 1
                    false.add(group)
            else:
                counts[variant + "_edits"] += 1
                counts["target_assessed"] += any(case["positions"][k][:2] == truth for k in scores)
                counts["target_unique_top"] += winner is not None and case["positions"][winner][:2] == truth
                old = incumbent["context_useful"] or incumbent["position_useful"]
                counts["incumbent_useful"] += old
                counts["combined_useful"] += old or span == truth
                if span == truth:
                    counts[variant + "_useful"] += 1
                    if not old:
                        counts["added_locations"] += 1
                        added.add(group)
                        families.add(variant)
                elif span is not None:
                    counts["wrong_only"] += 1
            if span is not None:
                locations.append([group, variant, *span, truth])
        introduced_false = false - inherited_false
        outcome[arm] = {"counts": dict(counts), "ordinary_groups": len(false), "incumbent_ordinary_groups": len(inherited_false),
                        "additional_ordinary_groups": len(introduced_false), "added_groups": len(added), "added_families": sorted(families),
                        "passes": len(added) >= 3 and len(families) == 2 and len(introduced_false) <= 1 and counts["wrong_only"] <= 2,
                        "locations": locations}
    return outcome


def replay(receipt):
    lock = receipt["freeze"]
    require(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["panels"] == json.loads(json.dumps(PANELS))
            and lock["checkpoint"] == BASELINE_REVISION and lock["assets"] == BASELINE_ASSETS
            and lock["incumbent_long_point"] == LONG_POINT, "The declared frozen method differs")
    point = points(receipt["point"])
    require(point == receipt["points"]["values"], "The frozen operating point does not replay")
    result = {}
    point_sha = text_hash(json.dumps(receipt["points"], allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
    for name in ("f1000", "elife"):
        data = receipt[name]
        require(data["freeze_sha256"] == receipt["point"]["freeze_sha256"] and data["points_sha256"] == point_sha,
                "A held run does not bind the fixed operating point")
        verify_panel(name, data)
        result[name] = decisions(data, point)
    require(result == receipt["results"], "A held decision, count, or range does not replay")
    return result


def native(args, receipt):
    for name in PANELS:
        source, old = read_panel(name, getattr(args, name))
        data = receipt[name]
        for raw, prior, recorded in zip(source, old, data["cases"], strict=True):
            expected, prepared = prepare_case(raw, prior)
            require({**expected, "centers": [[i, *map(text_hash, texts)] for i, texts in prepared]} == recorded,
                    "An omission input or evidence coordinate differs from its original")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "measure", "calibrate", "publish", "review", "native"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--panel", choices=PANELS)
    parser.add_argument("--point", type=Path)
    parser.add_argument("--f1000", type=Path)
    parser.add_argument("--elife", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args)
    elif args.phase == "measure":
        measure(args)
    elif args.phase == "calibrate":
        raw = read(args.run / "point.json")
        write(args.run / "points.json", {"values": points(raw), "freeze_sha256": raw["freeze_sha256"],
                                        "measurement_sha256": digest(args.run / "point.json")})
    else:
        if args.phase == "publish":
            receipt = {key: read(args.run / f"{key}.json") for key in ("freeze", "points", *PANELS)}
            receipt["results"] = {key: decisions(receipt[key], receipt["points"]["values"]) for key in ("f1000", "elife")}
            replay(receipt)
            write(args.receipt, receipt)
        else:
            receipt = read(args.receipt)
        result = replay(receipt)
        if args.phase == "native":
            native(args, receipt)
        print(json.dumps({key: {arm: {k: v for k, v in row.items() if k != "locations"} for arm, row in results.items()}
                          for key, results in result.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
