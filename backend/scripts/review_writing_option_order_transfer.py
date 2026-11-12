"""Independently replay publisher, prompt and evidence locations for the option-order test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

from lxml import html

FAMILIES = ("qwen", "smol2")
OPTIONS = ("Sentence 2", "Sentence 3", "Neither")
ARMS = ("current", "qwen_fixed", "qwen_balanced", "smol2_fixed", "smol2_balanced", "agreement", "union")


def read(path):
    return json.loads(path.read_text())


def sha(value):
    return hashlib.sha256(value).hexdigest()


def publisher_body(path, publisher):
    tree = html.fromstring(path.read_text())
    if publisher == "FAA":
        (body,) = (node for node in tree.find_class("clearfix") if "mb-4" in node.classes and node.findall("p"))
    else:
        (body,) = (node for node in tree.find_class("field--name-body") if len(node.findall(".//p")) >= 4)
    retained = []
    for node in body.findall(".//p"):
        if any(parent.tag in ("blockquote", "figure") for parent in node.iterancestors()):
            continue
        text = " ".join("".join(node.itertext()).split())
        if len(re.findall(r"[^\W_]+(?:[’'-][^\W_]+)*", text)) >= 8 and not any(
            term in text for term in ('“', '”', '"', 'Contact:', '@', 'http', 'www.', 'USDA is an equal opportunity')
        ):
            retained.append(text)
    return " ".join(retained)


def span_positions(text, group):
    return {i for i, char in enumerate(text) if not char.isspace() and any(a <= i < b for a, b in group)}


def read_choices(readings):
    outcomes = []
    by_semantic = [[], [], []]
    fixed = None
    for index, (order, probabilities) in enumerate(readings):
        assert len(probabilities) == 3 and all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities)
        if math.fsum(probabilities) < .2 or min(probabilities) == 0:
            continue
        current = {index: value for index, value in zip(order, probabilities, strict=True)}
        ranked = sorted(order, key=current.__getitem__, reverse=True)
        if index == 0 and current[ranked[0]] / current[ranked[1]] >= 1.5:
            fixed = ranked[0]
        outcomes.append(ranked[0])
        for choice, probability in current.items():
            by_semantic[choice].append(probability)
    balanced = None
    if len(outcomes) == 6:
        scores = [math.prod(values) ** (1 / 6) for values in by_semantic]
        first, second, _ = sorted(range(3), key=scores.__getitem__, reverse=True)
        if outcomes.count(first) >= 4 and scores[first] / scores[second] >= 1.5:
            balanced = first
    return {"fixed": fixed, "balanced": balanced, "orders_available": len(outcomes),
            "order_disagreement": len(set(outcomes)) > 1}


def audit(run):
    freeze = run / "order-freeze.json"
    result, readout = read(run / "order-result.json"), run / "private-order-readout.json"
    assert result["freeze_sha256"] == sha(freeze.read_bytes())
    assert result["readout_sha256"] == sha(readout.read_bytes())
    request_path = run / "requests.json"
    assert read(freeze)["requests_sha256"] == sha(request_path.read_bytes())
    inputs = read(request_path)
    checkpoints = {family: read(run / f"order-{family}.json") for family in FAMILIES}
    generations = {family: read(run / f"generate-{family}.json")["inputs"] for family in FAMILIES}
    originals, sources = {}, read(run / "sources.json")["sources"]
    verified = Counter()
    for key, source in sources.items():
        file = run / (key + ".html")
        assert sha(file.read_bytes()) == source["html_sha256"]
        body = publisher_body(file, source["publisher"])
        assert sha(body.encode()) == source["normalized_body_sha256"]
        original = []
        for anchor in source["anchors"]:
            a, b = anchor["publisher_codepoints"]
            assert 0 <= a < b <= len(body) and sha(body[a:b].encode()) == anchor["sha256"]
            original.append(body[a:b])
            verified["publisher_ranges"] += 1
        originals[key] = original
    counts, useful, human = defaultdict(lambda: defaultdict(Counter)), defaultdict(set), defaultdict(set)
    useful_by_family = defaultdict(set)
    for record in read(readout):
        case = record["case"]
        group, kind, key, target = (case[field] for field in ("group", "kind", "id", "target_index"))
        assert target == sources[group]["target_index"]
        values = originals[group][:4]
        if kind == "human_splice":
            values[target] = originals[group][4]
        elif kind in FAMILIES:
            raw = generations[kind][group]
            assert raw["eos"] and raw["input_sha256"] == sha(values[target].encode())
            values[target] = raw["text"].strip()
        body = " ".join(values)
        source = (run / "fixtures" / (key + ".txt")).read_bytes()
        assert sha(source) == case["sha256"] and source.decode() == body
        bounds, end = [], -1
        for sentence in values:
            start = end + 1
            end = start + len(sentence)
            bounds.append([start, end])
        assert case["replacement"] == bounds[target] and case["replacement_sha256"] == sha(values[target].encode())
        for i, focus in enumerate(record["focus"], 1):
            assert focus["span"] == bounds[i] and focus["sha256"] == sha(values[i].encode())
            verified["focus_ranges"] += 1
        response = {}
        for family, packet in checkpoints.items():
            assert packet["freeze_sha256"] == result["freeze_sha256"]
            reads = []
            for order, prompt, recorded in zip(permutations(range(3)), inputs[key], packet["cases"][key], strict=True):
                assert prompt["source_spans"] == bounds
                heading, options = prompt["prompt"].rsplit("\n\nOptions:\n", 1)
                expected_body = "\n".join(f"{i}. {value}" for i, value in enumerate(values, 1))
                assert heading.split("\n\n", 1)[1] == expected_body
                assert options.splitlines() == [f"{chr(65 + i)}. {OPTIONS[index]}" for i, index in enumerate(order)]
                assert recorded["prompt_sha256"] == sha(prompt["prompt"].encode()) and 0 < recorded["input_tokens"] <= 512
                reads.append((order, recorded["label_probability"]))
                verified["exact_prompts"] += 1
            response[family] = read_choices(reads)
        assert response == record["choices"]
        selected = []
        for i, value in enumerate(record["contexts"]):
            first, last = max(0, i - 1), min(len(bounds), i + 2)
            assert value["spans"] == bounds[first:last]
            assert value["sha256"] == sha(" ".join(values[first:last]).encode())
            assert 0 <= value["native"]["score"] <= 1
            if value["native"]["score"] >= .5:
                selected.append(value["spans"])
            verified["current_context_ranges"] += 1
        proposed = {"current": selected[:6]}
        for family, variants in response.items():
            for variant in ("fixed", "balanced"):
                mark = variants[variant]
                proposed[f"{family}_{variant}"] = [[bounds[mark + 1]]] if mark in (0, 1) else []
        proposed["agreement"] = proposed["qwen_balanced"] if proposed["qwen_balanced"] == proposed["smol2_balanced"] else []
        proposed["union"] = proposed["current"] + proposed["agreement"]
        assert proposed == record["proposals"]
        truth = span_positions(body, [bounds[target]])
        for scope in ("all", sources[group]["publisher"]):
            count = counts[scope][kind]
            count["cases"] += 1
            for family in FAMILIES:
                count[f"{family}_all_orders"] += response[family]["orders_available"] == 6
                count[f"{family}_disagreed_by_order"] += response[family]["order_disagreement"]
                for mode in ("fixed", "balanced"):
                    choice = response[family][mode]
                    count[f"{family}_{mode}_abstentions"] += choice is None
                    count[f"{family}_{mode}_neither"] += choice == 2
            for arm, windows in proposed.items():
                positive = False
                for window in windows:
                    actual = span_positions(body, window)
                    positive |= 10 * len(truth & actual) >= 3 * len(truth | actual)
                count[f"{arm}_flags"] += bool(windows)
                count[f"{arm}_locations"] += positive and kind != "original"
                if kind in FAMILIES and positive:
                    useful[scope, arm].add(group)
                    if scope == "all":
                        useful_by_family[kind, arm].add(group)
                if kind not in FAMILIES and windows:
                    human[scope, arm].add(group)
        verified["fixture_files"] += 1
    assert counts == result["counts"] and verified["fixture_files"] == result["cases"]
    for scope in counts:
        assert result["useful_groups"][scope] == {arm: len(useful[scope, arm]) for arm in ARMS}
        assert result["human_groups"][scope] == {arm: len(human[scope, arm]) for arm in ARMS}
        assert result["extra_useful_groups"][scope] == len(useful[scope, "union"] - useful[scope, "current"])
        assert result["extra_human_groups"][scope] == len(human[scope, "union"] - human[scope, "current"])
    family_gains = {family: len(useful_by_family[family, "union"] - useful_by_family[family, "current"])
                    for family in FAMILIES}
    assert result["extra_useful_groups_by_generator"] == family_gains
    assert result["advance"] == (
        all(useful[scope, "union"] - useful[scope, "current"] for scope in counts if scope != "all")
        and all(family_gains.values()) and len(human["all", "union"] - human["all", "current"]) <= 1
    )
    verified["outcome_counts"] = sum(len(row) for kinds in counts.values() for row in kinds.values()) + len(counts) * 16 + 3
    return dict(verified)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2, sort_keys=True))
