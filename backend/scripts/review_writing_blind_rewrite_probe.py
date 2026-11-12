"""Recheck source locations, probe distances and outcome counts without models."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from lxml import html

from app.detectors.text.segment import spans

FAMILIES = ("qwen", "smol2")
ARMS = ("current", *FAMILIES, "both", "union")
WORD = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*")


def load(path):
    return json.loads(path.read_bytes())


def digest(value):
    return hashlib.sha256(value).hexdigest()


def published_text(path):
    tree = html.fromstring(path.read_text())
    (body,) = tree.find_class("nps-news-release__body")
    for node in body.iterdescendants():
        if node.tag == "br":
            node.tail = " " + (node.tail or "")
    ignored = body.xpath('.//script|.//style|.//svg|.//figure|.//ul|.//ol|.//div[contains(@class,"TagList")]'
                         '|.//p[a[contains(@href,"getNewsRSS")]]|.//p[strong[contains(.,"News Release Date")]]')
    for node in ignored:
        node.drop_tree()
    return " ".join("".join(body.itertext()).split())


def normalized_edit_distance(source, response):
    left, right = (WORD.findall(value.casefold()) for value in (source, response))
    matrix = {(i, 0): i for i in range(len(left) + 1)} | {(0, j): j for j in range(len(right) + 1)}
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1):
            matrix[i, j] = min(matrix[i - 1, j - 1] + (a != b), matrix[i - 1, j] + 1, matrix[i, j - 1] + 1)
    return matrix[len(left), len(right)] / max(len(left), len(right))


def valid(source, output):
    text = output["text"].strip()
    if not (output["eos"] and 0 < len(text) <= 450 and "\n" not in text and text[0] not in '"“\'‘#*[' and text[-1] in ".!?"):
        return False
    if not 0.55 <= len(WORD.findall(text)) / len(WORD.findall(source)) <= 1.65:
        return False
    return spans(text, [(0, len(text))]) == [(0, len(text), 0)]


def audit(run):
    freeze = run / "scoring-freeze.json"
    raw = run / "private-responses.json"
    result = load(run / "result.json")
    assert digest(freeze.read_bytes()) == result["scoring_freeze_sha256"]
    assert digest(raw.read_bytes()) == result["response_sha256"]
    originals = {}
    checks = Counter()
    for group, source in load(run / "sources.json")["sources"].items():
        file = run / (group + ".html")
        assert digest(file.read_bytes()) == source["html_sha256"]
        body = published_text(file)
        assert digest(body.encode()) == source["normalized_body_sha256"]
        values = []
        for anchor in source["anchors"]:
            start, end = anchor["publisher_codepoints"]
            text = body[start:end]
            assert digest(text.encode()) == anchor["sha256"]
            values.append(text)
            checks["publisher_sentence_ranges"] += 1
        originals[group] = values
    probes = {family: load(run / f"probe-{family}.json")["inputs"] for family in FAMILIES}
    generation = {family: load(run / f"generate-{family}.json")["inputs"] for family in FAMILIES}
    counts, human, found = defaultdict(Counter), defaultdict(set), defaultdict(set)
    for record in load(raw):
        case = record["case"]
        group, kind, target = case["group"], case["kind"], case["target_index"]
        data = (run / "fixtures" / (case["id"] + ".txt")).read_bytes()
        text = data.decode()
        assert digest(data) == case["sha256"]
        values = originals[group][:4]
        if kind == "human_splice":
            values[target] = originals[group][4]
        if kind in generation:
            response = generation[kind][group]
            assert response["input_sha256"] == digest(values[target].encode()) and response["eos"]
            values[target] = response["text"].strip()
        expected = " ".join(values)
        assert text == expected
        source = []
        for sentence in values:
            position = source[-1][1] + 1 if source else 0
            source.append([position, position + len(sentence)])
        assert case["replacement"] == source[target]
        assert digest(values[target].encode()) == case["replacement_sha256"]
        for index, record_focus in enumerate(record["focus"], 1):
            a, b = record_focus["span"]
            assert [a, b] == source[index] and text[a:b] == values[index]
            assert record_focus["sha256"] == digest(values[index].encode())
            checks["exact_focus_ranges"] += 1
        current = []
        for i, context in enumerate(record["contexts"]):
            first, last = max(0, i - 1), min(len(source), i + 2)
            assert context["spans"] == source[first:last]
            assert context["sha256"] == digest(" ".join(values[first:last]).encode())
            assert 0 <= context["native"]["score"] <= 1
            if context["native"]["score"] >= .5:
                current.append(context["spans"])
            checks["native_context_ranges"] += 1
        marked = {"current": current[:6]}
        for family in FAMILIES:
            independently_measured = []
            for index, focus in enumerate(record["focus"], 1):
                output = probes[family][focus["sha256"]]
                assert output["input_sha256"] == focus["sha256"]
                distance = normalized_edit_distance(values[index], output["text"].strip()) if valid(values[index], output) else None
                independently_measured.append(distance)
                checks["probe_distances"] += 1
            assert independently_measured == record["distances"][family]
            a, b = independently_measured
            assessable = a is not None and b is not None
            counts[kind][f"{family}_available"] += assessable
            marked[family] = [[source[1 if a < b else 2]]] if assessable and abs(a - b) >= .15 else []
        marked["both"] = marked["qwen"] if marked["qwen"] == marked["smol2"] else []
        marked["union"] = marked["current"] + marked["both"]
        assert marked == record["selected"]
        counts[kind]["cases"] += 1
        a, b = source[target]
        truth = {i for i, c in enumerate(text) if a <= i < b and not c.isspace()}
        for arm, windows in marked.items():
            useful = False
            for window in windows:
                positions = {i for i, c in enumerate(text) if not c.isspace() and any(start <= i < stop for start, stop in window)}
                useful |= (10 * len(positions & truth)) >= (3 * len(positions | truth))
            counts[kind][f"{arm}_flags"] += bool(windows)
            counts[kind][f"{arm}_locates_replacement"] += useful and kind != "original"
            if useful and kind in generation:
                found[arm].add(group)
            if windows and kind not in generation:
                human[arm].add(group)
        checks["complete_fixtures"] += 1
    assert result["counts"] == {key: dict(row) for key, row in counts.items()}
    assert result["human_groups"] == {arm: len(human[arm]) for arm in ARMS}
    assert result["useful_groups"] == {arm: len(found[arm]) for arm in ARMS}
    assert result["new_useful_groups"] == len(found["union"] - found["current"])
    assert result["extra_human_groups"] == len(human["union"] - human["current"])
    checks["aggregate_counts"] = sum(len(row) for row in counts.values()) + 12
    return dict(checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2, sort_keys=True))
