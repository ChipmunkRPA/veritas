"""Replay grammar-seam inputs, thresholds, highlights and outcome counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from app.detectors.text.segment import spans
from scripts.writing_grammar_seams import source_blocks

ARMS = ("current", "unigram", "sequence", "union")
TOKEN = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*|[.!?;:,-]")
WORD = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*")


def read(path):
    return json.loads(path.read_text())


def sha(value):
    return hashlib.sha256(value).hexdigest()


def bags(value, classes, order):
    codes = []
    for word in TOKEN.findall(value.casefold()):
        code = "sentence" if word in (".", "!", "?") else word
        if code != "sentence" and word not in (";", ":", ",", "-"):
            code = classes.get(word, "number" if any(c.isdecimal() for c in word) else "content")
        codes.append(code)
    if len(codes) < 20:
        return None
    frequencies = Counter(zip(*(codes[offset:] for offset in range(order))))
    total = sum(frequencies.values())
    return {key: math.sqrt(count / total) for key, count in frequencies.items()}


def separation(left, right):
    return math.fsum((left.get(key, 0) - right.get(key, 0)) ** 2 for key in left.keys() | right.keys()) / 2


def replay_scores(text, bounds, classes, order):
    result = {}
    vectors = [bags(text[start:end], classes, order) for start, end in bounds]
    for i in range(1, len(vectors) - 1):
        left, focus, right = vectors[i - 1:i + 2]
        if left and focus and right:
            result[i] = min(separation(left, focus), separation(focus, right)) - separation(left, right)
    return result


def selected(scores, cutoff):
    if not scores:
        return None
    winners = sorted(scores, key=lambda index: -scores[index])
    first = winners[0]
    second = winners[1] if len(winners) > 1 else None
    if scores[first] > cutoff + 1e-12 and (second is None or scores[first] - scores[second] > 1e-12):
        return first
    return None


def chars(text, ranges):
    return {i for start, end in ranges for i in range(start, end) if not text[i].isspace()}


def admitted(original, response):
    value = response["text"].strip()
    before = [word.casefold() for word in WORD.findall(original)]
    after = [word.casefold() for word in WORD.findall(value)]
    ratio = len(WORD.findall(value)) / len(WORD.findall(original))
    if (response["finish"] != "stop" or not .65 <= ratio <= 1.35 or "\n" in value
            or value.startswith(('"', '“', '*', '#')) or not value.endswith((".", "!", "?"))):
        return False
    costs = list(range(len(before) + 1))
    for j, right in enumerate(after, 1):
        diagonal, costs[0] = costs[0], j
        for i, left in enumerate(before, 1):
            previous = costs[i]
            costs[i] = min(costs[i - 1] + 1, previous + 1, diagonal + int(left != right))
            diagonal = previous
    return costs[-1] / max(len(before), len(after)) >= .25


def verify(run):
    freeze_path = run / "protocol.json"
    freeze = read(freeze_path)
    families = freeze["weights"].keys()
    source_path = run / "sources.json"
    dev_path, points_path = run / "development.json", run / "points.json"
    assert freeze["source_sha256"] == sha(source_path.read_bytes())
    assert freeze["originals_sha256"] == sha((run / "originals.json").read_bytes())
    assert freeze["development_sha256"] == sha(dev_path.read_bytes())
    assert freeze["points_sha256"] == sha(points_path.read_bytes())
    classes = {word: tag for tag, items in freeze["word_classes"].items() for word in items.split()}
    points = read(points_path)
    verified = Counter()
    development = read(dev_path)
    for order, point in points.items():
        groups = defaultdict(list)
        for row in development:
            assert sha(row["text"].encode()) == row["sha256"]
            groups[row["group"]].extend(replay_scores(row["text"], row["bounds"], classes, int(order)).values())
        maxima = {key: max(values) for key, values in groups.items() if values}
        assert maxima.keys() == point["group_maxima"].keys()
        assert all(abs(value - point["group_maxima"][key]) < 1e-12 for key, value in maxima.items())
        rank = math.ceil(9 * len(maxima) / 10) - 1
        assert abs(point["cutoff"] - sorted(maxima.values())[rank]) < 1e-12
        verified["development_group_maxima"] += len(maxima)

    sources = read(source_path)["admitted"]
    originals = {}
    for group, source in sources.items():
        path = run / "sources" / (group + (".pdf" if source["publisher"] == "copyright" else ".html"))
        assert sha(path.read_bytes()) == source["sha256"]
        available = {tuple(location): value for location, value in source_blocks(path, source["publisher"])[3]}
        paragraphs = []
        for anchor in source["source_paragraphs"]:
            value = available[tuple(anchor["location"])]
            assert sha(value.encode()) == anchor["sha256"]
            paragraphs.append(value)
        assert len(paragraphs) == 5
        originals[group] = paragraphs
        verified["source_blocks"] += len(paragraphs)

    generation = {family: read(run / f"generated-{family}.json") for family in families}
    assert all(packet["protocol_sha256"] == sha(freeze_path.read_bytes()) for packet in generation.values())
    casefile = read(run / "cases.json")
    for family, packet in generation.items():
        assert packet["groups"].keys() == sources.keys()
        for group, response in packet["groups"].items():
            original = originals[group][sources[group]["target"]]
            assert response["input_sha256"] == sha(original.encode())
            assert (casefile["admission"][f"{group}/{family}"] == "admitted") == admitted(original, response)
            verified["generation_dispositions"] += 1
    expected_keys = {(group, kind) for group in sources for kind in ("original", "human_splice")}
    expected_keys.update(tuple(key.split("/")) for key, status in casefile["admission"].items() if status == "admitted")
    assert expected_keys == {(case["group"], case["kind"]) for case in casefile["cases"]}
    details = {row["case"]: row for row in read(run / "private-readout.json")}
    assert len(details) == len(expected_keys) == len(casefile["cases"])
    baseline = read(run / "baseline.json")
    counts, useful, humans = defaultdict(Counter), defaultdict(set), defaultdict(set)
    for case in casefile["cases"]:
        group, kind, target = (case[field] for field in ("group", "kind", "target"))
        publisher, text, bounds, truth = (case[field] for field in ("publisher", "text", "bounds", "span"))
        assert publisher == sources[group]["publisher"] and target == sources[group]["target"] and truth == bounds[target]
        expected = originals[group][:4]
        if kind == "human_splice":
            expected[target] = originals[group][4]
        elif kind in families:
            response = generation[kind]["groups"][group]
            assert response["input_sha256"] == sha(expected[target].encode()) and response["finish"] == "stop"
            expected[target] = " ".join(response["text"].split())
        assert text == "\n\n".join(expected) and sha(text.encode()) == case["text_sha256"]
        assert expected == [text[start:end] for start, end in bounds]
        assert bounds[0][0] == 0 and bounds[-1][1] == len(text)
        assert all(text[end:following] == "\n\n" for (_, end), (following, _) in zip(bounds, bounds[1:]))
        verified["fixture_paragraph_ranges"] += len(bounds)
        label = f"{group}/{kind}"
        record = details[label]
        sentences = [[start, end] for start, end, _ in spans(text, bounds)]
        current = []
        for index, context in enumerate(baseline["contexts"][label]):
            expected_window = sentences[max(0, index - 1):index + 2]
            assert context["spans"] == expected_window
            request = " ".join(text[start:end] for start, end in expected_window)
            assert context["sha256"] == sha(request.encode())
            score = baseline["native"][context["sha256"]]["score"]
            assert 0 <= score <= 1
            if score >= .5:
                current.append(expected_window)
            verified["current_context_ranges"] += 1
        assert len(baseline["contexts"][label]) == len(sentences)
        arms = {"current": current[:6]}
        for order, arm in (("1", "unigram"), ("3", "sequence")):
            values = replay_scores(text, bounds, classes, int(order))
            assert {str(index) for index in values} == record["scores"][order].keys()
            assert all(abs(value - record["scores"][order][str(index)]) < 1e-12 for index, value in values.items())
            candidate = selected(values, points[order]["cutoff"])
            assert candidate == record["selected"][order]
            arms[arm] = [[bounds[candidate]]] if candidate is not None else []
            verified["paragraph_scores"] += len(values)
        arms["union"] = arms["current"] + arms["sequence"]
        assert arms == record["regions"]
        known = chars(text, [truth])
        for scope in ("all", publisher):
            row = counts[f"{scope}/{kind}"]
            row["files"] += 1
            for order, arm in (("1", "unigram"), ("3", "sequence")):
                row[f"{arm}_unassessed"] += not record["scores"][order]
            for arm, regions in arms.items():
                area = [chars(text, window) for window in regions]
                hit = any(10 * len(known & pixels) >= 3 * len(known | pixels) for pixels in area)
                assert hit == record["useful"][arm]
                row[f"{arm}_flags"] += bool(regions)
                row[f"{arm}_useful"] += hit and kind in families
                if kind in families and hit:
                    useful[scope, arm].add(group)
                    useful[kind, arm].add(group)
                elif kind not in families and regions:
                    humans[scope, arm].add(group)
        verified["fixtures"] += 1
    result = read(run / "result.json")
    assert counts == result["counts"]
    gains = {axis: len(useful[axis, "union"] - useful[axis, "current"]) for axis in result["additional_useful_groups"]}
    assert gains == result["additional_useful_groups"]
    assert {arm: len(humans["all", arm]) for arm in ARMS} == result["human_groups"]
    extra = len(humans["all", "union"] - humans["all", "current"])
    assert extra == result["additional_human_groups"]
    represented = {case["kind"] for case in casefile["cases"]} & set(families)
    if "assessed_generator_families" in result:
        assert result["assessed_generator_families"] == sorted(represented)
    required = set(gains) - {"all"} - (set(families) - represented)
    assert result["advance"] == (extra <= 1 and len(represented) >= 2 and all(gains[axis] > 0 for axis in required))
    verified["count_values"] = sum(map(len, counts.values())) + len(gains) + len(ARMS) + 2
    return dict(verified)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2, sort_keys=True))
