"""Independently replay source commitments and outcomes without model imports."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

ARMS = ("current", "second", "agreement")
HUMAN = {"before", "edited"}


class TextReader(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def normalize(value):
    return " ".join(value.split())


def sha(value):
    if isinstance(value, Path):
        return hashlib.sha256(value.read_bytes()).hexdigest()
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def source_text(run, row, side, receipt):
    archive = run / "publisher" / receipt["record"]
    check(sha(archive) == receipt["publisher_record_sha256"], "Publisher record differs")
    if row["publisher"] == "18F":
        revision, name = receipt["revision"], row["group"].removeprefix("guide-")
        revisions = {entry["sha"]: entry["commit"]["committer"]["date"] for entry in read(archive)}
        check(revisions[revision] == receipt["date"] < "2020-01-01T00:00:00Z", "Guide time or commit differs")
        check(sha(run / "publisher/guide-license.md") == receipt["license_sha256"], "Guide terms differ")
        document = run / "publisher" / f"guide-{name}-{revision}.md"
        check(sha(document) == receipt["body_sha256"], "Guide body differs")
        value = re.sub(r"\[([^\]\n]+)\]\([^\n)]+\)", r"\1", document.read_text()).replace("**", "").replace("__", "")
    else:
        selected = [entry for entry in read(archive)["items"] if entry.get("revision_guid") == receipt["revision"]
                    and "body" in entry and entry.get("content_license") == receipt["license"]]
        match, = selected
        check(match["creation_date"] < 1577836800 and match["post_id"] == int(row["group"].rsplit("-", 1)[1]),
              "Unexpected question revision or date")
        check(sha(match["body"]) == receipt["body_sha256"], "Question body differs")
        parser = TextReader()
        parser.feed(match["body"])
        value = html.unescape("".join(parser.parts))
    cursor = 0
    text = normalize(value)
    for sentence in row[side]:
        cursor = text.find(sentence, cursor)
        check(cursor >= 0, "A scored source sentence was absent from its named revision")
        cursor += len(sentence)


def mask(text, regions):
    return sum(1 << i for start, end in regions for i in range(start, end) if not text[i].isspace())


def recorded_cases(run):
    sources = read(run / "source-receipt.json")
    originals = read(run / "originals.json")
    check(sha(run / "originals.json") == sources["originals_sha256"], "Source roster digest differs")
    original_by_group = {row["group"]: row for row in originals}
    for receipt in sources["groups"]:
        row = original_by_group[receipt["group"]]
        focus = row["target"]
        check(len(row["before"]) == len(row["after"]) == 4 and focus in (1, 2)
              and all((a != b) == (i == focus) for i, (a, b) in enumerate(zip(row["before"], row["after"], strict=True))),
              "Human edit geometry differs")
        for side in ("before", "after"):
            check([sha(value) for value in row[side]] == receipt[side + "_sentences_sha256"], "Sentence commitments differ")
            source_text(run, row, side, receipt["revisions"][side])
    cases = read(run / "cases.json")["cases"]
    groups = Counter((case["group"], case["role"]) for case in cases)
    check(all(count == 1 for count in groups.values())
          and all((group, "before") in groups and (group, "edited") in groups for group in original_by_group),
          "Missing or duplicate human counterparts")
    for case in cases:
        row, role = original_by_group[case["group"]], case["role"]
        parts = row["after"] if role == "edited" else row["before"]
        text = case["text"]
        i = case["target"]
        start, end = case["focus"]
        check(i == row["target"] and sha(text) == case["text_sha256"]
              and text[:start] == " ".join(parts[:i]) + " "
              and text[end:] == " " + " ".join(parts[i + 1:]), "Fixture boundaries differ")
        focus = text[start:end]
        if role in HUMAN:
            check(focus == parts[i], "Human source focus differs")
        else:
            packet = read(run / f"generated-{role}.json")
            value = packet["results"][case["group"]]
            check(value["eos"] and value["text"].strip() == focus and value["input_sha256"] == sha(row["before"][i]),
                  "Admitted replacement differs from its model receipt")
        yield case, [*parts[:i], focus, *parts[i + 1:]]


def reconstruct(case, parts, scores):
    windows, left = [], 0
    anchors = []
    for value in parts:
        right = left + len(value)
        anchors.append([left, right])
        left = right + 1
    text = case["text"]
    positives = {name: [] for name in ARMS}
    for position in range(4):
        a, b = max(0, position - 1), position + 2
        source = " ".join(parts[a:b])
        digest = sha(source)
        windows.append({"spans": anchors[a:b], "sha256": digest})
        for arm, responses in scores.items():
            value = responses[digest]
            coverage = value["coverage"]
            check(coverage["token_windows"] == 1 and 0 < coverage["tokens"] <= 512
                  and coverage.get("omitted_tokens", 0) == 0, "Incomplete model input")
            if value["score"] >= .5:
                positives[arm].append(position)
        if all(position in positives[name] for name in ("current", "second")):
            positives["agreement"].append(position)
    target = mask(text, [case["focus"]])
    relevant = {}
    for position, window in enumerate(windows):
        marked = mask(text, window["spans"])
        relevant[position] = (target & marked).bit_count() * 10 >= (target | marked).bit_count() * 3
    return {key: case[key] for key in ("group", "publisher", "role", "text_sha256", "focus", "target")} | {
        "windows": windows,
        "selected": positives,
        "useful": {arm: any(relevant[position] for position in positions) for arm, positions in positives.items()},
        "exact_focus": {arm: case["target"] in positions for arm, positions in positives.items()},
    }


def tally(subset, arm, expected):
    positives = {row["group"] for row in subset if row["role"] == "before" and row["selected"][arm]}
    negatives = {row["group"] for row in subset if row["role"] == "edited" and row["selected"][arm]}
    groups = {row["group"] for row in subset}
    check(len(positives | negatives) == expected["human_source_groups_flagged"], "Human group count differs")
    check(expected["human_before_to_after"] == {
        "no_flag_either": len(groups - positives - negatives), "added": len(negatives - positives),
        "removed": len(positives - negatives), "flagged_both": len(positives & negatives)}, "Human transitions differ")
    for role, published in expected["roles"].items():
        rows = [row for row in subset if row["role"] == role]
        marked = [row for row in rows if row["selected"][arm]]
        relevant = [row for row in marked if row["useful"][arm]]
        counts = {"files": len(rows), "flagged": len(marked)}
        if role == "edited":
            counts["human_edit_overlapped"] = len(relevant)
        elif role not in HUMAN:
            counts.update(useful=len(relevant), wrong_location=len(marked) - len(relevant), unflagged=len(rows) - len(marked),
                          exact_focus=sum(row["exact_focus"][arm] for row in rows),
                          lost_useful=sum(row["useful"]["current"] and not row["useful"][arm] for row in rows))
        check(counts == published, "One arm and role count differs")


def review(run):
    result = read(run / "aggregate.json")
    check(result["generation_sha256"] == sha(run / "generation-freeze.json")
          and result["scoring_sha256"] == sha(run / "scoring-freeze.json")
          and result["readout_sha256"] == sha(run / "private-readout.json"), "A committed run artifact differs")
    freeze = read(run / "scoring-freeze.json")
    check(freeze["cases_sha256"] == sha(run / "cases.json")
          and all(sha(run / f"generated-{family}.json") == digest for family, digest in freeze["generated_sha256"].items()),
          "The frozen case or generation packet differs")
    scores = {}
    for arm in ("current", "second"):
        path = run / f"{arm}.json"
        check(sha(path) == result["score_receipts"][arm]["sha256"], "Classifier receipt differs")
        scores[arm] = read(path)["responses"]
    rows = [reconstruct(case, parts, scores) for case, parts in recorded_cases(run)]
    inputs = {window["sha256"] for row in rows for window in row["windows"]}
    check(all(set(arm) == inputs for arm in scores.values()), "Classifier input set differs")
    check(rows == read(run / "private-readout.json") and len(rows) == result["files"]
          and len(rows) * 4 == result["windows"] and len(inputs) == result["unique_inputs"], "Exact readout or denominator differs")
    check(set(result["outcomes"]) == {"all", *(row["publisher"] for row in rows)}, "Missing publisher aggregate")
    for scope, record in result["outcomes"].items():
        subset = [row for row in rows if scope in ("all", row["publisher"])]
        for arm in ARMS:
            tally(subset, arm, record[arm])
    return {"source_versions_checked": 2 * len(result["sources"]["groups"]), "verified_files": len(rows),
            "exact_windows": len(rows) * 4, "unique_inputs": len(inputs), "verified_scope_arms": len(result["outcomes"]) * len(ARMS)}


def review_elife(run, source):
    result = read(run / "aggregate.json")
    readout = run / "private-readout.json"
    archive = source / "private-controls.jsonl"
    check(result["freeze_sha256"] == sha(run / "freeze.json") and result["readout_sha256"] == sha(readout)
          and result["prior_controls_sha256"] == sha(archive), "Scientific publisher commitments differ")
    controls = [json.loads(line) for line in archive.read_bytes().splitlines()]
    rows = read(readout)
    prior = {row["case"]: row for row in controls if row["role"] == "confirmation" and row["status"] == "constructed"}
    check({row["case"] for row in rows} == set(prior), "Scientific control roster differs")
    packets = {}
    for arm in ("current", "second"):
        path = run / f"{arm}.json"
        check(sha(path) == result["score_receipts"][arm]["sha256"], "Scientific classifier receipt differs")
        packets[arm] = read(path)["responses"]
    inputs, n_windows = set(), 0
    for row in rows:
        record = prior[row["case"]]
        source_ranges = row["source_paragraphs"]
        text = record["text"][source_ranges[0][0]:source_ranges[-1][1]]
        check(len(source_ranges) == 4 and source_ranges in [record["paragraph_codepoints"][i:i + 4] for i in (0, 1)]
              and row["group"] == record["article_group"] and row["arm"] == record["arm"]
              and row["full_input_sha256"] == sha(record["text"]) and row["text_sha256"] == sha(text),
              "Scientific excerpt differs")
        windows = row["windows"]
        sentence_ranges = [window["spans"][int(i != 0)] for i, window in enumerate(windows)]
        responses = {arm: [] for arm in ARMS}
        for index, window in enumerate(windows):
            spans = sentence_ranges[max(0, index - 1):index + 2]
            check(window["spans"] == spans, "Scientific context is misaligned")
            digest = sha(" ".join(text[a:b] for a, b in spans))
            check(window["sha256"] == digest, "Scientific context digest differs")
            inputs.add(digest)
            n_windows += 1
            for arm, packet in packets.items():
                response = packet[digest]
                coverage = response["coverage"]
                check(coverage["token_windows"] == 1 and 0 < coverage["tokens"] <= 512
                      and coverage.get("omitted_tokens", 0) == 0, "Scientific classifier saw a partial context")
                if response["score"] >= .5 and len(responses[arm]) < 6:
                    responses[arm].append(index)
        responses["agreement"] = [index for index in responses["current"] if index in responses["second"]]
        check(row["marked"] == responses, "Scientific control marks differ")
    check(len(rows) == result["cases"] and n_windows == result["windows"] and len(inputs) == result["inputs"]
          and all(set(packet) == inputs for packet in packets.values()), "Scientific control denominator differs")
    for scope, expected in result["outcomes"].items():
        selected = [row for row in rows if scope in ("all", row["arm"])]
        flags = {arm: [row for row in selected if row["marked"][arm]] for arm in ARMS}
        check(expected == {"human_files": len(selected), "human_groups": len({row["group"] for row in selected}),
                           "arms": {arm: {"flagged_files": len(values), "flagged_groups": len({row["group"] for row in values})}
                                    for arm, values in flags.items()}}, "Scientific control aggregate differs")
    return {"verified_files": len(rows), "exact_windows": n_windows, "unique_inputs": len(inputs),
            "verified_scope_arms": len(result["outcomes"]) * len(ARMS)}


def review_olmo(run):
    result = read(run / "aggregate.json")
    bindings = {"freeze_sha256": "freeze.json", "cases_sha256": "cases.json", "generated_sha256": "generated-olmo.json",
                "readout_sha256": "private-readout.json"}
    check(all(result[key] == sha(run / path) for key, path in bindings.items())
          and result["initial_result_sha256"] == sha(run.parent / "aggregate.json")
          and result["originals_sha256"] == sha(run.parent / "originals.json"), "Supplemental freeze differs")
    rows = read(run / "cases.json")["cases"]
    sources = {row["group"]: row for row in read(run.parent / "originals.json")}
    generated = read(run / "generated-olmo.json")["results"]
    check(set(sources) == set(generated) and len(sources) == result["planned"], "Supplemental attempt roster differs")
    scores = {}
    for arm in ("current", "second"):
        path = run / f"{arm}.json"
        check(sha(path) == result["score_receipts"][arm]["sha256"], "Supplemental model receipt differs")
        scores[arm] = read(path)["responses"]
    rebuilt = []
    for row in rows:
        source = sources[row["group"]]
        reply = generated[row["group"]]
        i = source["target"]
        start, end = row["focus"]
        before = source["before"]
        replacement = row["text"][start:end]
        parts = [*before[:i], replacement, *before[i + 1:]]
        check(row["role"] == "olmo" and i == row["target"] and reply["eos"] and reply["text"].strip() == replacement
              and reply["input_sha256"] == sha(before[i]) and " ".join(parts) == row["text"]
              and row["text_sha256"] == sha(row["text"]) and row["publisher"] == source["publisher"],
              "Supplemental generated fixture differs")
        rebuilt.append(reconstruct(row, parts, scores))
    check(rebuilt == read(run / "private-readout.json") and len(rebuilt) == result["admitted"], "Supplemental readout differs")
    digests = {window["sha256"] for row in rebuilt for window in row["windows"]}
    check(all(set(packet) == digests for packet in scores.values()) and len(digests) == result["unique_inputs"],
          "Supplemental input count differs")
    for arm in ARMS:
        marked = [row for row in rebuilt if row["selected"][arm]]
        hits = [row for row in rebuilt if row["useful"][arm]]
        counts = {"flagged": len(marked), "useful": len(hits), "wrong_location": len(marked) - len(hits),
                  "unflagged": len(rebuilt) - len(marked), "exact_focus": sum(row["exact_focus"][arm] for row in rebuilt),
                  "lost_useful": sum(row["useful"]["current"] and not row["useful"][arm] for row in rebuilt)}
        check(result["outcomes"][arm] == counts, "Supplemental model aggregate differs")
    return {"verified_attempts": len(sources), "verified_files": len(rebuilt), "unique_inputs": len(digests)}


def asset_rows(pins):
    return [{"file": name, "sha256": value} for name, value in sorted(pins.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--publish", type=Path)
    parser.add_argument("--elife", type=Path)
    parser.add_argument("--elife-source", type=Path)
    parser.add_argument("--olmo", type=Path)
    args = parser.parse_args()
    audit = review(args.run)
    check(bool(args.elife) == bool(args.elife_source), "Supply both eLife paths")
    aggregate = read(args.run / "aggregate.json") | {"independent_replay": dict(audit)}
    if args.elife:
        elife_audit = review_elife(args.elife, args.elife_source)
        aggregate["publisher_controls"] = read(args.elife / "aggregate.json") | {"independent_replay": elife_audit}
        audit["publisher_controls"] = elife_audit
    if args.olmo:
        olmo_audit = review_olmo(args.olmo)
        aggregate["third_generator"] = read(args.olmo / "aggregate.json") | {"independent_replay": olmo_audit}
        audit["third_generator"] = olmo_audit
    if args.publish:
        aggregate["checkpoints"] = {arm: asset_rows(pins) for arm, pins in aggregate["checkpoints"].items()}
        for record in aggregate["generators"]["weights"].values():
            record["assets"] = asset_rows(record["assets"])
        if args.elife:
            publisher = aggregate["publisher_controls"]
            publisher["checkpoints"] = {arm: asset_rows(pins) for arm, pins in publisher["checkpoints"].items()}
        if args.olmo:
            model = aggregate["third_generator"]["model"]
            model["assets"] = asset_rows(model["assets"])
        with args.publish.open("x") as out:
            json.dump(aggregate, out, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
            out.write("\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
