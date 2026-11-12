# SPDX-License-Identifier: Apache-2.0
"""Independently reopen MIME source coordinates and verify saved writing responses."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from email import policy
from email.parser import BytesParser
from pathlib import Path

from app.detectors.text.extract import extract
from app.detectors.text.segment import segment
from app.evidence.document_matched_context import one_complete_window
from scripts.writing_agreement_document_format_transfer import source_inputs
from scripts.writing_checkpoint_agreement import HUMAN


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def require(valid, message):
    if not valid:
        raise ValueError(message)


def review(folder, initial, earlier, panels, publication):
    freeze = read(folder / "freeze.json")
    aggregate = read(folder / "aggregate.json")
    source_rows, secondary, _ = source_inputs(panels)
    original = read(earlier / "result.json")
    outcomes = {(row["source"], row["layout"]): row for row in read(folder / "decisions.json")}
    initial_primary, initial_secondary = (read(initial / f"{arm}.json") for arm in ("primary", "second"))
    first = initial_primary["responses"]
    native_scores = read(folder / "responses.json")
    scores = native_scores["readings"]
    seal = sha((folder / "freeze.json").read_bytes())
    require(aggregate["freeze_sha256"] == native_scores["freeze_sha256"] == seal and set(scores) == set(freeze["requests"])
            and sha((earlier / "result.json").read_bytes()) == freeze["scoped_inputs"]
            and all(sha((initial / name).read_bytes()) == expected for name, expected in freeze["initial"].items()),
            "The earlier exact-input result or native score commitment differs")
    require(all(len(scores[key]) == len(hashes)
                and key == sha(json.dumps(hashes, separators=(",", ":")).encode())
                and all(0 <= float.fromhex(item["score_hex"]) <= 1 and type(item["tokens"]) is int
                        and 0 < item["tokens"] <= 512 for item in scores[key]) for key, hashes in freeze["requests"].items()),
            "A native request lacks a complete second response")
    seen_requests = set()

    def agree(windows):
        hashes = [w["sha256"] for w in windows]
        request = sha(json.dumps(hashes, separators=(",", ":")).encode())
        require(freeze["requests"].get(request) == hashes, "The exact service batch differs")
        seen_requests.add(request)
        return request, [w for w, response in zip(windows, scores[request], strict=True) if float.fromhex(response["score_hex"]) >= .5]

    before = {}
    for index, source in enumerate(source_rows):
        selected = [w for w in source["windows"][:-1] if w["current"] >= .5][:6]
        _, shared = agree(selected)
        significant = {i for i, ch in enumerate(source["text"]) if not ch.isspace()}
        truth = significant.intersection(range(*source["truth"]))
        useful = False
        for window in shared:
            included = significant.intersection(i for start, stop in window["spans"] for i in range(start, stop))
            useful = useful or 10 * len(truth & included) >= 3 * len(truth | included)
        generated = source["kind"] not in HUMAN
        before[index] = generated and useful, not generated and bool(shared)
    seen_fresh = set()
    counts, false_groups, positions = defaultdict(Counter), defaultdict(set), 0
    for row in original["cases"]:
        index, count = row["index"], row["paragraph_count"]
        file = earlier / "eml" / f"{index:03d}-{count}.eml"
        raw = file.read_bytes()
        require(sha(raw) == row["file_sha256"], "Native EML bytes changed")
        message = BytesParser(policy=policy.default).parsebytes(raw)
        plain = message.get_body(preferencelist=("plain",))
        require(plain is not None and plain.get_content_type() == "text/plain", "The selected MIME part is missing")
        mime = plain.get_content()
        doc = extract(file, ".eml")
        mapped = doc.source_map
        require(len(mapped) > count and all(p["quote_role"] == "body" for p in mapped[:count])
                and {p["quote_role"] for p in mapped[count:]} <= {"signature", "quoted"}
                and {p["mime_part"] for p in mapped} == {doc.meta["email"]["body_part"]},
                "Body, quote and signature do not map to one verified MIME part")
        for p in mapped[:count]:
            cursor = p["start"]
            for start, stop, a, b in p["source_ranges"]:
                content = mime[a:b]
                decoded = doc.text[start:stop]
                require(start == cursor and (content == decoded or content.isspace() and decoded == " "),
                        "Decoded reply positions differ from the native source")
                cursor = stop
                positions += 1
            require(cursor == p["end"], "Reply paragraph has missing source ranges")
        candidate_text = doc.text[:mapped[count-1]["end"]]
        source = source_rows[index]
        original_positions = [i for i, char in enumerate(source["text"]) if not char.isspace()]
        after = [i for i, char in enumerate(candidate_text) if not char.isspace()]
        require("".join(source["text"][i] for i in original_positions) == "".join(candidate_text[i] for i in after),
                "Repacked reply altered source words")
        truth = {right for left, right in zip(original_positions, after, strict=True) if source["truth"][0] <= left < source["truth"][1]}
        target_indices = set(after)
        units = segment(doc)
        chosen = []
        native = {w["sha256"]: w["current"] for w in source["windows"]}
        for i in range(len(units)):
            context = units[max(0, i-1):i+2]
            if context[-1].end > mapped[count-1]["end"]:
                continue
            for item in context:
                require(any(p["start"] <= item.start < item.end <= p["end"] for p in mapped[:count]),
                        "A model input crosses an excluded MIME role")
            content = " ".join(doc.text[item.start:item.end] for item in context)
            key = sha(content.encode())
            if key in native:
                response = native[key]
            else:
                seen_fresh.add(key)
                require(one_complete_window(first[key]["coverage"]), "A fresh context has incomplete primary token coverage")
                response = first[key]["score"]
            if response >= .5 and len(chosen) < 6:
                covered = target_indices.intersection(j for item in context for j in range(item.start, item.end))
                chosen.append({"sha256": key, "overlap": 10 * len(truth & covered) >= 3 * len(truth | covered)})
        request, corroborated = agree(chosen)
        generated = source["kind"] not in HUMAN
        useful = generated and any(window["overlap"] for window in corroborated)
        matched = [window["sha256"] for window in corroborated]
        false = not generated and bool(matched)
        saved = outcomes.pop((index, count))
        require((saved["request"], saved["corroborated"], saved["useful"], saved["human_flag"])
                == (request, matched, useful, false), "Source replay differs from the selected outcome")
        prior_useful, prior_false = before[index]
        if false and not prior_false:
            false_groups[str(count)].add(source["group"])
        counts[str(count)].update({"files": 1, "generated": generated, "candidate_contexts": len(chosen),
              "corroborated_contexts": len(matched), "segmentation_changed": bool(row.get("unavailable")),
              "old_useful": prior_useful, "new_useful": useful, "old_human_flags": prior_false,
              "new_human_flags": false, "added_human_flags": false and not prior_false})
    initial_lock = read(initial / "freeze.json")
    require(not outcomes and seen_fresh == set(initial_lock["forward_input_hashes"]) - set(initial_lock["parity_input_hashes"])
            and seen_requests == set(scores)
            and counts == aggregate["layout_outcomes"]
            and {key: len(groups) for key, groups in false_groups.items()} == aggregate["new_human_groups"],
            "Independent receipt does not cover the full frozen input roster")
    verified = {layout: 10 * row["new_useful"] >= 9 * row["old_useful"] and row["added_human_flags"] <= 1
                for layout, row in counts.items()}
    require(verified == aggregate["layouts_pass"] and all(verified.values()) == aggregate["all_pass"], "Layout decision differs")
    controls = []
    for i in range(12):
        html = (earlier / "eml" / f"control-{i:02d}-html.eml").read_bytes()
        quoted = earlier / "eml" / f"control-{i:02d}-inline_quote.eml"
        roles = [p["quote_role"] for p in extract(quoted, ".eml").source_map]
        require(BytesParser(policy=policy.default).parsebytes(html).get_body(preferencelist=("plain",)) is None
                and "quoted" in roles and roles[-1] == "body", "A source exclusion control changed")
        controls.extend((sha(quoted.read_bytes()), sha(html)))
    require(controls == original["controls"], "The MIME control roster changed")
    parity_hashes = initial_lock["parity_input_hashes"]
    historic_primary = {w["sha256"]: w["current"] for row in source_rows for w in row["windows"]}
    parity = aggregate["initial_measurements"]
    for arm, actual, expected in (("primary", first, historic_primary),
                                  ("second", initial_secondary["responses"], {h: row["score"] for h, row in secondary.items()})):
        deltas = [abs(actual[h]["score"] - expected[h]) for h in parity_hashes]
        crossings = sum((actual[h]["score"] >= .5) != (expected[h] >= .5) for h in parity_hashes)
        require(abs(max(deltas) - parity["initial_max_raw_deltas"][arm]) <= 1e-6
                and crossings == parity["initial_point_crossings"][arm], "Initial batch comparison changed")
    output = {**aggregate, "independent_native_ranges": positions, "independent_cases": len(original["cases"]),
              "independent_fresh_hashes": len(seen_fresh), "decision_sha256": sha((folder / "decisions.json").read_bytes())}
    with publication.open("x") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Verified {len(original['cases'])} native replies, {positions} source ranges, {len(seen_fresh)} new inputs and {len(controls)} excluded controls.")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("folder", "initial", "earlier", "panels", "publication"):
        cli.add_argument(key, type=Path)
    args = cli.parse_args()
    review(args.folder, args.initial, args.earlier, args.panels, args.publication)


if __name__ == "__main__":
    main()
