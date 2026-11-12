# SPDX-License-Identifier: Apache-2.0
"""Replay reply-body results independently from the serving eligibility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from email import policy
from email.parser import BytesParser
from pathlib import Path

SCHEMA = "chipmunk.writing-email-body-scope-transfer/1"
DELIVERIES = {"quote": ["quoted"], "signature": ["signature"],
              "signature_and_quote": ["signature", "quoted"]}
CONTROLS = {"quote_only", "inline_quote", "html"}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def summary(document):
    require(document["schema"] == SCHEMA and document["source_groups"] == 28, "Unexpected reply evaluation")
    seen = set()
    hashes = {}
    summary = defaultdict(Counter)
    per_format_cases = defaultdict(set)
    for row in document["outcomes"]:
        delivery = row["delivery"]
        case = row["case"]
        sha = row["file_sha256"]
        require(type(case) is int and 0 <= case < 100 and (delivery, case) not in seen
                and re.fullmatch("[0-9a-f]{64}", sha)
                and (sha not in hashes or hashes[sha] == delivery == "quote_only"), "Duplicate or invalid reply fixture")
        seen.add((delivery, case))
        hashes[sha] = delivery
        per_format_cases[delivery].add(case)
        current = summary[delivery]
        current["files"] += 1
        current["admitted"] += row["admitted"]
        if delivery in CONTROLS:
            require(row["admitted"] is False, "Unsupported email was admitted")
            continue
        spans = row["body_spans"]
        require(delivery in DELIVERIES and row["admitted"] is True and len(spans) in {1, 4}
                and row["excluded_roles"] == DELIVERIES[delivery] and spans[0][0] == 0
                and all(0 <= a < b for a, b in spans)
                and all(x[1] + 2 == y[0] for x, y in zip(spans, spans[1:])),
                "Invalid selected MIME source geometry")
        selected, corroborated = row["selected_contexts"], row["corroborated_contexts"]
        require(0 <= len(corroborated) <= len(selected) <= 6 and set(corroborated) <= set(selected)
                and row["reply_eligible_contexts"] >= len(selected)
                and row["all_primary_contexts"] > row["reply_eligible_contexts"], "Inconsistent saved response coverage")
        require(not row["useful"] or row["synthetic"] and row["baseline_useful"] and corroborated,
                "Invalid retained synthetic location")
        require(row["human_flag"] == (not row["synthetic"] and bool(corroborated))
                and (not row["human_flag"] or row["baseline_human_flag"]), "Invalid retained human flag")
        for name in ("synthetic", "baseline_useful", "useful", "baseline_human_flag", "human_flag",
                     "reply_eligible_contexts", "source_positions"):
            current[name] += row[name]
        current["selected_contexts"] += len(selected)
        current["corroborated_contexts"] += len(corroborated)
    expected = {delivery: 100 for delivery in DELIVERIES} | {delivery: 8 for delivery in CONTROLS}
    require({name: cell["files"] for name, cell in summary.items()} == expected and len(seen) == 324
            and len({frozenset(per_format_cases[k]) for k in CONTROLS}) == 1, "Incomplete paired evaluation")
    reconstructed = {key: dict(value) for key, value in summary.items()}
    require(document["aggregate"] == reconstructed, "Aggregate counts differ from the saved outcomes")
    return reconstructed


def native(run, panels):
    from scripts.writing_agreement_document_format_transfer import source_inputs
    from scripts.writing_checkpoint_agreement import HUMAN, useful

    result = read(run / "result.json")
    freeze_path = run / "freeze.json"
    frozen = read(freeze_path)
    require(result["freeze_sha256"] == digest(freeze_path.read_bytes()), "Source and outcome freezes differ")
    rows, responses, marks = source_inputs(panels)
    checks = Counter()
    for row in result["outcomes"]:
        i, delivery = row["case"], row["delivery"]
        body = rows[i]
        raw = (run / "fixtures" / f"{i:03d}-{delivery}.eml").read_bytes()
        require(digest(raw) == row["file_sha256"], "A retained email differs from its delivered bytes")
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        part = msg.get_body(preferencelist=("plain", "html"))
        if delivery in CONTROLS:
            text = part.get_content().replace("\r\n", "\n")
            if delivery == "quote_only":
                require(text.startswith("On a prior date someone wrote:\n> "), "Quote-only control contains an added reply")
            elif delivery == "inline_quote":
                require(text.startswith(body["text"] + "\n\n> ") and text.endswith("\n\nA reply follows the inline quotation.\n"),
                        "Inline quote control differs")
            else:
                require(part.get_content_type() == "text/html", "HTML control changed format")
            checks["controls"] += 1
            continue
        source = part.get_content().replace("\r\n", "\n")
        text, tail = source.split("\n\n-- ", 1) if "signature" in delivery else source.split("\n\nOn a prior date", 1)
        require(part.get_content_type() == "text/plain" and text == body["text"]
                and digest(text.encode()) == frozen["cases"][i]["text_sha256"] and tail.strip(), "The decoded reply body differs")
        a = 0
        offsets = []
        for paragraph in text.split("\n\n"):
            b = a + len(paragraph)
            offsets.append([a, b])
            a = b + 2
        require(row["body_spans"] == offsets, "Recorded body positions differ from native MIME bytes")
        old = body["windows"]
        # A trailing role adds a third sentence to the final two-sentence context.
        # Earlier exact primary contexts are invariant to the appended MIME text.
        admitted = old[:-1]
        selected = [w for w in admitted if w["current"] >= .5][:6]
        matching = [w for w in selected if responses[w["sha256"]]["score"] >= .5]
        is_ai = body["kind"] not in HUMAN
        require(row["reply_eligible_contexts"] == len(admitted)
                and row["selected_contexts"] == [w["sha256"] for w in selected]
                and row["corroborated_contexts"] == [w["sha256"] for w in matching]
                and row["useful"] == (is_ai and any(useful(text, body["truth"], w["spans"]) for w in matching))
                and row["baseline_useful"] == marks[i]["useful"]["agreement"], "The saved reply-body decisions differ")
        for window in selected:
            require(digest(" ".join(text[a:b] for a, b in window["spans"]).encode()) == window["sha256"]
                    and all(any(a <= c < d <= b for a, b in offsets) for c, d in window["spans"]),
                    "A selected response covers text outside its body paragraph")
        checks["decoded_reply_bodies"] += 1
        checks["exact_selected_inputs"] += len(selected)
    summary(result)
    return result, dict(checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--panels", type=Path)
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args()
    if args.panels:
        result, checks = native(args.source, args.panels)
        if args.publish:
            with args.publish.open("x") as stream:
                json.dump({**result, "independent_native_replay": checks}, stream, indent=2, sort_keys=True)
                stream.write("\n")
        print(json.dumps(checks, sort_keys=True))
    else:
        document = read(args.source)
        print(json.dumps(summary(document), sort_keys=True))


if __name__ == "__main__":
    main()
