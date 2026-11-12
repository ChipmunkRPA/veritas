# SPDX-License-Identifier: Apache-2.0
"""Verify raw MIME positions and recompute decisions for a saved reply-layout study."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from email import policy
from email.parser import BytesParser
from pathlib import Path

from app.detectors.text.extract import extract, from_string
from app.detectors.text.segment import segment
from scripts.writing_agreement_document_format_transfer import source_inputs
from scripts.writing_checkpoint_agreement import HUMAN


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def require(test, reason):
    if not test:
        raise ValueError(reason)


def part(path, preferred="plain"):
    mime = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    return mime.get_body(preferencelist=(preferred,))


def edit_overlap(text, truth, spans):
    significant = {i for i, ch in enumerate(text) if not ch.isspace()}
    targets = significant.intersection(range(*truth))
    checked = significant.intersection(i for start, end in spans for i in range(start, end))
    return 10 * len(targets & checked) >= 3 * len(targets | checked)


def verify(folder, panels, publication):
    rows, scores, _ = source_inputs(panels)
    freeze = read(folder / "freeze.json")
    observed = read(folder / "result.json")
    require(observed["freeze_sha256"] == digest((folder / "freeze.json").read_bytes())
            and freeze["source_cases"] == [row["text_sha256"] for row in rows], "Frozen sources differ")
    variants = {(row["index"], row["paragraph_count"]): row for row in observed["cases"]}
    totals, altered_groups = defaultdict(Counter), defaultdict(set)
    positions = 0
    for i, source in enumerate(rows):
        canonical = [item.text for item in segment(from_string(source["text"]))]
        eligible = len(canonical) - 1
        expected = [w for w in source["windows"][:eligible] if w["current"] >= .5][:6]
        shared = [w for w in expected if scores[w["sha256"]]["score"] >= .5]
        generated = source["kind"] not in HUMAN
        useful = generated and any(edit_overlap(source["text"], source["truth"], row["spans"]) for row in shared)
        for count in freeze["layouts"]:
            cell = totals[str(count)]
            if count > len(canonical):
                cell["source_unavailable"] += 1
                continue
            row = variants.pop((i, count))
            path = folder / "eml" / f"{i:03d}-{count}.eml"
            require(digest(path.read_bytes()) == row["file_sha256"], "A MIME fixture changed")
            raw = part(path).get_content()
            raw_blocks = raw.replace("\r\n", "\n").split("\n\n")
            indices = [len(canonical) * step // count for step in range(count + 1)]
            expected_blocks = [" ".join(canonical[indices[k]:indices[k + 1]]) for k in range(count)]
            require(raw_blocks[:count] == expected_blocks and raw_blocks[count].startswith("-- \n")
                    and raw_blocks[count + 1].splitlines()[1].startswith("> "), "The original MIME structure differs")
            document = extract(path, ".eml")
            mapped = document.source_map[:count]
            require(len(mapped) == count and all(item["quote_role"] == "body" and item["mime_part"] == "1.1.1"
                    for item in mapped), "Reply scope differs from the decoded MIME roles")
            current = [s for s in segment(document) if s.end <= mapped[-1]["end"]]
            changed = [s.text for s in current] != canonical
            if changed:
                altered_groups[str(count)].add(source["group"])
                require(row.get("unavailable") == "sentences_changed" and row["same_inputs"] == 0,
                        "An input with changed sentence boundaries reused responses")
                cell["sentences_changed"] += 1
            else:
                require(row["body"] == [{k: item[k] for k in ("start", "end")} for item in mapped]
                        and row["selected"] == [value["sha256"] for value in expected],
                        "Exact body offsets or selected model input hashes differ")
                for n in range(eligible):
                    actual = " ".join(s.text for s in current[max(0, n - 1):n + 2])
                    require(digest(actual.encode()) == source["windows"][n]["sha256"], "Recorded context differs from the raw input")
            for block in mapped:
                for start, stop, a, b in block["source_ranges"]:
                    native, decoded = raw[a:b], document.text[start:stop]
                    require(native == decoded or decoded == " " and native.isspace(), "Source coordinates lost native text")
                    positions += 1
            cell.update({"files": 1, "admitted": 1, "eligible": eligible, "generated": generated,
                         "same_inputs": 0 if changed else eligible, "useful": useful and not changed,
                         "human_flag": not generated and not changed and bool(shared),
                         "baseline_useful": useful, "baseline_human_flag": not generated and bool(shared)})
    require(not variants and totals == observed["result"], "Independent counts differ from the frozen comparison")
    controls = []
    for i in range(12):
        inline = folder / "eml" / f"control-{i:02d}-inline_quote.eml"
        html = folder / "eml" / f"control-{i:02d}-html.eml"
        roles = [row["quote_role"] for row in extract(inline, ".eml").source_map]
        html_body = part(html, "html")
        require("quoted" in roles and roles[-1] == "body" and part(html) is None
                and html_body is not None and html_body.get_content_type() == "text/html",
                "An exclusion control no longer represents its declared source structure")
        controls.extend(digest(path.read_bytes()) for path in (inline, html))
    require(controls == observed["controls"], "The exclusion controls changed")
    limits = freeze["gate"]
    qualifies = all(row["files"] and row["same_inputs"] >= limits["min_context_parity"] * row["eligible"]
                    and row["useful"] >= limits["min_generated_locations_retained"] * row["baseline_useful"]
                    and row["human_flag"] <= row["baseline_human_flag"] for row in totals.values())
    require(bool(qualifies) == observed["passed"], "Independent advancement decision differs")
    public = {"schema": observed["schema"], "freeze_sha256": observed["freeze_sha256"], "result": totals,
              "excluded_controls": len(controls), "native_body_source_ranges_checked": positions,
              "affected_original_groups": {count: len(groups) for count, groups in altered_groups.items()},
              "passed": observed["passed"], "serving_change": False}
    with publication.open("x") as out:
        json.dump(public, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps(public, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("folder", "panels", "publication"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    verify(args.folder, args.panels, args.publication)


if __name__ == "__main__":
    main()
