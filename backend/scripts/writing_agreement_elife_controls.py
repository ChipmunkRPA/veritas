"""Measure writing agreement on retained eLife publisher prose and constructed human controls."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from time import monotonic

from app.detectors.text.extract import _ocr_language, from_string
from app.detectors.text.segment import segment
from scripts.video_appearance_native_v108 import read, require
from scripts.video_appearance_native_v108 import write as save
from scripts.writing_blind_rewrite_probe import text_hash
from scripts.writing_focus_v3 import BASELINE_ASSETS, digest
from scripts.writing_paired_email_revision import SECONDARY, encoded, native_contexts, native_scores

SCHEMA = "chipmunk.writing-agreement-elife-controls/1"
ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "docs/evaluation-panels/text-elife-editorial-controls-v301-complete.json"
ARMS = ("current", "second", "agreement")
METHOD = "Confirmation-role eLife v301 inputs, previously unscored. " \
         "Select four of the retained five paragraphs around the target, keeping both adjacent paragraphs. " \
         "Measure exact current and second native contexts at the already fixed .5 point. Agreement keeps current " \
         "responses when the same input is positive in the second check. Human cases only; report original, " \
         "constructed copyedit and paragraph splice separately."


def inputs(source):
    published = read(PRIOR)
    raw = source / "private-controls.jsonl"
    require(digest(raw) == published["private_controls_sha256"], "Private control archive differs")
    controls = [json.loads(line) for line in raw.read_bytes().splitlines()]
    works = {row["group"]: row for row in published["works"]
             if row["status"] == "admitted" and row["role"] == "confirmation"}
    rows, texts, unavailable = [], {}, defaultdict(int)
    for row in controls:
        if row["role"] != "confirmation":
            continue
        require(row["article_group"] in works, "Control source is not in the confirmed source set")
        if row["status"] != "constructed":
            unavailable[row["arm"]] += 1
            continue
        text = row["text"]
        ranges = row["paragraph_codepoints"]
        work = works[row["article_group"]]
        require(text_hash(text) == row["text_sha256"] and row["source_sha256"] == work["text_sha256"]
                and row["source_target"] == work["edit_target_codepoints"] and row["ai_spans"] == []
                and work["license_url"] == "https://creativecommons.org/licenses/by/4.0/", "Unexpected source or edit receipt")
        target = work["paragraph_codepoints"].index(row["source_target"])
        start = 0 if target < 3 else 1
        retained = ranges[start:start + 4]
        selected = text[retained[0][0]:retained[-1][1]]
        doc = from_string(selected)
        if (doc.text != selected or len(doc.paragraphs) != 4 or not 68 <= len(selected.split()) <= 420
                or _ocr_language(selected).get("code") != "en"):
            unavailable[row["arm"]] += 1
            continue
        geometry = []
        for nearby in native_contexts(segment(doc)):
            entry, value = encoded(nearby)
            geometry.append(entry)
            texts[entry["sha256"]] = value
        rows.append({"case": row["case"], "group": row["article_group"], "arm": row["arm"],
                     "text_sha256": text_hash(selected), "full_input_sha256": row["text_sha256"],
                     "source_paragraphs": retained, "windows": geometry})
    require(len(works) == 7 and all(any(row["group"] == group and row["arm"] == "original" for row in rows) for group in works),
            "One or more required eLife original controls is unscored")
    return rows, texts, dict(unavailable)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "current", "second", "report"))
    parser.add_argument("run", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Run with local model access only")
    require(all(digest(args.current / name) == value for name, value in BASELINE_ASSETS.items())
            and all(digest(args.second / name) == value for name, value in SECONDARY.items()), "Classifier files differ")
    cases, texts, unavailable = inputs(args.source)
    lock = {"schema": SCHEMA, "method": METHOD, "code_sha256": digest(Path(__file__)),
            "prior_receipt_sha256": digest(PRIOR), "prior_controls_sha256": digest(args.source / "private-controls.jsonl"),
            "checkpoints": {"current": BASELINE_ASSETS, "second": SECONDARY}, "scope_unavailable": unavailable,
            "inputs": len(texts), "cases": len(cases)}
    freeze = args.run / "freeze.json"
    if args.stage == "freeze":
        save(freeze, lock)
        print(len(cases), len(texts), unavailable, flush=True)
        return
    require(read(freeze) == lock, "Publisher control freeze differs")
    pin = digest(freeze)
    if args.stage != "report":
        import torch

        torch.set_num_threads(1)
        start = monotonic()
        arm = args.stage
        scores, forwards = native_scores(getattr(args, arm), texts, second=arm == "second")
        save(args.run / f"{arm}.json", {"freeze_sha256": pin, "responses": scores,
                                        "seconds": round(monotonic() - start, 3), "forwards": forwards})
        return
    paths = {key: args.run / f"{key}.json" for key in ("current", "second")}
    packets = {key: read(path) for key, path in paths.items()}
    require(all(packet["freeze_sha256"] == pin and set(packet["responses"]) == set(texts) for packet in packets.values()),
            "One response set does not match the source freeze")
    for row in cases:
        answers = {arm: [i for i, item in enumerate(row["windows"]) if packet["responses"][item["sha256"]]["score"] >= .5][:6]
                   for arm, packet in packets.items()}
        answers["agreement"] = [position for position in answers["current"] if position in answers["second"]]
        row["marked"] = answers
    save(args.run / "private-readout.json", cases)
    outcomes = {}
    for scope in ("all", "original", "copyedit", "human_paragraph_splice"):
        subset = [case for case in cases if scope in ("all", case["arm"])]
        outcomes[scope] = {"human_files": len(subset), "human_groups": len({case["group"] for case in subset}),
                           "arms": {arm: {"flagged_files": sum(bool(case["marked"][arm]) for case in subset),
                                          "flagged_groups": len({case["group"] for case in subset if case["marked"][arm]})}
                                    for arm in ARMS}}
    save(args.run / "aggregate.json", {**lock, "freeze_sha256": pin, "outcomes": outcomes,
                                      "windows": sum(len(case["windows"]) for case in cases),
                                      "readout_sha256": digest(args.run / "private-readout.json"),
                                      "score_receipts": {key: {k: packet[k] for k in ("seconds", "forwards")}
                                                         | {"sha256": digest(paths[key])} for key, packet in packets.items()}})


if __name__ == "__main__":
    main()
