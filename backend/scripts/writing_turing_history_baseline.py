# SPDX-License-Identifier: Apache-2.0
"""Freeze and replay the local passage check on genuine historical editorial pairs."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from app.bench.writing_entropy_prominence import compact_range
from scripts.verify_writing_entropy_prominence import matching
from scripts.writing_entropy_prominence import existing
from scripts.writing_focus_v3 import BASELINE_ASSETS, BASELINE_REVISION
from scripts.writing_multi_passage_panel import score
from scripts.writing_transition_deviation import file_hash, read, save, text_hash
from scripts.writing_turing_editorial_history import CHAPTER, DIGEST, LICENSE, REVISION, SETTINGS, SHA, SOURCE
from scripts.writing_turing_editorial_history import SCHEMA as SOURCE_SCHEMA

SCHEMA = "chipmunk.writing-turing-history-baseline/1"
PIPELINE = Path(__file__).resolve().parents[1] / "app/evidence"
FILENAMES = ("document_passage_intervals.py", "passage_interval_selection.py", "report_passage_calibration.py")


def freeze(source, publisher, output):
    roster, upstream = read(source), read(publisher)
    if (roster["schema"] != SOURCE_SCHEMA or roster["plan_sha256"] != file_hash(publisher)
            or upstream["revision"] != REVISION or upstream["settings"] != SETTINGS
            or len(roster["works"]) < SETTINGS["min_works"]):
        raise ValueError("Use the qualified publisher roster and prospective intake settings")
    save(output, {"schema": SCHEMA, "source_sha256": file_hash(source), "publisher": upstream,
                  "model_revision": BASELINE_REVISION, "model_assets": BASELINE_ASSETS,
                  "operating_mode": {"document_baseline": True, "report_review": True, "device": "cpu"},
                  "decision_code_sha256": {name: file_hash(PIPELINE / name) for name in FILENAMES},
                  "runner_sha256": file_hash(Path(__file__))})


def mapped(row, focus):
    if row["sentences"][focus["sentence"]][:2] != focus["focus"]:
        raise ValueError("The measured sentence differs from the exact historical edit location")
    regions, assessed = existing(row, row)
    return {"assessed": assessed, "flagged": bool(regions),
            "selected": [compact_range(row["sentences"], interval) for interval in regions],
            "focus": compact_range(row["sentences"], (focus["sentence"], focus["sentence"] + 1))}


def summarize(works):
    totals = Counter()
    for work in works:
        before, after = (mapped(row, focus) for row, focus in zip(work["variants"], work["witnesses"], strict=True))
        if "text" in work or any("text" in row for row in work["variants"]):
            raise ValueError("The portable receipt must not include publisher prose")
        totals.update(works=1, earlier_assessed=before["assessed"], later_assessed=after["assessed"],
                      earlier_alerts=before["flagged"], later_alerts=after["flagged"],
                      either_alert=before["flagged"] or after["flagged"],
                      earlier_focus_alerts=matching(before["selected"], [before["focus"]]),
                      later_focus_alerts=matching(after["selected"], [after["focus"]]))
        if before["assessed"] and after["assessed"]:
            totals.update({f"{int(before['flagged'])}_to_{int(after['flagged'])}": 1})
    return dict(totals)


def validate_works(works):
    from datetime import datetime

    for key, limit in (("author_group", SETTINGS["max_per_author"]), ("chapter", SETTINGS["max_per_chapter"]), ("path", 1)):
        if max(Counter(work[key] for work in works).values(), default=0) > limit:
            raise ValueError("The source grouping differs from the preregistered coverage")
    for work in works:
        if (not CHAPTER.fullmatch(work["path"]) or work["document"] != text_hash(work["path"]) or work["license"] != LICENSE
                or not all(SHA.fullmatch(work[key]) for key in ("before_commit", "after_commit"))
                or not all(DIGEST.fullmatch(digest) for digest in work["files_sha256"] + work["license_sha256"])
                or not datetime.fromisoformat(work["before_date"]) < datetime.fromisoformat(work["committed"])
                < datetime.fromisoformat(SETTINGS["source_cutoff"])):
            raise ValueError("The historical source or license witness is invalid")
        if [row["version"] for row in work["variants"]] != ["before", "after"]:
            raise ValueError("Each editorial pair must contain the original and its chronological revision")


def evaluate(freeze, source, snapshot, result):
    sealed, roster = read(freeze), read(source)
    if (sealed["source_sha256"] != file_hash(source) or sealed["runner_sha256"] != file_hash(Path(__file__))
            or sealed["decision_code_sha256"] != {name: file_hash(PIPELINE / name) for name in FILENAMES}):
        raise ValueError("The scoring code or input plan differs from the freeze")
    validate_works(roster["works"])
    scratch = result.with_name("baseline-readings.json.gz")
    names = sorted({work["author_group"] for work in roster["works"]})
    aliases = {name: f"A{index:03d}" for index, name in enumerate(names, 1)}
    works = []
    for work in roster["works"]:
        public = dict(work, author_group=aliases[work["author_group"]])
        public.pop("editorial_group")
        works.append(public)
    score({"works": works}, snapshot, scratch)
    measured = read(scratch)
    if measured["model_revision"] != sealed["model_revision"] or measured["model_assets"] != sealed["model_assets"]:
        raise ValueError("The detector weights differ from the fixed baseline")
    works = measured["works"]
    save(result, {"schema": SCHEMA, "frozen": sealed, "works": works, "summary": summarize(works)})
    review(result)


def review(path):
    receipt = read(path)
    if (receipt["schema"] != SCHEMA or receipt["frozen"]["publisher"]["publisher"] != SOURCE
            or receipt["frozen"]["publisher"]["revision"] != REVISION):
        raise ValueError("Receipt belongs to a different historical source")
    works = receipt["works"]
    validate_works(works)
    if any(not re.fullmatch(r"A\d{3}", work["author_group"]) or "editorial_group" in work for work in works):
        raise ValueError("A public contributor group contains intake identifiers")
    results = summarize(works)
    if results != receipt["summary"]:
        raise ValueError("The historical outcomes disagree with native scores and positions")
    print(results, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for command, fields in (("freeze", ("source", "publisher", "output")),
                            ("evaluate", ("freeze", "source", "snapshot", "result")), ("review", ("path",))):
        action = modes.add_parser(command)
        for field in fields:
            action.add_argument(field, type=Path)
    args = vars(parser.parse_args())
    {"freeze": freeze, "evaluate": evaluate, "review": review}[args.pop("mode")](**args)
