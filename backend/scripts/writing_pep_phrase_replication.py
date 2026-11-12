"""Replicate the frozen eight-word check on author-separated Python PEP prose."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from app.detectors.text.extract import text_hash
from scripts import writing_archive_author_intake as history
from scripts import writing_archive_prospective_intake as prose
from scripts import writing_pep_revision_intake as pep
from scripts import writing_phrase_context_stability as trial
from scripts.text_elife_source_intake_v272 import overlap
from scripts.text_short_insert_intake_v203 import fingerprints
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_focus_v3 import digest

SCHEMA = "chipmunk.writing-pep-phrase-replication/1"
BASE = trial.BASE
ARCHIVE = BASE / "writing-pep-center-contribution-2026-11-11/python-peps.git"
PRIOR = BASE / "writing-phrase-context-stability-2026-11-12-sentence-fragments"
PUBLIC_PRIOR = trial.ROOT / "docs/evaluation-panels/writing-phrase-context-stability-2026-11-12.json"
GROUP_LIMIT = 20
MINIMUM_GROUPS = 12
METHOD = {
    "source": "Exact Python PEP files at the retained pre-2020 archive anchor. Require an explicit public-domain or CC0 notice "
              "in each file, a named credited author, and two disjoint five-sentence plain-prose excerpts with valid eight-word geometry.",
    "roster": "Hash order by PEP number. Exclude earlier PEPs and exact 12-word overlaps with the retained inventories. "
              "Exclude repeated recorded authors across the previous phrase study, earlier PEPs and the held role. "
              "Qualify source only before any generator or classifier call. Each ordinary control is a same-work substitution.",
    "points": "Reuse all three archived Kubernetes operating points exactly, including the original 0.50 current-window point. "
              "No held recalibration. Recompute the older points from their already committed response packet in separate audit.",
    "advancement": trial.METHOD["advance"],
}


def previous_rosters():
    return {path.parent.name: path for path in sorted(BASE.glob("writing-pep-*/originals.json"))}


def held_fragments(row):
    return fingerprints(row["before"]["text"]) | fingerprints(row["after"]["text"])


def qualify():
    previous = read(PRIOR / "sources.json")
    earlier = previous_rosters()
    used_numbers = set()
    names = {history.name_key(name) for role in ("calibration", "holdout") for row in previous[role]
             for name in [*row["byline"], *row["source"].get("editor", [])]}
    packets = [(name, digest(path), None, history.historical_texts(path)) for name, path in history.HISTORY.items()]
    for label, path in earlier.items():
        rows = read(path)
        require(isinstance(rows, list), "Retained PEP roster structure differs")
        for row in rows:
            source = row["source"]["path"]
            match = pep.PATH.fullmatch(source)
            require(match, "Earlier PEP path is invalid")
            used_numbers.add(match[1])
            names.update(history.name_key(name) for name in row["attribution"])
        packets.append((label, digest(path), None, [row[side]["text"] for row in rows for side in ("before", "after")]))
    used, later, _ = trial.paths()
    packets.extend([
        ("previous_phrase_works", digest(PRIOR / "sources.json"), None,
         [row[side]["text"] for role in ("calibration", "holdout") for row in previous[role] for side in ("before", "after")]),
        ("previous_blog_works", digest(used), None, [row[side]["text"] for row in read(used) for side in ("before", "after")]),
        ("previous_blog_bodies", digest(later), None, [block["text"] for row in read(later) for block in row["paragraphs"]]),
    ])

    paths = [path for path in pep.git(ARCHIVE, "ls-tree", "--name-only", pep.ANCHOR).splitlines()
             if (match := pep.PATH.fullmatch(path)) and int(match[1]) >= 100]
    paths.sort(key=lambda path: text_hash("pep-phrase-replication:" + path[4:8]))
    candidates, dispositions = [], Counter()
    for path in paths:
        number = path[4:8]
        if number in used_numbers:
            dispositions["earlier_pep"] += 1
            continue
        version = pep.exact_version(ARCHIVE, pep.ANCHOR, path)
        if version is None:
            dispositions["missing_explicit_terms_or_author"] += 1
            continue
        raw, copyright_start, credited = version
        authors = {history.name_key(name) for name in credited}
        if not authors or authors & names:
            dispositions["earlier_recorded_author"] += 1
            continue
        header = re.search(r"\n[ \t]*\n", raw)
        if not header:
            dispositions["missing_header_boundary"] += 1
            continue
        excerpt = [item for item in prose.excerpts(raw[:copyright_start], {"span": [0, header.end() - 1]})
                   if trial.phrase_ranges(item["focus_text"])]
        pair = prose.chosen_pair(excerpt)
        if pair is None:
            dispositions["no_disjoint_phrase_pair"] += 1
            continue
        license_tail = raw[copyright_start:]
        license_heading = pep.HEADING.match(license_tail)
        require(license_heading, "Original copyright heading differs")
        notice = next(block for block in license_tail[license_heading.end():].split("\n\n") if block.strip())
        notice_start = copyright_start + license_tail.index(notice, license_heading.end())
        candidates.append({"group": f"pep-{number}", "byline": sorted(authors), "before": pair[0], "after": pair[1],
                           "source": {"path": path, "commit": pep.ANCHOR, "blob_sha256": text_hash(raw),
                                      "notice_sha256": text_hash(notice), "notice_span": [notice_start, notice_start + len(notice)]}})

    marks = {row["group"]: held_fragments(row) for row in candidates}
    inventory, historical, _ = overlap(packets, marks)
    selected, fragments = [], set()
    for row in candidates:
        group_marks, authors = marks[row["group"]], set(row["byline"])
        if row["group"] in historical or group_marks & fragments:
            dispositions["known_twelve_word_fragment"] += 1
        elif authors & names:
            dispositions["held_recorded_author_overlap"] += 1
        elif len(selected) == GROUP_LIMIT:
            dispositions["qualified_over_cap"] += 1
        else:
            selected.append(row)
            names.update(authors)
            fragments.update(group_marks)
    require(len(selected) >= MINIMUM_GROUPS, f"Only {len(selected)} author-separated PEPs passed exact source qualification")
    return {"archive_commit": pep.ANCHOR, "intake_paths": len(paths), "source_qualification": dict(dispositions),
            "prior_corpus": inventory, "holdout": selected, "calibration": []}


def prior_points():
    published = read(PUBLIC_PRIOR)
    for name in ("freeze.json", "sources.json", "fixtures.json", "calibration-responses.json"):
        require(digest(PRIOR / name) == published["commitments"][name], "Earlier frozen point packet differs from public commitment")
    points = read(PRIOR / "points.json")
    require(points["points"] == published["operating_points"]
            and points["response_sha256"] == published["commitments"]["calibration-responses.json"]
            and points["fixture_sha256"] == published["commitments"]["fixtures.json"], "Earlier operating points changed")
    return points["points"]


def freeze(run, snapshot):
    method = trial.seal(run, snapshot, "sentence_fragment")
    return method | {"replication_schema": SCHEMA, "replication_method": METHOD, "replication_sha256": digest(Path(__file__)),
                     "frozen_points": prior_points(), "prior_receipt_sha256": digest(PUBLIC_PRIOR)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "generate", "fixtures", "holdout"))
    parser.add_argument("run", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--family", choices=trial.SEEDS)
    args = parser.parse_args()
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Require offline model runtime")
    os.umask(0o077)
    run = args.run
    if args.phase == "freeze":
        sources = qualify()
        run.mkdir(exist_ok=False)
        write(run / "sources.json", sources)
        write(run / "freeze.json", freeze(run, args.snapshot))
        print(f"Froze {len(sources['holdout'])} separate PEP works and the original fixed points", flush=True)
        return
    frozen = read(run / "freeze.json")
    require(frozen == freeze(run, args.snapshot), "Replicated method, source, point, or assets differ from freeze")
    if args.phase == "generate":
        require(args.family is not None, "A frozen generator family is required")
        trial.generation(run, frozen, args.family)
    elif args.phase == "fixtures":
        fixtures = trial.build(run)
        write(run / "fixtures.json", fixtures)
        print(json.dumps(fixtures["counts"], sort_keys=True), flush=True)
    else:
        result = trial.summarize(trial.score(run, args.snapshot, "holdout"), frozen["frozen_points"])
        write(run / "result.json", result)
        print(json.dumps({key: value for key, value in result.items() if key != "cases"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
