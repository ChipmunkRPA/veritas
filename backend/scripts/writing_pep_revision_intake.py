"""Freeze and replay pre-2020 Python PEP editorial revision pairs."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from app.detectors.text.extract import from_string, text_hash
from app.detectors.text.segment import segment
from scripts.text_elife_source_intake_v272 import overlap
from scripts.text_short_insert_intake_v203 import fingerprints
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_blind_rewrite_probe import distance, valid_sentence, words
from scripts.writing_focus_v3 import digest
from scripts.writing_journal_body_holdout import DATA, HISTORY, historical_texts

SCHEMA = "chipmunk.writing-pep-revision-intake/1"
REPOSITORY = "https://github.com/python/peps"
ANCHOR = "475d6cdf8caf61cfe02ec527f84d9489566349e6"
FROM = 1420070400  # 2015-01-01 UTC
UNTIL = 1577836800  # 2020-01-01 UTC
PATH = re.compile(r"pep-(\d{4})\.(txt|rst)")
HEADING = re.compile(r"(?m)^Copyright[ \t]*\n[-=~^*+]{3,}[ \t]*\n")
LICENSE = re.compile(
    r"(?:this (?:document|pep) (?:has been placed|is placed|is) in the public domain"
    r"|creative commons zero v?1\.0|cc0-1\.0-universal)", re.IGNORECASE
)
POOL = 200
COMMITS_PER_WORK = 12
MAX_GROUPS = 24
MAX_FILE_BYTES = 400_000
EXPOSURE = HISTORY | {"wellcome-gates": DATA / "writing-journal-body-transfer-2026-11-11/originals.json"}
PLAN = {
    "schema": SCHEMA, "repository": REPOSITORY, "anchor": ANCHOR, "start_unix": FROM,
    "end_exclusive_unix": UNTIL, "pool": POOL, "commits_per_work": COMMITS_PER_WORK, "cap": MAX_GROUPS,
    "sampling": "Rank historical PEP filenames and one-parent dated commits by SHA-256, select independently of content.",
    "license": "Both exact versions must explicitly place their source text in the public domain or CC0 in Copyright.",
    "prose": "Five consecutive native sentences from adjacent plain body blocks only. Require material change to exactly "
             "one 12–40-word center, equal author header, same flanks and unique matching neighborhood.",
    "separation": "Use one pair per PEP, disallow repeated documented authors and revision commit authors, and "
                  "compare twelve-token fragments against nine earlier project packets and other admitted works.",
}


def git(repo, *args, limit=5_000_000):
    result = subprocess.run(["git", f"--git-dir={repo}", *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            check=True, timeout=12)
    require(len(result.stdout) <= limit, "Bounded local Git result exceeded")
    return result.stdout.decode("utf-8", errors="strict")


def catalog(repo):
    log = git(repo, "log", ANCHOR, "--since=2015-01-01T00:00:00Z", "--no-merges",
              "--format=%x1e%H%x1f%P%x1f%at%x1f%ct%x1f%an%x1f%s", "--name-only", "--",
              "pep-*.txt", "pep-*.rst")
    entries = defaultdict(list)
    for record in log.split("\x1e")[1:]:
        header, *paths = record.strip().splitlines()
        commit, parent, author_time, committer_time, editor, subject = header.split("\x1f", 5)
        if (not re.fullmatch(r"[0-9a-f]{40}", parent)
                or not all(FROM <= int(x) < UNTIL for x in (author_time, committer_time))
                or re.match(r"(?i)^(?:revert|merge|auto)", subject)
                or re.search(r"(?i)\[bot\]|^(?:github|pep bot)$", editor)):
            continue
        for path in paths:
            found = PATH.fullmatch(path)
            if found and int(found[1]) >= 100:
                entries[found[1]].append({"path": path, "after": commit, "before": parent,
                                         "after_authored": int(author_time), "after_committed": int(committer_time),
                                         "editor": " ".join(editor.casefold().split())})
    return {number: sorted(entries[number], key=lambda row: text_hash(row["after"]))[:COMMITS_PER_WORK]
            for number in sorted(entries, key=text_hash)[:POOL]}


def freeze(repo):
    require(git(repo, "rev-parse", ANCHOR).strip() == ANCHOR, "Official fixed Git anchor missing")
    return {"plan": PLAN, "code_sha256": digest(Path(__file__)),
            "history_sha256": {key: digest(path) for key, path in EXPOSURE.items()}, "work_roster": catalog(repo)}


def authors(text):
    header = re.split(r"\n[ \t]*\n", text, maxsplit=1)[0]
    record = re.search(r"(?m)^Author:\s*(.*(?:\n[ \t]+[^\n]+)*)", header)
    if not record:
        return []
    parts = re.split(r",|;|\s+and\s+", re.sub(r"<[^>]*>|[\w.+-]+@[\w.-]+", "", record[1]).replace("\n", " "))
    return sorted({" ".join(part.split()).casefold() for part in parts if len(part.split()) >= 2})


def plain(raw):
    if not re.match(r"^[A-Za-z]", raw) or re.search(r"(?m)^\s|^[-+*]\s|^\d+[.)]\s|^[-=_+*~^:#]{3,}[ \t]*$", raw):
        return False
    if re.search(r"`|\[|\]|\{\{|\}\}|https?://|::|_\b|\*|(?m:^[A-Z][\w -]+:)", raw):
        return False
    return raw.rstrip()[-1:] in ".!?" and len(words(raw)) >= 12


def sentence_runs(text, stop):
    header = re.search(r"\n[ \t]*\n", text)
    if not header:
        return []
    start, runs, run = header.end(), [], []
    for divider in list(re.finditer(r"\n[ \t]*\n", text, re.MULTILINE)) + [None]:
        end = divider.start() if divider else len(text)
        if end < start:
            continue
        if end > stop:
            break
        raw = text[start:end].strip("\n")
        if plain(raw):
            doc = from_string(raw)
            for item in segment(doc):
                if item.text[-1:] not in ".!?":
                    run = []
                    break
                run.append({"text": item.text, "block": [start, end],
                            "block_sha256": text_hash(raw), "normalized_span": [item.start, item.end]})
        elif run:
            runs.append(run)
            run = []
        start = divider.end() if divider else len(text)
    if run:
        runs.append(run)
    return runs


def windows(text, stop):
    result = defaultdict(list)
    for run in sentence_runs(text, stop):
        for index in range(2, len(run) - 2):
            items = run[index - 2:index + 3]
            bodies = [item["text"] for item in items]
            if any(not 8 <= len(words(value)) <= 45 for value in bodies):
                continue
            if any(not 12 <= len(words(bodies[n])) <= 40 for n in (1, 2, 3)):
                continue
            value = " ".join(bodies)
            units = segment(from_string(value))
            if [unit.text for unit in units] != bodies or not 68 <= len(words(value)) <= 180:
                continue
            focus = units[2]
            key = tuple(bodies[:2] + bodies[3:])
            result[key].append({"text": value, "focus": [focus.start, focus.end], "focus_text": focus.text,
                                "witness": items})
    return result


def exact_version(repo, commit, path):
    value = git(repo, "show", f"{commit}:{path}", limit=MAX_FILE_BYTES)
    header = HEADING.search(value)
    if not header:
        return None
    license_section = value[header.end():].split("\n\n", maxsplit=2)
    first = next((block.strip() for block in license_section if block.strip()), "")
    if not LICENSE.search(first) or re.search(r"all rights reserved|non.commercial|except ", first, re.I):
        return None
    people = authors(value)
    return (value, header.start(), people) if people else None


def pair(repo, row):
    try:
        old = exact_version(repo, row["before"], row["path"])
        new = exact_version(repo, row["after"], row["path"])
    except (subprocess.CalledProcessError, UnicodeError, ValueError):
        return "unavailable_blob", None
    if not old or not new:
        return "missing_explicit_terms_or_authors", None
    if old[2] != new[2]:
        return "author_header_changed", None
    a, b = windows(old[0], old[1]), windows(new[0], new[1])
    options = []
    for key in a.keys() & b.keys():
        if len(a[key]) != 1 or len(b[key]) != 1:
            continue
        first, second = a[key][0], b[key][0]
        old_focus, new_focus = first["focus_text"], second["focus_text"]
        if distance(old_focus, new_focus) < .15 or not valid_sentence({"text": new_focus, "eos": True}, old_focus, generated=False):
            continue
        options.append((first, second))
    if not options:
        return "no_material_center", None
    first, second = min(options, key=lambda pair: text_hash(pair[0]["text"] + "\0" + pair[1]["text"]))
    parent_time = git(repo, "show", "-s", "--format=%ct", row["before"]).strip()
    if not parent_time.isdigit() or int(parent_time) > row["after_committed"]:
        return "nonchronological_parent", None
    return "qualified", {
        "publisher": "python-peps", "license": "Public-domain or CC0 statement in both exact Git blobs",
        "attribution": old[2], "before": first, "after": second,
        "source": row | {"before_committed": int(parent_time),
                         "before_blob_sha256": text_hash(old[0]), "after_blob_sha256": text_hash(new[0])},
    }


def qualify(repo, plan):
    found, dispositions = {}, Counter()
    for number, candidates in sorted(plan["work_roster"].items(), key=lambda item: text_hash(item[0])):
        for row in candidates:
            status, record = pair(repo, row)
            dispositions[status] += 1
            if record is not None:
                found[number] = record | {"group": f"pep-{number}"}
                break
    fragments = {number: fingerprints(record["before"]["text"]) | fingerprints(record["after"]["text"])
                 for number, record in found.items()}
    history, external, internal = overlap(
        [(key, digest(path), None, historical_texts(path)) for key, path in EXPOSURE.items()], fragments
    )
    selected, names, editors = [], set(), set()
    for number, record in found.items():
        editor = record["source"]["editor"]
        if number in external or number in internal:
            dispositions["text_overlap_rejected"] += 1
        elif set(record["attribution"]) & names or editor in editors:
            dispositions["identity_overlap_rejected"] += 1
        elif len(selected) == MAX_GROUPS:
            dispositions["qualified_over_cap"] += 1
        else:
            names.update(record["attribution"])
            editors.add(editor)
            selected.append(record)
    return selected, {"schema": SCHEMA, "work_pool": len(plan["work_roster"]), "counts": dict(dispositions),
                      "mechanical_groups": len(found), "qualified_groups": len(selected), "history": history}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "qualify", "verify"))
    cli.add_argument("run", type=Path)
    cli.add_argument("repo", type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    lock = args.run / "source-freeze.json"
    frozen = freeze(args.repo)
    if args.phase == "freeze":
        write(lock, frozen)
        return
    require(read(lock) == frozen, "Frozen source method or catalog differs")
    selected, summary = qualify(args.repo, frozen)
    originals, receipt = args.run / "originals.json", args.run / "source-receipt.json"
    if args.phase == "qualify":
        write(originals, selected)
        write(receipt, summary | {"source_freeze_sha256": digest(lock), "originals_sha256": digest(originals)})
    else:
        require(read(originals) == selected and read(receipt) == summary | {
            "source_freeze_sha256": digest(lock), "originals_sha256": digest(originals)
        }, "Independent source replay differs")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
