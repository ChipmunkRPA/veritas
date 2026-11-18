# SPDX-License-Identifier: Apache-2.0
"""Build exact paired editorial controls from The Turing Way's historic Git snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from scripts.writing_blind_rewrite_probe import distance, words
from scripts.writing_economic_paraphrase_holdout import find_texts, fragments
from scripts.writing_multi_passage_panel import native
from scripts.writing_transition_deviation import file_hash, read, save, text_hash

SCHEMA = "chipmunk.writing-turing-editorial-history/1"
SOURCE = "https://github.com/the-turing-way/the-turing-way"
REVISION = "e2394b4940f88c8e571dba6c496d0f7ea7a52d2c"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
WIDTH = 12
CUTOFF = datetime.fromisoformat("2021-01-01T00:00:00+00:00")
CHAPTER = re.compile(r"book/(content|website)/([^/]+)/(.+\.md)")
SHA = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"[0-9a-f]{64}")
MARKUP = re.compile(r"(?m)^\s*(?:[#>+<!|]|[-*]\s|\d+\.\s|\{[\{%])|`|https?:|www\.|@[\w.-]+|\[|\]|[©“”«»\"]")
BORROWED = re.compile(r"\b(?:adapted from|reproduced from|reproduced with permission|copyright|all rights reserved)\b", re.I)
SETTINGS = {"sentences": WIDTH, "source_cutoff": CUTOFF.isoformat(), "max_per_author": 2, "max_per_chapter": 2,
            "max_works": 24, "min_works": 8, "range_matching": {"iou": .30, "recall": .50},
            "source_body": "eligible Markdown prose paragraphs in chapter reading order"}


def git(repo, *args, missing=False):
    result = subprocess.run(["git", "--no-replace-objects", "-C", str(repo), *args],
                            env=os.environ | {"GIT_NO_LAZY_FETCH": "1"}, capture_output=True, check=False)
    if result.returncode == 0:
        return result.stdout
    if missing:
        return b""
    raise ValueError(f"Cannot read the frozen publisher archive ({args[0]})")


def ledger(repo):
    pattern = "%H%x09%P%x09%aI%x09%cI%x09%an%x09%ae"
    raw = git(repo, "log", REVISION, f"--before={CUTOFF.isoformat()}", "--no-merges", "--no-renames", "--diff-filter=M",
              f"--format={pattern}", "--name-only", "--", "book/content", "book/website").decode("utf-8")
    commit = None
    for line in raw.splitlines():
        if "\t" in line:
            commit = None
            new, previous, authored, committed, name, email = line.split("\t")
            if not SHA.fullmatch(new) or not SHA.fullmatch(previous):
                raise ValueError("The editorial commit must have an unambiguous single parent")
            if not name.strip() or not email or re.search(r"\bbot\b|\[bot\]|action|dependabot", f"{name} {email}", re.I):
                continue
            if max(datetime.fromisoformat(authored), datetime.fromisoformat(committed)) >= CUTOFF:
                continue
            commit = {"before_commit": previous, "after_commit": new, "author_group": text_hash(name.casefold().strip()),
                      "authored": authored, "committed": committed}
        elif commit and (match := CHAPTER.fullmatch(line)):
            family = match[2].casefold()
            if family not in {"afterword", "foreword", "community-handbook"}:
                chapter = match[3].split("/", 1)[0].removesuffix(".md") if match[1] == "website" else family
                yield commit | {"path": line, "chapter": chapter.casefold()}


@lru_cache(maxsize=512)
def snapshot(repo, revision, path):
    return git(repo, "show", f"{revision}:{path}", missing=True)


def license_digest(repo, revision):
    raw = snapshot(repo, revision, "LICENSE.md")
    terms = raw.decode("utf-8").casefold()
    if ("all documentation and chapter materials" in terms and "cc by 4.0" in terms
            and "attribution" in terms and "commercially" in terms):
        return hashlib.sha256(raw).hexdigest()
    return None


def _sections(raw):
    from scripts.writing_agreement_revision_sources import prose_markdown

    value = raw.decode("utf-8")
    if re.search(r"\bcopyright\b|©|\bthird.party license\b", value, re.I):
        return []
    value = re.sub(r"(?ms)^\s*~{3,}[^\n]*\n.*?^\s*~{3,}\s*$", "", value)
    paragraphs = [block for block in prose_markdown(value)
                  if block[0].isalpha() and block[-1] in ".?!" and not MARKUP.search(block) and not BORROWED.search(block)]
    return ["\n\n".join(paragraphs)] if paragraphs else []


@lru_cache(maxsize=512)
def _windows(raw):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    windows = defaultdict(list)
    for number, prose in enumerate(_sections(raw)):
        doc = from_string(prose)
        units = segment(doc)
        for start in range(len(units) - WIDTH + 1):
            sentence_set = units[start:start + WIDTH]
            if not 130 <= sum(unit.n_words for unit in sentence_set) <= 550:
                continue
            lo, hi = sentence_set[0].start, sentence_set[-1].end
            for position in range(2, WIDTH - 2):
                focus = sentence_set[position]
                if not 10 <= focus.n_words <= 50:
                    continue
                anchors = (position, *(unit.text for index, unit in enumerate(sentence_set) if index != position))
                windows[anchors].append({"text": doc.text[lo:hi], "section": number, "section_sha256": text_hash(doc.text),
                                         "section_range": [lo, hi], "focus": [focus.start - lo, focus.end - lo],
                                         "sentence": position, "focus_text": focus.text})
    return windows


def matches(before, after):
    left, right = _windows(before), _windows(after)
    for key, values in right.items():
        if len(values) != 1 or len(left.get(key, ())) != 1:
            continue
        old, new = left[key][0], values[0]
        before_words, after_words = words(old["focus_text"]), words(new["focus_text"])
        change = distance(old["focus_text"], new["focus_text"])
        if (.60 <= len(after_words) / max(1, len(before_words)) <= 1.55 and change <= .70
                and round(change * max(len(before_words), len(after_words))) >= 2):
            yield old, new


def prepare(repo, output, previous):
    plan, roster = output / "lock.json.gz", output / "source.json.gz"
    candidates = sorted(ledger(repo), key=lambda row: (row["committed"][:4] >= "2020",
                        text_hash(SCHEMA + row["after_commit"] + row["path"])))
    save(plan, {"schema": SCHEMA, "publisher": SOURCE, "revision": REVISION, "settings": SETTINGS,
                "historical_inputs": {path.name: file_hash(path) for path in previous}, "candidate_count": len(candidates),
                "source_plan_sha256": text_hash(json.dumps(candidates, sort_keys=True)),
                "implementation_sha256": file_hash(Path(__file__))})
    seen = {key for path in previous for text in set(find_texts(read(path))) for key in fragments(text)}
    by_author, by_chapter, exclusions = Counter(), Counter(), Counter()
    result, used_paths = [], set()
    for row in candidates:
        author, chapter, path = (row[key] for key in ("author_group", "chapter", "path"))
        if by_author[author] >= SETTINGS["max_per_author"] or by_chapter[chapter] >= SETTINGS["max_per_chapter"] or path in used_paths:
            exclusions["source_quota"] += 1
            continue
        old_rev, new_rev = (row[key] for key in ("before_commit", "after_commit"))
        before, after = (snapshot(repo, rev, path) for rev in (old_rev, new_rev))
        if not before or not after:
            exclusions["unavailable_source"] += 1
            continue
        for old, new in matches(before, after):
            signatures = fragments(old["text"]) | fragments(new["text"])
            if seen & signatures:
                continue
            variants = [native(version["text"], ()) for version in (old, new)]
            if any(value is None or len(value["sentences"]) != WIDTH for value in variants):
                continue
            if any(value["text"] != source["text"] or value["sentences"][source["sentence"]][:2] != source["focus"]
                   for source, value in zip((old, new), variants, strict=True)):
                continue
            old_date = git(repo, "log", "-1", "--format=%cI", old_rev).decode().strip()
            if git(repo, "log", "-1", "--format=%P", new_rev).decode().strip() != old_rev:
                raise ValueError("A source comparison must use the exact parent commit")
            if datetime.fromisoformat(old_date) >= datetime.fromisoformat(row["committed"]):
                exclusions["unordered_date"] += 1
                break
            terms = [license_digest(repo, rev) for rev in (old_rev, new_rev)]
            if not all(terms):
                exclusions["terms"] += 1
                break
            found = row | {"document": text_hash(path), "editorial_group": author, "split": "test", "before_date": old_date,
                           "license": LICENSE, "license_sha256": terms, "files_sha256": [hashlib.sha256(data).hexdigest()
                                                                                         for data in (before, after)],
                           "witnesses": [{key: val for key, val in row.items() if key not in {"text", "focus_text"}}
                                         for row in (old, new)],
                           "variants": [item | {"version": name} for item, name in zip(variants, ("before", "after"), strict=True)]}
            result.append(found)
            by_author[author] += 1
            by_chapter[chapter] += 1
            used_paths.add(path)
            seen.update(signatures)
            break
        else:
            exclusions["native_alignment_or_overlap"] += 1
        if len(result) >= SETTINGS["max_works"]:
            break
    save(roster, {"schema": SCHEMA, "plan_sha256": file_hash(plan), "works": result, "excluded": dict(exclusions)})
    print(f"Qualified {len(result)} historical editing pairs in {len(by_chapter)} chapters from {len(by_author)} attributed contributors",
          dict(exclusions), flush=True)
    if len(result) < SETTINGS["min_works"]:
        raise ValueError("Too few source-qualified editing pairs for comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("previous", nargs="+", type=Path)
    prepare(**vars(parser.parse_args()))
