"""Prospectively select genuine bylined technical-blog passages before scoring."""

from __future__ import annotations

import argparse
import bisect
import html
import os
import re
from collections import Counter
from pathlib import Path

from app.detectors.text.extract import from_string, text_hash
from app.detectors.text.segment import segment
from scripts import writing_archive_author_intake as history
from scripts.text_elife_source_intake_v272 import overlap
from scripts.text_short_insert_intake_v203 import fingerprints
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_blind_rewrite_probe import words
from scripts.writing_focus_v3 import digest

SCHEMA = "chipmunk.writing-archive-prospective-intake/1"
ANCHOR = "0ef096e4cb577c30739f19e19a017217e5ca4807"
MAX_GROUPS = 36
FRONT = re.compile(r"^---[^\S\n]*\n(.*?)\n---[^\S\n]*\n", re.S)
BYLINE = re.compile(r"(?im)^\s*(?:[*_]\s*){0,2}(?:guest\s+)?authors?(?:[*_]\s*){0,2}:(?:[*_]\s*){0,2}\s*([^\n]+)")
NAME = re.compile(r"[^\W\d_]+(?:[.'’-][^\W\d_]+)*\.?", re.UNICODE)
LINK = re.compile(r"(?<![!\[])\[([^\[\]\n]{1,110})\](?:\([^()\n]*(?:\([^()\n]*\)[^()\n]*)*\)|\[[^\[\]\n]*\])")
EMPHASIS = re.compile(r"(?<![\w`*_])(``?|\*{1,3}|_{1,2})(?=\S)([^\n]+?)(?<=\S)\1(?![`*_])")
FENCE = re.compile(r"(?m)^[ \t]{0,3}(`{3,}|~{3,})[^\n]*\n.*?^\s*\1[^\n]*$", re.S)
PLAN = {
    "source": history.SOURCES["kubernetes"],
    "anchor": ANCHOR,
    "selection": "Enumerate each archived English Kubernetes blog source in hash order before any inference. Require a named author or author team in its exact source. Exclude repeated author names and previously exposed fragments. At most 36 independent source groups including the fixed historic edit.",
    "geometry": "Construct native contiguous five-sentence paragraphs from genuine prose, allowing visible Markdown link labels and inline formatting with original character offsets. Use exact text and verified extraction. Both passages must have at least two eligible interior centers and each center 12–40 words. The source passages may cross adjacent prose blocks and must never overlap.",
    "matching": "Among eligible non-overlapping pairs from one article, minimize absolute center and excerpt word-count differences, then take a deterministic hash tie-break. Give the earlier location the generation role. The other passage remains a genuine unedited matching control.",
    "historic_control": "Carry forward the single verified archived same-contributor edit as an independently reported stratum.",
    "interpretation": "A recorded byline identifies the work's public credit, not the writer of every line. Matched passages are real original writing from the same work, not evidence of historical edits or real AI-assisted use.",
}


def freeze(archive, historical):
    roster = history.git(archive, "ls-tree", "-r", "--name-only", ANCHOR, "content/en/blog").splitlines()
    paths = [value for value in roster if re.fullmatch(r"content/en/blog/_posts/20(?:1[5-9])-[A-Za-z0-9_./-]+\.md", value)]
    rights = history.rights_record(archive, ANCHOR, history.SOURCES["kubernetes"])
    require(rights and paths and history.git(archive, "show", "-s", "--format=%ct", ANCHOR).strip().isdigit(), "No verified archive")
    require(int(history.git(archive, "show", "-s", "--format=%ct", ANCHOR)) < history.END, "Archive exceeds frozen date")
    return {"schema": SCHEMA, "method": PLAN, "code_sha256": digest(Path(__file__)),
            "historical_source_impl": digest(Path(history.__file__)),
            "archived_receipt_sha256": digest(historical / "source-receipt.json"),
            "archived_originals_sha256": digest(historical / "originals.json"),
            "archive_rights_sha256": rights[0],
            "past_text_sha256": {name: digest(path) for name, path in history.HISTORY.items()},
            "paths": sorted(paths, key=text_hash)}


def byline(raw, front):
    evidence = re.search(r"(?im)^authors?:[^\S\n]*(\S[^\n]*)", front[1])
    if evidence:
        line = evidence[1]
        offset = front.start(1) + evidence.start(1)
    else:
        body = raw[front.end():]
        evidence = BYLINE.search("\n".join(body.split("\n", 16)[:16]))
        if not evidence:
            return None
        line, offset = evidence[1], front.end() + evidence.start(1)
    line = LINK.sub(lambda m: m[1], line)
    line = re.sub(r"\([^()]*\)|<[^>]*>|@[A-Za-z0-9_.-]+|[*_`]", "", line.strip())
    parts = re.split(r"\s*[,;&]\s*|\s+and\s+", line)
    names = []
    for item in parts:
        parts_of_name = item.strip(" \"'[]").split()
        if not 2 <= len(parts_of_name) <= 5 or not all(NAME.fullmatch(part) for part in parts_of_name):
            break
        if any(word.casefold() in {"community", "engineer", "lead", "contributor", "senior", "director", "engineering",
                                   "product", "architect", "manager", "founder", "cloud", "technical", "team"}
               for word in parts_of_name):
            break
        names.append(history.name_key(" ".join(parts_of_name)))
    return {"names": sorted(set(names)), "span": [offset, offset + len(evidence[1])]} if names else None


def visible(raw, a, b):
    raw_block = raw[a:b]
    if (not re.match(r"^[ \t\n]*[A-Za-z`*_'“\"(\[]", raw_block) or len(raw_block) > 8_000
            or re.search(r"(?m)^[ \t]*(?:[#>|]|[-+*]\s|\d+[.)]\s|```|~~~|!\[)|^[ \t]{4,}\S", raw_block)
            or re.search(r"(?i)\{\{|\}\}|\[\^|^\s*(?:authors?|editors?|source)\s*:", raw_block)
            or history.RESTRICTED.search(raw_block)):
        return "", []
    hidden = set()
    for pattern, inner in ((LINK, 1), (EMPHASIS, 2)):
        for match in pattern.finditer(raw_block):
            lo, hi = match.span(inner)
            hidden.update(i for i in range(*match.span()) if i < lo or i >= hi)
    chars, offsets = [], []
    i = 0
    while i < len(raw_block):
        if i in hidden:
            i += 1
            continue
        entity = history.ENTITY.match(raw_block, i)
        stop = entity.end() if entity else i + 1
        value = html.unescape(entity[0]) if entity else raw_block[i]
        if len(value) != 1 or (not entity and value in "<>[`]|"):
            return "", []
        if not value.isspace() or (chars and chars[-1] != " "):
            chars.append(" " if value.isspace() else value)
            offsets.append((a + i, a + stop))
        i = stop
    if chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()
    text = "".join(chars)
    if re.search(r"(?i)https?://|\*\*|\{\{|\}\}", text):
        return "", []
    return text, offsets


def excerpts(raw, authors):
    forbidden = [match.span() for match in FENCE.finditer(raw)] + [authors["span"]]
    line_starts = [0] + [match.end() for match in re.finditer("\n", raw)]
    runs, current = [], []
    for start, stop in history.region_blocks(raw, True):
        hidden = any(start < hi and lo < stop for lo, hi in forbidden)
        rendered, positions = visible(raw, start, stop) if not hidden else ("", [])
        doc = from_string(rendered)
        native = segment(doc) if doc.text == rendered and rendered.endswith((".", "!", "?")) else []
        if not native or not all(unit.text.endswith((".", "!", "?")) for unit in native):
            if current:
                runs.append(current)
                current = []
            continue
        for unit in native:
            lo, hi = positions[unit.start][0], positions[unit.end - 1][1]
            current.append({"text": unit.text, "raw_span": [lo, hi], "raw_sha256": text_hash(raw[lo:hi]),
                            "lines": [bisect.bisect_right(line_starts, lo), bisect.bisect_right(line_starts, hi - 1)]})
    if current:
        runs.append(current)
    values = []
    for run in runs:
        for i in range(len(run) - 4):
            five = run[i:i + 5]
            texts = [item["text"] for item in five]
            sizes = [len(words(value)) for value in texts]
            if (not 12 <= sizes[2] <= 40 or sum(12 <= count <= 40 for count in sizes[1:4]) < 2
                    or not all(6 <= count <= 65 for count in sizes) or not 60 <= sum(sizes) <= 225):
                continue
            text = " ".join(texts)
            units = segment(from_string(text))
            if [unit.text for unit in units] != texts:
                continue
            middle = units[2]
            values.append({"text": text, "focus": [middle.start, middle.end], "focus_text": middle.text, "witness": five,
                           "raw_span": [five[0]["raw_span"][0], five[-1]["raw_span"][1]],
                           "words": sum(sizes), "focus_words": sizes[2]})
    return values


def chosen_pair(options):
    matches = ((a, b) for i, a in enumerate(options) for b in options[i + 1:] if a["raw_span"][1] < b["raw_span"][0])
    return min(matches, key=lambda items: (
        abs(items[0]["focus_words"] - items[1]["focus_words"]) + abs(items[0]["words"] - items[1]["words"]),
        text_hash(items[0]["text"] + "\0" + items[1]["text"])), default=None)


def qualify(archive, historical, frozen):
    raw_prior = read(historical / "originals.json")
    require(len(raw_prior) == 1 and raw_prior[0]["recorded_authorship"] == "same_recorded_contributor"
            and read(historical / "source-receipt.json")["originals_sha256"] == digest(historical / "originals.json"),
            "Source-only historical receipt differs")
    paths = {row["source"]["path"] for row in raw_prior}
    identities = {identity for row in raw_prior for identity in row["source"]["editor"] + row["byline"]}
    for path in paths:
        archived = history.git(archive, "show", f"{ANCHOR}:{path}")
        front = FRONT.match(archived)
        inferred = byline(archived, front) if front else None
        if inferred:
            identities.update(inferred["names"])
    counts, found = Counter(), []
    for path in frozen["paths"]:
        if path in paths:
            counts["historic_control_already_used"] += 1
            continue
        raw = history.git(archive, "show", f"{ANCHOR}:{path}", limit=600_000)
        front = FRONT.match(raw)
        authors = byline(raw, front) if front else None
        if not authors:
            counts["no_supported_named_byline"] += 1
            continue
        if history.RESTRICTED.search(raw):
            counts["restrictive_notice_or_quoted_material"] += 1
            continue
        pair = chosen_pair(excerpts(raw, authors))
        if pair is None:
            counts["no_disjoint_matching_passages"] += 1
            continue
        selected, control = pair
        found.append({"group": "blog-" + text_hash(path)[:16], "archive": "kubernetes", "license": "CC-BY-4.0",
                      "recorded_authorship": "same_bylined_work_matched_passages", "before": selected, "after": control,
                      "byline": authors["names"], "source": {"path": path, "genre": "blog", "commit": ANCHOR,
                      "blob_sha256": text_hash(raw), "license_sha256": frozen["archive_rights_sha256"],
                      "byline_span": authors["span"], "byline_sha256": text_hash(raw[slice(*authors["span"])])}})
    signatures = {row["group"]: fingerprints(row["before"]["text"]) | fingerprints(row["after"]["text"])
                  for row in raw_prior + found}
    packets = [(name, digest(path), None, history.historical_texts(path)) for name, path in history.HISTORY.items()]
    exposure, external, _ = overlap(packets, signatures)
    selected = raw_prior[:]
    fragments = set().union(*(signatures[row["group"]] for row in raw_prior))
    for row in found:
        group, authors = row["group"], set(row["byline"])
        if group in external:
            counts["prior_text_overlap"] += 1
        elif authors & identities:
            counts["writer_overlap"] += 1
        elif signatures[group] & fragments:
            counts["within_panel_text_overlap"] += 1
        elif len(selected) == MAX_GROUPS:
            counts["over_cap"] += 1
        else:
            counts["admitted_matching_group"] += 1
            selected.append(row)
            identities.update(authors)
            fragments.update(signatures[group])
    return selected, {"schema": SCHEMA, "history": exposure, "counts": dict(counts), "qualified_groups": len(selected),
                      "archived_edit_groups": len(raw_prior), "matched_passage_groups": len(selected) - len(raw_prior)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "qualify", "verify"))
    cli.add_argument("run", type=Path)
    cli.add_argument("historical", type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    archive, run = args.historical / history.SOURCES["kubernetes"]["archive"], args.run
    lock, originals, receipt, notice = (run / name for name in ("source-freeze.json", "originals.json",
                                                                  "source-receipt.json", "source-notice.json"))
    frozen = freeze(archive, args.historical)
    if args.phase == "freeze":
        run.mkdir(exist_ok=True)
        write(lock, frozen)
        print(f"Frozen {len(frozen['paths'])} pre-2020 byline candidate paths.", flush=True)
        return
    require(read(lock) == frozen, "Prospective archive metadata changed")
    rows, summary = qualify(archive, args.historical, frozen)
    license_record = history.rights_record(archive, ANCHOR, history.SOURCES["kubernetes"])
    notice_value = {"sha256": license_record[0], "text": license_record[1]}
    if args.phase == "qualify":
        write(originals, rows)
        write(notice, notice_value)
        write(receipt, summary | {"source_freeze_sha256": digest(lock), "originals_sha256": digest(originals),
                                  "notice_sha256": digest(notice)})
    else:
        require(read(originals) == rows and read(notice) == notice_value and read(receipt) == summary | {
            "source_freeze_sha256": digest(lock), "originals_sha256": digest(originals),
            "notice_sha256": digest(notice)}, "Prospective source replay differs")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
