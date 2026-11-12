"""Freeze, qualify and replay licensed pre-2020 source revisions with exact center spans."""

from __future__ import annotations

import argparse
import bisect
import html
import os
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from app.detectors.text.extract import from_string, text_hash
from app.detectors.text.segment import segment
from scripts.text_elife_source_intake_v272 import overlap
from scripts.text_short_insert_intake_v203 import fingerprints
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_blind_rewrite_probe import distance, valid_sentence, words
from scripts.writing_focus_v3 import digest
from scripts.writing_journal_body_holdout import historical_texts
from scripts.writing_pep_revision_intake import DATA, EXPOSURE

SCHEMA = "chipmunk.writing-archive-author-intake/1"
START = 1420070400
END = 1577836800
MAX_GROUPS = 36
SOURCES = {
    "kubernetes": {
        "remote": "https://github.com/kubernetes/website",
        "anchor": "e20b8746f2444ae9fe58f49fd0f0acbfbabbb938",
        "archive": "kubernetes-website.git",
        "rights": "LICENSE",
        "license": "CC-BY-4.0",
        "directories": ["content/en/blog", "content/en/docs"],
        "per_editor": {"blog": 16, "docs": 8},
        "maximum": {"blog": 550, "docs": 650},
    },
    "postgres": {
        "remote": "https://github.com/postgres/postgres",
        "anchor": "3f302f0ed06f69c5ebd33d4df95895323a3cbef6",
        "archive": "postgres.git",
        "rights": "COPYRIGHT",
        "license": "PostgreSQL",
        "directories": ["doc/src/sgml"],
        "per_editor": {"docs": 15},
        "maximum": {"docs": 850},
    },
}
HISTORY = EXPOSURE | {"pep": DATA / "writing-pep-prose-diff-extension-2026-11-11/originals.json"}
PLAN = {
    "scope": "Historical English blog and documentation revisions in two public project archives. One file and contributor group per admission.",
    "archive": SOURCES,
    "dates_unix": [START, END],
    "roster": "Freeze one-parent modification metadata only. Hash order per source and genre, cap each recorded editor before seeing prose.",
    "geometry": "Exact five-sentence native snippets from contiguous plain prose. Each 12–40 word center must overlap corresponding old and new changed Git lines with at least one exact unchanged sentence on each side within the snippet. At least two eligible interior centers in each file.",
    "source_offsets": "Keep original character offsets and inclusive Git line ranges for every normalized sentence. Decode listed inline markup only and collapse whitespace. Compare native segmentation before any scoring.",
    "author": "Require at least one changed after-center line attributed to the revision. Same contributor requires all before and after focus lines to have a non-boundary Git attribution with the revision's recorded name or email. Otherwise retain unchanged named Kubernetes blog authorship as a separate editorial stratum.",
    "reuse": "Both exact repository root license files must explicitly permit the relevant text reuse. Retain both notices locally. Reject explicit restrictive notices or quoted third-party material in selected text.",
    "separation": "No repeated source path, revision, recorded writer names, editor names or emails. Reject normalized twelve-word source fragments matching ten prior work packets or another selected source.",
    "cap": MAX_GROUPS,
}
SGML_INLINE = re.compile(r"</?(?:acronym|application|command|emphasis|envar|filename|firstterm|function|literal|option|parameter|productname|quote|replaceable|systemitem|term|type|varname)>", re.I)
MD_INLINE = re.compile(r"`([^`\n]+)`|\*\*([^*\n]+)\*\*")
ENTITY = re.compile(r"&(?:amp|gt|lt|quot|apos|nbsp|#[0-9]{1,5}|#x[0-9a-fA-F]{1,5});")
HUNK = re.compile(r"(?m)^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
ID = re.compile(r"[0-9a-f]{40}")
MANY = re.compile(r"(?im)^(?:revert|merge|automated|cherry.pick)\b")
BOT = re.compile(r"(?i)\[bot\]|^(?:github|k8s-ci-robot|kubernetes robot|postgresql buildfarm)$")
RESTRICTED = re.compile(r"(?i)all rights reserved|non-commercial|noncommercial|used with permission|licensed under (?:a |the )?(?!creative commons attribution 4\.0)")


def git(repo, *args, limit=24_000_000):
    result = subprocess.run(["git", f"--git-dir={repo}", *args], capture_output=True, check=True, timeout=45)
    require(len(result.stdout) <= limit, "Git archive result exceeds the frozen bound")
    return result.stdout.decode("utf-8", errors="strict")


def name_key(value):
    value = unicodedata.normalize("NFKD", value.casefold())
    return " ".join(re.findall(r"[^\W_]+", "".join(c for c in value if not unicodedata.combining(c))))


def catalog(repo, source):
    lines = git(repo, "log", source["anchor"], "--since=2015-01-01T00:00:00Z",
                "--before=2020-01-01T00:00:00Z", "--no-merges", "--no-renames", "--diff-filter=M",
                "--format=%x1e%H%x1f%P%x1f%at%x1f%ct%x1f%an%x1f%ae%x1f%s",
                "--name-only", "--", *source["directories"])
    roster = defaultdict(list)
    for record in lines.split("\x1e")[1:]:
        header, *paths = record.strip().splitlines()
        commit, parent, authored, committed, name, email, subject = header.split("\x1f", 6)
        if not (ID.fullmatch(parent) and START <= int(authored) < END and START <= int(committed) < END):
            continue
        editor = [name_key(name), email.lower()]
        if not editor[0] or not editor[1] or BOT.search(name) or MANY.search(subject):
            continue
        paths = [path for path in paths if path.strip()]
        if len(paths) > 8:
            continue
        for path in paths:
            genre = "blog" if path.startswith("content/en/blog/") else "docs"
            if not (re.fullmatch(r"(?:doc/src/sgml/[A-Za-z0-9_./-]+\.sgml|content/en/(?:docs|blog)/[A-Za-z0-9_./-]+\.md)", path)
                    and not re.search(r"(?i)(?:release-\d|generated|changelog|glossary|reference/generated)", path)):
                continue
            roster[genre].append({"path": path, "genre": genre, "before": parent, "after": commit,
                                  "after_authored": int(authored), "after_committed": int(committed), "editor": editor})
    selected = []
    for genre, entries in sorted(roster.items()):
        seen = Counter()
        for row in sorted(entries, key=lambda value: text_hash(value["path"] + value["after"])):
            editor = row["editor"][0]
            if seen[editor] < source["per_editor"][genre]:
                selected.append(row)
                seen[editor] += 1
            if sum(seen.values()) == source["maximum"][genre]:
                break
    return selected


def freeze(run):
    return {"schema": SCHEMA, "plan": PLAN, "code_sha256": digest(Path(__file__)),
            "history": {key: digest(path) for key, path in HISTORY.items()},
            "roster": {key: catalog(run / row["archive"], row) for key, row in SOURCES.items()}}


def region_blocks(raw, markdown):
    if markdown:
        front = re.match(r"^---[^\S\n]*\n.*?\n---[^\S\n]*\n", raw, re.S)
        start = front.end() if front else 0
        cursor = start
        for gap in re.finditer(r"\n[ \t]*\n(?:[ \t]*\n)*", raw[start:]):
            yield cursor, start + gap.start()
            cursor = start + gap.end()
        yield cursor, len(raw)
    else:
        for match in re.finditer(r"<para>\s*(.*?)\s*</para>", raw, re.S | re.I):
            yield match.start(1), match.end(1)


def rendered(raw, lo, hi, markdown):
    """Return literal prose and each visible character's raw source span."""
    value = raw[lo:hi]
    if (not re.match(r"^[ \t\n]*[A-Za-z<]", value) or len(value) > 12_000
            or RESTRICTED.search(value) or re.search(r"(?m)^\s*(?:[#>|]|[-+*]\s|\d+[.)]\s|```|~~~)", value)
            or re.search(r"(?i)https?://|\{\{|\}\}|\[|\]", value)):
        return "", []
    if markdown and re.search(r"(?m)^[ \t]{4,}\S", value):
        return "", []
    removed = set()
    for match in (MD_INLINE if markdown else SGML_INLINE).finditer(value):
        inner = match.span(match.lastindex) if markdown else (-1, -1)
        removed.update(i for i in range(*match.span()) if not inner[0] <= i < inner[1])
    visible, spans = [], []
    cursor = 0
    while cursor < len(value):
        if cursor in removed:
            cursor += 1
            continue
        entity = ENTITY.match(value, cursor)
        stop = entity.end() if entity else cursor + 1
        piece = html.unescape(entity[0]) if entity else value[cursor]
        if len(piece) != 1 or (not entity and piece in "<>`"):
            return "", []
        if piece.isspace():
            if visible and visible[-1] != " ":
                visible.append(" ")
                spans.append((lo + cursor, lo + stop))
        else:
            visible.append(piece)
            spans.append((lo + cursor, lo + stop))
        cursor = stop
    if visible and visible[-1] == " ":
        visible.pop()
        spans.pop()
    return "".join(visible), spans


def snippets(raw, markdown, changed):
    line_starts = [0] + [match.end() for match in re.finditer(r"\n", raw)]
    runs, run = [], []
    previous = 0
    for lo, hi in region_blocks(raw, markdown):
        gap = raw[previous:lo] if markdown else re.sub(r"</?para>", "", raw[previous:lo], flags=re.I)
        prose, anchors = rendered(raw, lo, hi, markdown)
        doc = from_string(prose)
        native = segment(doc) if prose.endswith((".", "!", "?")) and doc.text == prose else []
        if not all(unit.text.endswith((".", "!", "?")) for unit in native):
            native = []
        if run and (gap.strip() or not native):
            runs.append(run)
            run = []
        previous = hi
        for unit in native:
            raw_span = [anchors[unit.start][0], anchors[unit.end - 1][1]]
            lines = [bisect.bisect_right(line_starts, raw_span[0]), bisect.bisect_right(line_starts, raw_span[1] - 1)]
            run.append({"text": unit.text, "raw_span": raw_span, "lines": lines,
                        "raw_sha256": text_hash(raw[slice(*raw_span)])})
    if run:
        runs.append(run)
    candidates = []
    for run in runs:
        for i in range(2, len(run) - 2):
            items = run[i - 2:i + 3]
            a, b = items[2]["lines"]
            if not any(lo <= b and a <= hi for lo, hi in changed):
                continue
            texts = [entry["text"] for entry in items]
            sizes = [len(words(value)) for value in texts]
            if (not 12 <= sizes[2] <= 40 or sum(12 <= n <= 40 for n in sizes[1:4]) < 2
                    or not all(6 <= n <= 65 for n in sizes) or not 60 <= sum(sizes) <= 225):
                continue
            text = " ".join(texts)
            units = segment(from_string(text))
            if [unit.text for unit in units] != texts:
                continue
            center = units[2]
            candidates.append({"text": text, "focus": [center.start, center.end], "focus_text": center.text, "witness": items})
    return candidates


def blog_byline(raw):
    front = re.match(r"^---[^\S\n]*\n(.*?)\n---[^\S\n]*\n", raw, re.S)
    if not front:
        return []
    found = re.search(r"(?im)^authors?\s*:\s*(\S[^\n]*)", front[1])
    if not found or re.search(r"[{\[|>]", found[1]):
        return []
    return sorted({name_key(value) for value in re.split(r",|;|\s+(?:and|&)\s+", found[1].strip("\"'"))
                   if len(name_key(value).split()) >= 2})


def modified_ranges(diff):
    ranges = []
    for hunk in HUNK.finditer(diff):
        a, n, b, m = [int(x or 1) for x in hunk.groups()]
        if a and b and 0 < n <= 25 and 0 < m <= 25:
            ranges.append(([a, a + n - 1], [b, b + m - 1]))
    return ranges


def intersects(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def blame(repo, commit, path, lines):
    raw = git(repo, "blame", "--line-porcelain", "-L", f"{lines[0]},{lines[1]}", commit, "--", path, limit=100_000)
    proof = []
    for record in re.finditer(r"(?ms)^([0-9a-f]{40}) \d+ (\d+)(?: \d+)?\n(.*?)(?=^\t)", raw):
        data = dict(line.split(" ", 1) for line in record[3].splitlines() if " " in line)
        proof.append({"commit": record[1], "line": int(record[2]), "boundary": "boundary" in record[3].splitlines(),
                      "identity": [name_key(data["author"]), data["author-mail"].strip("<>").lower()]})
    require([row["line"] for row in proof] == list(range(lines[0], lines[1] + 1)), "Incomplete focus line attribution")
    return proof


def rights_record(repo, commit, source):
    value = git(repo, "show", f"{commit}:{source['rights']}", limit=100_000)
    normalized = " ".join(value.lower().split())
    permitted = (normalized.startswith("attribution 4.0 international") and "worldwide, royalty-free" in normalized
                 if source["license"] == "CC-BY-4.0"
                 else "permission to use, copy, modify, and distribute this software and its documentation for any purpose" in normalized)
    return (text_hash(value), value) if permitted else None


def exact_pair(repo, row, source, rendered_at):
    old, new = row["before"], row["after"]
    path = row["path"]
    diff = git(repo, "diff", "--unified=0", "--no-renames", "--no-ext-diff", "--no-textconv", old, new, "--", path, limit=600_000)
    hunks = modified_ranges(diff)
    if not hunks:
        return "no_bounded_changed_lines", None
    raw_old, raw_new = (rendered_at(commit, path) for commit in (old, new))
    if not raw_old or not raw_new:
        return "unsupported_source_blob", None
    markdown = path.endswith(".md")
    old_options = snippets(raw_old, markdown, [left for left, _ in hunks])
    new_options = snippets(raw_new, markdown, [right for _, right in hunks])
    if not old_options or not new_options:
        return "no_complete_native_excerpt", None
    options = []
    for left in old_options:
        for right in new_options:
            a, b = [item["text"] for item in left["witness"]], [item["text"] for item in right["witness"]]
            if not set(a[:2]) & set(b[:2]) or not set(a[3:]) & set(b[3:]):
                continue
            if not any(intersects(left["witness"][2]["lines"], x) and intersects(right["witness"][2]["lines"], y) for x, y in hunks):
                continue
            if .10 <= distance(a[2], b[2]) <= .70 and valid_sentence({"text": b[2], "eos": True}, a[2], generated=False):
                options.append((left, right))
    if not options:
        return "no_matched_center_replacement", None
    licenses = [rights_record(repo, commit, source) for commit in (old, new)]
    if not all(licenses):
        return "historical_license_missing", None
    before_authored, before_committed = map(int, git(repo, "show", "-s", "--format=%at %ct", old).split())
    if max(before_authored, before_committed) > row["after_committed"]:
        return "nonchronological_parent", None
    editorial = None
    byline = blog_byline(raw_old) if row["genre"] == "blog" else []
    for left, right in sorted(options, key=lambda pair: text_hash(pair[0]["text"] + "\0" + pair[1]["text"])):
        proof = [blame(repo, commit, path, item["witness"][2]["lines"])
                 for commit, item in zip((old, new), (left, right), strict=True)]
        if not any(item["commit"] == new for item in proof[1]):
            continue
        editor = row["editor"]
        attributable = all(not item["boundary"] and any(a and a == b for a, b in zip(editor, item["identity"], strict=True))
                           for side in proof for item in side)
        label = "same_recorded_contributor" if attributable else "publisher_edited_named_blog"
        if not attributable and not (byline and byline == blog_byline(raw_new)):
            continue
        result = {"before": left, "after": right, "recorded_authorship": label, "byline": byline,
                  "source": row | {"before_authored": before_authored, "before_committed": before_committed,
                                   "blob_sha256": list(map(text_hash, (raw_old, raw_new))), "diff_sha256": text_hash(diff),
                                   "license_sha256": [item[0] for item in licenses], "focus_line_attribution": proof}}
        if attributable:
            return label, result
        if editorial is None:
            editorial = result
    return ("publisher_edited_named_blog", editorial) if editorial else ("recorded_same_author_unverified", None)


def qualify(run, lock):
    found, dispositions, best = [], defaultdict(Counter), {}
    for key, metadata in SOURCES.items():
        repo = run / metadata["archive"]

        @lru_cache(maxsize=32)
        def get(commit, path, *, archive=repo):
            content = git(archive, "show", f"{commit}:{path}", limit=1_500_000)
            return content if len(content) <= 130_000 else None

        for row in lock["roster"][key]:
            try:
                status, item = exact_pair(repo, row, metadata, get)
            except (UnicodeDecodeError, subprocess.CalledProcessError):
                status, item = "unavailable_blob", None
            dispositions[f"{key}/{row['genre']}"][status] += 1
            if item is None:
                continue
            item |= {"group": f"{key}-{text_hash(row['path'] + row['after'])[:16]}", "archive": key, "license": metadata["license"]}
            group = (key, row["path"], row["editor"][0], row["editor"][1], status)
            if group not in best:
                found.append(item)
                best[group] = True
        get.cache_clear()
    signatures = {row["group"]: fingerprints(row["before"]["text"]) | fingerprints(row["after"]["text"]) for row in found}
    packets = [(key, digest(path), None, historical_texts(path)) for key, path in HISTORY.items()]
    history, external, _ = overlap(packets, signatures)
    selected, paths, identities, fragments, commits = [], set(), set(), set(), set()
    for row in sorted(found, key=lambda item: (item["recorded_authorship"] != "same_recorded_contributor", text_hash(item["group"]))):
        source, group = row["source"], row["group"]
        srcpath = (row["archive"], source["path"])
        ids = set(source["editor"] + row["byline"])
        if group in external:
            status = "prior_text_overlap"
        elif srcpath in paths or source["after"] in commits or ids & identities:
            status = "source_or_identity_overlap"
        elif signatures[group] & fragments:
            status = "within_panel_text_overlap"
        elif len(selected) == MAX_GROUPS:
            status = "over_cap"
        else:
            status = "admitted"
            selected.append(row)
            paths.add(srcpath)
            identities.update(ids)
            fragments.update(signatures[group])
            commits.add(source["after"])
        dispositions[f"{row['archive']}/{source['genre']}"][status] += 1
    return selected, {"schema": SCHEMA, "history": history,
                      "counts": {key: dict(values) for key, values in dispositions.items()},
                      "qualified_groups": len(selected), "strata": dict(Counter(
                          f"{row['archive']}/{row['source']['genre']}/{row['recorded_authorship']}" for row in selected))}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "qualify", "verify"))
    cli.add_argument("run", type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    run = args.run
    lock, originals, receipt = (run / name for name in ("source-freeze.json", "originals.json", "source-receipt.json"))
    expected = freeze(run)
    if args.phase == "freeze":
        write(lock, expected)
        print({key: dict(Counter(row["genre"] for row in values)) for key, values in expected["roster"].items()})
        return
    require(read(lock) == expected, "Source plan or historical archive changed")
    rows, summary = qualify(run, expected)
    notices = {}
    for row in rows:
        metadata = SOURCES[row["archive"]]
        for revision in (row["source"]["before"], row["source"]["after"]):
            checksum, value = rights_record(run / metadata["archive"], revision, metadata)
            notices[checksum] = {"archive": row["archive"], "license": metadata["license"], "text": value}
    notice_file = run / "source-notices.json"
    if args.phase == "qualify":
        write(originals, rows)
        write(notice_file, notices)
        write(receipt, summary | {"source_freeze_sha256": digest(lock), "originals_sha256": digest(originals),
                                  "notice_sha256": digest(notice_file)})
    else:
        require(read(originals) == rows and read(notice_file) == notices and read(receipt) == summary | {
            "source_freeze_sha256": digest(lock), "originals_sha256": digest(originals), "notice_sha256": digest(notice_file)
        }, "Historical source replay differs")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
