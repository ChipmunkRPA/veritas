# SPDX-License-Identifier: Apache-2.0
"""Reserve historical rewrites and test the Rich Text earlier-draft comparison."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path

from scripts.writing_transition_deviation import file_hash, read, save, text_hash
from scripts.writing_turing_editorial_history import (
    CHAPTER,
    LICENSE,
    REVISION,
    SHA,
    SOURCE,
    git,
    license_digest,
    matches,
    snapshot,
)

SCHEMA = "chipmunk.writing-rtf-draft-holdout/1"
AFTER = datetime.fromisoformat("2021-01-01T00:00:00+00:00")
BEFORE = datetime.fromisoformat("2021-11-11T00:00:00+00:00")
MIN_WORKS = 8
MAX_WORKS = 18
FORMATS = (".rtf:.rtf", ".docx:.rtf", ".txt:.rtf", ".rtf:.docx")
RULES = {"date_range": [AFTER.isoformat(), BEFORE.isoformat()], "min_works": MIN_WORKS, "max_works": MAX_WORKS,
         "max_per_chapter_or_contributor": 2, "excluded": "Prior editorial paths and all prior panels' normalized 12-word fragments.",
         "location": "Each version must mark at least one token inside the verified changed sentence. No marked token may fall outside it.",
         "unchanged": "Formatting-only controls must yield no wording changes.", "formats": list(FORMATS)}


def candidates(repo):
    output = git(repo, "log", REVISION, "--no-merges", "--no-renames", "--diff-filter=M", "--name-only",
                 f"--after={AFTER.isoformat()}", f"--before={BEFORE.isoformat()}",
                 "--format=%H%x09%P%x09%aI%x09%cI%x09%an%x09%ae", "--", "book/content", "book/website")
    commit = None
    for line in output.decode().splitlines():
        if "\t" in line:
            new, previous, authored, committed, contributor, contact = line.split("\t")
            commit = None
            if (all(SHA.fullmatch(value) for value in (new, previous)) and contributor.strip() and contact
                    and all(AFTER <= datetime.fromisoformat(value) < BEFORE for value in (authored, committed))
                    and not re.search(r"\bbot\b|\[bot\]|action|dependabot", f"{contributor} {contact}", re.I)):
                commit = {"before_commit": previous, "after_commit": new, "committed": committed,
                          "author_group": text_hash(contributor.casefold().strip())}
        elif commit and (match := CHAPTER.fullmatch(line)):
            family = match[2].casefold()
            if family not in {"afterword", "foreword", "community-handbook"}:
                chapter = match[3].split("/", 1)[0].removesuffix(".md") if match[1] == "website" else family
                yield commit | {"path": line, "chapter": chapter.casefold()}


def freeze(repo, previous, output):
    from scripts.writing_economic_paraphrase_holdout import find_texts, fragments
    from scripts.writing_multi_passage_panel import native

    previous_paths = list(previous)
    prior = read(previous_paths[0])["works"]
    prior_chapters = {row["chapter"] for row in prior}
    prior_authors = {row["author_group"] for row in prior}
    used_paths = {row["path"] for row in prior}
    chapters, authors = Counter(), {}
    seen = {fragment for path in previous_paths for text in find_texts(read(path)) for fragment in fragments(text)}
    rows = sorted(candidates(repo), key=lambda row: text_hash(SCHEMA + row["after_commit"] + row["path"]))
    if not rows:
        raise ValueError("The pinned publisher tag has no qualifying chronological history")
    exclusions = Counter()
    works = []
    for row in rows:
        path = row["path"]
        chapter, author = row["chapter"], row["author_group"]
        if path in used_paths or chapters[chapter] >= 2 or authors.get(author, (None, 0))[1] >= 2:
            exclusions["shared_group"] += 1
            continue
        old_rev, new_rev = (row[key] for key in ("before_commit", "after_commit"))
        before, after = (snapshot(repo, version, path) for version in (old_rev, new_rev))
        if not before or not after:
            exclusions["source_unavailable"] += 1
            continue
        for old, new in matches(before, after):
            signatures = fragments(old["text"]) | fragments(new["text"])
            if signatures & seen:
                continue
            terms = [license_digest(repo, revision) for revision in (old_rev, new_rev)]
            old_date = git(repo, "log", "-1", "--format=%cI", old_rev).decode().strip()
            if (not all(terms) or not datetime.fromisoformat(old_date) < datetime.fromisoformat(row["committed"])):
                exclusions["missing_terms_or_sequence"] += 1
                break
            versions = [native(version["text"], ()) for version in (old, new)]
            if any(value is None or value["text"] != selected["text"] or len(value["sentences"]) != 12
                   or value["sentences"][selected["sentence"]][:2] != selected["focus"]
                   for value, selected in zip(versions, (old, new), strict=True)):
                continue
            alias, appearances = authors.get(author, (f"A{len(authors) + 1:03d}", 0))
            works.append(row | {"author_group": alias, "prior_contributor_group": author in prior_authors,
                                "prior_chapter_group": chapter in prior_chapters, "document": text_hash(path), "split": "test",
                                "before_date": old_date, "license_sha256": terms,
                                "file_sha256": [text_hash(data.decode()) for data in (before, after)],
                                "witnesses": [{key: val for key, val in item.items() if key not in {"text", "focus_text"}}
                                              for item in (old, new)], "variants": versions})
            chapters[chapter] += 1
            authors[author] = (alias, appearances + 1)
            used_paths.add(path)
            seen.update(signatures)
            break
        else:
            exclusions["no_native_match"] += 1
        if len(works) == MAX_WORKS:
            break
    save(output, {"schema": SCHEMA, "rules": RULES, "candidate_count": len(rows), "publisher": SOURCE,
                  "revision": REVISION, "license": LICENSE, "implementation_sha256": file_hash(Path(__file__)),
                  "prior_sha256": [file_hash(path) for path in previous_paths], "works": works, "excluded": dict(exclusions)})
    print(f"Reserved {len(works)} unique-file chronological pairs across {len(chapters)} chapters and {len(authors)} contributors",
          dict(exclusions), flush=True)
    if len(works) < MIN_WORKS:
        raise ValueError("The publisher archive has too few qualifying fresh pairs")


def rtf(text, *, styled=False):
    parts = []
    for para in text.split("\n\n"):
        chars = []
        for ch in para:
            if ch in "\\{}":
                chars.append("\\" + ch)
            elif 32 <= ord(ch) < 127:
                chars.append(ch)
            else:
                units = ch.encode("utf-16-le")
                chars.extend(f"\\u{int.from_bytes(units[index:index + 2], 'little', signed=True)}?"
                             for index in range(0, len(units), 2))
        plain = "".join(chars)
        parts.append(f"{{\\b {plain}}}" if styled else plain)
    return ("{\\rtf1\\ansi\\ansicpg1252\\uc1 " + "\\par ".join(parts) + "}").encode("ascii")


def package(text, suffix, *, styled=False):
    if suffix == ".rtf":
        return rtf(text, styled=styled)
    if suffix == ".txt":
        return text.encode()
    from docx import Document

    document = Document()
    for para in text.split("\n\n"):
        document.add_paragraph(para)
    stream = BytesIO()
    document.save(stream)
    return stream.getvalue()


def locations(comparison, earlier, later):
    positions = []
    for key, body in zip(("earlier", "later"), (earlier, later), strict=True):
        ranges = []
        starts = [match.end() for match in re.finditer(r"\n\n", body)]
        for change in comparison["changes"]:
            if (value := change[key]) is None:
                continue
            start, end = value["start"], value["end"]
            if value["text"] != body[start:end]:
                raise ValueError("The exported excerpt differs from its original decoded range")
            if "paragraphs" in value:
                boundaries = zip([0, *starts], [stop - 2 for stop in starts] + [len(body)], strict=True)
                anchors = [{"paragraph": index + 1, "body_index": index, "start": max(start, a) - a, "end": min(end, b) - a}
                           for index, (a, b) in enumerate(boundaries)
                           if max(start, a) < min(end, b) or start == end and a <= start <= b]
                # Word's body_index is a physical XML block index. The test files use one body block per paragraph.
                if anchors != value["paragraphs"]:
                    raise ValueError("Paragraph coordinates differ from the source fixture")
            ranges.extend(value["changed_tokens"])
        positions.append(ranges)
    return positions


def compare(sources, focus):
    from app.evidence.document_revision import RTF_COORDINATES, compare_with_sources

    before, after = sources
    rows = {}
    for pair in FORMATS:
        old, new = pair.split(":")
        data = (package(before, old), package(after, new))
        comparison, old_text, new_text, _ = compare_with_sources(*data, new, earlier_extension=old)
        if (old_text, new_text) != (before, after):
            raise ValueError("The two document carriers failed to preserve exact extracted text")
        for side, suffix in (("earlier", old), ("later", new)):
            coordinate = comparison["coordinate_space"] if old == new else comparison[side].get("coordinate_space")
            if suffix == ".rtf" and coordinate != RTF_COORDINATES:
                raise ValueError("RTF files must be marked with extracted Rich Text coordinates")
        unchanged = compare_with_sources(data[0], package(before, new, styled=True), new, earlier_extension=old)[0]
        rows[pair] = {"focus": focus, "positions": locations(comparison, before, after),
                      "restyled_changed_ranges": locations(unchanged, before, before)}
    return rows


def summarize(records):
    summary = {}
    for pair in FORMATS:
        totals = Counter()
        for work in records:
            for lane, readings in work["readings"].items():
                row = readings[pair]
                outside = sum(not lo <= a < b <= hi
                              for (lo, hi), regions in zip(row["focus"], row["positions"], strict=True) for a, b in regions)
                totals.update({f"{lane}_pairs": 1, f"{lane}_locations": int(not outside and all(row["positions"])),
                               f"{lane}_outside_tokens": outside,
                               f"{lane}_ordinary_untouched_flags": int(any(row["restyled_changed_ranges"]))})
        summary[pair] = dict(totals)
    return summary


def measure(source, output, ai=None):
    sealed = read(source)
    if sealed["rules"] != RULES or sealed["schema"] != SCHEMA or sealed["implementation_sha256"] != file_hash(Path(__file__)):
        raise ValueError("Source reservation and the frozen settings differ")
    generated = {}
    if ai:
        report = read(ai)
        if report["source_sha256"] != file_hash(source):
            raise ValueError("Generated rewrites belong to a different source reservation")
        generated = {row["document"]: row["variants"] for row in report["works"]}
    works = []
    for item in sealed["works"]:
        earlier, later = (row["text"] for row in item["variants"])
        before, after = (row["focus"] for row in item["witnesses"])
        observations = {"historical": compare((earlier, later), [before, after])}
        for row in generated.get(item["document"], ()):
            observations[row["version"]] = compare((earlier, row["text"]), [before, row["truth"][0]["source"]])
        works.append({key: value for key, value in item.items() if key != "variants"} | {"readings": observations})
    save(output, {"schema": SCHEMA, "rules": RULES, "publisher": SOURCE, "license": LICENSE, "revision": REVISION,
                  "source_sha256": file_hash(source), "generated_sha256": file_hash(ai) if ai else None,
                  "prior_sha256": sealed["prior_sha256"], "source_exclusions": sealed["excluded"],
                  "generation_rejections": report["rejected"] if ai else None, "works": works, "counts": summarize(works),
                  "comparison_sha256": file_hash(Path(__file__).resolve().parents[1] / "app/evidence/document_revision.py")})
    review(output)


def review(path):
    result = read(path)
    if result["schema"] != SCHEMA or result["rules"] != RULES or result["counts"] != summarize(result["works"]):
        raise ValueError("The recorded range measurements do not replay")
    print(result["counts"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    reserve = actions.add_parser("freeze")
    reserve.add_argument("repo", type=Path)
    reserve.add_argument("output", type=Path)
    reserve.add_argument("previous", type=Path, nargs="+")
    evaluate = actions.add_parser("measure")
    evaluate.add_argument("source", type=Path)
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("--ai", type=Path)
    actions.add_parser("review").add_argument("path", type=Path)
    args = vars(parser.parse_args())
    {"freeze": freeze, "measure": measure, "review": review}[args.pop("action")](**args)
