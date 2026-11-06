"""Prepare local author-disjoint controls from released federal correspondence.

Only fully released records with federal sender addresses in the archive are used.
The result contains extracted source prose and stays in the ignored data directory.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from prepare_text_upgrade import digest, file_hash, normalized_hash, suitable

SOURCE = "https://www.kaggle.com/datasets/kaggle/hillary-clinton-emails"
FEDERAL_SENDER = re.compile(r"([\w.+-]+)[@©]([\w.-]+\.gov)\b", re.I)
QUOTE = re.compile(
    r"^\s*(?:>|from\s*:|sent\s*:|to\s*:|cc\s*:|subject\s*:|"
    r"-{2,}\s*(?:original message|forwarded message)|on .{1,300} wrote\s*:)",
    re.I,
)
RELEASE_LINE = re.compile(
    r"^\s*(?:unclassified\b|u\.?s\.? department of state\b|case no\.?\s+f-\d|"
    r"(?:\d{4}-\d+\s+)?doc(?:ument)? no\.?|release in full\b|"
    r"state dept\.\s*-\s*produced to house\b|page\s+\d+\s+of\s+\d+)", re.I
)
CONTACT = re.compile(r"(?:[\w.+-]+[@©][\w.-]+\.[a-z]{2,}|https?://\S+|\b(?:\+1[ .-]?)?\d{3}[ .-]\d{3}[ .-]\d{4}\b)", re.I)
SIGNATURE = re.compile(
    r"^\s*(?:\*+\s*)?(?:sent from my|--\s*$|privacy notice\s*:|confidentiality notice\s*:|"
    r"this e-mail is unclassified\b)", re.I
)
SCHEDULE_TIME = re.compile(r"(?im)^\s*\d{1,2}:\d{2}\s*(?:am|pm)\b")


def body(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        if QUOTE.match(line) or SIGNATURE.match(line):
            break
        if not RELEASE_LINE.match(line):
            lines.append(line.strip())
    text = "\n".join(lines).strip()
    # Removing the entire input is preferable to leaving corpus-specific markup in one class.
    return "" if CONTACT.search(text) or len(SCHEDULE_TIME.findall(text)) >= 3 else text


def trigrams(text: str) -> set[tuple[str, str, str]]:
    words = re.findall(r"\w+", text.casefold())
    return set(zip(words, words[1:], words[2:]))


def selected_rows(source: Path) -> dict[str, list[dict]]:
    authors = defaultdict(list)
    seen = set()
    seen_trigrams: list[set[tuple[str, str, str]]] = []
    with zipfile.ZipFile(source) as archive, archive.open("Emails.csv") as stream:
        for row in csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig")):
            sender = FEDERAL_SENDER.search(row["ExtractedFrom"])
            if not sender or row["ExtractedReleaseInPartOrFull"] != "RELEASE IN FULL":
                continue
            author = row["SenderPersonId"]
            if not author or not row["DocNumber"]:
                continue
            text = body(row["ExtractedBodyText"])
            if not suitable(text) or len(text.split()) > 220:
                continue
            key = normalized_hash(text)
            if key in seen:
                continue
            grams = trigrams(text)
            if any(len(grams & old) / len(grams | old) >= 0.6 for old in seen_trigrams):
                continue
            seen.add(key)
            seen_trigrams.append(grams)
            authors[author].append({
                "id": f"federal-email:{row['DocNumber']}", "group": digest(f"federal-email:{row['DocNumber']}"),
                "author_group": digest(f"federal-email-author:{author}"),
                "archive_document": row["DocNumber"], "archive_path": row["MetadataPdfLink"],
                "text": text, "text_sha256": digest(text),
            })
    return authors


def split_rows(authors: dict[str, list[dict]], per_split: int, max_per_author: int) -> list[dict]:
    ordered = sorted(authors, key=lambda a: digest(f"email-upgrade-v1:{a}"))
    # Place whole authors in the split with fewer available emails. Limit each
    # writer before assignment so a single writer cannot dominate the fit.
    splits: dict[str, list[dict]] = {"fit": [], "holdout": []}
    for author in ordered:
        entries = sorted(authors[author], key=lambda r: digest(r["id"]))[:max_per_author]
        split = min(splits, key=lambda name: len(splits[name]))
        splits[split].extend(entries)
    out = []
    for split, entries in splits.items():
        if len(entries) < per_split:
            raise ValueError(f"Only {len(entries)} eligible {split} emails, need {per_split}")
        for i, row in enumerate(sorted(entries, key=lambda r: digest(f"order:{r['id']}"))[:per_split]):
            out.append({**row, "split": split, "generator": ("qwen", "smol2")[i % 2],
                        "rewrite_style": ("direct", "minimal")[(i // 2) % 2]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-split", type=int, default=60)
    parser.add_argument("--max-per-author", type=int, default=12)
    args = parser.parse_args()
    if min(args.per_split, args.max_per_author) < 1:
        parser.error("Selection limits must be positive")
    rows = split_rows(selected_rows(args.archive), args.per_split, args.max_per_author)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump({"protocol": "email-upgrade-v2", "source": SOURCE, "source_sha256": file_hash(args.archive),
                   "originals": rows}, output, indent=2, ensure_ascii=False)
    for split in ("fit", "holdout"):
        cases = [row for row in rows if row["split"] == split]
        print(f"{split}: {len(cases)} originals across {len({row['author_group'] for row in cases})} authors")


if __name__ == "__main__":
    main()
