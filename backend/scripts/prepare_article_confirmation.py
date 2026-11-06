"""Freeze a second-source news comparison before running the installed classifiers."""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from prepare_text_upgrade import digest, file_hash, normalized_hash, suitable

SOURCE = "Rajarshi-Roy-research/Defactify_Text_Dataset@f510e7ced81c9af0aecc4648f5fe470b3c0132c7"
TRANSCRIPT = re.compile(r"(?:\b\d{2}:\d{2}\s*[-–]\s*\d{2}:\d{2}\b|^new video loaded:)", re.I)
HEADING = re.compile(r"(?im)^\s*(?:#+\s+|(?:title|byline|published|date):\s*)")
PLACEHOLDER = re.compile(r"(?i)\[(?:your|author|publisher|publication|insert) [^\]\n]{1,35}\]|\[date\]")


def plain_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = HEADING.sub("", value)
    value = PLACEHOLDER.sub("", value)
    return re.sub(r"[*_]{2,}", "", value).strip()


def shingles(text: str) -> set[tuple[str, str, str]]:
    words = re.findall(r"\w+", text.casefold())
    return set(zip(words, words[1:], words[2:]))


def distinct(signature: set, seen: list[set]) -> bool:
    # Candidate and reference articles may share a topic. Exclude overlaps of
    # the actual prose rather than treating shared words as duplicates.
    return all(len(signature & prior) / len(signature | prior) < 0.5 for prior in seen)


def freeze(source: Path, previous: Path, per_generator: int) -> list[dict]:
    old = json.loads(previous.read_text(encoding="utf-8"))["cases"]
    hashes = {normalized_hash(plain_text(case["text"])) for case in old}
    seen = [shingles(plain_text(case["text"])) for case in old if case["domain"] == "article"]
    pools = defaultdict(list)
    for index, row in enumerate(pq.read_table(source, columns=["Text", "Label_A", "Label_B"]).to_pylist()):
        raw = row["Text"]
        if not isinstance(raw, str) or TRANSCRIPT.search(raw):
            continue
        clean = plain_text(raw)
        key = normalized_hash(clean)
        if not suitable(clean) or key in hashes:
            continue
        if row["Label_A"] not in (0, 1):
            raise ValueError("Unexpected source label")
        label = "ai" if row["Label_A"] else "human"
        generator = row["Label_B"] if label == "ai" else "human"
        pools[(label, generator)].append({
            "id": f"news-confirmation:{index}", "domain": "article", "split": "holdout",
            "label": label, "generator": generator, "text": clean, "text_sha256": digest(clean),
            "source_sha256": digest(raw), "words": len(clean.split()), "normalized_sha256": key,
        })
    humans = pools[("human", "human")]
    selected = []
    for key, candidates in sorted(pools.items()):
        if key[0] == "human":
            continue
        n = 0
        for ai in sorted(candidates, key=lambda row: digest(f"article-confirmation-v1:{row['id']}")):
            ai_signature = shingles(ai["text"])
            if not distinct(ai_signature, seen):
                continue
            matched = sorted(humans, key=lambda row: (abs(row["words"] - ai["words"]), row["text_sha256"]))
            for human in matched:
                if human["normalized_sha256"] in hashes or abs(human["words"] - ai["words"]) > 35:
                    continue
                human_signature = shingles(human["text"])
                if not distinct(human_signature, [*seen, ai_signature]):
                    continue
                group = digest(f"length-match:{human['id']}:{ai['id']}")
                selected.extend([{**human, "group": group}, {**ai, "group": group}])
                seen.extend((ai_signature, human_signature))
                hashes.update((ai["normalized_sha256"], human["normalized_sha256"]))
                n += 1
                break
            if n >= per_generator:
                break
        if n != per_generator:
            raise ValueError(f"Only {n} matched examples for {key[1]}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("previous_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-generator", type=int, default=12)
    args = parser.parse_args()
    if args.per_generator < 1:
        parser.error("Specify at least one pair per generator")
    cases = freeze(args.source, args.previous_manifest, args.per_generator)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump({"protocol": "article-confirmation-v1", "source": SOURCE, "source_sha256": file_hash(args.source),
                   "previous_manifest_sha256": file_hash(args.previous_manifest), "cases": cases},
                  output, indent=2, ensure_ascii=False)
    print(dict(Counter((row["label"], row["generator"]) for row in cases)))


if __name__ == "__main__":
    main()
