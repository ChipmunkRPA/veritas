"""Freeze local, grouped authorship fixtures before scoring. The output contains source text."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_REVISIONS = {
    "llmtrace-test.jsonl": "iitolstykh/LLMTrace_classification@252e21e3dca713bc82bc3c1ef73bf533d7c9fc7e",
    "email.jsonl": "HarmJ0y/PhishingSpamDataSet@d13e9bb73d03f458e6d5b6cb6c7b0a708ea47e3d",
    "ait-span.parquet": "UKPLab/AITDNA@21d77f7d44833507750c0da309295c4147c1bda5",
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def suitable(text: object) -> bool:
    if not isinstance(text, str) or not 50 <= len(text.split()) <= 750:
        return False
    letters = [char for char in text if char.isalpha()]
    return bool(letters and sum(char.isascii() for char in letters) / len(letters) >= 0.9)


def article_pairs(path: Path):
    grouped = defaultdict(list)
    for index, row in enumerate(read_jsonl(path)):
        if (
            row["lang"] == "eng"
            and row["data_type"] in {"article", "news"}
            and row["topic_id"]
            and suitable(row["text"])
        ):
            grouped[row["topic_id"]].append(
                (index, row["label"], row["model"], row["text"].strip(), row["prompt_type"])
            )
    return grouped


def email_pairs(path: Path):
    grouped = defaultdict(list)
    for row in read_jsonl(path):
        if row["Type"] != "Valid" or not suitable(row["Body"]) or not row["Original email No."]:
            continue
        author = row["Created by"]
        if author == "Human":
            label, variant = "human", "original"
        elif row["Source"] == "LLM rephrase" and author not in {"", "Huamn"}:
            label, variant = "ai", "ai_rephrased"
        else:
            continue
        grouped[str(row["Original email No."])].append(
            (row["No."], label, author, row["Body"].strip(), variant)
        )
    return grouped


def normalized_hash(text: str) -> str:
    return digest(re.sub(r"[^\w]+", " ", text.casefold()).strip())


def select_pairs(grouped: dict, domain: str, limit: int, fit: int) -> list[dict]:
    selected = []
    used = set()
    generator_counts: Counter[str] = Counter()
    for group in sorted(grouped, key=lambda g: digest(f"text-upgrade-v1:{domain}:{g}")):
        rows = grouped[group]
        humans = [r for r in rows if r[1] == "human" and normalized_hash(r[3]) not in used]
        if not humans:
            continue
        human = min(humans, key=lambda r: digest(r[3]))
        human_hash = normalized_hash(human[3])
        machines = [
            r for r in rows if r[1] == "ai" and normalized_hash(r[3]) not in used | {human_hash}
        ]
        if not machines:
            continue
        machine = min(machines, key=lambda r: (generator_counts[r[2]], digest(r[3])))
        generator_counts[machine[2]] += 1
        split = "fit" if len(selected) < 2 * fit else "holdout"
        group_id = digest(f"{domain}:{group}")
        for index, label, generator, text, variant in (human, machine):
            selected.append({
                "id": f"{domain}:{index}", "group": group_id, "domain": domain, "split": split,
                "label": label, "generator": generator, "variant": variant, "text": text,
                "text_sha256": digest(text),
            })
            used.add(normalized_hash(text))
        if len(selected) == 2 * limit:
            return selected
    raise ValueError(f"Found {len(selected) // 2} eligible pairs for {domain}, wanted {limit}")


def mixed_examples(path: Path, limit: int) -> list[dict]:
    import pyarrow.parquet as pq

    # User and Bot are actual provenance annotations in this dataset.
    # Report the fraction of labeled non-whitespace characters, not a binary label.
    authors = set()
    selected = []
    rows = pq.read_table(path).to_pylist()
    indices = sorted(range(len(rows)), key=lambda i: digest(f"text-upgrade-v1:mixed:{i}"))
    for i in indices:
        row = rows[i]
        author = row["metadata"]["author"]
        if author in authors:
            continue
        text = "".join(s["text"] for s in row["data"])
        if not suitable(text):
            continue
        counts = Counter()
        for span in row["data"]:
            counts[span["author"]] += sum(not char.isspace() for char in span["text"])
        if min(counts["User"], counts["Bot"]) == 0 or set(counts) != {"User", "Bot"}:
            continue
        selected.append({
            "id": f"mixed:{i}", "group": digest(f"mixed:{author}"), "domain": "mixed",
            "split": "challenge", "label": "mixed", "generator": row["metadata"]["model"],
            "variant": row["metadata"]["task"], "text": text, "text_sha256": digest(text),
            "ai_character_fraction": counts["Bot"] / counts.total(),
        })
        authors.add(author)
        if len(selected) == limit:
            return selected
    raise ValueError(f"Found {len(selected)} distinct mixed authors, wanted {limit}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--article-pairs", type=int, default=200)
    parser.add_argument("--article-fit", type=int, default=80)
    parser.add_argument("--email-pairs", type=int, default=120)
    parser.add_argument("--mixed", type=int, default=40)
    args = parser.parse_args()
    if not 0 < args.article_fit < args.article_pairs or min(args.email_pairs, args.mixed) < 1:
        parser.error("Need nonempty fit, holdout and challenge sets")
    cases = select_pairs(article_pairs(args.sources / "llmtrace-test.jsonl"), "article",
                         args.article_pairs, args.article_fit)
    # This source is an evaluation challenge. No serving calibration is fitted from it.
    cases += select_pairs(email_pairs(args.sources / "email.jsonl"), "email", args.email_pairs, 0)
    cases += mixed_examples(args.sources / "ait-span.parquet", args.mixed)
    records = {
        "protocol": "text-upgrade-v1", "sources": {
            name: {"revision": revision, "sha256": file_hash(args.sources / name)}
            for name, revision in SOURCE_REVISIONS.items()
        }, "cases": cases,
    }
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(records, target, ensure_ascii=False, indent=2)
    print(dict(Counter((row["domain"], row["split"], row["label"]) for row in cases)))


if __name__ == "__main__":
    main()
