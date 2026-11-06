"""Join completed local rewrites with their exact originals and retain every exclusion."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from prepare_text_upgrade import digest, file_hash, read_jsonl


def assemble(originals: Path, journals: list[Path]) -> dict:
    payload = json.loads(originals.read_text(encoding="utf-8"))
    original_sha = file_hash(originals)
    originals_by_id = {row["id"]: row for row in payload["originals"]}
    if len(originals_by_id) != len(payload["originals"]):
        raise ValueError("The original manifest repeats an email ID")
    seen = set()
    excluded = {}
    cases = []
    sources = {"originals": {"sha256": original_sha, "archive_sha256": payload["source_sha256"],
                             "url": payload["source"]}}
    for journal in journals:
        records = read_jsonl(journal)
        metadata = next(records)
        generator = metadata["generator"]
        if generator in sources or metadata["originals_sha256"] != original_sha:
            raise ValueError("Unexpected generator or original manifest")
        sources[generator] = {**metadata, "journal_sha256": file_hash(journal)}
        for record in records:
            row = originals_by_id[record["id"]]
            if (row["generator"] != generator or row["id"] in seen or
                    digest(row["text"]) != record["input_text_sha256"] or row["text_sha256"] != record["input_text_sha256"]):
                raise ValueError("A generated sample does not match its original")
            seen.add(row["id"])
            if record["exclusion"]:
                excluded[row["id"]] = record["exclusion"]
                continue
            rewrite = record["text"]
            if not isinstance(rewrite, str) or digest(rewrite) != record["text_sha256"]:
                raise ValueError("Generated sample hash mismatch")
            shared = {"group": row["group"], "author_group": row["author_group"], "domain": "email",
                      "split": row["split"], "archive_document": row["archive_document"]}
            cases.append({**shared, "id": row["id"], "label": "human", "generator": "human",
                          "variant": "original", "text": row["text"], "text_sha256": row["text_sha256"]})
            cases.append({**shared, "id": f"{row['id']}:{generator}", "label": "ai", "generator": generator,
                          "variant": row["rewrite_style"], "text": rewrite, "text_sha256": record["text_sha256"]})
    if seen != originals_by_id.keys():
        raise ValueError("Wait for all generation journals to finish")
    return {"protocol": "email-upgrade-v2", "sources": sources, "exclusions": excluded, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("originals", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("journals", nargs="+", type=Path)
    args = parser.parse_args()
    manifest = assemble(args.originals, args.journals)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False)
    print(dict(Counter((row["split"], row["label"], row["generator"]) for row in manifest["cases"])))
    print("Exclusions", dict(Counter(manifest["exclusions"].values())))


if __name__ == "__main__":
    main()
