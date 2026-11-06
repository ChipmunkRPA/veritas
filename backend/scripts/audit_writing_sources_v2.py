"""Audit mixed-writing provenance/offsets before selecting a new experiment.

No inference, fitted parameters, source text, author names or span annotations
are exported. An unused-author pool is not a frozen independent confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from app.detectors.text.extract import from_string
from scripts.prepare_text_upgrade import normalized_hash

SPAN_SHA = "0641d20f9326522061bee792cad5f8963faf26bf37268e491f901cb7c2ce5cfb"
PRIOR_SHA = "e10769c188e69889c54f1116c5a4f8f03cbb58f4194a19a3a1ea9e969a92d28c"
CARDS = {
    "aitdna-card.md": "23c9f9af9dde11accd92ec3d94c87f190ceafda209d9749c3995c44a98010cb0",
    "tropa-card.md": "b511193473ce1507ca578c465561bd8ea06f37df8549d972beaad8b9143350a7",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_row(row):
    spans, metadata = row["data"], row["metadata"]
    if not spans or not isinstance(metadata["author"], str) or not metadata["author"].strip():
        raise ValueError("Declared author and source spans required")
    if type(metadata["human_only"]) is not bool:
        raise ValueError("Explicit human-only metadata required")
    labels, pieces = [], []
    for span in spans:
        if span["author"] not in ("User", "Bot") or not isinstance(span["text"], str):
            raise ValueError("Known character-provenance labels required")
        if not isinstance(span["queries"], list) or any(not isinstance(q, str) for q in span["queries"]):
            raise ValueError("Declared query records required")
        pieces.append(span["text"])
        labels.extend([span["author"]] * len(span["text"]))
    text = "".join(pieces)
    counts = Counter(label for char, label in zip(text, labels, strict=True) if not char.isspace())
    if metadata["human_only"] and counts["Bot"]:
        raise ValueError("Human-only metadata conflicts with retained Bot provenance")
    doc = from_string(text)
    projected = {}
    for paragraph in doc.source_map:
        for start, end, raw_start, raw_end in paragraph["source_ranges"]:
            if not 0 <= raw_start <= raw_end <= len(text) or not 0 <= start <= end <= len(doc.text):
                raise ValueError("Source coordinate outside its immutable text")
            if end - start != raw_end - raw_start:
                if any(not c.isspace() for c in doc.text[start:end]):
                    raise ValueError("Non-whitespace normalization cannot preserve character provenance")
                continue
            for index, label in zip(range(start, end), labels[raw_start:raw_end], strict=True):
                if index in projected and projected[index] != label:
                    raise ValueError("Conflicting source provenance projection")
                projected[index] = label
    if any(i not in projected for i, char in enumerate(doc.text) if not char.isspace()):
        raise ValueError("Extracted non-whitespace characters lack source provenance")
    return text, {
        "human_only": metadata["human_only"],
        "human_characters": counts["User"],
        "bot_characters": counts["Bot"],
        "mixed": counts["User"] > 0 and counts["Bot"] > 0,
        "extracted_nonwhitespace": sum(not c.isspace() for c in doc.text),
    }


def audit(rows, previous):
    old = previous["cases"]
    declared_authors = {
        r["metadata"]["author"]
        for r in rows
        if isinstance(r, dict)
        and isinstance(r.get("metadata"), dict)
        and isinstance(r["metadata"].get("author"), str)
        and r["metadata"]["author"].strip()
    }
    used = set()
    for case in old:
        if case["label"] != "mixed":
            continue
        index = int(case["id"].split(":")[1])
        text = "".join(s["text"] for s in rows[index]["data"])
        if hashlib.sha256(text.encode()).hexdigest() != case["text_sha256"] or text != case["text"]:
            raise ValueError("Historical mixed sample differs from the original provenance")
        used.add(rows[index]["metadata"]["author"])
    old_texts = {normalized_hash(c["text"]) for c in old}
    outcomes, inventory = [], []
    for index, row in enumerate(rows):
        try:
            text, info = check_row(row)
            known_author = row["metadata"]["author"] in used
            exact_reuse = normalized_hash(text) in old_texts
            inventory.append((row, known_author, exact_reuse, info))
            outcomes.append(
                {
                    "index": index,
                    "status": "ok",
                    "known_author": known_author,
                    "normalized_text_reuse": exact_reuse,
                    **info,
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            outcomes.append({"index": index, "status": "failed", "reason": str(error)})
    eligible = [(r, info) for r, known, repeated, info in inventory if not known and not repeated]
    return {
        "schema": "chipmunk.writing-source-preflight/2",
        "passed": len(inventory) == len(rows),
        "source_rows": len(rows),
        "prior_mixed_authors": len(used),
        "unused_authors_before_text_checks": len(declared_authors - used),
        "audited_rows": len(inventory),
        "failed_rows": sum(r["status"] == "failed" for r in outcomes),
        "unused_author_nonreused_pool": {
            "rows": len(eligible),
            "authors": len({r["metadata"]["author"] for r, _ in eligible}),
            "mixed_rows": sum(info["mixed"] for _, info in eligible),
            "human_only_rows": sum(info["human_only"] for _, info in eligible),
            "generator_rows": dict(Counter(r["metadata"]["model"] or "human-only" for r, _ in eligible)),
        },
        "outcomes": outcomes,
        "source_terms": "Pinned dataset card says cc-by-sa-4.0 in metadata and gpl-2.0 in prose. Unresolved; do not assign an unambiguous license or publish source text/annotations from this audit.",
        "detector_terms": "Pinned detector card consistently declares Apache-2.0 and commercial use; no new weights are downloaded or executed here.",
        "scope": "Provenance/Unicode-offset inventory only. Pool is not selected/frozen confirmation. Near-duplicate, shared-prompt, training-overlap and cross-source checks remain; no accuracy, authorship confidence or outside review is established.",
        "new_inference_run": False,
        "fitting_started": False,
        "deployment_eligible": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spans", type=Path)
    parser.add_argument("previous", type=Path)
    parser.add_argument("cards", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if digest(args.spans) != SPAN_SHA or digest(args.previous) != PRIOR_SHA:
        raise ValueError("Original declared writing source bytes changed")
    if any(digest(args.cards / name) != expected for name, expected in CARDS.items()):
        raise ValueError("Pinned publisher card bytes changed")
    import pyarrow.parquet as pq

    result = audit(pq.read_table(args.spans).to_pylist(), json.loads(args.previous.read_text()))
    result.update(
        source_sha256=SPAN_SHA,
        previous_manifest_sha256=PRIOR_SHA,
        publisher_cards_sha256=CARDS,
        audit_code_sha256=digest(Path(__file__)),
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "outcomes"}), flush=True)
