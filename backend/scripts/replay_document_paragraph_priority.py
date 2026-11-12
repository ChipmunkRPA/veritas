"""Replay first-card localization against saved paragraph transfer receipts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from app.detectors.text.extract import extract_text_like, text_hash
from app.evidence import document_paragraph_priority as priority
from app.evidence.document_packets import MAX_PACKETS
from scripts.review_speech_paragraph_transfer import read, records, require, sha

REF = {"id": "study", "display_id": "S01"}


def measured_packets(text: str, response: dict) -> list[dict]:
    current = response["current"]
    proof = response["current_proof"]
    require(proof["foci"] == proof["actual_input_windows"] == len(current), "Context proof needs one input per focus")
    positions = [{key: row[key] for key in ("start", "end")} for row in current]
    paragraphs = response["paragraphs"]
    for paragraph in paragraphs:
        (span,) = paragraph["spans"]
        start, end = span["start"], span["end"]
        require(text_hash(text[start:end]) == paragraph["input_sha256"], "Paragraph input does not match source")

    packets = []
    for i, row in enumerate(current):
        if row["raw"] < 0.5:
            continue
        focus = positions[i]
        spans = positions[max(0, i - 1):i + 2]
        context = {"spans": spans, "input_sha256": text_hash(" ".join(text[s["start"]:s["end"]] for s in spans)),
                   "score": row["raw"], "score_exact_hex": row["raw"].hex(),
                   # The receipt commits one model input per focus. Exact counts were not exported.
                   "token_coverage": {"tokens": 1, "token_windows": 1}}
        packet = {"id": f"T{len(packets) + 1:02d}", "focus": focus, "spans": spans,
                  "supporting_windows": [{**context, "signal_ref": REF, "direction": "supports_ai"}]}
        for paragraph in paragraphs:
            (span,) = paragraph["spans"]
            if span["start"] <= focus["start"] < focus["end"] <= span["end"]:
                packet["context_comparison"] = {"isolated_paragraph": {
                    "signal_ref": REF, "context": context, "paragraph": paragraph,
                    "supports_inspection": float.fromhex(paragraph["score_exact_hex"]) >= 0.5,
                }}
                break
        packets.append(packet)
        if len(packets) == MAX_PACKETS:
            break
    return packets


def useful(packet: dict | None, expected: set[int], text: str) -> bool:
    if packet is None:
        return False
    measured = {i for span in packet["spans"] for i in range(span["start"], span["end"]) if not text[i].isspace()}
    return bool(expected) and 10 * len(measured & expected) >= 3 * len(measured | expected)


def replay(directory: Path, receipt_path: Path) -> dict:
    receipt = read(receipt_path)
    controls = directory / "private-controls.jsonl"
    positives = directory / "private-positives.jsonl"
    responses = directory / "private-responses.jsonl"
    require(sha(controls) == receipt["source"]["private_controls_sha256"], "Control receipt differs")
    require(sha(positives) == receipt["preparation"]["case_sha256"], "Generated receipt differs")
    require(sha(responses) == receipt["terminal"]["responses_sha256"], "Response receipt differs")
    human, generated, gained_groups, lost_groups = Counter(), Counter(), set(), set()
    unavailable = 0
    with TemporaryDirectory() as scratch:
        original = Path(scratch) / "source.txt"
        for case, response in zip(records(controls) + records(positives), records(responses), strict=True):
            require(case["case"] == response["case"] and case["status"] == response["status"], "Case alignment differs")
            if case["status"] == "unavailable":
                unavailable += 1
                continue
            text = case["text"]
            require(text_hash(text) == case["text_sha256"] == response["text_sha256"], "Source text differs")
            original.write_text(text, encoding="utf-8")
            doc = extract_text_like(original, ".txt")
            require(doc.text == text and priority.eligible(text, doc.source_map, doc.meta["method"]), "Source is outside pilot scope")
            packets = measured_packets(text, response)
            before = packets[0] if packets else None
            preferred = priority.preferred_packet(packets, REF)
            after = preferred if preferred is not None else before
            expected = {i for a, b in case["ai_spans"] for i in range(a, b) if not text[i].isspace()}
            counts = generated if expected else human
            counts.update(assessed=1, eligible_first=preferred is not None, first_changed=before is not after,
                          files_with_retained_passages=bool(packets), retained_passages=len(packets))
            if expected:
                first_useful, priority_useful = useful(before, expected, text), useful(after, expected, text)
                gained, lost = priority_useful and not first_useful, first_useful and not priority_useful
                counts.update(first_useful_before=first_useful, first_useful_after=priority_useful, gained=gained, lost=lost)
                if gained:
                    gained_groups.add(case["article_group"])
                if lost:
                    lost_groups.add(case["article_group"])
    return {"source_receipt": receipt_path.name, "source_receipt_sha256": sha(receipt_path), "unavailable": unavailable,
            "human": dict(human), "generated": dict(generated), "source_groups_gained": len(gained_groups),
            "source_groups_lost": len(lost_groups)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.run_dir, args.receipt), indent=2, sort_keys=True))
