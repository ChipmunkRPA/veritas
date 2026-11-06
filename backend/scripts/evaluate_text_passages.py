"""Measure source-mapped sentence evidence on held-out human–AI coauthored documents."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from prepare_text_upgrade import digest, file_hash
from scipy.stats import spearmanr
from score_text_upgrade import sequence_predictor, snapshot
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.detectors.text.extract import Document, from_string
from app.detectors.text.pipeline import windows
from app.detectors.text.segment import segment


def authorship(doc: Document, source_labels: list[str]) -> dict[int, str]:
    authors = {}
    for para in doc.source_map:
        for start, end, raw_start, raw_end in para["source_ranges"]:
            if end - start != raw_end - raw_start:
                if any(not char.isspace() for char in doc.text[start:end]):
                    raise ValueError("Cannot project this normalization onto source authorship")
                continue
            authors.update(zip(range(start, end), source_labels[raw_start:raw_end], strict=True))
    if any(i not in authors for i, char in enumerate(doc.text) if not char.isspace()):
        raise ValueError("Extracted text is missing authorship coordinates")
    return authors


def assess_file(case: dict, row: dict, predict) -> list[dict]:
    original = "".join(span["text"] for span in row["data"])
    if original != case["text"] or digest(original) != case["text_sha256"]:
        raise ValueError("Span annotation does not bind to the frozen source")
    labels = [span["author"] for span in row["data"] for _ in span["text"]]
    doc = from_string(original)
    authors = authorship(doc, labels)
    sents = segment(doc)
    inputs = windows(sents, settings.text_window_sentences)
    half = settings.text_window_sentences // 2

    def fraction(indices):
        relevant = [authors[i] for i in indices if not doc.text[i].isspace()]
        return relevant.count("Bot") / len(relevant)

    positions = [range(s.start, s.end) for s in sents]
    results = []
    for i, (focus, score) in enumerate(zip(sents, predict(inputs), strict=True)):
        context = positions[max(0, i - half):i + half + 1]
        results.append({
            "id": case["id"], "sentence": i, "start": focus.start, "end": focus.end,
            "extracted_text_sha256": digest(doc.text), "context_sha256": digest(inputs[i]),
            "score": score, "focus_ai_fraction": fraction(positions[i]),
            "context_ai_fraction": fraction(j for r in context for j in r),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("desklib", "tropa", "modernbert"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("spans", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text())
    spans_sha = file_hash(args.spans)
    if spans_sha != manifest["sources"]["ait-span.parquet"]["sha256"]:
        parser.error("Span data do not match the frozen archive")
    checkpoint = args.checkpoint
    if args.model == "desklib":
        checkpoint = snapshot("hf.desklib.ai_text_detector_v1_01")
    if checkpoint is None:
        parser.error("This model requires a local checkpoint")
    predict = sequence_predictor(args.model, checkpoint, args.device, batch=8, max_len=512)
    annotations = pq.read_table(args.spans).to_pylist()
    records = []
    for case in manifest["cases"]:
        if case["label"] == "mixed":
            index = int(case["id"].split(":")[1])
            records.extend(assess_file(case, annotations[index], predict))

    summary = {}
    for region in ("focus", "context"):
        key = f"{region}_ai_fraction"
        pure = [row for row in records if row[key] in {0, 1}]
        summary[region] = {
            "count": len(records), "pure_count": len(pure),
            "pure_roc_auc": float(roc_auc_score([row[key] for row in pure], [row["score"] for row in pure])),
            "rank_correlation": float(spearmanr([row[key] for row in records], [row["score"] for row in records]).statistic),
        }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump({"model": args.model, "manifest_sha256": file_hash(args.manifest), "spans_sha256": spans_sha,
                   "summary": summary, "records": records}, output, indent=2, allow_nan=False)
    print(summary)


if __name__ == "__main__":
    main()
