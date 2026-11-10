"""Exercise product context comparisons on retained model responses; no inference.

Contexts are reconstructed with the original reader's three-sentence recipe.
This does not independently reproduce the recorded model-input token hashes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import psutil

from app.detectors.text.extract import from_string, text_hash
from app.detectors.text.segment import segment
from app.evidence.document_context import build, validate

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
TRIAL = BASE / "text-format-comparison-v94"
INPUT = BASE / "text-format-audit-v93"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run():
    folder = ROOT / "backend/data/qa/document-context-recorded-model-v100"
    folder.mkdir(exist_ok=False)
    started = time.monotonic()
    process = psutil.Process()
    observations = []

    def observe(initial=False):
        row = {
            "elapsed_s": time.monotonic() - started,
            "rss_bytes": process.memory_info().rss,
            "available_ram_bytes": psutil.virtual_memory().available,
            "free_disk_bytes": shutil.disk_usage(folder).free,
        }
        minimum = 3 * 1024**3 if initial else 1536 * 1024**2
        assert row["elapsed_s"] < 900 and row["rss_bytes"] < 1536 * 1024**2
        assert min(row["available_ram_bytes"], row["free_disk_bytes"]) >= minimum
        observations.append(row)

    observe(initial=True)
    fixed = json.loads((TRIAL / "result.json").read_text())
    cases = [json.loads(line) for line in (INPUT / "inputs.jsonl").read_text().splitlines()]
    assert sha(INPUT / "inputs.jsonl") == fixed["source_input_packet_sha256"]
    counts = Counter()
    with (folder / "measurements.jsonl").open("x") as out:
        for batch in range(8):
            path = TRIAL / f"batch-{batch:02d}.jsonl"
            assert sha(path) == fixed["journal_sha256"][batch]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for case, row in zip(cases[batch * 36 : (batch + 1) * 36], rows, strict=True):
                assert case["id"] == row["id"]
                for view, response in zip(case["effective"], row["effective_responses"], strict=True):
                    assert response["status"] == "ok"
                    text = view["text"]
                    doc = from_string(text)
                    foci = segment(doc)
                    proposals = response["arms"]["current"]
                    assert (
                        doc.text == text and len(foci) == len(proposals) == response["current_native"]["foci"]
                    )
                    windows = []
                    for i, (focus, proposal) in enumerate(zip(foci, proposals, strict=True)):
                        assert (focus.start, focus.end) == (proposal["start"], proposal["end"])
                        group = foci[max(0, i - 1) : i + 2]
                        windows.append(
                            {
                                "focus": {"start": focus.start, "end": focus.end},
                                "spans": [{"start": s.start, "end": s.end} for s in group],
                                "score": proposal["raw"],
                                "input_sha256": text_hash(" ".join(s.text for s in group)),
                            }
                        )
                    result = {
                        "signals": [
                            {
                                "id": "recorded.current:global",
                                "detector_id": "recorded.current",
                                "display_id": "S01",
                                "status": "ok",
                                "details": {
                                    "text_evidence": {
                                        "version": "scored-context/1",
                                        "offset_unit": "unicode_codepoint",
                                        "extracted_text_sha256": text_hash(text),
                                        "windows": windows,
                                    }
                                },
                            }
                        ]
                    }
                    for focus in foci:
                        span = {"start": focus.start, "end": focus.end}
                        block = build(result, text, span)
                        validate(block, focus=span, chars=len(text), text=text)
                        counts["focus_comparisons"] += 1
                        counts[block["status"]] += 1
                        assert (
                            block["authorship_confidence_pct"] is None
                            and block["affects_overall_confidence"] is False
                        )
                        for check in block["checks"]:
                            expected = [
                                w["score"]
                                for w in windows
                                if any(s["start"] <= focus.start < focus.end <= s["end"] for s in w["spans"])
                            ]
                            assert check["lowest_response"] == min(expected) and check[
                                "highest_response"
                            ] == max(expected)
                            counts["nonzero_recorded_ranges"] += check["response_spread"] > 0
                        out.write(json.dumps(block, allow_nan=False) + "\n")
                    counts["effective_inputs"] += 1
                    observe()
    assert counts["effective_inputs"] == 800
    assert not {"torch", "transformers"}.intersection(sys.modules)
    result = {
        "schema": "chipmunk.document-context-recorded-model-check/100",
        "status": "passed",
        "counts": dict(counts),
        "new_neural_calls": 0,
        "neural_libraries_imported": False,
        "source_model_result_sha256": sha(TRIAL / "result.json"),
        "output_sha256": sha(folder / "measurements.jsonl"),
        "duration_s": time.monotonic() - started,
        "maximum_observed_self_rss_bytes": max(r["rss_bytes"] for r in observations),
        "minimum_observed_available_ram_bytes": min(r["available_ram_bytes"] for r in observations),
        "code_sha256": {
            name: sha(ROOT / name)
            for name in [
                "backend/scripts/check_document_context_v100.py",
                "backend/app/evidence/document_context.py",
            ]
        },
        "scope": "All 800 retained effective inputs, source focus geometry and raw current responses. Contexts are reconstructed using the prior reader recipe, not an independent token-input/model execution or fresh accuracy test. Per-input observations are not continuous RSS peaks. Source text and per-input records remain private.",
    }
    with (folder / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    run()
