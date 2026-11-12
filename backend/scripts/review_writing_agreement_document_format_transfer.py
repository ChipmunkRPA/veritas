# SPDX-License-Identifier: Apache-2.0
"""Recompute per-format writing parity outcomes without reading source prose."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_image_small_boundary_control_stress import encoded, require, sha  # noqa: E402

HUMAN = {"original", "human_splice"}


def summarize(frozen, measured):
    require(measured["freeze_sha256"] == sha(encoded(frozen)), "The result belongs to a different format freeze")
    expected = {row["index"]: row for row in frozen["cases"]}
    by_format = defaultdict(Counter)
    by_panel = defaultdict(Counter)
    omissions = defaultdict(Counter)
    for row in measured["cases"]:
        case = expected.pop(row["index"])
        require([item["format"] for item in row["formats"]] == frozen["formats"], "Missing or repeated delivery format")
        baseline = row["formats"][0]
        require(baseline["format"] == "txt" and baseline["text_sha256"] == case["text_sha256"]
                and baseline["native_windows"] == len(case["contexts"]) and baseline["context_equal"], "TXT baseline differs from the pinned case")
        label = "human" if case["kind"] in HUMAN else "ai_edit"
        for item in row["formats"]:
            fmt = item["format"]
            text_equal = item["text_sha256"] == case["text_sha256"]
            context_equal = item["ordered_contexts_sha256"] == baseline["ordered_contexts_sha256"]
            require(item["text_equal"] == text_equal and item["context_equal"] == context_equal, "Format parity flags differ")
            exact = text_equal and context_equal and item["source_scope_qualified"] and item["verified_format_positions"] > 0
            if exact:
                require(item["app_source_gate"] and item["selections"] == row["baseline_selections"]
                        and item["prospective_gate"] == baseline["existing_gate"]
                        and item["context_span_anchors"] >= len(case["contexts"]), "Native source parity did not preserve the detector inputs")
            else:
                reason = ("different_decoded_text" if not text_equal else "different_ordered_contexts" if not context_equal else
                          "unsupported_source_map" if not item["source_scope_qualified"] else "unverified_native_positions")
                omissions[fmt][reason] += 1
            if item["existing_gate"]:
                require(fmt == "txt", "A prior-format route ran outside TXT")
            outcomes = {"files": 1, "text_equal": text_equal, "context_equal": context_equal, "exact_source_parity": exact,
                        "prospective_gate": item["prospective_gate"], "old_gate": item["existing_gate"],
                        "source_eligible": item["app_source_gate"], "native_contexts": item["native_windows"] if exact else 0,
                        "source_positions": item["verified_format_positions"], "context_span_anchors": item["context_span_anchors"],
                        "agreement_flags": bool(exact and item["selections"]["agreement"]),
                        "agreement_useful": exact and row["baseline_useful"]["agreement"]}
            by_format[f"{fmt}/{label}"].update(outcomes)
            by_panel[f"{fmt}/{case['panel']}"].update(outcomes)
    require(not expected, "A held file is missing")
    controls = {(index, role) for index in frozen["control_indices"] for role in ("quoted_eml", "html_eml", "raster_pdf")}
    negative = defaultdict(Counter)
    for row in measured["controls"]:
        role = row["role"]
        controls.remove((row["index"], role))
        negative[role].update({"views": 1, "admitted": row["admitted"], "app_source_gate": row["app_source_gate"],
                               "text_equal": row["text_equal"], "requires_ocr": row["extraction"] == "requires_ocr"})
    require(not controls, "A negative control is missing")
    return {"by_format_and_origin": by_format, "by_format_and_shape": by_panel, "omissions": omissions,
            "negative_controls": negative, "admitted_negative_controls": sum(row["app_source_gate"] for row in negative.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args()
    if args.source.is_file():
        package = json.loads(args.source.read_bytes())
        require(sha(encoded(package["measurements"])) == package["measurements_sha256"], "Published native measurements differ")
        result = summarize(package["freeze"], package["measurements"])
        require(result == package["summary"], "Published counts differ")
    else:
        lock, measured = (json.loads((args.source / f"{name}.json").read_bytes()) for name in ("freeze", "measurements"))
        result = summarize(lock, measured)
        if args.publish:
            with args.publish.open("xb") as stream:
                stream.write(encoded({"schema": lock["schema"], "freeze": lock, "measurements": measured,
                                      "measurements_sha256": sha(encoded(measured)), "summary": result}))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
