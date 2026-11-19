# SPDX-License-Identifier: Apache-2.0
"""Recount the published disclosure-scope and disclosure-location audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.writing-joss-disclosures/1"
RECEIPT_SHA256 = "ba4aa2a20d3d99c3690af1854d3a15ea8d2ba7ea992e5b7577c7f7daaa4a3e53"


def recount(record):
    cases = record["observations"]
    if (record["schema"] != SCHEMA or record["issue"] != 126
            or record["model_calls"] != 0 or record["authorship_weight"] != 0
            or [row["id"] for row in cases] != list(range(1, 21))):
        raise ValueError("The receipt differs from the closed audit scope")

    semantic = Counter()
    positions = Counter()
    for row in cases:
        reference, predicted = row["reference"], row["inferred_scope"]
        semantic[reference, predicted] += 1
        if row["semantic_anchors"]:
            if row["anchors_replayed"] is not True:
                raise ValueError("A reported semantic anchor lacks source replay")
            positions.update(files_with_semantic_anchors=1, replayed_semantic_anchors=row["semantic_anchors"])
        start, stop = row["heading_page"], row["end_page"]
        if start is None:
            if stop is not None or row["selected_lines"] or row["source_location_replayed"] or row["end_matches_reference"]:
                raise ValueError("A missing heading cannot supply reported positions")
            continue
        if not (type(start) is type(stop) is int and 1 <= start <= stop and row["selected_lines"] > 0
                and row["source_location_replayed"] is True):
            raise ValueError("A routed heading lacks ordered source pages and replayed lines")
        positions.update(headings_replayed=1, correct_extents=int(row["end_matches_reference"]),
                         correct_cross_page_extents=int(row["end_matches_reference"] and stop > start))

    controls = record["semantic_controls"]
    forgery, = (row for row in controls if row["reference"] == "indistinguishable_from_claim")
    interpreted = [row for row in controls if row is not forgery]
    routes = {row["case"]: row for row in record["heading_controls"]}
    return {
        "papers": len(cases),
        "matching_scope": sum(count for (reference, prediction), count in semantic.items() if reference == prediction),
        "scope_table": {f"{a} -> {b}": count for (a, b), count in sorted(semantic.items())},
        "position_counts": dict(positions),
        "semantic_controls_matching": sum(row["reference"] == row["scope"] for row in interpreted),
        "semantic_controls_classifiable": len(interpreted),
        "unmarked_forgery_inferred_scope": forgery["scope"],
        "quote_preface_misrouted": routes["quote_preface_separate_block"]["routes"],
        "nested_quote_extent_correct": routes["nested_separate_section"]["end_matches"],
    }


def review(path):
    with path.open("rb") as stream:
        payload = stream.read(20_001)
    if len(payload) > 20_000 or hashlib.sha256(payload).hexdigest() != RECEIPT_SHA256:
        raise ValueError("The receipt bytes differ from the published record")
    return recount(json.loads(payload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    print(json.dumps(review(parser.parse_args().receipt), sort_keys=True))
