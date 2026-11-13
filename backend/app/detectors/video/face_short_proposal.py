# SPDX-License-Identifier: Apache-2.0
"""Propose a short face range from a bounded native-pixel shortlist."""

from __future__ import annotations

import numpy as np

from app.detectors.video import face_paired_edge
from app.evidence import face_short_proposal_binding as proof


def shortlist(reduced, anchor_boxes, check):
    if len(anchor_boxes) < 4:
        return []
    box = {key: float(np.median([item[key] for item in anchor_boxes])) for key in ("x", "y", "w", "h")}
    ranked = []
    for edge in proof.EDGE_SCAN:
        check("brief face proposal shortlist")
        if result := face_paired_edge.motion(edge, reduced, box):
            ranked.append((float.fromhex(result["central_hex"]), edge))
    kept = []
    measured = set(proof.ANCHOR_SAMPLES)
    for _, edge in sorted(ranked, key=lambda item: (-item[0], item[1])):
        group = set(proof.edge_frames(edge))
        if len(measured | group) > proof.MAX_FACE_READS:
            continue
        kept.append(edge)
        measured.update(group)
        if len(kept) == proof.MAX_EDGES:
            break
    return kept


def select(edge_candidates, values, reduced, check, excluded):
    edges = face_paired_edge.appearance_edges(edge_candidates, values, reduced, check)
    ranked = []
    for start, entry in edges.items():
        for length in proof.DURATIONS:
            stop = start + length
            if stop not in edges or excluded(start, stop):
                continue
            response, measurements, gaps = face_paired_edge.pair_response(entry, edges[stop])
            if response > proof.POINT:
                ranked.append(((-response, start, stop), measurements, gaps))
    return min(ranked, key=lambda item: item[0], default=None)
