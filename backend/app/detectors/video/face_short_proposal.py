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
    return _rank(reduced, lambda _: box, check)


def moving_shortlist(reduced, anchors, check):
    """Follow the recorded face centers while proposing candidate motion edges."""
    if len(anchors) < 4:
        return []
    indices = sorted(anchors)
    keys = ("x", "y", "w", "h")
    coordinates = np.asarray([[anchors[i][key] for key in keys] for i in indices])
    boxes = np.column_stack([np.interp(proof.EDGE_SCAN, indices, values) for values in coordinates.T])

    def box_at(edge):
        return dict(zip(keys, boxes[edge - proof.EDGE_SCAN.start], strict=True))

    return _rank(reduced, box_at, check)


def _rank(reduced, box_at, check):
    ranked = []
    for edge in proof.EDGE_SCAN:
        check("brief face proposal shortlist")
        if result := face_paired_edge.motion(edge, reduced, box_at(edge)):
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


def select(edge_candidates, values, reduced, check, excluded, *, durations=proof.DURATIONS):
    edges = face_paired_edge.appearance_edges(edge_candidates, values, reduced, check)
    ranked = []
    for start, entry in edges.items():
        for length in durations:
            stop = start + length
            if stop not in edges or excluded(start, stop):
                continue
            response, measurements, gaps = face_paired_edge.pair_response(entry, edges[stop])
            if response > proof.POINT:
                ranked.append(((-response, start, stop), measurements, gaps))
    return min(ranked, key=lambda item: item[0], default=None)


def has_measurable_pair(edges, faces, durations):
    complete = {edge for edge in edges if all(i in faces for i in proof.edge_frames(edge))}
    return any(start + duration in complete for start in complete for duration in durations)
