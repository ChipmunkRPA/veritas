# SPDX-License-Identifier: Apache-2.0
"""Associate native face detections using the portable geometric rules."""

from app.evidence import face_primary_track_binding as binding


def find_anchors(detections, *, position_only=False):
    return binding.anchors({i: [face.box() for face in found] for i, found in detections.items()},
                           position_only=position_only)


def follow(index, detections, anchors, *, position_only=False):
    selected = binding.follow(index, [face.box() for face in detections], anchors, position_only=position_only)
    return next((face for face in detections if face.box() == selected), None)
