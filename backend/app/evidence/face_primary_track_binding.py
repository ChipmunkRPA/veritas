# SPDX-License-Identifier: Apache-2.0
"""Replay the geometric association from saved native detection boxes."""

from bisect import bisect_left

AREA_RATIO = 4
MIN_OVERLAP = 0.35
MAX_DISTRACTOR_OVERLAP = 0.10
POSITION_MAX_ANCHOR_GAP = 50
PANE_DIVIDER = 640
PANE_GAP = 24
PANE_CHANGE_TILES = 8


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    height = max(0, min(ay + ah, by + bh) - max(ay, by))
    shared = width * height
    return shared / (aw * ah + bw * bh - shared)


def dominant(boxes):
    if not boxes:
        return None
    main, *rest = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)
    if any(
        box[2] * box[3] * AREA_RATIO > main[2] * main[3] or overlap(main, box) > MAX_DISTRACTOR_OVERLAP
        for box in rest
    ):
        return None
    return main


def anchors(detections, *, position_only=False):
    boxes = {
        i: main for i, candidates in sorted(detections.items()) if (main := dominant(candidates)) is not None
    }
    if len(boxes) < 4:
        return {}
    indices = list(boxes)
    maximum_gap = POSITION_MAX_ANCHOR_GAP if position_only else 20
    if any(b - a > maximum_gap or overlap(boxes[a], boxes[b]) < MIN_OVERLAP for a, b in zip(indices, indices[1:])):
        return {}
    return boxes


def follow(index, boxes, reference, *, position_only=False):
    if not reference:
        return None
    indices = list(reference)
    offset = bisect_left(indices, index)
    a = indices[max(0, offset - 1)]
    b = indices[min(len(indices) - 1, offset)]
    if index in reference:
        predicted = reference[index]
    elif a == b:
        if abs(index - a) > 10:
            return None
        predicted = reference[a]
    else:
        predicted = tuple(
            left + (right - left) * (index - a) / (b - a)
            for left, right in zip(reference[a], reference[b], strict=True)
        )
    expected_area = predicted[2] * predicted[3]

    def comparable(box):
        return overlap(box, predicted) >= MIN_OVERLAP and 0.5 <= box[2] * box[3] / expected_area <= 2

    if not position_only or not a <= index <= b:
        main = dominant(boxes)
        return main if main is not None and comparable(main) else None
    plausible = [box for box in boxes if comparable(box)]
    if len(plausible) != 1:
        return None
    main, = plausible
    if any(box != main and overlap(box, main) > MAX_DISTRACTOR_OVERLAP for box in boxes):
        return None
    return main


def pane_faces(boxes, pane):
    if pane == "left":
        return [box for box in boxes if box[0] + box[2] <= PANE_DIVIDER - PANE_GAP]
    return [box for box in boxes if box[0] >= PANE_DIVIDER + PANE_GAP]


def pane_anchors(detections, pane):
    separated = {i: pane_faces(boxes, pane) for i, boxes in detections.items()}
    other = "right" if pane == "left" else "left"
    if sum(len(boxes) == len(pane_faces(detections[i], other)) == 1
           for i, boxes in separated.items()) < 8:
        return {}
    return anchors(separated)


def pane_pick(index, boxes, reference, pane):
    found = pane_faces(boxes, pane)
    return follow(index, found, reference) if len(found) == 1 else None


def validate(record, source, planned, faces, anchor_samples, *, position_only=False, pane=None):
    rows = record["face_detection_boxes"]
    width, height = source["width"], source["height"]
    if (
        not isinstance(rows, list)
        or len(rows) != len(planned)
        or any(not isinstance(row, dict) or set(row) != {"frame_index", "boxes"} for row in rows)
        or [row["frame_index"] for row in rows] != planned
    ):
        raise ValueError("Dominant face evidence lacks its scheduled detection boxes")
    detections = {}
    for row in rows:
        boxes = row["boxes"]
        if (
            not isinstance(boxes, list)
            or len(boxes) > 50
            or any(
                not isinstance(box, list)
                or len(box) != 4
                or any(type(value) is not int for value in box)
                or not (
                    0 <= box[0] < width
                    and 0 <= box[1] < height
                    and 48 <= box[2] <= width - box[0]
                    and 48 <= box[3] <= height - box[1]
                )
                for box in boxes
            )
        ):
            raise ValueError("Dominant face evidence has an invalid native detection box")
        detections[row["frame_index"]] = [tuple(box) for box in boxes]
    sampled = {index: detections[index] for index in anchor_samples}
    reference = pane_anchors(sampled, pane) if pane else anchors(sampled, position_only=position_only)
    if not reference or not any(len(detections[index]) > 1 for index in faces):
        raise ValueError("Dominant face evidence lacks a continuous sampled position")
    for index, reading in faces.items():
        selected = tuple(reading["bbox"][key] for key in ("x", "y", "w", "h"))
        expected = (pane_pick(index, detections[index], reference, pane) if pane else
                    follow(index, detections[index], reference, position_only=position_only))
        if expected != selected:
            raise ValueError("The measured face does not follow its recorded native detections")

def position_context(record, anchor_samples):
    reference = anchors({row["frame_index"]: [tuple(box) for box in row["boxes"]]
                         for row in record["face_detection_boxes"] if row["frame_index"] in anchor_samples},
                        position_only=True)
    middle = record["anchors"]["during"]
    indices = list(reference)
    offset = bisect_left(indices, middle)
    if middle in reference:
        return (middle,)
    return tuple(dict.fromkeys((indices[max(0, offset-1)], indices[min(len(indices)-1, offset)])))
