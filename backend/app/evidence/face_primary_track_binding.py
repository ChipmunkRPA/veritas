# SPDX-License-Identifier: Apache-2.0
"""Replay the geometric association from saved native detection boxes."""

from bisect import bisect_left

AREA_RATIO = 4
MIN_OVERLAP = 0.35
MAX_DISTRACTOR_OVERLAP = 0.10


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


def anchors(detections):
    boxes = {
        i: main for i, candidates in sorted(detections.items()) if (main := dominant(candidates)) is not None
    }
    if len(boxes) < 4:
        return {}
    indices = list(boxes)
    if any(b - a > 20 or overlap(boxes[a], boxes[b]) < MIN_OVERLAP for a, b in zip(indices, indices[1:])):
        return {}
    return boxes


def follow(index, boxes, reference):
    main = dominant(boxes)
    if main is None or not reference:
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
    return (
        main
        if (
            overlap(main, predicted) >= MIN_OVERLAP
            and 0.5 <= main[2] * main[3] / (predicted[2] * predicted[3]) <= 2
        )
        else None
    )


def validate(record, source, planned, faces, anchor_samples):
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
    reference = anchors({index: detections[index] for index in anchor_samples})
    if not reference or not any(len(detections[index]) > 1 for index in faces):
        raise ValueError("Dominant face evidence lacks a continuous sampled position")
    for index, reading in faces.items():
        selected = tuple(reading["bbox"][key] for key in ("x", "y", "w", "h"))
        if follow(index, detections[index], reference) != selected:
            raise ValueError("The measured face does not follow its recorded native detections")
