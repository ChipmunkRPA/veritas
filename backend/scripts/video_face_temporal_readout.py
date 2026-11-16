# SPDX-License-Identifier: MIT
"""Read local face-texture changes on the delivered picture clock."""
from __future__ import annotations

import argparse
import hashlib
from fractions import Fraction
from pathlib import Path

from scripts import video_common_sources_v89 as files
from scripts.video_face_temporal_delivery import FPS, FRAME_COUNT, HEIGHT, WIDTH

STRIDE = 2
WINDOW = 18
WINDOW_STEP = 8
CANVAS = 112
LANDMARK_MIN = 18.0
MIN_FACE = 56
MIN_IOU = .35
ARMS = ("fixed", "flow")
FEATURES = ("luma_curvature", "chroma_curvature", "luma_reversal",
            "chroma_reversal", "luma_lateral_change", "chroma_lateral_change")
REGIONS = (
    (slice(48, 69), slice(39, 53)),
    (slice(48, 69), slice(59, 73)),
    (slice(49, 70), slice(25, 37)),
    (slice(49, 70), slice(75, 87)),
)


def align(rgb, face):
    import cv2
    import numpy as np

    eye, other = sorted(face.landmarks[:2])
    dx, dy = (other[axis] - eye[axis] for axis in range(2))
    length = dx * dx + dy * dy
    if length < LANDMARK_MIN**2:
        return None
    r = 36 * dx / length
    s = 36 * dy / length
    matrix = np.array([[r, s, 38 - r * eye[0] - s * eye[1]],
                       [-s, r, 42 + s * eye[0] - r * eye[1]]])
    inverse = cv2.invertAffineTransform(matrix)
    inspected = [(x, y) for x in (22, 90) for y in (45, 75)]
    native = np.asarray(inspected) @ inverse[:, :2].T + inverse[:, 2]
    if (native.min(axis=0) < 1).any() or (native.max(axis=0) >= np.asarray(rgb.shape[1::-1]) - 1).any():
        return None
    aligned = cv2.warpAffine(rgb, matrix, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return cv2.GaussianBlur(cv2.cvtColor(aligned, cv2.COLOR_RGB2YCrCb), (3, 3), .7)


def warp(source, current, grid):
    import cv2

    field = cv2.calcOpticalFlowFarneback(current[..., 0].copy(), source[..., 0].copy(), None,
                                       .5, 2, 15, 2, 5, 1.2, 0)
    return cv2.remap(source, grid + field, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def pair_features(before, central, after):
    """Measure what changes direction across three stabilized pictures."""
    import numpy as np

    before, central, after = (image.astype(np.float32) for image in (before, central, after))
    first, second = central - before, after - central
    curvature = np.abs(second - first)
    change = np.abs(first) + np.abs(second)
    features = []
    for plane in (slice(0, 1), slice(1, 3)):
        inter = [float(curvature[y, x, plane].mean()) for y, x in REGIONS]
        total = [float(change[y, x, plane].mean()) for y, x in REGIONS]
        middle, outer = sum(inter[:2]) / 2, sum(inter[2:]) / 2
        reversal = [a / (b + 1.) for a, b in zip(inter, total, strict=True)]
        features.append((np.log((middle + 1.) / (outer + 1.)),
                         (sum(reversal[:2]) - sum(reversal[2:])) / 2,
                         abs(inter[0] - inter[1]) / (middle + 1.) - abs(inter[2] - inter[3]) / (outer + 1.)))
    return tuple(float(value) for pair in zip(*features, strict=True) for value in pair)


def measure(images):
    import numpy as np

    grid = np.stack(np.meshgrid(np.arange(CANVAS), np.arange(CANVAS)), axis=-1).astype(np.float32)
    for index in range(1, len(images) - 1):
        before, central, after = images[index-1:index+2]
        if any(item is None for item in (before, central, after)):
            yield None
            continue
        yield (pair_features(before, central, after), pair_features(warp(before, central, grid), central, warp(after, central, grid)))


def table_for(path):
    from app.detectors.video.decode import probe_frames

    table = probe_frames(path, cap=FRAME_COUNT + 1)
    assert (table.width, table.height, table.rotation) == (WIDTH, HEIGHT, 0), path.name
    assert not table.truncated and not table.interpolated
    assert len(table) == FRAME_COUNT and [point * table.tb for point in table.pts] == [
        Fraction(index, FPS) for index in range(FRAME_COUNT)], path.name
    return table


def scan(path, detector):
    from statistics import median

    import cv2

    from app.detectors.image.faces import detect_faces, iou
    from app.detectors.video.decode import decode_frames

    table = table_for(path)
    scheduled = tuple(range(0, FRAME_COUNT, STRIDE))
    faces, images, hashes = [], [], []
    for frame, expected in zip(decode_frames(path, table, scheduled), scheduled, strict=True):
        assert frame.index == expected and frame.pts_verified, path.name
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        found = detect_faces(bgr, max_side=WIDTH, detector=detector, min_size=MIN_FACE)
        face = found[0] if found and (len(found) == 1 or found[0].w * found[0].h > 2 * found[1].w * found[1].h) else None
        faces.append(face)
        images.append(align(frame.rgb, face) if face is not None else None)
        hashes.append(hashlib.sha256(frame.rgb).hexdigest())
    values = list(measure(images))
    windows = []
    for start in range(0, len(scheduled) - WINDOW + 1, WINDOW_STEP):
        stop = start + WINDOW
        track = faces[start:stop]
        triplets = values[start:stop-2]
        if any(f is None for f in track) or any(item is None for item in triplets):
            continue
        if any(iou(left, right) < MIN_IOU for left, right in zip(track, track[1:])):
            continue
        first, middle, last = start, start + WINDOW // 2, stop - 1
        indexes = [scheduled[i] for i in (first, middle, last)]
        measurements = {arm: [round(median(row[view][dim] for row in triplets), 9)
                              for dim in range(len(FEATURES))] for view, arm in enumerate(ARMS)}
        windows.append({"frames": indexes, "pts": [table.pts[i] for i in indexes], "boxes": [track[i-start].box()
                        for i in (first, middle, last)], "rgb_sha256": [hashes[i] for i in (first, middle, last)],
                        "measurements": measurements})
    return {"time_base": table.time_base, "scheduled_pictures": len(scheduled),
            "face_pictures": sum(img is not None for img in images), "windows": windows}


def run(folder, output, split):
    import cv2

    from app.detectors.image.adapters.loaders import _entry, _file
    from app.detectors.image.faces import load_yunet
    from app.engine.fetchers import ensure_verified

    cv2.setNumThreads(1)
    entry = _entry("yunet.2023mar")
    ensure_verified(entry)
    asset = _file(entry, "face_detection_yunet_2023mar.onnx")
    detector = load_yunet(asset)
    manifest = folder / "manifest.json"
    rows = []
    for row in files.read(manifest)["rows"]:
        if row["role"] != split:
            continue
        if "file" in row:
            path = folder / row["file"]
            assert files.sha(path) == row["sha256"], path.name
            row = {key: value for key, value in row.items() if key != "native_map"} | scan(path, detector)
        rows.append(row)
        if "file" in row:
            print(row["file"], len(row["windows"]), flush=True)
    files.write(output, {"manifest_sha256": files.sha(manifest), "features": FEATURES, "arms": ARMS,
                         "yunet_sha256": files.sha(asset), "split": split, "rows": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deliveries", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("split", choices=("development", "held"))
    args = parser.parse_args()
    run(args.deliveries, args.output, args.split)
