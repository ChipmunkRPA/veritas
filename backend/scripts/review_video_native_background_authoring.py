# SPDX-License-Identifier: Apache-2.0
"""Independently rebuild the native background panel from pinned source pictures."""

from __future__ import annotations

import argparse
import bisect
import hashlib
from collections import Counter
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from app.detectors.video.decode import FrameTable, decode_frames, probe_frames
from scripts import video_native_background_panel as panel

common, require = panel.common, panel.require
SIZE = (640, 360)
COUNT = 120
CUT_SPANS = ((54, 60), (67, 75), (54, 64), (65, 77))
LONG_SPANS = ((50, 68), (56, 78))
SHORT_SPANS = ((54, 60), (62, 73))
ORDINARY = (67, 75)


def _nearest(times: tuple[Fraction, ...], k: int) -> int:
    target = Fraction(k, 25)
    i = bisect.bisect_left(times, target)
    neighbors = range(max(0, i - 1), min(i + 1, len(times)))
    return min(neighbors, key=lambda j: (abs(times[j] - target), j))


def _read(archive: Path, pin: dict, variant: str, indices: range) -> list[np.ndarray]:
    shape, conditions, directory = panel.previous.prior.durations.source_set(pin, archive)
    source = conditions[variant]
    path = directory / source["file"]
    require(common.digest(path) == source["sha256"], "A source file differs from its archived bytes")
    table = probe_frames(path, timeout_s=25)
    require(table == FrameTable.from_json(source["frame_table"]) and (table.width, table.height) == shape,
            "The verified source clock differs")
    origin = table.pts[0]
    times = tuple((pts - origin) * table.tb for pts in table.pts)
    schedule = [_nearest(times, i) for i in indices]
    require(schedule == [pin["source_frame_indices"][i] for i in indices] and len(set(schedule)) == len(schedule),
            "Selected native inputs differ from the earlier-nearest-time rule")
    frames = []
    for index, frame in zip(schedule, decode_frames(path, table, schedule, max_side=None, timeout_s=90), strict=True):
        require(frame.index == index and frame.pts_verified and frame.pts == table.pts[index]
                and hashlib.sha256(frame.rgb.tobytes()).hexdigest() == source["expected_rgb_sha256"][index],
                "A selected picture differs from its archived pixels or clock")
        frames.append(cv2.resize(frame.rgb, SIZE, interpolation=cv2.INTER_AREA))
    return frames


def _expected(pair: int, direction: int) -> dict:
    cases = {"cross_cut": (CUT_SPANS[(pair + direction * 2) % 4], None),
             "long_swap": (LONG_SPANS[direction], ("swap_a", "swap_b")[direction]),
             ("light" if direction else "camera_punch"): (ORDINARY, None)}
    if pair < 4:
        cases["short_swap"] = (SHORT_SPANS[direction], ("swap_b", "swap_a")[direction])
    if direction == 0:
        cases["original"] = (None, None)
    return cases


def _change(kind: str, host: list[np.ndarray], guest: list[np.ndarray], span: tuple[int, int]) -> list[np.ndarray]:
    a, b = span
    if kind == "cross_cut":
        return guest[a:b]
    if kind == "light":
        return [np.rint(np.minimum(image.astype(np.float32) * 1.20, 255)).astype(np.uint8) for image in host[a:b]]
    require(kind == "camera_punch", "Unknown controlled intervention")
    return [cv2.resize(image[8:-8, 8:-8], SIZE, interpolation=cv2.INTER_LINEAR) for image in host[a:b]]


def _audit(folder: Path, archive: Path) -> dict:
    freeze = common.read(folder / "freeze.json")
    manifest = common.read(folder / "media.json")
    freeze_hash = common.digest(folder / "freeze.json")
    require(manifest["freeze_sha256"] == freeze_hash, "The authored panel differs from its source freeze")
    pairs = freeze["pairs"]
    roster = {row["index"]: row for row in freeze["source_roster"]}
    require(len(pairs) == 8 and len(roster) == 16 and set(roster) == {i for _, a, b in pairs for i in (a, b)}
            and all(0 <= value <= .15 and a != b for value, a, b in pairs), "The source roles are not distinct and complete")
    rows = {(row["pair"], row["direction"], row["condition"]): row for row in manifest["media"]}
    require(len(rows) == len(manifest["media"]) == 64, "The verification panel needs all 64 distinct roles")
    counts = Counter()
    with TemporaryDirectory(prefix="background-authoring-", dir=folder) as temporary:
        destination = Path(temporary) / "rebuild.mp4"
        for pair, (_, first, second) in enumerate(pairs):
            originals = [_read(archive, roster[index], "original", range(COUNT)) for index in (first, second)]
            counts["authenticated_source_pictures"] += 2 * COUNT
            for direction, index in enumerate((first, second)):
                host, guest = originals[direction], originals[1 - direction]
                for kind, (span, variant) in _expected(pair, direction).items():
                    row = rows.pop((pair, direction, kind))
                    expected_span = list(span) if span is not None else None
                    require(row["split"] == "held" and row["group"] == 2 * pair + direction and row["interval"] == expected_span
                            and row["truth"] == (expected_span if variant else None),
                            "A result uses a different declared source group, edit, or control")
                    path = folder / "media" / row["file"]
                    require(path.is_file() and common.digest(path) == row["sha256"], "An authored file differs from its manifest")
                    pictures = list(host)
                    if span is not None:
                        start, end = span
                        if variant:
                            replacement = _read(archive, roster[index], variant, range(start, end))
                            counts["authenticated_source_pictures"] += len(replacement)
                        else:
                            replacement = _change(kind, host, guest, span)
                        require(len(replacement) == end - start and
                                all(not np.array_equal(p, q) for p, q in zip(host[start:end], replacement, strict=True)),
                                "The declared changed interval contains an identical picture")
                        pictures[start:end] = replacement
                        counts["verified_changed_intervals"] += 1
                    panel.old.encoded.encode(destination, frames=iter(pictures), crf=23)
                    require(common.digest(destination) == row["sha256"], "Independently authored MP4 bytes differ")
                    destination.unlink()
                    counts["identical_mp4s"] += 1
            print(f"Rebuilt pair {pair + 1} of {len(pairs)}", flush=True)
    require(not rows and counts["identical_mp4s"] == 64, "The source replay is incomplete")
    return {"status": "verified", "freeze_sha256": freeze_hash, "media_sha256": common.digest(folder / "media.json"),
            "reviewer_sha256": common.digest(Path(__file__)), "counts": dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, default=panel.ARCHIVE)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = _audit(args.folder, args.archive)
    common.write(args.folder / "independent-authoring.json", result)
    print(result["counts"])


if __name__ == "__main__":
    main()
