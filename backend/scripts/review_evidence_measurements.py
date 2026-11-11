"""Recompute selected PNG seam responses or video refinement schedules from an original."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import subprocess
from fractions import Fraction
from pathlib import Path
from threading import Timer

MAX_EVIDENCE_BYTES = 16 * 1024 ** 2
MAX_SOURCE_BYTES = 512 * 1024 ** 2
MAX_FRAMES = 108_000
SEAM_TITLE = "Sharp boundary to review"
SIDES = ("Left side", "Top side", "Right side", "Bottom side")
COMBINED = "Combined boundary response"
# The producer sums in float32 before rounding the packet to six decimals.
SEAM_ABS_TOLERANCE = .00005


def _regular(path: Path, maximum: int) -> None:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
        raise ValueError("Expected a nonempty regular file within the reviewer size limit")


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _side_response(pixel, *, position: int, start: int, stop: int, vertical: bool) -> float:
    def step(index: int, along: int) -> float:
        if index == 0:
            return 0.
        a, b = ((index, along), (index - 1, along)) if vertical else ((along, index), (along, index - 1))
        return sum(abs(u - v) for u, v in zip(pixel[a], pixel[b], strict=True)) / 3

    def response(along: int) -> float:
        before, center, after = (step(position + delta, along) for delta in (-3, 0, 3))
        return max(0., min(3., math.log((center + 3.) / ((before + after) / 2 + 3.))))

    return math.fsum(response(along) for along in range(start, stop)) / (stop - start)


def seam_values(image, box: dict) -> dict[str, float]:
    """Measure the published rectangle directly without the candidate search or its prefix sums."""
    if (set(box) != {"x", "y", "w", "h"} or any(type(box[key]) is not int for key in box)):
        raise ValueError("Expected an integer native seam rectangle")
    x, y, w, h = (box[key] for key in ("x", "y", "w", "h"))
    width, height = image.size
    if (w not in {round(Fraction(width, n)) for n in (6, 4, 3)}
            or h not in {round(Fraction(height, n)) for n in (6, 4, 3)}
            or not 3 <= x < x + w < width - 3 or not 3 <= y < y + h < height - 3):
        raise ValueError("The seam rectangle falls outside the method's native search geometry")
    pixel = image.load()
    measured = [
        _side_response(pixel, position=x, start=y, stop=y + h, vertical=True),
        _side_response(pixel, position=y, start=x, stop=x + w, vertical=False),
        _side_response(pixel, position=x + w, start=y, stop=y + h, vertical=True),
        _side_response(pixel, position=y + h, start=x, stop=x + w, vertical=False),
    ]
    return dict(zip((*SIDES, COMBINED), (*measured, math.prod(measured) ** .25), strict=True))


def review_image(path: Path, document: dict) -> list[dict]:
    from PIL import Image, ImageOps

    packets = [item for item in document["packets"] if item["title"] == SEAM_TITLE]
    if not packets:
        return []
    source = document["source"]
    with Image.open(path) as raw:
        if (raw.format != "PNG" or raw.mode != "RGB" or raw.n_frames != 1
                or min(raw.size) < 512 or max(raw.size) > 768):
            raise ValueError("Seam replay requires a single-frame RGB PNG from 512 to 768 pixels per side")
        orientation = int(raw.getexif().get(274, 1) or 1)
        with ImageOps.exif_transpose(raw) as image:
            if (source["coordinate_space"] != "exif-oriented-original" or orientation != source["exif_orientation"]
                    or image.size != (source["width"], source["height"])):
                raise ValueError("The source does not match the recorded oriented PNG geometry")
            output = []
            for packet in packets:
                anchor = packet["anchor"]
                if (packet["evidence_type"] != "anomaly" or packet["semantics"] != "residual"
                        or anchor["original_sha256"] != source["sha256"]
                        or anchor["coordinate_space"] != source["coordinate_space"]
                        or anchor["exif_orientation"] != orientation
                        or anchor["region_to_original"] != [1, 0, 0, 0, 1, 0]
                        or anchor["map_grid"] != list(image.size)):
                    raise ValueError("The seam packet lacks a native source-coordinate binding")
                measured = seam_values(image, packet["region"])
                rows = packet["measurements"]
                if len(rows) != len(measured) or {row["label"] for row in rows} != set(measured):
                    raise ValueError("The seam packet must contain four side values and one combined value")
                largest_error = 0.
                for row in rows:
                    actual = row["value"]
                    if (row["unit"] != "relative response" or type(actual) not in (float, int)
                            or not math.isfinite(actual) or abs(measured[row["label"]] - actual) > SEAM_ABS_TOLERANCE):
                        raise ValueError(f"Reported {row['label']} differs from the original pixel measurement")
                    largest_error = max(largest_error, abs(measured[row["label"]] - actual))
                output.append({"packet": packet["id"], "relative_responses_reproduced": len(measured),
                               "maximum_absolute_difference": largest_error})
    return output


def _native_clock(path: Path) -> tuple[Fraction, list[int]]:
    command = ["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v:0",
               "-show_entries", "stream=time_base:frame=pts,best_effort_timestamp", "-of", "compact",
               "file:" + str(path.resolve())]
    pts, bases = [], []
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                          stdin=subprocess.DEVNULL) as process:
        watchdog = Timer(90, process.kill)
        watchdog.start()
        try:
            for line in process.stdout:
                kind, *fields = line.rstrip().split("|")
                values = dict(field.split("=", 1) for field in fields if "=" in field)
                if kind == "stream":
                    bases.append(Fraction(values["time_base"]))
                if kind == "frame":
                    stamp = values.get("pts")
                    if stamp in (None, "N/A"):
                        stamp = values.get("best_effort_timestamp")
                    if stamp in (None, "N/A"):
                        raise ValueError("The clock contains an inferred timestamp. This reviewer requires native timestamps.")
                    if len(pts) >= MAX_FRAMES:
                        raise ValueError("The clock exceeds the reviewer frame limit")
                    pts.append(int(stamp))
            if process.wait() != 0:
                raise ValueError("FFprobe could not finish the local source clock read")
        finally:
            watchdog.cancel()
            if process.poll() is None:
                process.kill()
    if len(bases) != 1 or bases[0] <= 0 or not pts or any(a > b for a, b in zip(pts, pts[1:])):
        raise ValueError("Expected one nondecreasing native presentation clock")
    return bases[0], pts


def _scheduled_indices(pts: list[int], base: Fraction, start: int, stop: int) -> list[int]:
    """Sweep the clock once, with integer nearest distances and the earliest index on ties."""
    if not 0 < (stop - start) * base <= 1:
        raise ValueError("A refined face range requires a positive measured range of at most one second")
    # Keeping the first index for each timestamp also makes all clock distances unimodal.
    unique = [(stamp, i) for i, stamp in enumerate(pts) if i == 0 or stamp != pts[i - 1]]
    origin = pts[0] * base
    low, high = start * base - Fraction(5, 8), stop * base + Fraction(5, 8)
    first = max(0, math.floor((low - origin) * 12) - 1)
    last = min(math.floor((pts[-1] * base - origin) * 12), math.ceil((high - origin) * 12) + 1)
    selected: set[int] = set()
    factor = 12 * base.numerator
    pointer = 0
    for tick in range(first, last + 1):
        target = pts[0] * factor + tick * base.denominator
        while (pointer + 1 < len(unique)
               and abs(unique[pointer + 1][0] * factor - target) < abs(unique[pointer][0] * factor - target)):
            pointer += 1
        stamp, index = unique[pointer]
        if low <= stamp * base <= high:
            selected.add(index)
    return sorted(selected)


def _source_pt(index: int, pts: list[int]) -> int:
    if type(index) is not int or not 0 <= index < len(pts):
        raise ValueError("A reported frame index falls outside the original clock")
    return pts[index]


def _video_packet(packet: dict, pts: list[int], base: Fraction) -> dict:
    coarse, refined = packet["face_sequence"], packet["face_boundary"]
    if (refined["method"] != "face-appearance-boundary/1" or Fraction(refined["time_base"]) != base
            or refined["track_id"] != coarse["track_id"]):
        raise ValueError("The refined range does not use the selected appearance track and source clock")
    during = coarse["centers"]["during"]
    start, stop = _source_pt(during[0], pts), _source_pt(during[-1], pts)
    if (coarse["start_pts"], coarse["end_pts"]) != (start, stop):
        raise ValueError("The coarse review range differs from its recorded native centers")
    schedule = _scheduled_indices(pts, base, start, stop)
    if schedule != refined["scheduled_frame_indices"]:
        raise ValueError("The recorded refinement schedule differs from the source clock")
    support = refined["frame_inputs"]
    observed = [item["frame_index"] for item in support]
    if (observed != sorted(set(observed)) or not set(observed).issubset(schedule)
            or any(_source_pt(item["frame_index"], pts) != item["pts"] for item in support)):
        raise ValueError("Measured timing inputs do not match the reproduced schedule and clock")

    def transition(name: str) -> list[int]:
        group = refined["edges"][name]["frame_indices"]
        if len(group) != 4 or any(type(i) is not int or i not in observed for i in group):
            raise ValueError("A transition omits four measured frame references")
        first = schedule.index(group[0])
        if group != schedule[first:first + 4]:
            raise ValueError("A transition omits adjacent scheduled inputs")
        return group

    entry, exit_ = transition("entry"), transition("exit")
    begin, end = pts[entry[2]], pts[exit_[2]]
    if not begin < end or (refined["start_pts"], refined["end_pts"]) != (begin, end):
        raise ValueError("The refined range differs from the published transition frames")
    middle = min((i for i in observed if entry[2] <= i < exit_[2]), key=lambda i: (abs(2 * pts[i] - begin - end), i))
    roles = {"before": entry[1], "during": middle, "after": exit_[2]}
    anchors = {frame["interval_role"]: frame["anchor"] for frame in packet["face_boundary_frames"]}
    if (refined["anchors"] != roles or len(anchors) != len(packet["face_boundary_frames"])
            or set(anchors) != set(roles)
            or any((anchors[role]["frame_index"], anchors[role]["pts"]) != (i, pts[i]) for role, i in roles.items())):
        raise ValueError("The displayed timing frame roles differ from the measured transitions")
    return {"packet": packet["id"], "scheduled_frames_reproduced": len(schedule), "measured_inputs_checked": len(support),
            "displayed_frame_roles_checked": len(anchors), "refined_start_pts": begin, "refined_end_pts": end,
            "time_base": str(base)}


def review_video(path: Path, document: dict) -> list[dict]:
    packets = [item for item in document["packets"] if "face_boundary" in item or "face_boundary_frames" in item]
    if not packets:
        return []
    source = document["source"]
    if document["version"] != "video-evidence-11" or source["stream_selector"] != "v:0":
        raise ValueError("Unsupported video refinement packet version or source stream")
    base, pts = _native_clock(path)
    if Fraction(source["time_base"]) != base or source["frame_count"] != len(pts):
        raise ValueError("The decoded clock differs from the recorded frame count or time base")
    return [_video_packet(packet, pts, base) for packet in packets]


def review(kind: str, source: Path, evidence: Path) -> dict:
    _regular(source, MAX_SOURCE_BYTES)
    _regular(evidence, MAX_EVIDENCE_BYTES)
    data = evidence.read_bytes()
    document = json.loads(data)
    digest = _sha256(source)
    if (document["schema"] != f"chipmunk.{kind}-evidence/1" or document["status"] != "ready"
            or document["source"]["sha256"] != digest):
        raise ValueError("The evidence schema or its original source digest does not match")
    matches = (review_image if kind == "image" else review_video)(source, document)
    if _sha256(source) != digest:
        raise ValueError("The original source changed during the replay")
    return {"schema": "chipmunk.evidence-measurement-review/1", "modality": kind,
            "source_sha256": digest, "evidence_sha256": hashlib.sha256(data).hexdigest(),
            "status": "matched" if matches else "no_measurement_recorded", "packets": matches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("image", "video"))
    parser.add_argument("source", type=Path)
    parser.add_argument("evidence", type=Path, help="evidence.json extracted from an independently verified ZIP")
    args = parser.parse_args()
    try:
        result = review(args.kind, args.source, args.evidence)
    except (ImportError, OSError, KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as error:
        parser.exit(1, f"Evidence replay failed: {error}\n")
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
