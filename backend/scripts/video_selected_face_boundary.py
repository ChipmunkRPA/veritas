# SPDX-License-Identifier: Apache-2.0
"""Replay a selected face window for persistent local boundary residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import tarfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from scripts import video_selected_landmark_reliability as prior

common = prior.common
require = prior.require
ROLES = prior.ROLES
ARCS = {"right_cheek": 20, "right_jaw": 50, "chin": 90, "left_jaw": 130, "left_cheek": 160}
SPEC = {
    "minimum_eye_pixels": 24,
    "minimum_geometry_overlap": .35,
    "maximum_interframe_scale_ratio": 1.25,
    "maximum_roundtrip_error_px": 1.,
    "maximum_point_movement_patch_widths": .75,
    "minimum_patch_luma_std": 4.,
    "grid_period_px": 8,
    "maximum_grid_residual_ratio": 1.8,
    "stabilizer_gray_levels": .5,
    "minimum_signed_log_gap_above_flanks": math.log(1.5),
    "minimum_core_centers": 3,
    "minimum_core_fraction": .60,
    "minimum_eligible_arcs": 2,
}
POLICY = "Positive corroboration only on the exact selected interval. Preserve baseline flags and endpoints. Missing or unsupported residuals supply no originality inference."
ADVANCE = "Research replication only if at least 18 of 36 matched swaps in at least four source groups have local support, at least 80% of selected flags are measured, and none of the saved genuine stress views has support."


def journal(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def freeze(folder, baseline):
    old = prior.frozen(baseline)
    run = common.read(baseline / "run.json")
    require(run["scored_sha256"] == common.digest(baseline / "scored.jsonl"), "Original selected windows differ from receipt")
    common.write(folder / "freeze.json", {
        "schema": "chipmunk.video-selected-face-boundary-freeze/1",
        "baseline_freeze_sha256": common.digest(baseline / "freeze.json"),
        "baseline_journal_sha256": run["scored_sha256"],
        "bindings": {**common.bindings(), "backend/scripts/video_selected_face_boundary.py": common.digest(Path(__file__))},
        "rtfs": old["rtfs"], "camera": old["camera"],
        "specification": SPEC, "arc_angles_degrees": ARCS,
        "geometry": "Eye-oriented elliptical lower face boundary. Four native square patches at one and three half-spacings per side. Local translation to both immediate neighbors by forward-back Lucas–Kanade.",
        "measurement": "Median absolute signed luminance high-pass prediction error. Signed log gap across the boundary after removing the larger within-side control gap. Require same direction toward both neighbors and contrast to both flanks. Reject codec-grid concentration separately.",
        "policy": POLICY, "pilot_advancement": ADVANCE,
    })


def frozen(folder, baseline):
    lock = common.read(folder / "freeze.json")
    require(lock["baseline_freeze_sha256"] == common.digest(baseline / "freeze.json")
            and lock["baseline_journal_sha256"] == common.digest(baseline / "scored.jsonl")
            and lock["bindings"] == {**common.bindings(), "backend/scripts/video_selected_face_boundary.py": common.digest(Path(__file__))}
            and lock["specification"] == SPEC and lock["arc_angles_degrees"] == ARCS,
            "Frozen source or method binding differs")
    return lock


def indices(roles):
    return [i + offset for role in ROLES for i in roles[role] for offset in (-1, 0, 1)]


def exact_roles(times, row, rgb):
    """Recover skipped original cadence centers using the archived selected RGB commitment."""
    start, end = (Fraction(row["original"][key]) for key in ("start", "end"))
    schedule = [i for i in common.schedule(times, 4) if 0 < i < len(times) - 1]
    left, right = schedule.index(times.index(start)), schedule.index(times.index(end))
    possible = []
    for before in combinations(schedule[max(0, left - 4):left], 2):
        for after in combinations(schedule[right + 1:right + 5], 2):
            inner = schedule[left + 1:right]
            for count in range(1, len(inner) + 1):
                for middle in combinations(inner, count):
                    during = (schedule[left], *middle, schedule[right])
                    all_centers = (*before, *during, *after)
                    if any(times[b] - times[a] > Fraction(11, 20) for a, b in zip(all_centers, all_centers[1:])):
                        continue
                    roles = dict(zip(ROLES, (before, during, after), strict=True))
                    commitment = hashlib.sha256("".join(rgb[i] for i in indices(roles)).encode()).hexdigest()
                    if commitment == row["selected_input_sha256"]:
                        possible.append(roles)
    require(len(possible) == 1, "Archived selected native centers are not uniquely identified")
    return possible[0]


@dataclass(frozen=True)
class Picture:
    gray: np.ndarray
    x: int
    y: int
    face: object


def geometry(face):
    eyes = np.asarray(sorted(face.landmarks[:2]), np.float32)
    origin = eyes.mean(axis=0)
    horizontal = eyes[1] - eyes[0]
    spacing = float(np.linalg.norm(horizontal))
    if spacing < SPEC["minimum_eye_pixels"] or not np.isfinite(spacing):
        return None
    horizontal /= spacing
    vertical = np.asarray([-horizontal[1], horizontal[0]])
    mouth = np.asarray(face.landmarks[3:]).mean(axis=0)
    if float((mouth - origin) @ vertical) < .35 * spacing:
        return None
    return origin, horizontal, vertical, spacing


def load_pictures(path, expected_sha, table_meta, rgb, wanted, yunet):
    require(common.digest(path) == expected_sha, "Source bytes differ from native receipt")
    table = common.probe_frames(path, timeout_s=25)
    times = common.relative_times(table)
    original = table_meta["pts"]
    tb = Fraction(table_meta["time_base"])
    require(not table.truncated and not table.interpolated and table.rotation == 0
            and (table.width, table.height) == table_meta["size"]
            and [pts * table.tb for pts in table.pts] == [pts * tb for pts in original],
            "Native presentation clock or picture dimensions differ")
    pictures = {}
    for frame in common.decode_frames(path, table, sorted(wanted), timeout_s=65):
        index = frame.index
        require(frame.pts_verified and hashlib.sha256(frame.rgb).hexdigest() == rgb[index]
                and index not in pictures, "Native source sample differs from receipt")
        faces = common.detect_faces(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), detector=yunet, min_size=48)
        pictures[index] = None
        if len(faces) == 1:
            face = faces[0]
            pad = round(.55 * max(face.w, face.h)) + 18
            x, y = max(0, face.x - pad), max(0, face.y - pad)
            right, bottom = min(table.width, face.x + face.w + pad), min(table.height, face.y + face.h + pad)
            gray = cv2.cvtColor(frame.rgb[y:bottom, x:right], cv2.COLOR_RGB2GRAY)
            pictures[index] = Picture(gray, x, y, face)
    require(pictures.keys() == wanted, "Incomplete verified native source roster")
    return times, pictures


def common_area(pictures):
    x, y = max(p.x for p in pictures), max(p.y for p in pictures)
    right = min(p.x + p.gray.shape[1] for p in pictures)
    bottom = min(p.y + p.gray.shape[0] for p in pictures)
    if right - x < 56 or bottom - y < 56:
        return None
    return (x, y), [np.ascontiguousarray(p.gray[y - p.y:bottom - p.y, x - p.x:right - p.x]) for p in pictures]


def patch_centers(face, pose):
    origin, horizontal, vertical, spacing = pose
    side = max(11, min(21, 2 * round(.10 * spacing) + 1))
    center = origin + .16 * face.h * vertical
    points = []
    for degrees in ARCS.values():
        angle = math.radians(degrees)
        outward = math.cos(angle) * horizontal + math.sin(angle) * vertical
        edge = center + .44 * face.w * math.cos(angle) * horizontal + .39 * face.h * math.sin(angle) * vertical
        points.extend(edge + offset * (side + 2) * outward for offset in (-1.5, -.5, .5, 1.5))
    return np.asarray(points, np.float32), side


def track(center, neighbor, points):
    kwargs = {"winSize": (19, 19), "maxLevel": 2, "minEigThreshold": .0003}
    points = points.reshape(-1, 1, 2).astype(np.float32)
    forward, good, _ = cv2.calcOpticalFlowPyrLK(center, neighbor, points, None, **kwargs)
    reverse, back_good, _ = cv2.calcOpticalFlowPyrLK(neighbor, center, forward, None, **kwargs)
    error = np.linalg.norm(reverse[:, 0] - points[:, 0], axis=1)
    ok = good[:, 0].astype(bool) & back_good[:, 0].astype(bool) & (error <= SPEC["maximum_roundtrip_error_px"])
    return forward[:, 0], ok


def inside(point, image, radius):
    x, y = point
    return radius <= x < image.shape[1] - radius and radius <= y < image.shape[0] - radius


def residual(current, nearby, point, matched, side, origin):
    radius = side // 2 + 2
    if not inside(point, current, radius) or not inside(matched, nearby, radius):
        return None
    a = cv2.getRectSubPix(current, (side, side), tuple(point), patchType=cv2.CV_32F)
    b = cv2.getRectSubPix(nearby, (side, side), tuple(matched), patchType=cv2.CV_32F)
    if min(float(a.std()), float(b.std())) < SPEC["minimum_patch_luma_std"]:
        return None
    delta = a - b
    error = np.abs(delta - cv2.GaussianBlur(delta, (5, 5), 1.0))[1:-1, 1:-1]
    offsets = np.arange(side - 2) - (side // 2 - 1)
    phase = SPEC["grid_period_px"]
    x = (round(float(point[0]) + origin[0]) + offsets) % phase
    y = (round(float(point[1]) + origin[1]) + offsets) % phase
    grid = np.isin(x[None, :], (0, phase - 1)) | np.isin(y[:, None], (0, phase - 1))
    grid_ratio = (float(error[grid].mean()) + .25) / (float(error[~grid].mean()) + .25)
    return float(np.median(error)), grid_ratio


def boundary_gap(errors):
    levels = [math.log(x + SPEC["stabilizer_gray_levels"]) for x in errors]
    inner, boundary, outer = (b - a for a, b in zip(levels, levels[1:]))
    return math.copysign(max(0., abs(boundary) - max(abs(inner), abs(outer))), boundary)


def center_measurement(pictures):
    """Measure a local straddling pair and two same-side control pairs on each arc."""
    if any(p is None for p in pictures):
        return {}, "single_face_missing"
    before, current, after = pictures
    poses = [geometry(p.face) for p in pictures]
    if any(pose is None for pose in poses) or any(
        common.iou(current.face, neighbor.face) < SPEC["minimum_geometry_overlap"]
        for neighbor in (before, after)
    ):
        return {}, "unstable_face_geometry"
    spacings = [pose[3] for pose in poses]
    area = common_area(pictures)
    if max(spacings) / min(spacings) > SPEC["maximum_interframe_scale_ratio"] or area is None:
        return {}, "unstable_face_geometry"
    origin, (left, center, right) = area
    full_points, side = patch_centers(current.face, poses[1])
    points = full_points - np.asarray(origin, np.float32)
    directions = []
    for neighbor in (left, right):
        matched, good = track(center, neighbor, points)
        good &= np.linalg.norm(matched - points, axis=1) <= side * SPEC["maximum_point_movement_patch_widths"]
        directions.append([residual(center, neighbor, p, q, side, origin) if valid else None
                           for p, q, valid in zip(points, matched, good, strict=True)])
    rows = {}
    for index, name in enumerate(ARCS):
        start, end = index * 4, (index + 1) * 4
        pairs = [direction[start:end] for direction in directions]
        if any(item is None for pair in pairs for item in pair):
            rows[name] = {"status": "patch_unmeasured"}
            continue
        grid_ratio = max(float(np.median([item[1] for item in pair])) for pair in pairs)
        if grid_ratio > SPEC["maximum_grid_residual_ratio"]:
            rows[name] = {"status": "grid_concentrated", "grid_ratio": grid_ratio}
            continue
        a, b = (boundary_gap([item[0] for item in pair]) for pair in pairs)
        gap = math.copysign(min(abs(a), abs(b)), a) if a * b > 0 else 0.
        bounds = []
        for x, y in full_points[start:end]:
            half = side // 2
            bounds.append([round(float(x)) - half, round(float(y)) - half, side, side])
        rows[name] = {"status": "measured", "gap": gap, "grid_ratio": grid_ratio, "patches_xywh": bounds}
    return rows, None


def assess(by_role):
    arcs = {}
    for name in ARCS:
        parts = [[row["arcs"][name]["gap"] for row in by_role[role]
                  if row["arcs"].get(name, {}).get("status") == "measured"] for role in ROLES]
        before, during, after = parts
        minimum = max(SPEC["minimum_core_centers"], math.ceil(SPEC["minimum_core_fraction"] * len(by_role["during"])))
        if not before or not after or len(during) < minimum:
            continue
        signed_flanks = {}
        for sign in (-1, 1):
            reference = max(0., float(np.median(before)) * sign, float(np.median(after)) * sign)
            signed_flanks[sign] = sum(sign * value - reference >= SPEC["minimum_signed_log_gap_above_flanks"] for value in during)
        arcs[name] = {"measured_centers": list(map(len, parts)), "support_centers": signed_flanks,
                      "support_direction": next((sign for sign, count in signed_flanks.items() if count >= minimum), None)}
    supported = [name for name, arc in arcs.items() if arc["support_direction"] is not None]
    status = "unmeasured" if len(arcs) < SPEC["minimum_eligible_arcs"] else "supported" if supported else "measured_no_support"
    return status, arcs, supported if status == "supported" else []


def contact_sheet(folder, name, selected, pictures, arc, direction):
    """Keep a private before/during/after crop with the four measured patch locations."""
    tiles = []
    colors = [(220, 160, 40), (55, 165, 250), (60, 90, 240), (220, 160, 40)]
    for role in ROLES:
        measured = [row for row in selected[role] if row["arcs"].get(arc, {}).get("status") == "measured"]
        row = (max(measured, key=lambda item: item["arcs"][arc]["gap"] * direction)
               if role == "during" and direction else measured[len(measured) // 2])
        picture = pictures[row["index"]]
        boxes = row["arcs"][arc]["patches_xywh"]
        x0, y0 = max(picture.x, min(b[0] for b in boxes) - 16), max(picture.y, min(b[1] for b in boxes) - 16)
        x1 = min(picture.x + picture.gray.shape[1], max(b[0] + b[2] for b in boxes) + 16)
        y1 = min(picture.y + picture.gray.shape[0], max(b[1] + b[3] for b in boxes) + 16)
        tile = cv2.cvtColor(picture.gray[y0 - picture.y:y1 - picture.y, x0 - picture.x:x1 - picture.x], cv2.COLOR_GRAY2BGR)
        for (x, y, w, h), color in zip(boxes, colors, strict=True):
            cv2.rectangle(tile, (x - x0, y - y0), (x + w - x0 - 1, y + h - y0 - 1), color, 1)
        tile = cv2.resize(tile, (tile.shape[1] * 2, tile.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        label = f"{role}  {float(Fraction(row['time'])):.3f}s"
        tiles.append((tile, label))
    width = max(130, *(tile.shape[1] for tile, _ in tiles)) + 12
    height = max(tile.shape[0] for tile, _ in tiles) + 32
    sheet = np.full((height, width * 3, 3), 26, dtype=np.uint8)
    for number, (tile, label) in enumerate(tiles):
        x = number * width + 6
        sheet[26:26 + tile.shape[0], x:x + tile.shape[1]] = tile
        cv2.putText(sheet, label, (x, 17), cv2.FONT_HERSHEY_SIMPLEX, .43, (245, 245, 245), 1, cv2.LINE_AA)
    require(cv2.imwrite(str(folder / f"{name}.jpg"), sheet), "Could not write private crop evidence")


def measure_selected(row, roles, pictures, times, evidence, name, *, stress=False):
    selected = {}
    for role in ROLES:
        selected[role] = []
        for index in roles[role]:
            require(times[index + 1] - times[index - 1] <= Fraction(13, 100), "Original native triplet span differs")
            arcs, reason = center_measurement([pictures[i] for i in (index - 1, index, index + 1)])
            selected[role].append({"index": index, "time": str(times[index]), "arcs": arcs, "unmeasured": reason})
    start, end = (str(times[roles["during"][i]]) for i in (0, -1))
    require(start == row["original"]["start"] and end == row["original"]["end"], "Selected interval differs")
    status, eligible, supported = assess(selected)
    shown = (supported or list(eligible)) if stress else supported
    if shown:
        arc = shown[0]
        contact_sheet(evidence, name, selected, pictures, arc, eligible[arc]["support_direction"])
    return {"status": status, "centers": selected, "eligible_arcs": eligible, "supported_arcs": supported,
            "private_crop": f"{name}.jpg" if shown else None}


def unselected(row):
    require(row["status"] in {"baseline_unassessed", "no_baseline_flag"}, "Unexpected archived selection status")
    return {"status": row["status"], "supported_arcs": []}


def rtfs_group(archive, pin, group, yunet, evidence):
    index = pin["index"]
    receipt = common.DATA / f"group-{index:03d}-complete.json"
    review = common.DATA / f"group-{index:03d}-command-review-v96/result.json"
    require(common.digest(receipt) == pin["receipt_sha256"] and common.digest(review) == pin["review_sha256"],
            "Native RTFS receipt differs from freeze")
    cases = {c["condition"]: c for c in common.read(receipt)["cases"]}
    folder = archive / "video-native-inserts-v95" / f"group-{index:03d}"
    original, times, intervals = prior.comparison.match_cases(folder, cases, camera=False)
    plain_hashes = original["expected_rgb_sha256"]
    plans, wanted, measured = [], defaultdict(set), {}
    for offset, row in enumerate(group["rows"]):
        if not row["status"].startswith("flag_"):
            measured[offset] = unselected(row)
            continue
        case = cases[row["condition"]]
        a, b = intervals[row["duration"]]
        rgb = [case["expected_rgb_sha256"][i] if a <= i < b else sha for i, sha in enumerate(plain_hashes)]
        roles = exact_roles(times, row, rgb)
        wanted["original"].update(i for i in indices(roles) if not a <= i < b)
        wanted[row["condition"]].update(i for i in indices(roles) if a <= i < b)
        plans.append((offset, row, roles, a, b))

    def load(name):
        case = cases[name]
        clock = case["frame_table"]
        observed, pictures = load_pictures(folder / case["file"], case["sha256"],
            {**clock, "size": (clock["width"], clock["height"])}, case["expected_rgb_sha256"], wanted[name], yunet)
        require(observed == times, "Matched variant has a different presentation clock")
        return pictures

    original_pictures = load("original") if wanted["original"] else {}
    for name in (key for key in wanted if key != "original"):
        variant_pictures = load(name)
        for offset, row, roles, a, b in plans:
            if row["condition"] != name:
                continue
            pictures = {i: (variant_pictures if a <= i < b else original_pictures)[i] for i in indices(roles)}
            measured[offset] = measure_selected(row, roles, pictures, times, evidence, f"rtfs_{index:03d}_{offset:02d}")
    require(len(measured) == len(group["rows"]), "Incomplete selected-window comparison")
    return [{**row, "boundary": measured[offset]} for offset, row in enumerate(group["rows"])], sum(map(len, wanted.values()))


def camera_saved_group(pins, group, yunet, evidence):
    camera = prior.camera_saved
    roster = common.read(camera.TRIAL / "private-plan.json")
    rows, samples = [], 0
    for offset, (pin, row) in enumerate(zip(pins, group["rows"], strict=True)):
        require(row["condition"] == pin["role"], "Camera delivery role differs")
        condition = pin["condition"]
        source = roster[condition]
        tar, sidecar = camera.archive_paths(condition)
        receipt = common.read(sidecar)
        require(common.digest(sidecar) == pin["receipt_sha256"] and receipt["archive_sha256"] == pin["archive_sha256"]
                and source["sha256"] == pin["source_sha256"] and source["qualification_sha256"] == pin["qualification_sha256"],
                "Historical camera archive differs from freeze")
        prior.archive_io.verify_archive(tar, receipt)
        with tarfile.open(tar, "r:") as stream:
            select = prior.scoped(prior.webcam.select_record, {"sqlite3": SimpleNamespace(connect=prior.saved.memory_connect)})
            record, table, missing = select(stream, condition, source)
        if missing:
            rows.append({**row, "boundary": unselected(row)})
            continue
        expected = camera.expected_case(source)
        prior.saved.validate_sequence_record(record, {**table, "frame_count": len(table["pts"])}, table_pts=table["pts"])
        require(table["pts"] == expected["pts"] and Fraction(table["time_base"]) == Fraction(expected["time_base"])
                and all(expected["rgb"][item["frame_index"]] == item["decoded_rgb_sha256"] for item in record["frame_inputs"]),
                "Archived camera triplet clock or RGB commitment differs")
        roles = record["centers"]
        require(float.fromhex(record["measurement_hex"]["response"]) == row["original"]["score"]
                and set(indices(roles)) == {frame["frame_index"] for frame in record["frame_inputs"]},
                "Archived camera selection differs from baseline")
        source_dir = camera.PREPARATION / "slot-09-attempt-01"
        source_case = common.read(source_dir / "qualification.json")["cases"][pin["role"]]
        times, pictures = load_pictures(source_dir / source_case["file"], pin["source_sha256"], expected,
                                        expected["rgb"], set(indices(roles)), yunet)
        samples += len(pictures)
        rows.append({**row, "boundary": measure_selected(row, roles, pictures, times, evidence, f"camera_{offset}", stress=True)})
    return rows, samples


def run(folder, baseline, archive):
    lock = frozen(folder, baseline)
    require(archive.resolve() == prior.camera_saved.ARCHIVE, "Exact existing video archive required")
    evidence = folder / "private-crops"
    evidence.mkdir(mode=0o700)
    cv2.setNumThreads(2)
    yunet = common.load_yunet(common.FACE_FILE)
    groups = journal(baseline / "scored.jsonl")
    start, samples = time.monotonic(), 0
    with (folder / "scored.jsonl").open("x") as stream:
        for group in groups:
            name = group["group"]
            if name == "camera_saved":
                rows, n = camera_saved_group(lock["camera"]["9"], group, yunet, evidence)
            elif name == "camera_native":
                rows, n = [{**row, "boundary": unselected(row)} for row in group["rows"]], 0
            else:
                pin = next(pin for pin in lock["rtfs"] if name == f"rtfs_{pin['index']:03d}")
                rows, n = rtfs_group(archive, pin, group, yunet, evidence)
            samples += n
            stream.write(json.dumps({"group": name, "cohort": group["cohort"], "rows": rows, "native_samples": n},
                                    sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            counts = Counter(row["boundary"]["status"] for row in rows)
            print(name, dict(counts), "verified frames", n, flush=True)
    common.write(folder / "run.json", {"native_samples": samples, "elapsed_s": time.monotonic() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "private_crop_bytes": sum(path.stat().st_size for path in evidence.iterdir()),
        "scored_sha256": common.digest(folder / "scored.jsonl")})


def readout(folder, baseline):
    frozen(folder, baseline)
    run = common.read(folder / "run.json")
    scored = journal(folder / "scored.jsonl")
    require(run["scored_sha256"] == common.digest(folder / "scored.jsonl")
            and run["native_samples"] == sum(group["native_samples"] for group in scored), "Boundary journal differs from runtime receipt")
    counts, by_duration, groups = defaultdict(Counter), defaultdict(Counter), defaultdict(set)
    supporting_groups, locations = defaultdict(set), Counter()
    patch_statuses, center_statuses = Counter(), Counter()
    for old, new in zip(journal(baseline / "scored.jsonl"), scored, strict=True):
        require(old["group"] == new["group"] and old["cohort"] == new["cohort"], "Comparison group differs from baseline")
        for base, current in zip(old["rows"], new["rows"], strict=True):
            boundary = current["boundary"]
            require({k: v for k, v in current.items() if k != "boundary"} == base, "Baseline changed")
            status = boundary["status"]
            cohort = "known_swap" if base["truth"] else old["cohort"]
            counts[cohort][status] += 1
            groups[cohort].add(old["group"])
            if base["truth"]:
                by_duration[base["duration"]][status] += 1
                overlap = common.overlap(base["original"], base["truth"])
                locations["baseline_iou_at_least_half"] += overlap >= Fraction(1, 2)
                locations["supported_iou_at_least_half"] += overlap >= Fraction(1, 2) and status == "supported"
            if status in {"baseline_unassessed", "no_baseline_flag"}:
                require(not base["status"].startswith("flag_") and not boundary["supported_arcs"], "Support escaped selected windows")
                continue
            require(base["status"].startswith("flag_"), "Boundary assessed a window without a flag")
            # JSON changes integer direction keys to strings, but the status and arcs must replay.
            replayed = json.loads(json.dumps(assess(boundary["centers"])))
            require(replayed == [status, boundary["eligible_arcs"], boundary["supported_arcs"]], "Saved arc decisions do not replay")
            if status == "supported":
                supporting_groups[cohort].add(old["group"])
            for rows in boundary["centers"].values():
                for row in rows:
                    center_statuses[row["unmeasured"] or "eligible_geometry"] += 1
                    patch_statuses.update(arc["status"] for arc in row["arcs"].values())
    status_totals = sum(counts.values(), Counter())
    assessed = status_totals["supported"] + status_totals["measured_no_support"]
    selected = assessed + status_totals["unmeasured"]
    advance = (counts["known_swap"]["supported"] >= 18 and len(supporting_groups["known_swap"]) >= 4
               and selected > 0 and assessed / selected >= .8 and counts["genuine_camera_saved"]["supported"] == 0)
    common.write(folder / "aggregate.json", {"schema": "chipmunk.video-selected-face-boundary/1",
        "freeze_sha256": common.digest(folder / "freeze.json"), "cohorts": {
            cohort: {"views": dict(count), "source_groups": len(groups[cohort]),
                     "source_groups_with_support": len(supporting_groups[cohort])} for cohort, count in counts.items()},
        "known_swap_by_duration": {duration: dict(values) for duration, values in by_duration.items()},
        "locations": dict(locations), "center_statuses": dict(center_statuses), "arc_observation_statuses": dict(patch_statuses),
        "selected_flags_total": selected, "selected_flags_measured": assessed, "native_runtime": run,
        "pilot_advances": advance, "policy": POLICY, "pilot_advancement": ADVANCE})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=prior.camera_saved.ARCHIVE)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.folder, args.baseline)
    elif args.stage == "run":
        run(args.folder, args.baseline, args.archive)
    else:
        readout(args.folder, args.baseline)


if __name__ == "__main__":
    main()
