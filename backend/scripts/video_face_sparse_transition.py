# SPDX-License-Identifier: Apache-2.0
"""Evaluate scene support for single face transitions where the background is sparse."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

ARMS = ("incumbent", "fine_tiles", "off_face_motion")
CONTEXT = {
    "scaled_size": [160, 90],
    "minimum_region": [40, 12],
    "minimum_gradient": 5.0,
    "minimum_gradient_points": 32,
    "minimum_horizontal_groups": 2,
    "minimum_points_per_group": 8,
    "minimum_appearance": .90,
    "maximum_residual": 9.,
    "maximum_background": 8.,
    "maximum_jump_ratio": 2.5,
    "minimum_neighbors": 4,
}
CALIBRATED = "calibrated_off_face_motion"
FLOOR = "off_face_motion_at_floor"
CALIBRATION = {"minimum_ordinary_groups": 4, "floor": .06, "allowed_development_groups": 1}
ADMISSION = {
    "minimum_added_localizations": 5,
    "minimum_added_sources": 3,
    "maximum_added_ordinary_alerts": 3,
    "maximum_added_ordinary_sources": 2,
    "maximum_added_cut_alerts": 0,
    "maximum_added_misplaced_edits": 2,
    "boundary_tolerance_pictures": 1,
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def freeze(manifest, output):
    from app.detectors.video import face_return_fine_scene as fine
    from app.detectors.video import face_single_boundary as current
    from scripts.video_face_local_measure import assets

    packet = read(manifest)
    rows = packet["rows"]
    identities, digests = defaultdict(set), defaultdict(set)
    for row in rows:
        group = str(row["group"])
        if row["split"] not in {"development", "held"}:
            raise ValueError("Unknown study split")
        identities[row["split"]].add(group)
        digests[row["split"]].add(row["sha256"])
        if not 0 < len(group) < 90 or sha(Path(row["path"])) != row["sha256"]:
            raise ValueError("Invalid frozen source binding")
    if identities["development"] & identities["held"] or digests["development"] & digests["held"]:
        raise ValueError("A source group crosses the development and held split")
    write(output, {"manifest_sha256": sha(manifest), "reviewer_sha256": sha(Path(__file__)),
                   "baseline_sha256": sha(Path(current.__file__)), "fine_sha256": sha(Path(fine.__file__)),
                   "licensed_weights": assets(), "arms": ARMS, "context": CONTEXT, "admission": ADMISSION,
                   "point": current.POINT, "calibration": CALIBRATION, "exploratory_floor_arm": FLOOR, "groups": {key: sorted(value) for key, value in identities.items()}})


def center_region(box_a, box_b):
    """Use a fixed rectangle below both detected heads, in downsampled picture coordinates."""
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b
    width, height = CONTEXT["scaled_size"]
    mid = (xa + wa / 2 + xb + wb / 2) / 2
    w, h = (wa + wb) / 2, (ha + hb) / 2
    y = max(ya + ha, yb + hb)
    left = max(0, math.floor((mid - 1.05 * w) * width / 640))
    right = min(width, math.ceil((mid + 1.05 * w) * width / 640))
    top = max(0, math.floor((y + .12 * h) * height / 360))
    bottom = min(height, math.ceil((y + .78 * h) * height / 360))
    return left, top, right, bottom


def pixels(rgb):
    import cv2
    import numpy as np

    yuv = cv2.cvtColor(cv2.resize(rgb, tuple(CONTEXT["scaled_size"]), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2YCrCb)
    softened = cv2.GaussianBlur(yuv.astype(np.float32), (3, 3), 0)
    gray = softened[:, :, 0]
    gradient = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, scale=.25), cv2.Sobel(gray, cv2.CV_32F, 0, 1, scale=.25))
    return softened, gradient


def _face_cut_mask(boxes):
    import numpy as np

    width, height = CONTEXT["scaled_size"]
    y, x = np.meshgrid(np.arange(height) * 360 / height, np.arange(width) * 640 / width, indexing="ij")
    outside = np.ones((height, width), dtype=bool)
    for xx, yy, ww, hh in boxes:
        outside &= (x < xx - .12 * ww) | (x >= xx + 1.12 * ww) | (y < yy - .12 * hh) | (y >= yy + 1.12 * hh)
    return outside


def motion_support(cached, samples, a, b):
    """Ask whether off-face texture and motion are consistent at this adjacent boundary."""
    import numpy as np

    box_a, box_b = samples[a].face.box(), samples[b].face.box()
    bounds = center_region(box_a, box_b)
    x, y, right, bottom = bounds
    mw, mh = CONTEXT["minimum_region"]
    if right - x < mw or bottom - y < mh:
        return None
    region = np.s_[y:bottom, x:right]
    luma_a, gradient_a = cached[a]
    luma_b, gradient_b = cached[b]
    first, second = luma_a[region][:, :, 0], luma_b[region][:, :, 0]
    shared = (gradient_a[region] >= CONTEXT["minimum_gradient"]) & (gradient_b[region] >= CONTEXT["minimum_gradient"])
    points = int(np.count_nonzero(shared))
    groups = int(sum(np.count_nonzero(area) >= CONTEXT["minimum_points_per_group"] for area in np.array_split(shared, 3, axis=1)))
    first, second = first - first.mean(), second - second.mean()
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    similarity = float(np.clip((first * second).sum() / denom, -1, 1)) if denom > 0 else 0.
    off_face = _face_cut_mask((box_a, box_b))
    diff = np.abs(luma_a - luma_b)
    broad = diff[:, :, 0] + .5 * np.maximum(diff[:, :, 1], diff[:, :, 2])
    surroundings = float(np.quantile(broad[off_face], .85))

    def distance(left, right):
        delta = np.abs(cached[left][0][region] - cached[right][0][region])
        combined = delta[:, :, 0] + .5 * np.maximum(delta[:, :, 1], delta[:, :, 2])
        return float(combined.mean())

    change = distance(a, b)
    step = b - a
    neighbors = [(a - k * step, a - (k - 1) * step) for k in (1, 2, 3)]
    neighbors.extend((b + (k - 1) * step, b + k * step) for k in (1, 2, 3))
    values = [distance(left, right) for left, right in neighbors if left in cached and right in cached]
    relative = change / max(1.5, float(np.median(values))) if len(values) >= CONTEXT["minimum_neighbors"] else 999.
    return {"region": list(bounds), "texture_points": points, "texture_groups": groups, "appearance": similarity,
            "surroundings": surroundings, "residual": change, "relative_to_neighbor_motion": relative,
            "neighbor_pairs": len(values)}


def qualify_motion(row):
    return bool(row and row["texture_points"] >= CONTEXT["minimum_gradient_points"]
                and row["texture_groups"] >= CONTEXT["minimum_horizontal_groups"]
                and row["appearance"] >= CONTEXT["minimum_appearance"]
                and row["residual"] <= CONTEXT["maximum_residual"]
                and row["surroundings"] <= CONTEXT["maximum_background"]
                and row["relative_to_neighbor_motion"] <= CONTEXT["maximum_jump_ratio"])


def appearance_candidates(samples):
    """The serving appearance formula before the scene-context gate."""
    import numpy as np
    from app.detectors.video import face_single_boundary as original

    present = [i for i in original.SAMPLE if samples[i].vector is not None]
    if len(present) < 2 * original.FLANK:
        return []
    vectors = np.stack([samples[index].vector for index in present])
    distances = 1 - np.clip(vectors @ vectors.T, -1, 1)
    positions = {index: pos for pos, index in enumerate(present)}

    def within(indices):
        values = distances[np.ix_(indices, indices)]
        return np.quantile(values[np.triu_indices(len(indices), k=1)], .75)

    def separation(before, after):
        return float(np.quantile(distances[np.ix_(before, after)], .25) - max(within(before), within(after)))

    candidates = []
    for edge in range(22, 79, 2):
        near_before = list(range(edge - 2 * original.FLANK, edge, 2))
        near_after = list(range(edge, edge + 2 * original.FLANK, 2))
        if not set(near_before + near_after) <= positions.keys():
            continue
        before = [positions[index] for index in present if index < edge]
        after = [positions[index] for index in present if index >= edge]
        if min(len(before) / (edge // 2), len(after) / ((len(original.SAMPLE) * 2 - edge) // 2)) < original.MINIMUM_PRESENCE:
            continue
        left, right = ([positions[index] for index in group] for group in (near_before, near_after))
        persistent = min(separation(before, after), separation(left, right))
        neighbors = np.r_[distances[left[:-1], left[1:]], distances[right[:-1], right[1:]]]
        bump = float(distances[left[-1], right[0]] - np.quantile(neighbors, .75))
        candidates.append({"coarse": edge, "persistent": round(persistent, 9),
                           "edge_supported": round(min(persistent, bump), 9), "jump_excess": round(bump, 9)})
    return candidates


def edge_readings(samples, native_pixels, cached, fine_frames, a, b):
    import numpy as np
    from app.detectors.image.faces import iou
    from app.detectors.video import face_return
    from app.detectors.video import face_return_fine_scene as fine
    from app.detectors.video import face_single_boundary as original

    left, right = samples[a], samples[b]
    if left.face is None or right.face is None or iou(left.face, right.face) < original.MINIMUM_IOU:
        return {}
    coarse = face_return.scene_lower_quartile(left, right)
    if coarse is not None:
        if coarse < original.MINIMUM_SCENE:
            return {}
        base = {"method": "existing", "score": round(coarse, 9)}
        return dict.fromkeys(ARMS, base)
    coarse_count = int(np.count_nonzero(left.scene[1] & right.scene[1]))
    for index, sample in ((a, left), (b, right)):
        if index not in fine_frames:
            fine_frames[index] = fine.frame(native_pixels[index], sample.face.box())
    refined, extra = fine.edge(fine_frames[a], fine_frames[b], coarse_count)
    result = {}
    if refined is not None:
        result["fine_tiles"] = {"method": "fine_tiles", "score": round(refined, 9), "coarse_tiles": coarse_count, **extra}
    motion = motion_support(cached, samples, a, b)
    if qualify_motion(motion):
        result["off_face_motion"] = {"method": "off_face_motion", "score": round(motion["appearance"], 9),
                                      "coarse_tiles": coarse_count, **motion}
    return result


def refinement(coarse, origins, samples, rgb, cached, method, table, locator, encoder):
    import numpy as np
    from app.detectors.video import face_local_detail

    centers = []
    for region in (range(coarse - 12, coarse - 3, 2), range(coarse + 2, coarse + 12, 2)):
        center = np.mean([origins[index].vector for index in region], axis=0)
        centers.append(center / max(np.linalg.norm(center), 1e-6))

    def input_at(index):
        if index not in samples:
            sample, _, pin, _ = face_local_detail.observe(rgb[index], locator, encoder, with_texture=True)
            samples[index] = sample, pin
        return samples[index]

    visits = tuple(range(coarse - 3, coarse + 3))
    rows = {i: input_at(i)[0] for i in visits}
    proposals = []
    fine_frames = {}
    for edge in range(coarse - 2, coarse + 3):
        left, right = rows[edge - 1], rows[edge]
        if left.vector is None or right.vector is None:
            continue
        context = edge_readings(rows, rgb, cached, fine_frames, edge - 1, edge).get(method)
        if context is None:
            continue
        old, new = centers
        margin = min(float(left.vector @ old - left.vector @ new), float(right.vector @ new - right.vector @ old))
        if margin > 0:
            proposals.append((margin, -abs(edge - coarse), -edge, context))
    if not proposals:
        return None
    margin, _, inverse_edge, context = max(proposals, key=lambda candidate: candidate[:3])
    edge = -inverse_edge
    indices = (edge - 1, edge, coarse + 11)
    pins = [input_at(i)[1] for i in indices]
    if any(pin is None for pin in pins):
        return None
    return {"boundary": edge, "fine_margin": round(margin, 9), "time_base": table.time_base,
            "fine_context": context, "inputs": [
                {"index": index, "pts": table.pts[index], **pin,
                 "eyes": [list(eye) for eye in pin["eyes"]]} for index, pin in zip(indices, pins, strict=True)]}


def score_clip(path, locator, encoder):
    from app.detectors.video import face_local_detail
    from app.detectors.video import face_single_boundary as original
    from app.detectors.video.decode import decode_frames, probe_frames
    from app.detectors.video.frames import probe
    from app.detectors.video.shots import ShotDetector

    table = probe_frames(path)
    delivery = probe(path)
    if not original.supported(path, table, (delivery["codec"], delivery["pix_fmt"])):
        raise ValueError("The input is outside the held native four-second picture profile")
    observer = ShotDetector(table, min_shot_s=.0)
    coarse, native_rgb = {}, {}
    with closing(decode_frames(path, table, None, timeout_s=12)) as frames:
        for index, frame in enumerate(frames):
            if frame.index != index or frame.pts != table.pts[index] or not frame.pts_verified:
                raise ValueError("An unverified timestamp entered evaluation")
            observer.update(index, frame.rgb)
            native_rgb[index] = frame.rgb
            if index in original.SAMPLE:
                coarse[index] = face_local_detail.observe(frame.rgb, locator, encoder, with_texture=False)[0]
    if len(native_rgb) != len(table):
        raise ValueError("One or more original impressions are missing")
    if observer.boundaries:
        return {"scanned": len(table), "cuts": [cut["frame_index"] for cut in observer.boundaries],
                "measured": {name: None for name in ARMS}, "coarse_counts": {name: 0 for name in ARMS}}

    computed, baseline = appearance_candidates(coarse), original.proposals(coarse, lambda: None)
    present = {row["coarse"]: row for row in computed}
    fields = ("coarse", "persistent", "edge_supported", "jump_excess")
    if any({key: row[key] for key in fields} != present[row["coarse"]] for row in baseline):
        raise ValueError("Candidate appearance math no longer matches the current check")
    measured = dict.fromkeys(ARMS)
    baseline_top = max(baseline, key=lambda row: (row["persistent"], row["jump_excess"], -row["coarse"]), default=None)
    if baseline_top is not None:
        witness = original._refine(path, table, coarse, baseline_top, locator, encoder, check=lambda: None, timeout_s=lambda: 12.)
        if witness is not None:
            measured["incumbent"] = {"boundary": witness["boundary"], "score": witness["persistent"],
                                     "coarse": witness["coarse"], "fine_context": {"method": "existing"}}

    if (current := measured["incumbent"]) and current["score"] > original.POINT:
        return {"scanned": len(table), "cuts": [], "measured": dict.fromkeys(ARMS, current),
                "coarse_counts": dict.fromkeys(ARMS, len(baseline))}
    cached = {i: pixels(rgb) for i, rgb in native_rgb.items()}
    fine_frames, proposals, fine_samples = {}, {arm: [] for arm in ARMS[1:]}, {}
    for row in computed:
        candidate = row["coarse"]
        contexts = edge_readings(coarse, native_rgb, cached, fine_frames, candidate - 2, candidate)
        for arm in ARMS[1:]:
            if (context := contexts.get(arm)) is not None:
                proposals[arm].append((row, context))
    for arm, pool in proposals.items():
        if pool:
            (chosen, context) = max(pool, key=lambda item: (item[0]["persistent"], item[0]["jump_excess"], -item[0]["coarse"]))
            # Keep a distinct positive only if the alternate context contributed at the coarse or exact boundary.
            reading = refinement(chosen["coarse"], coarse, fine_samples, native_rgb, cached, arm, table, locator, encoder)
            if reading is not None and arm in (context["method"], reading["fine_context"]["method"]):
                measured[arm] = reading | {"coarse": chosen["coarse"], "score": chosen["persistent"], "coarse_context": context}
    for arm in ARMS[1:]:
        if measured[arm] is None:
            measured[arm] = measured["incumbent"]
    return {"scanned": len(table), "cuts": [], "measured": measured,
            "coarse_counts": {"incumbent": len(baseline), **{arm: len(pool) for arm, pool in proposals.items()}}}


def score(scope, lock_path, output, split, calibration_path=None):
    import cv2
    from app.detectors.image.adapters.loaders import load_yunet
    from app.detectors.video import face_return_fine_scene as fine
    from app.detectors.video import face_single_boundary as original
    from app.detectors.video.adapters.loaders import load_face_appearance
    from scripts.video_face_local_measure import ENCODER, LOCATOR, assets

    lock = read(lock_path)
    if (lock["reviewer_sha256"] != sha(Path(__file__)) or lock["manifest_sha256"] != sha(scope)
            or lock["baseline_sha256"] != sha(Path(original.__file__)) or lock["fine_sha256"] != sha(Path(fine.__file__))
            or lock["licensed_weights"] != assets()):
        raise ValueError("Locked inputs, code or assets differ")
    calibration = read(calibration_path) if calibration_path is not None else None
    if split == "held" and (calibration is None or calibration["plan_sha256"] != sha(lock_path)
            or set(lock["groups"]["held"]).intersection(calibration["source_groups"])):
        raise ValueError("Freeze the disjoint development operating point before held scoring")
    cv2.setNumThreads(1)
    locator = load_yunet(LOCATOR, "cpu", "float32").obj
    encoder = load_face_appearance(ENCODER, "cpu", "float32").obj
    data = read(scope)
    with output.open("x") as out:
        out.write(json.dumps({"scope_sha256": sha(scope), "lock_sha256": sha(lock_path), "split": split,
                              "calibration_sha256": None if calibration_path is None else sha(calibration_path)}) + "\n")
        for i, row in enumerate((x for x in data["rows"] if x["split"] == split), 1):
            path = Path(row["path"])
            if sha(path) != row["sha256"]:
                raise ValueError("A submitted file changed since freezing")
            result = score_clip(path, locator, encoder)
            out.write(json.dumps({key: value for key, value in row.items() if key != "path"} | result, allow_nan=False) + "\n")
            out.flush()
            print(split, i, row["group"], row["role"], row["expected_frame"],
                  {key: (value["boundary"], round(value["score"], 3)) if value else None
                   for key, value in result["measured"].items()}, flush=True)


def read_journal(path, lock_path):
    with path.open() as stream:
        header = json.loads(next(stream))
        if header["lock_sha256"] != sha(lock_path):
            raise ValueError("Score rows refer to a different freeze")
        rows = [json.loads(line) for line in stream]
    return header, rows


def is_motion_candidate(measured):
    return measured is not None and any((measured.get(key) or {}).get("method") == "off_face_motion"
                                        for key in ("coarse_context", "fine_context"))


def calibrate(path, lock_path, output):
    header, rows = read_journal(path, lock_path)
    lock = read(lock_path)
    if header["split"] != "development":
        raise ValueError("Only development inputs can set a point")
    groups = {}
    for row in rows:
        if row["expected_frame"] is not None:
            continue
        candidate = row["measured"]["off_face_motion"]
        if is_motion_candidate(candidate):
            group = row["group"]
            groups[group] = max(candidate["score"], groups.get(group, -2))
    point = None
    if len(groups) >= lock["calibration"]["minimum_ordinary_groups"]:
        order = sorted(groups.values(), reverse=True)
        point = max(lock["calibration"]["floor"], order[lock["calibration"]["allowed_development_groups"]])
    details = {"plan_sha256": sha(lock_path), "scored_development_sha256": sha(path),
               "point": point, "group_maxima": groups, "rule": lock["calibration"],
               "source_groups": sorted({row["group"] for row in rows})}
    write(output, details)
    print(json.dumps(details, sort_keys=True), flush=True)


def selected(row, arm, point, alternative_point):
    readings = row["measured"]
    if arm not in {CALIBRATED, FLOOR}:
        measure = readings[arm]
    else:
        measure = readings["off_face_motion"]
        if is_motion_candidate(measure):
            point = CALIBRATION["floor"] if arm == FLOOR else alternative_point
    return measure, measure["boundary"] if measure and measure["score"] > point else None


def review(path, lock_path, calibration_path=None):
    lock = read(lock_path)
    header, rows = read_journal(path, lock_path)
    point = lock["point"]
    alternative_point = None
    if calibration_path is not None:
        calibration = read(calibration_path)
        if calibration["plan_sha256"] != sha(lock_path) or calibration["rule"] != lock["calibration"]:
            raise ValueError("The operating point refers to another freeze")
        if header["split"] == "held" and {row["group"] for row in rows}.intersection(calibration["source_groups"]):
            raise ValueError("Held scoring shares a development recording group")
        alternative_point = calibration["point"]
    if header["split"] == "held" and (calibration_path is None or header.get("calibration_sha256") != sha(calibration_path)):
        raise ValueError("Holdout scoring predates this calibration")
    by_profile = defaultdict(list)
    for row in rows:
        by_profile[row.get("profile", "native")].append(row)
    result = {"rows": len(rows), "calibrated_point": alternative_point,
              "outcomes": summarize(rows, point, alternative_point),
              "by_profile": {profile: summarize(group_rows, point, alternative_point)
                             for profile, group_rows in sorted(by_profile.items())}}
    print(json.dumps(result, indent=2, sort_keys=True))


def summarize(rows, point, alternative_point):
    outcomes = {}
    arms = (*ARMS, FLOOR) if alternative_point is None else (*ARMS, FLOOR, CALIBRATED)
    for arm in arms:
        counts, located_sources, ordinary_sources = Counter(), set(), set()
        for row in rows:
            truth = row["expected_frame"]
            measure, guessed = selected(row, arm, point, alternative_point)
            _, before = selected(row, "incumbent", point, None)
            positive = truth is not None
            good = positive and guessed is not None and abs(guessed - truth) <= ADMISSION["boundary_tolerance_pictures"]
            prior_good = positive and before is not None and abs(before - truth) <= ADMISSION["boundary_tolerance_pictures"]
            added = good and not prior_good
            ordinary = not positive and guessed is not None
            new_ordinary = ordinary and before is None
            counts.update(positives=positive, ordinary=not positive, located=good, misplaced=positive and guessed is not None and not good,
                          inconclusive=measure is None, alerts=ordinary, new_locations=added, lost_locations=prior_good and not good,
                          new_ordinary_alerts=new_ordinary, new_cut_alerts=new_ordinary and "cut" in row["role"],
                          added_misplaced=positive and guessed is not None and not good and before != guessed)
            if added:
                located_sources.add(row["group"])
            if new_ordinary:
                ordinary_sources.add(row["group"])
        outcome = dict(counts, added_sources=len(located_sources), new_ordinary_sources=len(ordinary_sources))
        if arm != "incumbent":
            outcome["pass"] = (outcome["new_locations"] >= ADMISSION["minimum_added_localizations"]
                and outcome["added_sources"] >= ADMISSION["minimum_added_sources"] and outcome["lost_locations"] == 0
                and outcome["new_ordinary_alerts"] <= ADMISSION["maximum_added_ordinary_alerts"]
                and outcome["new_ordinary_sources"] <= ADMISSION["maximum_added_ordinary_sources"]
                and outcome["new_cut_alerts"] <= ADMISSION["maximum_added_cut_alerts"]
                and outcome["added_misplaced"] <= ADMISSION["maximum_added_misplaced_edits"])
        outcomes[arm] = outcome
    return outcomes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("freeze")
    lock.add_argument("manifest", type=Path)
    lock.add_argument("output", type=Path)
    run = commands.add_parser("score")
    run.add_argument("manifest", type=Path)
    run.add_argument("lock", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--split", choices=["development", "held"], required=True)
    run.add_argument("--calibration", type=Path)
    fit = commands.add_parser("calibrate")
    fit.add_argument("journal", type=Path)
    fit.add_argument("lock", type=Path)
    fit.add_argument("output", type=Path)
    check = commands.add_parser("review")
    check.add_argument("journal", type=Path)
    check.add_argument("lock", type=Path)
    check.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.manifest, args.output)
    elif args.command == "score":
        score(args.manifest, args.lock, args.output, args.split, args.calibration)
    elif args.command == "calibrate":
        calibrate(args.journal, args.lock, args.output)
    else:
        review(args.journal, args.lock, args.calibration)
