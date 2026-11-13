# SPDX-License-Identifier: Apache-2.0
"""Test long point-trajectory agreement in a single uploaded short video."""

from __future__ import annotations

import argparse
import subprocess
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from app.detectors.video.shots import ShotDetector
from scripts import video_mouth_contour_phase as prior
from scripts.video_common_sources_v89 import read, require, sha, write

ARMS = ("trajectory_loss", "cycle_separation")
FLOOR = (0.08, 0.10)
HOPS = (3, 4)
SPANS = (8, 12, 16, 20)
FLANK = 5
SPEC = {
    "signal": "Compare local point track survival and position disagreement between direct and adjacent routes over 3 and 4 native pictures. Rigid eye normalization, seed points from tracked mouth and cheek regions, match normalized corner strength and reject globally unreliable comparisons. No pixel-energy evidence, audio, earlier copy, source timing markers or rendered support enters measurement.",
    "tracks": "Greedy one-to-one matching of strongest mouth corners to cheek corners within a 4x strength ratio. Require three matched starts. Track with 9-pixel Lucas–Kanade, one pyramid level, and a fixed 1.25-pixel direct round-trip tolerance. Cycle arm includes pairs with all routes valid; adjacent chains and direct paths are independently computed.",
    "extent": "Search 8-, 12-, 16- and 20-picture cores inside one verified shot. Both five-picture flanks and the core need at least 60% measured entries per horizon. Use minimum excess over the two flanks, then require both horizons to support a positive excess. Keep the strongest interval, with exact before, middle and first-after source frames/crops.",
    "ordinary_derivatives": "Six fresh speaker groups outside MuseTalk holdout. Add two 16-picture ordinary crop controls per group using fixed same-time subpixel resampling and a same-time local two-picture temporal blend with unchanged outer pixels. Encode native RGB losslessly, preserve presentation times, and keep all derived files in held only.",
    "point": "Third-largest ordinary GRID development maximum per arm, with fixed floors 0.08 and 0.10 respectively, and strict exceedance. A useful location uses native half-open interval IoU >= 0.50.",
    "gate": "At least four strict LivePortrait locations across three groups and the strict independent MuseTalk location, at least 70% measurement coverage on each main class, at most two ordinary alerts, none on untouched, hard-cut or recompressed deliveries. One strict MuseTalk case is descriptive, not a stable rate across generators.",
}


def freeze(lombard, fresh, muse):
    data = prior.plan(lombard, fresh)
    m_lock, m_audit, m_render = (muse / (name + ".json") for name in ("freeze", "audit", "render"))
    audited, construction = read(m_audit), read(m_render)
    require(
        audited["freeze_sha256"] == construction["freeze_sha256"] == sha(m_lock)
        and audited["render_sha256"] == sha(m_render),
        "Fresh neural native source audit differs",
    )
    source = read(fresh / "source-receipt.json")
    newer = {r["source"]["speaker"] for r in read(m_lock)["sources"]}
    older = {r["source"]["speaker"] for r in read(lombard / "freeze.json")["sources"]}
    require(not newer & older and len(newer) == 6, "Neural source groups overlap")
    audits = {r["source_ordinal"]: r for r in audited["rows"]}
    for entry in construction["cases"]:
        verified = audits[entry["ordinal"]]
        role = entry["role"]
        neural = role in {"same_time_neural", "earlier_sound_neural"}
        require(
            entry["status"] == "authored" and verified["roles"][role]["native_pictures"] > 0,
            "Incomplete neural held delivery",
        )
        data["rows"].append(
            {
                "split": "held",
                "family": "musetalk",
                "id": entry["ordinal"],
                "group": verified["group"],
                "role": role,
                "file": str(muse / "deliveries" / f"{entry['token']}.mov"),
                "sha": entry["file_sha256"],
                "native_truth": verified["edit"] if neural else None,
                "aligned": bool(neural and role == "same_time_neural" and verified["qualifies_local_event"]),
                "source_event_retained": bool(
                    neural
                    and role == "same_time_neural"
                    and verified["roles"][role]["status"] == "source_event_retained"
                ),
            }
        )
    speaker_by_id = {r["ordinal"]: r["speaker"] for r in source["rows"]}
    controls = {}
    for row in data["rows"]:
        if row["family"] == "fresh" and speaker_by_id[row["id"]] not in newer:
            controls.setdefault(row["group"], row)
    require(
        len(controls) == 6
        and sum(r.get("aligned", False) for r in data["rows"] if r["family"] == "musetalk") == 1,
        "Expected six novel ordinary groups and one strict independent second-generator case",
    )
    data.update(
        protocol=SPEC,
        tool_sha256=sha(Path(__file__)),
        input_plan_sha256=sha(Path(prior.__file__)),
        geometry_code_sha256=sha(Path(prior.panel.prior.__file__)),
        muse_audit_sha256=sha(m_audit),
        control_sources=list(controls.values()),
    )
    return data


YY, XX = np.mgrid[:96, :96]
LK = {
    "winSize": (9, 9),
    "maxLevel": 1,
    "criteria": (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 16, 0.025),
    "minEigThreshold": 0.0001,
}


def seeds(gray, mouth):
    dx, dy = XX - mouth[0], YY - mouth[1]
    center = ((dx / 15) ** 2 + (dy / 8) ** 2) <= 1
    cheek = (((dx / 32) ** 2 + (dy / 19) ** 2) <= 1) & (np.abs(dx) >= 17) & (dy <= 10)
    safe = (XX >= 10) & (XX < 86) & (YY >= 10) & (YY < 86)
    strengths = cv2.cornerMinEigenVal(gray, 5)
    entries = []
    for region in (center, cheek):
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=28,
            qualityLevel=0.02,
            minDistance=2.5,
            mask=(region & safe).astype(np.uint8),
            blockSize=5,
        )
        if points is None:
            return None
        coords = np.rint(points[:, 0]).astype(int)
        values = strengths[coords[:, 1], coords[:, 0]]
        entries.append((points, values))
    mouth_pts, mouth_strength = entries[0]
    cheek_pts, cheek_strength = entries[1]
    available = np.ones(len(cheek_pts), dtype=bool)
    left, right = [], []
    for i, strength in enumerate(mouth_strength):
        if strength <= 0:
            continue
        disparity = np.abs(np.log(np.maximum(1e-9, cheek_strength) / strength))
        j = int(np.argmin(np.where(available, disparity, np.inf)))
        if available[j] and disparity[j] <= np.log(4):
            left.append(mouth_pts[i])
            right.append(cheek_pts[j])
            available[j] = False
    return np.asarray([*left, *right], dtype=np.float32) if len(left) >= 3 else None


def move(a, b, points):
    target, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, **LK)
    valid = ok[:, 0].astype(bool) & ((target[:, 0] >= 4) & (target[:, 0] <= 91)).all(axis=1)
    return np.where(valid[:, None, None], target, points), valid


def measure_cycle(frames, points):
    direct, found = move(frames[0], frames[-1], points)
    reverse, returned = move(frames[-1], frames[0], direct)
    chain = points
    sequential = np.ones(len(points), dtype=bool)
    for first, second in pairwise(frames):
        chain, step = move(first, second, chain)
        sequential &= step
    rolled = chain
    for first, second in pairwise(reversed(frames)):
        rolled, step = move(first, second, rolled)
        sequential &= step
    roundtrip = np.linalg.norm((reverse - points)[:, 0], axis=1)
    chained_roundtrip = np.linalg.norm((rolled - points)[:, 0], axis=1)
    endpoint = np.linalg.norm((direct - chain)[:, 0], axis=1)
    good = found & returned & sequential & (roundtrip <= 1.25) & (chained_roundtrip <= 1.25)
    n = len(points) // 2
    if good[n:].sum() * 2 < n:
        return [np.nan, np.nan]
    lost = float(good[n:].mean() - good[:n].mean())
    valid = (found & returned & sequential).reshape(2, n).all(axis=0)
    disagreement = np.maximum.reduce((roundtrip, chained_roundtrip, endpoint))
    paired = disagreement[:n] - disagreement[n:]
    return [lost, float(np.median(np.clip(paired[valid], -4, 4))) if valid.sum() >= 3 else np.nan]


def inspect(path, detector, cache):
    table = probe_frames(path, timeout_s=15)
    require(
        not (table.truncated or table.interpolated or table.rotation) and 40 <= len(table) <= 100,
        "Expected short native picture table",
    )
    shots, inputs, points, anchors = [], [], [], []
    track = ShotDetector(table)
    for frame in decode_frames(path, table, None, timeout_s=28):
        require(
            frame.index == len(shots) and frame.pts_verified and frame.pts == table.pts[frame.index],
            "Picture clock differs",
        )
        track.update(frame.index, frame.rgb)
        shots.append(len(track.boundaries))
        digest = prior.panel.raw_digest(frame.rgb)
        if digest not in cache:
            geometry = prior.panel.prior.aligned_face(frame.rgb, detector)
            if geometry is None:
                cache[digest] = None, None, None
            else:
                gray, _, mouth, _ = geometry
                x0, y0, x1, y1 = prior.panel.native_bounds(geometry)
                cache[digest] = (
                    gray,
                    seeds(gray, mouth),
                    ([x0, y0, x1, y1], prior.panel.raw_digest(frame.rgb[y0:y1, x0:x1])),
                )
        gray, seed, proof = cache[digest]
        inputs.append(gray)
        points.append(seed)
        anchors.append(
            None
            if proof is None
            else {
                "frame": frame.index,
                "pts": frame.pts,
                "rgb_sha256": digest,
                "crop_xyxy": proof[0],
                "crop_sha256": proof[1],
            }
        )
    require(len(shots) == len(table), "Incomplete native source pictures")
    data = np.full((len(table), len(HOPS), len(ARMS)), np.nan)
    for column, horizon in enumerate(HOPS):
        for a in range(len(table) - horizon):
            end = a + horizon
            frames = inputs[a : end + 1]
            if points[a] is not None and shots[a] == shots[end] and all(f is not None for f in frames):
                data[a + horizon // 2, column] = measure_cycle(frames, points[a])
    return table, shots, data, anchors


def select(table, shots, measured, anchors):
    candidates = dict.fromkeys(ARMS)
    assessed = dict.fromkeys(ARMS, False)
    windows = {}
    for width in (FLANK, *SPANS):
        parts = np.lib.stride_tricks.sliding_window_view(measured, width, axis=0)
        good = np.isfinite(parts).sum(axis=-1) * 5 >= width * 3
        median = np.ma.median(np.ma.masked_invalid(parts), axis=-1).filled(np.nan)
        windows[width] = np.where(good, median, np.nan)
    for width in SPANS:
        for a in range(FLANK, len(table) - width - FLANK + 1):
            b = a + width
            chosen = [anchors[i] for i in (a - 1, (a + b) // 2, b)]
            if shots[a - FLANK] != shots[b + FLANK - 1] or not all(chosen):
                continue
            excess = windows[width][a] - np.maximum(windows[FLANK][a - FLANK], windows[FLANK][b])
            for i, arm in enumerate(ARMS):
                if not np.isfinite(excess[:, i]).all():
                    continue
                assessed[arm] = True
                strength = float(np.min(excess[:, i]))
                if strength <= 0 or candidates[arm] and strength <= candidates[arm]["score"]:
                    continue
                candidates[arm] = {
                    "score": strength,
                    "frames": [a, b],
                    "seconds": [str(table.pts[j] * table.tb) for j in (a, b)],
                    "time_base": table.time_base,
                    "anchors": chosen,
                    "horizon_responses": excess[:, i].tolist(),
                }
    return {arm: {"measured": assessed[arm], "proposal": candidates[arm]} for arm in ARMS}


def derived_picture(sequence, i, span, geometry, mode):
    start, end = span
    rgb = sequence[i]
    if not start <= i < end or geometry is None:
        return rgb
    _, matrix, mouth, _ = geometry
    x0, y0, x1, y1 = prior.panel.native_bounds(geometry)
    y, x = np.mgrid[y0:y1, x0:x1]
    relative = np.stack((x, y), axis=-1) @ matrix[:, :2].T + matrix[:, 2] - mouth
    alpha = (
        0.45
        * min(2, i - start + 1, end - i)
        / 2
        * np.clip((1 - np.linalg.norm(relative / [23, 13], axis=-1)) * 2, 0, 1)
    )[..., None]
    if mode == "ordinary_subpixel_resample":
        translated = np.float32([[1, 0, 0.65], [0, 1, -0.65]])
        patch = rgb[y0:y1, x0:x1]
        shifted = cv2.warpAffine(
            patch,
            translated,
            (patch.shape[1], patch.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        target = cv2.warpAffine(
            shifted,
            cv2.invertAffineTransform(translated),
            (patch.shape[1], patch.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    else:
        target = cv2.addWeighted(sequence[i - 1][y0:y1, x0:x1], 0.5, sequence[i + 1][y0:y1, x0:x1], 0.5, 0)
    output = rgb.copy()
    output[y0:y1, x0:x1] = np.rint(rgb[y0:y1, x0:x1] * (1 - alpha) + target * alpha).astype(np.uint8)
    return output


def prepare(lock, run):
    detector = load_yunet(prior.WEIGHT)
    folder = run / "controls"
    folder.mkdir(mode=0o700)
    cases = []
    for row in lock["control_sources"]:
        path = Path(row["file"])
        require(sha(path) == row["sha"], "Frozen source differs")
        table = probe_frames(path, timeout_s=15)
        times = [p * table.tb for p in table.pts]
        steps = {b - a for a, b in pairwise(times)}
        require(
            table.tb.numerator == 1 and len(steps) == 1 and times[0] == 0,
            "Control needs a complete native clock",
        )
        frames = []
        for frame in decode_frames(path, table, None, timeout_s=30):
            require(
                frame.index == len(frames) and frame.pts_verified and frame.pts == table.pts[frame.index],
                "Original clock differs",
            )
            frames.append(frame.rgb)
        span = [len(frames) // 2 - 8, len(frames) // 2 + 8]
        geometry = {i: prior.panel.prior.aligned_face(frames[i], detector) for i in range(*span)}
        for role in ("ordinary_subpixel_resample", "ordinary_temporal_blend"):
            rendered = [derived_picture(frames, i, span, geometry.get(i), role) for i in range(len(frames))]
            changed = [
                i for i, (a, b) in enumerate(zip(frames, rendered, strict=True)) if not np.array_equal(a, b)
            ]
            out = folder / f"{row['id']:02d}-{role}.mov"
            argv = [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-n",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{table.width}x{table.height}",
                "-framerate",
                str(1 / steps.copy().pop()),
                "-i",
                "pipe:0",
                "-c:v",
                "libx264rgb",
                "-qp",
                "0",
                "-bf",
                "0",
                "-preset",
                "veryfast",
                "-threads:v",
                "1",
                "-video_track_timescale",
                str(table.tb.denominator),
                str(out),
            ]
            subprocess.run(
                argv, input=b"".join(frame.tobytes() for frame in rendered), check=True, timeout=50
            )
            check = probe_frames(out, timeout_s=15)
            require(
                [p * check.tb for p in check.pts] == times, "Derived ordinary delivery alters source times"
            )
            for frame in decode_frames(out, check, None, timeout_s=25):
                require(
                    frame.pts_verified and np.array_equal(frame.rgb, rendered[frame.index]),
                    "Derived pixel replay differs",
                )
            cases.append(
                {
                    **row,
                    "family": "derived_ordinary",
                    "role": role,
                    "file": str(out),
                    "sha": sha(out),
                    "source_sha": row["sha"],
                    "authored_control_frames": span,
                    "changed_pictures": changed,
                    "verified_pictures": len(frames),
                }
            )
        print("controls", row["id"], flush=True)
    write(run / "controls.json", {"freeze_sha256": sha(run / "freeze.json"), "rows": cases})


def points_for(rows):
    points = {}
    for arm, floor in zip(ARMS, FLOOR, strict=True):
        values = sorted(
            (r["arms"][arm]["proposal"]["score"] for r in rows if r["arms"][arm]["proposal"]), reverse=True
        )
        points[arm] = max(floor, values[2] if len(values) >= 3 else 0)
    return points


def score(lock, run, phase):
    roster, groups = lock["rows"], {}
    if phase == "held":
        control = read(run / "controls.json")
        require(
            control["freeze_sha256"] == sha(run / "freeze.json") and len(control["rows"]) == 12,
            "Held derived roster differs",
        )
        roster = [*roster, *control["rows"]]
    for row in roster:
        if row["split"] == phase:
            groups.setdefault((row["family"], row["id"]), []).append(row)
    detector = load_yunet(prior.WEIGHT)
    all_rows = []
    for entries in groups.values():
        cache = {}
        for row in entries:
            path = Path(row["file"])
            require(sha(path) == row["sha"], "Frozen video bytes differ")
            inputs = inspect(path, detector, cache)
            table = inputs[0]
            truth = (
                [str(table.pts[i] * table.tb) for i in row["edit"]]
                if row.get("edit")
                else row.get("native_truth")
            )
            all_rows.append(
                {
                    **row,
                    "arms": select(*inputs),
                    "native_frames": len(table),
                    "geometry_frames": sum(x is not None for x in inputs[3]),
                    "edit_truth": truth,
                    "supported_cycles": np.isfinite(inputs[2]).sum(axis=0).tolist(),
                }
            )
        print(phase, entries[0]["family"], entries[0]["id"], len(all_rows), flush=True)
    path = run / (phase + ".json")
    write(
        path,
        {
            "freeze_sha256": sha(run / "freeze.json"),
            "derived_controls_sha256": sha(run / "controls.json") if phase == "held" else None,
            "rows": all_rows,
        },
    )
    if phase == "development":
        write(run / "points.json", {"development_sha256": sha(path), "points": points_for(all_rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "development", "held"))
    for name in ("run", "lombard", "fresh", "muse"):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    lock = freeze(a.lombard, a.fresh, a.muse)
    path = a.run / "freeze.json"
    if a.phase == "freeze":
        a.run.mkdir(mode=0o700)
        write(path, lock)
        return
    require(read(path) == lock, "Frozen inputs, runtime or method differ")
    cv2.setNumThreads(1)
    if a.phase == "prepare":
        prepare(lock, a.run)
        return
    if a.phase == "held":
        dev = a.run / "development.json"
        require(
            read(a.run / "points.json")
            == {"development_sha256": sha(dev), "points": points_for(read(dev)["rows"])},
            "Held points were not frozen from development",
        )
    score(lock, a.run, a.phase)


if __name__ == "__main__":
    main()
