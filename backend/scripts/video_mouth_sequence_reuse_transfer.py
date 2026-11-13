# SPDX-License-Identifier: Apache-2.0
"""Compare two separated native intervals for repeated moving mouth pixels."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.audio.decode import decode_audio, wav_bytes
from app.detectors.image.faces import detect_faces, load_yunet
from app.detectors.video import lipsync, mouth_local_hold
from scripts import video_lipsync_local_delay_transfer as clock
from scripts import video_lipsync_short_bracket as current
from scripts import video_mouth_local_hold_panel as sources
from scripts.video_reference_soundtrack_panel import digest, read, require, save

N, SR, HOP = 50, 16_000, 640
POSITIVES = ("mouth_later", "mouth_earlier", "mouth_later_tandem")
CONTROLS = ("untouched", "tandem", "whole_sequence", "mouth_hold", "gain", "matched_picture_audio")
POINTS = (0.28, 0.38, 0.48, 0.58)
CONTRACT = {
    "ranges": "Both disjoint half-open source ranges and both native endpoints. Search moving repeats of 7–12 frames, separated by 9–32 frames, with two observed flank pictures.",
    "test": "Compare brightness-centred matching mouth pixels and the adjacent temporal difference sequence. Require moving texture, distinct nearby face regions, and distinct concurrent active speech in both intervals.",
    "development_point": "Maximize correct source-pair locations with at most one control flag, then prefer stricter point.",
    "iou": "Each of two ranges must have at least 3/5 intersection-over-union against its source span, and match the known frame lag exactly.",
    "scope": "Fixed-mouth crop in single-face 360x288 or 480x360, H.264 yuv420p, 50 frames at 25fps, 16kHz continuous audio.",
    "gate": {
        "min_correct_per_bank": 8,
        "min_groups_per_bank": 4,
        "max_controls_per_bank": 1,
        "min_coverage": 0.75,
        "max_whole_or_matched_av_controls": 0,
    },
}


def pcm_sha(pcm):
    return hashlib.sha256(np.asarray(pcm, dtype="<i2").tobytes()).hexdigest()


def roster():
    grid = [
        dict(bank="grid", group=r["group"], split=r["split"], path=r["path"], sha256=r["source_sha256"])
        for r in read(sources.GRID / "freeze.json")["roster"]
        if r["id"].endswith("P-1")
    ]
    archive = read(sources.CREMA / "intake-receipt.json")["rows"]
    require(len({r["actor"] for r in archive}) == len(archive) == 15, "CREMA-D actor roster differs")
    crema = [
        dict(
            bank="crema",
            group=r["ordinal"],
            split=r["split"],
            path=str((sources.CREMA / r["file"]).relative_to(sources.ROOT)),
            sha256=r["sha256"],
        )
        for r in archive
        if r["ordinal"] not in {8, 10, 13}
    ]
    require(len(grid) == 8 and len(crema) == 12, "Frozen source roster is incomplete")
    return grid + crema


def contract():
    return {
        "contract": CONTRACT,
        "candidate_points": list(POINTS),
        "positive": list(POSITIVES),
        "controls": list(CONTROLS),
        "sources": roster(),
        "opencv": cv2.__version__,
        "bindings": {
            p.name: digest(p)
            for p in (
                Path(__file__),
                Path(sources.__file__),
                Path(current.__file__),
                Path(clock.__file__),
                Path(mouth_local_hold.__file__),
                sources.WEIGHTS,
                sources.GRID / "freeze.json",
                sources.CREMA / "intake-receipt.json",
            )
        },
    }


def original(row, detector):
    frames, source_pts = sources.source(row)
    path = sources.ROOT / row["path"]
    audio = decode_audio(path, source_sha256=row["sha256"], max_seconds=8)
    first, pcm, _ = clock.resample_clock(audio)
    table = sources.probe_frames(path, timeout_s=10)
    offset = round(Fraction(table.pts[0]) * table.tb * SR) - first
    require(
        offset >= -16 and offset + N * HOP <= len(pcm), "The video interval lacks synchronized source audio"
    )
    pcm = np.pad(pcm, (max(0, -offset), 0))[max(0, offset) : max(0, offset) + N * HOP]
    geometry = mouth_local_hold.stable_geometry(frames, detector)
    require(geometry is not None, "The source roster requires a visible stable single mouth")
    width, height = round(geometry.width * 1.52), round(geometry.width * 1.10)
    x, y = round(geometry.x - width / 2), round(geometry.y - height / 2)
    h, w = frames[0].shape[:2]
    require(0 <= x < x + width <= w and 0 <= y < y + height <= h, "Source patch lies outside picture")
    return frames, source_pts, pcm, [x, y, width, height]


def paste(frames, first, last, box, *, full=False, held=False):
    start, end = last
    left, _ = first
    output = frames.copy()
    if full:
        output[start:end] = frames[left : left + end - start]
    else:
        x, y, w, h = box
        col, row = np.arange(w), np.arange(h)
        edge = np.minimum(np.minimum(col + 1, w - col)[None, :], np.minimum(row + 1, h - row)[:, None])
        alpha = np.clip(edge / 4, 0, 1)[..., None]
        for i in range(end - start):
            edited = frames[start + i].copy()
            target = edited[y : y + h, x : x + w]
            donor = frames[left if held else left + i][y : y + h, x : x + w]
            target[:] = np.rint((1 - alpha) * target + alpha * donor).astype(np.uint8)
            output[start + i] = edited
    require(
        min(np.count_nonzero(np.any(frames[i] != output[i], axis=2)) for i in range(start, end)) >= 8,
        "Expected actual changed pictures",
    )
    return output


def author(frames, pcm, box, group):
    length = 8 + 2 * (group % 2)
    a = 10 + group % 3
    b = a + 16 + 2 * (group % 4)
    early, late = [a, a + length], [b, b + length]
    later = paste(frames, early, late, box)
    earlier = paste(frames, late, early, box)
    yield "untouched", frames, pcm, None, None
    yield "tandem", frames, pcm, None, None
    yield "mouth_later", later, pcm, early, late
    yield "mouth_earlier", earlier, pcm, late, early
    yield "mouth_later_tandem", later, pcm, early, late
    yield "whole_sequence", paste(frames, early, late, box, full=True), pcm, None, None
    yield "mouth_hold", paste(frames, early, late, box, held=True), pcm, None, None
    wave = pcm.copy()
    wave[b * HOP : (b + length) * HOP] = np.rint(pcm[b * HOP : (b + length) * HOP] * 0.64).astype("<i2")
    yield "gain", frames, wave, None, None
    wave = pcm.copy()
    wave[b * HOP : (b + length) * HOP] = pcm[a * HOP : (a + length) * HOP]
    yield "matched_picture_audio", later, wave, None, None


def mouth_boxes(geometry, shape):
    h, w = shape[:2]
    width, height = max(8, round(0.73 * geometry.width)), max(8, round(0.47 * geometry.width))
    move = max(2, round(0.11 * geometry.width))
    return [
        (x, y, width, height)
        for j in (-move, 0, move)
        for i in (-move, 0, move)
        if 0 <= (x := round(geometry.x - width / 2) + i) <= w - width
        and 0 <= (y := round(geometry.y - height / 2) + j) <= h - height
    ]


def crop(gray, rect):
    x, y, w, h = rect
    return gray[:, y : y + h, x : x + w].reshape(len(gray), -1)


def centered(crop_pixels):
    return crop_pixels - crop_pixels.mean(axis=1, keepdims=True)


def differences(features):
    return np.mean(np.abs(features[:, None, :] - features[None, :, :]), axis=2)


def pairs(gray, pcm, geometry):
    regions = mouth_local_hold.context_regions(geometry, gray.shape[1:])
    if len(regions) < 3:
        return []
    outer = np.stack([differences(centered(crop(gray, r))) for r in regions])
    context = np.sort(outer, axis=0)[-2]
    rms = np.sqrt(np.mean((pcm.astype(np.float64).reshape(N, HOP) / 32768.0) ** 2, axis=1))
    active = rms >= max(0.0018, float(np.percentile(rms, 90)) * 0.12)
    eligible = []
    wave = pcm.astype(float)
    for lag in range(9, 33):
        for a in range(2, N - lag - 8):
            b = a + lag
            for length in range(7, min(12, lag, N - 2 - b) + 1):
                if (
                    min(np.count_nonzero(active[a : a + length]), np.count_nonzero(active[b : b + length]))
                    < 3
                ):
                    continue
                t = np.arange(length)
                i, j = a + t, b + t
                neighbor = context[i, j]
                if np.median(neighbor) < 1.7 or np.count_nonzero(neighbor > 1.7) < length * 0.60:
                    continue
                if (
                    abs(
                        lipsync._pearson(
                            wave[a * HOP : (a + length) * HOP], wave[b * HOP : (b + length) * HOP]
                        )
                    )
                    >= 0.35
                ):
                    continue
                eligible.append((a, b, length, i, j, float(np.median(neighbor))))
    hypotheses = []
    for rect in mouth_boxes(geometry, gray.shape[1:]):
        m = centered(crop(gray, rect))
        diff = differences(m)
        movement = np.mean(np.abs(np.diff(m, axis=0)), axis=1)
        texture = m.std(axis=1)
        vectors = np.diff(m, axis=0)
        for a, b, length, i, j, neighbor in eligible:
            match = diff[i, j]
            if np.median(match) > 3.1 or np.quantile(match, 0.8) > 4.8:
                continue
            move = np.minimum(movement[i[:-1]], movement[j[:-1]])
            median_move = float(np.median(move))
            if (
                median_move < 1.6
                or np.count_nonzero(move >= 2.0) < 3
                or min(np.median(texture[i]), np.median(texture[j])) < 9
            ):
                continue
            temporal = np.mean(np.abs(vectors[i[:-1]] - vectors[j[:-1]]), axis=1)
            error = float(
                max(np.median(match) / (median_move + 1.0), np.median(temporal) / (median_move + 1.0) * 0.78)
            )
            if error <= max(POINTS):
                hypotheses.append(
                    {
                        "ranges": [[a, a + length], [b, b + length]],
                        "lag": b - a,
                        "box": list(rect),
                        "error": error,
                        "context_difference": neighbor,
                        "mouth_movement": median_move,
                    }
                )
    return hypotheses


def best(choices, point):
    return min(
        (r for r in choices if r["error"] <= point),
        key=lambda r: (-(r["ranges"][0][1] - r["ranges"][0][0]), r["error"], r["ranges"], r["box"]),
        default=None,
    )


def incumbent(frames, pcm, detector):
    hold_status, hold = mouth_local_hold.inspect(frames, detector)
    faces, patches = {}, {}
    for i, rgb in enumerate(frames):
        observed = detect_faces(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
        if len(observed) == 1:
            face = observed[0]
            faces[i] = {
                "frame_index": i,
                "bbox": dict(zip(("x", "y", "w", "h"), face.box(), strict=True)),
                "landmarks": face.landmarks,
            }
            patches[i] = lipsync.mouth_patch(rgb, faces[i])
    grid = lipsync.TickGrid(0.0, tuple(range(N)), N)
    return hold_status, hold, current.current(faces, patches, pcm.astype(np.float32) / 32768.0, grid)


def deliver(folder, path, frames, pcm, role):
    if role in {"tandem", "mouth_later_tandem"}:
        _, frames = mouth_local_hold.native(
            folder / ("untouched.mov" if role == "tandem" else "mouth_later.mov")
        )
    base, wave = folder / "current.mp4", folder / "current.wav"
    sources.encode(base, frames, 34 if role in {"tandem", "mouth_later_tandem"} else 22)
    wave.write_bytes(wav_bytes(pcm.astype(np.float32) / 32768.0))
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(base),
            "-i",
            str(wave),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "pcm_s16le",
            "-video_track_timescale",
            "25000",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=12,
    )
    base.unlink()
    wave.unlink()
    table, delivered = mouth_local_hold.native(path)
    require(
        list(table.pts) == [1000 * i for i in range(N)] and table.time_base == "1/25000",
        "Output native clock differs",
    )
    data = decode_audio(path, source_sha256=digest(path), max_seconds=3)
    first, observed, overlap = clock.resample_clock(data)
    require(first == 0 and not overlap and np.array_equal(pcm, observed), "Output PCM differs")
    return table, delivered, observed


def witness(choice, frames, pts):
    if choice is None:
        return []
    x, y, w, h = choice["box"]
    return [
        {
            "frame": i,
            "pts": pts[i],
            "rgb_sha256": hashlib.sha256(frames[i].tobytes()).hexdigest(),
            "crop_sha256": hashlib.sha256(frames[i][y : y + h, x : x + w].tobytes()).hexdigest(),
        }
        for a, b in choice["ranges"]
        for i in (a - 1, a, (a + b) // 2, b)
    ]


def run(folder, row, detector):
    frames, source_pts, pcm, box = original(row, detector)
    local = folder / f"{row['bank']}-{row['group']}"
    local.mkdir(mode=0o700)
    cases = []
    for role, pictures, audio, source, target in author(frames, pcm, box, row["group"]):
        path = local / f"{role}.mov"
        table, decoded, observed = deliver(local, path, pictures, audio, role)
        geometry = mouth_local_hold.stable_geometry(decoded, detector)
        gray = np.stack([cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) for rgb in decoded]).astype(np.float32)
        choices = pairs(gray, observed, geometry) if geometry else []
        selections = [best(choices, point) for point in POINTS]
        hold_status, hold, av = incumbent(decoded, observed, detector)
        if role == "matched_picture_audio":
            require(pcm_sha(audio) != pcm_sha(pcm), "Duplicated audio does not differ")
        cases.append(
            {
                "role": role,
                "source": source,
                "target": target,
                "assessed": geometry is not None,
                "matches": selections,
                "hold_status": hold_status,
                "hold_range": hold["frames"] if hold else None,
                "current": av,
                "delivery_sha256": digest(path),
                "pcm_sha256": pcm_sha(observed),
                "native_witnesses": [witness(choice, decoded, table.pts) for choice in selections],
            }
        )
        if role not in {"untouched", "mouth_later", "mouth_earlier"}:
            path.unlink()
    save(
        local / "result.json",
        {
            "bank": row["bank"],
            "group": row["group"],
            "source_sha256": row["sha256"],
            "source_pts": source_pts,
            "source_audio_sha256": pcm_sha(pcm),
            "cases": cases,
        },
    )
    print(row["bank"], row["group"], "complete", flush=True)


def iou(p, q):
    return Fraction(max(0, min(p[1], q[1]) - max(p[0], q[0])), max(p[1], q[1]) - min(p[0], q[0]))


def report(rows, index):
    result = {}
    for bank in ("grid", "crema"):
        totals, roles, groups = Counter(), {r: Counter() for r in (*POSITIVES, *CONTROLS)}, set()
        for row in (r for r in rows if r["bank"] == bank):
            for case in row["cases"]:
                role, hit = case["role"], case["matches"][index]
                expected = sorted((case["source"], case["target"])) if case["target"] else None
                correct = bool(
                    hit
                    and expected
                    and hit["lag"] == expected[1][0] - expected[0][0]
                    and all(iou(x, y) >= Fraction(3, 5) for x, y in zip(hit["ranges"], expected, strict=True))
                )
                key = (
                    "correct"
                    if correct
                    else "wrong"
                    if expected and hit
                    else "false"
                    if hit
                    else "no_match"
                    if case["assessed"]
                    else "unassessed"
                )
                roles[role][key] += 1
                totals[key] += 1
                if correct:
                    groups.add(row["group"])
                if case["current"]["measured_windows"]:
                    totals["current_measured"] += 1
                if case["current"]["ranges"]:
                    totals["current_ranges"] += 1
                if case["hold_status"] != "no_stable_face":
                    totals["hold_measured"] += 1
                if case["hold_range"]:
                    totals["hold_ranges"] += 1
        result[bank] = {
            "totals": dict(totals),
            "roles": {r: dict(c) for r, c in roles.items()},
            "groups": len(groups),
            "measured": sum(totals[k] for k in ("correct", "wrong", "false", "no_match")),
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "development", "point", "held", "report"))
    p.add_argument("folder", type=Path)
    a = p.parse_args()
    cv2.setNumThreads(1)
    plan = contract()
    if a.phase == "freeze":
        a.folder.mkdir(mode=0o700)
        save(a.folder / "freeze.json", plan)
        return
    require(read(a.folder / "freeze.json") == plan, "Frozen contract differs")
    if a.phase in {"development", "held"}:
        if a.phase == "held":
            require(
                read(a.folder / "point.json")["point"] in POINTS, "Point must be frozen before held inference"
            )
        detector = load_yunet(sources.WEIGHTS)
        for row in (r for r in plan["sources"] if r["split"] == a.phase):
            run(a.folder, row, detector)
    else:
        subset = "development" if a.phase == "point" else "held"
        rows = [
            read(a.folder / f"{r['bank']}-{r['group']}" / "result.json")
            for r in plan["sources"]
            if r["split"] == subset
        ]
        if a.phase == "point":
            options = [report(rows, i) for i in range(len(POINTS))]
            allowed = [
                i
                for i, result in enumerate(options)
                if sum(r["totals"].get("false", 0) for r in result.values()) <= 1
            ]
            index = max(
                allowed,
                key=lambda i: (sum(r["totals"].get("correct", 0) for r in options[i].values()), -i),
                default=0,
            )
            save(a.folder / "point.json", {"point": POINTS[index], "development": options})
        else:
            point = read(a.folder / "point.json")["point"]
            result = report(rows, POINTS.index(point))
            save(a.folder / "result.json", {"point": point, "summary": result, "rows": rows})
            print(result, flush=True)


if __name__ == "__main__":
    main()
