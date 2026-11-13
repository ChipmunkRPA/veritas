# SPDX-License-Identifier: Apache-2.0
"""Test repeated short soundtrack ranges against two observed native mouth sequences."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

from app.detectors.audio.decode import decode_audio, wav_bytes
from app.detectors.image.faces import detect_faces, load_yunet
from app.detectors.video import lipsync
from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import audio_exact_repetition, audio_similarity
from scripts import video_lipsync_local_delay_transfer as clock
from scripts import video_lipsync_short_bracket as baseline
from scripts import video_reference_soundtrack_panel as delivery
from scripts.video_common_sources_v89 import read, require, sha, write

N, SR, HOP = 63, 16000, 640
WEIGHT = Path(__file__).resolve().parents[2] / "models/weights/face_detection_yunet_2023mar.onnx"
POSITIVE = ("sound_later", "sound_earlier", "local_late", "local_early")
CONTROLS = (
    "genuine",
    "gain",
    "lowpass",
    "global",
    "picture_copy",
    "matched_av",
    "matched_mouth_sound",
    "independent_sound",
    "quiet_repeated_bed",
)
CODECS = {
    "pcm": ("mov", ("-c:a", "pcm_s16le")),
    "aac": ("mp4", ("-c:a", "aac", "-b:a", "48k")),
}
GATE = {
    "minimum_added": 16,
    "minimum_groups": 7,
    "maximum_ordinary": 2,
    "maximum_synchronized": 0,
    "minimum_coverage": 0.75,
    "minimum_iou": "1/2",
}
METHOD = {
    "source": "63 complete 25 Hz pictures, one observed frontal face, continuous 40,320 mono PCM samples at 16 kHz plus any decoder padding in the delivery. First sufficiently long qualified recording from the preordered source choices.",
    "nominate": "Reuse active two-band audio similarity. A nonoverlapping acoustic range lasts at least 240 ms.",
    "measure": "Same source audio sample clock and midpoint-qualified consecutive picture indices. Observed eye alignment and a normalized mouth crop. Require variable patches in both ranges and at least two thirds measured picture pairs.",
    "guards": "Require distinct moving mouth pixels across at least three observed pairs and reject duplicated wide pictures and locally matched mouth trajectories. Reject conspicuous pose change.",
    "scope": "Descriptive source-time correspondence only, no estimate of manipulation direction or AI authenticity.",
    "training": "None. Compare original guard and native-translation follow-up defined on development only.",
}
ARMS = ("moving_mouth", "native_mouth_guard")


def pcm_sha(pcm):
    return hashlib.sha256(np.asarray(pcm, dtype="<i2").tobytes()).hexdigest()


def binding():
    return {
        p.name: sha(p)
        for p in (
            Path(__file__),
            Path(clock.__file__),
            Path(baseline.__file__),
            Path(audio_similarity.__file__),
            Path(delivery.__file__),
            WEIGHT,
        )
    }


def freeze(run, sources):
    receipt = read(sources / "intake-receipt.json")
    require(
        receipt["roster_sha256"] == sha(sources / "intake-freeze.json") and len(receipt["rows"]) >= 90,
        "Complete fresh source acquisition required",
    )
    dev = {row["actor"] for row in receipt["rows"] if row["split"] == "development"}
    held = {row["actor"] for row in receipt["rows"] if row["split"] == "held"}
    require(len(dev) == 4 and len(held) >= 11 and not dev & held, "Actor sets must be independent")
    require(
        all(sha(sources / row["file"]) == row["sha256"] for row in receipt["rows"]), "Source bytes differ"
    )
    run.mkdir(mode=0o700)
    write(
        run / "freeze.json",
        {
            "schema": "chipmunk.video-lipsync-repeat-audio/1",
            "source_rows": receipt["rows"],
            "sources_sha256": sha(sources / "intake-receipt.json"),
            "method": METHOD,
            "gate": GATE,
            "positive": POSITIVE,
            "control": CONTROLS,
            "codecs": CODECS,
            "bindings": binding(),
            "ffmpeg": delivery.command("ffmpeg", "-version").decode().splitlines()[0],
        },
    )


def picture(frames, path):
    sequence = np.ascontiguousarray(frames)
    command = (
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "480x360",
        "-framerate",
        "25",
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-video_track_timescale",
        "16000",
        str(path),
    )
    subprocess.run(command, input=sequence.tobytes(), capture_output=True, check=True, timeout=20)


def original(path, digest):
    table = probe_frames(path, timeout_s=10)
    require(
        (table.width, table.height) == (480, 360)
        and not (table.rotation or table.truncated or table.interpolated),
        "Publisher pictures outside the prepared profile",
    )
    decoded = decode_audio(path, source_sha256=digest, max_seconds=8)
    first, pcm, _ = clock.resample_clock(decoded)
    # Use the fixed common three-second extent after independent qualification. No pad or invented frames.
    base = max(Fraction(first, SR), table.pts[0] * table.tb)
    aligned = round(base * SR)
    i = aligned - first
    if i + N * HOP > len(pcm) or base + Fraction(N - 1, 25) > table.pts[-1] * table.tb:
        return None
    targets = [base + Fraction(j, 25) for j in range(N)]
    native = [p * table.tb for p in table.pts]
    indices = [min(range(len(native)), key=lambda k: abs(native[k] - target)) for target in targets]
    require(
        len(set(indices)) == N
        and max(abs(native[j] - t) for j, t in zip(indices, targets, strict=True)) <= Fraction(1, 40),
        "Qualification needs distinct timestamp-matched publisher pictures",
    )
    frames = list(decode_frames(path, table, indices, timeout_s=12))
    require(
        len(frames) == N
        and all(f.pts_verified and f.pts == table.pts[j] for f, j in zip(frames, indices, strict=True)),
        "Publisher source verification failed",
    )
    return (
        np.stack([f.rgb for f in frames]),
        pcm[i : i + N * HOP].copy(),
        [f.pts for f in frames],
        table.time_base,
        aligned,
    )


def copies(frames, early, late, mouth_box):
    a, b = early
    c, d = late
    whole = frames.copy()
    whole[c:d] = frames[a:b]
    mouth = frames.copy()
    x, y, w, h = mouth_box
    mouth[c:d, y : y + h, x : x + w] = frames[a:b, y : y + h, x : x + w]
    return whole, mouth


def edits(frames, pcm, independent, index, mouth_box):
    a, c = 14 + index % 3, 40 + index % 3
    width = 12
    first, second = (a, a + width), (c, c + width)
    early, late = tuple(i * HOP for i in first), tuple(i * HOP for i in second)
    donor_earlier, donor_later = pcm.copy(), pcm.copy()
    donor_earlier[slice(*early)] = pcm[slice(*late)]
    donor_later[slice(*late)] = pcm[slice(*early)]
    local = tuple(i * HOP for i in (28, 44))
    delta = 8 * HOP
    whole, mouth = copies(frames, first, second, mouth_box)
    wave_earlier, wave_later = delivery._mix(pcm, donor_earlier, early), delivery._mix(pcm, donor_later, late)
    yield "genuine", frames, pcm, None
    yield "sound_later", frames, wave_later, [list(early), list(late)]
    yield "sound_earlier", frames, wave_earlier, [list(early), list(late)]
    for role, shift, expected in (
        ("local_late", delta, [[local[0] - delta, local[0]], [local[0], local[0] + delta]]),
        ("local_early", -delta, [[local[1] - delta, local[1]], [local[1], local[1] + delta]]),
    ):
        yield role, frames, delivery._mix(pcm, clock.shift(pcm, shift), local), expected
    yield "gain", frames, delivery._mix(pcm, pcm * 0.65, late), None
    low = sosfiltfilt(butter(4, 2500, fs=SR, output="sos"), pcm)
    yield "lowpass", frames, delivery._mix(pcm, low, late), None
    yield "global", frames, clock.shift(pcm, delta), None
    yield "picture_copy", whole, pcm, None
    yield "matched_av", whole, wave_later, None
    yield "matched_mouth_sound", mouth, wave_later, None
    yield "independent_sound", frames, delivery._mix(pcm, independent, late), None
    # An ordinary repeated bed below foreground speech, with same underlying speaker in each location.
    rng = np.random.default_rng(index)
    bed = sosfiltfilt(
        butter(4, (160, 3600), fs=SR, btype="bandpass", output="sos"), rng.standard_normal(width * HOP)
    )
    rms = min(float(np.std(pcm[slice(*span)])) for span in (early, late))
    bed *= 0.30 * rms / np.std(bed)
    background = pcm.astype(np.float64)
    for span in early, late:
        background[slice(*span)] += bed
    yield (
        "quiet_repeated_bed",
        frames,
        delivery._mix(delivery._mix(pcm, background, early), background, late),
        None,
    )


def measure_frame(rgb, landmarks):
    points = np.asarray(landmarks, dtype=np.float32)
    eye = points[1] - points[0]
    distance = np.linalg.norm(eye)
    if distance < 35:
        return None
    # Similarity warp around both eyes, independent of the mouth geometry.
    anchor = np.array((30.0, 28.0), dtype=np.float32)
    rotation = np.array(((eye[0], eye[1]), (-eye[1], eye[0])), dtype=np.float32) * (36 / distance**2)
    matrix = np.column_stack((rotation, anchor - rotation @ points[0]))
    normalized = points @ rotation.T + matrix[:, 2]
    center = normalized[3:5].mean(axis=0)
    x, y = round(float(center[0])), round(float(center[1]))
    if not (31 <= x <= 66 and 49 <= y <= 78):
        return None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    region = cv2.warpAffine(gray, matrix, (96, 96), flags=cv2.INTER_LINEAR)
    patch = region[y - 9 : y + 9, x - 15 : x + 15].astype(np.float32)
    patch -= patch.mean()
    wide = cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    wide -= wide.mean()
    absolute = tuple(map(int, np.rint(points[3:5].mean(axis=0))))
    context = tuple(map(int, np.rint(points[:2].mean(axis=0))))
    return (
        patch,
        wide,
        normalized[2],
        [float(v) for v in matrix.ravel()],
        [x - 15, y - 9, 30, 18],
        gray,
        absolute,
        context,
    )


def video_inputs(path, detector):
    table = probe_frames(path, timeout_s=10)
    require(
        len(table) == N
        and all(table.tb * (p - table.pts[0]) == Fraction(i, 25) for i, p in enumerate(table.pts))
        and not (table.rotation or table.truncated or table.interpolated),
        "Unexpected delivered picture clock",
    )
    observed, patches, faces, rows = {}, {}, {}, []
    for native in decode_frames(path, table, list(range(N)), timeout_s=12):
        require(native.pts_verified and native.pts == table.pts[native.index], "Unverified picture input")
        i = native.index
        record = {
            "index": i,
            "pts": native.pts,
            "rgb_sha256": hashlib.sha256(native.rgb.tobytes()).hexdigest(),
        }
        rows.append(record)
        detection = detect_faces(cv2.cvtColor(native.rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
        if len(detection) != 1:
            continue
        face = detection[0]
        faces[i] = {
            "frame_index": i,
            "pts": native.pts,
            "landmarks": face.landmarks,
            "bbox": dict(zip(("x", "y", "w", "h"), face.box(), strict=True)),
        }
        patches[i] = lipsync.mouth_patch(native.rgb, faces[i])
        if (features := measure_frame(native.rgb, face.landmarks)) is not None:
            patch, _, _, transform, box, _, _, _ = features
            record.update(
                mouth_sha256=hashlib.sha256(patch.tobytes()).hexdigest(), transform=transform, box=box
            )
            observed[i] = features
    require(len(rows) == N, "Missing native delivered pictures")
    return table, observed, patches, faces, rows


def translation_correlation(first, second, half_width, half_height, radius):
    gray_a, (ax, ay) = first
    gray_b, (bx, by) = second
    width, height = half_width + radius, half_height + radius
    if any(
        not (width <= x < gray.shape[1] - width and height <= y < gray.shape[0] - height)
        for gray, (x, y) in (first, second)
    ):
        return 0.0
    patch = gray_a[ay - half_height : ay + half_height, ax - half_width : ax + half_width]
    search = gray_b[by - height : by + height, bx - width : bx + width]
    if patch.std() < 6:
        return 0.0
    return float(cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED).max())


def mouth_support(first, second, measurements):
    frame_pairs = [
        (k, (k * HOP + HOP // 2 + second[0] - first[0]) // HOP)
        for k in range(N)
        if first[0] + HOP // 4 <= k * HOP + HOP // 2 < first[1] - HOP // 4
    ]
    qualified = [(a, b) for a, b in frame_pairs if a in measurements and b in measurements]
    if len(qualified) < 5 or 3 * len(qualified) < 2 * len(frame_pairs):
        return None
    a, b = ([measurements[p[i]] for p in qualified] for i in (0, 1))
    lip_a, lip_b = (np.stack([row[0] for row in values]) for values in (a, b))
    change_a, change_b = (np.mean(np.abs(np.diff(lips, axis=0)), axis=(1, 2)) for lips in (lip_a, lip_b))
    pose = np.linalg.norm(np.stack([x[2] - y[2] for x, y in zip(a, b, strict=True)]), axis=1)
    if min(np.quantile(change_a, 0.65), np.quantile(change_b, 0.65)) < 2.0 or np.quantile(pose, 0.80) > 7.0:
        return None
    mouth = np.mean(np.abs(lip_a - lip_b), axis=(1, 2))
    context = np.asarray([np.mean(np.abs(x[1] - y[1])) for x, y in zip(a, b, strict=True)])
    # A copied picture, or a matching mouth in independently moving context, is ordinary correspondence.
    if np.median(context) < 1.4 or np.median(mouth) < 4.0 or np.count_nonzero(mouth >= 5.5) < 3:
        return None
    similarity = [
        translation_correlation((x[5], x[6]), (y[5], y[6]), 12, 8, 5) for x, y in zip(a, b, strict=True)
    ]
    static_reference = [
        translation_correlation((x[5], x[7]), (y[5], y[7]), 25, 15, 5) for x, y in zip(a, b, strict=True)
    ]
    return {
        "frame_pairs": [list(p) for p in qualified],
        "mouth_mae": [float(v) for v in mouth],
        "wide_mae": [float(v) for v in context],
        "max_robust_pose_distance": float(np.quantile(pose, 0.8)),
        "motion_65th": [float(np.quantile(v, 0.65)) for v in (change_a, change_b)],
        "native_mouth_correlation": similarity,
        "native_upper_face_correlation": static_reference,
    }


def matched(pcm, observed, rows):
    data = np.asarray(pcm, dtype="<i2").tobytes()
    exact = audio_exact_repetition.find_repeated_spans(data)
    acoustic = audio_similarity.find_similar_spans(data)
    voice = lipsync.vad(pcm.astype(np.float32) / 32768)
    scopes = []
    for span in acoustic.spans:
        first, second = span.first, span.second
        if first[1] - first[0] < 6 * HOP:
            continue
        a, b = (voice.speech[x // HOP : y // HOP] for x, y in (first, second))
        if min(a.mean(), b.mean()) < 0.4:
            continue
        if (m := mouth_support(first, second, observed)) is not None:
            scopes.append(
                {
                    "samples": [list(first), list(second)],
                    "pcm_sha256": [pcm_sha(pcm[x:y]) for x, y in (first, second)],
                    **m,
                    "pictures": [rows[i] for i in sorted({i for p in m["frame_pairs"] for i in p})],
                }
            )
    scopes.sort(key=lambda r: (r["samples"][0][0] - r["samples"][0][1], r["samples"]))
    return {
        "audio_status": acoustic.status,
        "assessed": len(observed) >= N * 3 // 4 and acoustic.status == "measured",
        "exact_audio": [[list(s.first), list(s.second)] for s in exact.spans],
        "similar_audio": [[list(s.first), list(s.second)] for s in acoustic.spans],
        "matches": {
            "moving_mouth": scopes[:2],
            "native_mouth_guard": [
                scope
                for scope in scopes
                if sum(
                    m >= 0.90 or (m >= 0.82 and m >= s)
                    for m, s in zip(
                        scope["native_mouth_correlation"], scope["native_upper_face_correlation"], strict=True
                    )
                )
                * 3
                < len(scope["frame_pairs"]) * 2
            ][:2],
        },
    }


def score(run, source, row, input_frames, input_pcm, donor_pcm, qualified, detector):
    index = row["ordinal"]
    folder = run / f"group-{row['group']:02d}"
    folder.mkdir(mode=0o700)
    a = 14 + index % 3
    geometries = [
        detect_faces(cv2.cvtColor(input_frames[i], cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
        for i in (a, a + 5, a + 10)
    ]
    require(all(len(g) == 1 for g in geometries), "Authored matched mouth control requires one source face")
    mouth = np.median([face[0].landmarks[3:5] for face in geometries], axis=0)
    width = float(np.linalg.norm(mouth[0] - mouth[1]))
    x, y = mouth.mean(axis=0)
    w, h = round(width * 2.25), round(width * 1.55)
    box = [round(x - w / 2), round(y - h / 2), w, h]
    require(
        box[0] > 0 and box[1] > 0 and box[0] + w < 480 and box[1] + h < 360,
        "Authored mouth box outside source",
    )
    source_picture = folder / "unchanged.mov"
    picture(input_frames, source_picture)
    cases = []
    mouth_file = folder / "mouth-only.mov"
    picture_file = folder / "picture-copy.mov"
    for role, frames, wave, truth in edits(input_frames, input_pcm, donor_pcm, index, box):
        if role == "picture_copy":
            picture(frames, picture_file)
        if role == "matched_mouth_sound":
            picture(frames, mouth_file)
        target = (
            picture_file
            if role in {"picture_copy", "matched_av"}
            else mouth_file
            if role == "matched_mouth_sound"
            else source_picture
        )
        wav = folder / "input.wav"
        wav.write_bytes(wav_bytes(wave.astype(np.float32) / 32768))
        for codec, (ext, options) in CODECS.items():
            path = folder / f"{role}-{codec}.{ext}"
            delivery.remux(target, wav, path, options)
            digest = sha(path)
            decoded = decode_audio(path, source_sha256=digest, max_seconds=4)
            start, pcm, _ = clock.resample_clock(decoded)
            require(
                start == 0 and N * HOP <= len(pcm) <= N * HOP + 1024, "Unexpected delivered audio coverage"
            )
            if codec == "pcm":
                require(np.array_equal(pcm, wave), "Lossless soundtrack differs from authored samples")
            table, observed, patches, faces, records = qualified.setdefault(
                target.name, None
            ) or video_inputs(path, detector)
            qualified[target.name] = table, observed, patches, faces, records
            delivered = probe_frames(path, timeout_s=6)
            require(delivered == table, "Remux changed the delivered native picture clock")
            grid = lipsync.tick_grid(table.times(), N / 25, len(pcm), audio_start_s=0.0)
            current = baseline.current(faces, patches, pcm.astype(np.float32) / 32768, grid)
            result = matched(pcm, observed, records)
            cases.append(
                {
                    "role": role,
                    "carrier": codec,
                    "truth": truth,
                    "digest": digest,
                    "samples": len(pcm),
                    "pcm_sha256": pcm_sha(pcm),
                    "pts_sha256": hashlib.sha256(str(table.pts).encode()).hexdigest(),
                    "frame_time_base": table.time_base,
                    "observed": len(observed),
                    "candidate": result,
                    "current": current,
                }
            )
            if row["group"] != 5:
                path.unlink()
    wav.unlink()
    write(
        folder / "results.json",
        {
            "group": row["group"],
            "source_sha256": row["sha256"],
            "source_pts": source[2],
            "source_time_base": source[3],
            "audio_start_sample": source[4],
            "cases": cases,
        },
    )
    print(
        row["group"],
        row["phrase"],
        [
            (c["carrier"], c["role"], {a: len(x) for a, x in c["candidate"]["matches"].items()})
            for c in cases
            if any(c["candidate"]["matches"].values())
        ],
        flush=True,
    )


def overlap(a, b):
    return Fraction(max(0, min(a[1], b[1]) - max(a[0], b[0])), max(a[1], b[1]) - min(a[0], b[0]))


def outcome(rows, arm):
    summary = {}
    for codec in CODECS:
        result = {role: Counter() for role in POSITIVE + CONTROLS}
        groups = set()
        for group in rows:
            for row in group["cases"]:
                if row["carrier"] != codec:
                    continue
                c = result[row["role"]]
                c["count"] += 1
                candidate, truth = row["candidate"], row["truth"]
                c["assessed"] += candidate["assessed"]
                c["current_measured"] += bool(row["current"]["measured_windows"])
                c["current_ranges"] += bool(row["current"]["ranges"])
                c["audio_exact_pairs"] += bool(candidate["exact_audio"])
                c["audio_similar_pairs"] += bool(candidate["similar_audio"])
                spans = [r["samples"] for r in candidate["matches"][arm]]
                c["candidate_flags"] += bool(spans)
                if truth:
                    useful = any(
                        min(overlap(a, b) for a, b in zip(span, truth, strict=True))
                        >= Fraction(GATE["minimum_iou"])
                        for span in spans
                    )
                    c["useful"] += useful
                    c["wrong"] += bool(spans and not useful)
                    if useful:
                        groups.add(group["group"])
        total = sum(result.values(), Counter())
        positives = sum((result[r] for r in POSITIVE), Counter())
        controls = sum((result[r] for r in CONTROLS), Counter())
        checks = {
            "locations": positives["useful"] >= GATE["minimum_added"],
            "groups": len(groups) >= GATE["minimum_groups"],
            "ordinary": controls["candidate_flags"] <= GATE["maximum_ordinary"],
            "synchronized": sum(
                result[r]["candidate_flags"] for r in ("matched_av", "matched_mouth_sound", "global")
            )
            == 0,
            "coverage": total["assessed"] / total["count"] >= GATE["minimum_coverage"],
        }
        summary[codec] = {
            "roles": result,
            "positives": positives,
            "controls": controls,
            "total": total,
            "new_groups": len(groups),
            "gate": checks,
            "advance": all(checks.values()),
        }
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "development", "held", "report"))
    p.add_argument("run", type=Path)
    p.add_argument("sources", type=Path)
    args = p.parse_args()
    if args.phase == "freeze":
        freeze(args.run, args.sources)
        return
    plan = read(args.run / "freeze.json")
    require(
        plan["bindings"] == binding() and plan["sources_sha256"] == sha(args.sources / "intake-receipt.json"),
        "Freeze differs",
    )
    if args.phase == "report":
        journal = [read(path) for path in sorted(args.run.glob("group-*/results.json"))]
        result = {
            split: {
                arm: outcome([r for r in journal if (r["group"] <= 4) == (split == "development")], arm)
                for arm in ARMS
            }
            for split in ("development", "held")
        }
        write(args.run / "aggregate.json", {"freeze_sha256": sha(args.run / "freeze.json"), "result": result})
        print(result)
        return
    cv2.setNumThreads(1)
    detector = load_yunet(WEIGHT)
    picks = {}
    for row in (r for r in plan["source_rows"] if r["split"] == args.phase):
        if row["group"] in picks:
            continue
        path = args.sources / row["file"]
        if (item := original(path, row["sha256"])) is not None:
            picks[row["group"]] = row, item
    require(
        len(picks) >= (4 if args.phase == "development" else 8),
        "Incomplete independently qualified source roster",
    )
    write(
        args.run / f"{args.phase}-qualification.json",
        {
            "qualified_groups": list(picks),
            "unqualified_groups": sorted(
                {row["group"] for row in plan["source_rows"] if row["split"] == args.phase} - picks.keys()
            ),
            "planned_groups": len(
                {row["group"] for row in plan["source_rows"] if row["split"] == args.phase}
            ),
            "rule": "Require a complete common 63-picture and 40,320-sample extent, with no padding or extrapolation.",
        },
    )
    groups = list(picks)
    for position, group in enumerate(groups):
        row, source = picks[group]
        _, donor = picks[groups[(position + 1) % len(groups)]]
        score(args.run, source, row, source[0], source[1], donor[1], {}, detector)


if __name__ == "__main__":
    main()
