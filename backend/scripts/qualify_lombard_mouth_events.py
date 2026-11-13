# SPDX-License-Identifier: Apache-2.0
"""Qualify distinct acoustic and visible stop-release brackets on a fresh source bank."""

from __future__ import annotations

import argparse
import hashlib
import struct
from collections import Counter, defaultdict
from contextlib import closing
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence.video_reference_timing import _mesh
from scripts import grid_fresh_release_labels as release
from scripts import grid_reserve_neural_mouth as observer
from scripts import lombard_mouth_source_intake as intake
from scripts.video_common_sources_v89 import read, require, sha, write

SPEC = {
    "audio": "Decode the publisher-aligned external 16 kHz WAV. Detect one distinct 15 ms high-band burst with speech following, using the publisher forced-phone end only as a coarse neighborhood. Record competing events and exact PCM hash.",
    "video": "Decode the silent front video. A separate muted pass finds two consecutive near-closed observations and the first clear crossing followed by a sustained reopening. Use the same coarse forced-phone anchor, with no acoustic measurements. Retain native picture PTS, RGB and crop commitments.",
    "joint": "At most 80 ms between acoustic and visual bracket midpoints. The audible and visible event and temporal flanks must fit within the original paired release coordinates. Native audio and video start at their released zero origins. This is independent measured local-event agreement, without external human validation or whole-clip synchronization.",
    "normalization": "Keep complete uniform native picture cadence from 23.5 to 25 fps and every exact float32 sample. A separate rendering contract must preserve each exact cadence.",
    "maximum_bracket_midpoint_difference": "2/25",
    "render_span": "Half-open 16-picture span from six pictures before the last closed picture. Require three-picture outside flanks and keep both labelled event brackets inside the interval.",
    "scope": "Source-only qualification. No renderer or detection checkpoint is loaded.",
}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def seal(folder):
    paths = (Path(__file__), Path(release.__file__), Path(observer.__file__), Path(intake.__file__))
    committed = read(folder / "source-receipt.json")
    require(committed["freeze_sha256"] == sha(folder / "freeze.json"), "Input acquisition differs")
    require(
        len(committed["rows"]) == 32 and len({r["group"] for r in committed["rows"]}) == 8,
        "Frozen source roster incomplete",
    )
    return {
        "spec": SPEC,
        "source_receipt_sha256": sha(folder / "source-receipt.json"),
        "worker_sha256": {p.name: sha(p) for p in paths},
        "geometry_sha256": sha(observer.ASSETS / "permissions/face_landmarker.task"),
    }


def source(folder, row, name):
    record = row["inputs"][name]
    path = folder / "sources" / record["file"]
    require(sha(path) == record["sha256"], "Selected publisher member differs")
    return path


def native_pcm(path):
    payload = path.read_bytes()
    require(
        payload[:4] == b"RIFF"
        and payload[8:12] == b"WAVE"
        and int.from_bytes(payload[4:8], "little") + 8 == len(payload),
        "Expected complete publisher RIFF/WAVE",
    )
    chunks = {}
    i = 12
    while i + 8 <= len(payload):
        name, size = struct.unpack_from("<4sI", payload, i)
        require(
            name not in chunks and i + 8 + size <= len(payload), "Invalid duplicated or truncated WAV chunk"
        )
        chunks[name] = payload[i + 8 : i + 8 + size]
        i += 8 + size + (size & 1)
    require(
        i == len(payload) and len(chunks.get(b"fmt ", b"")) >= 16 and b"data" in chunks,
        "Incomplete WAV chunks",
    )
    fmt, channels, rate, byte_rate, block, bits = struct.unpack_from("<HHIIHH", chunks[b"fmt "])
    require(
        (fmt, channels, rate, byte_rate, block, bits) == (3, 1, 16000, 64000, 4, 32),
        "Require native mono float32 publisher WAV",
    )
    data = chunks[b"data"]
    require(len(data) % 4 == 0, "Truncated float32 sample")
    values = np.frombuffer(data, dtype="<f4")
    require(
        20000 <= len(values) <= 64000 and np.isfinite(values).all() and np.max(np.abs(values)) <= 1,
        "Native sample count, finiteness or amplitude unsupported",
    )
    return data, values


def audio(folder, rows):
    output = []
    for row in rows:
        path = source(folder, row, "audio")
        pcm_source, pcm = native_pcm(path)
        rate, width = 16000, 4
        n = len(pcm)
        guess = round(Fraction(row["aligned_following_phone_s"]) * rate)
        observation = release.audio_only(pcm.astype(np.float64) * 32768, guess)
        if "range_samples" in observation:
            a, b = observation["range_samples"]
            observation["burst_native_pcm_sha256"] = digest(pcm_source[a * width : b * width])
        output.append(
            {
                "ordinal": row["ordinal"],
                "samples": n,
                "sample_rate": rate,
                "sample_bytes": width,
                "native_pcm_sha256": digest(pcm_source),
                "audio_sha256": sha(path),
                **observation,
            }
        )
    return output


def visual_releases(apertures, times, hint):
    events = []
    for i in range(1, len(times) - 4):
        near = apertures[i - 1 : i + 5]
        if not (hint - Fraction(12, 100) <= (times[i] + times[i + 1]) / 2 <= hint + Fraction(16, 100)) or any(
            v is None for v in near
        ):
            continue
        if (
            apertures[i - 1] <= 0.075
            and apertures[i] <= 0.055 < apertures[i + 1]
            and max(apertures[i + 1 : i + 5]) - apertures[i] >= 0.032
        ):
            events.append([i, i + 1])
    if len(events) != 1:
        return {"status": "ambiguous_reopening" if events else "no_clear_reopening", "candidates": events}
    [pair] = events
    return {"status": "observed_reopening", "frames": pair, "seconds": [str(times[i]) for i in pair]}


def visual(folder, rows):
    module, task = observer.mp_task()
    output = []
    with closing(task):
        for row in rows:
            path = source(folder, row, "front")
            table = probe_frames(path, timeout_s=12)
            times = [p * table.tb for p in table.pts]
            require(
                38 <= len(times) <= 100
                and times[0] == 0
                and (table.width, table.height) == (720, 480)
                and not (table.rotation or table.truncated or table.interpolated),
                "Publisher video outside verified input scope",
            )
            step = times[1] - times[0]
            uniform = Fraction(1, 25) <= step <= Fraction(2, 47) and all(
                b - a == step for a, b in pairwise(times)
            )
            apertures = [None] * len(times)
            anchors = {}
            hint = Fraction(row["aligned_following_phone_s"])
            indices = [i for i, t in enumerate(times) if abs(t - hint) <= Fraction(1, 2)]
            for frame in decode_frames(path, table, indices, timeout_s=12):
                require(
                    frame.pts_verified and frame.pts == table.pts[frame.index],
                    "Observed frame lacks native timestamp",
                )
                features = _mesh(task, module, frame.rgb)
                if features is None:
                    continue
                measured, box = features
                apertures[frame.index] = float(np.median(measured[:3]))
                x0, y0, x1, y1 = box
                anchors[frame.index] = {
                    "frame": frame.index,
                    "pts": frame.pts,
                    "rgb_sha256": digest(frame.rgb.tobytes()),
                    "box_xyxy": box,
                    "crop_sha256": digest(frame.rgb[y0:y1, x0:x1].tobytes()),
                }
            observed = visual_releases(apertures, times, hint)
            if "frames" in observed:
                observed["anchors"] = [anchors[i] for i in observed["frames"]]
            output.append(
                {
                    "ordinal": row["ordinal"],
                    "time_base": table.time_base,
                    "pts": list(table.pts),
                    "qualified_native_cadence": uniform,
                    "native_cadence": str(1 / step),
                    "measured_frames": len(anchors),
                    "video_sha256": sha(path),
                    **observed,
                }
            )
            print(row["ordinal"], observed["status"], flush=True)
    return output


def compare(folder, rows, lock):
    sound, pictures = (read(folder / f"{k}-labels.json") for k in ("audio", "visual"))
    require(
        all(d["freeze_sha256"] == sha(folder / "event-freeze.json") for d in (sound, pictures)),
        "Separate observation ledgers differ",
    )
    found = []
    for row, audio, visual in zip(rows, sound["rows"], pictures["rows"], strict=True):
        require(row["ordinal"] == audio["ordinal"] == visual["ordinal"], "Source observation roster differs")
        result = {k: row[k] for k in ("ordinal", "group", "split", "condition", "phoneme")}
        result.update(audio_status=audio["status"], visual_status=visual["status"], status="unavailable")
        if "range_samples" in audio and "frames" in visual:
            sound_time = [Fraction(s, 16000) for s in audio["range_samples"]]
            vision_time = list(map(Fraction, visual["seconds"]))
            delta = (sum(vision_time) - sum(sound_time)) / 2
            start = visual["frames"][0] - 6
            stop = start + 16
            native = [Fraction(v) * Fraction(visual["time_base"]) for v in visual["pts"]]
            result.update(
                audio_range_s=list(map(str, sound_time)),
                visual_range_s=visual["seconds"],
                signed_midpoint_delta_s=str(delta),
                status="unresolved_clock_agreement_or_flanks",
            )
            if (
                abs(delta) <= Fraction(lock["spec"]["maximum_bracket_midpoint_difference"])
                and visual["qualified_native_cadence"]
                and 3 <= start < stop <= len(native) - 4
                and native[start] <= min(sound_time[0], vision_time[0])
                and max(sound_time[1], vision_time[1]) < native[stop]
                and native[stop] + Fraction(1, 8) <= Fraction(audio["samples"], 16000)
            ):
                result.update(
                    status="qualified_original_event",
                    native_edit_frames=[start, stop],
                    native_edit_range_s=[str(native[i]) for i in (start, stop)],
                    audio_commitment=audio["burst_native_pcm_sha256"],
                    visual_anchors=visual["anchors"],
                )
        found.append(result)
    by_group = defaultdict(list)
    for row in found:
        if row["status"] == "qualified_original_event":
            by_group[row["group"]].append(row)
    chosen = []
    for items in by_group.values():
        seen = set()
        for item in items:
            if item["condition"] not in seen:
                chosen.append(item["ordinal"])
                seen.add(item["condition"])
    counts = Counter((r["split"], r["status"]) for r in found)
    write(
        folder / "qualification.json",
        {
            "freeze_sha256": sha(folder / "event-freeze.json"),
            "cases": found,
            "scheduled_render_ids": chosen,
            "counts": [[*k, n] for k, n in counts.items()],
            "qualified_groups": len(by_group),
        },
    )
    print(counts, "groups", len(by_group), "render ids", chosen, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "audio", "visual", "compare"))
    p.add_argument("folder", type=Path)
    a = p.parse_args()
    lock = seal(a.folder)
    frozen = a.folder / "event-freeze.json"
    if a.phase == "freeze":
        write(frozen, lock)
        return
    require(lock == read(frozen), "Source event contract differs")
    rows = read(a.folder / "source-receipt.json")["rows"]
    cv2.setNumThreads(1)
    if a.phase == "compare":
        compare(a.folder, rows, lock)
    else:
        output = (audio if a.phase == "audio" else visual)(a.folder, rows)
        write(a.folder / f"{a.phase}-labels.json", {"freeze_sha256": sha(frozen), "rows": output})


if __name__ == "__main__":
    main()
