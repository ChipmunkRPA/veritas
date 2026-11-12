# SPDX-License-Identifier: Apache-2.0
"""Test a single-clip visible release and acoustic transient timing cue on exact native media."""

from __future__ import annotations

import argparse
import subprocess
from contextlib import closing
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

from app.detectors.image.faces import load_yunet
from app.detectors.video import lipsync
from scripts import grid_paired_closure_panel as native
from scripts import grid_reserve_neural_mouth as neural
from scripts import grid_reserve_neural_mouth_readout as existing
from scripts import grid_reserve_word_clocks as clocks
from scripts import review_video_lipsync_local_delay_transfer as original_audio
from scripts import video_lipsync_local_contrast_transfer as baseline
from scripts.video_common_sources_v89 import read, require, sha, write

RATE = 16000
TICK = RATE // 25
STUDY = clocks.planning.BASE / "grid-reserve-neural-mouth-2026-11-12-admitted"
QUALIFICATION = clocks.planning.BASE / "grid-reserve-word-clocks-2026-11-12"
RELEASE = {
    "video": "Eye-normalized three-gap median. Last picture at or below .055 before a crossing and a rise of at least .032 over the next four native pictures. Anchor is the midpoint of the bracketing pair.",
    "audio": "5 ms bins. Broadband speech after a quiet interval and a high-pass transient at 1,600 Hz. Require an eightfold high-band rise, and normalize floor and sustained energy to the clip's 90th percentile. Keep the strongest candidate per 70 ms neighborhood.",
    "association": "Unique acoustic and visual association after removing ambiguous alternatives. One concordant association on either side must agree within 60 ms. The middle pair needs a 160–400 ms change with no close alternative in either modality.",
    "normal_radius_ms": 80,
    "flank_consistency_ms": 60,
    "flank_max_gap_ms": 800,
    "local_min_ms": 160,
    "local_max_ms": 400,
    "known_shift_tolerance_ms": 80,
    "authored_crossfade_samples": 160,
    "gate": {
        "correct": 8,
        "correct_source_groups": 3,
        "minimum_assessed_fraction": 0.60,
        "maximum_control_flags": 1,
    },
}
# Local shift is applied to audio. A global audio offset is a control for the claim of a *local* change.
ROLES = {
    "untouched": (None, 0),
    "remux": (None, 0),
    "early_audio_later": ((18, 38), 8),
    "early_audio_earlier": ((18, 38), -8),
    "middle_audio_later": ((27, 49), 6),
    "middle_audio_earlier": ((27, 49), -6),
    "local_gain": ((18, 38), 0),
    "local_lowpass": ((27, 49), 0),
    "global_audio_later": (None, 8),
    "global_audio_earlier": (None, -8),
}


def visual_candidates(geometry):
    aperture = np.array([np.nan if item is None else np.median(item[:3]) for item in geometry])
    releases = []
    for i in range(1, len(aperture) - 4):
        around = aperture[i - 1 : i + 5]
        if (
            np.isfinite(around).all()
            and aperture[i] <= 0.055 < aperture[i + 1]
            and max(aperture[i + 1 : i + 5]) - aperture[i] >= 0.032
        ):
            releases.append({"picture": i, "anchor": i * TICK + TICK // 2})
    return releases


def mouth_crop(mesh, picture):
    height, width = picture.shape[:2]
    points = np.asarray(
        [(mesh[k].x * width, mesh[k].y * height) for k in (78, 82, 87, 13, 14, 308, 312, 317)]
    )
    margin = max(4, float(np.ptp(points[:, 0])) * 0.22)
    lo = np.maximum(0, np.floor(points.min(axis=0) - margin)).astype(int)
    hi = np.minimum((width, height), np.ceil(points.max(axis=0) + margin)).astype(int)
    return [*map(int, lo), *map(int, hi)]


def audio_candidates(pcm):
    hop = RATE // 200
    signal = pcm.astype(np.float64) / 32768
    high = sosfiltfilt(butter(3, 1600, fs=RATE, btype="highpass", output="sos"), signal)
    count = len(pcm) // hop * hop
    power = np.mean(signal[:count].reshape(-1, hop) ** 2, axis=1)
    high_power = np.mean(high[:count].reshape(-1, hop) ** 2, axis=1)
    regular, bright = (max(1e-12, np.percentile(v, 90)) for v in (power, high_power))
    proposed = []
    for i in range(7, len(power) - 17):
        peak = max(high_power[i : i + 2])
        previous = power[i - 6 : i - 1].mean()
        following = np.percentile(power[i + 3 : i + 16], 60)
        strength = peak / max(high_power[i - 6 : i - 1].mean(), bright * 0.005)
        if (
            strength > 8
            and peak >= bright * 0.20
            and following > regular * 0.12
            and following / max(previous, regular * 0.01) > 3
            and high_power[i - 1] < high_power[i] >= high_power[i + 1]
        ):
            proposed.append((float(strength), -i))
    return [
        {"anchor": -i * hop, "strength": strength}
        for strength, i in proposed
        if (strength, i) == max(other for other in proposed if abs(i - other[1]) < 14)
    ]


def measure(geometry, pcm):
    visual, sound = visual_candidates(geometry), audio_candidates(pcm)
    radius = RELEASE["normal_radius_ms"] * 16
    anchors = [
        (i, j)
        for i, j in product(range(len(visual)), range(len(sound)))
        if abs(sound[j]["anchor"] - visual[i]["anchor"]) <= radius
    ]
    certain = [
        (i, j) for i, j in anchors if sum(i == a for a, _ in anchors) == sum(j == b for _, b in anchors) == 1
    ]

    def offset(i, j):
        return sound[j]["anchor"] - visual[i]["anchor"]

    alternatives = {}
    for left, right in product(certain, repeat=2):
        a, b = left
        c, d = right
        if a >= c or b >= d or abs(offset(*left) - offset(*right)) > RELEASE["flank_consistency_ms"] * 16:
            continue
        central = (offset(*left) + offset(*right)) // 2
        for i, j in product(range(a + 1, c), range(b + 1, d)):
            if (
                any(i == x or j == y for x, y in anchors)
                or max(
                    visual[i]["anchor"] - visual[a]["anchor"],
                    visual[c]["anchor"] - visual[i]["anchor"],
                    sound[j]["anchor"] - sound[b]["anchor"],
                    sound[d]["anchor"] - sound[j]["anchor"],
                )
                > RELEASE["flank_max_gap_ms"] * 16
            ):
                continue
            delta = offset(i, j) - central
            if RELEASE["local_min_ms"] * 16 <= abs(delta) <= RELEASE["local_max_ms"] * 16:
                row = (abs(offset(*left) - offset(*right)), left, right, delta)
                alternatives[(i, j)] = min(row, alternatives.get((i, j), row))
    result = {
        "status": "unassessed",
        "visual_events": len(visual),
        "sound_events": len(sound),
        "clear_ordinary_pairs": len(certain),
    }
    if len(certain) >= 2 and not alternatives:
        result["status"] = "no_supported_local_change"
    if len(alternatives) == 1:
        (i, j), (_, before, after, delta) = next(iter(alternatives.items()))
        result.update(
            status="local_timing_change",
            audio_minus_picture_change_samples=delta,
            events=[
                {
                    "picture": visual[v]["picture"],
                    "picture_anchor": visual[v]["anchor"],
                    "acoustic_anchor": sound[s]["anchor"],
                }
                for v, s in (before, (i, j), after)
            ],
        )
    return result


def bindings():
    return {
        path.name: sha(path)
        for path in map(
            Path,
            (
                __file__,
                native.__file__,
                existing.__file__,
                baseline.__file__,
                baseline.baseline.__file__,
                lipsync.__file__,
                original_audio.__file__,
                neural.__file__,
            ),
        )
    }


def discovery(out):
    native_audit = read(STUDY / "independent-replay.json")
    require(
        native_audit["status"] == "verified"
        and native_audit["construction_sha256"] == sha(STUDY / "construction.json"),
        "Require verified native discovery material",
    )
    prior = existing.freeze(STUDY)
    module, task = neural.mp_task()
    rows = []
    with closing(task):
        for entry in prior["sources"]:
            path = Path(entry["path"])
            require(sha(path) == entry["sha256"], "Discovery source bytes differ")
            _, pictures, pcm = native.video(path)
            measured = measure(native.motion(module, task, pictures), pcm)
            rows.append(
                {"role": entry["role"], "ordinal": entry["ordinal"], "sha256": entry["sha256"], **measured}
            )
            print(entry["role"], measured, flush=True)
    write(out / "discovery.json", {"bindings": bindings(), "method": RELEASE, "rows": rows})


def freeze(out):
    qualified = read(QUALIFICATION / "qualification.json")
    clocks_path = QUALIFICATION / "private-word-clocks.json"
    require(
        sha(clocks_path) == qualified["private_word_clocks_sha256"],
        "Exact qualified native clock roster required",
    )
    sources, plan = clocks.plan()
    discovery_result = read(out / "discovery.json")
    discovery_groups = {sources[r["ordinal"]]["component_index"] for r in discovery_result["rows"]}
    require(
        len(discovery_groups) == 2
        and len(discovery_result["rows"]) == 8
        and discovery_result["bindings"] == bindings()
        and discovery_result["method"] == RELEASE,
        "Current method needs a complete discovery record",
    )
    held = []
    for row in read(clocks_path)["records"]:
        item = sources[row["ordinal"]]
        if item["component_index"] not in discovery_groups:
            require(
                item["input"]["source_sha256"] == row["source_sha256"],
                "Source roster and clock qualification differ",
            )
            held.append(
                {
                    "ordinal": row["ordinal"],
                    "group": item["component_index"],
                    "source_sha256": row["source_sha256"],
                }
            )
    require(len(held) == 12 and len({row["group"] for row in held}) == 6, "Require all disjoint held sources")
    sealed = {
        "method": RELEASE,
        "roles": ROLES,
        "held": held,
        "discovery_groups": sorted(discovery_groups),
        "discovery_sha256": sha(out / "discovery.json"),
        "word_clock_roster_sha256": sha(clocks_path),
        "plan_sha256": plan,
        "bindings": bindings(),
        "prior_exposure": "Earlier source qualification and paired or prior short timing testing. New exact media deliveries and no overlap with this trial's discovery group.",
    }
    write(out / "freeze.json", sealed)


def verify_lock(out):
    seal = read(out / "freeze.json")
    require(
        seal["bindings"] == bindings()
        and seal["method"] == RELEASE
        and seal["discovery_sha256"] == sha(out / "discovery.json")
        and {role: [list(span) if span else None, shift] for role, (span, shift) in ROLES.items()}
        == seal["roles"],
        "Frozen method, media construction or discovery differs",
    )
    return seal


def edited_audio(pcm, role, span, shift):
    if role in ("untouched", "remux"):
        return pcm
    if span is None:
        return original_audio.rebuilt(
            pcm, "global_late" if shift > 0 else "global_early", 0, 0, abs(shift), 160
        )
    a, b = (tick * TICK for tick in span)
    if shift:
        return original_audio.rebuilt(
            pcm, "local_late" if shift > 0 else "local_early", *span, abs(shift), 160
        )
    if role == "local_gain":
        return original_audio.rebuilt(pcm, "gain", *span, 0, 160)
    other = np.rint(sosfiltfilt(butter(3, 2200, fs=RATE, output="sos"), pcm.astype(np.float64)))
    blend = np.minimum(np.arange(1, b - a + 1), np.arange(b - a, 0, -1)).clip(0, 160) / 160
    result = pcm.copy()
    result[a:b] = np.rint((1 - blend) * pcm[a:b] + blend * other[a:b]).clip(-32768, 32767).astype("<i2")
    return result


def author(out):
    seal = verify_lock(out)
    sources, digest = clocks.plan()
    require(digest == seal["plan_sha256"], "Held source plan differs")
    directory = out / "media"
    directory.mkdir(mode=0o700)
    cases = []
    for pin in seal["held"]:
        source = Path(sources[pin["ordinal"]]["input"]["source_file"])
        require(sha(source) == pin["source_sha256"], "Held source file differs")
        _, pictures, pcm = native.video(source)
        rgb = [original_audio.sha(frame.tobytes()) for frame in pictures]
        for role, (span, shift) in ROLES.items():
            expected = edited_audio(pcm, role, span, shift)
            path = source
            if role != "untouched":
                path = directory / f"{pin['ordinal']:02}-{role}.mkv"
                wav = path.with_suffix(".wav")
                native.save_pcm(wav, expected)
                subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-i",
                        str(source),
                        "-i",
                        str(wav),
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "pcm_s16le",
                        str(path),
                    ],
                    capture_output=True,
                    check=True,
                    timeout=15,
                )
                wav.unlink()
                _, actual_pictures, actual_pcm = native.video(path)
                require(
                    np.array_equal(actual_pcm, expected)
                    and [original_audio.sha(x.tobytes()) for x in actual_pictures] == rgb,
                    "Delivered clip differs from authored PCM or original native pictures",
                )
            changed = np.flatnonzero(pcm != expected)
            if span:
                require(
                    len(changed) > 0 and changed.min() >= span[0] * TICK and changed.max() < span[1] * TICK,
                    "Local treatment escaped its declared bounds",
                )
            cases.append(
                {
                    **pin,
                    "role": role,
                    "path": str(path),
                    "file_sha256": sha(path),
                    "span": span,
                    "shift": shift,
                    "whole_pcm_sha256": original_audio.sha(expected.tobytes()),
                    "changed_samples": len(changed),
                    "native_rgb": rgb,
                }
            )
        print("authored", pin["ordinal"], flush=True)
    write(out / "media.json", {"freeze_sha256": sha(out / "freeze.json"), "cases": cases})


def localized(proposal, span, shift):
    if proposal["status"] != "local_timing_change" or not span or not shift:
        return False
    anchor = proposal["events"][1]["acoustic_anchor"]
    return (
        span[0] * TICK + 160 <= anchor < span[1] * TICK - 160
        and abs(proposal["audio_minus_picture_change_samples"] - shift * TICK)
        <= RELEASE["known_shift_tolerance_ms"] * 16
    )


def score(out):
    verify_lock(out)
    media = read(out / "media.json")
    require(
        media["freeze_sha256"] == sha(out / "freeze.json") and len(media["cases"]) == 120,
        "Authoring freeze differs",
    )
    cv2.setNumThreads(1)
    module, task = neural.mp_task()
    detector = load_yunet(native.geometry.WEIGHT)
    cache, scores = {}, []
    with closing(task):
        for row in media["cases"]:
            path = Path(row["path"])
            require(sha(path) == row["file_sha256"], "Frozen delivered input differs")
            table, pictures, pcm = native.video(path)
            require(
                original_audio.sha(pcm.tobytes()) == row["whole_pcm_sha256"]
                and [original_audio.sha(x.tobytes()) for x in pictures] == row["native_rgb"],
                "Held native media differs",
            )
            if row["ordinal"] not in cache:
                observed, grid, faces, patches, _, _ = existing.observe(path, row["file_sha256"], detector)
                require(np.array_equal(pcm, observed), "Current check and candidate soundtrack inputs differ")
                observations = [neural.observed(module, task, picture) for picture in pictures]
                geometry = [value for _, value in observations]
                boxes = [
                    mouth_crop(mesh, picture) if mesh is not None else None
                    for (mesh, _), picture in zip(observations, pictures, strict=True)
                ]
                cache[row["ordinal"]] = geometry, grid, faces, patches, boxes
            geometry, grid, faces, patches, boxes = cache[row["ordinal"]]
            proposal = measure(geometry, pcm)
            current = baseline.baseline.current(faces, patches, pcm.astype(np.float32) / 32768, grid)
            if proposal["status"] == "local_timing_change":
                for event in proposal["events"]:
                    frame, anchor = event["picture"], event["acoustic_anchor"]
                    x0, y0, x1, y1 = boxes[frame]
                    a, b = anchor - 640, anchor + 640
                    event.update(
                        native_pts=[table.pts[frame], table.pts[frame + 1]],
                        time_base=table.time_base,
                        frame_sha256s=row["native_rgb"][frame : frame + 2],
                        crop_xyxy=[x0, y0, x1, y1],
                        crop_sha256s=[
                            original_audio.sha(pictures[k][y0:y1, x0:x1].tobytes())
                            for k in (frame, frame + 1)
                        ],
                        pcm_range=[a, b],
                        pcm_sha256=original_audio.sha(pcm[a:b].tobytes()),
                    )
            result = {
                "ordinal": row["ordinal"],
                "group": row["group"],
                "role": row["role"],
                "span": row["span"],
                "shift": row["shift"],
                "proposal": proposal,
                "current": {"measured": current["measured_windows"], "ranges": current["ranges"]},
                "correct": localized(proposal, row["span"], row["shift"]),
                "current_correct": bool(
                    row["span"]
                    and row["shift"]
                    and any(baseline.qualifies(flag, row["span"], row["shift"]) for flag in current["ranges"])
                ),
            }
            scores.append(result)
            print(row["ordinal"], row["role"], proposal["status"], result["correct"], flush=True)
    write(
        out / "readout.json",
        {
            "freeze_sha256": sha(out / "freeze.json"),
            "media_sha256": sha(out / "media.json"),
            "scores": scores,
        },
    )


def report(out):
    result = read(out / "readout.json")
    require(
        result["freeze_sha256"] == sha(out / "freeze.json")
        and result["media_sha256"] == sha(out / "media.json"),
        "Readout bindings differ",
    )
    results = result["scores"]
    by_role = {}
    for role, (span, shift) in ROLES.items():
        subset = [r for r in results if r["role"] == role]
        by_role[role] = {
            "local_shift_truth": bool(span and shift),
            "views": len(subset),
            "assessed": sum(r["proposal"]["status"] != "unassessed" for r in subset),
            "flags": sum(r["proposal"]["status"] == "local_timing_change" for r in subset),
            "correct": sum(r["correct"] for r in subset),
            "current_assessed": sum(bool(r["current"]["measured"]) for r in subset),
            "current_flags": sum(bool(r["current"]["ranges"]) for r in subset),
            "current_correct": sum(r["current_correct"] for r in subset),
        }
    criteria = RELEASE["gate"]
    correct_groups = len({r["group"] for r in results if r["correct"]})
    gates = {
        "correct": sum(r["correct"] for r in results) >= criteria["correct"],
        "groups": correct_groups >= criteria["correct_source_groups"],
        "coverage": sum(r["assessed"] for r in by_role.values())
        >= criteria["minimum_assessed_fraction"] * len(results),
        "control_flags": sum(r["flags"] for r in by_role.values() if not r["local_shift_truth"])
        <= criteria["maximum_control_flags"],
    }
    write(
        out / "aggregate.json",
        {
            "schema": "chipmunk.grid-single-clip-release-study/1",
            "method": RELEASE,
            "discovery_roles": read(out / "discovery.json")["rows"],
            "held": by_role,
            "correct_groups": correct_groups,
            "gates": gates,
            "advances": all(gates.values()),
            "freeze_sha256": result["freeze_sha256"],
            "readout_sha256": sha(out / "readout.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("discovery", "freeze", "author", "score", "report"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.phase == "discovery":
        args.output.mkdir(mode=0o700, parents=True)
    globals()[args.phase](args.output)


if __name__ == "__main__":
    main()
