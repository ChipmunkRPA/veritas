# SPDX-License-Identifier: Apache-2.0
"""Test two complementary picture signals for continuous local mouth redraws."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-mouth-motion-transport/1"
ARMS = ("deformation", "detail_transport", "agreement")
SPANS = (8, 12, 16, 20)
FLANK = 5
FLOORS = (.12, .10, .10)
SPLITS = {"development": 56, "held": 146, "codec_stress": 34}
CONTROLS = ("original", "dubbed", "silent", "cadence")
ROOT = Path(__file__).resolve().parents[2]
STRESS_PLAN = ROOT / "docs/evaluation-panels/video-lipsync-repeat-public-speech-sources-2026-11-13.json"
LOCKED = {"freeze.json": "af7feee3a877e09414ff92892b8af923c2f2441a4007596eacf4910bd6e89ff3",
          "codec_stress_roster.json": "0ab4c837497e7d5f2c4ba890f7a4c1830f448d7b27f57e2e5a11aa28f4a35cc8"}
BOUND = {"before": 0, "first": 1, "during": 2, "after": 3}
SPEC = {
    "transport": "Estimate forward and return flow from low-pass neighboring native pictures in eye-normalized coordinates. "
                 "At pixels with verified geometry and a stable round trip, compare the signed fine-detail vectors transported "
                 "with this flow. Compare mouth skin with skin above and beside it. Fine detail is not used for the flow fit.",
    "deformation": "At mouth-shell and matched reference points, measure the discrepancy between the flow of each point and "
                   "opposite points a few pixels away. Use the worse of horizontal and vertical normalized bending, conditioned "
                   "on stable flow and comparable local texture.",
    "agreement": "Require both visual cues at the same frame for this arm. Report each cue independently as well. "
                 "The cues use shared geometry and motion support; their results are not independent probabilities.",
    "window": "Require verified upright native pictures, one shot, frame steps 22–55 ms near that clip’s median, and at least "
              "80% support in each flank and the selected 8, 12, 16 or 20-picture window. Select the greatest positive core "
              "difference against both five-frame flanks. Commit before, first, during and first following picture/crop hashes.",
    "calibration": "Lock the third-largest ordinary development maximum with arm-specific floors. Report isolated and "
                   "joint signals. Every actual alert on ordinary editing or untouched footage counts against specificity.",
    "ordinary_extra": "Use the first planned offset of four previously acquired public speech sources outside GRID/Lombard. "
                      "Create a common H.264 25 fps delivery and remux unchanged pictures with time-shifted or silent audio. "
                      "Separately retime lossless pictures with fixed alternating 32 and 48 ms presentation steps.",
    "gates": "At least four strict LivePortrait hits in three groups, the one strict MuseTalk hit, at least 70% measured "
             "each neural family and ordinary controls, at most two ordinary alerts and none on untouched, recompressed, "
             "hard-cut, dubbed, silent or cadence-altered controls. Require exact native interval IoU of at least one half.",
    "scope": "Single-upload picture comparison without previous media, sound, text, model/renderer output, or prior region "
             "selections. A candidate would contribute zero AI confidence before new independent transfer.",
}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def frozen_inputs(archive):
    rows, input_pins = roster(archive)
    return {"schema": SCHEMA, "protocol": SPEC,
            "prior_rosters": LOCKED, "source_count": dict(Counter(row["split"] for row in rows)),
            "ordinary_extra_plan_sha256": sha(STRESS_PLAN), "face_geometry_sha256": input_pins["geometry"],
            "face_geometry_weights_sha256": input_pins["detector_weights"]}


def points_for(rows):
    result = {}
    for arm, floor in zip(ARMS, FLOORS, strict=True):
        scores = sorted((row["arms"][arm]["score"] or 0 for row in rows), reverse=True)
        result[arm] = max(floor, scores[2])
    return result


def roster(archive):
    for name, digest in LOCKED.items():
        require(sha(archive / name) == digest, "Previously held archive input roster differs")
    old, stress = (read(archive / name) for name in LOCKED)
    rows = [*old["rows"], *stress["rows"]]
    require(Counter(row["split"] for row in rows) == SPLITS, "Input roster changed")
    return rows, old["inputs_sha256"]


def pair_observation(a, b, mouth, elapsed):
    import cv2
    import numpy as np

    # The two measurements share a low-pass motion fit, never a fine-detail fit.
    def flow(first, second):
        low = [cv2.GaussianBlur(frame, (5, 5), 1.2) for frame in (first, second)]
        return cv2.calcOpticalFlowFarneback(*low, None, .5, 2, 13, 3, 5, 1.2, 0)

    forward, reverse = flow(a, b), flow(b, a)
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]].astype(np.float32)
    displaced = np.stack((xx, yy), axis=-1) + forward

    def warp(values):
        return cv2.remap(values, displaced, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    cycle = np.linalg.norm(forward + warp(reverse), axis=2)
    dx, dy = xx - mouth[0], yy - mouth[1]
    # Keep the textured skin shell next to the mouth, apart from teeth and lip opening.
    mouth_mask = ((dx / 22) ** 2 + (dy / 15) ** 2 <= 1) & ((dx / 13) ** 2 + (dy / 6) ** 2 >= 1.15)
    surrounding = (((dx-27) / 9) ** 2 + ((dy+6) / 12) ** 2 <= 1)
    surrounding |= (((dx+27) / 9) ** 2 + ((dy+6) / 12) ** 2 <= 1)
    surrounding |= ((dx / 15) ** 2 + ((dy+23) / 7) ** 2 <= 1)
    surrounding &= ~mouth_mask
    valid = (cycle <= 1.10) & (xx >= 5) & (yy >= 5) & (xx < a.shape[1]-5) & (yy < a.shape[0]-5)
    valid &= np.all((displaced >= 2) & (displaced < np.asarray(a.shape[::-1])-2), axis=2)
    gradient = np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1)) / 8
    # Match textured regions by using the same usable gradient bins in both areas.
    bend = []
    for axis in (0, 1):
        deviation = np.roll(forward, 3, axis=axis) + np.roll(forward, -3, axis=axis) - 2*forward
        bend.append(np.linalg.norm(deviation, axis=2) * (.04 / elapsed))
    bending = np.maximum(*bend)

    high = [frame.astype(np.float32) - cv2.GaussianBlur(frame.astype(np.float32), (5, 5), 1.0) for frame in (a, b)]
    previous, current = high[0], warp(high[1])
    def local_mean(values):
        return cv2.boxFilter(values, -1, (5, 5))

    first_energy, second_energy = local_mean(previous*previous), local_mean(current*current)
    coherence = local_mean(previous*current) / np.sqrt((first_energy+.7)*(second_energy+.7))
    signal = []
    for edge_low, edge_high in ((2, 6), (6, 13), (13, 27)):
        available = valid & (gradient >= edge_low) & (gradient < edge_high)
        mouth_band, surround_band = mouth_mask & available, surrounding & available
        if min(np.count_nonzero(mouth_band), np.count_nonzero(surround_band)) < 18:
            continue
        a_bend, b_bend = (float(np.median(bending[mask])) for mask in (mouth_band, surround_band))
        a_coh, b_coh = (float(np.median(coherence[mask])) for mask in (mouth_band, surround_band))
        signal.append((math.log((a_bend+.08)/(b_bend+.08)), b_coh-a_coh))
    if len(signal) < 2:
        return (np.nan,) * len(ARMS)
    first, second = np.median(signal, axis=0)
    return (float(first), float(second), float(min(first, second*3)))


def select(table, shots, values, anchors):
    import numpy as np

    windows = {}
    for width in (FLANK, *SPANS):
        parts = np.lib.stride_tricks.sliding_window_view(values, width, axis=0)
        good = np.isfinite(parts).sum(axis=-1) * 5 >= width * 4
        windows[width] = np.where(good, np.ma.median(np.ma.masked_invalid(parts), axis=-1).filled(np.nan), np.nan)
    best = {arm: {"measured": False, "score": None} for arm in ARMS}
    for width in SPANS:
        for start in range(FLANK, len(table)-width-FLANK+1):
            end = start+width
            picked = [anchors[i] for i in (start-1, start, (start+end)//2, end)]
            if shots[start-FLANK] != shots[end+FLANK-1] or not all(picked):
                continue
            center = windows[width][start]
            flanks = np.maximum(windows[FLANK][start-FLANK], windows[FLANK][end])
            for arm, amount, middle, side in zip(ARMS, center-flanks, center, flanks, strict=True):
                if not np.isfinite(amount):
                    continue
                best[arm]["measured"] = True
                score = round(float(amount), 8)
                if score > (best[arm]["score"] or 0):
                    best[arm].update(score=score, frames=[start, end], anchors=picked,
                                     core=round(float(middle), 8), flank=round(float(side), 8))
    return best


def inspect(path, detector, cache):
    import numpy as np

    from app.detectors.video.decode import decode_frames, probe_frames
    from app.detectors.video.shots import ShotDetector
    from scripts.video_mouth_jaw_transfer import native_bounds, raw_digest
    from scripts.video_mouth_reconstruction_transfer import aligned_face

    table = probe_frames(path, timeout_s=15)
    require(not (table.truncated or table.interpolated or table.rotation) and 40 <= len(table) <= 100,
            "Expected verified short native clip")
    require(all(a < b for a, b in zip(table.pts, table.pts[1:])), "Native PTS must strictly increase")
    shot_detector = ShotDetector(table)
    observations = np.full((len(table), len(ARMS)), np.nan)
    step = float(np.median(np.diff(table.pts)) * table.tb)
    anchors, shots = [], []
    previous = None
    for frame in decode_frames(path, table, None, timeout_s=28):
        i = len(anchors)
        require(frame.index == i and frame.pts_verified and frame.pts == table.pts[i], "Decoded picture clock differs")
        shot_detector.update(i, frame.rgb)
        shots.append(len(shot_detector.boundaries))
        digest = raw_digest(frame.rgb)
        if digest not in cache:
            location = aligned_face(frame.rgb, detector)
            if location is None:
                cache[digest] = None
            else:
                bounds = native_bounds(location)
                x0, y0, x1, y1 = bounds
                cache[digest] = (location[0], location[2], bounds, raw_digest(frame.rgb[y0:y1, x0:x1]))
        current = cache[digest]
        anchors.append(None if current is None else {"frame": i, "pts": frame.pts, "rgb_sha256": digest,
                       "crop_xyxy": current[2], "crop_sha256": current[3]})
        if current is not None and previous is not None and shots[-1] == shots[-2]:
            delta = float((table.pts[i]-table.pts[i-1])*table.tb)
            if max(.022, step*.8) <= delta <= min(.055, step*1.2):
                mouth = (current[1]+previous[1])*.5
                observations[i] = pair_observation(previous[0], current[0], mouth, delta)
        previous = current
    require(len(anchors) == len(table), "Incomplete native picture delivery")
    return table, select(table, shots, observations, anchors), dict(zip(ARMS, map(int, np.isfinite(observations).sum(axis=0))))


def capture(table, readings, support):
    return {"arms": readings, "frame_support": support, "time_base": table.time_base, "pts": list(table.pts),
            "width": table.width, "height": table.height}


def labeled(input_row, table, readings, support):
    edit = input_row.get("edit")
    truth = [str(table.pts[i]*table.tb) for i in edit] if edit else input_row.get("native_truth")
    return {"family": input_row["family"], "role": input_row["role"], "id": input_row["id"], "group": input_row["group"],
            "source_sha256": input_row["sha"], "strict": input_row.get("aligned", False), "truth": truth,
            **capture(table, readings, support)}


def score(archive, partition, work):
    import cv2

    from app.detectors.image.faces import load_yunet
    from scripts.video_mouth_reconstruction_transfer import WEIGHT

    rows, input_pins = roster(archive)
    from scripts import video_mouth_reconstruction_transfer as geometry_source

    require(sha(WEIGHT) == input_pins["detector_weights"] and sha(Path(geometry_source.__file__)) == input_pins["geometry"],
            "Different face geometry or weights")
    cv2.setNumThreads(1)
    detector = load_yunet(WEIGHT)
    groups = {}
    for row in rows:
        if row["split"] == partition:
            groups.setdefault((row["family"], row["id"]), []).append(row)
    outcomes = []
    for group in groups.values():
        cache = {}
        for row in group:
            path = Path(row["file"])
            require(sha(path) == row["sha"], "Source video bytes differ")
            outcomes.append(labeled(row, *inspect(path, detector, cache)))
        print(partition, len(outcomes), flush=True)
    result = {"schema": SCHEMA, "protocol": SPEC, "partition": partition, "rows": outcomes}
    write(work / f"{partition}.json", result)
    if partition == "development":
        write(work / "points.json", {"points": points_for(outcomes), "development_sha256": sha(work / "development.json")})
        print(points_for(outcomes), flush=True)


def materialize_controls(source_cache, work):
    import subprocess

    from scripts.video_temporal_insert_native import ffmpeg, input_file, process

    for index, source in enumerate(read(STRESS_PLAN)["sources"]):
        path = source_cache / (source["id"]+".webm")
        require(sha(path) == source["sha256"], "Public conversation bytes differ")
        original = work / f"control-{index}-original.mp4"
        dub = work / f"control-{index}-dubbed.mov"
        silent = work / f"control-{index}-silent.mov"
        cadence = work / f"control-{index}-cadence.mp4"
        offset = str(source["offsets"][0])
        process([*ffmpeg(), "-ss", offset, *input_file(path), "-t", "2.8", "-map", "0:v:0", "-map", "0:a:0",
                 "-vf", "fps=25,scale=480:360:flags=area,setsar=1", "-c:v", "libx264", "-threads", "1", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "1", "-ar", "16000", "-video_track_timescale",
                 "25000", str(original)])
        for target, audio in ((dub, "atrim=start=0.32,asetpts=PTS-STARTPTS,apad,atrim=end=2.8"),
                              (silent, "volume=0")):
            process([*ffmpeg(), *input_file(original), "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy",
                     "-af", audio, "-c:a", "pcm_s16le", "-video_track_timescale", "25000", str(target)])
        command = [*ffmpeg(), *input_file(original), "-map", "0:v:0", "-an",
                   "-vf", "settb=1/25000,setpts=(1000*N-200*mod(N\\,2))", "-fps_mode", "passthrough",
                   "-enc_time_base", "1/25000", "-video_track_timescale", "25000", "-use_editlist", "0",
                   "-c:v", "libx264", "-qp", "0", "-bf", "0",
                   "-pix_fmt", "yuv420p", "-threads", "1", str(cadence)]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, timeout=30)
        yield index, source, (original, dub, silent, cadence)


def ordinary_controls(source_cache, work):
    import cv2

    from app.detectors.image.faces import load_yunet
    from scripts.video_mouth_reconstruction_transfer import WEIGHT

    cv2.setNumThreads(1)
    detector = load_yunet(WEIGHT)
    cases = []
    for index, source, paths in materialize_controls(source_cache, work):
        cache = {}
        for role, path in zip(CONTROLS, paths, strict=True):
            case = {"family": "public_conversation", "role": role, "id": index, "group": index, "sha": sha(path)}
            cases.append({**labeled(case, *inspect(path, detector, cache)), **control_streams(path)})
        print("ordinary controls", source["id"], flush=True)
    review_controls(cases)
    write(work / "ordinary_extra.json", {"schema": SCHEMA, "protocol": SPEC, "partition": "ordinary_extra", "rows": cases})


def control_streams(path):
    from scripts.video_temporal_insert_native import ffmpeg, input_file, process

    packets = process([*ffmpeg(), *input_file(path), "-map", "0:v:0", "-an", "-vf", "format=yuv420p",
                       "-fps_mode", "passthrough", "-threads", "1", "-f", "rawvideo", "pipe:1"])
    audio = process([*ffmpeg(), *input_file(path), "-map", "0:a:0?", "-vn", "-ac", "1", "-ar", "16000",
                     "-f", "s16le", "pipe:1"]) if path.name.endswith(("-original.mp4", "-dubbed.mov", "-silent.mov")) else b""
    return {"control_video_sha256": hashlib.sha256(packets).hexdigest(),
            "control_audio_sha256": hashlib.sha256(audio).hexdigest(), "control_audio_bytes": len(audio),
            "control_audio_is_silent": not any(audio)}


def review_controls(rows):
    require(len(rows) == len(CONTROLS)*len(read(STRESS_PLAN)["sources"]), "Incomplete ordinary source panel")
    for i in range(0, len(rows), len(CONTROLS)):
        original, dub, silent, cadence = rows[i:i+len(CONTROLS)]
        require([row["role"] for row in (original, dub, silent, cadence)] == list(CONTROLS) and
                all(row["family"] == "public_conversation" and row["id"] == row["group"] == i//len(CONTROLS)
                    and row["truth"] is None and not row["strict"] for row in (original, dub, silent, cadence)),
                "Ordinary source and condition order differs")
        observed = ("pts", "time_base", "frame_support", "arms")
        for changed in (dub, silent):
            require(all(original[k] == changed[k] for k in observed), "Changing audio changed picture-only evidence")
        require(len({row["control_video_sha256"] for row in (original, dub, silent, cadence)}) == 1
                and len({row["control_audio_sha256"] for row in (original, dub, silent)}) == 3
                and all(row["control_audio_bytes"] > 0 for row in (original, dub, silent))
                and silent["control_audio_is_silent"] and not cadence["control_audio_bytes"],
                "Audio-only and cadence variants do not bind the same decoded source pictures")
        recorded = [tick * Fraction(cadence["time_base"]) for tick in cadence["pts"]]
        require(recorded == [Fraction(25*n-5*(n%2), 625) for n in range(len(original["pts"]))],
                "Cadence control picture ticks differ")


def reopen_frames(path, row):
    import subprocess

    from scripts.video_temporal_insert_native import ffmpeg, input_file

    dimensions = row["width"], row["height"]
    frame_size = dimensions[0]*dimensions[1]*3
    anchors = {item["frame"]: item for arm in row["arms"].values() for item in arm.get("anchors", ())}
    if not anchors:
        return 0
    ffprobe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts",
                                      "-of", "json", str(path)], timeout=25)
    require([int(item["pts"]) for item in json.loads(ffprobe)["frames"]] == row["pts"], "Independent native clock differs")
    wanted = "+".join(f"eq(n\\,{i})" for i in sorted(anchors))
    raw = subprocess.check_output([*ffmpeg(), "-copyts", *input_file(path), "-map", "0:v:0", "-an", "-vf",
                                   "select="+wanted+",format=rgb24", "-fps_mode", "passthrough", "-threads", "1", "-f",
                                   "rawvideo", "pipe:1"], timeout=25)
    require(len(raw) == frame_size*len(anchors), "Independent FFmpeg did not retrieve each evidence picture")
    for j, i in enumerate(sorted(anchors)):
        rgb = memoryview(raw)[j*frame_size:(j+1)*frame_size]
        anchor = anchors[i]
        require(hashlib.sha256(rgb).hexdigest() == anchor["rgb_sha256"], "Independent source screenshot differs")
        x0, y0, x1, y1 = anchor["crop_xyxy"]
        crop = hashlib.sha256()
        for y in range(y0, y1):
            crop.update(rgb[(y*dimensions[0]+x0)*3:(y*dimensions[0]+x1)*3])
        require(crop.hexdigest() == anchor["crop_sha256"], "Independent source evidence crop differs")
    return len(anchors)


def verify_evidence(archive, work):
    inputs, _ = roster(archive)
    paths = {row["sha"]: Path(row["file"]) for row in inputs}
    checks = Counter()
    for partition in ("held", "codec_stress"):
        for row in read(work / f"{partition}.json")["rows"]:
            path = paths[row["source_sha256"]]
            require(sha(path) == row["source_sha256"], "Input source hash differs")
            if row["strict"] or any((reading["score"] or 0) > .10 for reading in row["arms"].values()):
                checks[partition] += reopen_frames(path, row)
    return checks


def overlap(a, b):
    start, stop = map(Fraction, a)
    other_start, other_end = map(Fraction, b)
    return max(Fraction(0), min(stop, other_end)-max(start, other_start))/(max(stop, other_end)-min(start, other_start))


def summary(rows, points):
    totals = {}
    for arm in ARMS:
        counts = Counter()
        by_role = {}
        recovered_groups = set()
        for row in rows:
            pts, tb = row["pts"], Fraction(row["time_base"])
            require(40 <= len(pts) <= 100 and tb > 0 and all(a < b for a, b in zip(pts, pts[1:])), "Invalid native clock")
            require(0 <= row["frame_support"][arm] <= len(pts), "Invalid number of supported picture transitions")
            proposal = row["arms"][arm]
            amount = proposal["score"]
            aligned = False
            if amount is not None:
                a, b = proposal["frames"]
                require(a >= FLANK and b+FLANK <= len(pts) and b-a in SPANS and proposal["measured"]
                        and amount > 0 and abs(proposal["core"]-proposal["flank"]-amount) <= 2e-8,
                        "Invalid proposal amplitude or frame interval")
                observed = [a-1, a, (a+b)//2, b]
                require(len(proposal["anchors"]) == len(BOUND), "Incomplete evidence frames")
                for index, anchor in zip(observed, proposal["anchors"], strict=True):
                    x0,y0,x1,y1 = anchor["crop_xyxy"]
                    require((anchor["frame"], anchor["pts"]) == (index, pts[index]) and
                            0 <= x0 < x1 <= row["width"] and 0 <= y0 < y1 <= row["height"] and
                            all(isinstance(anchor[key], str) and len(anchor[key]) == 64 and
                                all(c in "0123456789abcdef" for c in anchor[key]) for key in ("rgb_sha256", "crop_sha256")),
                            "Invalid evidence picture timestamp, crop, or hash")
                aligned = bool(row["truth"] and overlap([pts[a]*tb, pts[b]*tb], row["truth"]) >= Fraction(1, 2))
            flag = amount is not None and amount > points[arm]
            marks = {"files": 1, "measured": proposal["measured"], "flags": flag, "exact_enough": flag and aligned,
                     "below_threshold_useful": aligned and not flag}
            role = f"{row['family']}:{row['role']}"
            by_role.setdefault(role, Counter()).update(marks)
            label = "positive" if row["truth"] else "ordinary"
            counts.update({f"{label}_{k}": v for k, v in marks.items()})
            if row["strict"]:
                counts.update({f"strict_{row['family']}_{k}": v for k, v in marks.items()})
            if row["strict"] and row["family"] == "lombard" and marks["exact_enough"]:
                recovered_groups.add(row["group"])
            if flag and not row["truth"] and row["role"] in ("original", "reencode", "hard_cut", "dubbed", "silent", "cadence"):
                counts["essential_ordinary_flags"] += 1
        gates = {"strict_liveportrait": counts["strict_lombard_exact_enough"] >= 4 and len(recovered_groups) >= 3,
                 "strict_musetalk": counts["strict_musetalk_exact_enough"] == counts["strict_musetalk_files"] == 1,
                 "coverage": all(counts[f"{k}_files"] and 10*counts[f"{k}_measured"] >= 7*counts[f"{k}_files"]
                                 for k in ("strict_lombard", "strict_musetalk", "ordinary")),
                 "ordinary": counts["ordinary_flags"] <= 2 and counts["essential_ordinary_flags"] == 0}
        totals[arm] = {"counts": counts, "role": by_role, "strict_liveportrait_groups": len(recovered_groups), "gates": gates}
    return totals


def portable(ledger):
    documents = {phase: read(ledger / f"{phase}.json") for phase in (*SPLITS, "ordinary_extra")}
    frozen = read(ledger / "points.json")
    require(frozen["development_sha256"] == sha(ledger / "development.json"), "Development commitment differs")
    return {"points": frozen["points"], "scored_inputs": documents}


def verify_portable(receipt):
    documents, points = receipt["scored_inputs"], receipt["points"]
    require(points == points_for(documents["development"]["rows"]), "Development points differ")
    report = {}
    for partition, expected_count in {**SPLITS, "ordinary_extra": 16}.items():
        record = documents[partition]
        require((record["schema"], record["protocol"], record["partition"], len(record["rows"])) ==
                (SCHEMA, SPEC, partition, expected_count), "Results differ from frozen roster or method")
        if partition == "ordinary_extra":
            review_controls(record["rows"])
        report[partition] = summary(record["rows"], points)
    return {"points": points, "summary": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "held", "codec_stress", "ordinary_extra", "review",
                                          "native_review", "pack"))
    parser.add_argument("source", type=Path)
    parser.add_argument("work", type=Path, nargs="?")
    args = parser.parse_args()
    if args.phase == "freeze":
        print(json.dumps(frozen_inputs(args.source), sort_keys=True, indent=2))
    elif args.phase == "native_review" and args.work:
        print(json.dumps(verify_evidence(args.source, args.work), sort_keys=True))
    elif args.phase == "pack" and args.work:
        receipt = portable(args.source)
        verify_portable(receipt)
        with args.work.open("xb") as stream:
            stream.write(gzip.compress(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
    elif args.phase == "review":
        receipt = portable(args.source) if args.source.is_dir() else json.loads(gzip.decompress(args.source.read_bytes()))
        print(json.dumps(verify_portable(receipt), sort_keys=True))
    else:
        if args.work is None:
            parser.error("Scoring needs an output directory")
        if args.phase == "ordinary_extra":
            ordinary_controls(args.source, args.work)
        else:
            score(args.source, args.phase, args.work)


if __name__ == "__main__":
    main()
