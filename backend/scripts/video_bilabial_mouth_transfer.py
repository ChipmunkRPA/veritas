# SPDX-License-Identifier: Apache-2.0
"""Compare blind speech timestamps with independently measured lip releases."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts import video_neural_phone_clock as phone

SCHEMA = "chipmunk.video-bilabial-mouth-transfer/1"
FACE_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
RATE = phone.RATE
MAX_CLOCK_ERROR = 1440
MINIMUM_OFFSET, MAXIMUM_OFFSET = 1760, 5440
DISTINCT_MARGIN = 640
ROLES = ("original", "filtered", "gain", "global_early", "global_late", "early", "late")
EDITS = {"early": -phone.SHIFT, "late": phone.SHIFT}
METHODS = ("old_nominal", "neural_nominal", "old_flanked", "neural_flanked")
SHA256 = re.compile(r"[0-9a-f]{64}")
DESIGN = {
    "observations": "Observe silent native pictures independently. Find sustained low aperture with a local reopening, "
                    "using the exact three eye-normalized inner-lip gaps. Read speech phones without supplying labels.",
    "bias": "First-cohort genuine development only. Fit the median phone-minus-release offset from a matching labeled "
            "stop and a unique nearby visible release. Fit each phone method separately.",
    "candidate": "An emitted B/P event must have a unique nearest release with absolute residual offset 110–340 ms. "
                 "Require at least 40 ms separation from the next visible release by absolute distance. "
                 "The flanked arm also requires distinct aligned phone/visible pairs before and after the event.",
    "points": "On second-cohort genuine control speakers, use strict excess above the highest and second-highest "
              "speaker maximum ordinary scores. Fit before recounting either held cohort.",
    "controls": "Original, low-pass, local gain, and two constant 200 ms shifts. Replay exact earlier 700 ms local splices. "
                "Preserve genuine native pictures in every case.",
    "utility": "Sound minus picture sign within 80 ms and temporal overlap of at least 0.40 on the soundtrack clock. "
               "The evidence span includes both event times and 80 ms of context at either edge.",
}


def source_rows(first, second):
    paths = list(map(Path, (first, second)))
    panels = [json.loads(path.read_bytes()) for path in paths]
    for panel in panels:
        phone.require(phone.review(panel) == panel["outcomes"], "Parent timing benchmark did not replay")
    phone.verify_previous(panels[1], paths[0])
    return [(i, row) for i, panel in enumerate(panels, 1) for row in panel["rows"]]


def treatments(original, edit):
    import numpy as np

    authored = dict(phone.variants(original, edit))
    lo, hi = edit
    gains = np.full(hi - lo, .65)
    gains[:128] = np.linspace(1, .65, 128, endpoint=False)
    gains[-128:] = gains[:128][::-1]
    quiet = original.copy()
    quiet[lo:hi] = np.rint(original[lo:hi] * gains).astype("<i2")
    authored["gain"] = quiet
    for name, amount in EDITS.items():
        out = np.zeros_like(original)
        if amount > 0:
            out[amount:] = original[:-amount]
        else:
            out[:amount] = original[-amount:]
        authored["global_" + name] = out
    return {name: authored[name] for name in ROLES}


def visual_releases(frames):
    events = []
    for i in range(1, len(frames) - 4):
        near = [row.get("gap") for row in frames[i - 1:i + 5]]
        if None in near:
            continue
        before, closed, opened, *following = near
        rise = max(opened, *following) - closed
        if before <= .075 and closed <= .055 < opened and rise >= .032:
            events.append({
                "frames": [i, i + 1], "sample": (frames[i]["sample"] + frames[i + 1]["sample"]) // 2,
                "reopening": round(rise, 6),
            })
    return events


def picture_observations(path, samples, mp, task, yunet):
    import cv2
    import numpy as np

    from app.detectors.image.faces import detect_faces
    from app.detectors.video import lipsync, sampling, xcorr
    from app.detectors.video.decode import decode_frames, probe_frames
    from app.evidence.video_reference_timing import _mesh

    table = probe_frames(path, cap=160, timeout_s=18, max_pixels=1920 * 1080)
    phone.require(len(table) >= 24 and not table.truncated and not table.interpolated
                  and all(a < b for a, b in zip(table.pts, table.pts[1:])), "Native picture clock is unavailable")
    times = (np.array(table.pts) - table.pts[0]) * float(table.tb)
    duration = sampling.duration_s(table)
    grid = lipsync.tick_grid(times, duration, samples, audio_start_s=0.)
    wanted = set(grid.frames) - {-1}
    frames, faces, native_patches = [], {}, {}
    for frame in decode_frames(path, table, list(range(len(table))), timeout_s=25):
        phone.require(frame.pts_verified and frame.index == len(frames) and frame.pts == table.pts[frame.index],
                      "Decoded picture differs from its native timestamp")
        record = {"index": frame.index, "pts": frame.pts,
                  "sample": round((frame.pts - table.pts[0]) * table.tb * RATE), "gap": None,
                  "rgb_sha256": phone.digest(frame.rgb.tobytes())}
        if observed := _mesh(task, mp, frame.rgb):
            features, box = observed
            record["gap"] = round(float(np.median(features[:3])), 7)
            x0, y0, x1, y1 = box
            record["mouth_box"] = box
            record["mouth_crop_sha256"] = phone.digest(frame.rgb[y0:y1, x0:x1].tobytes())
        frames.append(record)
        if frame.index in wanted:
            detected = detect_faces(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), detector=yunet, min_size=48)
            if len(detected) == 1:
                face = detected[0]
                faces[frame.index] = {"frame_index": frame.index, "pts": frame.pts, "landmarks": face.landmarks,
                                      "bbox": dict(zip(("x", "y", "w", "h"), face.box(), strict=True))}
                native_patches[frame.index] = lipsync.mouth_patch(frame.rgb, faces[frame.index])
    phone.require(len(frames) == len(table), "A native source picture was missed")
    patches = {i: native_patches[frame] for i, frame in enumerate(grid.frames) if frame in native_patches}
    return frames, table.time_base, (faces, patches, {i: xcorr.aperture(patch) for i, patch in patches.items()}, grid)


def observe(first, second, folder, runtime, model_path, checkpoint, wheel, output):
    import cv2
    import numpy as np

    from app.detectors.image.faces import load_yunet
    from app.detectors.video import lipsync
    from app.evidence.video_reference_timing import _landmarker, available
    from scripts.video_lipsync_signed_transitions import current
    from scripts.video_lipsync_window_schedule import FACE_FILE, FACE_SHA

    rows = source_rows(first, second)
    paths, _ = phone.source_archives(folder)
    phone.require(phone.sha(model_path) == FACE_SHA256 and phone.sha(FACE_FILE) == FACE_SHA,
                  "A reviewed face or geometry asset differs")
    sys.path.insert(0, str(runtime))
    phone.require(available(model_path), "The pinned face runtime could not load")
    output.mkdir(mode=0o700)
    neural, device = phone.neural_decoder(checkpoint)
    old = phone.old_decoder(wheel, output / "baseline")
    cv2.setNumThreads(1)
    yunet = load_yunet(FACE_FILE)
    result = []
    with (ZipFile(paths["audio"]) as audio, ZipFile(paths["front"]) as front,
          TemporaryDirectory(prefix="av-bilabial-") as staging, _landmarker(model_path) as (mp, task)):
        path = Path(staging) / "original.mov"
        for index, (cohort, row) in enumerate(rows, 1):
            data = {key: archive.read(row["members"][key]["name"]) for key, archive in (("front", front), ("audio", audio))}
            phone.require(all(phone.digest(value) == row["members"][key]["sha256"] for key, value in data.items()),
                          "Publisher originals differ from committed sources")
            path.write_bytes(data["front"])
            pcm = phone.pcm(data["audio"])
            frames, time_base, (faces, patches, lookup, grid) = picture_observations(path, len(pcm), mp, task, yunet)
            cases = {}
            for role, wave in treatments(pcm, row["edit"]).items():
                payload = phone.digest(wave.tobytes())
                if saved := row["measurements"].get(role):
                    phone.require(saved["sha256"] == payload, "The prior staged waveform differs")
                    audio_events = {name: saved[name] for name in ("old", "neural")}
                else:
                    audio_events = {"old": old(wave), "neural": neural(wave)}
                current_result = current(faces, patches, lookup, lipsync.vad(wave.astype(np.float32) / 32768), grid)
                cases[role] = {"pcm_sha256": payload, "existing": current_result, **audio_events}
            result.append({field: row[field] for field in ("speaker", "split", "source", "condition", "samples",
                           "target", "edit", "labels", "members")}
                          | {"cohort": cohort, "time_base": time_base, "frames": frames,
                             "releases": visual_releases(frames), "cases": cases})
            if index % 4 == 0:
                print("Measured picture and sound", index, "of", len(rows), flush=True)
    phone.write(output / "observed.json", {"schema": SCHEMA, "method": DESIGN,
                "geometry_sha256": FACE_SHA256, "incumbent_face_sha256": FACE_SHA, "device": device,
                "parents": [phone.sha(path) for path in (first, second)], "rows": result})


def bias_fit(rows):
    fitted = {}
    for method in ("old", "neural"):
        contributions = defaultdict(list)
        for row in rows:
            if row["cohort"] != 1 or row["split"] != "development":
                continue
            for label, sound in phone.match(row["cases"]["original"][method], row["labels"], phone.LABEL_PADDING):
                nearby = [r for r in row["releases"] if abs(r["sample"] - label[2]) <= 1600]
                if len(nearby) == 1:
                    contributions[row["speaker"]].append(sound[1] - nearby[0]["sample"])
        # Give each speaker the same weight across multiple stops on one track.
        values = [statistics.median(samples) for samples in contributions.values()]
        fitted[method] = {"value": round(statistics.median(values)) if values else 0,
                          "speakers": len(values), "anchors": sum(map(len, contributions.values()))}
    return fitted


def associate(audio, releases, bias):
    pairs = []
    for sound in audio:
        ranked = sorted((abs(sound[1] - bias - event["sample"]), order) for order, event in enumerate(releases))
        if not ranked or ranked[0][0] > MAXIMUM_OFFSET:
            continue
        distance, order = ranked[0]
        if len(ranked) > 1 and ranked[1][0] - distance < DISTINCT_MARGIN:
            continue
        release = releases[order]
        pairs.append({"audio": sound, "release": release, "raw_offset": sound[1] - release["sample"]})
    return pairs


def proposals(row, role, method, bias):
    audio_method, scope = method.split("_")
    pairs = associate(row["cases"][role][audio_method], row["releases"], bias)
    measured = 0
    strongest = None
    for candidate in pairs:
        sample = candidate["audio"][1]
        release = candidate["release"]
        flanks = []
        if scope == "flanked":
            lower, upper = min(sample, release["sample"]), max(sample, release["sample"])
            for side in (-1, 1):
                supports = [p for p in pairs if abs(p["raw_offset"] - bias) <= MAX_CLOCK_ERROR
                            and (max(p["audio"][1], p["release"]["sample"]) <= lower - 1600 if side < 0 else
                                 min(p["audio"][1], p["release"]["sample"]) >= upper + 1600)]
                if supports:
                    flanks.append(min(supports, key=lambda p: abs(p["audio"][1] - sample)))
            if len(flanks) != 2 or abs(flanks[0]["raw_offset"] - flanks[1]["raw_offset"]) > MAX_CLOCK_ERROR:
                continue
            baseline = sum(flank["raw_offset"] for flank in flanks) // 2
        else:
            baseline = bias
        measured += 1
        offset = candidate["raw_offset"] - baseline
        if not MINIMUM_OFFSET <= abs(offset) <= MAXIMUM_OFFSET:
            continue
        strength = round((abs(offset) - MAX_CLOCK_ERROR) * min(1., release["reopening"] / .08), 6)
        if strongest is not None and strongest["strength"] >= strength:
            continue
        start, end = min(sample, release["sample"]), max(sample, release["sample"])
        nearby = sorted({max(0, release["frames"][0] - 2), *release["frames"],
                         min(len(row["frames"]) - 1, release["frames"][1] + 2)})
        strongest = candidate | {
            "offset_samples": offset, "baseline_samples": baseline, "strength": strength,
            "span_samples": [max(0, start - 1280), min(row["samples"], end + 1280)],
            "witness": nearby, "flanks": flanks,
        }
    return measured, strongest


def points(rows, fit):
    boundaries = {}
    for method in METHODS:
        maxima = defaultdict(float)
        for row in rows:
            if row["cohort"] != 2 or row["split"] != "development":
                continue
            speaker = row["speaker"]
            bias = fit[method.split("_")[0]]["value"]
            scores = [found["strength"] for role in ROLES if role not in EDITS
                      and (found := proposals(row, role, method, bias)[1])]
            maxima[speaker] = max(maxima[speaker], max(scores, default=0.))
        ordered = sorted(maxima.values(), reverse=True)
        phone.require(len(ordered) == 4, "The calibration needs all four separate speakers")
        boundaries[method] = {"tight": ordered[0], "coverage": ordered[1], "speaker_maxima": dict(maxima)}
    return boundaries


def useful(span, offset, truth, signed):
    a, b = span
    c, d = truth
    overlap = min(b, d) - max(a, c)
    return abs(offset - signed) <= 1280 and 5 * overlap >= 2 * (max(b, d) - min(a, c))


def summarize(rows, fit, thresholds):
    counts = defaultdict(Counter)
    examples = []
    for row in rows:
        if row["split"] != "held":
            continue
        for role in ROLES:
            key = "local" if role in EDITS else "constant" if role.startswith("global") else "ordinary"
            truth = row["edit"]
            signed = EDITS.get(role)
            earlier = row["cases"][role]["existing"]
            existing = counts[f"current/{key}"]
            existing["files"] += 1
            existing["measured"] += int(bool(earlier["starts"]))
            existing["flagged"] += int(bool(earlier["ranges"]))
            good = bool(signed and any(
                lag is not None and useful([a * 320 + lag * 640, b * 320 + lag * 640], lag * 640, truth, signed)
                for _, a, b, lag in earlier["ranges"]
            ))
            existing["correct"] += int(good)
            for method in METHODS:
                measured, candidate = proposals(row, role, method, fit[method.split("_")[0]]["value"])
                correct = bool(candidate and signed and useful(candidate["span_samples"], candidate["offset_samples"], truth, signed))
                for policy in ("tight", "coverage"):
                    selected = bool(candidate and candidate["strength"] > thresholds[method][policy])
                    tally = counts[f"{method}/{policy}/{key}"]
                    tally["files"] += 1
                    tally["measured"] += int(bool(measured))
                    tally["flagged"] += int(selected)
                    tally["correct"] += int(selected and correct)
                    tally["correct_before_point"] += int(correct)
                    if selected:
                        examples.append({"source": row["source"], "cohort": row["cohort"], "role": role,
                                         "mode": method, "point": policy, "useful": correct, **candidate})
    return {"outcomes": {name: dict(value) for name, value in counts.items()}, "examples": examples}


def validate_rows(document, parents):
    phone.require(document["schema"] == SCHEMA and document["method"] == DESIGN
                  and document["geometry_sha256"] == FACE_SHA256 and document["parents"] == list(map(phone.sha, parents)),
                  "The visual method or committed acoustic parents differ")
    originals = source_rows(*parents)
    phone.require(len(originals) == len(document["rows"]), "Missing planned source rows")
    for (cohort, source), row in zip(originals, document["rows"], strict=True):
        fields = ("speaker", "split", "source", "condition", "samples", "target", "edit", "labels", "members")
        phone.require(cohort == row["cohort"] and all(row[key] == source[key] for key in fields)
                      and set(row["cases"]) == set(ROLES), "Source groups, waveforms or label roles changed")
        frames = row["frames"]
        scale = Fraction(row["time_base"])
        phone.require(len(frames) >= 24 and frames[0]["sample"] == 0 and scale > 0
                      and all(a["pts"] < b["pts"] for a, b in zip(frames, frames[1:]))
                      and all(frame["index"] == i and frame["sample"] == round((frame["pts"] - frames[0]["pts"]) * scale * RATE)
                              for i, frame in enumerate(frames)) and row["releases"] == visual_releases(frames),
                      "A picture clock or visible mouth event changed")
        for frame in frames:
            phone.require(SHA256.fullmatch(frame["rgb_sha256"]), "Missing native picture hash")
            if frame["gap"] is not None:
                x0, y0, x1, y1 = frame["mouth_box"]
                phone.require(math.isfinite(frame["gap"]) and frame["gap"] >= 0
                              and all(type(value) is int for value in (x0, y0, x1, y1))
                              and 0 <= x0 < x1 and 0 <= y0 < y1
                              and SHA256.fullmatch(frame["mouth_crop_sha256"]),
                              "Invalid visible mouth measurement")
        for role in phone.TREATMENTS:
            old = source["measurements"][role]
            case = row["cases"][role]
            phone.require(case["pcm_sha256"] == old["sha256"] and case["old"] == old["old"] and case["neural"] == old["neural"],
                          "An earlier acoustic measurement differs from its source")


def analyze(path, parents, output):
    document = json.loads(path.read_bytes())
    phone.write(output, document | {"result": recount(document, parents)})


def recount(document, parents):
    validate_rows(document, parents)
    fit = bias_fit(document["rows"])
    boundaries = points(document["rows"], fit)
    return {"bias": fit, "points": boundaries, **summarize(document["rows"], fit, boundaries)}


def review(document, parents):
    result = recount(document, parents)
    phone.require(document["result"] == result, "Held timing results changed under replay")
    return result["outcomes"]


def replay(path, parents, sources, output):
    from app.detectors.video.decode import decode_frames, probe_frames

    document = json.loads(path.read_bytes())
    review(document, parents)
    locations, _ = phone.source_archives(sources)
    witnesses = defaultdict(set)
    for example in document["result"]["examples"]:
        key = example["cohort"], example["source"]
        witnesses[key].update(example["witness"])
        for flank in example["flanks"]:
            witnesses[key].update(flank["release"]["frames"])

    tally = Counter()
    with (ZipFile(locations["audio"]) as audio, ZipFile(locations["front"]) as front,
          TemporaryDirectory(prefix="av-bilabial-replay-") as staging):
        media = Path(staging) / "original.mov"
        for row in document["rows"]:
            sound = audio.read(row["members"]["audio"]["name"])
            picture = front.read(row["members"]["front"]["name"])
            phone.require(phone.digest(sound) == row["members"]["audio"]["sha256"]
                          and phone.digest(picture) == row["members"]["front"]["sha256"],
                          "Publisher source bytes differ")
            for role, wave in treatments(phone.pcm(sound), row["edit"]).items():
                phone.require(phone.digest(wave.tobytes()) == row["cases"][role]["pcm_sha256"],
                              "A comparison waveform differs")
                tally["waveforms"] += 1
            media.write_bytes(picture)
            table = probe_frames(media, cap=160, timeout_s=18, max_pixels=1920 * 1080)
            records = row["frames"]
            phone.require(not table.truncated and not table.interpolated and table.time_base == row["time_base"]
                          and tuple(table.pts) == tuple(frame["pts"] for frame in records),
                          "The native picture clock differs")
            tally["native_timestamps"] += len(table)
            tally["source_pairs"] += 1
            wanted = witnesses.get((row["cohort"], row["source"]))
            if not wanted:
                continue
            decoded = set()
            for frame in decode_frames(media, table, sorted(wanted), timeout_s=25):
                record = records[frame.index]
                phone.require(frame.pts_verified and frame.pts == record["pts"]
                              and phone.digest(frame.rgb.tobytes()) == record["rgb_sha256"],
                              "A witness picture differs")
                decoded.add(frame.index)
                if box := record.get("mouth_box"):
                    x0, y0, x1, y1 = box
                    height, width = frame.rgb.shape[:2]
                    phone.require(x1 <= width and y1 <= height
                                  and phone.digest(frame.rgb[y0:y1, x0:x1].tobytes()) == record["mouth_crop_sha256"],
                                  "A witness mouth crop differs")
                    tally["witness_mouth_crops"] += 1
            phone.require(decoded == wanted, "Missing witness pictures")
            tally["witness_pictures"] += len(decoded)
    phone.write(output, {"receipt_sha256": phone.sha(path), **tally})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("observe", "analyze", "review", "replay"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--parents", type=Path, nargs=2, required=True)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "observe":
        observe(*args.parents, args.sources, args.runtime, args.geometry, args.checkpoint, args.wheel, args.path)
    elif args.phase == "analyze":
        analyze(args.path, args.parents, args.output)
    elif args.phase == "replay":
        replay(args.path, args.parents, args.sources, args.output)
    else:
        print(json.dumps(review(json.loads(args.path.read_bytes()), args.parents), sort_keys=True))


if __name__ == "__main__":
    main()
