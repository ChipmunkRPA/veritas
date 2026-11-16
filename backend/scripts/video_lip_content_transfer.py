# SPDX-License-Identifier: MIT
"""Compare lip timing and local mouth disagreement on matched soundtrack replacements."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from math import gcd
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from app.evidence import video_viseme_binding as incumbent
from app.evidence import video_viseme_ranked as ranked
from app.evidence import video_viseme_timing as timing
from scripts import video_lip_dense_posteriors as dense
from scripts import video_lip_viseme_transfer as study

phone = study.phone
RATE = timing.RATE
SCHEMA = "chipmunk.video-lip-content-transfer/1"
SPLITS = {"calibration": 6, "held": 16}
ROLES = {
    "original": "ordinary", "gain": "ordinary", "same_words": "same_words",
    "different_words": "content", "global_early": "constant", "global_late": "constant",
}
ARMS = ("existing", "native_alignment", "shared_flank_lag")
LEVELS = ("tight", "wide")
PHASES = ("before", "during", "after")
SHA256 = re.compile(r"[0-9a-f]{64}")
SALT = "2026-11-15-lip-content/1"
METHOD = {
    "inputs": "Use verified speaker/utterance labels and forced-aligned letter and digit boundaries from Lombard Grid. "
              "Both donors are from the target speaker and recording condition. One repeats both words, while the other "
              "changes both. Restrict donor lengths and splice each to the same sample range with the same interpolation "
              "and endpoint blend. All original video frames and all outside samples stay byte-identical.",
    "split": "Choose 6 calibration and up to 16 held speakers in seeded hash order, excluding the two preceding native "
             "lip timing source panels. Take one plain and one Lombard recording per speaker. Same-word audio replacement, "
             "gain changes, and global delay are distinct controls.",
    "check": "The incumbent searches for a brief change in sound-picture lag. The two candidates scan where the predicted "
             "and observed aperture disagree more than on both flanks. Fix the comparison at native alignment or at the "
             "best shared lag across both flanks. A candidate reports disagreement and does not claim a local delay.",
    "points": "Require a score strictly above the largest and second-largest per-speaker maxima on all calibration controls. "
              "Freeze all geometry, rankings and evidence rules. Count at least 0.40 interval intersection over union as "
              "useful on different-word clips. Count original, gain, repeated-word and constant-delay alerts separately.",
}


def keyed(text):
    return phone.digest(f"{SALT}/{text}".encode())


def word_bounds(phones):
    words = []
    opened = None
    for row in phones:
        name = row["phone"]
        if name.startswith("SIL"):
            if opened is not None:
                return None
            continue
        a = round(Fraction(str(row["offset"])) * RATE)
        b = round((Fraction(str(row["offset"])) + Fraction(str(row["duration"]))) * RATE)
        tail = name.rsplit("_", 1)[-1]
        if tail in {"S", "B"}:
            if opened is not None:
                return None
            opened = a
        if opened is None or tail not in {"B", "I", "E", "S"} or b <= a:
            return None
        if tail in {"S", "E"}:
            words.append([opened, b])
            opened = None
    if opened is None and len(words) == 6 and all(left[1] <= right[0] for left, right in zip(words, words[1:])):
        start, end = words[3][0], words[4][1]
        if 6400 <= end - start <= 12000:
            return [start, end]
    return None


def qualified(label, position, archive, audio, ear, upper):
    wave = audio.read(ear[label])
    samples = phone.pcm(wave)
    if position[0] < 6400 or position[1] + 4800 > len(samples) or len(samples) > upper:
        return None
    return {"id": label, "span": position, "samples": len(samples), "audio": ear[label].filename,
            "audio_sha256": phone.digest(wave), "alignment_sha256": phone.digest(archive)}


def roster_group(group, audio, ear, front, eye):
    order = sorted(group, key=keyed)
    qualified_sources = {}
    for target in order:
        target_words = target.rsplit("_", 1)[-1][3:5]
        location = group[target][0]
        target_length = location[1] - location[0]
        same, distinct = [], []
        for donor in order:
            if donor == target:
                continue
            words = donor.rsplit("_", 1)[-1][3:5]
            if words == target_words:
                pool = same
            elif all(a != b for a, b in zip(words, target_words, strict=True)):
                pool = distinct
            else:
                continue
            start, end = group[donor][0]
            if 85 * target_length <= 100 * (end - start) <= 115 * target_length:
                pool.append(donor)
        for uid in (target, *same, *distinct):
            if uid not in qualified_sources:
                qualified_sources[uid] = qualified(uid, *group[uid], audio, ear, 64000)
        reference = qualified_sources[target]
        if reference and (control := next((qualified_sources[uid] for uid in same if qualified_sources[uid]), None)):
            a, b = control["span"]
            changed = next((qualified_sources[uid] for uid in distinct
                            if qualified_sources[uid] and abs((group[uid][0][1] - group[uid][0][0]) - (b - a)) <= 960), None)
            if changed:
                picture = front.read(eye[target])
                return {"target": reference | {"front": eye[target].filename, "front_sha256": phone.digest(picture)},
                        "same_words": control, "different_words": changed}
    return None


def build_roster(folder, output):
    from scripts.lombard_mouth_source_intake import index_zip

    paths, publisher = phone.source_archives(folder)
    source_rows = study.timings()["rows"] + study.load(dense.CARRIER_SOURCE)["rows"]
    excluded = {row["speaker"] for row in source_rows}
    groups, rows = defaultdict(lambda: defaultdict(dict)), []
    with (ZipFile(folder / "publisher-json.zip") as metadata, ZipFile(folder / "alignment.zip") as align,
          ZipFile(paths["front"]) as front, ZipFile(paths["audio"]) as audio):
        eye, ear = index_zip(front, ".mov"), index_zip(audio, ".wav")
        labels = index_zip(align, ".json")
        for meta in index_zip(metadata, ".json").values():
            for row in json.loads(metadata.read(meta)):
                group, condition, code = row["SPKR"], row["COND"], row["UTTERANCE"]
                if group in excluded or condition not in ("p", "l") or row["STATUS"] != "CORRECT":
                    continue
                uid = f"{group}_{condition}_{code}"
                if not re.fullmatch(r"[a-z]{4}[0-9][a-z]", code) or uid not in labels or uid not in eye or uid not in ear:
                    continue
                alignment = align.read(labels[uid])
                if span := word_bounds(json.loads(alignment)[uid]):
                    groups[group][condition][uid] = span, alignment
        admitted = 0
        for group in sorted(groups, key=keyed):
            clips = {condition: roster_group(groups[group][condition], audio, ear, front, eye) for condition in ("p", "l")}
            if not all(clips.values()):
                continue
            split = "calibration" if admitted < SPLITS["calibration"] else "held"
            rows.extend({"speaker": keyed(group), "condition": condition, "split": split, **case} for condition, case in clips.items())
            admitted += 1
            if admitted == sum(SPLITS.values()):
                break
    phone.require(admitted >= SPLITS["calibration"] + 10, f"Only {admitted} matched speakers qualified")
    plan = {"schema": SCHEMA, "method": METHOD, "sources": publisher, "exclusion_sha256": phone.sha(dense.CARRIER_SOURCE), "rows": rows}
    phone.write(output, plan)
    print(f"Froze {admitted} speakers and {len(rows)} target clips", flush=True)


def splice(original, location, donor, origin):
    import numpy as np
    from scipy.signal import resample_poly

    start, stop = location
    first, last = origin
    clip = donor[first:last].astype(np.float64)
    common = gcd(stop - start, len(clip))
    fitted = resample_poly(clip, (stop - start) // common, len(clip) // common)
    earlier = original[start:stop].astype(np.float64)
    phone.require(len(fitted) == len(earlier) and np.any(fitted), "The rate-matched speech donor differs in duration")
    power = np.linalg.norm(earlier) / np.linalg.norm(fitted)
    fitted *= np.clip(power, .5, 2.)
    edge = np.linspace(0., 1., 128, endpoint=False)
    fitted[:128] = earlier[:128] * (1. - edge) + fitted[:128] * edge
    fitted[-128:] = earlier[-128:] * (1. - edge[::-1]) + fitted[-128:] * edge[::-1]
    result = original.copy()
    result[start:stop] = np.rint(fitted).clip(-32768, 32767).astype("<i2")
    return result


def pictures(path, samples, mp, task, *, with_shape=False):
    from app.detectors.video.decode import decode_stream, probe_frames
    from app.evidence.video_reference_timing import _mesh

    table = probe_frames(path, cap=120, timeout_s=18, max_pixels=1280 * 720)
    frame_count = len(table)
    phone.require(not table.truncated and not table.interpolated and not table.rotation and 24 <= frame_count <= 120
                  and all(Fraction(1, 28) <= (b - a) * table.tb <= Fraction(1, 22)
                          for a, b in zip(table.pts, table.pts[1:])), "The frozen source is outside the exact picture clock")
    frames = []
    for frame in decode_stream(path, table, max_side=None, timeout_s=20):
        index = len(frames)
        phone.require((frame.index, frame.pts, frame.pts_verified) == (index, table.pts[index], True),
                      "The picture does not have its exact native position")
        record = {"index": index, "pts": frame.pts, "sample": round((frame.pts - table.pts[0]) * table.tb * RATE), "gap": None}
        if observed := _mesh(task, mp, frame.rgb):
            features, box = observed
            x, y, right, bottom = box
            record.update(gap=round(float(median(features[:3])), 7), mouth_box=box,
                          rgb_sha256=phone.digest(frame.rgb.tobytes()),
                          mouth_crop_sha256=phone.digest(frame.rgb[y:bottom, x:right].tobytes()))
            if with_shape:
                record["shape"] = [round(float(value) * timing.SCORE_UNITS) for value in features]
        frames.append(record)
    phone.require(len(frames) == frame_count, "The picture scan did not cover the full source")
    return {"frames": frames, "samples": samples, "time_base": table.time_base}


def evidence(row, proposal):
    if proposal["status"] != "measured":
        return proposal
    bound = timing.picture_evidence(row["frames"], proposal, row["samples"])
    frames = bound["picture_brackets"]
    if not all(frames.values()):
        return {"status": "incomplete_context"}
    return {"status": "measured", **{key: proposal[key] for key in ("score", "span", "baseline_lag", "change")},
            "picture_span": bound["picture_span"],
            "pictures": {role: {key: value for key, value in frame.items() if key != "gap"}
                         for role, frame in frames.items()}}


def disagreement(events, samples, *, fixed):
    if len(events) < 10:
        return {"status": "insufficient_events"}
    lags = timing.LAGS
    candidates = []
    for width in timing.WIDTHS:
        for start in range(timing.STEP, samples - width - timing.STEP + 1, timing.STEP):
            stop = start + width
            left = [event for event in events if event[0] < start]
            middle = [event for event in events if start <= event[0] < stop]
            right = [event for event in events if event[0] >= stop]
            if len(left) < 3 or len(right) < 3 or len(middle) < 4:
                continue
            a, b, c = ([total / (len(section) * timing.SCORE_UNITS) for total in timing.totals(section)]
                       for section in (left, middle, right))
            lane = lags.index(0) if fixed else max(range(len(lags)), key=lambda i: (a[i] + c[i], -abs(lags[i]), -lags[i]))
            score = round(min(a[lane], c[lane]) - b[lane], 7)
            if score > 0:
                candidates.append({"score": score, "span": [start, stop], "baseline_lag": lags[lane], "change": 0,
                                   "before_frame": left[-1][3][lane], "after_frame": right[0][3][lane],
                                   "frames": sorted({item[3][lane] for item in middle})})
    if not candidates:
        return {"status": "no_supported_change"}
    best = min(candidates, key=lambda p: (-p["score"], p["span"]))
    if any(item["score"] >= best["score"] - .015 and not overlap(best["span"], item["span"]) for item in candidates):
        return {"status": "ambiguous_change"}
    return {"status": "measured", **best}


def overlap(first, second):
    a, b = first
    c, d = second
    return 5 * (min(b, d) - max(a, c)) >= 2 * (max(b, d) - min(a, c))


def measurements(row, speech, events):
    sparse = evidence(row, timing.hypotheses(timing.compatible(row, events, incumbent.REFERENCE), row["samples"]))
    series = ranked.scores(row, ranked.predict(speech))
    aperture = evidence(row, timing.hypotheses(series, row["samples"]))
    return {"sparse": sparse, "aperture": aperture, **{
        arm: evidence(row, disagreement(series, row["samples"], fixed=fixed))
        for arm, fixed in zip(ARMS[1:], (True, False), strict=True)}}


def choose(readings, arm, points, level):
    if readings["sparse"]["status"] == "measured" and readings["sparse"]["score"] > incumbent.POINT:
        return readings["sparse"]
    if readings["aperture"]["status"] == "measured" and readings["aperture"]["score"] > ranked.POINT:
        return readings["aperture"]
    if arm != "existing" and readings[arm]["status"] == "measured" and readings[arm]["score"] > points[arm][level]:
        return readings[arm]
    return None


def calculate(rows):
    points = {}
    for arm in ARMS[1:]:
        maxima = {}
        for row in rows:
            if row["split"] == "calibration":
                group = row["speaker"]
                values = [measure[arm].get("score", 0.) for role, measure in row["readings"].items()
                          if ROLES[role] != "content"]
                maxima[group] = max(maxima.get(group, 0.), *values)
        phone.require(len(maxima) == SPLITS["calibration"], "Missing separate calibration speakers")
        points[arm] = dict(zip(LEVELS, sorted(maxima.values(), reverse=True)[:2], strict=True))
    results = {}
    unassessed = {"insufficient_events", "incomplete_context", "ambiguous_change"}
    for arm in ARMS:
        for level in LEVELS:
            counts = defaultdict(Counter)
            for row in rows:
                if row["split"] != "held":
                    continue
                for role, reads in row["readings"].items():
                    proposed = choose(reads, arm, points, level)
                    baseline = choose(reads, "existing", points, level)
                    kind = ROLES[role]
                    useful = kind == "content" and proposed is not None and overlap(row["truth"], proposed["span"])
                    matched = choose(row["readings"]["same_words"], arm, points, level) if useful else None
                    clean = useful and (not matched or not overlap(proposed["span"], matched["span"]))
                    assessed = any(reads[key]["status"] not in unassessed for key in
                                   ("sparse", "aperture") + (() if arm == "existing" else (arm,)))
                    for bucket in {kind, f"{row['condition']}/{kind}"}:
                        counts[bucket].update(files=1, assessed=assessed, flags=proposed is not None, useful=useful,
                                              useful_without_same_word_collision=clean, misplaced=kind == "content" and proposed is not None and not useful,
                                              added=bool(proposed and not baseline))
            results[f"{arm}/{level}"] = {key: dict(value) for key, value in counts.items()}
    return points, results


def capture(args):
    from app.detectors.video.viseme_worker import aperture_reader
    from app.evidence.video_reference_timing import _landmarker, available

    plan = json.loads(args.plan.read_bytes())
    phone.require(plan["schema"] == SCHEMA and plan["method"] == METHOD, "The source plan did not match the frozen protocol")
    sys.path.insert(0, str((args.assets / "runtime").resolve()))
    geometry = args.assets / "face_landmarker.task"
    phone.require(available(geometry) and phone.sha(geometry) == study.parent.FACE_SHA256, "The pinned face asset is required")
    acoustic, device = aperture_reader(args.assets / "checkpoint")
    rows = []
    with (ZipFile(args.sources / "lombardgrid_audio.zip") as archive,
          ZipFile(args.sources / "lombardgrid_front.zip") as front, TemporaryDirectory() as scratch,
          _landmarker(geometry) as (mp, task)):
        path = Path(scratch) / "original.mov"
        for row in plan["rows"]:
            target = row["target"]
            payload = front.read(target["front"])
            phone.require(phone.digest(payload) == target["front_sha256"], "The original picture file changed")
            path.write_bytes(payload)
            wave = {}
            stems = ("target", "same_words", "different_words")
            for role in stems:
                source = row[role]
                payload = archive.read(source["audio"])
                phone.require(phone.digest(payload) == source["audio_sha256"], "The original audio file changed")
                wave[role] = phone.pcm(payload)
            pcm, span = wave["target"], target["span"]
            screen = pictures(path, len(pcm), mp, task)
            versions = {role: value for role, value in study.parent.treatments(pcm, span).items()
                        if role in ("original", "gain", "global_early", "global_late")}
            for role in stems[1:]:
                versions[role] = splice(pcm, span, wave[role], row[role]["span"])
                a, b = span
                phone.require(versions[role][:a].tobytes() == pcm[:a].tobytes()
                              and versions[role][b:].tobytes() == pcm[b:].tobytes()
                              and versions[role][a:b].tobytes() != pcm[a:b].tobytes(), "A replacement changed its native extent")
            scores = {role: measurements(screen, *acoustic(wave, include_events=True)) for role, wave in versions.items()}
            rows.append({key: row[key] for key in ("speaker", "condition", "split")} | {
                "media": {role: {key: row[role][key] for key in ("audio_sha256", "alignment_sha256", "span")}
                          for role in stems},
                "picture_sha256": target["front_sha256"], "time_base": screen["time_base"],
                "picture_pts": [frame["pts"] for frame in screen["frames"]],
                "samples": len(pcm), "truth": span, "readings": scores,
                "pcm": {key: phone.digest(values.tobytes()) for key, values in versions.items()},
            })
            print(f"Checked {len(rows)}/{len(plan['rows'])} native recordings", flush=True)
    packet = {key: plan[key] for key in ("schema", "method", "sources", "exclusion_sha256")}
    packet.update(device=device, plan_sha256=phone.sha(args.plan), rows=rows)
    packet["points"], packet["result"] = calculate(rows)
    review(packet)
    dense.seal(args.output, packet)


def review(packet):
    phone.require(packet["schema"] == SCHEMA and packet["method"] == METHOD
                  and packet["exclusion_sha256"] == phone.sha(dense.CARRIER_SOURCE), "The source plan or method changed")
    groups = defaultdict(list)
    consumed_audio, consumed_video = set(), set()
    for row in packet["rows"]:
        groups[row["speaker"]].append((row["split"], row["condition"]))
        source_hashes = {item["audio_sha256"] for item in row["media"].values()}
        phone.require(set(row["readings"]) == set(row["pcm"]) == set(ROLES) and row["truth"] == row["media"]["target"]["span"],
                      "A delivery or truth interval changed")
        phone.require(set(row["media"]) == {"target", "same_words", "different_words"} and len(source_hashes) == 3
                      and not source_hashes & consumed_audio and row["picture_sha256"] not in consumed_video
                      and all(SHA256.fullmatch(value) for value in (*source_hashes, row["picture_sha256"], row["speaker"],
                          *(item["alignment_sha256"] for item in row["media"].values()), *row["pcm"].values())),
                      "Duplicate or invalid original-media commitments")
        consumed_audio.update(source_hashes)
        consumed_video.add(row["picture_sha256"])
        start, stop = row["truth"]
        pts = row["picture_pts"]
        time_base = Fraction(row["time_base"])
        phone.require(24 <= len(pts) <= 120 and all(type(value) is int for value in pts)
                      and all(Fraction(1, 28) <= (b - a) * time_base <= Fraction(1, 22) for a, b in zip(pts, pts[1:])),
                      "Invalid native picture clock")
        durations = [b - a for a, b in (row["media"][role]["span"] for role in ("same_words", "different_words"))]
        phone.require(6400 <= stop - start <= 12000 and start >= 6400 and row["samples"] - stop >= 4800
                      and all(85 * (stop - start) <= 100 * length <= 115 * (stop - start) for length in durations)
                      and abs(durations[0] - durations[1]) <= 960, "The matched donor timing differs from the frozen plan")
        for comparison in row["readings"].values():
            phone.require(set(comparison) == {"sparse", "aperture", *ARMS[1:]}, "A named comparison is missing")
            for name, proposal in comparison.items():
                if proposal["status"] != "measured":
                    phone.require(proposal["status"] in {
                        "insufficient_events", "incomplete_context", "ambiguous_change", "no_supported_change"},
                        "Unknown observation status")
                    continue
                start, stop = proposal["span"]
                phone.require(0 <= start < stop <= row["samples"] and proposal["score"] > 0
                              and proposal["baseline_lag"] in timing.LAGS and (name in ARMS[1:]) == (proposal["change"] == 0),
                              "Invalid interval evidence")
                lag = proposal["baseline_lag"] + proposal["change"]
                pa, pb = (max(0, min(row["samples"], t - lag)) for t in (start, stop))
                phone.require([pa, pb] == proposal["picture_span"], "The picture range differs from the audio range")
                pictured = [proposal["pictures"][phase] for phase in PHASES]
                left, middle, right = (p["sample"] for p in pictured)
                phone.require(left < pa <= middle < pb <= right
                              and all(0 <= p["index"] < len(pts) and p["pts"] == pts[p["index"]]
                                      and p["sample"] == round((p["pts"] - pts[0]) * time_base * RATE) for p in pictured)
                              and all(SHA256.fullmatch(p[key]) for p in pictured for key in ("rgb_sha256", "mouth_crop_sha256"))
                              and pictured[0]["index"] < pictured[1]["index"] < pictured[2]["index"],
                              "Before, during and after frames do not match the original clock")
    phone.require(all(roles in ([("calibration", "p"), ("calibration", "l")], [("held", "p"), ("held", "l")])
                      for roles in groups.values()), "A source appears across roles or twice in a condition")
    held = sum(roles[0][0] == "held" for roles in groups.values())
    phone.require(10 <= held <= SPLITS["held"], "The held speaker population fell outside the declared design")
    points, results = calculate(packet["rows"])
    phone.require(points == packet["points"] and results == packet["result"], "The final counts do not match the numeric replay")
    return results


def audit_pictures(receipt, plan_path, sources):
    from app.detectors.video.decode import decode_frames, probe_frames

    packet = study.load(receipt)
    review(packet)
    phone.require(phone.sha(plan_path) == packet["plan_sha256"], "The picture roster differs from the receipt")
    plan = json.loads(plan_path.read_bytes())
    counts = Counter()
    with ZipFile(sources / "lombardgrid_front.zip") as archive, TemporaryDirectory() as scratch:
        path = Path(scratch) / "source.mov"
        for source, row in zip(plan["rows"], packet["rows"], strict=True):
            picture = archive.read(source["target"]["front"])
            phone.require(phone.digest(picture) == source["target"]["front_sha256"] == row["picture_sha256"],
                          "A source video commitment differs")
            path.write_bytes(picture)
            table = probe_frames(path, cap=120, timeout_s=18, max_pixels=1280 * 720)
            phone.require(not table.truncated and not table.interpolated and table.time_base == row["time_base"]
                          and list(table.pts) == row["picture_pts"], "The source picture clock differs")
            observed = {}
            for methods in row["readings"].values():
                for proposal in methods.values():
                    for witness in proposal.get("pictures", {}).values():
                        index = witness["index"]
                        phone.require(index not in observed or observed[index] == witness,
                                      "A repeated original frame has conflicting evidence")
                        observed[index] = witness
                        counts["picture_witnesses"] += 1
            if observed:
                indices = sorted(observed)
                decoded = decode_frames(path, table, indices, timeout_s=25)
                for index, picture in zip(indices, decoded, strict=True):
                    witness = observed[index]
                    x, y, right, bottom = witness["mouth_box"]
                    phone.require((picture.index, picture.pts, picture.pts_verified) == (index, witness["pts"], True)
                                  and 0 <= x < right <= table.width and 0 <= y < bottom <= table.height
                                  and phone.digest(picture.rgb.tobytes()) == witness["rgb_sha256"]
                                  and phone.digest(picture.rgb[y:bottom, x:right].tobytes()) == witness["mouth_crop_sha256"],
                                  "Original pixels do not reproduce the recorded crop")
                    counts["unique_decoded_frames"] += 1
            counts["source_clocks"] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    plan = phases.add_parser("freeze")
    for name in ("sources", "output"):
        plan.add_argument(name, type=Path)
    native = phases.add_parser("run")
    for name in ("plan", "sources", "assets", "output"):
        native.add_argument(name, type=Path)
    portable = phases.add_parser("review")
    portable.add_argument("receipt", type=Path)
    frames = phases.add_parser("audit-pictures")
    for name in ("receipt", "plan", "sources"):
        frames.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        build_roster(args.sources, args.output)
    elif args.phase == "run":
        capture(args)
    else:
        result = (review(study.load(args.receipt)) if args.phase == "review"
                  else audit_pictures(args.receipt, args.plan, args.sources))
        print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
