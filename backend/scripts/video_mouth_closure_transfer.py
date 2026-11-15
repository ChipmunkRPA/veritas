# SPDX-License-Identifier: Apache-2.0
"""Test whether audible M events have a corresponding visible lip closure."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts import video_bilabial_mouth_transfer as releases

phone = releases.phone
RATE = phone.RATE
SCHEMA = "chipmunk.video-mouth-closure-transfer/1"
SPLICE_HALF = 3360
HALF_WINDOWS = {"100ms": 1600, "160ms": 2560}
CONDITIONS = ("p", "l")
M_ROLES = ("m_original", "m_lowpass", "m_gain", "m_global_early", "m_global_late", "m_same_letter")
ROLES = (*M_ROLES, "vowel_original", "vowel_same_letter", "vowel_dubbed_with_m")
POSITIVE = ROLES[-1]
PHONES = {"m": "m_E", "a": "ey_S", "i": "ay_S"}
M_NAMES = ("m", "mʲ")
ANCHOR_POINT, FLOOR = .55, .060
DESIGN = {
    "sources": "Source codes outside the two phone-timing panels and earlier exclusion list. "
               "Choose six development speakers by hash order, then use all remaining qualified speakers. "
               "Require plain and Lombard conditions for each speaker, each with two M and two A or I utterances. "
               "Choose from publisher labels and media availability before any detector inference.",
    "edits": "Preserve the submitted pictures. Copy a same-speaker excerpt around the target letter midpoint "
             "from another recording under the same condition. Use a 420 ms edit, center-aligned with 8 ms edge blends "
             "and at most 3 dB level adjustment. Include analogous same-letter dubs, gain, low-pass and constant 120 ms offsets.",
    "measure": "Read audible M proposals independently. Require emission >= 0.55, and a second M reading from "
               "a 3.4 kHz low-pass copy within 40 ms. Use median clock bias from development genuine M. "
               "Compare minimum observed lip aperture within 100 or 160 ms of the corrected acoustic event. "
               "Require at least four consecutive native pictures with complete mouth geometry and at most 50 ms per step.",
    "points": "Use strict exceedance above the highest (tight) and second-highest (broad) ordinary development "
              "speaker maxima, with floor 0.060. Select the strongest aperture observation per file and each window size.",
    "utility": "For the known changed interval, require the scored event inside the inserted M and 0.40 temporal IoU. "
               "No AI authorship claim. Flag an ordinary control as a false local-edit lead even when a constant offset is present.",
}


def order(key):
    return phone.digest(f"{SCHEMA}/{key}".encode())


def labelled_member(archive, uid, member, letter):
    payload = archive.read(member)
    labels = json.loads(payload)[uid]
    candidates = [row for row in labels if row["phone"] == PHONES[letter] and row["duration"] >= .08]
    if len(candidates) != 1:
        return None
    label, = candidates
    start = Fraction(str(label["offset"]))
    end = start + Fraction(str(label["duration"]))
    return [round(start * RATE), round(end * RATE)], phone.digest(payload)


def eligible_sources(info, labels, front, audio, excluded):
    from scripts.lombard_mouth_source_intake import index_zip

    label_index, picture_index, audio_index = (index_zip(bank, suffix) for bank, suffix in
                                             ((labels, ".json"), (front, ".mov"), (audio, ".wav")))
    sources = defaultdict(list)
    for entry in index_zip(info, ".json").values():
        for row in json.loads(info.read(entry)):
            speaker, condition, prompt = (row[key] for key in ("SPKR", "COND", "UTTERANCE"))
            uid, letter = f"{speaker}_{condition}_{prompt}", prompt[3:4]
            if (speaker in excluded or condition not in CONDITIONS or row["STATUS"] != "CORRECT" or letter not in PHONES
                    or uid not in label_index or uid not in picture_index or uid not in audio_index):
                continue
            if not (verified := labelled_member(labels, uid, label_index[uid], letter)):
                continue
            interval, aligned_sha = verified
            center = sum(interval) // 2
            picture, sound = picture_index[uid], audio_index[uid]
            pcm = phone.pcm(audio.read(sound))
            if center - SPLICE_HALF < 0 or center + SPLICE_HALF > len(pcm):
                continue
            sources[speaker, condition, letter].append({"source": uid, "letter": letter, "aligned": interval,
                    "center": center, "samples": len(pcm), "audio_pcm_sha256": phone.digest(pcm.tobytes()),
                    "files": {"front": picture.filename, "audio": sound.filename, "alignment": label_index[uid].filename},
                    "alignment_sha256": aligned_sha})
    return sources


def roster(root, prior):
    paths, metadata = phone.source_archives(root)
    excluded = phone.EXCLUDED | {row["speaker"] for _, row in releases.source_rows(*prior)}
    with (ZipFile(root / "publisher-json.zip") as info, ZipFile(root / "alignment.zip") as labels,
          ZipFile(paths["front"]) as front, ZipFile(paths["audio"]) as audio):
        sources = eligible_sources(info, labels, front, audio, excluded)
        admitted = []
        for speaker in sorted({key[0] for key in sources}, key=order):
            group = []
            for condition in CONDITIONS:
                candidates = {letter: sorted(sources[speaker, condition, letter],
                                             key=lambda item: order(item["source"]))[:2] for letter in PHONES}
                center_m = candidates["m"]
                vowels = [candidates[letter] for letter in ("a", "i") if len(candidates[letter]) == 2]
                if len(center_m) != 2 or not vowels:
                    continue
                center_vowel = min(vowels, key=lambda rows: order(rows[0]["source"]))
                variants = dict(zip(("m", "m_alternate", "vowel", "vowel_alternate"), center_m + center_vowel, strict=True))
                for row in variants.values():
                    row["front_sha256"] = phone.digest(front.read(row["files"]["front"]))
                group.append({"speaker": speaker, "condition": condition, "sources": variants})
            if len(group) == len(CONDITIONS):
                split = "development" if len(admitted) < 6 * len(CONDITIONS) else "held"
                admitted.extend(row | {"split": split} for row in group)
    phone.require(len(admitted) >= 26, "Too few disjoint plain and Lombard pairs")
    return {"schema": SCHEMA, "design": DESIGN, "parents": [phone.sha(p) for p in prior], "excluded": sorted(excluded),
            "assets": {"acoustic": phone.WEIGHT_PIN, "geometry": releases.FACE_SHA256}, "archives": metadata, "rows": admitted}


def blend(base, donor, center, alternative):
    import numpy as np

    start, stop = center - SPLICE_HALF, center + SPLICE_HALF
    donor_start, donor_stop = alternative - SPLICE_HALF, alternative + SPLICE_HALF
    first, second = base[start:stop].astype("float64"), donor[donor_start:donor_stop].astype("float64")
    ratio = np.linalg.norm(first) / max(np.linalg.norm(second), 1.)
    gain = min(max(ratio, 10 ** (-3 / 20)), 10 ** (3 / 20))
    fades = np.ones(stop - start)
    fades[:128] = np.linspace(0, 1, 128, endpoint=False)
    fades[-128:] = fades[:128][::-1]
    changed = base.copy()
    changed[start:stop] = np.rint(first * (1 - fades) + gain * second * fades).clip(-32768, 32767).astype("<i2")
    return changed


def filtered(samples):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    smooth = sosfiltfilt(butter(4, 3400, fs=RATE, output="sos"), samples)
    return np.rint(smooth).clip(-32768, 32767).astype("<i2")


def deliveries(sounds, sources):
    import numpy as np

    base_m, base_vowel = sounds["m"], sounds["vowel"]
    offset = 1920
    result = {"m_original": base_m, "m_lowpass": filtered(base_m), "m_gain": np.rint(base_m * .65).astype("<i2"),
              "m_global_early": np.pad(base_m[offset:], (0, offset)),
              "m_global_late": np.pad(base_m[:-offset], (offset, 0)), "vowel_original": base_vowel}
    for role, first, second in (("m_same_letter", "m", "m_alternate"), ("vowel_same_letter", "vowel", "vowel_alternate"),
                                (POSITIVE, "vowel", "m_alternate")):
        result[role] = blend(sounds[first], sounds[second], sources[first]["center"], sources[second]["center"])
    return {role: result[role] for role in ROLES}


def observe(args, frozen):
    import cv2
    import numpy as np

    from app.detectors.image.faces import load_yunet
    from app.detectors.video import lipsync
    from app.evidence.video_reference_timing import _landmarker, available
    from scripts.video_lipsync_signed_transitions import current
    from scripts.video_lipsync_window_schedule import FACE_FILE, FACE_SHA

    paths, metadata = phone.source_archives(args.sources)
    phone.require(metadata == frozen["archives"], "Publisher commitments differ")
    phone.require(phone.sha(args.face) == releases.FACE_SHA256 and phone.sha(FACE_FILE) == FACE_SHA, "Geometry assets changed")
    sys.path.insert(0, str(args.runtime))
    phone.require(available(args.face), "Face geometry runtime unavailable")
    cv2.setNumThreads(1)
    detector = load_yunet(FACE_FILE)
    infer, device = phone.neural_decoder(args.model, names=M_NAMES)
    measured = []
    with ExitStack() as context:
        front, audio, labels = [context.enter_context(ZipFile(path))
                                for path in (paths["front"], paths["audio"], args.sources / "alignment.zip")]
        work = Path(context.enter_context(TemporaryDirectory(prefix="mouth-closure-")))
        mp, task = context.enter_context(_landmarker(args.face))
        for source in (row for row in frozen["rows"] if row["split"] == args.phase):
            sources = source["sources"]
            sounds = {key: phone.pcm(audio.read(row["files"]["audio"])) for key, row in sources.items()}
            for key, row in sources.items():
                actual = phone.digest(sounds[key].tobytes()), phone.digest(labels.read(row["files"]["alignment"]))
                phone.require(actual == (row["audio_pcm_sha256"], row["alignment_sha256"]), "Publisher source mismatch")
            pictures, geometry = {}, {}
            for key in ("m", "vowel"):
                row = sources[key]
                data = front.read(row["files"]["front"])
                phone.require(phone.digest(data) == row["front_sha256"], "Original picture archive mismatch")
                path = work / f"{key}.mov"
                path.write_bytes(data)
                frames, time_base, details = releases.picture_observations(path, len(sounds[key]), mp, task, detector)
                pictures[key] = {"time_base": time_base, "frames": frames}
                geometry[key] = details
            cases = {}
            for role, samples in deliveries(sounds, sources).items():
                key = "m" if role in M_ROLES else "vowel"
                faces, patches, aperture, grid = geometry[key]
                acoustic = infer(samples)
                supported = infer(filtered(samples)) if acoustic else []
                cases[role] = {"pcm_sha256": phone.digest(samples.tobytes()), "frames": key,
                               "acoustic": acoustic, "lowpass": supported,
                               "existing": current(faces, patches, aperture, lipsync.vad(samples.astype(np.float32) / 32768), grid)}
            measured.append({"speaker": source["speaker"], "condition": source["condition"], "picture": pictures, "cases": cases})
            print("Measured", args.phase, len(measured), flush=True)
    return {"device": device, "rows": measured}


def anchors(case):
    return [event for event in case["acoustic"] if event[2] >= ANCHOR_POINT
            and any(abs(event[1] - other[1]) <= 640 for other in case["lowpass"])]


def fit_clock(rows, measured):
    samples = defaultdict(list)
    for row, observation in zip(rows, measured, strict=True):
        source = row["sources"]["m"]
        matches = [event[1] - source["center"] for event in anchors(observation["cases"]["m_original"])
                   if abs(event[1] - source["center"]) <= 2560]
        if len(matches) == 1:
            samples[row["speaker"]].append(matches[0])
    phone.require(len(samples) >= 3, "Too few independently labeled M events to establish a clock")
    return {"offset": round(statistics.median(statistics.median(values) for values in samples.values())),
            "speakers": len(samples), "events": sum(map(len, samples.values()))}


def reading(case, pictures, bias, half):
    frames = pictures[case["frames"]]["frames"]
    proposals = []
    for _, auditory_sample, posterior in anchors(case):
        center = auditory_sample - bias
        kept = [row for row in frames if abs(row["sample"] - center) <= half]
        if (center - half < frames[0]["sample"] or center + half > frames[-1]["sample"]
                or len(kept) < 4 or any(row["gap"] is None for row in kept)
                or any(not 0 < right["sample"] - left["sample"] <= 800 for left, right in zip(kept, kept[1:]))):
            continue
        proposals.append({"audio_event": auditory_sample, "posterior": posterior, "expected_closure": center,
                          "range": [center - half, center + half], "frames": [row["index"] for row in kept],
                          "minimum_aperture": min(row["gap"] for row in kept)})
    return max(proposals, key=lambda row: (row["minimum_aperture"], -row["audio_event"]), default=None)


def calibrate(rows, observed):
    bias = fit_clock(rows, observed)
    points = {}
    for name, half in HALF_WINDOWS.items():
        maxima = defaultdict(float, {row["speaker"]: 0. for row in rows})
        for row, reading_set in zip(rows, observed, strict=True):
            for role in ROLES[:-1]:
                match = reading(reading_set["cases"][role], reading_set["picture"], bias["offset"], half)
                if match:
                    maxima[row["speaker"]] = max(maxima[row["speaker"]], match["minimum_aperture"])
        highest, second, *_ = sorted(maxima.values(), reverse=True)
        points[name] = {"tight": max(FLOOR, highest), "broad": max(FLOOR, second), "speaker_maxima": dict(maxima)}
    return {"clock": bias, "points": points}


def overlaps(span, truth):
    a, b = span
    start, end = truth
    intersection = max(0, min(b, end) - max(a, start))
    return 5 * intersection >= 2 * (max(b, end) - min(a, start))


def useful(candidate, truth, inserted):
    return (inserted[0] - 640 <= candidate["expected_closure"] <= inserted[1] + 640
            and overlaps(candidate["range"], truth))


def counts(rows, observed, fit):
    results = defaultdict(Counter)
    incumbent = defaultdict(Counter)
    overlap_counts = defaultdict(Counter)
    findings = []
    for row, record in zip(rows, observed, strict=True):
        source = row["sources"]
        center = source["vowel"]["center"]
        truth = (center - SPLICE_HALF, center + SPLICE_HALF)
        inserted = [value + center - source["m_alternate"]["center"] for value in source["m_alternate"]["aligned"]]
        for role in ROLES:
            case = record["cases"][role]
            existing = case["existing"]
            spans = [(a * 320 + (lag or 0) * 640, b * 320 + (lag or 0) * 640) for _, a, b, lag in existing["ranges"]]
            current_hit = role == POSITIVE and any(overlaps(span, truth) for span in spans)
            incumbent[role].update(files=1, measured=int(bool(existing["starts"])), flags=int(bool(spans)), useful=int(current_hit))
            for name, half in HALF_WINDOWS.items():
                match = reading(case, record["picture"], fit["clock"]["offset"], half)
                for arm in ("tight", "broad"):
                    flag = match is not None and match["minimum_aperture"] > fit["points"][name][arm]
                    located = role == POSITIVE and flag and useful(match, truth, inserted)
                    cell = results[name, arm, role]
                    cell.update(files=1, anchors=int(bool(anchors(case))), measured=int(match is not None),
                                flags=int(flag), useful=int(located))
                    overlap_counts[name, arm, role].update(useful_union=int(located or current_hit),
                                                           added_useful=int(located and not current_hit),
                                                           added_flags=int(flag and not spans))
                    if flag:
                        findings.append({"speaker": row["speaker"], "condition": row["condition"], "role": role,
                                         "window": name, "point": arm, "reading": match, "located": located})
    return {"counts": {"/".join(keys): dict(value) for keys, value in sorted(results.items())},
            "baseline": {key: dict(value) for key, value in incumbent.items()},
            "additive": {"/".join(keys): dict(value) for keys, value in sorted(overlap_counts.items())}, "findings": findings}


def verify_observations(frozen, measured, phase):
    expected = [row for row in frozen["rows"] if row["split"] == phase]
    for source, record in zip(expected, measured, strict=True):
        phone.require(all(source[key] == record[key] for key in ("speaker", "condition")) and set(record["cases"]) == set(ROLES),
                      "Observed case does not match its frozen role")
        phone.require(set(record["picture"]) == {"m", "vowel"}, "Missing paired pictures")
        for pictures in record["picture"].values():
            frames, base = pictures["frames"], Fraction(pictures["time_base"])
            first = frames[0]["pts"]
            phone.require(all(frame["index"] == pos and frame["sample"] == round((frame["pts"] - first) * base * RATE)
                              and (frame["gap"] is None or 0 <= frame["gap"] <= 1 and math.isfinite(frame["gap"]))
                              and releases.SHA256.fullmatch(frame["rgb_sha256"])
                              and (frame["gap"] is None or releases.SHA256.fullmatch(frame["mouth_crop_sha256"]))
                              for pos, frame in enumerate(frames)) and all(a["sample"] < b["sample"] for a, b in zip(frames, frames[1:])),
                          "A native picture commitment differs")
        for role, case in record["cases"].items():
            side = "m" if role in M_ROLES else "vowel"
            phone.require(side == case["frames"] and releases.SHA256.fullmatch(case["pcm_sha256"])
                          and all(name == "m" and 0 <= sample < source["sources"][side]["samples"] and .35 <= value <= 1
                                  for key in ("acoustic", "lowpass") for name, sample, value in case[key]),
                          "Audio evidence exceeds the submitted source")
    return expected


def verify_split(frozen, parents):
    seen = defaultdict(set)
    used = set()
    groups = defaultdict(set)
    expected_excluded = phone.EXCLUDED | {row["speaker"] for _, row in releases.source_rows(*parents)}
    phone.require(frozen["excluded"] == sorted(expected_excluded) and frozen["parents"] == [phone.sha(p) for p in parents],
                  "Source exclusion ledger changed")
    for row in frozen["rows"]:
        speaker, split, condition = (row[k] for k in ("speaker", "split", "condition"))
        phone.require(speaker not in expected_excluded and split in {"development", "held"}
                      and condition in CONDITIONS and condition not in groups[speaker], "Speaker or condition was reused")
        groups[speaker].add(condition)
        seen[split].add(speaker)
        sources = row["sources"]
        phone.require(set(sources) == {"m", "m_alternate", "vowel", "vowel_alternate"}
                      and sources["m"]["letter"] == sources["m_alternate"]["letter"] == "m"
                      and sources["vowel"]["letter"] == sources["vowel_alternate"]["letter"] in {"a", "i"},
                      "Same-letter replacement controls differ")
        for item in sources.values():
            a, b = item["aligned"]
            center = item["center"]
            phone.require(item["source"] not in used and item["source"].startswith(f"{speaker}_{condition}_")
                          and b - a >= 1280 and center == (a + b) // 2
                          and 0 <= center - SPLICE_HALF < a <= center <= b < center + SPLICE_HALF <= item["samples"],
                          "A labeled source or native interval was reused or changed")
            used.add(item["source"])
    phone.require(seen["development"].isdisjoint(seen["held"]) and len(seen["development"]) == 6
                  and len(seen["held"]) >= 7 and all(group == set(CONDITIONS) for group in groups.values()),
                  "Incomplete disjoint speaker roles")


def replay(path, sources, document, output):
    from app.detectors.video.decode import decode_frames, probe_frames

    roster = document["freeze"]
    media, metadata = phone.source_archives(sources)
    phone.require(metadata == roster["archives"], "Publisher archive identities differ")
    selected = defaultdict(set)
    for finding in document["outcome"]["findings"]:
        key = finding["speaker"], finding["condition"], "m" if finding["role"] in M_ROLES else "vowel"
        selected[key].update(finding["reading"]["frames"])
    inputs = document["fitted"]["development"]["rows"] + document["held"]["rows"]
    tally = Counter()
    with (ZipFile(media["front"]) as front, ZipFile(media["audio"]) as audio,
          ZipFile(sources / "alignment.zip") as labels, TemporaryDirectory(prefix="mouth-closure-replay-") as directory):
        movie = Path(directory) / "original.mov"
        for source, observed in zip(roster["rows"], inputs, strict=True):
            group = source["sources"]
            streams = {}
            for role, item in group.items():
                native = item["files"]
                streams[role] = phone.pcm(audio.read(native["audio"]))
                phone.require(phone.digest(streams[role].tobytes()) == item["audio_pcm_sha256"]
                              and labelled_member(labels, item["source"], native["alignment"], item["letter"])
                              == (item["aligned"], item["alignment_sha256"]), "Original audio or label changed")
                raw = front.read(native["front"])
                phone.require(phone.digest(raw) == item["front_sha256"], "Original picture source changed")
                tally["committed_sources"] += 1
                if role not in observed["picture"]:
                    continue
                movie.write_bytes(raw)
                table = probe_frames(movie, cap=160, timeout_s=18, max_pixels=1920 * 1080)
                record = observed["picture"][role]
                frames = record["frames"]
                phone.require(not table.truncated and not table.interpolated and table.time_base == record["time_base"]
                              and tuple(table.pts) == tuple(frame["pts"] for frame in frames), "Picture timeline changed")
                tally["native_timestamps"] += len(table)
                wanted = selected[source["speaker"], source["condition"], role]
                if not wanted:
                    continue
                inspected = set()
                for frame in decode_frames(movie, table, sorted(wanted), timeout_s=25):
                    stored = frames[frame.index]
                    x0, y0, x1, y1 = stored["mouth_box"]
                    h, w = frame.rgb.shape[:2]
                    phone.require(frame.pts_verified and frame.pts == stored["pts"]
                                  and 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
                                  and phone.digest(frame.rgb.tobytes()) == stored["rgb_sha256"]
                                  and phone.digest(frame.rgb[y0:y1, x0:x1].tobytes()) == stored["mouth_crop_sha256"],
                                  "Selected mouth crop differs from submitted pixels")
                    inspected.add(frame.index)
                phone.require(inspected == wanted, "Missing mouth witnesses")
                tally["witness_frames_and_crops"] += len(inspected)
            for role, samples in deliveries(streams, group).items():
                phone.require(phone.digest(samples.tobytes()) == observed["cases"][role]["pcm_sha256"],
                              "A reproduced waveform differs")
                if role in {POSITIVE, "m_same_letter", "vowel_same_letter"}:
                    base = "m" if role in M_ROLES else "vowel"
                    low, high = (group[base]["center"] + s for s in (-SPLICE_HALF, SPLICE_HALF))
                    phone.require((samples[:low] == streams[base][:low]).all() and (samples[high:] == streams[base][high:]).all()
                                  and (samples[low:high] != streams[base][low:high]).any(), "Replacement crossed its bounds")
                tally["waveforms"] += 1
    phone.write(output, {"receipt_sha256": phone.sha(path), **tally})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "held", "review", "replay"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--parents", type=Path, nargs=2)
    for field in ("sources", "runtime", "face", "model", "fit", "output"):
        parser.add_argument(f"--{field}", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        phone.write(args.path, roster(args.sources, args.parents))
        return
    document = json.loads(args.path.read_bytes())
    finished = args.phase in {"review", "replay"}
    frozen = document["freeze"] if finished else document
    phone.require(frozen["schema"] == SCHEMA and frozen["design"] == DESIGN, "Study protocol differs")
    if finished:
        verify_split(frozen, args.parents)
    if args.phase == "development":
        output = observe(args, frozen)
        rows = verify_observations(frozen, output["rows"], args.phase)
        phone.write(args.output, {"freeze_sha256": phone.sha(args.path), "development": output, "fit": calibrate(rows, output["rows"])})
    else:
        fitted = json.loads(args.fit.read_bytes()) if args.phase == "held" else document["fitted"]
        if args.phase == "held":
            phone.require(phone.sha(args.path) == fitted["freeze_sha256"], "Calibration does not match the roster")
        development = fitted["development"]["rows"]
        fit = calibrate(verify_observations(frozen, development, "development"), development)
        phone.require(fit == fitted["fit"], "Calibration does not replay")
        held = observe(args, frozen) if args.phase == "held" else document["held"]
        rows = verify_observations(frozen, held["rows"], "held")
        outcome = counts(rows, held["rows"], fit)
        if args.phase == "held":
            phone.write(args.output, {"freeze": frozen, "fitted": fitted, "held": held, "outcome": outcome})
        else:
            phone.require(outcome == document["outcome"], "Observation counts differ")
            if args.phase == "replay":
                replay(args.path, args.sources, document, args.output)
        print(json.dumps({"fit": fit, "counts": outcome["counts"], "baseline": outcome["baseline"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
