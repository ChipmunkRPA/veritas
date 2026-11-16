# SPDX-License-Identifier: MIT
"""Measure short same-speaker word replacements with multivariate visual articulation."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts import video_lip_content_transfer as parent
from scripts import video_lip_viseme_transfer as study

phone, timing = parent.phone, parent.timing
ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "docs/evaluation-panels/video-lip-content-transfer-2026-11-15.json.gz"
SCHEMA = "chipmunk.video-lip-articulation-transfer/1"
SALT = "2026-11-15-lip-articulation/1"
FIT_SPEAKERS, CALIBRATION_SPEAKERS, HELD_SPEAKERS = 14, 8, 24
FIT_PER_CONDITION = 3
ROLES = parent.ROLES
ARMS = ("aperture_classes", "full_shape")
LEVELS = ("tight", "wide")
METHOD = {
    "sources": "Select disjoint fit, ordinary-only calibration and held speakers by fixed hash order. Exclude media used "
               "in the prior same-speaker comparison. Use unchanged speech for fitting, and locally replace a letter and "
               "digit with duration-matched same-speaker recordings for the paired tests.",
    "fit": "Match dense acoustic phone events on the exact source clock to normalized visual lip landmarks. "
           "Compare a visual class model using aperture alone with one using inner corner width and separate lip gaps. "
           "Use five groups including labiodental sounds, and fit on audio labels from unchanged examples only.",
    "review": "On a local disagreement, use one lag shared by the pre- and post-interval context. Require confirmed picture "
              "timestamps and brackets. Compare the two candidates against the old motion-timing check using the same native "
              "samples, and count same-word splices and uniform delays separately.",
}


def keyed(text):
    return phone.digest(f"{SALT}/{text}".encode())


def frozen_inputs(sources, output):
    from scripts.lombard_mouth_source_intake import index_zip

    previous = json.loads(gzip.decompress(PREVIOUS.read_bytes()))
    used_audio = {source["audio_sha256"] for row in previous["rows"] for source in row["media"].values()}
    used_pictures = {row["picture_sha256"] for row in previous["rows"]}
    paths, publisher = phone.source_archives(sources)
    inventory = defaultdict(lambda: defaultdict(dict))
    rows = []
    with (ZipFile(sources / "publisher-json.zip") as labels, ZipFile(sources / "alignment.zip") as align,
          ZipFile(paths["audio"]) as audio, ZipFile(paths["front"]) as front):
        eye, ear = index_zip(front, ".mov"), index_zip(audio, ".wav")
        bounds = index_zip(align, ".json")
        for member in index_zip(labels, ".json").values():
            for row in json.loads(labels.read(member)):
                group, condition = row["SPKR"], row["COND"]
                uid = f"{group}_{condition}_{row['UTTERANCE']}"
                if condition not in ("l", "p") or row["STATUS"] != "CORRECT" or uid not in bounds or uid not in ear or uid not in eye:
                    continue
                timing_bytes = align.read(bounds[uid])
                span = parent.word_bounds(json.loads(timing_bytes)[uid])
                if span:
                    inventory[group][condition][uid] = (span, timing_bytes)
        fit = 0
        for group in sorted(inventory, key=keyed):
            by_condition = inventory[group]
            if fit < FIT_SPEAKERS:
                clips = {cond: fit_records(by_condition[cond], audio, front, ear, eye, used_audio, used_pictures)
                         for cond in ("p", "l")}
                if any(len(items) != FIT_PER_CONDITION for items in clips.values()):
                    continue
                rows.extend({"speaker": keyed(group), "split": "fit", "condition": cond, "target": item}
                            for cond, cases in clips.items() for item in cases)
                fit += 1
        fit_groups = {row["speaker"] for row in rows}
        used_audio.update(row["target"]["audio_sha256"] for row in rows)
        used_pictures.update(row["target"]["front_sha256"] for row in rows)
        paired = 0
        for group in sorted(inventory, key=keyed):
            if keyed(group) in fit_groups:
                continue
            by_condition = inventory[group]
            clips = {cond: matched_records(by_condition[cond], audio, front, ear, eye, used_audio, used_pictures)
                     for cond in ("p", "l")}
            if not all(clips.values()):
                continue
            split = "calibration" if paired < CALIBRATION_SPEAKERS else "held"
            rows.extend({"speaker": keyed(group), "split": split, "condition": cond, **case} for cond, case in clips.items())
            paired += 1
            if paired == CALIBRATION_SPEAKERS + HELD_SPEAKERS:
                break
    phone.require(fit == FIT_SPEAKERS and paired >= CALIBRATION_SPEAKERS + 16, "Too few distinct source groups")
    phone.write(output, {"schema": SCHEMA, "method": METHOD, "publisher": publisher,
                         "previous_sha256": phone.sha(PREVIOUS), "rows": rows})
    print(f"{fit} fit speakers and {paired} calibration and held speakers frozen", flush=True)


def source_record(uid, case, audio, front, ear, eye, used_audio, used_pictures):
    value = parent.qualified(uid, *case, audio, ear, 64000)
    if value is None or value["audio_sha256"] in used_audio:
        return None
    digest = phone.digest(front.read(eye[uid]))
    if digest in used_pictures:
        return None
    return value | {"front": eye[uid].filename, "front_sha256": digest}


def fit_records(cases, audio, front, ear, eye, used_audio, used_pictures):
    selected = []
    for uid in sorted(cases, key=keyed):
        if (value := source_record(uid, cases[uid], audio, front, ear, eye, used_audio, used_pictures)) is not None:
            selected.append(value)
            if len(selected) == FIT_PER_CONDITION:
                break
    return selected


def matched_records(cases, audio, front, ear, eye, used_audio, used_pictures):
    # Read only initially selected triples, then remove an ineligible member
    # from this recording set and repeat the established pairing rule.
    pool = dict(cases)
    while len(pool) >= 3:
        case = parent.roster_group(pool, audio, ear, front, eye)
        if case is None:
            return None
        excluded = [item["id"] for item in case.values() if item["audio_sha256"] in used_audio
                    or item.get("front_sha256") in used_pictures]
        if not excluded:
            return case
        for uid in excluded:
            pool.pop(uid)
    return None


def audio_reader(checkpoint):
    from app.detectors.video.viseme_worker import aperture_reader

    infer, device = aperture_reader(checkpoint, articulation=True)

    def score(pcm):
        groups, standard, articulation = infer(pcm, include_events=True)
        return {"groups": groups, "standard": standard, "articulation": articulation,
                "pcm_sha256": phone.digest(pcm.tobytes())}

    return score, device


def payload(plan, sources, assets, phase, output, *, aac=False):
    from app.evidence.video_reference_timing import _landmarker, available

    if aac and phase != "held":
        raise ValueError("AAC transfer uses the held clips after freezing the PCM fit and thresholds")
    lock = json.loads(plan.read_bytes())
    phone.require(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["previous_sha256"] == phone.sha(PREVIOUS),
                  "The frozen plan or its parent changed")
    sys.path.insert(0, str((assets / "runtime").resolve()))
    geometry = assets / "face_landmarker.task"
    phone.require(available(geometry) and phone.sha(geometry) == study.parent.FACE_SHA256, "The pinned landmark asset is required")
    measure, device = audio_reader(assets / "checkpoint")
    fields = ("speaker", "split", "condition")
    rows = []
    with (ZipFile(sources / "lombardgrid_audio.zip") as audio, ZipFile(sources / "lombardgrid_front.zip") as front,
          TemporaryDirectory() as scratch, _landmarker(geometry) as (mp, task)):
        path = Path(scratch) / "source.mov"
        for row in lock["rows"]:
            if row["split"] != phase:
                continue
            target = row["target"]
            picture = front.read(target["front"])
            phone.require(phone.digest(picture) == target["front_sha256"], "A frozen picture source changed")
            path.write_bytes(picture)

            def samples(item):
                body = audio.read(item["audio"])
                phone.require(phone.digest(body) == item["audio_sha256"], "A frozen soundtrack changed")
                return phone.pcm(body)

            pcm = samples(target)
            screen = parent.pictures(path, len(pcm), mp, task, with_shape=True)
            versions = {"original": pcm}
            if phase != "fit":
                span = target["span"]
                versions.update({role: value for role, value in study.parent.treatments(pcm, span).items()
                                 if role in ("gain", "global_early", "global_late")})
                for role in ("same_words", "different_words"):
                    variants = parent.splice(pcm, span, samples(row[role]), row[role]["span"])
                    a, b = span
                    phone.require(variants[:a].tobytes() == pcm[:a].tobytes() and variants[b:].tobytes() == pcm[b:].tobytes()
                                  and variants[a:b].tobytes() != pcm[a:b].tobytes(), "The splice left its qualified interval")
                    versions[role] = variants
            if aac:
                from scripts.video_lip_event_alignment import aac_clocked

                versions = {role: aac_clocked(variant, Path(scratch) / "carrier.m4a") for role, variant in versions.items()}
            rows.append({key: row[key] for key in fields} | screen | {
                "truth": target["span"], "picture_sha256": target["front_sha256"],
                "source_audio_sha256": {key: row[key]["audio_sha256"] for key in ("target", "same_words", "different_words") if key in row},
                "deliveries": {role: measure(variant) for role, variant in versions.items()},
            })
            print(f"{phase}: {len(rows)} source clips measured", flush=True)
    packet = {"schema": SCHEMA, "plan_sha256": phone.sha(plan), "phase": phase, "device": device, "rows": rows}
    if aac:
        packet["codec"] = "aac-lc-m4a-48k-16k-mono"
    seal(output, packet)


def seal(path, packet):
    data = json.dumps(packet, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with path.open("xb") as stream:
        stream.write(gzip.compress(data, mtime=0))


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def readout(row, models):
    from app.evidence import video_viseme_articulation as visual

    proposals = {}
    for role, audio in row["deliveries"].items():
        original = parent.measurements(row, audio["groups"], audio["standard"])
        candidates = {key: original[key] for key in ("sparse", "aperture")}
        for arm, model in models.items():
            raw = visual.locate(row["frames"], audio["articulation"], row["samples"], model)
            finding = parent.evidence(row, raw)
            if finding["status"] == "measured":
                anchors = raw["disagreements"]
                finding["disagreements"] = [
                    {key: event[key] for key in ("sample", "heard", "best_visual_class", "score")} | {
                        "frame": row["frames"][event["frame"]]["index"], "pts": row["frames"][event["frame"]]["pts"]}
                    for event in anchors
                ]
            candidates[arm] = finding | {"eligible_events": raw["eligible_events"]}
        proposals[role] = candidates
    return {key: row[key] for key in ("speaker", "split", "condition", "samples", "time_base", "truth", "picture_sha256",
                                     "source_audio_sha256")} | {
        "picture_pts": [frame["pts"] for frame in row["frames"]],
        "pcm": {role: audio["pcm_sha256"] for role, audio in row["deliveries"].items()},
        "readings": proposals,
    }


def operating_points(rows):
    maxima = {arm: {} for arm in ARMS}
    for row in rows:
        for arm in ARMS:
            score = max(finding[arm].get("score", 0.) for role, finding in row["readings"].items() if role != "different_words")
            scores = maxima[arm]
            scores[row["speaker"]] = max(scores.get(row["speaker"], 0.), score)
    phone.require(all(len(scores) == CALIBRATION_SPEAKERS for scores in maxima.values()), "Invalid separate calibration cohort")
    return {arm: dict(zip(LEVELS, sorted(scores.values(), reverse=True)[:2], strict=True)) for arm, scores in maxima.items()}


def calibrate(fit_path, calibration_path, output):
    from app.evidence import video_viseme_articulation as visual

    training, calibration = read(fit_path), read(calibration_path)
    phone.require(training["schema"] == calibration["schema"] == SCHEMA
                  and training["plan_sha256"] == calibration["plan_sha256"]
                  and (training["phase"], calibration["phase"]) == ("fit", "calibration"), "The captures do not use the same plan")
    groups = {row["speaker"] for row in training["rows"]}
    phone.require(len(groups) == FIT_SPEAKERS and not groups & {row["speaker"] for row in calibration["rows"]},
                  "Fit and calibration speakers overlap")
    models = {arm: visual.fit(training["rows"], shape=arm == "full_shape") for arm in ARMS}
    rows = [readout(row, models) for row in calibration["rows"]]
    record = {"schema": SCHEMA, "method": METHOD, "plan_sha256": training["plan_sha256"], "models": models,
              "fit_speakers": sorted(groups), "fit_sha256": phone.sha(fit_path), "calibration_sha256": phone.sha(calibration_path),
              "previous_sha256": phone.sha(PREVIOUS),
              "implementation_sha256": phone.sha(Path(visual.__file__)), "points": operating_points(rows), "calibration": rows}
    seal(output, record)
    print(json.dumps({"points": record["points"], "shape_fit_events": models["full_shape"]["counts"]}, indent=2))


def summarize(rows, points):
    from collections import Counter

    scores = {}
    for arm in ("existing", *ARMS):
        for level in LEVELS:
            totals = defaultdict(Counter)
            for row in rows:
                for role, readings in row["readings"].items():
                    selected = parent.choose(readings, arm, points, level)
                    existing = parent.choose(readings, "existing", points, level)
                    checks = ("sparse", "aperture") if arm == "existing" else ("sparse", "aperture", arm)
                    assessed = any(readings[key]["status"] in {"measured", "no_supported_change"} for key in checks)
                    category = ROLES[role]
                    useful = bool(selected and category == "content" and parent.overlap(row["truth"], selected["span"]))
                    control = parent.choose(row["readings"]["same_words"], arm, points, level) if useful else None
                    same_word_collision = bool(control and parent.overlap(selected["span"], control["span"]))
                    for bucket in (category, f"{row['condition']}/{category}"):
                        totals[bucket].update(files=1, assessed=assessed, flags=bool(selected), useful=useful,
                            unique_correct=useful and not same_word_collision, added=bool(selected and not existing),
                            misplaced=bool(selected and category == "content" and not useful))
            scores[f"{arm}/{level}"] = {kind: dict(count) for kind, count in totals.items()}
    return scores


def evaluate(calibration_path, captures, output):
    from app.evidence import video_viseme_articulation as visual

    lock, held = read(calibration_path), read(captures)
    phone.require(lock["schema"] == held["schema"] == SCHEMA and lock["plan_sha256"] == held["plan_sha256"] and held["phase"] == "held"
                  and lock["implementation_sha256"] == phone.sha(Path(visual.__file__))
                  and lock["points"] == operating_points(lock["calibration"]), "The operating points or held plan changed")
    rows = [readout(row, lock["models"]) for row in held["rows"]]
    packet = lock | {"held_sha256": phone.sha(captures), "cohort": rows, "summary": summarize(rows, lock["points"])}
    if codec := held.get("codec"):
        packet["held_codec"] = codec
    review(packet)
    seal(output, packet)
    print(json.dumps(packet["summary"], indent=2))


def review(packet):
    from fractions import Fraction

    phone.require(packet["schema"] == SCHEMA and packet["method"] == METHOD
                  and packet["previous_sha256"] == phone.sha(PREVIOUS) and packet["points"] == operating_points(packet["calibration"])
                  and packet.get("held_codec") in {None, "aac-lc-m4a-48k-16k-mono"},
                  "The operating points or method differ")
    audio, picture_pairs = set(), set()
    cohorts = {"fit": set(packet["fit_speakers"]), "calibration": set(), "held": set()}
    phones = set(timing.PHONES) | {"lower_lip"}
    phase_rows = {"calibration": packet["calibration"], "held": packet["cohort"]}
    for phase, rows in phase_rows.items():
        per_speaker = defaultdict(set)
        for row in rows:
            per_speaker[row["speaker"]].add(row["condition"])
            clock = Fraction(row["time_base"])
            picture_pts = row["picture_pts"]
            inputs = row["source_audio_sha256"]
            phone.require(24 <= len(picture_pts) <= 120 and all(Fraction(1, 28) <= (b - a) * clock <= Fraction(1, 22)
                          for a, b in zip(picture_pts, picture_pts[1:])), "The source picture clock differs")
            phone.require(row["split"] == phase and set(row["readings"]) == set(row["pcm"]) == set(ROLES)
                          and set(inputs) == {"target", "same_words", "different_words"}
                          and len(set(inputs.values())) == 3 and not audio.intersection(inputs.values())
                          and row["picture_sha256"] not in picture_pairs, "The delivered source cohorts overlap")
            picture_pairs.add(row["picture_sha256"])
            audio.update(inputs.values())
            for readings in row["readings"].values():
                phone.require(set(readings) == {"sparse", "aperture", *ARMS}, "Missing comparison arm")
                for key, finding in readings.items():
                    if finding["status"] != "measured":
                        continue
                    a, b = finding["span"]
                    photos = finding["pictures"]
                    picture_a, picture_b = finding["picture_span"]
                    phone.require(0 <= a < b <= row["samples"] and photos["before"]["sample"] < picture_a
                                  <= photos["during"]["sample"] < picture_b <= photos["after"]["sample"],
                                  "The selected range lacks source context")
                    for photo in photos.values():
                        i = photo["index"]
                        phone.require(photo["pts"] == picture_pts[i]
                                      and photo["sample"] == round((picture_pts[i] - picture_pts[0]) * clock * timing.RATE),
                                      "A review picture differs from its original clock")
                    if key in ARMS:
                        phone.require(finding["change"] == 0 and finding["disagreements"]
                                      and all(a <= event["sample"] < b and event["score"] < 0
                                              and event["heard"] in phones and event["best_visual_class"] in phones
                                              and picture_pts[event["frame"]] == event["pts"] for event in finding["disagreements"]),
                                      "A reported articulation mismatch lacks source evidence")
        phone.require(all(conditions == {"p", "l"} for conditions in per_speaker.values())
                      and len(rows) == 2 * len(per_speaker), "A paired native condition is missing")
        cohorts[phase] = set(per_speaker)
    phone.require(len(cohorts["fit"]) == FIT_SPEAKERS and len(cohorts["calibration"]) == CALIBRATION_SPEAKERS
                  and len(cohorts["held"]) >= 16 and len(set().union(*cohorts.values())) == sum(map(len, cohorts.values())),
                  "A source speaker appears in more than one split")
    summary = summarize(packet["cohort"], packet["points"])
    phone.require(summary == packet["summary"], "Held results differ on replay")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    lock = command.add_parser("freeze")
    lock.add_argument("sources", type=Path)
    lock.add_argument("output", type=Path)
    capture = command.add_parser("capture")
    capture.add_argument("phase", choices=("fit", "calibration", "held"))
    capture.add_argument("--aac", action="store_true")
    for name in ("plan", "sources", "assets", "output"):
        capture.add_argument(name, type=Path)
    calibration = command.add_parser("calibrate")
    for name in ("fit", "calibration", "output"):
        calibration.add_argument(name, type=Path)
    held = command.add_parser("held")
    for name in ("calibration", "captures", "output"):
        held.add_argument(name, type=Path)
    audit = command.add_parser("review")
    audit.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        frozen_inputs(args.sources, args.output)
    elif args.command == "capture":
        payload(args.plan, args.sources, args.assets, args.phase, args.output, aac=args.aac)
    elif args.command == "calibrate":
        calibrate(args.fit, args.calibration, args.output)
    elif args.command == "held":
        evaluate(args.calibration, args.captures, args.output)
    else:
        print(json.dumps(review(read(args.receipt)), indent=2))


if __name__ == "__main__":
    main()
