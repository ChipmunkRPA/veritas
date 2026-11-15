# SPDX-License-Identifier: MIT
"""Measure whether unconditional speech-phone patterns improve local lip timing."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from app.evidence.video_viseme_timing import PHONES, STEP, compatible, hypotheses, nearest, observations, picture_evidence
from scripts import video_lip_event_alignment as prior

parent = prior.parent
phone = parent.phone
RATE = parent.RATE
SCHEMA = "chipmunk.video-lip-viseme-transfer/1"
TRANSFER_SCHEMA = "chipmunk.video-lip-viseme-carrier/1"
TRANSFER_SPEAKERS = 8
CODECS = ("pcm", "aac48")
METHOD = {
    "input": "Use the pinned unprompted phone decoder and all native picture aperture observations. In addition "
             "to lip closures, measure three classes of vowels and glides. Apply the same supported picture clock "
             "at every candidate lag. Existing B/P timing results are retained for comparison.",
    "fit": "Learn the median normalized aperture for each acoustic class from untouched first-cohort development "
           "recordings, all with their original clocks. Normalize visible inner-lip aperture independently inside "
           "each clip by its 10th-to-90th percentile range. Use at least eight observations and two fit speakers per class.",
    "comparison": "Score one shared lag over the same acoustic events as a two-lag hypothesis. Evaluate 400, 560 "
                  "and 720 ms intervals every 40 ms. Require support on both flanks and at least two interior events. "
                  "Both flanks must individually support the same background lag over the proposed local lag. "
                  "Count the improvement in total compatibility over the best single lag, weighted by event confidence.",
    "freeze": "Set tight and wider points from per-speaker maxima on second-cohort ordinary gain, filtering, original "
              "and constant-delay controls before opening held comparisons. Evaluate each waveform on its own.",
}


def timings():
    document = json.loads(prior.SOURCE.read_bytes())
    parent.review(document, prior.ACOUSTIC)
    return document


def load(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_bytes())


def is_fit(row):
    return row["cohort"] == 1 and row["split"] == "development"


def is_calibration(row):
    return row["cohort"] == 2 and row["split"] == "development"


def decoder(checkpoint):
    label_by_prefix = {name[0]: label for label, names in PHONES.items() for name in names}
    acoustic, device = phone.neural_decoder(checkpoint, names=tuple(token for names in PHONES.values() for token in names))

    def events(pcm):
        return [[label_by_prefix[letter], sample, prob] for letter, sample, prob in acoustic(pcm)]

    return events, device


def capture(args):
    base = timings()
    declared, _ = phone.source_archives(args.sources)
    acoustic, device = decoder(args.checkpoint)
    output = []
    with ZipFile(declared["audio"]) as archive:
        for i, row in enumerate(base["rows"], 1):
            spec = row["members"]["audio"]
            source = archive.read(spec["name"])
            phone.require(phone.digest(source) == spec["sha256"], "Source waveform differs from the committed member")
            deliveries = parent.treatments(phone.pcm(source), row["edit"])
            record = {"source": row["source"], "cohort": row["cohort"], "cases": {}}
            for role, pcm in deliveries.items():
                if is_fit(row) and role != "original":
                    continue
                digest = phone.digest(pcm.tobytes())
                phone.require(digest == row["cases"][role]["pcm_sha256"], "Edited or control waveform differs")
                record["cases"][role] = {"pcm_sha256": digest, "phones": acoustic(pcm)}
            output.append(record)
            if i % 8 == 0:
                print(f"Measured phoneme classes for {i}/{len(base['rows'])} recordings.", flush=True)
    phone.write(args.output, {"method": METHOD, "device": device, "parent_sha256": phone.sha(prior.SOURCE), "rows": output})


def fit(base, captured):
    groups = defaultdict(list)
    for row, measured in zip(base, captured, strict=True):
        if not is_fit(row) or (points := observations(row)) is None:
            continue
        for label, sample, _ in measured["cases"]["original"]["phones"]:
            if (value := nearest(points, sample)) is not None:
                groups[label].append((row["speaker"], value[0]))
    result = {label: {"median": round(median(point for _, point in values), 7),
                      "events": len(values), "speakers": len({speaker for speaker, _ in values})}
              for label, values in groups.items() if len(values) >= 8 and len({speaker for speaker, _ in values}) >= 2}
    phone.require("closure" in result and len(result) >= 2, "Insufficient shape calibration in the original recordings")
    return result


def arm(record, label):
    if label == "all_visible":
        return record
    return [event for event in record if event[0] == "closure"]


def summarize(rows, points):
    outcomes = defaultdict(Counter)
    for row in rows:
        direction = parent.EDITS.get(row["role"])
        kind = "local" if direction else "constant" if row["role"].startswith("global_") else "ordinary"
        for name, values in points.items():
            result = row["arms"][name]
            for point, threshold in values.items():
                counts = outcomes[f"{name}/{point}/{kind}"]
                hit = result.get("score", 0.) > threshold
                useful = bool(direction and hit and parent.useful(result["span"], result["change"], row["truth"], direction))
                counts.update(cases=1, assessed=result["status"] != "insufficient_events",
                              proposals=result["status"] == "measured", flagged=hit,
                              useful=useful, misplaced=bool(direction and hit and not useful),
                              useful_without_point=bool(direction and result["status"] == "measured"
                                                        and parent.useful(result["span"], result["change"], row["truth"], direction)),
                              added_useful=useful and not row["prior_useful"])
    return dict(outcomes)


def score_rows(base, captured, references, *, calibration, original, ordered):
    threshold = original["result"]["points"]["neural_flanked"]["coverage"]
    bias = original["result"]["bias"]["neural"]["value"]
    for row, data in zip(base, captured, strict=True):
        if is_fit(row) or is_calibration(row) != calibration:
            continue
        for role, value in data["cases"].items():
            direction = parent.EDITS.get(role)
            truth = row["edit"] if direction else None
            older = ordered.get((row["cohort"], row["source"], role))
            _, flanked = parent.proposals(row, role, "neural_flanked", bias)
            old_useful = bool(direction and flanked and flanked["strength"] > threshold
                              and parent.useful(flanked["span_samples"], flanked["offset_samples"], truth, direction))
            ordered_useful = bool(direction and older and older.get("improvement", 0.) > ordered["point"]
                                  and parent.useful(older["supported"], older["offset_samples"], truth, direction))
            yield {"source": row["source"], "speaker": row["speaker"], "role": role, "truth": truth,
                   "prior_useful": old_useful or ordered_useful,
                   "arms": {label: hypotheses(compatible(row, arm(value["phones"], label), references), row["samples"])
                            for label in ("closure", "all_visible")}}


def valid_events(events, extent):
    samples = [sample for _, sample, _ in events]
    return samples == sorted(set(samples)) and all(
        label in PHONES and .35 <= probability <= 1. and 0 <= sample < extent
        and (sample-phone.CENTER) % phone.STRIDE == 0 for label, sample, probability in events
    )


def measure(captured, *, points_path=None):
    source = timings()
    base = source["rows"]
    phone.require(captured["parent_sha256"] == phone.sha(prior.SOURCE) and captured["method"] == METHOD
                  and len(base) == len(captured["rows"]), "Acoustic evidence belongs to a different source protocol")
    for row, data in zip(base, captured["rows"], strict=True):
        expected = {"original"} if is_fit(row) else set(parent.ROLES)
        phone.require(data["source"] == row["source"] and data["cohort"] == row["cohort"] and set(data["cases"]) == expected,
                      "Acoustic row differs from the frozen source roster")
        for role, value in data["cases"].items():
            phone.require(value["pcm_sha256"] == row["cases"][role]["pcm_sha256"], "Acoustic row has a different source clock")
            phone.require(valid_events(value["phones"], row["samples"]),
                "Invalid decoded phone timestamps or probabilities")
    references = fit(base, captured["rows"])
    ordered_path = prior.PANEL / "video-lip-event-alignment-2026-11-15.json"
    ordered = json.loads(ordered_path.read_bytes())
    phone.require(ordered["freeze"]["parent_sha256"] == captured["parent_sha256"], "Earlier timing sources differ")
    previous = {(r["cohort"], r["source"], r["role"]): r for r in ordered["held"]} | {"point": ordered["points"]["wider"]}
    calibration = list(score_rows(base, captured["rows"], references, calibration=True, original=source, ordered=previous))
    points = {}
    for name in ("closure", "all_visible"):
        maxima = defaultdict(float)
        for row in calibration:
            if row["truth"] is None:
                maxima[row["speaker"]] = max(maxima[row["speaker"]], row["arms"][name].get("score", 0.))
        strength = sorted(maxima.values(), reverse=True)
        phone.require(len(strength) == 4, "Expected four separate point speakers")
        points[name] = {"tight": strength[0], "wider": strength[1]}
    if points_path is not None:
        phone.write(points_path, {"references": references, "points": points, "parents": [captured["parent_sha256"], phone.sha(ordered_path)]})
    held = list(score_rows(base, captured["rows"], references, calibration=False, original=source, ordered=previous))
    return {"schema": SCHEMA, "ordered_sha256": phone.sha(ordered_path), "references": references, "points": points, "calibration": calibration,
            "held": held, "outcomes": summarize(held, points)}


def transfer_input(primary):
    result = measure(primary["acoustic"])
    phone.require(result == primary["result"], "Primary timing commitments differ")
    base = timings()
    excluded = sorted({row["speaker"] for row in base["rows"]})
    return result, excluded


def transfer_result(record, primary):
    from fractions import Fraction

    committed, excluded = transfer_input(primary)
    phone.require(record["schema"] == TRANSFER_SCHEMA and record["excluded_speakers"] == excluded
                  and record["references"] == committed["references"] and record["points"] == committed["points"],
                  "Transfer changed the scope or operating points")
    speakers = defaultdict(set)
    for row in record["rows"]:
        phone.require(row["speaker"] not in set(excluded).union(phone.EXCLUDED) and row["source"] not in speakers[row["speaker"]],
                      "The transfer repeats a fit, calibration or held speaker")
        speakers[row["speaker"]].add(row["source"])
    phone.require(len(speakers) == TRANSFER_SPEAKERS and all(len(ids) == 2 for ids in speakers.values()), "The transfer roster differs")
    for speaker in speakers:
        phone.require({row["condition"] for row in record["rows"] if row["speaker"] == speaker} == {"p", "l"},
                      "Plain and Lombard speech are both required")
    points, references = committed["points"], committed["references"]
    outcomes, witnesses = {}, {}
    for codec in CODECS:
        held, existing = [], defaultdict(Counter)
        for row in record["rows"]:
            frames = row["frames"]
            clock = Fraction(row["time_base"])
            phone.require(len(row["edit"]) == 2 and 0 < row["edit"][0] < row["edit"][1] <= row["samples"]
                          and len(frames) >= 24 and all(a["pts"] < b["pts"] for a, b in zip(frames, frames[1:]))
                          and all(i == frame["index"] and frame["sample"] == round((frame["pts"] - frames[0]["pts"]) * clock * RATE)
                                  for i, frame in enumerate(frames)), "Source picture timing differs")
            phone.require(set(row["cases"]) == set(CODECS) and set(row["cases"][codec]) == set(parent.ROLES), "Missing matched carrier cases")
            for role, case in row["cases"][codec].items():
                expected = row["cases"]["pcm"][role]["sha256"]
                phone.require(case["original_sha256"] == expected and valid_events(case["phones"], row["samples"]),
                              "Carrier input or decoded phone timing differs")
                shift, picture = parent.EDITS.get(role), case["current"]
                truth = row["edit"] if shift is not None else None
                kind = "local" if shift else "constant" if role.startswith("global_") else "ordinary"
                old_useful = bool(shift and any(lag is not None and parent.useful(
                    [a * 320 + lag * STEP, b * 320 + lag * STEP], lag * STEP, truth, shift) for _, a, b, lag in picture["ranges"]))
                existing[kind].update(files=1, assessed=bool(picture["starts"]), flagged=bool(picture["ranges"]), useful=old_useful)
                arms = {name: hypotheses(compatible(row, arm(case["phones"], name), references), row["samples"])
                        for name in points}
                held.append({"role": role, "truth": truth, "prior_useful": old_useful, "arms": arms})
                result = arms["all_visible"]
                if shift and result.get("score", 0.) > points["all_visible"]["tight"] and parent.useful(
                        result["span"], result["change"], truth, shift):
                    witnesses[f"{codec}/{row['source']}/{role}"] = picture_evidence(frames, result, row["samples"])
        outcomes[codec] = {"candidate": summarize(held, points), "current": dict(existing)}
    return {"outcomes": outcomes, "witnesses": witnesses}


def transfer(args):
    import sys

    import cv2
    import numpy as np

    from app.detectors.image.faces import load_yunet
    from app.detectors.video import lipsync
    from app.evidence.video_reference_timing import _landmarker
    from scripts.video_lipsync_signed_transitions import current
    from scripts.video_lipsync_window_schedule import FACE_FILE, FACE_SHA

    primary = load(args.primary)
    verified, excluded = transfer_input(primary)
    phone.require(phone.sha(args.geometry) == parent.FACE_SHA256 and phone.sha(FACE_FILE) == FACE_SHA, "Face assets differ")
    prepared = phone.source_rows(args.sources, extra_excluded=excluded, speaker_count=TRANSFER_SPEAKERS, clips_per_condition=1)
    roster = {key: prepared[key] for key in ("archives", "metadata_sha256", "revision", "model_sha256", "rows")}
    sys.path.insert(0, str(args.runtime))
    cv2.setNumThreads(1)
    locations = {name: args.sources / f"lombardgrid_{name}.zip" for name in ("front", "audio")}
    acoustic, device = decoder(args.checkpoint)
    rows = []
    with (ZipFile(locations["audio"]) as sound, ZipFile(locations["front"]) as front,
          TemporaryDirectory(prefix="av-viseme-carrier-") as directory, _landmarker(args.geometry) as (mp, task)):
        run = Path(directory)
        media = run / "original.mov"
        yunet = load_yunet(FACE_FILE)
        for i, row in enumerate(roster["rows"], 1):
            photo = front.read(row["members"]["front"]["name"])
            waveform = sound.read(row["members"]["audio"]["name"])
            phone.require(phone.digest(photo) == row["members"]["front"]["sha256"]
                          and phone.digest(waveform) == row["members"]["audio"]["sha256"], "Source pair differs")
            media.write_bytes(photo)
            original = phone.pcm(waveform)
            frames, time_base, (faces, patches, lookup, grid) = parent.picture_observations(media, len(original), mp, task, yunet)
            deliveries = parent.treatments(original, row["edit"])
            cases = {name: {} for name in CODECS}
            for role, wave in deliveries.items():
                digest = phone.digest(wave.tobytes())
                received = prior.aac_clocked(wave, run / "carrier.m4a")
                for codec, pcm in zip(CODECS, (wave, received), strict=True):
                    cases[codec][role] = {"sha256": phone.digest(pcm.tobytes()), "original_sha256": digest,
                                          "phones": acoustic(pcm),
                                          "current": current(faces, patches, lookup, lipsync.vad(pcm.astype(np.float32)/32768.), grid)}
            rows.append({key: row[key] for key in ("source", "speaker", "condition", "samples", "edit", "members")}
                        | {"frames": frames, "time_base": time_base, "cases": cases})
            print(f"Measured PCM and AAC on fresh source {i}/{len(roster['rows'])}.", flush=True)
    record = {"schema": TRANSFER_SCHEMA, "primary_sha256": phone.sha(args.primary), "excluded_speakers": excluded,
              "points": verified["points"], "references": verified["references"], "device": device,
              "archives": roster["archives"], "metadata_sha256": roster["metadata_sha256"], "rows": rows}
    phone.write(args.output, record | transfer_result(record, primary))


def native_audit(args):
    from app.detectors.video.decode import decode_frames, probe_frames

    receipt = load(args.receipt)
    phone.require(receipt["primary_sha256"] == phone.sha(args.primary), "Primary comparison digest differs")
    result = transfer_result(receipt, load(args.primary))
    phone.require(all(result[key] == receipt[key] for key in result), "Transfer outcome differs")
    locations, metadata = phone.source_archives(args.sources)
    phone.require(metadata == receipt["archives"], "Publisher archive bindings differ")
    wanted = defaultdict(set)
    for key, example in receipt["witnesses"].items():
        _, source, _ = key.split("/")
        anchors = example["timing_anchor_frames"]
        frames = [anchors["left_audio_context"], *anchors["local_audio_events"], anchors["right_audio_context"],
                  *example["picture_brackets"].values()]
        wanted[source].update(frame["index"] for frame in frames if frame is not None)
    counts = Counter()
    with (ZipFile(locations["front"]) as pictures, ZipFile(locations["audio"]) as audio,
          TemporaryDirectory(prefix="av-viseme-source-audit-") as directory):
        scratch = Path(directory)
        path = scratch / "original.mov"
        for row in receipt["rows"]:
            video_spec, audio_spec = row["members"]["front"], row["members"]["audio"]
            video, sound = pictures.read(video_spec["name"]), audio.read(audio_spec["name"])
            phone.require(phone.digest(video) == video_spec["sha256"] and phone.digest(sound) == audio_spec["sha256"],
                          "Publisher member bytes differ")
            for kind, pcm in parent.treatments(phone.pcm(sound), row["edit"]).items():
                decoded = prior.aac_clocked(pcm, scratch / "carrier.m4a")
                for codec, wave in zip(CODECS, (pcm, decoded), strict=True):
                    phone.require(phone.digest(wave.tobytes()) == row["cases"][codec][kind]["sha256"],
                                  "Reconstructed carrier samples differ")
                    counts[f"{codec}_waveforms"] += 1
            path.write_bytes(video)
            table = probe_frames(path, cap=160, timeout_s=18, max_pixels=1920*1080)
            recorded = row["frames"]
            phone.require(not table.interpolated and not table.truncated and table.time_base == row["time_base"]
                          and tuple(table.pts) == tuple(frame["pts"] for frame in recorded), "Source picture clock differs")
            counts["picture_timestamps"] += len(table)
            indices = sorted(wanted[row["source"]])
            if indices:
                decoded = []
                for frame in decode_frames(path, table, indices, timeout_s=25):
                    record = recorded[frame.index]
                    x0, y0, x1, y1 = record["mouth_box"]
                    phone.require(frame.pts_verified and frame.pts == record["pts"]
                                  and phone.digest(frame.rgb.tobytes()) == record["rgb_sha256"]
                                  and phone.digest(frame.rgb[y0:y1, x0:x1].tobytes()) == record["mouth_crop_sha256"],
                                  "Recorded source picture or mouth crop differs")
                    decoded.append(frame.index)
                phone.require(decoded == indices, "Source picture replay omitted a recorded witness")
                counts["source_frame_and_mouth_crops"] += len(decoded)
    print(json.dumps(dict(counts), sort_keys=True, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase, names in (("capture", ("sources", "checkpoint", "output")), ("measure", ("acoustic", "output")),
                         ("review", ("receipt",)), ("transfer", ("primary", "sources", "checkpoint", "geometry", "runtime", "output")),
                         ("review-transfer", ("receipt", "primary")), ("audit-source", ("receipt", "primary", "sources"))):
        sub = phases.add_parser(phase)
        for name in names:
            sub.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.phase == "capture":
        capture(args)
    elif args.phase == "measure":
        captured = load(args.acoustic)
        phone.write(args.output, {"acoustic": captured,
                                  "result": measure(captured, points_path=args.output.with_suffix(".points.json"))})
    elif args.phase == "transfer":
        transfer(args)
    elif args.phase == "audit-source":
        native_audit(args)
    elif args.phase == "review-transfer":
        record = load(args.receipt)
        phone.require(record["primary_sha256"] == phone.sha(args.primary), "Primary comparison digest differs")
        result = transfer_result(record, load(args.primary))
        phone.require(all(result[key] == record[key] for key in result), "Transfer outcome differs")
        print(json.dumps(result["outcomes"], sort_keys=True, indent=2))
    else:
        record = load(args.receipt)
        result = measure(record["acoustic"])
        phone.require(result == record["result"], "Viseme timing observations or counts differ")
        print(json.dumps({"references": result["references"], "points": result["points"], "outcomes": result["outcomes"]},
                         sort_keys=True, indent=2))
