# SPDX-License-Identifier: MIT
"""Compare native mouth width and aperture on the same held lip-timing cases."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from app.evidence import video_viseme_binding as binding
from app.evidence import video_viseme_ranked as ranked
from app.evidence import video_viseme_timing as timing
from scripts import video_lip_dense_native as native
from scripts import video_lip_dense_posteriors as dense

study = dense.prior
PANEL = study.prior.PANEL
PRIMARY = PANEL / "video-lip-dense-posteriors-2026-11-15.json.gz"
CARRIER = PANEL / "video-lip-dense-carrier-2026-11-15.json.gz"
SPARSE = PANEL / "video-lip-viseme-transfer-2026-11-15.json.gz"
ARMS = ("sparse", "aperture", "width", "both")
LEVELS = ("tight", "wider")
SCHEMA = "chipmunk.video-lip-mouth-width/1"
NATIVE_SCHEMA = "chipmunk.video-lip-mouth-width-native/1"
METHOD = {
    "pixels": "Replay the committed native pictures and require the original mouth openings, crops, and picture hashes.",
    "fit": "Apply the existing aperture ridge fit separately to eye-normalized horizontal lip width, with the same first-cohort originals.",
    "scores": "Compare the shipped aperture ranks, width ranks and the equal-weight mean of both lag score vectors. Keep native frame indices, event sample positions, and lag hypotheses fixed.",
    "points": "Keep the shipped aperture and sparse points. For width and combined ranks use the largest and second-largest separate speaker maxima among the second-cohort ordinary and constant-delay controls.",
    "fallback": "Compare sparse first, then aperture, then the width or two-shape rule on cases without a qualifying prior interval. Count signed correctly placed intervals and all ordinary and constant-delay flags.",
}


def picture_widths(path, row, mp, task):
    from app.detectors.video.decode import decode_frames, probe_frames
    from app.evidence.video_reference_timing import _mesh

    table = probe_frames(path, cap=160, timeout_s=18, max_pixels=1920 * 1080)
    expected = row["frames"]
    if table.truncated or table.interpolated or table.time_base != row["time_base"] or list(table.pts) != [f["pts"] for f in expected]:
        raise ValueError("The picture clock differs from the committed source")
    widths = []
    digest = study.phone.digest
    for saved, frame in zip(expected, decode_frames(path, table, list(range(len(table))), timeout_s=25), strict=True):
        if (not frame.pts_verified or (frame.index, frame.pts) != (saved["index"], saved["pts"])
                or digest(frame.rgb.tobytes()) != saved["rgb_sha256"]):
            raise ValueError("A source picture differs from the committed frame")
        measured = _mesh(task, mp, frame.rgb)
        if measured is None:
            if saved["gap"] is not None:
                raise ValueError("Committed mouth landmarks are missing")
            widths.append(None)
            continue
        features, box = measured
        x, y, right, bottom = box
        if (round(float(median(features[:3])), 7) != saved["gap"] or box != saved["mouth_box"]
                or digest(frame.rgb[y:bottom, x:right].tobytes()) != saved["mouth_crop_sha256"]):
            raise ValueError("Mouth landmarks or crop differ from the committed frame")
        widths.append(round(float(features[3]) * timing.SCORE_UNITS))
    return widths


def capture_pictures(args):
    from app.evidence.video_reference_timing import _landmarker, available

    sys.path.insert(0, str(args.runtime.resolve()))
    if study.phone.sha(args.geometry) != study.parent.FACE_SHA256 or not available(args.geometry):
        raise ValueError("The pinned face asset and runtime are required")
    rows = study.timings()["rows"] + study.load(dense.CARRIER_SOURCE)["rows"]
    if len({row["source"] for row in rows}) != len(rows):
        raise ValueError("Primary and carrier recordings must remain separate")
    widths = {}
    with (ZipFile(args.sources / "lombardgrid_front.zip") as archive, TemporaryDirectory() as scratch,
          _landmarker(args.geometry) as (mp, task)):
        path = Path(scratch) / "original.mov"
        for i, row in enumerate(rows, 1):
            member = row["members"]["front"]
            payload = archive.read(member["name"])
            if study.phone.digest(payload) != member["sha256"]:
                raise ValueError("Publisher picture bytes differ from the committed member")
            path.write_bytes(payload)
            widths[row["source"]] = picture_widths(path, row, mp, task)
            if i % 8 == 0:
                print(f"Verified mouth measurements for {i}/{len(rows)} recordings.", flush=True)
    dense.seal(args.output, widths)


def capture_carrier(args):
    read, device = dense.reader(args.checkpoint)
    rows = study.load(dense.CARRIER_SOURCE)["rows"]
    with (ZipFile(args.sources / "lombardgrid_audio.zip") as archive, TemporaryDirectory() as scratch,
          args.output.open("x", encoding="utf-8", buffering=1) as output):
        delivery = Path(scratch) / "carrier.m4a"
        for i, row in enumerate(rows, 1):
            member = row["members"]["audio"]
            payload = archive.read(member["name"])
            if study.phone.digest(payload) != member["sha256"]:
                raise ValueError("Publisher audio bytes differ from the committed member")
            cases = {codec: {} for codec in study.CODECS}
            for role, wave in study.parent.treatments(study.phone.pcm(payload), row["edit"]).items():
                aac = study.prior.aac_clocked(wave, delivery)
                for codec, pcm in zip(study.CODECS, (wave, aac), strict=True):
                    digest = study.phone.digest(pcm.tobytes())
                    if digest != row["cases"][codec][role]["sha256"]:
                        raise ValueError("An acoustic carrier differs from the committed samples")
                    cases[codec][role] = {"pcm_sha256": digest, "groups": read(pcm)}
            output.write(json.dumps({"source": row["source"], "device": device, "cases": cases}) + "\n")
            print(f"Measured PCM and AAC for {i}/{len(rows)} recordings.", flush=True)


def width_row(row, widths):
    frames = []
    for frame, width in zip(row["frames"], widths, strict=True):
        if (frame["gap"] is None) != (width is None) or (width is not None and (type(width) is not int or width <= 0)):
            raise ValueError("Mouth width must be present on exactly the same face frames as aperture")
        frames.append(frame | {"gap": None if width is None else width / timing.SCORE_UNITS})
    return row | {"frames": frames}


def combined(aperture, width):
    if not aperture or not width:
        return []
    events = []
    for left, right in zip(aperture, width, strict=True):
        sample, _, scores, frames = left
        if right[0] != sample or right[3] != frames:
            raise ValueError("Both mouth measurements must use the same source samples and pictures")
        events.append((sample, "both", [round((a + b) / 2) for a, b in zip(scores, right[2], strict=True)], frames))
    return events


def decide(row, wide, aperture, width, phones, references):
    if (len(aperture) != len(width) or any((a is None) != (w is None) for a, w in zip(aperture, width, strict=True))
            or any(w is not None and (type(w) is not int or not -ranked.UNIT // 2 <= w <= 3 * ranked.UNIT // 2) for w in width)):
        raise ValueError("Predicted mouth width differs from the expected acoustic clock or scale")
    opened, spread = ranked.scores(row, aperture), ranked.scores(wide, width)
    inputs = dict(zip(ARMS, (timing.compatible(row, phones, references), opened, spread, combined(opened, spread)), strict=True))
    readings = {}
    for name, events in inputs.items():
        decision = timing.hypotheses(events, row["samples"])
        if decision["status"] == "measured":
            evidence = timing.picture_evidence(row["frames"], decision, row["samples"])
            if not all(evidence["picture_brackets"].values()):
                decision = {"status": "incomplete_context", "eligible_events": len(events)}
        readings[name] = decision
    return readings


def read_lines(path):
    value = {}
    with path.open(encoding="utf-8") as entries:
        for entry in map(json.loads, entries):
            source = entry["source"]
            if source in value:
                raise ValueError("A recorded source is duplicated")
            value[source] = entry
    return value


def predictions(raw, existing, fitted):
    if set(raw) != set(existing):
        raise ValueError("The acoustic deliveries differ from the earlier replay")
    result = {}
    for role, measured in raw.items():
        earlier = existing[role]
        groups = measured["groups"]
        if measured["pcm_sha256"] != earlier["pcm_sha256"] or ranked.predict(groups) != earlier["prediction"]:
            raise ValueError("The aperture comparator changed during reconstruction")
        result[role] = ranked.predict(groups, fitted)
    return result


def cases(rows, widths, original, spread, phones, references):
    for row in rows:
        source = row["source"]
        wide = width_row(row, widths[source])
        if set(original[source]) != set(spread[source]) or set(original[source]) != set(phones[source]):
            raise ValueError("A comparison delivery is missing")
        for role, existing in original[source].items():
            decision = decide(row, wide, existing["prediction"], spread[source][role], phones[source][role], references)
            yield row, role, decision


def operating_points(calibration, sparse_points):
    maxima = {name: {} for name in ARMS[1:]}
    for row, role, readings in calibration:
        if role in study.parent.EDITS:
            continue
        for name, groups in maxima.items():
            speaker = row["speaker"]
            groups[speaker] = max(groups.get(speaker, 0.), readings[name].get("score", 0.))
    points = {"sparse": sparse_points}
    for name, groups in maxima.items():
        values = sorted(groups.values(), reverse=True)
        if len(values) != 4:
            raise ValueError("Calibration requires four distinct speaker controls")
        points[name] = dict(zip(LEVELS, values[:2], strict=True))
    if points["aperture"]["tight"] != ranked.POINT:
        raise ValueError("The aperture comparator must reproduce the shipped point")
    return points


def summarize(deliveries, points):
    counts, evidence = {}, {}
    routes = {"current": ARMS[:2], "add_width": ARMS[:3], "add_both": (*ARMS[:2], "both")}
    routes.update({name: (name,) for name in ARMS})
    for row, role, readings in deliveries:
        shift = study.parent.EDITS.get(role)
        kind = "local" if shift else "constant" if role.startswith("global_") else "ordinary"
        for level in LEVELS:
            flagged = {name for name in ARMS if readings[name].get("score", 0.) > points[name][level]}
            for route, order in routes.items():
                name = next((name for name in order if name in flagged), None)
                result = readings[name] if name else None
                useful = bool(shift and result and study.parent.useful(result["span"], result["change"], row["edit"], shift))
                key = f"{route}/{level}/{kind}"
                counts.setdefault(key, Counter()).update(
                    files=1, flagged=bool(result), useful=useful, misplaced=bool(shift and result and not useful),
                    assessed=any(readings[arm]["status"] not in {"insufficient_events", "ambiguous_change", "incomplete_context"}
                                 for arm in order),
                )
                if useful and name == order[-1] and route in {"add_width", "add_both"}:
                    evidence[f"{route}/{level}/{row['source']}/{role}"] = timing.picture_evidence(row["frames"], result, row["samples"])
    return {"counts": counts, "new_locations": evidence}


def inputs():
    primary, carrier, sparse = map(study.load, (PRIMARY, CARRIER, SPARSE))
    references = {label: {"median": values["median"]} for label, values in sparse["result"]["references"].items()}
    if (tuple(primary["fit"]["weights"]) != ranked.WEIGHTS
            or sparse["result"]["points"]["all_visible"]["tight"] != binding.POINT
            or references != binding.REFERENCE
            or carrier["primary_sha256"] != study.phone.sha(PRIMARY)):
        raise ValueError("The archived predictions and sound classes must match the local lip checks")
    return primary, carrier, sparse


def evaluate(widths, values):
    primary, carrier, sparse = inputs()
    main_rows = study.timings()["rows"]
    carrier_source = study.load(dense.CARRIER_SOURCE)
    if set(widths) != {row["source"] for row in main_rows + carrier_source["rows"]}:
        raise ValueError("A native picture recording is missing")
    phones = {row["source"]: {role: data["phones"] for role, data in row["cases"].items()}
              for row in sparse["acoustic"]["rows"]}
    split = {"calibration": [row for row in main_rows if study.is_calibration(row)],
             "held": [row for row in main_rows if row["split"] == "held"]}
    roles = ([row for row in main_rows if study.is_fit(row)], split["calibration"], split["held"], carrier_source["rows"])
    speaker_sets = [{row["speaker"] for row in cohort} for cohort in roles]
    if any(a & b for a, b in combinations(speaker_sets, 2)):
        raise ValueError("Fit, calibration and test speakers must remain separate")

    def primary_cases(part):
        return cases(split[part], widths, primary["predictions"], values["primary"], phones, sparse["result"]["references"])

    points = operating_points(primary_cases("calibration"), sparse["result"]["points"]["all_visible"])
    panels = {"primary_pcm": summarize(primary_cases("held"), points)}
    for codec in study.CODECS:
        original = {source: entries[codec] for source, entries in carrier["predictions"].items()}
        carrier_phones = {row["source"]: {role: item["phones"] for role, item in row["cases"][codec].items()}
                          for row in carrier_source["rows"]}
        panels[f"transfer_{codec}"] = summarize(cases(carrier_source["rows"], widths, original, values[codec],
                                                      carrier_phones, carrier_source["references"]), points)
    return {"points": points, "panels": panels}


def report(args):
    primary, carrier, _ = inputs()
    widths = study.load(args.widths)
    source = read_lines(args.primary_sound)
    raw = read_lines(args.carrier_sound)
    rows = study.timings()["rows"]
    if set(source) != {row["source"] for row in rows} or set(raw) != set(carrier["predictions"]):
        raise ValueError("An acoustic recording was added or removed")
    for row in rows:
        if study.is_fit(row) and source[row["source"]]["cases"]["original"]["pcm_sha256"] != row["cases"]["original"]["pcm_sha256"]:
            raise ValueError("Width training must use the unchanged fit recordings")
    width_fit = dense.fit([width_row(row, widths[row["source"]]) for row in rows], source)
    values = {"primary": {key: predictions(source[key]["cases"], old, width_fit["weights"])
                          for key, old in primary["predictions"].items()}}
    for codec in study.CODECS:
        values[codec] = {key: predictions(raw[key]["cases"][codec], old[codec], width_fit["weights"])
                         for key, old in carrier["predictions"].items()}
    result = evaluate(widths, values)
    dense.seal(args.output, {"schema": SCHEMA, "method": METHOD, "sources": source_pins(), "width_fit": width_fit,
                             "widths": widths, "width_predictions": values, "result": result})
    return result


def source_pins():
    return {path.name: study.phone.sha(path) for path in (PRIMARY, CARRIER, SPARSE, study.prior.SOURCE, dense.CARRIER_SOURCE)}


def review(path):
    receipt = study.load(path)
    if (receipt["schema"] != SCHEMA or receipt["method"] != METHOD or receipt["sources"] != source_pins()
            or receipt["result"] != evaluate(receipt["widths"], receipt["width_predictions"])):
        raise ValueError("The mouth-width receipt differs from its replay")
    return receipt["result"]


def native_deliveries(args, qualified, read, weights, references):
    for number, item in enumerate(qualified, 1):
        source = args.media / f"asset-{item['index']:03d}.flv"
        frames, table = native.pixels(source, item, args.geometry, measure_width=True)
        original = native.audio(source, item)
        extent, window = len(original), native.native.EDIT_WIDTH
        origin = round(Fraction(item["audio_clocks"]["frames"][0]["start"]) * timing.RATE)
        edit = [(extent - window) // 2, (extent + window) // 2]
        picture_frames = [frame | {"sample": round(frame["pts"] * table.tb * timing.RATE) - origin,
                                   "gap": None if frame["gap_units"] is None else frame["gap_units"] / timing.SCORE_UNITS}
                          for frame in frames]
        row = {"source": item["index"], "frames": picture_frames, "samples": extent, "edit": edit}
        wide = width_row(row, [frame.get("width_units") for frame in frames])
        for role, pcm in study.parent.treatments(original, edit).items():
            groups, phones = read(pcm, include_events=True)
            aperture, width = ranked.predict(groups), ranked.predict(groups, weights)
            yield row, role, decide(row, wide, aperture, width, phones, references)
        print(f"Measured original frames and speech for {number}/{len(qualified)} actors in this panel.", file=sys.stderr, flush=True)


def native_transfer(args):
    result = review(args.receipt)
    weights = study.load(args.receipt)["width_fit"]["weights"]
    references = study.load(SPARSE)["result"]["references"]
    sys.path.insert(0, str(args.runtime.resolve()))
    if study.phone.sha(args.geometry) != study.parent.FACE_SHA256:
        raise ValueError("Face geometry differs from the reviewed asset")
    read, _ = dense.reader(args.checkpoint)
    panels = {}
    for name, remaining in (("first", False), ("remaining", True)):
        qualified = native.native.roster(args.audit, remaining=remaining)
        cases = native_deliveries(args, qualified, read, weights, references)
        outcomes = summarize(cases, result["points"])
        panels[name] = {"actors": len(qualified), "counts": outcomes["counts"],
                        "qualifications_sha256": study.phone.digest(json.dumps(qualified, sort_keys=True).encode())}
    record = {"schema": NATIVE_SCHEMA, "grid_receipt_sha256": study.phone.sha(args.receipt), "panels": panels}
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    pictures = phases.add_parser("pictures")
    for name in ("sources", "geometry", "runtime", "output"):
        pictures.add_argument(name, type=Path)
    carrier = phases.add_parser("carrier-sound")
    for name in ("sources", "checkpoint", "output"):
        carrier.add_argument(name, type=Path)
    compare = phases.add_parser("report")
    for name in ("widths", "primary_sound", "carrier_sound", "output"):
        compare.add_argument(name, type=Path)
    replay = phases.add_parser("review")
    replay.add_argument("receipt", type=Path)
    transfer = phases.add_parser("native-transfer")
    for name in ("receipt", "audit", "media", "checkpoint", "geometry", "runtime", "output"):
        transfer.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.phase == "pictures":
        capture_pictures(args)
    elif args.phase == "carrier-sound":
        capture_carrier(args)
    elif args.phase == "native-transfer":
        native_transfer(args)
    else:
        result = report(args) if args.phase == "report" else review(args.receipt)
        print(json.dumps({name: panel["counts"] for name, panel in result["panels"].items()}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
