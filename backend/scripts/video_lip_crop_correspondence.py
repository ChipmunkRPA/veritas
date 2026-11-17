"""Test whether native mouth appearance adds speech-to-picture localization."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts import video_lip_correspondence as study

SCHEMA = "chipmunk.video-lip-crop-correspondence/1"
ARM = "shape_texture"
METHOD = {
    "shape": "Use the same recorded lip geometry and matched, original-only fit.",
    "appearance": "On the exact recorded mouth crop, measure a 3-by-4 grid of intensity relative to the outer crop "
                  "and three vertical chroma summaries.",
    "test": "Select a control budget on the separate calibration actors, then hold it fixed on PCM, AAC and the separate "
            "capture cohort. Keep actual AI authorship outside a measurement of speech-to-picture disagreement.",
}


def descriptor(pixels):
    import cv2
    import numpy as np

    if min(pixels.shape[:2]) < 12:
        return None
    rgb = cv2.resize(pixels, (20, 16), interpolation=cv2.INTER_AREA).astype(np.float64)
    red, green, blue = rgb.transpose(2, 0, 1)
    luma = .2126 * red + .7152 * green + .0722 * blue
    chroma = red - .5 * (green + blue)
    border = np.ones(luma.shape, dtype=bool)
    border[3:-3, 3:-3] = False
    level = np.median(luma[border])
    contrast = max(12., float(np.std(luma[border])))
    red_level = np.median(chroma[border])
    window = ((luma[2:-2, 2:-2] - level) / contrast).clip(-4., 4.)
    brightness = window.reshape(3, 4, 4, 4).mean(axis=(1, 3))
    color = ((chroma[2:-2, 2:-2] - red_level) / 24.).clip(-4., 4.).reshape(3, 4, 16).mean(axis=(1, 2))
    return np.concatenate((brightness.ravel(), color)).round(7).tolist()


def native_texture(path, record):
    from app.detectors.video.decode import decode_stream, probe_frames

    table = probe_frames(path, cap=120, timeout_s=18, max_pixels=1280 * 720)
    expected = record["frames"]
    if (table.truncated or table.interpolated or list(table.pts) != [row["pts"] for row in expected]
            or str(table.tb) != record["time_base"]):
        raise ValueError("Native picture clock does not reproduce the declared observations")
    rows = []
    with closing(decode_stream(path, table, max_side=None, timeout_s=20)) as frames:
        for picture in frames:
            note = dict(expected[len(rows)])
            if picture.index != note["index"] or not picture.pts_verified or picture.pts != note["pts"]:
                raise ValueError("Picture returned at a different source position")
            if "shape" in note:
                x, y, right, bottom = note["mouth_box"]
                crop = picture.rgb[y:bottom, x:right]
                if (hashlib.sha256(picture.rgb.tobytes()).hexdigest() != note["rgb_sha256"]
                        or hashlib.sha256(crop.tobytes()).hexdigest() != note["mouth_crop_sha256"]):
                    raise ValueError("Frozen source picture or its mouth crop changed")
                note["texture"] = descriptor(crop)
            rows.append(note)
    if len(rows) != len(expected):
        raise ValueError("Picture decode did not cover the original frames")
    return rows


def capture(capture, plan, sources, output):
    frozen = study.read(capture)
    if study.sha(plan) != frozen["plan_sha256"]:
        raise ValueError("Native capture uses a different source plan")
    specs = {row["target"]["front_sha256"]: row["target"] for row in json.loads(plan.read_bytes())["rows"]}
    added = []
    with ZipFile(sources / "lombardgrid_front.zip") as pictures, TemporaryDirectory(prefix="lip-texture-") as work:
        path = Path(work) / "source.mov"
        for row in frozen["rows"]:
            spec = specs[row["picture_sha256"]]
            payload = pictures.read(spec["front"])
            if hashlib.sha256(payload).hexdigest() != spec["front_sha256"]:
                raise ValueError("Publisher video bytes changed")
            path.write_bytes(payload)
            added.append(row | {"frames": native_texture(path, row)})
            if len(added) % 12 == 0:
                print(f"{frozen['phase']} {len(added)}/{len(frozen['rows'])} exact clips", flush=True)
    study.save(output, frozen | {"rows": added, "parent_capture_sha256": study.sha(capture), "texture_method": METHOD})


def transfer_capture(roster, stereo_receipt, sources, assets, output):
    import numpy as np

    from app.evidence.video_reference_timing import _landmarker
    from scripts import video_lip_articulation_transfer as prior
    from scripts.video_lip_delivery_transfer import qualified_48k
    from scripts.video_lip_stereo_transfer import verify as verify_stereo

    sources_plan = json.loads(roster.read_bytes())
    stereo = study.read(stereo_receipt)
    verify_stereo(stereo)
    assert stereo["freeze_sha256"] == study.sha(roster)
    clocks = {row["source"]: row for row in stereo["rows"] if row["split"] == "new_speaker"}
    sys.path.insert(0, str((assets / "runtime").resolve()))
    measure, device = prior.audio_reader(assets / "checkpoint")
    captured = []
    with (ZipFile(sources / "lombardgrid_front.zip") as pictures, ZipFile(sources / "lombardgrid_audio.zip") as audio,
          TemporaryDirectory(prefix="lip-texture-transfer-") as work, _landmarker(assets / "face_landmarker.task") as (mp, task)):
        source = Path(work) / "source.mov"
        carrier = Path(work) / "sound.m4a"
        for row in sources_plan["rows"]:
            if row["split"] != "new_speaker":
                continue
            target = row["target"]
            expected = clocks[target["front_sha256"]]
            video = pictures.read(target["front"])
            assert prior.phone.digest(video) == target["front_sha256"]
            source.write_bytes(video)

            def sound(item):
                payload = audio.read(item["audio"])
                assert prior.phone.digest(payload) == item["audio_sha256"]
                return prior.phone.pcm(payload)

            original = sound(row["target"])
            known = target["span"]
            versions = {role: value for role, value in prior.study.parent.treatments(original, known).items()
                        if role in {"original", "gain", "global_early", "global_late"}}
            for role in ("same_words", "different_words"):
                data = prior.parent.splice(original, known, sound(row[role]), row[role]["span"])
                a, b = known
                assert np.array_equal(data[:a], original[:a]) and np.array_equal(data[b:], original[b:])
                versions[role] = data
            assert set(versions) == study.ROLES.keys()
            observed = prior.parent.pictures(source, len(original), mp, task, with_shape=True)
            assert observed["time_base"] == expected["time_base"] and [frame["pts"] for frame in observed["frames"]] == expected["pts"]
            observed["frames"] = native_texture(source, observed)
            deliveries = {"pcm": {}, "aac48": {}}
            for role, data in versions.items():
                rendered, _, timing = qualified_48k(data, carrier)
                deliveries["pcm"][role] = measure(data)
                deliveries["aac48"][role] = measure(rendered) | {"clock": timing}
            captured.append({"speaker": row["speaker"], "condition": row["condition"], "split": "transfer", "truth": known,
                             "picture_sha256": target["front_sha256"], "source_audio_sha256": {
                                 role: row[role]["audio_sha256"] for role in ("target", "same_words", "different_words")},
                             **observed, "deliveries": deliveries})
            print(f"Checked source clocks and both audio deliveries for {len(captured)}/{len(clocks)} clips", flush=True)
    assert len(captured) == len(clocks) == 19
    study.save(output, {"schema": SCHEMA, "method": METHOD, "device": device, "phase": "transfer", "rows": captured,
                        "roster_sha256": study.sha(roster), "stereo_receipt_sha256": study.sha(stereo_receipt),
                        "plan_sha256": sources_plan["retained_plan_sha256"]})


def calibration(rows, arms):
    values = {}
    for arm in arms:
        maxima = dict.fromkeys({row["speaker"] for row in rows}, 0.)
        for row in rows:
            for role, readings in row["cases"].items():
                if study.ROLES[role] != "content" and readings["incumbent"] is None:
                    maxima[row["speaker"]] = max(maxima[row["speaker"]], readings[arm].get("score", 0.))
        scores = sorted(maxima.values(), reverse=True)
        values[arm] = {key: {"point": scores[n], "outcomes": study.totals(rows, arm, scores[n])}
                       for key, n in study.POINTS.items()}
    return values


def select(points):
    def order(option):
        arm, choice = option
        totals = points[arm][choice]["outcomes"]
        return (totals["content"]["confirmed"], -sum(totals[name]["added"] for name in ("ordinary", "same_word", "whole_delay")),
                totals["content"]["useful"], choice == "tight")

    arm, key = max(((arm, key) for arm, choices in points.items() for key in choices), key=order)
    return {"arm": arm, "point": points[arm][key]["point"], "level": key}


def held_counts(rows, points):
    return {"current": study.totals(rows, "incumbent", 0)} | {
        f"{arm}/{choice}": study.totals(rows, arm, result["point"])
        for arm, group in points.items() for choice, result in group.items()
    }


def fit(training, controls, comparator, frozen_shape, path):
    frozen = study.read(frozen_shape)
    genuine, calibration_set, previous = map(study.read, (training, controls, comparator))
    if (frozen["plan_sha256"] != genuine["plan_sha256"] or genuine["plan_sha256"] != calibration_set["plan_sha256"]
            or previous["plan_sha256"] != calibration_set["plan_sha256"]
            or (genuine["parent_capture_sha256"], calibration_set["parent_capture_sha256"]) != (
                previous["fit_sha256"], previous["calibration_sha256"])):
        raise ValueError("The texture measurements and earlier comparison cover different recordings")
    heads = {"shape": frozen["heads"]["shape"], ARM: study.fit(genuine["rows"], ARM)}
    rows = study.readouts(calibration_set["rows"], previous["calibration"], previous["points"], heads)
    points = calibration(rows, heads)
    output = {"schema": SCHEMA, "method": METHOD, "heads": heads, "points": points, "selected": select(points), "rows": rows,
              "fit_speakers": frozen["fit_speakers"], "plan_sha256": genuine["plan_sha256"],
              "inputs": {label: study.sha(p) for label, p in (("fit", training), ("control", controls), ("incumbent", comparator))}}
    study.save(path, output)
    print(json.dumps({"selected": output["selected"], "points": points}, indent=2))


def test(freeze, capture, incumbent, path, *, aac_capture=None):
    frozen, observed, prior = map(study.read, (freeze, capture, incumbent))
    assert observed["plan_sha256"] == prior["plan_sha256"] == frozen["plan_sha256"]
    if aac_capture is None:
        assert observed["parent_capture_sha256"] == prior["held_sha256"]
    rows = observed["rows"]
    if aac_capture is not None:
        delivered = study.read(aac_capture)
        assert delivered["plan_sha256"] == frozen["plan_sha256"] and study.sha(aac_capture) == prior["held_sha256"]
        clips = {row["picture_sha256"]: row for row in rows}
        for row in delivered["rows"]:
            bound = clips[row["picture_sha256"]]
            assert [{k: v for k, v in item.items() if k != "texture"} for item in bound["frames"]] == row["frames"]
            assert row["source_audio_sha256"] == bound["source_audio_sha256"] and row["samples"] == bound["samples"]
        rows = [row | {"frames": clips[row["picture_sha256"]]["frames"]} for row in delivered["rows"]]
    assert not {row["speaker"] for row in rows}.intersection(
        set(frozen["fit_speakers"]) | {row["speaker"] for row in frozen["rows"]})
    measurements = study.readouts(rows, prior["cohort"], prior["points"], frozen["heads"])
    counts = held_counts(measurements, frozen["points"])
    output = {"schema": SCHEMA, "method": METHOD, "freeze": frozen, "rows": measurements, "outcomes": counts,
              "capture_sha256": study.sha(capture), "prior_sha256": study.sha(incumbent)}
    if aac_capture is not None:
        output["aac_capture_sha256"] = study.sha(aac_capture)
    study.save(path, output)
    print(json.dumps(counts, indent=2))


def test_transfer(freeze, capture, comparator, path):
    from scripts import video_lip_articulation_transfer as prior

    frozen, observed, earlier = map(study.read, (freeze, capture, comparator))
    assert observed["phase"] == "transfer" and observed["plan_sha256"] == frozen["plan_sha256"] == earlier["plan_sha256"]
    assert observed["method"] == METHOD and frozen["selected"] == select(frozen["points"])
    assert study.sha(comparator) == frozen["inputs"]["incumbent"]
    groups = {row["speaker"] for row in observed["rows"]}
    assert not groups.intersection(set(frozen["fit_speakers"]) | {row["speaker"] for row in frozen["rows"]})
    results = {}
    for profile in ("pcm", "aac48"):
        rows = [item | {"deliveries": item["deliveries"][profile]} for item in observed["rows"]]
        incumbent = [prior.readout(row, earlier["models"]) for row in rows]
        readings = study.readouts(rows, incumbent, earlier["points"], frozen["heads"])
        for source, replay in zip(rows, readings, strict=True):
            if profile == "aac48":
                for role, recording in source["deliveries"].items():
                    replay["cases"][role]["clock"] = recording["clock"]
        results[profile] = {"rows": readings, "outcomes": held_counts(readings, frozen["points"])}
    report = {"schema": SCHEMA, "method": METHOD, "freeze": frozen, "transfer": results, "speakers": len(groups),
              "capture_sha256": study.sha(capture), "prior_sha256": study.sha(comparator),
              "roster_sha256": observed["roster_sha256"], "stereo_receipt_sha256": observed["stereo_receipt_sha256"]}
    study.save(path, report)
    print(json.dumps({kind: rows["outcomes"] for kind, rows in results.items()}, indent=2))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="action", required=True)
    for command, fields in (
        ("capture", ("capture", "plan", "sources", "output")),
        ("capture-transfer", ("roster", "stereo_receipt", "sources", "assets", "output")),
        ("fit", ("training", "controls", "comparator", "frozen_shape", "output")),
        ("assess", ("freeze", "capture", "incumbent", "output")),
        ("assess-transfer", ("freeze", "capture", "comparator", "output")),
        ("verify", ("receipt",)),
    ):
        parser = actions.add_parser(command)
        for name in fields:
            parser.add_argument(name, type=Path)
        if command == "assess":
            parser.add_argument("--aac-capture", type=Path)
    args = cli.parse_args()
    if args.action == "capture":
        capture(args.capture, args.plan, args.sources, args.output)
    elif args.action == "capture-transfer":
        transfer_capture(args.roster, args.stereo_receipt, args.sources, args.assets, args.output)
    elif args.action == "fit":
        fit(args.training, args.controls, args.comparator, args.frozen_shape, args.output)
    elif args.action == "assess":
        test(args.freeze, args.capture, args.incumbent, args.output, aac_capture=args.aac_capture)
    elif args.action == "assess-transfer":
        test_transfer(args.freeze, args.capture, args.comparator, args.output)
    else:
        record = study.read(args.receipt)
        assert record["schema"] == SCHEMA and record["method"] == METHOD
        frozen = record["freeze"]
        assert calibration(frozen["rows"], frozen["heads"]) == frozen["points"]
        assert select(frozen["points"]) == frozen["selected"]
        deliveries = record.get("transfer") or {"held": record}
        for delivery in deliveries.values():
            assert held_counts(delivery["rows"], frozen["points"]) == delivery["outcomes"]
        print(json.dumps({profile: delivery["outcomes"] for profile, delivery in deliveries.items()}, indent=2))


if __name__ == "__main__":
    main()
