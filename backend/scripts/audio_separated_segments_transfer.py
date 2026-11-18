# SPDX-License-Identifier: Apache-2.0
"""Confirmation of the frozen short speech detector on two voices in a different measured room."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_separated_segments as earlier
from scripts import audio_short_edit_training_sources as identifiers
from scripts import audio_two_voice_components as render
from scripts import audio_two_voice_isolation as parent
from scripts.audio_focused_context_transfer import matches

SCHEMA = "chipmunk.separated-speech-segments-transfer/1"
PANELS = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
DEVELOPMENT = PANELS / "audio-separated-segments-development-2026-11-17.json.gz"
DEVELOPMENT_SHA = "055841bb5f2a9f2530b50e4b4c15c57a8f1e164f6ea7c9c3aba528cf7b78919a"
NATIVE_SHA = "0f7dad649ce02661b5a17a5c5daaded438ca3e7d7a8ee4c52bd4dcc0b17e0c77"
HOLDOUT_SHA = "12926246246eeee0522191b8fc619c331e171167f52b15b5a28266a7cdbf26b4"
ANGULAR_SHA = "5ef67c11589b7b7c1068bbd8caf5741d65e40baca5786283bc5573e6487d5444"
VAD_SHA = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
CONDITIONS = render.CONDITIONS


def sha(value):
    with value.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rank(*keys):
    return hashlib.sha256("/".join((SCHEMA, *map(str, keys))).encode()).hexdigest()


def validation_points():
    if sha(DEVELOPMENT) != DEVELOPMENT_SHA:
        raise ValueError("The previously frozen development result differs")
    result = earlier.portable.read(DEVELOPMENT)
    if result["points"] != earlier.calibrated_points(result["rows"]) or result["input_pins"]["variant"] != "2048/plain":
        raise ValueError("Development rows do not reproduce the prescribed points")
    return result["points"], result["input_pins"]


def impulses(directory):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    from scripts import audio_room_assets_v299 as room

    asset_info = room.check_huddersfield(directory)["candidate_pairs"]
    outputs = {}
    for position, assets in room.HUDDERSFIELD_ASSETS.items():
        left, right = (sf.read(io.BytesIO(room.pinned_bytes(directory, asset)[0]), dtype="float64")[0] for asset in assets)
        stereo = np.column_stack((left, right))
        sample_rate = asset_info[position][0]["sample_rate_hz"]
        output = resample_poly(stereo, 160, sample_rate // 100, axis=0)
        if output.shape[1] != 2 or sample_rate != 44100 or not np.isfinite(output).all():
            raise ValueError("The admitted stereo filter or sample rate differs")
        outputs[position] = output
        asset_info[position] = {
            "source": asset_info[position], "filtered_sha256": earlier.portable.sha(output.astype("<f8").tobytes()),
            "native_rate": sample_rate, "rendered_rate": render.RATE, "rendered_samples": len(output),
        }
    return outputs, asset_info


def dry_cases(source):
    import numpy as np
    from scipy.signal import resample_poly

    identity = source / "freeze.json"
    if sha(identity) != NATIVE_SHA:
        raise ValueError("The local verified matched-original plan differs")
    plan = earlier.portable.read(identity)
    pairs = {pair["edit"]: pair for pair in plan["pairs"]}
    rows = {row["id"].split("/")[0]: row for row in plan["cases"] if row["condition"] == "clean"}
    prior = {voice["speaker"] for scene in parent.parent()["plan"]["scenes"] for voice in scene["voices"]}

    def read(row):
        name = row["pcm_sha256"]
        raw = (source / "pcm" / f"{name}.pcm").read_bytes()
        if len(raw) != 2 * row["samples"] or earlier.portable.sha(raw) != name:
            raise ValueError("The matched original or edit PCM differs")
        return np.frombuffer(raw, dtype="<i2")

    groups = defaultdict(list)
    for identity, pair in pairs.items():
        modified, ordinary = rows[identity], rows[pair["ordinary"]]
        speaker = identifiers.order(modified["group"])
        if speaker in prior or modified["group"] != ordinary["group"] or modified["samples"] != ordinary["samples"]:
            raise ValueError("A transfer pair reuses an earlier speaker or lacks a shared clock")
        control, edited = read(ordinary), read(modified)
        spans = [(a, b) for a, b in modified["truth"] if 3840 <= b - a <= 20000 and a >= 4800 and len(control) - b >= 4800]
        if not spans:
            raise ValueError("A reserved utterance lacks a complete edit at this duration")
        a, b = min(spans, key=lambda span: rank(speaker, span))
        crop = min(max(0, a - 40000), len(control) - render.SAMPLES)
        aa, bb = a - crop, b - crop
        original = control[crop:crop + render.SAMPLES].copy()
        samples = {"untouched": original}
        truth_signal = original[aa:bb].astype(np.float64)
        replacement = edited[a:b].astype(np.float64)
        filtered = resample_poly(resample_poly(truth_signal, 1, 2), 2, 1)[:b - a]
        fade = min(320, (b - a) // 4)
        weights = np.ones(b - a)
        weights[:fade] = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
        weights[-fade:] = weights[:fade][::-1]
        for kind, phrase in (("synthetic", replacement), ("genuine", filtered)):
            variant = original.copy()
            variant[aa:bb] = np.rint(truth_signal * (1 - weights) + phrase * weights).clip(-32768, 32767).astype("<i2")
            samples[kind] = variant
        groups[modified["family"]].append({
            "speaker": speaker, "group": rank("utterance", identity), "family": modified["family"],
            "source_files": [ordinary["pcm_sha256"], modified["pcm_sha256"]],
            "source_crop": [crop, crop + render.SAMPLES], "source_edit": [a, b], "original_span": [aa, bb],
            "variants": {name: earlier.portable.sha(values.tobytes()) for name, values in samples.items()},
            "control": "same speech 8 kHz round trip", "samples": samples,
        })
    if sum(map(len, groups.values())) != 14:
        raise ValueError("The original matched roster has a different size")
    for members in groups.values():
        members.sort(key=lambda value: value["group"], reverse=True)
    while any(groups.values()):
        choices = sorted((family for family in groups if groups[family]), key=lambda key: (-len(groups[key]), rank(key)))
        if len(choices) < 2:
            raise ValueError("Could not retain disjoint two-generator pairs")
        yield [groups[family].pop() for family in choices[:2]]


def prepare(source, room, output):
    import numpy as np

    points, model = validation_points()
    filters, filter_assets = impulses(room)
    clips = list(dry_cases(source))
    output.mkdir(mode=0o700)
    (output / "pcm").mkdir()

    def save(samples):
        raw = samples.astype("<i2").tobytes()
        digest = earlier.portable.sha(raw)
        file = output / "pcm" / f"{digest}.pcm"
        if not file.exists():
            file.write_bytes(raw)
        return digest

    rows = []
    for pair, voices in enumerate(clips):
        positions = sorted(filters, key=lambda key: rank(pair, key))[:2]
        for schedule, direction in zip(render.LOCATIONS, (positions, positions[::-1]), strict=True):
            descriptors, stems = render.render(voices, schedule, filters, direction)
            for control, (a, b) in CONDITIONS.items():
                first, second = stems[0][a], stems[1][b]
                quiet = np.rint(first.astype(np.float64) * earlier.QUIET_SCALE).astype("<i2")
                signal = quiet.astype(np.int32) + second.astype(np.int32)
                if np.max(np.abs(signal)) >= 32768:
                    raise ValueError("The new stereo mixture would clip")
                truths = [{"span": item["span"], "side": side, "family": item["family"]}
                          for side, (kind, item) in enumerate(zip((a, b), descriptors, strict=True)) if kind == "synthetic"]
                rows.append({"pair": pair, "schedule": schedule, "condition": control, "truth": truths, "cohort": "huddersfield",
                             "voices": descriptors, "input_sha256": save(signal), "stem_sha256": [save(quiet), save(second)]})
    plan = {"schema": SCHEMA, "source_sha256": NATIVE_SHA, "development_sha256": DEVELOPMENT_SHA, "points": points,
            "model": model, "rooms": filter_assets, "rows": rows,
            "gate": {"minimum_new_events": 2, "minimum_new_pairs": 2, "maximum_new_ordinary": 2}}
    earlier.portable.write(output / "freeze.json", plan)
    print(f"Froze {len(rows)} stereo inputs from {len(clips)} source-disjoint pairs", flush=True)


def decisions(rows, points):
    totals = {rule: Counter() for rule in ("mixture", "angular", "independent", "combined")}
    paired = Counter()
    pair_gains = set()
    for row in rows:
        truth = [item["span"] for item in row["truth"]]
        readings = {method: [lane["peaks"]["contrast"]["range"] for lane in lanes
                              if lane["peaks"]["contrast"] is not None and lane["peaks"]["contrast"]["score"] >
                              points[method]["contrast"]["strict"]] for method, lanes in row["measurements"].items()}
        readings["combined"] = readings["angular"] + readings["independent"]
        assessed = {method: any(lane["peaks"]["contrast"] is not None for lane in lanes)
                    for method, lanes in row["measurements"].items()}
        assessed["combined"] = assessed["angular"] or assessed["independent"]
        obtained = {}
        for rule, ranges in readings.items():
            areas = sorted(set(map(tuple, ranges)))
            hits = len(matches(areas, truth))
            obtained[rule] = hits
            count = totals[rule]
            count.update(edits=len(truth), correct=hits, ordinary=not truth, ordinary_flags=bool(areas and not truth),
                         misplaced_files=bool(truth and areas and not hits), assessed=assessed[rule])
        if obtained["combined"] > obtained["angular"]:
            paired["new_events"] += obtained["combined"] - obtained["angular"]
            pair_gains.add(row["pair"])
        if not truth and readings["combined"] and not readings["angular"]:
            paired["new_ordinary_files"] += 1
    paired["new_pairs"] = len(pair_gains)
    return {"policies": {rule: dict(count) for rule, count in totals.items()},
            "paired": {label: paired[label] for label in ("new_events", "new_ordinary_files", "new_pairs")}}


def measure(inputs, model, driver, output):
    import numpy as np

    from app.evidence.audio_short_review_binding import speech_counts
    from scripts import audio_spatial_mask_voice_transfer as spatial
    from scripts import audio_two_channel_separator as blind
    from scripts.audio_joint_generator_transfer import prior

    path = inputs / "freeze.json"
    vad = Path("models/weights/audio-upgrade/silero_vad.jit")
    plan = earlier.portable.read(path)
    points, model_pins = validation_points()
    if (sha(path) != HOLDOUT_SHA or plan["points"] != points or plan["model"] != model_pins
            or sha(Path(earlier.__file__)) != model_pins["reader_sha256"]
            or sha(Path(spatial.__file__)) != ANGULAR_SHA or sha(vad) != VAD_SHA
            or sha(model) != model_pins["model_state_sha256"] or sha(driver / "sal_smoke.py") != model_pins["sal_driver_sha256"]
            or sha(Path(blind.__file__)) != model_pins["separator_sha256"]):
        raise ValueError("The parameters or the frozen inference assets differ")
    driver = earlier.model_driver(model, driver)
    activity = prior.runtime(vad)
    rows = []
    for raw in plan["rows"]:
        samples = render.load_stereo(inputs, raw["input_sha256"])
        stems = [render.load_stereo(inputs, identity) for identity in raw["stem_sha256"]]
        if not np.array_equal(samples, stems[0].astype(np.int32) + stems[1].astype(np.int32)):
            raise ValueError("Retained voice tracks do not reproduce the original stereo input")
        angular, _ = spatial.separate(samples.astype(np.float64))
        independent = blind.separate(samples.astype(np.float64) / 32768., frame_size=2048, whiten=False)
        views = {"mixture": [samples.mean(axis=1) / 32768.],
                 "angular": [wave.astype(np.float64) / 32768. for wave in angular if wave is not None],
                 "independent": [] if independent is None else list(independent.T)}
        all_readings = {}
        for method, lanes in views.items():
            readings = []
            for lane in lanes:
                pcm = np.asarray(lane, dtype=np.float32)
                scores = driver.measure(pcm).tolist()
                steps = [(i * earlier.STEP, (i + 1) * earlier.STEP) for i in range(len(scores))]
                voiced = speech_counts(activity.spans(pcm), steps)
                readings.append({"peaks": earlier.peak_readings(scores, voiced),
                                 "sha256": earlier.portable.sha(pcm.astype("<f4").tobytes())})
            all_readings[method] = readings
        rows.append(raw | {"measurements": all_readings})
        if len(rows) % 5 == 0:
            print(f"Completed {len(rows)} of {len(plan['rows'])} held deliveries", flush=True)
    result = plan | {"plan_sha256": sha(path), "rows": rows, "outcome": decisions(rows, points)}
    earlier.portable.write(output, result)
    print(json.dumps(result["outcome"], sort_keys=True), flush=True)


def review(receipt):
    points, model = validation_points()
    result = earlier.portable.read(receipt)
    if (result["schema"] != SCHEMA or result["points"] != points or result["model"] != model
            or result["outcome"] != decisions(result["rows"], points)):
        raise ValueError("The confirmation does not reproduce")
    rows = [{key: value for key, value in row.items() if key != "measurements"} for row in result["rows"]]
    plan = {key: value for key, value in result.items() if key not in {"outcome", "plan_sha256"}} | {"rows": rows}
    plan_bytes = (json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if earlier.portable.sha(plan_bytes) != HOLDOUT_SHA or result["plan_sha256"] != HOLDOUT_SHA:
        raise ValueError("Measured rows differ from their original freeze")
    prior = {voice["speaker"] for scene in parent.parent()["plan"]["scenes"] for voice in scene["voices"]}
    identities = {voice["speaker"] for row in rows for voice in row["voices"]}
    if (len(rows) != 70 or len(identities) != 14 or len({row["input_sha256"] for row in rows}) != 70
            or identities & prior):
        raise ValueError("Confirmation speaker or file coverage changed")
    counts = Counter((row["pair"], row["schedule"], row["condition"]) for row in rows)
    if set(counts) != {(pair, schedule, kind) for pair in range(7) for schedule in render.LOCATIONS for kind in CONDITIONS}:
        raise ValueError("A confirmation edit condition or control is missing")
    for row in rows:
        states = CONDITIONS[row["condition"]]
        expected = []
        for side, (start, voice) in enumerate(zip(render.LOCATIONS[row["schedule"]], row["voices"], strict=True)):
            a, b = voice["original_span"]
            arrival = voice["direct_arrival"]
            if voice["shift"] != start - a or voice["span"] != [start + arrival, start + arrival + b - a]:
                raise ValueError("A published edit no longer uses the original declared clock")
            if states[side] == "synthetic":
                expected.append({"span": voice["span"], "side": side, "family": voice["family"]})
        if row["truth"] != expected:
            raise ValueError("A truth position differs from its prepared interval")
    print(json.dumps(result["outcome"], sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    steps = parser.add_subparsers(dest="phase", required=True)
    for function, names in ((prepare, ("source", "room", "output")), (measure, ("inputs", "model", "driver", "output")),
                            (review, ("receipt",))):
        sub = steps.add_parser(function.__name__)
        sub.set_defaults(call=function)
        for name in names:
            sub.add_argument(name, type=Path)
    options = vars(parser.parse_args())
    options.pop("phase")
    options.pop("call")(**options)
