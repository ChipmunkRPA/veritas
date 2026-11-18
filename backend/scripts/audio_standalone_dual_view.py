# SPDX-License-Identifier: Apache-2.0
"""Evaluate shared short speech-edit intervals on raw and enhanced standalone audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from collections import Counter, defaultdict
from pathlib import Path

from scripts.video_stereo_covariance_panel import file_hash, read, save

SCHEMA = "chipmunk.standalone-dual-view/1"
BASELINE_RECEIPT = (Path(__file__).resolve().parents[2] / "docs" / "evaluation-panels"
                    / "audio-short-edit-duration-serving-2026-11-15.json")
PLAN_SHA = "725f2a5643fbfeb836a77dbaf8a99e9955ec39bdd1dafff73bf47f506496d59a"
SOURCE_SHA = "858f3210701cf5cc6aaa3b01100756342cbd790543e786affab1f4b1e9e05507"
RATE = 16000
SAMPLES = RATE * 6
HOP = RATE // 100
FLANK = 30
GAP = 5
WIDTHS = (30, 45, 65, 85, 105)
STEP = 5
LEVELS = (-5, 3)
SCALE = 1_000_000
METHODS = ("raw", "enhanced", "agreement")
ORDINARY = ("original", "genuine_splice")
HELD_METHODS = ("baseline", *METHODS, "fallback")
MODEL_SHA = "432bb64ef1f2e16a052d1d9ae36089bf4910ca35c2d6958ec92b83a38e285ce8"
ENHANCER_SHA = "a630d992cf792daf4ce2bb5bcf9c4d389f740a8f09c6e0971184697fe6371b79"
VAD_SHA = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"


class Deadline:
    def check(self, stage):
        pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_plan(plan, sources):
    frozen, selected = read(plan), read(sources)
    if (file_hash(plan) != PLAN_SHA or file_hash(sources) != SOURCE_SHA
            or selected["recipe_sha256"] != PLAN_SHA):
        raise ValueError("The pre-measurement method and source selection must match exactly")
    foreground = {row["reader"] for row in selected["pairs"]}
    if (len(foreground) != len(selected["pairs"]) or foreground.intersection(selected["known_readers"])
            or Counter(row["split"] for row in selected["pairs"]) != {"calibration": 12, "held": 16}
            or len({row["pcm_sha256"] for row in selected["backgrounds"]}) != 8):
        raise ValueError("Foreground readers and room excerpts must be distinct within their planned roles")
    return frozen, selected


def restore(plan, sources, output, caches):
    """Restore the already reserved source members from verified publisher prefixes."""
    from scripts.audio_sine_resynthesis_transfer import shard_members, source_listing

    _, selected = verify_plan(plan, sources)
    pins = {row["shard"]: row for row in selected["source_prefixes"]}
    grouped = defaultdict(list)
    for source in selected["pairs"]:
        grouped[source["shard"]].append(source)
    output.mkdir(mode=0o700)
    clips = output / "sine"
    clips.mkdir(mode=0o700)
    for number, rows in sorted(grouped.items()):
        cache = next((folder for folder in caches if (folder / f"split-{number}-192.range").is_file()), None)
        if cache is None:
            raise ValueError(f"Missing the reserved SINE prefix {number}")
        expected = source_listing(cache)[f"data/split-{number}.tar.gz"]
        wavs, labels, measured = shard_members(cache, number, expected)
        if measured != pins[number]:
            raise ValueError("A cached archive prefix differs from the exact source reservation")
        for row in rows:
            stem = row["stem"]
            if labels["medium_cut_paste.txt"][f"dev_cut_paste_medium-{stem}"] != (row["publisher_cut_paste_clock"], "0"):
                raise ValueError("The published edit clock differs from the retained study source")
            for name, kind in (("original", "real"), ("variant", "cut_paste")):
                data = wavs[kind][stem]
                if digest(data) != row["files"][name]["wav_sha256"]:
                    raise ValueError("A retained reader delivery differs from its exact byte commitment")
                (clips / f"{row['id']:03}-{name}.wav").write_bytes(data)


def crop(path, facts, start):
    import numpy as np
    from scipy.signal import resample_poly

    if file_hash(path) != facts["wav_sha256"]:
        raise ValueError("A native SINE file differs from the initially acquired content")
    with wave.open(str(path), "rb") as handle:
        expected = (1, 2, 24000, facts["samples"], "NONE")
        if handle.getparams()[:5] != expected or digest(handle.readframes(facts["samples"])) != facts["pcm_sha256"]:
            raise ValueError("The publisher PCM differs from its underlying commitment")
        handle.setpos(start)
        samples = np.frombuffer(handle.readframes(144000), dtype="<i2")
    if len(samples) != 144000:
        raise ValueError("The publisher crop does not contain the required six seconds")
    return np.rint(resample_poly(samples.astype(float), 2, 3)).clip(-32768, 32767)


def ordinary_splice(source, bounds):
    import numpy as np

    a, b = bounds
    width = b - a
    target_power = np.mean(source[a:b] ** 2)
    positions = [index for index in range(0, len(source) - width + 1, 3200)
                 if index + width + 3200 <= a or index >= b + 3200]
    donor = min(positions, key=lambda index: (abs(np.mean(source[index:index + width] ** 2) - target_power), index))
    copy = source.copy()
    fade = 400
    mix = np.ones(width)
    mix[:fade] = np.linspace(0, 1, fade, endpoint=True)
    mix[-fade:] = mix[:fade][::-1]
    copy[a:b] = source[a:b] * (1 - mix) + source[donor:donor + width] * mix
    return copy, [donor, donor + width]


def samples_for(split, root, plan, rooms):
    import numpy as np

    room_slice = slice(0, 4) if split == "calibration" else slice(4, 8)
    backgrounds = plan["backgrounds"][room_slice]
    ambient = {}
    for row in backgrounds:
        name = row["room"]
        data = (rooms / f"{name}.pcm").read_bytes()
        if digest(data) != row["pcm_sha256"] or len(data) != 2 * SAMPLES:
            raise ValueError("A competing-speaker input differs from its earlier recording commitment")
        ambient[name] = np.frombuffer(data, dtype="<i2")
        yield {"split": split, "room": name, "kind": "room_only", "reader": None, "level": None, "truth": None}, ambient[name]
    for position, row in enumerate(entry for entry in plan["pairs"] if entry["split"] == split):
        clip = {name: crop(root / "sine" / f'{row["id"]:03}-{name}.wav', facts, row["native_start"])
                for name, facts in row["files"].items()}
        ordinary, donor = ordinary_splice(clip["original"], row["truth"])
        variants = {"original": clip["original"], "genuine_splice": ordinary, "edit": clip["variant"]}
        room = backgrounds[position % len(backgrounds)]["room"]
        background = ambient[room].astype(float)
        ratio = np.linalg.norm(clip["original"]) / np.linalg.norm(background)
        for level in LEVELS:
            noise = background * ratio * 10 ** (-level / 20)
            mixed = {key: values + noise for key, values in variants.items()}
            gain = min(1.0, 27_000 / max(float(np.max(np.abs(values))) for values in mixed.values()))
            for kind, values in mixed.items():
                if split == "calibration" and kind == "edit":
                    continue
                metadata = {"split": split, "id": row["id"], "reader": row["reader"], "room": room,
                            "level": level, "kind": kind, "truth": row["truth"] if kind == "edit" else None,
                            "paired_clock": row["truth"]}
                if kind == "genuine_splice":
                    metadata["genuine_donor"] = donor
                yield metadata, np.rint(values * gain).astype("<i2")


def enhanced_pcm(source, model):
    import numpy as np

    from scripts.audio_noise_view_v4 import enhance

    overlap = RATE * 2
    left, right = enhance(source[:2 * overlap], model), enhance(source[overlap:], model)
    ramp = np.sin(np.linspace(0, np.pi / 2, overlap, dtype=np.float32)) ** 2
    result = np.concatenate((left[:overlap], left[overlap:] * (1 - ramp) + right[:overlap] * ramp, right[overlap:]))
    if result.shape != source.shape or not np.isfinite(result).all():
        raise ValueError("The enhanced reading lost original sample positions")
    return result


def choose(pcm, enhanced, spans, model):
    import numpy as np

    centers = np.arange(SAMPLES // HOP) * HOP
    speech = np.zeros(centers.size, dtype=float)
    for start, end in spans:
        speech += np.clip(np.minimum(centers + HOP, end * RATE) - np.maximum(centers, start * RATE), 0, HOP)
    prefix = np.r_[0, np.cumsum(np.minimum(HOP, speech))]
    eligible = []
    for width in WIDTHS:
        starts = np.arange(FLANK + GAP, len(centers) - width - FLANK - GAP + 1, STEP)
        ends = starts + width
        inside = 2 * (prefix[ends] - prefix[starts]) >= HOP * width
        before = 2 * (prefix[starts - GAP] - prefix[starts - GAP - FLANK]) >= HOP * FLANK
        after = 2 * (prefix[ends + GAP + FLANK] - prefix[ends + GAP]) >= HOP * FLANK
        valid = inside & before & after
        eligible.extend(zip(starts[valid].tolist(), ends[valid].tolist(), strict=True))
    results = dict.fromkeys(METHODS)
    if not eligible:
        return {"assessed": False, "windows": 0, "proposals": results}
    windows = np.array(eligible)
    a, b = windows.T
    scores = []
    for input_view in (pcm, enhanced):
        local, edge = model.predict(input_view).astype(np.float64)
        local, edge = np.r_[0, np.cumsum(local)], np.r_[0, np.cumsum(edge)]
        segment = (local[b] - local[a]) / (b - a)
        left = (local[a - GAP] - local[a - GAP - FLANK]) / FLANK
        right = (local[b + GAP + FLANK] - local[b + GAP]) / FLANK
        edges = np.minimum(edge[a + 3] - edge[a - 2], edge[b + 3] - edge[b - 2]) / 5
        scores.append(np.rint((segment - np.maximum(left, right) + .25 * edges) * SCALE).astype(np.int64))
    for name, values in zip(METHODS, (*scores, np.minimum(*scores)), strict=True):
        i = min(range(len(windows)), key=lambda j: (-values[j], eligible[j]))
        results[name] = {"span": [int(a[i]) * HOP, int(b[i]) * HOP], "score_units": int(values[i]),
                         "view_score_units": [int(raw[i]) for raw in scores]}
    return {"assessed": True, "windows": len(eligible), "proposals": results}


def calibrate(rows):
    points = {}
    for method in METHODS:
        maxima = defaultdict(list)
        ambient = []
        for row in rows:
            if proposal := row["proposals"][method]:
                if row["kind"] == "room_only":
                    ambient.append(proposal["score_units"])
                else:
                    maxima[row["reader"]].append(proposal["score_units"])
        if len(maxima) < 8:
            raise ValueError("Too few comparable calibration voices to freeze this operating point")
        ordered = sorted(max(items) for items in maxima.values())
        point = max(ordered[math.ceil(.95 * len(ordered)) - 1], max(ambient, default=-math.inf))
        points[method] = {"score_units": point, "assessed_readers": len(ordered), "assessed_rooms": len(ambient)}
    return points


def candidate(plan, sources, root, rooms, weights, enhancer, output):
    import numpy as np
    import torch

    from app.detectors.audio.focused_edit_review import MODEL_ID, load_model
    from scripts import audio_noise_view_v4 as reduction
    from scripts.audio_short_edit_duration_serving import ROOM
    _, materials = verify_plan(plan, sources)
    if file_hash(weights / "silero_vad.jit") != VAD_SHA:
        raise ValueError("The original voice admission must use the pinned activity helper")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = load_model(MODEL_ID, "cpu", "float32")
    filter_model = reduction.load(enhancer)
    activity = ROOM.runtime(weights / "silero_vad.jit")
    output.mkdir(mode=0o700)
    readings = {}
    for split in ("calibration", "held"):
        captured = []
        with (output / f"{split}.jsonl").open("x") as stream:
            for index, (case, pcm) in enumerate(samples_for(split, root, materials, rooms), 1):
                native = pcm.astype(np.float32) / 32768
                filtered = enhanced_pcm(native, filter_model)
                row = case | {"input_sha256": digest(pcm.tobytes()), "enhanced_sha256": digest(filtered.tobytes())}
                row.update(choose(native, filtered, activity.spans(native), model.obj))
                captured.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                if index % 10 == 0:
                    print(f"Measured {index} {split} files", flush=True)
        readings[split] = captured
        if split == "calibration":
            save(output / "points.json", calibrate(captured))
    save(output / "candidate.json.gz", {"model_sha256": model.revision, "enhancer_sha256": reduction.PINS, "vad_sha256": VAD_SHA,
                                       "points": calibrate(readings["calibration"]), "rows": readings})


def baseline(plan, sources, root, rooms, weights, output):
    import numpy as np

    from app.detectors.audio import short_voice_review
    from scripts.audio_short_edit_duration_serving import ROOM
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    _, materials = verify_plan(plan, sources)
    activity = ROOM.runtime(weights / "silero_vad.jit")
    model = load_voice({"weights": weights, "assets": ASSETS})
    rows = []
    with output.with_suffix(".jsonl").open("x") as log:
        for index, (case, pcm) in enumerate(samples_for("held", root, materials, rooms), 1):
            waveform = pcm.astype(np.float32) / 32768
            result = short_voice_review.scan(waveform, activity.spans(waveform), model, Deadline(), codec="pcm_s16le")
            chosen = result["selected"]
            row = case | {"input_sha256": digest(pcm.tobytes()), "assessed": bool(result["responses"]),
                          "span": [chosen["start_sample"], chosen["end_sample"]] if chosen else None}
            rows.append(row)
            log.write(json.dumps(row, sort_keys=True) + "\n")
            log.flush()
            if index % 10 == 0:
                print(f"Measured {index} held baseline files", flush=True)
    save(output, {"assets": ASSETS, "rows": rows})


def useful(span, truth):
    if not span or not truth:
        return False
    shared = max(0, min(span[1], truth[1]) - max(span[0], truth[0]))
    return (2 * shared >= truth[1] - truth[0]
            and 10 * shared >= 3 * (span[1] - span[0] + truth[1] - truth[0] - shared))


def outcomes(measurements, baseline_rows, points):
    lookup = {}
    for row, incumbent in zip(measurements, baseline_rows, strict=True):
        fields = ("split", "reader", "room", "kind", "level", "truth", "input_sha256")
        if any(row[key] != incumbent[key] for key in fields):
            raise ValueError("Two mechanisms did not read precisely the same submitted PCM")
        kind = row["kind"]
        key = (row["reader"], row["level"], kind) if kind != "room_only" else (row["room"], None, kind)
        if key in lookup:
            raise ValueError("Each recorded case needs one result per mechanism")
        spans = {"baseline": incumbent["span"]}
        for name in METHODS:
            proposal = row["proposals"][name]
            spans[name] = proposal["span"] if proposal and proposal["score_units"] > points[name]["score_units"] else None
        spans["fallback"] = spans["baseline"] or spans["agreement"]
        lookup[key] = row, incumbent, spans
    summary = defaultdict(Counter)
    improvements = defaultdict(Counter)
    gains, ordinary, extra_room_flags = defaultdict(set), defaultdict(set), set()
    for (group, level, kind), (row, baseline_reading, spans) in lookup.items():
        label = f"{level}/{kind}" if level is not None else kind
        for method in HELD_METHODS:
            span, truth = spans[method], row["truth"]
            is_useful = useful(span, truth)
            collision = is_useful and any(useful(lookup[group, level, case][2][method], truth) for case in ORDINARY)
            coverage = {"baseline": baseline_reading["assessed"],
                        "fallback": baseline_reading["assessed"] or row["assessed"]}.get(method, row["assessed"])
            summary[f"{method}/{label}"].update(total=1, flagged=span is not None, useful=is_useful,
                                                  assessed=coverage, same_clock_control=collision,
                                                  without_control=is_useful and not collision)
        baseline_span, fallback = (spans[name] for name in ("baseline", "fallback"))
        if kind == "edit":
            truth = row["truth"]
            good = useful(fallback, truth)
            base_good = useful(baseline_span, truth)
            clear = good and not any(useful(lookup[group, level, case][2]["fallback"], truth) for case in ORDINARY)
            improvements[level]["gained_without_control"] += clear and not base_good
            improvements[level]["lost_useful"] += base_good and not good
            improvements[level]["more_misplaced"] += fallback is not None and not good and baseline_span is None
            if clear and not base_good:
                gains[level].add(group)
        elif baseline_span is None and fallback:
            if kind == "room_only":
                extra_room_flags.add(group)
            else:
                improvements[level]["more_ordinary"] += 1
                ordinary[level].add(group)
    for level in LEVELS:
        improvements[level]["gained_reader_groups"] = len(gains[level])
        improvements[level]["more_ordinary_reader_groups"] = len(ordinary[level])
    low, high = (improvements[snr] for snr in LEVELS)
    gate = (low["gained_without_control"] >= 3 and low["gained_reader_groups"] >= 3
            and low["more_ordinary"] <= 2 and low["more_ordinary_reader_groups"] <= 2 and not extra_room_flags
            and high["lost_useful"] <= 1 and all(improvements[level]["more_misplaced"] <= 2 for level in LEVELS))
    return {"summary": dict(summary), "fallback_deltas": {str(level): dict(row) for level, row in improvements.items()},
            "fallback_new_room_flags": len(extra_room_flags), "accept": gate}


def validate_cases(rows, materials, split, *, current=False):
    rosters = {item["id"]: item for item in materials["pairs"] if item["split"] == split}
    scene_positions = slice(0, 4) if split == "calibration" else slice(4, 8)
    rooms = [item["room"] for item in materials["backgrounds"][scene_positions]]
    expected = {(room, None, "room_only") for room in rooms}
    for source in rosters.values():
        for level in LEVELS:
            for kind in ORDINARY + (("edit",) if split == "held" else ()):
                expected.add((source["id"], level, kind))
    for row in rows:
        kind, level = row["kind"], row["level"]
        source = None if kind == "room_only" else rosters[row["id"]]
        key = (row["room"], None, kind) if source is None else (source["id"], level, kind)
        if key not in expected or row["split"] != split or not _valid_hash(row["input_sha256"]):
            raise ValueError("A result differs from the exact prepared case inventory")
        expected.remove(key)
        if source is not None:
            if (row["reader"] != source["reader"] or row["paired_clock"] != source["truth"]
                    or row["truth"] != (source["truth"] if kind == "edit" else None)
                    or row["room"] != rooms[(source["id"] - min(rosters)) % len(rooms)]):
                raise ValueError("A case lost its declared mixture, source group or exact target clock")
        elif any(row[key] is not None for key in ("reader", "truth", "level")):
            raise ValueError("An isolated room must have no foreground or edit truth")
        if current:
            from app.evidence.audio_short_review_binding import GRID

            if row["span"] and (not row["assessed"] or tuple(row["span"]) not in GRID):
                raise ValueError("The incumbent range differs from its existing two-second grid")
            continue
        if (set(row["proposals"]) != set(METHODS) or not _valid_hash(row["enhanced_sha256"])
                or bool(row["windows"]) != row["assessed"]):
            raise ValueError("A response lost its original or enhanced sample commitment")
        for name, proposal in row["proposals"].items():
            if bool(proposal) != row["assessed"]:
                raise ValueError("The readings need matching speech-window admission across both views")
            if proposal is None:
                continue
            a, b = proposal["span"]
            first, second = proposal["view_score_units"]
            points = {"raw": first, "enhanced": second, "agreement": min(first, second)}
            if ((b - a) not in [width * HOP for width in WIDTHS] or a % (STEP * HOP)
                    or not (FLANK + GAP) * HOP <= a < b <= SAMPLES - (FLANK + GAP) * HOP
                    or proposal["score_units"] != points[name]):
                raise ValueError("A reading lost its comparable timing window or shared-view score")
    if expected:
        raise ValueError("Missing prepared comparison results")


def _valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def checked(candidate_result, current, materials):
    if (candidate_result["model_sha256"] != MODEL_SHA or candidate_result["vad_sha256"] != VAD_SHA
            or candidate_result["enhancer_sha256"]["checkpoints/model_trained_on_dns3.tar"] != ENHANCER_SHA
            or current["assets"] != read(BASELINE_RECEIPT)["assets"]):
        raise ValueError("The candidate or incumbent assets differ from the declared local comparison")
    records = candidate_result["rows"]
    for role in ("calibration", "held"):
        validate_cases(records[role], materials, role)
    validate_cases(current["rows"], materials, "held", current=True)
    points = calibrate(records["calibration"])
    if candidate_result["points"] != points:
        raise ValueError("The recorded operating points do not derive from the independent calibration roles")
    from app.evidence.audio_short_review_binding import GRID

    metrics = outcomes(records["held"], current["rows"], points)
    proposed = [(a * HOP, (a + duration) * HOP) for duration in WIDTHS
                for a in range(FLANK + GAP, SAMPLES // HOP - duration - FLANK - GAP + 1, STEP)]
    diagnostics = {}
    for level in LEVELS:
        edits = [row for row in records["held"] if row["kind"] == "edit" and row["level"] == level]
        diagnostics[str(level)] = {
            "geometry_can_fit_baseline": sum(any(useful(span, row["truth"]) for span in GRID) for row in edits),
            "geometry_can_fit_narrow": sum(any(useful(span, row["truth"]) for span in proposed) for row in edits),
            "top_proposals_that_overlap": {name: sum(useful((row["proposals"][name] or {}).get("span"), row["truth"])
                                                      for row in edits) for name in METHODS},
        }
    return metrics | {"localization_diagnostics": diagnostics}


def combine(plan, sources, newer, current, output):
    _, materials = verify_plan(plan, sources)
    candidate_result, incumbent = read(newer), read(current)
    metrics = checked(candidate_result, incumbent, materials)
    save(output, {"schema": SCHEMA, "plan_sha256": PLAN_SHA, "sources_sha256": SOURCE_SHA,
                  "candidate": candidate_result, "incumbent": incumbent, "outcome": metrics})
    print(json.dumps(metrics, sort_keys=True), flush=True)


def review(plan, sources, receipt):
    _, materials = verify_plan(plan, sources)
    study = read(receipt)
    if (study["schema"] != SCHEMA or study["plan_sha256"] != PLAN_SHA or study["sources_sha256"] != SOURCE_SHA):
        raise ValueError("The published result differs from the locked study")
    metrics = checked(study["candidate"], study["incumbent"], materials)
    if study["outcome"] != metrics:
        raise ValueError("Reported results do not match the paired response commitments")
    print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="mode", required=True)
    stages = {candidate: ("root", "rooms", "weights", "output", "enhancer"),
              baseline: ("root", "rooms", "weights", "output"),
              combine: ("newer", "current", "output"), review: ("receipt",)}
    for function, fields in stages.items():
        command = commands.add_parser(function.__name__)
        for field in ("plan", "sources", *fields):
            command.add_argument(field, type=Path)
        command.set_defaults(action=function)
    command = commands.add_parser("restore")
    for name in ("plan", "sources", "output"):
        command.add_argument(name, type=Path)
    command.add_argument("caches", nargs="+", type=Path)
    command.set_defaults(action=restore)
    args = vars(cli.parse_args())
    action = args.pop("action")
    args.pop("mode")
    action(**args)
