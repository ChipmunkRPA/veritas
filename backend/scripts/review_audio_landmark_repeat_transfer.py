"""Audit frozen repeat spans against rebuilt copy geometry and delivery PCM."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RATE, SAMPLES = 16_000, 48_000
MIN_OVERLAP, MAX_LAG_ERROR = 2400, 160
CODECS = ("pcm", "opus")
ROLES = ("archived", "transfer")
COUNTS = {"archived": (17, 16, 236), "transfer": (16, 4, 200)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(data):
    return hashlib.sha256(data.read_bytes() if isinstance(data, Path) else data).hexdigest()


def parent_roster(parent, freeze):
    files = {name: parent / f"{name}.json" for name in ("freeze", "roster", "scores", "aggregate")}
    hashes = {f"{name}_sha256": digest(path) for name, path in files.items()}
    require(hashes == freeze["parent"], "Parent commitments differ")
    roster, plan, scores, outcome = (read(files[name]) for name in ("roster", "freeze", "scores", "aggregate"))
    require(plan["roster_sha256"] == hashes["roster_sha256"]
            and scores["freeze_sha256"] == outcome["freeze_sha256"] == hashes["freeze_sha256"]
            and outcome["scores_sha256"] == hashes["scores_sha256"], "Parent plan and scores do not match")
    require([(row["id"], row["mono_sha256"]) for row in roster] ==
            [(row["id"], row["mono_sha256"]) for row in scores["rows"]], "Parent input scores do not match roster")
    require([row["id"] for row in roster] == list(range(len(roster))), "Unexpected source roster order")
    return roster


def verify_split(roster, recipes):
    groups, rooms, hashes, families = {}, {}, {}, {}
    for role in ROLES:
        plan = recipes[role]
        clean = [row for row in roster if row["role"] == role and row["placement"] == "clean"]
        native = [row for row in roster if row["role"] == role and row["placement"] == "native"]
        require(plan["sources"] == [row["id"] for row in clean] and plan["native"] == [row["id"] for row in native],
                "Recipe omitted or repeated a frozen source")
        groups[role] = {row["group"] for row in clean}
        hashes[role] = {row["mono_sha256"] for row in clean + native}
        families[role] = dict(Counter(row["family"] for row in clean))
        rooms[role] = {row["room"] for row in native}
        require((len(groups[role]), len(native), 12 * len(clean) + 2 * len(native)) == COUNTS[role],
                "Incorrect source groups, room controls or planned view count")
        require(set(plan["scenes"]) == {"sequential", "simultaneous"}, "Incorrect room scenarios")
        for index, mode in enumerate(("sequential", "simultaneous")):
            choices = [row for row in native if row["mode"] == mode]
            room = sorted({row["room"] for row in choices})[index]
            require(plan["scenes"][mode] == next(row["id"] for row in choices if row["room"] == room),
                    "Room scenario differs from the frozen selection")
    require(not any(items["archived"] & items["transfer"] for items in (groups, rooms, hashes)),
            "Development and held source groups, exact bytes or rooms overlap")
    return families


def audio(parent, row):
    data = (parent / "mono" / f"{row['id']:03}.pcm").read_bytes()
    require(len(data) == SAMPLES * 2 and digest(data) == row["mono_sha256"], "Frozen source PCM differs")
    return np.frombuffer(data, dtype="<i2")


def materialize(source, duration):
    original = source.astype(np.float64)
    starts = tuple(range(1600, 9601, 1600))
    power = [np.square(original[start + 320:start + duration - 320]).sum() for start in starts]
    start, target = starts[int(np.argmax(power))], 28_800
    taper = .5 * (1 - np.cos(np.linspace(0, np.pi, 320)))
    weight = np.ones(duration)
    weight[:320], weight[-320:] = taper, taper[::-1]
    edited = original.copy()
    edited[target:target + duration] = (original[start:start + duration] * weight
                                         + original[target:target + duration] * (1 - weight))
    truth = [start + 320, start + duration - 320, target + 320, target + duration - 320]
    a, b, c, d = truth
    require(np.array_equal(edited[:target], original[:target]) and np.array_equal(edited[target + duration:], original[target + duration:])
            and np.array_equal(edited[c:d], original[a:b]), "Copy interior or exterior PCM differs")
    return original, edited, truth


def quantized(values):
    scaled = np.rint(values)
    require(scaled.min() >= -32768 and scaled.max() <= 32767, "PCM overflow")
    return scaled.astype("<i2").tobytes()


def room_mix(original, edited, room):
    room_gain = np.sqrt(np.square(original).sum() / np.square(room.astype(np.float64)).sum()) * 10 ** .25
    background = room * room_gain
    gain = min(1., 30_000 / np.max(np.abs(background) + np.maximum(np.abs(original), np.abs(edited))))
    return quantized((original + background) * gain), quantized((edited + background) * gain)


def opus_decode(data, scratch):
    wav, opus = (scratch / f"delivery.{suffix}" for suffix in ("wav", "opus"))
    with wave.open(str(wav), "wb") as stream:
        stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes(data)
    base = ["ffmpeg", "-nostdin", "-v", "error"]
    subprocess.run([*base, "-y", "-i", str(wav), "-ar", "48000", "-c:a", "libopus", "-b:a", "32k", str(opus)],
                   check=True, timeout=30)
    conversion = "aformat=sample_fmts=s16:channel_layouts=mono,aresample=16000,atrim=end_sample=48000,asetpts=N/SR/TB"
    rebuilt = subprocess.run([*base, "-copyts", "-i", str(opus), "-map", "0:a:0", "-af", conversion, "-vn",
                              "-t", "3", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"],
                             check=True, capture_output=True, timeout=30).stdout
    require(len(rebuilt) == SAMPLES * 2, "Opus delivery changed the sample count")
    return rebuilt


def cases(parent, roster, plan):
    with tempfile.TemporaryDirectory(prefix="chipmunk-landmark-review-") as scratch:
        def deliveries(source, scenario, raw, truth):
            yield source, scenario, "pcm", raw, truth
            yield source, scenario, "opus", opus_decode(raw, Path(scratch)), truth
        rooms = {mode: audio(parent, roster[row]) for mode, row in plan["scenes"].items()}
        for index, source in enumerate(plan["sources"]):
            original, edited, truth = materialize(audio(parent, roster[source]), 8000 if index % 2 else 10400)
            yield from deliveries(source, "clean", quantized(original), None)
            yield from deliveries(source, "clean", quantized(edited), truth)
            for mode, room in rooms.items():
                control, changed = room_mix(original, edited, room)
                yield from deliveries(source, mode, control, None)
                yield from deliveries(source, mode, changed, truth)
        for source in plan["native"]:
            yield from deliveries(source, "native", audio(parent, roster[source]).tobytes(), None)


def span_valid(span):
    return len(span) == 4 and all(type(value) is int for value in span) and 0 <= span[0] < span[1] <= span[2] < span[3] <= SAMPLES


def candidate_span(candidate):
    if candidate is None:
        return []
    score, start, stop, lag = candidate
    span = start, stop, start + lag, stop + lag
    require(span_valid(span) and type(score) is int and score > 0 and lag >= 38 * 80 and lag % 80 == start % 80 == 0
            and stop - start >= MIN_OVERLAP, "Candidate lies outside frozen sample or hop constraints")
    return [span]


def compare_geometry(span, truth):
    left = min(span[1], truth[1]) - max(span[0], truth[0])
    right = min(span[3], truth[3]) - max(span[2], truth[2])
    lag_error = abs((span[2] - span[0]) - (truth[2] - truth[0]))
    return max(0, left), max(0, right), lag_error


def localizes(geometry):
    return any(min(left, right) >= MIN_OVERLAP and error <= MAX_LAG_ERROR for left, right, error in geometry)


def verify_journal(journal, parent, roster, plan):
    keyed = {(row["source"], row["scenario"], row["codec"], row["truth"] is not None): row for row in journal}
    require(len(keyed) == len(journal), "Repeated journal case")
    counts = Counter()
    for source, scenario, codec, data, truth in cases(parent, roster, plan):
        key = source, scenario, codec, truth is not None
        row = keyed.pop(key, None)
        require(row is not None and row["truth"] == truth and row["sha256"] == digest(data),
                f"Missing or mismatched geometry or delivery at {key}")
        candidate_span(row["candidate"])
        require(all(map(span_valid, row["baseline"])), "Baseline contains an invalid interval")
        counts[codec] += 1
        counts["positive_truth" if truth else "controls"] += 1
    require(not keyed, "Unexpected journal cases")
    return dict(counts)


def minimum_from_development(rows, roster, plan):
    maxima = {roster[source]["group"]: 0 for source in plan["sources"]}
    for row in rows:
        if row["truth"] is None and row["scenario"] != "native" and row["candidate"]:
            group = roster[row["source"]]["group"]
            maxima[group] = max(maxima[group], row["candidate"][0])
    ordered = sorted(maxima.values(), reverse=True)
    return ordered[1] + 1, ordered


def held_outcomes(rows, roster, minimum):
    cells, added_groups, flagged_groups = defaultdict(Counter), set(), set()
    added_geometry = []
    for row in rows:
        truth = row["truth"]
        cell = cells[f"{row['scenario']}/{row['codec']}"]
        eligible = row["candidate"] is not None and row["candidate"][0] >= minimum
        incumbent = row["baseline"]
        group = roster[row["source"]]["group"]
        if truth is not None:
            geometry = [compare_geometry(span, truth) for span in candidate_span(row["candidate"])] if eligible else []
            current = localizes(compare_geometry(span, truth) for span in incumbent)
            hit = localizes(geometry)
            additional = hit and not current
            cell.update(edits=1, baseline_localized=current, candidate_localized=hit, additional=additional)
            if additional:
                added_groups.add(group)
                added_geometry.extend(geometry)
        else:
            extra = eligible and not incumbent
            cell.update(controls=1, baseline_flagged=bool(incumbent), candidate_flagged=eligible, extra_false=extra)
            if extra and row["scenario"] != "native":
                flagged_groups.add(group)
    added = sum(cell["additional"] for cell in cells.values())
    false = sum(cell["extra_false"] for cell in cells.values())
    native = sum(cell["candidate_flagged"] for name, cell in cells.items() if name.startswith("native/"))
    outcome = {"cells": cells, "extra_correct": added, "extra_false": false, "added_source_groups": len(added_groups),
               "false_source_groups": len(flagged_groups), "native_flags": native,
               "advance": added >= 8 and len(added_groups) >= 4 and false <= 2 and len(flagged_groups) <= 1 and native == 0}
    geometry = {"count": len(added_geometry),
                "minimum_overlap_samples": min((min(left, right) for left, right, _ in added_geometry), default=None),
                "maximum_lag_error_samples": max((error for _, _, error in added_geometry), default=None)}
    return outcome, geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "parent", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    freeze, points, aggregate = (read(args.run / f"{name}.json") for name in ("freeze", "points", "aggregate"))
    freeze_sha = digest(args.run / "freeze.json")
    require(freeze["code_sha256"] == digest(Path(__file__).with_name("audio_landmark_repeat_transfer.py"))
            and aggregate["schema"] == freeze["schema"] == "chipmunk.audio-landmark-repeat/1"
            and aggregate["method"] == freeze["method"] and aggregate["freeze_sha256"] == points["freeze_sha256"] == freeze_sha,
            "Freeze, candidate code or report method differs")
    expected = {role: digest(args.run / f"{role}.json") for role in ROLES}
    require(points["development_sha256"] == aggregate["fit_sha256"] == expected["archived"]
            and aggregate["held_sha256"] == expected["transfer"] and aggregate["point_sha256"] == digest(args.run / "points.json"),
            "Freeze, points or journals lack exact binding")
    roster = parent_roster(args.parent, freeze)
    families = verify_split(roster, freeze["recipes"])
    journals, counts = {}, {}
    for role in ROLES:
        journal = read(args.run / f"{role}.json")
        require(journal["freeze_sha256"] == freeze_sha and len(journal["cases"]) == COUNTS[role][2], "Unexpected journal scope")
        journals[role] = journal["cases"]
        counts[role] = verify_journal(journal["cases"], args.parent, roster, freeze["recipes"][role])
        print(role, counts[role], flush=True)
    minimum, maxima = minimum_from_development(journals["archived"], roster, freeze["recipes"]["archived"])
    require(aggregate["minimum"] == points["minimum"] == minimum and aggregate["views"] == [236, 200],
            "Development cutoff or totals differ")
    outcomes, geometry = held_outcomes(journals["transfer"], roster, minimum)
    require(all(aggregate[key] == value for key, value in outcomes.items()), "Held outcome arithmetic differs")
    report = {"status": "verified", "aggregate_sha256": digest(args.run / "aggregate.json"), "rebuilt_deliveries": counts,
              "foreground_source_families": families, "minimum": minimum, "development_control_source_maxima": maxima,
              "additional_location_geometry": geometry, **outcomes}
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print({key: report[key] for key in ("minimum", "extra_correct", "extra_false", "advance")}, flush=True)


if __name__ == "__main__":
    main()
