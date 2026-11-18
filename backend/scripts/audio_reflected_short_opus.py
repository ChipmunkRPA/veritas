# SPDX-License-Identifier: MIT
"""Evaluate one mirrored short-voice view on Opus remixes with known speech edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from scripts.review_audio_two_second_opus_transfer import CODEC_POINTS, matches, outputs, read


SCHEMA = "chipmunk.short-reflection-opus/1"
RATE = 16_000
WINDOW = 12_000
STEP = 8_000
SOURCE_ROOTS = ("audio-matched-originals/matched-panel", "audio-six-ten-context-comparison/prospective",
                "audio-two-second-transfer/native-panel")
ROOM_PATHS = {
    "ES2004a": "audio-ami-native-stereo-v291/raw/ES2004a.Array1-01.wav",
    "IS1009a": "audio-ami-native-stereo-v291/raw/IS1009a.Array1-01.wav",
    "ES2015d": "audio-stream-presence-ami-native-2026-11-11/ES2015d.wav",
}
ROLES = {"calibration": ("ES2004a",), "held": ("IS1009a", "ES2015d")}
VIEWS = ("direct", "reflected")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def rank(value):
    return sha(f"{SCHEMA}/{value}".encode())


def write(path, data):
    from scripts.audio_two_second_opus_transfer import write as serialize

    serialize(path, data)


def short_starts(samples):
    return list(range(0, samples - WINDOW + 1, STEP))


def room_sources(annotations, rooms, reference_rows):
    from scripts.audio_harmonic_overlap_transfer import room_activity

    with ZipFile(annotations) as archive:
        document = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        participants = {item.get("observation"): {person.get("global_name") for person in item if person.get("global_name")}
                        for item in document.iter() if item.get("observation")}
        previous = {row["room"] for row in reference_rows}
        roster = set().union(*(participants[room] for room in previous))
        results = {}
        for name, relative in ROOM_PATHS.items():
            if not participants[name] or participants[name] & roster:
                raise ValueError(f"Room participants overlap in {name}")
            roster.update(participants[name])
            occupancy = room_activity(archive, name)
            with wave.open(str(rooms / relative)) as source:
                if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, RATE):
                    raise ValueError(f"Unexpected speech source format: {name}")
                local_frames = (rooms.joinpath(relative).stat().st_size - 1024) // 2
                stop = min(source.getnframes(), local_frames, len(occupancy) * 160) // 160
                blocks = {}
                for start in range(1900, stop - 1400, 350):
                    part = occupancy[start:start + 1400]
                    speech, overlap = int((part >= 1).sum()), int((part >= 2).sum())
                    if speech >= 1050 and overlap >= 380:
                        blocks[start] = [speech, overlap]
                available = []
                for start in sorted(blocks, key=lambda k: rank(f"{name}/{k}")):
                    if any(abs(start - item[0]) < 1400 for item in available):
                        continue
                    source.setpos(start * 160)
                    pcm = source.readframes(14 * RATE)
                    if len(pcm) != 28 * RATE:
                        raise ValueError("Incomplete room samples")
                    available.append((start, blocks[start], pcm))
                results[name] = available
    return results


def sources(root, original):
    earlier = {row["group"] for row in original["rows"]}
    chosen = {}
    commitments = {}
    for relative in SOURCE_ROOTS:
        directory = root / relative
        raw = (directory / "freeze.json").read_bytes()
        commitments[relative] = sha(raw)
        for row in json.loads(raw)["cases"]:
            group, kind = row["group"], row["kind"]
            if row["condition"] != "clean" or kind not in ("edited", "ordinary") or group in earlier:
                continue
            if not 9 * RATE <= row["samples"] <= 14 * RATE:
                continue
            chosen.setdefault((group, kind), (relative, row))
    edits = sorted((group for group, kind in chosen if kind == "edited"), key=rank)
    matched = {group for group in edits if (group, "ordinary") in chosen
               and chosen[group, "ordinary"][0] == SOURCE_ROOTS[0] == chosen[group, "edited"][0]}
    other = sorted((group for group, kind in chosen if kind == "ordinary" and group not in edits), key=rank)
    reserve = len(edits) - len(matched)
    roles = {"calibration": other[reserve:], "held": edits + other[:reserve]}
    if len(edits) != 8 or len(matched) != 3 or len(roles["calibration"]) < 6:
        raise ValueError("Incomplete source roster")
    return chosen, matched, commitments, roles


def freeze(root, rooms, annotations, reference, output):
    import numpy as np

    data = read(reference)
    chosen, matched, commitments, roles = sources(root, data)
    backgrounds = room_sources(annotations, rooms, data["rows"])
    cases = []
    directory = output / "pcm"
    directory.mkdir(parents=True, mode=0o700)

    def quantize(samples):
        return np.rint(samples).astype("<i2").tobytes()

    for role, groups in roles.items():
        room_ids = ROLES[role]
        for index, group in enumerate(groups):
            name = room_ids[index % len(room_ids)]
            block = index // len(room_ids)
            start, annotation, room_raw = backgrounds[name][block]
            originals, waveforms, labels = {}, {}, {}
            for kind in ("ordinary", "edited"):
                if (group, kind) not in chosen or (kind == "ordinary" and (group, "edited") in chosen and group not in matched):
                    continue
                relative, row = chosen[group, kind]
                original = (root / relative / "pcm" / f"{row['pcm_sha256']}.pcm").read_bytes()
                if sha(original) != row["pcm_sha256"] or len(original) != 2 * row["samples"]:
                    raise ValueError("Source signed PCM differs from frozen manifest")
                originals[kind] = {"manifest": relative, "case": row["id"], "pcm_sha256": row["pcm_sha256"]}
                waveforms[kind] = np.frombuffer(original, dtype="<i2").astype(np.float64)
                labels[kind] = (row["family"], row["truth"])
            sizes = {len(value) for value in waveforms.values()}
            if len(sizes) != 1:
                raise ValueError("Paired edits and originals need the same clock")
            size, = sizes
            clipping_safe = float(np.iinfo(np.int16).max)
            competing = np.frombuffer(room_raw, dtype="<i2").astype(np.float64)[:size]
            foreground = waveforms.get("ordinary") if "ordinary" in waveforms else waveforms["edited"]
            nominal = 10 ** (10 / 20)
            rms = lambda signal: float(np.sqrt(np.mean(signal ** 2)))
            competing *= nominal * rms(foreground) / rms(competing)
            mixed = {kind: samples + competing for kind, samples in waveforms.items()}
            gain = min(1., clipping_safe / max(np.max(np.abs(v)) for v in (*mixed.values(), competing)))
            if block == 0:
                mixed["background"] = competing
            for kind, signal in mixed.items():
                raw = quantize(signal * gain)
                digest = sha(raw)
                with (directory / f"{digest}.pcm").open("xb") as stream:
                    stream.write(raw)
                family, truth = labels.get(kind, (None, []))
                cases.append({"id": f"{group}/{name}/{kind}", "group": group, "role": role, "kind": kind,
                              "family": family, "truth": truth, "room": name, "room_start": start * 160,
                              "room_speech_and_overlap_10ms": annotation, "room_pcm_sha256": sha(room_raw),
                              "source": originals.get(kind), "foreground_competing_db": None if kind == "background"
                              else 20 * math.log10(rms(waveforms[kind]) / rms(competing)),
                              "mix_gain": gain, "samples": size, "pcm_sha256": digest})
    plan = {
        "schema": SCHEMA, "prior_sha256": sha(reference.read_bytes()),
        "annotations_sha256": sha(annotations.read_bytes()), "source_manifests": commitments, "room_paths": ROOM_PATHS,
        "implementation_sha256": sha(Path(__file__).read_bytes()), "cases": cases, "assets": data["plan"]["assets"],
        "window_samples": WINDOW, "step_samples": STEP, "speech_fraction": .5,
        "selection_rule": "Strict excess of the highest adjacent-pair minimum over all calibration controls.",
        "scope": "Serve only after later independent validation of any positive held signal.",
    }
    write(output / "freeze.json", plan)
    print(dict(Counter((r["role"], r["kind"]) for r in cases)), flush=True)


def graded(row, view):
    series = {item["start_sample"]: item[view] for item in row["short_scores"]}
    return {a: min(series[a], series[b]) for a, b in ((offset, offset + STEP) for offset in series) if b in series}


def intervals(row, view, point):
    supported = {offset + shift for offset, response in graded(row, view).items() if response > point for shift in (0, STEP)}
    return sorted((a, a + WINDOW) for a in supported)


def operating_points(rows):
    return {
        view: max(response for row in rows for response in graded(row, view).values())
        for view in VIEWS
    }


def summary(rows, points):
    totals = defaultdict(Counter)
    for row in rows:
        current, extra = outputs(row, CODEC_POINTS["aac"])
        choices = {"baseline": current, "baseline_and_two_second": current + extra}
        choices.update({view: current + extra + intervals(row, view, points[view]) for view in VIEWS})
        ref = matches(choices["baseline_and_two_second"], row["truth"])
        fitted = [item["start_sample"] for item in row["short_scores"]]
        geometry = matches([(a, a + WINDOW) for a in short_starts(row["samples"])], row["truth"])
        for view, values in choices.items():
            found = matches(values, row["truth"])
            counts = totals[view, row["kind"]]
            counts.update(cases=1, flagged=int(bool(values)), located=len(found), labels=len(row["truth"]),
                          short_labels=sum(b - a < 9600 for a, b in row["truth"]), useful_gain=len(found - ref),
                          useful_lost=len(ref - found), geometric_ceiling=len(geometry), short_forward_windows=len(fitted),
                          short_labels_found=sum(row["truth"][i][1] - row["truth"][i][0] < 9600 for i in found))
    return {"/".join(key): dict(value) for key, value in sorted(totals.items())}


def score(directory, output, role, calibration):
    import numpy as np
    import torch

    from app.core.canonical import canonical_json
    from app.detectors.audio.extended_short_voice_review import scan
    from app.detectors.audio.short_voice_review import voice_logit
    from app.engine.boundary import Deadline
    from app.evidence import audio_long_review_binding as long
    from scripts.audio_joint_generator_transfer import prior as incumbent
    from scripts.audio_radar_transfer import adapter_result
    from scripts.audio_two_second_opus_transfer import measurements
    from scripts.run_audio_specialists_v15 import load_voice

    plan = read(directory / "freeze.json")
    if plan["implementation_sha256"] != sha(Path(__file__).read_bytes()):
        raise ValueError("Runner differs from the frozen procedure")
    weights = Path("models/weights/audio-upgrade")
    voice = load_voice({"weights": weights, "assets": plan["assets"]})
    torch.set_num_threads(2)
    activity = incumbent.runtime(weights / "silero_vad.jit")
    selected = [row for row in plan["cases"] if row["role"] == role]
    if role == "held":
        base = read(calibration)
        if base["freeze_sha256"] != sha((directory / "freeze.json").read_bytes()):
            raise ValueError("Calibration and held score must have the same source freeze")
        points = operating_points(base["rows"])
    rows = []
    with tempfile.TemporaryDirectory(prefix="reflect-opus-") as temporary, output.with_suffix(".jsonl").open("x") as journal:
        path = Path(temporary) / "delivery.ogg"
        for item in selected:
            source = directory / "pcm" / f"{item['pcm_sha256']}.pcm"
            if sha(source.read_bytes()) != item["pcm_sha256"]:
                raise ValueError("Input sample commitment differs from the freeze")
            pcm, clock = measurements(path, item, source)
            spans = [(round(a * RATE), round(b * RATE)) for a, b in activity.spans(pcm)]
            scores, reading = [], {}
            times = Counter()
            for start in short_starts(len(pcm)):
                end = start + WINDOW
                speech = sum(max(0, min(end, b) - max(start, a)) for a, b in spans)
                if speech * 2 < WINDOW:
                    continue
                window = pcm[start:end]
                result = {"start_sample": start, "speech_samples": speech}
                for key, model_input in (("direct", window), ("reflected", np.concatenate((window, window[::-1])))):
                    began = time.perf_counter()
                    result[key] = voice_logit(voice, model_input)
                    times[key] += time.perf_counter() - began
                scores.append(result)
            if role == "held":
                began = time.perf_counter()
                reading["excerpts"] = json.loads(canonical_json(scan(pcm, activity, voice, Deadline(None))))
                reading["current_windows"] = [
                    {"start_sample": a, "end_sample": b,
                     "direction": adapter_result(pcm[a:b], voice, activity)["decision"]}
                    for a, b in (long.focused_intervals(len(pcm)) or long.intervals(len(pcm)))
                ]
                times["comparator"] = time.perf_counter() - began
            row = {**item, **clock, **reading, "short_scores": scores, "seconds": dict(times)}
            rows.append(row)
            journal.write(json.dumps(row, separators=(",", ":")) + "\n")
            journal.flush()
            path.unlink()
            print(role, len(rows), len(selected), item["kind"], len(scores), flush=True)
    if role == "calibration":
        points = operating_points(rows)
    result = {"freeze_sha256": sha((directory / "freeze.json").read_bytes()), "role": role, "rows": rows, "points": points}
    if role == "held":
        result["counts"] = summary(rows, points)
    write(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    for field in ("root", "rooms", "annotations", "reference", "output"):
        prepare.add_argument(field, type=Path)
    run = commands.add_parser("score")
    for field in ("directory", "output"):
        run.add_argument(field, type=Path)
    run.add_argument("role", choices=ROLES)
    run.add_argument("--calibration", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "freeze":
        freeze(**args)
    else:
        if args["role"] == "held" and args["calibration"] is None:
            parser.error("held scoring requires --calibration")
        score(**args)


if __name__ == "__main__":
    main()
