# SPDX-License-Identifier: Apache-2.0
"""Test a temporary spectral floor change on one submitted noisy recording."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts import audio_standalone_dual_view as previous

PANELS = Path(__file__).resolve().parents[2] / "docs" / "evaluation-panels"
PLAN = PANELS / "audio-local-floor-return-plan-2026-11-18.json"
PLAN_SHA = "e1ba781f0dc91ae63ad58df54d04620fe33246cbe2f2946b41dde5435c8e672c"
PREFIX = "audio-standalone-dual-view"
PRIOR = PANELS / f"{PREFIX}-2026-11-18.json.gz"
PRIOR_SHA = "da1b3fc09b32a72ad958e3040f3fff5f6ffb6a46d26da2bae5859881b4a8ab42"
SCHEMA = "chipmunk.audio-local-floor-return/1"
FFT = 640
HOP = 160
SUPPORT = 30
GAP = 4
STEP = 4
WIDTHS = (40, 52, 68, 84, 104, 128, 156)
BANDS = (140, 220, 320, 450, 620, 820, 1050, 1350, 1700, 2100, 2600, 3150, 3800, 4550, 5400, 6400, 7600)
METHODS = {"lower_quintile": .2, "median": .5}
ORDINARY = (*previous.ORDINARY, "local_gain", "local_eq")
SCALE = 1_000_000


def identity(row):
    return row.get("id"), row["level"], row["kind"], row["room"]


def inputs():
    if (previous.file_hash(PLAN), previous.file_hash(PRIOR)) != (PLAN_SHA, PRIOR_SHA):
        raise ValueError("An inherited input receipt or prospective plan changed")
    source_plan, source_roles = (PANELS / f"{PREFIX}-{role}-2026-11-18.json" for role in ("plan", "sources"))
    _, bank = previous.verify_plan(source_plan, source_roles)
    exact = previous.read(PRIOR)
    calibration = {identity(row): row for row in exact["candidate"]["rows"]["calibration"]}
    held = {identity(row): row for row in exact["incumbent"]["rows"]}
    if (len(calibration), len(held)) != (52, 100):
        raise ValueError("The previous exact observations do not cover the reserved panel")
    return bank, {"calibration": calibration, "held": held}


def measure(samples):
    import numpy as np

    pcm = samples.astype(float) / 32768
    if pcm.shape != (previous.SAMPLES,) or not np.isfinite(pcm).all():
        raise ValueError("This test requires a bounded six-second mono input")
    frames = np.lib.stride_tricks.sliding_window_view(pcm, FFT)[::HOP]
    freq = np.fft.rfftfreq(FFT, 1 / previous.RATE)
    power = np.abs(np.fft.rfft(frames * np.hanning(FFT), axis=1)) ** 2
    bands = np.stack([power[:, (freq >= a) & (freq < b)].sum(axis=1) for a, b in zip(BANDS[:-1], BANDS[1:])], axis=1)
    logs = 10 * np.log10(np.maximum(bands, 1e-15))
    frame_power = np.mean(frames * frames, axis=1)
    rolling = np.lib.stride_tricks.sliding_window_view(logs, SUPPORT, axis=0)
    active = np.quantile(np.lib.stride_tricks.sliding_window_view(frame_power, SUPPORT), .2, axis=1) > 1e-6
    first_center = FFT // (2 * HOP)
    starts = range(first_center + SUPPORT + GAP, first_center + len(frames) - SUPPORT - GAP, STEP)
    windows = [(a, a + width) for a in starts for width in WIDTHS if a + width + GAP - first_center < len(rolling)]
    pairs = np.asarray(windows)
    a, b = pairs.T - first_center
    before, inner_start = a - SUPPORT - GAP, a + GAP
    inner_end, after = b - SUPPORT - GAP, b + GAP
    admissible = active[before] & active[inner_start] & active[inner_end] & active[after]
    if not np.any(admissible):
        return 0, dict.fromkeys(METHODS)
    before, inner_start, inner_end, after = (indices[admissible] for indices in (before, inner_start, inner_end, after))
    pairs = pairs[admissible]

    def distance(left, right):
        differences = np.minimum(np.abs(left - right), 20)
        return np.sqrt(np.mean(differences * differences, axis=1))

    result = {}
    for label, quantile in METHODS.items():
        profile = np.quantile(rolling, quantile, axis=2)
        profile -= np.median(profile, axis=1, keepdims=True)
        x, y, z, w = (profile[key] for key in (before, inner_start, inner_end, after))
        distances = np.stack([distance(x, y), distance(z, w), distance(x, w), distance(y, z)], axis=1)
        parts = np.rint(distances * SCALE).astype(np.int64)
        scores = np.minimum(parts[:, 0], parts[:, 1]) - np.maximum(parts[:, 2], parts[:, 3])
        index = int(np.argmax(scores))  # pairs are in increasing start, then increasing end order.
        result[label] = {"span": [int(value) * HOP for value in pairs[index]], "score_micro_db": int(scores[index]),
                         "distance_micro_db": parts[index].tolist()}
    return len(pairs), result


def expanded_cases(split, native, rooms, bank, exact):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    for meta, pcm in previous.samples_for(split, native, bank, rooms):
        expected = exact[identity(meta)]
        if previous.digest(pcm.tobytes()) != expected["input_sha256"]:
            raise ValueError("An inherited delivery does not match its exact PCM hash")
        yield meta, pcm

    split_ids = {"calibration": slice(0, 4), "held": slice(4, 8)}[split]
    background_rows = bank["backgrounds"][split_ids]
    values = [(rooms / f"{row['room']}.pcm").read_bytes() for row in background_rows]
    if any(previous.digest(value) != row["pcm_sha256"] for value, row in zip(values, background_rows, strict=True)):
        raise ValueError("An isolated room changed")
    room_tracks = [np.frombuffer(value, dtype="<i2").astype(float) for value in values]
    lowpass = butter(4, 2400, "lowpass", fs=previous.RATE, output="sos")
    for index, row in enumerate(item for item in bank["pairs"] if item["split"] == split):
        local = {name: previous.crop(native / "sine" / f'{row["id"]:03}-{name}.wav', facts, row["native_start"])
                 for name, facts in row["files"].items()}
        ordinary, _ = previous.ordinary_splice(local["original"], row["truth"])
        a, b = row["truth"]
        weights = np.zeros(previous.SAMPLES)
        weights[a:b] = 1
        fade = min(640, (b - a) // 2)
        weights[a:a + fade] = np.linspace(0, 1, fade)
        weights[b - fade:b] = weights[a:a + fade][::-1]
        room_index = index % len(room_tracks)
        donor_index = (room_index + 1) % len(room_tracks)
        room, donor = room_tracks[room_index], room_tracks[donor_index]
        foreground = local["original"]
        room_scale = np.linalg.norm(foreground) / np.linalg.norm(room)
        donor_scale = np.linalg.norm(foreground) / np.linalg.norm(donor)
        for level in previous.LEVELS:
            scale = 10 ** (-level / 20)
            original = foreground + room * room_scale * scale
            variants = (original, ordinary + room * room_scale * scale, local["variant"] + room * room_scale * scale)
            gain = min(1., 27_000 / max(float(np.max(np.abs(values))) for values in variants))
            matches = background_rows[room_index]["room"]
            reference = exact[(row["id"], level, "original", matches)]
            if previous.digest(np.rint(original * gain).astype("<i2").tobytes()) != reference["input_sha256"]:
                raise ValueError("Derived controls must share the original mix and gain")
            lowered = original * (1 + weights * (10 ** (-3 / 20) - 1))
            filtered = original + weights * (sosfiltfilt(lowpass, original) - original)
            extra = {"local_gain": (lowered, None, None), "local_eq": (filtered, None, None)}
            if split == "held":
                replacement = original + weights * (donor * donor_scale - room * room_scale) * scale
                extra["room_replacement"] = replacement, row["truth"], background_rows[donor_index]["room"]
            for kind, (signal, truth, donor_room) in extra.items():
                meta = {"split": split, "id": row["id"], "reader": row["reader"], "level": level, "room": matches,
                        "kind": kind, "donor_room": donor_room, "truth": truth, "paired_clock": row["truth"]}
                yield meta, np.rint(signal * gain).clip(-32768, 32767).astype("<i2")


def readings(split, native, rooms, bank, exact):
    measured = []
    for meta, pcm in expanded_cases(split, native, rooms, bank, exact):
        count, candidates = measure(pcm)
        measured.append(meta | {"input_sha256": previous.digest(pcm.tobytes()), "intervals_assessed": count,
                                "proposals": candidates})
    return measured


def verify_rows(rows, split, bank, exact):
    rooms = bank["backgrounds"][:4] if split == "calibration" else bank["backgrounds"][4:]
    readers = [row for row in bank["pairs"] if row["split"] == split]
    expected = {key: (row["truth"], row["reader"], row["input_sha256"], None) for key, row in exact.items()}
    added_roles = ("local_gain", "local_eq") if split == "calibration" else ("local_gain", "local_eq", "room_replacement")
    for index, reader in enumerate(readers):
        for level in previous.LEVELS:
            for role in added_roles:
                key = reader["id"], level, role, rooms[index % 4]["room"]
                replacement = role == "room_replacement"
                expected[key] = (reader["truth"] if replacement else None, reader["reader"], None,
                                 rooms[(index + 1) % 4]["room"] if replacement else None)
    for row in rows:
        truth, name, digest, donor = expected.pop(identity(row))
        if (row["split"] != split or (row["truth"], row["reader"], row.get("donor_room")) != (truth, name, donor)
                or (digest and row["input_sha256"] != digest) or row["proposals"].keys() != METHODS.keys()):
            raise ValueError("Wrong source role, split, label, or exact PCM digest")
        for proposal in row["proposals"].values():
            if (proposal is None) != (not row["intervals_assessed"]):
                raise ValueError("Proposal does not match measured coverage")
            if proposal:
                a, b = proposal["span"]
                n, m, x, y = proposal["distance_micro_db"]
                if (not 0 <= a < b <= previous.SAMPLES or (a - 36 * HOP) % (STEP * HOP)
                        or b - a not in tuple(width * HOP for width in WIDTHS)
                        or min(n, m, x, y) < 0 or max(n, m, x, y) > 20 * SCALE
                        or proposal["score_micro_db"] != min(n, m) - max(x, y)):
                    raise ValueError("A retained score disagrees with its distances or source span")
    if expected:
        raise ValueError("A planned source and carrier is missing")


def points(rows):
    return {method: max(proposal["score_micro_db"] for row in rows if (proposal := row["proposals"][method]))
            for method in METHODS}


def useful(span, truth):
    if not span or not truth:
        return False
    a, b = span
    x, y = truth
    overlap = max(0, min(b, y) - max(a, x))
    return 2 * overlap >= y - x and 10 * overlap >= 3 * (max(b, y) - min(a, x))


def outcomes(rows, baseline, cutoffs):
    def observed(row, method):
        prior = baseline.get(identity(row), {})
        old = prior.get("span")
        if method == "incumbent":
            return old, old is not None, prior.get("assessed", False)
        name = method.removesuffix("_fallback")
        choice = row["proposals"][name]
        new = choice["span"] if choice and choice["score_micro_db"] > cutoffs[name] else None
        if method.endswith("_fallback"):
            return old or new, bool(old or new), prior.get("assessed", False) or bool(choice)
        return new, bool(new), bool(choice)

    summary = {}
    for method in ("incumbent", *METHODS, *(f"{name}_fallback" for name in METHODS)):
        observed_spans = {identity(row): observed(row, method) for row in rows}
        tallies = {}
        donor_rooms = set()
        for row in rows:
            if method == "incumbent" and identity(row) not in baseline:
                continue
            span, flag, scored = observed_spans[identity(row)]
            top = row["proposals"].get(method.removesuffix("_fallback"))
            found = useful(span, row["truth"])
            collision = found and row["kind"] == "edit" and any(
                useful(observed_spans[identity(other)][0], row["truth"]) for other in rows if (
                    other["kind"] in ORDINARY and (row["id"], row["level"]) == (other["id"], other["level"])
                    or other["kind"] == "room_only" and row["room"] == other["room"]))
            key = f'{row["level"]}/{row["kind"]}'
            tally = tallies.setdefault(key, Counter())
            tally.update(total=1, assessed=scored, flagged=flag, located=found, located_without_control=found and not collision,
                         wrong_range=flag and row["truth"] is not None and not found,
                         top_before_cutoff=bool(top and useful(top["span"], row["truth"])))
            if row["kind"] == "room_replacement" and found and row["level"] == -5:
                donor_rooms.add(row["donor_room"])
        summary[method] = {"tallies": tallies, "located_donor_rooms_at_minus_5_db": len(donor_rooms)}

    incumbent = summary["incumbent"]["tallies"]
    for method in METHODS:
        candidate = summary[method]["tallies"]
        fallback = summary[f"{method}_fallback"]["tallies"]
        summary[method]["background_gate"] = (candidate["-5/room_replacement"]["located"] >= 8
            and summary[method]["located_donor_rooms_at_minus_5_db"] >= 3 and not candidate["None/room_only"]["flagged"]
            and all(sum(candidate[f"{level}/{role}"]["flagged"] for role in ORDINARY) <= 2 for level in previous.LEVELS))
        summary[method]["speech_fallback_gate"] = (
            fallback["-5/edit"]["located_without_control"] >= incumbent["-5/edit"]["located_without_control"] + 3
            and fallback["None/room_only"]["flagged"] == incumbent["None/room_only"]["flagged"]
            and all(sum(fallback[f"{level}/{role}"]["flagged"] - incumbent[f"{level}/{role}"]["flagged"]
                        for role in previous.ORDINARY) <= 2 for level in previous.LEVELS))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("calibrate", "held", "review"):
        command = commands.add_parser(mode)
        command.add_argument("freeze", type=Path)
        if mode != "review":
            command.add_argument("sources", type=Path)
            command.add_argument("rooms", type=Path)
        if mode != "calibrate":
            command.add_argument("receipt", type=Path)
    args = parser.parse_args()
    bank, exact = inputs()
    checks = {"schema": SCHEMA, "plan_sha256": PLAN_SHA, "previous_receipt_sha256": PRIOR_SHA,
              "implementation_sha256": previous.file_hash(Path(__file__))}
    if args.mode == "calibrate":
        if args.freeze.exists():
            parser.error("Use a new calibration output path")
        rows = readings("calibration", args.sources, args.rooms, bank, exact["calibration"])
        previous.save(args.freeze, checks | {"rows": rows, "points": points(rows)})
    freeze = previous.read(args.freeze)
    verify_rows(freeze["rows"], "calibration", bank, exact["calibration"])
    if any(freeze[key] != value for key, value in checks.items()) or points(freeze["rows"]) != freeze["points"]:
        raise ValueError("The calibration does not match its frozen procedure or retained scores")
    if args.mode == "calibrate":
        print(json.dumps(freeze["points"], sort_keys=True))
        return
    held_checks = checks | {"calibration_sha256": previous.file_hash(args.freeze)}
    if args.mode == "held":
        if args.receipt.exists():
            parser.error("Use a new held output path")
        rows = readings("held", args.sources, args.rooms, bank, exact["held"])
        verify_rows(rows, "held", bank, exact["held"])
        result = outcomes(rows, exact["held"], freeze["points"])
        previous.save(args.receipt, held_checks | {"rows": rows, "outcomes": result})
    else:
        receipt = previous.read(args.receipt)
        verify_rows(receipt["rows"], "held", bank, exact["held"])
        result = outcomes(receipt["rows"], exact["held"], freeze["points"])
        if result != receipt["outcomes"] or any(receipt[key] != value for key, value in held_checks.items()):
            raise ValueError("The receipt differs from its frozen procedure or recomputed outcomes")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
