# SPDX-License-Identifier: Apache-2.0
"""Compare brief spectral transients as evidence of paired speech-edit boundaries."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_staggered_sine_transfer as transfer

io = transfer.io
clock = transfer.previous.baseline.clock
SCHEMA = "chipmunk.audio-transient-edges-transfer/1"
DEVELOPMENT = io.PUBLIC / "audio-sine-resynthesis-transfer-2026-11-14.json"
HELD = io.PUBLIC / "audio-narrow-flag-transfer-2026-11-14.json"
CURRENT = io.PUBLIC / "audio-staggered-sine-2026-11-14.json"
RATE = 16000
SAMPLES = 6 * RATE
HOP = 160
FRAME = 640
GUARD = 3
CONTEXT = 12
LIMIT = 4 * HOP
SPANS = (16, 120)
SCENES = transfer.SCENES
ARMS = ("edge_band", "broadband")
BANDS = ((0, 90), (2200, 3600), (3600, 5500), (5500, 7900), (6800, 7900))
ROLES = ("original", "gain", "eq", "ordinary_splice", "cut_paste")
METHOD = {
    "frames": {"samples": FRAME, "hop": HOP, "bands_hz": BANDS, "guard_hops": GUARD, "flank_hops": CONTEXT},
    "edge_band": "Larger local transient excess in the below-90 Hz and uppermost measured bands.",
    "broadband": "Smallest local transient excess across three bands between 2200 and 7900 Hz.",
    "peak": "Hann-window measured band level over the louder 80th percentile flank, floored at -84 dBFS. "
            "Require two endpoints 160 to 1200 ms apart. The weaker endpoint sets the pair score.",
    "mixing": "Original and publisher variant retain their earlier exact samples. Independently reduce gain on a constructed "
              "control only when needed to prevent clipping. No clips or labels are provided to the two measurements.",
    "decision": "One strict control-maximum point per arm, pooled across scenes. Fit on distinct development readers. "
                "Advance an editing observation only with at least three useful held edits across at least two noisy scenes, "
                "at most one flagged no-local-edit source group and zero flagged isolated backgrounds. "
                "A located ordinary splice counts as an editing observation, and rules out treating that result as AI evidence.",
}


def contract():
    dev, held = (io.read(path)["freeze"] for path in (DEVELOPMENT, HELD))
    dev_readers = {row["reader"] for row in dev["pairs"] if row["role"] in {"cut_paste", "resyn"}}
    if dev_readers & {row["reader"] for row in held["pairs"]}:
        raise ValueError("A SINE reader appears in both stages")
    cases = []
    for stage, bank in (("development", dev), ("held", held)):
        for pair in bank["pairs"]:
            if pair["role"] == "cut_paste":
                case_roles = ROLES
            elif pair["role"] == "resyn" and stage == "development":
                case_roles = ("resyn",)
            else:
                continue
            for scene in SCENES:
                for role in case_roles:
                    edit = pair["truth"] if role in {"cut_paste", "ordinary_splice"} else None
                    cases.append({"stage": stage, "id": pair["id"], "reader": pair["reader"],
                                  "scene": scene, "role": role, "truth": edit})
    for scene, count in (("rain", 4), ("train", 4), ("competing_speech", len(held["backgrounds"]))):
        for slot in range(count):
            cases.append({"stage": "held", "id": slot, "reader": None, "scene": scene, "role": "isolated", "truth": None})
    return {"schema": SCHEMA, "method": json.loads(json.dumps(METHOD)), "cases": cases,
            "source_sha256": {file.name: io.sha(file) for file in (DEVELOPMENT, HELD, CURRENT)}}


def peaks(pcm):
    import numpy as np

    window = np.hanning(FRAME)
    frame = np.lib.stride_tricks.sliding_window_view(pcm.astype(np.float64) / 32768, FRAME)[::HOP]
    power = abs(np.fft.rfft(frame * window, axis=1)) ** 2 * 2 / (FRAME * np.dot(window, window))
    hz = np.fft.rfftfreq(FRAME, 1 / RATE)
    energy = np.stack([power[:, (hz >= low) & (hz < high)].sum(axis=1) for low, high in BANDS], axis=1)
    levels = 10 * np.log10(np.maximum(1e-12, energy))
    flank = np.quantile(np.lib.stride_tricks.sliding_window_view(levels, CONTEXT, axis=0), .80, axis=-1)
    ticks = np.arange(CONTEXT + GUARD, len(frame) - CONTEXT - GUARD + 1)
    surrounding = np.maximum(-84., np.maximum(flank[ticks - CONTEXT - GUARD], flank[ticks + GUARD - 1]))
    excess = levels[ticks] - surrounding
    return (ticks + FRAME // (2 * HOP)) * HOP, np.max(excess[:, [0, 4]], axis=1), np.min(excess[:, 1:4], axis=1)


def strongest(ticks, score):
    import numpy as np

    best = None
    for gap in range(SPANS[0], SPANS[1] + 1):
        ratings = np.minimum(score[:-gap], score[gap:])
        first = int(np.argmax(ratings))
        start, end = int(ticks[first]), int(ticks[first + gap])
        support = [round(float(score[i]), 6) for i in (first, first + gap)]
        candidate = {"range": [start, end], "db": support}
        if best is None or rank(candidate) > rank(best):
            best = candidate
    return best


def rank(proposal):
    return min(proposal["db"]), -proposal["range"][0], -proposal["range"][1]


def pair_bank(folder, freeze):
    from scripts.audio_sine_resynthesis_transfer import native_rows

    if io.read(folder / "freeze.json") != freeze:
        raise ValueError("Retained SINE bank differs from the public source manifest")
    dry, ambient = {}, {}
    for row, samples in native_rows(folder, freeze):
        if row["condition"] == "clean":
            dry[row["id"], row["side"]] = samples
        elif row["role"] == "background_only":
            ambient[row["side"]] = samples
    return dry, [ambient[row["room"]] for row in freeze["backgrounds"]]


def controls(original, truth):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    source = original.astype(np.float64)
    start, end = truth
    size = end - start
    donor_start = SAMPLES * 3 // 4 if end < SAMPLES // 2 else RATE // 8
    donor = source[donor_start:donor_start + size].copy()
    segment = source[start:end]
    donor *= min(2., np.linalg.norm(segment) / max(1., np.linalg.norm(donor)))
    fade = 2 * HOP
    weight = np.minimum(1, np.minimum(np.arange(size), np.arange(size)[::-1]) / fade)
    transformed = {"gain": .58 * segment, "ordinary_splice": donor}
    output = {"original": original, "eq": quantize(sosfiltfilt(butter(2, 2800, fs=RATE, output="sos"), source))}
    for role, replacement in transformed.items():
        changed = source.copy()
        changed[start:end] = (1 - weight) * segment + weight * replacement
        output[role] = quantize(changed)
    return output


def quantize(values):
    import numpy as np

    return np.clip(np.rint(values), -32768, 32767).astype("<i2")


def mix_group(signals, background):
    import numpy as np

    room = background.astype(np.float64)
    room *= 10 ** .25 * np.linalg.norm(signals["original"].astype(np.float64)) / np.linalg.norm(room)
    publisher_edits = {"cut_paste", "resyn"}
    maximum = max(float(np.max(np.abs(values + room))) for role, values in signals.items()
                  if role == "original" or role in publisher_edits)
    scale = min(1., 32000 / maximum)
    mixed = {}
    for key, values in signals.items():
        peak = float(np.max(np.abs(values + room)))
        cutoff = min(scale, 32000 / peak) if peak > 32766 / scale else scale
        mixed[key] = np.rint(values * cutoff) + np.rint(room * cutoff)
    return {key: values.astype("<i2") for key, values in mixed.items()}


def samples_for(cases, dev_dir, held_dir, sounds):
    from scripts.audio_reference_variable_window import read_noise

    backgrounds = read_noise(sounds)
    sources, environments = {}, {}
    for stage, folder, receipt in (("development", dev_dir, DEVELOPMENT), ("held", held_dir, HELD)):
        freeze = io.read(receipt)["freeze"]
        clean, competing = pair_bank(folder, freeze)
        sources[stage] = clean, {row["id"]: row for row in freeze["pairs"]}
        environments[stage] = backgrounds | {"competing_speech": competing}
    last, variants = None, {}
    for row in cases:
        stage, index, scene, role = (row[key] for key in ("stage", "id", "scene", "role"))
        ambient = environments[stage].get(scene)
        if role == "isolated":
            yield ambient[index]
            continue
        key = stage, index, scene
        if key != last:
            clean, manifest = sources[stage]
            original, alternate = (clean[index, side] for side in ("original", "variant"))
            truth = manifest[index]["truth"]
            variants = (controls(original, truth) | {"cut_paste": alternate} if truth
                        else {"original": original, "resyn": alternate})
            if ambient is not None:
                variants = mix_group(variants, ambient[index % len(ambient)])
            last = key
        yield variants[role]


def assess(freeze, rows):
    if freeze != contract():
        raise ValueError("Source manifest or measurement design changed")
    prior = io.read(CURRENT)
    current = {(row["id"], row["scene"], row["side"]): row for row in prior["rows"]}
    roles = {"cut_paste": "variant", "original": "original", "isolated": "isolated"}
    previous_cells = defaultdict(Counter)
    for facts, row in zip(freeze["cases"], rows, strict=True):
        if row["case"] != facts or not clock.SHA.fullmatch(row["pcm_sha256"]):
            raise ValueError("Unexpected source identity")
        role = roles.get(facts["role"])
        if facts["stage"] == "held" and role:
            original = current[facts["id"], facts["scene"], role]
            if original["pcm_sha256"] != row["pcm_sha256"]:
                raise ValueError("Held PCM differs from the previous voice check")
            spans = transfer.current({tuple(item["samples"]): item["direction"] for item in original["windows"]})
            cell = previous_cells[f"{facts['scene']}/{role}"]
            truth = [facts["truth"]] if facts["truth"] else []
            cell.update(files=1, flags=bool(spans), useful=clock.useful(spans, truth))
        if row["status"] != "measured" or set(row["peaks"]) != set(ARMS):
            raise ValueError("Missing measurement arm")
        for proposal in row["peaks"].values():
            start, end = proposal["range"]
            if (not (CONTEXT + GUARD + 2) * HOP <= start < end <= SAMPLES - (CONTEXT + GUARD + 1) * HOP
                    or start % HOP or end % HOP or not SPANS[0] <= (end - start) // HOP <= SPANS[1]
                    or len(proposal["db"]) != 2 or any(not math.isfinite(value) or abs(value) > 121 for value in proposal["db"])):
                raise ValueError("Invalid transient measurement")
    dev_controls = [row for row in rows if row["case"]["stage"] == "development" and row["case"]["truth"] is None]
    points = {name: max(rank(row["peaks"][name])[0] for row in dev_controls) for name in ARMS}
    cells, gains, alarms, isolated = defaultdict(Counter), defaultdict(set), defaultdict(set), Counter()
    for row in rows:
        case = row["case"]
        stage, scene, role, truth, reader = (case[key] for key in ("stage", "scene", "role", "truth", "reader"))
        cell = cells[f"{stage}/{scene}/{role}"]
        cell["files"] += 1
        for name, proposal in row["peaks"].items():
            span = proposal["range"]
            alert = rank(proposal)[0] > points[name]
            useful = truth is not None and clock.useful([span], [truth])
            accurate = truth is not None and all(abs(a - b) <= LIMIT for a, b in zip(span, truth, strict=True))
            cell.update({f"{name}_flags": alert, f"{name}_useful": alert and useful, f"{name}_edges": alert and accurate,
                         f"{name}_useful_before_point": useful, f"{name}_edges_before_point": accurate})
            if stage == "held" and alert:
                if role == "cut_paste" and useful and scene != "clean":
                    gains[name].add((reader, scene))
                if role == "isolated":
                    isolated[name] += 1
                elif truth is None:
                    alarms[name].add(reader)
    summary = {name: {"noisy_useful": len(gains[name]), "noisy_scenes": len({scene for _, scene in gains[name]}),
                      "flagged_control_readers": len(alarms[name]), "flagged_isolated": isolated[name],
                      "passes": len(gains[name]) >= 3 and len({scene for _, scene in gains[name]}) >= 2
                      and len(alarms[name]) <= 1 and isolated[name] == 0} for name in ARMS}
    return {"points_db": points, "cells": {key: dict(value) for key, value in cells.items()}, "summary": summary,
            "previous_voice_check": {key: dict(value) for key, value in previous_cells.items()}}


def measure(freeze, args):
    rows = []
    signals = samples_for(freeze["cases"], args.development, args.held, args.sounds)
    for case, pcm in zip(freeze["cases"], signals, strict=True):
        ticks, *signals_by_arm = peaks(pcm)
        row = {"case": case, "pcm_sha256": io.digest(pcm.tobytes()), "status": "measured",
               "peaks": {arm: strongest(ticks, scores) for arm, scores in zip(ARMS, signals_by_arm, strict=True)}}
        rows.append(row)
    return {"freeze": freeze, "rows": rows, "results": assess(freeze, rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "review"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--held", type=Path)
    parser.add_argument("--sounds", type=Path)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        with args.receipt.open("xb") as frozen:
            frozen.write(io.encoded({"design": contract(), "runner_sha256": io.sha(Path(__file__))}))
        return
    if args.phase == "run":
        if None in (args.development, args.held, args.sounds, args.freeze) or args.receipt.exists():
            parser.error("Specify both pinned banks, the noise directory and the design lock, and use a new output path")
        if io.read(args.freeze) != {"design": contract(), "runner_sha256": io.sha(Path(__file__))}:
            raise ValueError("Frozen study design or runner changed")
        report = measure(contract(), args)
        io.save(args.receipt, report)
    else:
        report = io.read(args.receipt)
        if assess(report["freeze"], report["rows"]) != report["results"]:
            raise ValueError("The saved result differs from the recount")
    print(json.dumps(report["results"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
