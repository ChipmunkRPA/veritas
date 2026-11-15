# SPDX-License-Identifier: MIT
"""Test local decoder-frequency evidence on retained noisy partial-speech panels."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from scripts import audio_shared_mel_context as parent

source = parent.source
SCHEMA = "chipmunk.audio-decoder-peak-transfer/1"
PANELS = ("development", "random_placement", "families", "sine")
PARENT = parent.PANELS / "audio-shared-mel-context-2026-11-14.json"
RATE = 16000
BLOCK = RATE // 5
FRAMES = 4
WIDTH = BLOCK * FRAMES
STARTS = tuple(range(0, 6 * RATE - WIDTH + 1, BLOCK))
PATTERNS = ((1000, 2000, 3000, 4000), (1500, 3000, 6000), (1378.125, 2756.25, 5512.5),
            (1837.5, 3675, 7350), (2000, 4000, 6000))
FREQUENCIES = tuple(sorted({hz for pattern in PATTERNS for hz in pattern}))
ARMS = ("absolute", "contrast", "paired_contrast")
METHOD = {
    "clock": [RATE, BLOCK, WIDTH], "frequencies": list(FREQUENCIES),
    "paired_patterns": [list(pattern) for pattern in PATTERNS],
    "prominence": "Measure peak log power within 5 Hz of each predicted location. Compare with the median bins "
                  "20 to 60 Hz to either side. Use a Hann FFT in four nonoverlapping 200 ms blocks.",
    "contrasts": "Use the second-smallest prominence across the four blocks. Subtract the largest of zero and "
                 "the 75th percentile prominence from up to four adjacent blocks on either available side. "
                 "The paired rule uses the weaker of the two strongest contrasts on one frequency pattern.",
    "level_floor": "Skip a block below -60 dBFS RMS. Skip frequency-specific readings when the peak is below "
                   "-75 dBFS on the Hann amplitude scale. Require every interior block to meet the level constraints, "
                   "and at least three exterior blocks of literal source context.",
    "point": "Same development genuine source-group and isolated-background rule as the shared Mel comparison. "
             "Freeze a separate point per arm. Keep existing flags in the supplement arm.",
    "advancement": "At least two added correct locations across two generator families, at most one added genuine "
                   "source group in each held panel, and no added isolated background flag. Use 50% truth coverage "
                   "and 0.30 time IoU. Keep same-word resynthesis separate.",
}


def readouts(raw, *, project=False):
    import numpy as np

    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768
    source.check(len(pcm) == 6 * RATE, "Expected six seconds of native PCM")
    blocks = pcm.reshape(-1, BLOCK)
    taper = np.hanning(BLOCK)
    bins = np.fft.rfftfreq(BLOCK, 1 / RATE)
    # The replay evaluates only the registered bins by direct projection.
    if project:
        distance = np.abs(bins[:, None] - FREQUENCIES)
        selected = np.any((distance <= 5) | ((distance >= 20) & (distance <= 60)), axis=1)
        bins = bins[selected]
        phase = -2j * np.pi * bins[:, None] * np.arange(BLOCK) / RATE
        transform = (blocks * taper) @ np.exp(phase).T
    else:
        transform = np.fft.rfft(blocks * taper)
    amplitude = np.abs(transform) * (2.0 / taper.sum())
    power_db = 20 * np.log10(np.maximum(amplitude, 1e-12))
    audible = np.mean(blocks**2, axis=1) >= 1e-6
    prominences, valid = [], []
    for hz in FREQUENCIES:
        distance = np.abs(bins - hz)
        peak = power_db[:, distance <= 5].max(axis=1)
        nearby = np.median(power_db[:, (distance >= 20) & (distance <= 60)], axis=1)
        prominences.append(peak - nearby)
        valid.append(audible & (peak >= -75))
    values = np.stack(prominences, axis=1)
    supported = np.stack(valid, axis=1)
    output = []
    for index in range(len(STARTS)):
        end = index + FRAMES
        inner = np.sort(values[index:end], axis=0)[1]
        flanks = []
        for a, b in ((max(0, index - FRAMES), index), (end, min(len(blocks), end + FRAMES))):
            if b - a >= 3:
                flanks.append(np.quantile(values[a:b], .75, axis=0))
        outside = np.maximum.reduce([np.zeros(len(FREQUENCIES)), *flanks])
        measured = supported[index:end].all(axis=0)
        absolute = [round(float(score), 7) if ok else None for score, ok in zip(inner, measured, strict=True)]
        contrast = [round(float(score), 7) if ok else None for score, ok in zip(inner - outside, measured, strict=True)]
        output.append([absolute, contrast])
    return output


def ranked(values, frequencies):
    return sorted(((value, hz) for value, hz in zip(values, frequencies, strict=True) if value is not None), reverse=True)


def point_measure(scores, arm):
    selected = []
    for index, (absolute, contrast) in enumerate(scores):
        if arm == "paired_contrast":
            by_hz = dict(zip(FREQUENCIES, contrast, strict=True))
            for pattern in PATTERNS:
                options = ranked((by_hz[hz] for hz in pattern), pattern)
                if len(options) >= 2:
                    selected.append((options[1][0], -index, (options[0][1], options[1][1])))
        else:
            options = ranked(absolute if arm == "absolute" else contrast, FREQUENCIES)
            if options:
                value, hz = options[0]
                selected.append((value, -index, (hz,)))
    if not selected:
        return None
    value, negative_index, frequencies = max(selected)
    start = STARTS[-negative_index]
    return [value, [start, start + WIDTH], list(frequencies)]


def points(rows):
    isolated = parent.interval.BACKGROUNDS
    result = {}
    for arm in ARMS:
        ordinary_scores, backgrounds = {}, []
        for row in rows:
            top = point_measure(row["measurements"], arm)
            if row["truth"] or top is None or parent.existing(row):
                continue
            if row["kind"] in isolated:
                backgrounds.append(top[0])
            else:
                group = row["group"]
                ordinary_scores[group] = max(ordinary_scores.get(group, -math.inf), top[0])
        source.check(len(ordinary_scores) >= 2, "Too few ordinary calibration groups")
        second = sorted(ordinary_scores.values(), reverse=True)[1]
        result[arm] = max(backgrounds, default=second)
        result[arm] = max(result[arm], second)
    return result


def tally(rows, arm, point=None, supplement=False):
    counts, ordinary_groups, baseline_groups, new_locations = Counter(), set(), set(), []
    for row in rows:
        baseline = parent.existing(row)
        top = None if arm == "existing" else point_measure(row["measurements"], arm)
        span = baseline if arm == "existing" or (supplement and baseline) else top[1] if top and top[0] > point else None
        truth = row["truth"]
        counts["measured"] += bool(top) if arm != "existing" else (
            row["existing_scores"] is not None if row["existing_summary"] else any(v is not None for v in row["existing_scores"]))
        if truth:
            useful = bool(span and parent.stock.useful(span, truth))
            counts.update({"edits": 1, "useful": useful, "misplaced": bool(span and not useful)})
            if arm != "existing" and not supplement:
                counts["useful_top_before_point"] += bool(top and parent.stock.useful(top[1], truth))
                counts["measurable_near_edit"] += any(
                    parent.stock.useful([start, start + WIDTH], truth) and point_measure([scores], arm) is not None
                    for start, scores in zip(STARTS, row["measurements"], strict=True))
            if useful and not (baseline and parent.stock.useful(baseline, truth)):
                new_locations.append({key: row[key] for key in ("pcm_sha256", "family", "truth")} | {"selection": top})
        elif row["kind"] in parent.interval.BACKGROUNDS:
            counts.update({"isolated": 1, "isolated_flags": bool(span), "added_isolated_flags": bool(span and not baseline)})
        elif row["kind"] == "resyn":
            counts.update({"same_words": 1, "same_words_flags": bool(span)})
        else:
            counts.update({"ordinary": 1, "ordinary_flags": bool(span), "added_ordinary_flags": bool(span and not baseline)})
            if span:
                ordinary_groups.add(row["group"])
            if baseline:
                baseline_groups.add(row["group"])
    return {"counts": dict(counts), "ordinary_groups": len(ordinary_groups),
            "new_ordinary_groups": len(ordinary_groups - baseline_groups),
            "added_locations": new_locations}


def outcome(panels, frozen):
    results = {panel: {"existing": tally(rows, "existing"), **{
        arm: {"alone": tally(rows, arm, frozen[arm]), "supplement": tally(rows, arm, frozen[arm], supplement=True)}
        for arm in ARMS}} for panel, rows in panels.items()}
    advances = {}
    for arm in ARMS:
        held = [results[panel][arm]["supplement"] for panel in PANELS[1:]]
        additions = [row for panel in held for row in panel["added_locations"]]
        advances[arm] = (len(additions) >= 2 and len({r["family"] for r in additions}) >= 2
                         and all(r["new_ordinary_groups"] <= 1 and not r["counts"].get("added_isolated_flags") for r in held))
    return {"panels": results, "advance": advances}


def identity(panel):
    return [row | {"existing_summary": panel in ("families", "sine")} for row in parent.input_rows(panel)]


def parent_bindings():
    return source.read(PARENT)["freeze"]["bindings"] | {"shared_mel_receipt": source.path_sha(PARENT)}


def verify(receipt):
    source.check(receipt["schema"] == SCHEMA and receipt["method"] == METHOD and set(receipt["panels"]) == set(PANELS)
                 and receipt["parents"] == parent_bindings(), "Frozen method or source records differ")
    for panel in PANELS:
        for row, expected in zip(receipt["panels"][panel], identity(panel), strict=True):
            source.check({key: row[key] for key in expected} == expected, "Parent labels, source or baseline differ")
            cells = row["measurements"]
            source.check(len(cells) == len(STARTS) and all(
                len(cell) == 2 and all(len(view) == len(FREQUENCIES) for view in cell)
                and all((a is None and b is None) or (math.isfinite(a) and math.isfinite(b) and a >= b)
                        for a, b in zip(*cell, strict=True)) for cell in cells), "Invalid spectral measurements")
    frozen = points(receipt["panels"]["development"])
    source.check(frozen == receipt["points"], "Points do not match development measurements")
    result = outcome(receipt["panels"], frozen)
    source.check(result == receipt["outcome"], "Reported counts differ from measurements")
    return result


def compare(primary, projection):
    source.check(verify(primary) == verify(projection) and primary["points"] == projection["points"],
                 "The native implementations disagree on points or outcomes")
    clips = values = rounding_differences = unavailable = 0
    maximum_error = 0.0
    for panel in PANELS:
        for first, second in zip(primary["panels"][panel], projection["panels"][panel], strict=True):
            clips += 1
            for cell, repeat in zip(first["measurements"], second["measurements"], strict=True):
                for view, replayed in zip(cell, repeat, strict=True):
                    for a, b in zip(view, replayed, strict=True):
                        if a is None or b is None:
                            source.check(a is b, "Measurement availability changed")
                            unavailable += 1
                            continue
                        values += 1
                        rounding_differences += a != b
                        maximum_error = max(maximum_error, abs(a - b))
    source.check(maximum_error <= 2e-7, "Native frequency measurements differ")
    return {"clips": clips, "numeric_readings": values, "unavailable_readings": unavailable,
            "rounding_differences": rounding_differences, "maximum_error_db": maximum_error}


def write_receipt(path, receipt):
    # One line per case keeps the per-window measurements available without a line for each bin.
    panels = receipt["panels"]
    header = json.dumps({k: v for k, v in receipt.items() if k != "panels"}, indent=2, allow_nan=False)
    with path.open("w") as out:
        out.write(header.removesuffix("\n}") + ',\n  "panels": {\n')
        for offset, (panel, rows) in enumerate(panels.items()):
            if offset:
                out.write(",\n")
            out.write(f'    {json.dumps(panel)}: [\n')
            out.write(",\n".join("      " + json.dumps(row, separators=(",", ":"), allow_nan=False) for row in rows))
            out.write("\n    ]")
        out.write("\n  }\n}\n")


def run(args):
    _, native, _ = parent.interval.native_rows(args.archive)
    parents = parent_bindings()
    source.check(parent.ledger_roles() == {k: v for k, v in parents.items() if k != "shared_mel_receipt"},
                 "Native source bindings differ from the parent receipt")
    sine_parent = source.read(parent.SINE_PARENT)
    parent.sine.audit_native(args.sine, sine_parent)
    for _, pcm in parent.sine.native_rows(args.sine, sine_parent["freeze"]):
        raw = pcm.tobytes()
        native[source.digest(raw)] = raw
    args.path.mkdir(mode=0o700)
    source.write(args.path / "method.json", METHOD)
    measured = {}
    for panel in PANELS:
        measured[panel] = []
        for row in identity(panel):
            raw = native[row["pcm_sha256"]]
            source.check(source.digest(raw) == row["pcm_sha256"], "Source PCM differs")
            measured[panel].append(row | {"measurements": readouts(raw, project=args.phase == "project")})
        if panel == "development":
            frozen = points(measured[panel])
            source.write(args.path / "points-before-held.json", frozen)
    result = outcome(measured, frozen)
    receipt = {"schema": SCHEMA, "method": METHOD, "parents": parents, "panels": measured,
               "points": frozen, "outcome": result}
    verify(receipt)
    write_receipt(args.path / "receipt.json", receipt)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("measure", "project", "review", "compare"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--sine", type=Path)
    parser.add_argument("--projection", type=Path)
    args = parser.parse_args()
    if args.phase in ("measure", "project") and (args.archive is None or args.sine is None):
        parser.error("Native replay requires --archive and --sine")
    if args.phase == "compare":
        if args.projection is None:
            parser.error("Comparing requires --projection")
        result = compare(source.read(args.path), source.read(args.projection))
    elif args.phase == "review":
        result = verify(source.read(args.path))
    else:
        result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
