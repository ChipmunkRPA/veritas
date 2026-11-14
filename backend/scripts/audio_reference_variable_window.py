# SPDX-License-Identifier: Apache-2.0
"""Compare shorter earlier-copy ranges under recorded rain and train sound."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_dense_clean_reference_transfer as dense_study
from scripts import audio_vocal_event_mixtures as io
from scripts import review_audio_reduced_background_localization as clock

RATE = 16_000
WIDTHS = (8, 12, 16)
HOP = 640
SCHEMA = "chipmunk.audio-reference-variable-window/1"
METHOD = {
    "widths_ms": [width * 40 for width in WIDTHS],
    "selection": "For each length, use the maximum response on matched ordinary development cases across all three backgrounds.",
    "combination": "Keep the 960 ms incumbent when it reports a range. Else choose the largest absolute excess over its length-specific point.",
    "inputs": "Use clean exact earlier copies, hold the clock fixed, add matched rain and train noise at -5 dB.",
    "processing_guard": "Compare with the existing twelve-second broad frequency balance rule, using its same three sections over six seconds.",
}
SCENES = ("clean", "rain", "train")
CONTROLS = ("untouched", "gain", "eq", "unrelated", "background")
require = dense_study.require


def compact(result, status):
    return {"status": status, "measurement": None if result is None else
            {key: result[key] for key in ("samples", "relative_drop")}}


def measurements(first, second):
    import numpy as np

    from app.evidence import audio_clean_speech as speech
    from app.evidence import audio_clean_speech_dense as dense
    from app.evidence import audio_clean_speech_long as long

    state, old = dense.locate(first, second)
    baseline = compact(old, state)
    if state == "reference_match_not_supported":
        return baseline, [compact(None, state) for _ in WIDTHS], False
    broad_eq, _ = long.frequency_balance(first, second, section_samples=2 * RATE)
    stats = list(speech.band_stats(first, second, np.hanning(dense.WIDTH), hop=HOP))
    frames = len(stats[0][1])
    outputs = []
    for width in WIDTHS:
        starts = np.arange(1, frames - width)
        ends = starts + width
        bounds = (
            (np.maximum(0, starts - dense.FLANK_FRAMES - dense.GUARD_FRAMES),
             np.maximum(0, starts - dense.GUARD_FRAMES)),
            (starts, ends),
            (np.minimum(frames, ends + dense.GUARD_FRAMES),
             np.minimum(frames, ends + dense.GUARD_FRAMES + dense.FLANK_FRAMES)),
        )
        deviations = []
        for _, strength, gain, cutoff, cap in stats:
            (left, middle, right), support = dense._averages(strength, gain, cutoff, cap, bounds)
            flank = np.minimum(left, right)
            valid = (flank > .25) & (np.maximum(left, right) <= 2.2 * flank) & (np.min(support, axis=0) >= 3)
            deviations.append(np.where(valid, 1 - middle / np.maximum(flank, 1e-10), np.nan))
        values = np.stack(deviations)
        eligible = np.flatnonzero(np.sum(np.isfinite(values), axis=0) >= 2)
        if not eligible.size:
            outputs.append(compact(None, "local_support_not_available"))
            continue
        responses = np.nanmedian(values[:, eligible], axis=0)
        best = int(np.argmax(responses))
        i = int(eligible[best])
        start = int(starts[i]) * HOP + (dense.WIDTH - HOP) // 2
        outputs.append(compact({"samples": [start, start + width * HOP],
                                "relative_drop": round(float(responses[best]), 8)}, "measured"))
    return baseline, outputs, broad_eq


def read_noise(folder):
    import numpy as np

    from scripts.audio_focused_window_transfer import SCENES as PINS

    backgrounds = {}
    for scene, spec in PINS.items():
        file = folder / f"{scene}.ogg"
        require(io.sha(file) == spec["sha256"], "Unpinned background recording")
        values = np.frombuffer(io.decode(file), dtype="<i2")
        backgrounds[scene] = [values[i * RATE:(i + 6) * RATE] for i in spec["offsets"]]
    return backgrounds


def source_groups(folder, role):
    from scripts.audio_sine_resynthesis_transfer import native_rows

    if role == "sine":
        parent = io.read(io.PUBLIC / "audio-sine-resynthesis-transfer-2026-11-14.json")
        frozen = io.read(folder / "freeze.json")
        require(frozen == parent["freeze"], "SINE selection differs from the publisher readout")
        previous = {(c["id"], c["condition"], c["side"]): c for c in parent["records"]}
        groups = defaultdict(dict)
        for item, values in native_rows(folder, frozen):
            if item["condition"] != "clean" or item["role"] not in {"cut_paste", "resyn"}:
                continue
            pin = previous[item["id"], item["condition"], item["side"]]["pcm_sha256"]
            require(io.digest(values.tobytes()) == pin, "SINE source PCM differs")
            groups[item["id"]][item["side"]] = item, values
        for group, views in groups.items():
            before = views["original"][1]
            meta, after = views["variant"]
            yield group, before, [(meta["role"], meta["truth"], after)]
        return
    builds = defaultdict(list)
    if role == "development":
        for group, kind, level, truth, first, second in dense_study.source_bank(folder, role):
            if level is None:
                builds[group].append((kind, truth, first, second))
    else:
        import numpy as np

        parent = io.read(io.PUBLIC / "audio-random-clock-stock-2026-11-13.json")
        lock = folder / "placements-freeze.json"
        require(io.sha(lock) == parent["native_freeze_sha256"], "Held stock roster changed")
        rows = io.read(lock)["native_manifest_rows"]
        with np.load(folder / "placements-native.npz", allow_pickle=False) as archive:
            dry = archive["dry"]
        for row, signal in zip(rows, dry, strict=True):
            if row["snr"] != 0 or type(row["group"]) is not int:
                continue
            require(signal.shape == (6 * RATE,) and signal.dtype == np.dtype("<i2")
                    and io.digest(signal.tobytes()) == row["dry_pcm_sha256"], "Held dry recording differs")
            builds[row["group"]].append((row["kind"], row["edit_clock"], signal, signal))
    for group, cases in sorted(builds.items()):
        before = next(first for kind, _, first, _ in cases if kind == "untouched")
        known_edit = next(truth for _, truth, _, _ in cases if truth)
        yield group, before, [(kind, truth or known_edit, second)
                              for kind, truth, _, second in cases if kind != "untouched"]


def authored(before, source, backgrounds, i):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    gain = before.astype(np.float64)
    gain[34880:44160] *= .7
    eq = sosfiltfilt(butter(2, 2300, fs=RATE, output="sos"), before)
    variants = [("untouched", None, before), ("gain", None, gain), ("eq", None, eq), *source]
    for scene in SCENES:
        if scene == "clean":
            added = np.zeros_like(gain)
        else:
            clip = backgrounds[scene][i % 4].astype(np.float64)
            added = clip * (np.linalg.norm(before) / np.linalg.norm(clip)) * 10**.25
        peak = max(np.max(np.abs(signal + added)) for _, _, signal in variants)
        scale = min(1., 32000 / peak)
        for kind, truth, signal in variants:
            yield scene, kind, truth, np.rint((signal + added) * scale).astype("<i2")
        if scene != "clean":
            yield scene, "background", None, np.rint(added * scale).astype("<i2")


def pairs(role, folder, sounds):
    group_rows = list(source_groups(folder, role))
    require(len(group_rows) == (24 if role == "sine" else 8), "Source groups differ from the fixed roster")
    noises = read_noise(sounds)
    for i, (group, before, sources) in enumerate(group_rows):
        wrong_reference = group_rows[(i + 1) % len(group_rows)][1]
        for scene, kind, truth, after in authored(before, sources, noises, i):
            yield group, scene, kind, truth, before, after
            if kind == "untouched":
                yield group, scene, "unrelated", None, wrong_reference, after


def assess(role, folder, sounds):
    rows = []
    for group, scene, kind, truth, first, after in pairs(role, folder, sounds):
        baseline, short, broad_eq = measurements(first, after)
        rows.append({"group": group, "scene": scene, "kind": kind, "truth": truth,
                     "earlier_sha256": io.digest(first.tobytes()), "submitted_sha256": io.digest(after.tobytes()),
                     "baseline": baseline, "shorter": short, "broad_eq": broad_eq})
    print(role, len(rows), "pairs", flush=True)
    return rows


def points(rows):
    return [max(row["shorter"][i]["measurement"]["relative_drop"] for row in rows if row["kind"] in CONTROLS
                and row["shorter"][i]["measurement"]) for i in range(len(WIDTHS))]


def pick(readings, cutoffs):
    eligible = [(item["relative_drop"] - point, -i, item) for i, (row, point) in enumerate(zip(readings, cutoffs, strict=True))
                if (item := row["measurement"]) and item["relative_drop"] > point]
    return max(eligible)[2] if eligible else None


def summarize(rows, cutoffs):
    cells = defaultdict(Counter)
    for row in rows:
        old = row["baseline"]["measurement"]
        short = pick(row["shorter"], cutoffs)
        truth = [row["truth"]] if row["truth"] else []
        cell = cells[row["scene"] + "/" + row["kind"]]
        cell["pairs"] += 1
        for arm, proposal in (("baseline", old), ("shorter", short), ("fallback", old or short),
                              ("guarded", None if row["broad_eq"] else old)):
            useful = proposal is not None and clock.useful([proposal["samples"]], truth)
            cell.update({f"{arm}_flagged": proposal is not None, f"{arm}_useful": useful})
        cell["same_clock_admitted"] += row["baseline"]["status"] != "reference_match_not_supported"
    return {name: dict(values) for name, values in cells.items()}


def verify_measurements(rows):
    states = {"measured", "local_change", "no_local_change", "reference_match_not_supported", "local_support_not_available"}
    for row in rows:
        require(len(row["shorter"]) == len(WIDTHS) and row["scene"] in SCENES and type(row["broad_eq"]) is bool
                and all(clock.SHA.fullmatch(row[key]) for key in ("earlier_sha256", "submitted_sha256")), "Unexpected pair or input pin")
        for reading, width in zip([row["baseline"], *row["shorter"]], [24, *WIDTHS], strict=True):
            require(reading["status"] in states, "Unknown disposition")
            measurement = reading["measurement"]
            require((measurement is not None) == (reading["status"] in {"measured", "local_change"}), "Missing measurement")
            if measurement:
                a, b = measurement["samples"]
                require(0 <= a < b <= 6 * RATE and b-a == width * HOP
                        and math.isfinite(measurement["relative_drop"]),
                        "Invalid earlier-copy range")


def run(folders, sounds, output):
    header = {"schema": SCHEMA, "method": METHOD}
    development = assess("development", folders["development"], sounds)
    cutoffs = points(development)
    output.mkdir(mode=0o700)
    io.save(output / "points.json", header | {"development_sha256": io.commitment(development), "points": cutoffs})
    roles = {"development": development}
    for role in ("assessment", "sine"):
        roles[role] = assess(role, folders[role], sounds)
    io.save(output / "receipt.json", header | {"points": cutoffs, "panels": roles,
             "results": {role: summarize(rows, cutoffs) for role, rows in roles.items()}})


def review(path, folders, sounds):
    data = io.read(path)
    require(data["schema"] == SCHEMA and data["method"] == METHOD, "Wrong policy")
    panels = data["panels"]
    require(set(panels) == {"development", "assessment", "sine"}, "Missing panel")
    require([len(panels[key]) for key in ("development", "assessment", "sine")] == [208, 208, 408],
            "A comparison panel is incomplete")
    for role, rows in panels.items():
        verify_measurements(rows)
        if sounds:
            require(rows == assess(role, folders[role], sounds), "Measurements differ from the original PCM")
    cutoffs = points(panels["development"])
    require(data["points"] == cutoffs, "Points differ from the development records")
    results = {role: summarize(rows, cutoffs) for role, rows in panels.items()}
    require(data["results"] == results, "Differences in the held counts")
    print(json.dumps(results, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "review"))
    parser.add_argument("path", type=Path)
    for key in ("development", "assessment", "sine", "sounds"):
        parser.add_argument(f"--{key}", type=Path)
    args = parser.parse_args()
    folders = {key: getattr(args, key) for key in ("development", "assessment", "sine")}
    if args.phase == "review":
        if args.sounds and not all(folders.values()):
            parser.error("Native replay requires all three source folders")
        review(args.path, folders, args.sounds)
    else:
        if not (args.sounds and all(folders.values())):
            parser.error("Run requires all three source folders and backgrounds")
        run(folders, args.sounds, args.path)


if __name__ == "__main__":
    main()
