"""Compare short background-tone phase changes with matched source controls."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_transient_edges_transfer as prior

io = prior.io
SCHEMA = "chipmunk.audio-tonal-phase-transfer/1"
RATE, SAMPLES, FADE = 16000, 73600, 240
ARMS = ("single", "harmonic")
TONES = {
    "Mains_hum_60_Hz.ogg": ("development", "dd11780ff2161f3a2f0ced1ed7128f9ddf6772e2df2b1bf246222abc2552aee5"),
    "60-hz.wav": ("development", "b973b5fc3d1fc9887497aa6ac7d927862c167909397d10eb7722c64e1554b538"),
    "MainsBrum50Hz.ogg": ("held", "88bd2a95cae498ffc4534ca51001c5375f99067eea42f10fa87ea5ff85f58997"),
    "60Hz_hum_square.ogg": ("held", "a4230848b204d806671a4ef858626013dc66c8ca57f941ab979a6118740ba7ca"),
}
SCENES = ("rain", "train", "competing_speech")
POSITIVES = ("tone_cut", "mixed_cut")
CONTROLS = ("untouched", "gain", "lowpass", "publisher_voice_edit", "whole_delay", "no_tone", "continuous_warp")
METHOD = {
    "source": "New source-disjoint SINE reader groups. Two previously unused tone sources per role. "
              "Rain, train or concurrent speech at +5 dB relative to original speech RMS.",
    "tone": "Mix recorded hum at -7 or -13 dB relative to the foreground. Test a temporary splice in the tone "
            "and separately in the entire mix. Preserve samples outside the labeled span.",
    "control": "Continuous tone, local gain, low-pass, publisher voice splice under continuous tone, whole delay, "
               "no added tone, smooth tone warp, and isolated ambient recordings.",
    "candidate": "320 ms Hann projections every 40 ms on a 1 kHz analysis copy. Require tones at centers 240 ms "
                 "before and after both boundaries. Interpolate phase across the outer supports and measure the "
                 "interior residual. Compare the best single tone and two compatible harmonics.",
    "point": "Strictly exceed the greater of .60 or the largest ordinary development-group score. "
             "A broader point uses the second-highest group maximum.",
    "utility": "A useful range covers at least half the labeled edit and reaches .30 temporal intersection over union. "
               "Also count both endpoints within 120 ms.",
}


def freeze():
    pairs = {row["reader"]: row["id"] for row in io.read(prior.DEVELOPMENT)["freeze"]["pairs"] if row["role"] == "cut_paste"}
    readers = sorted(pairs, key=lambda reader: io.digest(f"{SCHEMA}/{reader}".encode()))
    if len(readers) <= 8:
        raise ValueError("Missing speech reader groups")
    cases = []
    for index, reader in enumerate(readers):
        stage = "development" if index < 8 else "held"
        choices = [name for name, (role, _) in TONES.items() if role == stage]
        for slot, scene in enumerate(SCENES):
            start = 18000 + ((index * 11 + slot * 19) % 23) * 480
            end = start + (8960, 11520, 14080, 16640)[(index + slot) % 4]
            offset = (1495, 1712, 2093, 2192)[(index * 3 + slot) % 4]
            for role in (*POSITIVES, *CONTROLS):
                cases.append({"stage": stage, "reader": reader, "id": pairs[reader], "scene": scene,
                              "tone": choices[(index + slot) % len(choices)], "slot": index,
                              "tone_db": -13 if (index // 2 + slot) % 2 else -7,
                              "offset": -offset if (index // 3 + slot) % 2 else offset,
                              "authored_interval": [start, end], "role": role, "truth": [start, end] if role in POSITIVES else None})
    for stage in ("development", "held"):
        for scene in SCENES:
            for slot in range(4):
                cases.append({"stage": stage, "reader": None, "id": None, "scene": scene,
                              "slot": slot + (4 if stage == "held" else 0), "role": "isolated", "truth": None})
    return {"schema": SCHEMA, "method": METHOD, "sine_sha256": io.sha(prior.DEVELOPMENT),
            "assets": {name: list(spec) for name, spec in TONES.items()}, "files": cases}


def read_assets(folder):
    import numpy as np

    captures = {}
    for name, (_, expected) in TONES.items():
        path = folder / name
        if io.sha(path) != expected:
            raise ValueError("Tone does not match the publisher file")
        pcm = np.frombuffer(io.decode(path), dtype="<i2")[:SAMPLES].astype(np.float64)
        if len(pcm) != SAMPLES:
            raise ValueError("Recorded tone is shorter than the comparison")
        captures[name] = pcm - pcm.mean()
    return captures


def rms(pcm):
    import numpy as np

    return float(np.linalg.norm(pcm) / math.sqrt(len(pcm)))


def splice(pcm, start, end, offset, *, factor=1.):
    import numpy as np

    result = pcm.copy()
    weight = np.ones(end - start)
    weight[:FADE] = np.linspace(0, 1, FADE, endpoint=False)
    weight[-FADE:] = weight[:FADE][::-1]
    result[start:end] = (1 - weight) * pcm[start:end] + weight * pcm[start + offset:end + offset] * factor
    return result


def variants(case, original, edited, room, hum):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    start, end = case["authored_interval"]
    offset = case["offset"]
    foreground = original[:SAMPLES].astype(np.float64)
    background = room[:SAMPLES].astype(np.float64)
    background *= 10 ** (.25) * rms(foreground) / max(1., rms(background))
    hum = hum * (10 ** (case["tone_db"] / 20) * rms(foreground) / max(1., rms(hum)))
    unedited = foreground + background + hum
    time = np.arange(SAMPLES, dtype=np.float64)
    warp = 48 * (1 - np.cos(2 * np.pi * time / SAMPLES))
    results = {
        "tone_cut": foreground + background + splice(hum, start, end, offset),
        "mixed_cut": splice(unedited, start, end, offset),
        "untouched": unedited,
        "gain": splice(unedited, start, end, 0, factor=.60),
        "lowpass": sosfiltfilt(butter(2, 1900, fs=RATE, output="sos"), unedited),
        "publisher_voice_edit": edited[:SAMPLES] + background + hum,
        "whole_delay": np.roll(unedited, offset),
        "no_tone": foreground + background,
        "continuous_warp": foreground + background + np.interp(time + warp, time, hum),
    }
    scale = min(1.0, 32000.0 / max(float(np.max(np.abs(x))) for x in results.values()))
    return {name: np.rint(values * scale).astype("<i2") for name, values in results.items()}


def rows_for(plan, stage, bank, tones, sounds):
    from scripts.audio_reference_variable_window import read_noise

    exact, competing = prior.pair_bank(bank, io.read(prior.DEVELOPMENT)["freeze"])
    ambient = read_noise(sounds) | {"competing_speech": competing}
    hums = read_assets(tones)
    previous, grouped = None, {}
    for case in plan["files"]:
        if case["stage"] != stage:
            continue
        scene = case["scene"]
        room = ambient[scene][case["slot"] % len(ambient[scene])]
        if case["role"] == "isolated":
            yield case, room[:SAMPLES]
            continue
        key = case["id"], scene
        if key != previous:
            grouped = variants(case, exact[key[0], "original"], exact[key[0], "variant"], room, hums[case["tone"]])
            previous = key
        yield case, grouped[case["role"]]


def observation(pcm):
    from app.evidence import audio_continuity, audio_tonal_phase

    original = pcm.astype("float64") / 32768
    counts, candidates = audio_tonal_phase.measure(original)
    spectral = audio_continuity.select(audio_continuity.candidates(original))
    return {"sha256": io.digest(pcm.tobytes()), "comparisons": counts, "ranges": candidates,
            "spectral_boundaries": sorted(row["boundary_sample"] for row in spectral)}


def readouts(plan, stage, args):
    results = []
    for case, samples in rows_for(plan, stage, args.bank, args.hums, args.sounds):
        results.append({"case": case, **observation(samples)})
        if len(results) % 48 == 0:
            print(stage, len(results), flush=True)
    return results


def calibration(rows):
    points = {}
    for arm in ARMS:
        maxima = defaultdict(float)
        for row in rows:
            case = row["case"]
            if case["truth"] is None:
                reader = str(case["reader"])
                candidate = row["ranges"][arm]
                maxima[reader] = max(maxima[reader], candidate["score"] if candidate else 0)
        extremes = sorted(maxima.values(), reverse=True)
        points[arm] = {"tight": max(.60, extremes[0]), "broader": max(.60, extremes[1]), "ordinary_maxima": dict(maxima)}
    return points


def useful(interval, truth):
    a, b = interval
    x, y = truth
    overlap = max(0, min(b, y) - max(a, x))
    return 2 * overlap >= y - x and 10 * overlap >= 3 * (max(b, y) - min(a, x))


def summarize(rows, points):
    stats, false_controls = defaultdict(Counter), defaultdict(set)
    for row in rows:
        case = row["case"]
        truth = case["truth"]
        bands = row["spectral_boundaries"]
        recorded = [("spectral", [[a, b] for a in bands for b in bands if 8320 <= b - a <= 18560], True)]
        for arm in ARMS:
            candidate = row["ranges"][arm]
            for name in ("tight", "broader"):
                selected = bool(candidate and candidate["score"] > points[arm][name])
                spans = [candidate["tones"][0]["samples"]] if selected else []
                recorded.append((f"{arm}/{name}", spans, row["comparisons"][arm] > 0))
        for method, spans, assessed in recorded:
            for location in ("all", case["scene"]):
                cell = stats[f"{method}/{location}/{case['role']}"]
                cell.update(total=1, assessed=int(assessed), flags=int(bool(spans)))
                if truth is not None:
                    cell["useful"] += int(any(useful(span, truth) for span in spans))
                    cell["both_edges_within_120ms"] += int(any(
                        max(abs(a - b) for a, b in zip(span, truth, strict=True)) <= 1920 for span in spans
                    ))
            if truth is None and spans and case["reader"] is not None:
                false_controls[method].add(case["reader"])
    return {"counts": {key: dict(value) for key, value in stats.items()},
            "ordinary_reader_groups": {key: len(group) for key, group in false_controls.items()}}


def review(document):
    if document["freeze"] != freeze():
        raise ValueError("Source groups or the measurement design changed")
    expected = [case for stage in ("development", "held") for case in document["freeze"]["files"] if case["stage"] == stage]
    for case, row in zip(expected, document["development"] + document["held"], strict=True):
        if case != row["case"] or not prior.clock.SHA.fullmatch(row["sha256"]):
            raise ValueError("A source commitment changed")
        if set(row["comparisons"]) != set(ARMS) or set(row["ranges"]) != set(ARMS):
            raise ValueError("A comparison arm is missing")
        for arm in ARMS:
            count = row["comparisons"][arm]
            candidate = row["ranges"][arm]
            if type(count) is not int or count < 0 or candidate and not count:
                raise ValueError("Invalid comparison coverage")
            if candidate:
                tones = candidate["tones"]
                start, end = tones[0]["samples"]
                centers = [start - 3840, start + 3840, end - 3840, end + 3840]
                if (len(tones) != (1 if arm == "single" else 2) or not 8320 <= end - start <= 18560
                        or not 0 <= centers[0] < centers[-1] < SAMPLES or not -.1 <= candidate["score"] <= math.pi
                        or any(tone["samples"] != [start, end] or tone["support_centers"] != centers
                               or tone["tone_above_neighbors_db"] < 8 or tone["tone_level_dbfs"] < -58 for tone in tones)):
                    raise ValueError("A phase range differs from its measured supports")
    points = calibration(document["development"])
    summary = summarize(document["held"], points)
    if points != document["points"] or summary != document["results"]:
        raise ValueError("The frozen phase comparison does not replay")
    return summary


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "develop", "held", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--hums", type=Path)
    parser.add_argument("--sounds", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        write(args.path, freeze())
    elif args.phase == "develop":
        plan = io.read(args.path)
        if plan != freeze():
            raise ValueError("The source roster changed")
        development = readouts(plan, "development", args)
        write(args.output, {"freeze": plan, "development": development, "points": calibration(development)})
    elif args.phase == "held":
        study = io.read(args.path)
        if study["freeze"] != freeze() or calibration(study["development"]) != study["points"]:
            raise ValueError("The development measurements do not reproduce")
        held = readouts(study["freeze"], "held", args)
        result = study | {"held": held, "results": summarize(held, study["points"])}
        review(result)
        write(args.output, result)
    else:
        print(json.dumps(review(io.read(args.path)), sort_keys=True))


if __name__ == "__main__":
    main()
