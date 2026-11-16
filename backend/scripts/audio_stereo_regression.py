# SPDX-License-Identifier: MIT
"""Compare physical channels and mid-side interference removal on short speech edits."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from app.evidence.audio_short_review_binding import GRID, RATE, REVIEW_POINT, SAMPLES, speech_counts
from scripts import audio_short_edit_union_holdout as union
from scripts import audio_two_voice_components as voices

SOURCE = voices.SOURCE
PANELS = union.PANEL.parent
PHYSICAL = PANELS / "audio-two-voice-components-2026-11-15.json.gz"
SINE = PANELS / "audio-short-edit-union-holdout-2026-11-15.json.gz"
SCHEMA = "chipmunk.audio-stereo-regression/1"
VIEWS = ("mono", "left", "right", "residual")
METHODS = ("mono", "channels", "residual", "agreement")
FFT, HOP, RIDGE, MAX_GAIN = 512, 128, 0.10, 2.0
MIN_COHERENCE, LOW_HZ, HIGH_HZ = 0.15, 200, 6000
METHOD = {
    "reference": "Scan the five established two-second windows on the arithmetic mono of each stereo clip. "
    "Scan both untouched physical channels and a mid-side interference-reduced view independently "
    "at identical source times, with the same voice model and per-view speech gate.",
    "residual": "Estimate one complex least-squares predictor per frequency from the six-second side into mid. "
    "Use ridge protection proportional to mid power, limit transfer gain and omit low-coherence bins. "
    "Subtract estimated coherent interference from mid. This can suppress the target when it is off center.",
    "decision": "Compare highest available physical-channel logit, residual logit, and smaller same-window "
    "logit of the two. Freeze separate strict and wider points at the highest and second-highest "
    "ordinary speaker-pair maxima in development. Report standalone and serving-baseline fallback "
    "decisions. No source identities, source levels or edit times choose the signal at inference.",
    "sources": "Replay the two-voice stereo render benchmark. Add new matched mixtures of the retained SINE "
    "voices and six recorded meeting segments on microphones 01 and 05. Put the target equally in "
    "both channels on alternating reader pairs. Otherwise attenuate the far channel and delay it by "
    "three samples. Put room speech 5 dB above target by mono RMS and keep gain matched within each pair.",
}


def sources():
    return union.duration.read_gz(PHYSICAL), union.duration.read_gz(SINE)


def contract(sine):
    return {
        "schema": SCHEMA,
        "method": METHOD,
        "code_sha256": SOURCE.path_sha(Path(__file__)),
        "physical_sha256": SOURCE.path_sha(PHYSICAL),
        "sine_sha256": SOURCE.path_sha(SINE),
        "assets": sine["frozen"]["protocol"]["current_assets"],
        "grid": [list(span) for span in GRID],
        "filter": [FFT, HOP, RIDGE, MAX_GAIN, MIN_COHERENCE, LOW_HZ, HIGH_HZ],
        "mono_point": REVIEW_POINT,
    }


def audio_hash(values):
    return SOURCE.digest(values.astype("<f4").tobytes())


def pcm_hash(values):
    return SOURCE.digest(values.astype("<i2").tobytes())


def arithmetic_mono(stereo):
    import numpy as np

    return np.rint(stereo.mean(axis=1)).astype(np.float32) / 32768


def spectral(values):
    from scipy.signal import stft

    return stft(
        values, fs=RATE, window="hann", nperseg=FFT, noverlap=FFT - HOP, boundary="zeros", padded=True
    )


def fit_filter(pcm):
    import numpy as np

    frequency, _, mid = spectral(pcm.mean(axis=1))
    _, _, side = spectral((pcm[:, 0] - pcm[:, 1]) / 2)
    mid_power, side_power = (np.sum(abs(values) ** 2, axis=1) for values in (mid, side))
    cross = (mid * side.conj()).sum(axis=1)
    coherence = abs(cross) ** 2 / (mid_power * side_power + 1e-24)
    weights = cross / (side_power + RIDGE * mid_power + 1e-12)
    weights *= np.minimum(1, MAX_GAIN / (abs(weights) + 1e-12))
    active = (frequency >= LOW_HZ) & (frequency <= HIGH_HZ) & (coherence >= MIN_COHERENCE)
    return weights * active


def project(pcm, weights):
    from scipy.signal import istft

    _, _, mid = spectral(pcm.mean(axis=1))
    _, _, side = spectral((pcm[:, 0] - pcm[:, 1]) / 2)
    _, values = istft(mid - weights[:, None] * side, fs=RATE, window="hann", nperseg=FFT, noverlap=FFT - HOP)
    return values[: len(pcm)]


def acoustic_check(stems, weights, mixture, target):
    import numpy as np

    if target is None:
        return None
    signal, noise = stems[target], stems[1 - target]
    original = (signal.mean(axis=1), noise.mean(axis=1))
    filtered = (project(signal, weights), project(noise, weights))
    SOURCE.check(
        np.max(abs(filtered[0] + filtered[1] - mixture)) < 1e-6,
        "Fixed filter does not preserve exact stem addition",
    )

    def power(values):
        return float(values @ values)

    snrs = [10 * math.log10(max(power(a), 1e-16) / max(power(b), 1e-16)) for a, b in (original, filtered)]
    target_change = 10 * math.log10(max(power(filtered[0]), 1e-16) / max(power(original[0]), 1e-16))
    return {
        "mono_snr_db": round(snrs[0], 4),
        "residual_snr_db": round(snrs[1], 4),
        "target_energy_change_db": round(target_change, 4),
    }


def physical_meta(plan):
    records = []
    for phase in ("development", "held"):
        for index, scene, condition, item in voices.cases(plan, phase):
            targets = [i for i, role in enumerate(voices.CONDITIONS[condition]) if role == "synthetic"]
            if phase == "development" and targets:
                continue
            records.append(
                {
                    "source": "physical",
                    "cohort": scene["cohort"],
                    "group": scene["pair"],
                    "scene": index,
                    "role": condition,
                    "pcm_sha256": item["pcm_sha256"],
                    "stems": item["stems"],
                    "labels": [scene["voices"][i]["span"] for i in targets],
                    "target": targets[0] if len(targets) == 1 else None,
                }
            )
    return records


def stereo_target(pcm, layout):
    import numpy as np

    pcm = pcm.astype(np.float64)
    if layout == "center":
        return np.stack((pcm, pcm), axis=1)
    far = 0.55 * np.pad(pcm[:-3], (3, 0))
    return np.stack((pcm, far) if layout == "left" else (far, pcm), axis=1)


def render_pair(original, edited, room, layout):
    import numpy as np

    original, edited = (stereo_target(values, layout) for values in (original, edited))
    norm = np.linalg.norm(room.mean(axis=1))
    SOURCE.check(norm > 0, "Silent interference")
    room = room * (np.linalg.norm(original.mean(axis=1)) / norm) * 10**0.25
    gain = min(1.0, 32000 / max(np.max(abs(values + room)) for values in (original, edited)))
    background = np.rint(room * gain).astype("<i2")
    return [(np.rint(voice * gain).astype("<i2"), background) for voice in (original, edited)]


def recorded_mics(before, after, old_root, new_root):
    import numpy as np

    SOURCE.check(
        [(r["room"], r["start_sample"], r["url"].replace("-01.wav", "-05.wav")) for r in before]
        == [(r["room"], r["start_sample"], r["url"]) for r in after],
        "The recorded microphone clocks differ",
    )
    microphones = {}
    for a, b in zip(before, after, strict=True):
        data = [(root / f"{row['room']}.pcm").read_bytes() for root, row in ((old_root, a), (new_root, b))]
        SOURCE.check(
            all(
                len(raw) == 2 * SAMPLES and SOURCE.digest(raw) == row["pcm_sha256"]
                for raw, row in zip(data, (a, b), strict=True)
            ),
            "Recorded microphone PCM differs",
        )
        microphones[a["room"]] = np.column_stack([np.frombuffer(raw, "<i2") for raw in data])
    return microphones


def sine_stems(root, mics, plan):
    import numpy as np

    rooms = recorded_mics(plan["rooms"], SOURCE.read(mics / "receipt.json"), root / "background", mics)
    pending = {}
    for row, pcm in union.SINE.native_rows(root, {"pairs": plan["pairs"], "backgrounds": plan["rooms"]}):
        if row["condition"] != "clean":
            continue
        group = row["id"]
        if row["side"] == "original":
            pending[group] = pcm
            continue
        pair = plan["pairs"][group]
        layout = ("center", "left", "center", "right")[group % 4]
        rendered = render_pair(pending.pop(group), pcm, rooms[pair["room"]], layout)
        for role, (signal, noise) in zip(("untouched", "edited"), rendered, strict=True):
            mixed = signal.astype("<i4") + noise.astype("<i4")
            SOURCE.check(np.max(abs(mixed)) < 32768, "Matched stereo rendering clipped")
            entry = {
                "source": "sine",
                "cohort": "sine_" + ("center" if layout == "center" else "lateral"),
                "group": group,
                "scene": group,
                "role": role,
                "room": pair["room"],
                "layout": layout,
                "pcm_sha256": pcm_hash(mixed),
                "stems": [pcm_hash(signal), pcm_hash(noise)],
                "target": 0,
                "labels": [pair["truth"]] if role == "edited" else [],
            }
            yield entry, mixed.astype("<i2"), (signal, noise)
    for index, (name, room) in enumerate(rooms.items()):
        yield (
            {
                "source": "sine",
                "cohort": "recorded_rooms",
                "group": index,
                "scene": index,
                "role": "background",
                "room": name,
                "layout": None,
                "pcm_sha256": pcm_hash(room),
                "stems": [],
                "target": None,
                "labels": [],
            },
            room,
            (),
        )


def fixture(args, physical, sine):
    plan = SOURCE.read(args.physical / "freeze.json")
    SOURCE.check(plan == physical["plan"], "Physical stereo plan differs from the public receipt")
    SOURCE.check(SOURCE.read(args.sine / "freeze.json") == sine["frozen"], "SINE matched source plan differs")
    for entry in physical_meta(plan):
        stereo = voices.load_stereo(args.physical, entry["pcm_sha256"])
        stems = tuple(voices.load_stereo(args.physical, key) for key in entry["stems"])
        SOURCE.check((stems[0].astype("<i4") + stems[1] == stereo).all(), "Physical stem sum differs")
        yield entry, stereo, stems
    yield from sine_stems(args.sine, args.mics, sine["frozen"])


def scan(values, model, activity):
    from app.detectors.audio.calibration import logit

    counts = speech_counts(activity.spans(values))
    scores = []
    for (a, b), count in zip(GRID, counts, strict=True):
        scores.append(logit(model.predict([values[a:b]])[0]) if count >= RATE else None)
    return counts, scores


def from_current(row):
    positions = {(r["start_sample"], r["end_sample"]): r["logit"] for r in row["responses"]}
    return row["speech_counts"], [positions.get(span) for span in GRID]


def measure(args, phase):
    import numpy as np
    import torch

    from scripts.run_audio_specialists_v15 import load_voice

    physical, sine = sources()
    protocol = contract(sine)
    points = None
    if phase == "held":
        locked = SOURCE.read(args.calibration)
        SOURCE.check(
            locked["contract"] == protocol and locked["points"] == calibrate(locked["rows"]),
            "Calibration differs",
        )
        points = locked["points"]
        print("Loaded ordinary points before held model inference", flush=True)
    model = load_voice({"weights": args.weights, "assets": protocol["assets"]})
    torch.set_num_threads(4)
    activity = union.ROOM.runtime(args.weights / "silero_vad.jit")
    SOURCE.check(
        physical["current"]["assets"] == protocol["assets"], "The retained mono check used different weights"
    )
    archive = physical["current"]["rows"]
    cached = {row["pcm_sha256"]: from_current(row) for row in archive}
    results = []
    for entry, stereo, stems in fixture(args, physical, sine):
        if (entry["cohort"] == "development") != (phase == "development"):
            continue
        floating = stereo.astype(np.float64) / 32768
        weights = fit_filter(floating)
        filtered = project(floating, weights)
        target = acoustic_check(
            tuple(stem.astype(np.float64) / 32768 for stem in stems), weights, filtered, entry["target"]
        )
        safe_gain = min(1.0, (32767 / 32768) / max(float(np.max(abs(filtered))), 1e-8))
        views = (
            arithmetic_mono(stereo),
            floating[:, 0].astype(np.float32),
            floating[:, 1].astype(np.float32),
            (filtered * safe_gain).astype(np.float32),
        )
        observed = {}
        for name, values in zip(VIEWS, views, strict=True):
            if name == "mono" and entry["pcm_sha256"] in cached:
                counts, logits = cached[entry["pcm_sha256"]]
            else:
                counts, logits = scan(values, model, activity)
            observed[name] = {"digest": audio_hash(values), "speech_samples": counts, "logits": logits}
        results.append({"input": entry, "views": observed, "acoustics": target})
        if len(results) % 8 == 0:
            print(f"{phase}: measured {len(results)} recordings", flush=True)
    manifest = {"contract": protocol, "rows": results, "microphones": SOURCE.read(args.mics / "receipt.json")}
    manifest["points"] = points if points is not None else calibrate(results)
    SOURCE.write(args.output, manifest)


def readings(row):
    scores = {}
    for name in VIEWS:
        stats = row["views"][name]
        grid = list(zip(stats["speech_samples"], stats["logits"], strict=True))
        SOURCE.check(
            len(grid) == len(GRID)
            and all(
                type(count) is int
                and 0 <= count <= 2 * RATE
                and (
                    (value is None and count < RATE)
                    or (count >= RATE and value is not None and math.isfinite(value))
                )
                for count, value in grid
            ),
            "Speech admission differs",
        )
        scores[name] = stats["logits"]
    channels, agreement = [], []
    for left, right, residual in zip(scores["left"], scores["right"], scores["residual"], strict=True):
        available = [v for v in (left, right) if v is not None]
        channel = max(available) if available else None
        channels.append(channel)
        agreement.append(min(channel, residual) if channel is not None and residual is not None else None)
    return {
        "mono": scores["mono"],
        "channels": channels,
        "residual": scores["residual"],
        "agreement": agreement,
    }


def best(values):
    return max(((value, -i) for i, value in enumerate(values) if value is not None), default=None)


def calibrate(rows):
    groups = {arm: defaultdict(list) for arm in METHODS}
    SOURCE.check(
        rows and all(row["input"]["cohort"] == "development" and not row["input"]["labels"] for row in rows),
        "Ordinary calibration used an ineligible source",
    )
    for row in rows:
        for arm, values in readings(row).items():
            if selected := best(values):
                groups[arm][row["input"]["group"]].append(selected[0])
    output = {}
    for name, values in groups.items():
        maxima = sorted((max(group) for group in values.values()), reverse=True)
        SOURCE.check(
            len(maxima) == 3, "The three development speaker pairs must each have an ordinary response"
        )
        output[name] = {"groups": len(maxima), "strict": maxima[0], "wider": maxima[1]}
    return output


def useful(window, labels):
    return (
        [union.duration.compatible(window, truth) for truth in labels] if window else [False for _ in labels]
    )


def choose(value, limit):
    return GRID[-value[1]] if value is not None and value[0] > limit else None


def summarize(rows, points):
    results = defaultdict(Counter)
    for row in rows:
        entry = row["input"]
        scores = {arm: best(values) for arm, values in readings(row).items()}
        baseline = choose(scores["mono"], REVIEW_POINT)
        choices = {"mono/existing": (baseline, scores["mono"] is not None)}
        for arm, value in scores.items():
            for point in ("strict", "wider"):
                picked = choose(value, points[arm][point])
                choices[f"{arm}/{point}"] = (picked, value is not None)
                if arm != "mono":
                    choices[f"{arm}/{point}/fallback"] = (
                        baseline or picked,
                        value is not None or scores["mono"] is not None,
                    )
        for key, (window, assessed) in choices.items():
            stat = results[f"{entry['cohort']}/{key}"]
            labels = entry["labels"]
            exact = useful(window, labels)
            stat.update(
                edited=bool(labels),
                ordinary=not labels,
                unassessed=not assessed,
                useful_edits=any(exact),
                covered_labels=sum(exact),
                labeled_edits=len(labels),
                missed_edits=bool(labels) and not any(exact),
                wrong_range=bool(window and labels and not any(exact)),
                ordinary_flags=bool(window and not labels),
                gained=any(exact) and not any(useful(baseline, labels)),
                lost=any(useful(baseline, labels)) and not any(exact),
                added_ordinary=bool(window and not baseline and not labels),
                cleared_ordinary=bool(baseline and not window and not labels),
            )
    return {key: dict(counts) for key, counts in sorted(results.items())}


def review(packet):
    physical, sine = sources()
    study = contract(sine)
    ordinary, held = (packet[phase] for phase in ("development", "held"))
    SOURCE.check(
        ordinary["contract"] == held["contract"] == study and ordinary["microphones"] == held["microphones"],
        "Evaluation procedure or microphones differ",
    )
    points = calibrate(ordinary["rows"])
    SOURCE.check(points == ordinary["points"] == held["points"], "The ordinary points were changed")
    native = [row["input"] for row in ordinary["rows"] + held["rows"] if row["input"]["source"] == "physical"]
    SOURCE.check(native == physical_meta(physical["plan"]), "A physical test label or input differs")
    records = [row["input"] for row in held["rows"] if row["input"]["source"] == "sine"]
    expected = [
        (pair["id"], role, [pair["truth"]] if role == "edited" else [], pair["room"])
        for pair in sine["frozen"]["pairs"]
        for role in ("untouched", "edited")
    ]
    expected.extend((i, "background", [], row["room"]) for i, row in enumerate(sine["frozen"]["rooms"]))
    SOURCE.check(
        expected == [(row["group"], row["role"], row["labels"], row["room"]) for row in records],
        "A matched SINE label or room differs",
    )
    digests = [row["input"]["pcm_sha256"] for row in ordinary["rows"] + held["rows"]]
    SOURCE.check(len(digests) == len(set(digests)), "Delivered PCM is repeated across controls or roles")
    decisions = summarize(held["rows"], points)
    SOURCE.check(packet.get("results", decisions) == decisions, "Held results differ from the scored windows")
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for phase in ("development", "held"):
        mode = commands.add_parser(phase)
        for name in ("physical", "sine", "mics", "weights", "output"):
            mode.add_argument(name, type=Path)
        if phase == "held":
            mode.add_argument("calibration", type=Path)
    combine = commands.add_parser("combine")
    for name in ("development", "held", "output"):
        combine.add_argument(name, type=Path)
    verify = commands.add_parser("review")
    verify.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.mode in ("development", "held"):
        measure(args, args.mode)
    elif args.mode == "combine":
        packet = {name: SOURCE.read(getattr(args, name)) for name in ("development", "held")}
        packet["results"] = review(packet)
        data = (json.dumps(packet, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        args.output.write_bytes(gzip.compress(data, mtime=0))
    else:
        print(
            json.dumps(
                review(union.duration.read_gz(args.receipt)), indent=2, sort_keys=True, allow_nan=False
            )
        )


if __name__ == "__main__":
    main()
