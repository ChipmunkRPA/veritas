# SPDX-License-Identifier: Apache-2.0
"""Measure speech-window admission with continuous instead of reset activity state."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.evidence import audio_long_review_binding as binding
from scripts import audio_focused_window_transfer as focused
from scripts import audio_vocal_event_mixtures as io
from scripts import review_audio_reduced_background_localization as clock

PARENT = io.PUBLIC / "audio-focused-window-transfer-2026-11-12.json"
SCHEMA = "chipmunk.audio-continuous-activity-transfer/1"
METHOD = {
    "activity": "Compare a reset activity pass per window with one continuous pass per recording.",
    "admission": "Keep the existing half-speech minimum. The additive arm rescues only windows the reset pass skipped.",
    "classifier": "Score newly admitted original samples with the unchanged three-second classifier and calibration.",
    "panels": "Keep the 98 control and edited deliveries and all focused windows from the exact parent roster.",
}


def header():
    return {"schema": SCHEMA, "method": METHOD, "parent_sha256": io.sha(PARENT)}


def samples_of_speech(spans, start, end):
    return sum(max(0, min(end, b) - max(start, a)) for a, b in spans)


def sample_spans(activity, pcm):
    return [(round(start * binding.RATE), round(end * binding.RATE)) for start, end in activity.spans(pcm)]


def predict(voice, pcm):
    from app.detectors.audio import calibration

    score, = voice.predict([pcm])
    result = calibration.evaluate(float(score), pcm, [(0.0, pcm)], voice, NO_DEADLINE)
    focused.require(result["scope_supported"], "Original three-second calibration scope changed")
    return result["decision"]


class _Unbounded:
    @staticmethod
    def check(_):
        pass


NO_DEADLINE = _Unbounded()


def native_folder(folder, parent):
    focused.require(io.sha(folder / "roster.json") == parent["freeze"]["roster_sha256"],
                    "Parent delivery roster changed")


def measure(row, previous, raw, activity, voice=None):
    from scripts.audio_gtcrn_traffic import float_pcm

    pcm = float_pcm(raw)
    continuous = sample_spans(activity, pcm)
    windows = []
    for before in previous["windows"]:
        start, end = before["span"]
        clip = pcm[start:end]
        reset = samples_of_speech(sample_spans(activity, clip), 0, end - start)
        extended = samples_of_speech(continuous, start, end)
        recovered = (predict(voice, clip) if voice is not None
                     and reset * 2 < end - start <= extended * 2 else None)
        windows.append({"span": [start, end], "reset_samples": reset, "continuous_samples": extended,
                        "recovered_direction": recovered})
    return {"case": row["case"], "windows": windows}


def regions(samples, directions):
    readings = [{"start_sample": a, "end_sample": b, "direction": direction}
                for (a, b), direction in directions.items()]
    return [[r["start_sample"], r["end_sample"]] for r in binding.focused_regions(samples, readings)]


def compare(parent, observed):
    results = defaultdict(Counter)
    arms = ("baseline", "continued", "additive")
    for row, previous, record in zip(parent["roster"], parent["responses"], observed, strict=True):
        focused.require(row["case"] == record["case"] and [w["span"] for w in record["windows"]]
                        == [w["span"] for w in previous["windows"]], "Recorded window differs from the frozen schedule")
        meta = row["source"]
        label, truth = (meta["label"], meta["edits"]) if meta else ("isolated", [])
        cell = results[f"{row['scene']}/{label}"]
        cell["files"] += 1
        lookup = {arm: {} for arm in arms}
        for before, window in zip(previous["windows"], record["windows"], strict=True):
            a, b = window["span"]
            reset, continuous, recovery = (window[key] for key in ("reset_samples", "continuous_samples", "recovered_direction"))
            focused.require(type(reset) is type(continuous) is int and 0 <= reset <= b-a and 0 <= continuous <= b-a,
                            "Invalid speech counts")
            was_admitted, is_admitted = reset * 2 >= b-a, continuous * 2 >= b-a
            direction = before["direction"]
            focused.require(was_admitted == (direction != "unassessed"), "Reset activity differs from the measured baseline")
            focused.require(recovery in binding.DIRECTIONS - {"unassessed"} if is_admitted and not was_admitted
                            else recovery is None, "Unexpected new score")
            additive = recovery or direction
            continued = additive if is_admitted else "unassessed"
            for key, state in zip(arms, (direction, continued, additive), strict=True):
                lookup[key][a, b] = state
            cell.update(windows=1, previous_admitted=was_admitted, continuous_admitted=is_admitted,
                        new_windows=is_admitted and not was_admitted, lost_windows=was_admitted and not is_admitted,
                        recovered_ai=recovery == "ai", recovered_original=recovery == "original",
                        recovered_inconclusive=recovery == "inconclusive")
        old = regions(row["samples"], lookup["baseline"])
        found = clock.useful(old, truth)
        dense = [list(span) for span in binding.dense_intervals(row["samples"])]
        cell["dense_geometry_reachable"] += clock.useful(clock.intersections(dense, set(map(tuple, dense))), truth)
        for arm in arms:
            proposals = regions(row["samples"], lookup[arm])
            useful = clock.useful(proposals, truth)
            admitted = {span for span, state in lookup[arm].items() if state != "unassessed"}
            cell.update({f"{arm}_flags": bool(proposals), f"{arm}_useful": useful,
                         f"{arm}_gained": useful and not found, f"{arm}_lost": found and not useful,
                         f"{arm}_reachable_with_admission": clock.useful(clock.intersections(dense, admitted), truth),
                         f"{arm}_files_assessed": any(state != "unassessed" for state in lookup[arm].values())})
            cell[f"{arm}_genuine_samples_marked"] += sum(
                b-a-sum(clock.overlap((a, b), edit) for edit in truth) for a, b in clock.intervals_union(proposals)
            )
    return {name: dict(values) for name, values in results.items()}


def run(folder, output, parent):
    from scripts.audio_reduced_background_localization import assets_for_scores

    native_folder(folder, parent)
    output.mkdir(mode=0o700)
    lock = header()
    io.save(output / "freeze.json", lock)
    voice, activity = assets_for_scores(parent["freeze"])
    records = []
    for row, previous in zip(parent["roster"], parent["responses"], strict=True):
        raw = focused.native_wave(folder, row)
        observation = measure(row, previous, raw, activity, voice)
        records.append(observation)
        if len(records) % 10 == 0:
            print(f"Checked {len(records)} of {len(parent['roster'])} recordings", flush=True)
    io.save(output / "receipt.json", lock | {"recordings": records, "results": compare(parent, records)})


def review(receipt, parent, native=None):
    completion = io.read(receipt)
    focused.require(all(completion[key] == value for key, value in header().items()), "Study commitments differ")
    results = compare(parent, completion["recordings"])
    focused.require(results == completion["results"], "Recorded results differ")
    checked = 0
    if native:
        from app.detectors.audio.upgraded import SpeechActivity
        from scripts.audio_vocal_event_mixtures import WEIGHTS

        native_folder(native, parent)
        path = WEIGHTS / "silero_vad.jit"
        focused.require(io.sha(path) == parent["freeze"]["voice_assets"][path.name], "Activity asset differs")
        activity = SpeechActivity(path)
        for row, previous, record in zip(parent["roster"], parent["responses"], completion["recordings"], strict=True):
            raw = focused.native_wave(native, row)
            repeated = measure(row, previous, raw, activity)
            expected = [{"recovered_direction": None, **{k: w[k] for k in ("span", "reset_samples", "continuous_samples")}}
                        for w in record["windows"]]
            focused.require(repeated["windows"] == expected, "Original sample coverage differs")
            checked += len(expected)
    print(json.dumps({"results": results, "native_windows_verified": checked}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--native", type=Path)
    args = parser.parse_args()
    parent = io.read(PARENT)
    if args.action == "run":
        if args.native is None:
            parser.error("The run requires the exact parent PCM bank")
        run(args.native, args.path, parent)
    else:
        review(args.path, parent, args.native)


if __name__ == "__main__":
    main()
