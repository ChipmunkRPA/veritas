"""Compare frozen voice decisions after blind stereo separation on a reused panel."""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import audio_blind_spatial_transfer as acoustic

ARMS = ("original", "blind")
WINDOWS = ((16000, 48000), (48000, 80000))
RULE = {
    "scope": "Offline on the 52 already exposed stereo deliveries used for the acoustic preflight. A gain requires a later fresh test.",
    "admission": "Keep only windows that pass the incumbent speech gate on both arms. Original-only coverage remains visible. No source labels in routing.",
    "points": "Use each arm's strict maximum on all assessed genuine selection views. Retain the parent original point as a separate descriptive comparator.",
    "location": "Use the best admitted two-second score. Match the known edit with at least 0.30 temporal IoU and half the edit covered.",
    "advance": "At least two net additional useful held edits in different generator families and distinct groups, at most one additional held false genuine group, and no new background-only flags. Also report the historical original-track comparison point and retrospective equal-false-group operating points separately.",
}


def wav_view(parent, source, row):
    signal = acoustic.load_pcm(parent, source, "stereo")
    result = acoustic.separate(signal)
    if result is None or result[1] is None:
        acoustic.require(row["route"] is None, "Blind routing did not replay")
        return None
    candidates, index, _ = result
    acoustic.require(index == row["route"], "Blind routing did not replay")
    view = np.rint(candidates[:, index])
    if not np.all((view >= -32768) & (view <= 32767)):
        return None
    view = view.astype("<i2")
    acoustic.require(acoustic.sha(view.tobytes()) == row["results"][f"output_{index}"]["sha256"],
                     "Selected waveform does not match the acoustic receipt")
    return view


def score(args, parent_rows, measure, lock, parent, frozen):
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts import audio_measured_spatial_transfer as parent_trial
    from scripts.run_audio_specialists_v15 import load_voice

    parent_trial.OLD.old.spectra.incumbent.network_free()
    assets = parent["assets"]
    acoustic.require(acoustic.sha((args.weights / "silero_vad.jit").read_bytes()) == assets["silero_vad.jit"],
                     "Speech gate bytes differ")
    model = load_voice({"weights": str(args.weights), "assets": assets})
    gate = SpeechActivity(args.weights / "silero_vad.jit")
    began = time.monotonic()
    points, ledgers = None, {}
    for role in ("selection", "held"):
        rows = []
        calls = Counter()
        for source, receipt in zip(parent_rows, measure, strict=True):
            if source["panel"] != role:
                continue
            original = acoustic.load_pcm(args.parent, source, "original").astype("<i2")
            selected = wav_view(args.parent, source, receipt)
            inputs = {"original": original, "blind": selected}
            arms = {key: [] for key in ARMS}
            for key, integers in inputs.items():
                if integers is None:
                    arms[key] = [None] * len(WINDOWS)
                    continue
                audio = integers.astype(np.float32) / 32768
                speech = gate.spans(audio)
                for a, b in WINDOWS:
                    response = None
                    if parent_trial.OLD.old.window_in_speech(speech, a / acoustic.RATE, b / acoustic.RATE):
                        (probability,) = model.predict([audio[a:b]])
                        acoustic.require(math.isfinite(probability) and 0 <= probability <= 1, "Invalid classifier score")
                        response = float(parent_trial.OLD.old.calibration.logit(probability))
                        calls[key] += 1
                    arms[key].append({"sha256": acoustic.sha(integers[a:b].tobytes()), "score": response})
            rows.append({"case": receipt["case"], "group": source["group"], "kind": source["arm"],
                         "family": source["family"], "truth": source["truth"], "arms": arms})
            if len(rows) % 6 == 0:
                print(role, len(rows), "cases,", sum(calls.values()), "voice forwards", flush=True)
        ledgers[role] = {"rows": rows, "calls": dict(calls), "freeze_sha256": frozen}
        acoustic.write(args.run / f"voice-{role}.json", ledgers[role])
        if role == "selection":
            points = fit(rows, parent["original_point"])
            acoustic.write(args.run / "voice-points.json", points)
    result = readout(ledgers, points)
    result["execution"] = {"seconds": round(time.monotonic() - began, 3),
                           "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           "platform": sys.platform}
    result["commitments"] = {"acoustic": lock["acoustic_sha256"], "frozen_voice": frozen}
    acoustic.write(args.run / "voice-result.json", result)


def responses(row, *, shared=True):
    available = {key: [entry["score"] if entry is not None else None for entry in row["arms"][key]] for key in ARMS}
    response = {}
    for key in ARMS:
        values = [(score, i) for i, score in enumerate(available[key]) if score is not None
                  and (not shared or all(available[arm][i] is not None for arm in ARMS))]
        best = max(values, key=lambda value: (value[0], -value[1]), default=None)
        response[key] = (best[0], WINDOWS[best[1]]) if best else (None, None)
    return response


def fit(rows, parent_point):
    controls = [row for row in rows if row["truth"] is None]
    shared = [(row["group"], responses(row)) for row in controls]
    eligible = {group for group, response in shared if response["original"][0] is not None}
    acoustic.require(len(eligible) >= 5, "Too few assessed selection groups")
    return {"matched": {arm: max(response[arm][0] for _, response in shared if response[arm][0] is not None)
                        for arm in ARMS}, "historical_original": parent_point, "genuine_groups": len(eligible)}


def useful(truth, window):
    if not truth or not window:
        return False
    a, b = truth
    c, d = window
    overlap = max(0, min(b, d) - max(a, c))
    return 10 * overlap >= 3 * (max(b, d) - min(a, c)) and 2 * overlap >= b - a


def evaluate(rows, arm, point, *, shared=True):
    flags, located, false = set(), set(), set()
    false_cases, background, assessed, inconclusive = 0, 0, 0, 0
    for row in rows:
        score, window = responses(row, shared=shared)[arm]
        if score is None:
            continue
        assessed += 1
        if score <= point:
            inconclusive += 1
        elif row["truth"] is None:
            false.add(row["group"])
            false_cases += 1
            background += row["kind"] == "background_only"
        else:
            flags.add(row["case"])
            if useful(row["truth"], window):
                located.add(row["case"])
    return {"assessed": assessed, "inconclusive": inconclusive, "unassessed": len(rows) - assessed, "ai_flags": len(flags),
            "located": sorted(located), "false_groups": len(false), "false_cases": false_cases, "background_flags": background}


def equal_false_group(rows, arm, budget):
    groups = {}
    for row in rows:
        score = responses(row)[arm][0]
        if row["truth"] is None and score is not None:
            groups[row["group"]] = max(score, groups.get(row["group"], -math.inf))
    ceilings = sorted(groups.values(), reverse=True)
    return evaluate(rows, arm, ceilings[budget] if budget < len(ceilings) else -math.inf)


def readout(ledgers, points):
    roles = {}
    for role, ledger in ledgers.items():
        rows = ledger["rows"]
        acoustic.require(len(rows) == 26 and sum(row["truth"] is not None for row in rows) == 6, "Unexpected roster")
        fixed = {arm: evaluate(rows, arm, points["matched"][arm]) for arm in ARMS}
        added = set(fixed["blind"]["located"]) - set(fixed["original"]["located"])
        lost = set(fixed["original"]["located"]) - set(fixed["blind"]["located"])
        families = {row["family"] for row in rows if row["case"] in added}
        groups = {row["group"] for row in rows if row["case"] in added}
        roles[role] = {
            "fixed": fixed, "added_locations": len(added), "lost_locations": len(lost),
            "added_families": sorted(families),
            "historical_original": evaluate(rows, "original", points["historical_original"], shared=False),
            "original_full_coverage": evaluate(rows, "original", points["matched"]["original"], shared=False),
            "equal_false_group": {str(budget): {arm: equal_false_group(rows, arm, budget) for arm in ARMS}
                                  for budget in (0, 1, 2)},
            "calls": ledger["calls"],
            "advance": len(added) - len(lost) >= 2 and len(families) >= 2 and len(groups) >= 2
            and fixed["blind"]["false_groups"] <= fixed["original"]["false_groups"] + 1
            and fixed["blind"]["background_flags"] <= fixed["original"]["background_flags"],
        }
    return {"schema": "chipmunk.audio-blind-spatial-voice/1", "method": RULE, "roles": roles, "selection_points": points}


def audit(args, source_rows, measured, frozen, parent):
    ledgers = {role: acoustic.read(args.run / f"voice-{role}.json") for role in ("selection", "held")}
    local_result = acoustic.read(args.run / "voice-result.json")
    for role, ledger in ledgers.items():
        acoustic.require(ledger["freeze_sha256"] == frozen, "Voice ledger uses a different freeze")
        calls = Counter()
        cases = [(row, exact) for row, exact in zip(source_rows, measured, strict=True) if row["panel"] == role]
        for row, (source, acoustic_row) in zip(ledger["rows"], cases, strict=True):
            acoustic.require(
                (row["case"], row["group"], row["kind"], row["family"], row["truth"])
                == (acoustic_row["case"], source["group"], source["arm"], source["family"], source["truth"]),
                "Voice ledger case binding differs",
            )
            original = acoustic.load_pcm(args.parent, source, "original").astype("<i2")
            blind = wav_view(args.parent, source, acoustic_row)
            for key, signal in (("original", original), ("blind", blind)):
                for entry, (a, b) in zip(row["arms"][key], WINDOWS, strict=True):
                    if entry is None:
                        acoustic.require(signal is None, "An available waveform has no record")
                        continue
                    acoustic.require(signal is not None and entry["sha256"] == acoustic.sha(signal[a:b].tobytes()),
                                     "Voice window does not replay")
                    if entry["score"] is not None:
                        acoustic.require(math.isfinite(entry["score"]), "Invalid saved voice response")
                        calls[key] += 1
        acoustic.require(dict(calls) == ledger["calls"], "Voice forward count differs")
    points = fit(ledgers["selection"]["rows"], parent["original_point"])
    acoustic.require(points == acoustic.read(args.run / "voice-points.json"), "Selection points do not replay")
    result = readout(ledgers, points)
    acoustic.require(all(local_result[key] == value for key, value in result.items()), "Voice outcomes do not replay")
    print(json.dumps(result["roles"]["held"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "audit"))
    parser.add_argument("parent", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("models/weights/audio-upgrade"))
    args = parser.parse_args()
    acoustic_lock, roster = acoustic.binding(args.parent)
    acoustic_path = args.run / "result.json"
    measured = acoustic.read(acoustic_path)
    source = acoustic.read(args.parent / "freeze.json")
    lock_path = args.run / "voice-freeze.json"
    acoustic.require(acoustic.read(args.run / "freeze.json") == acoustic_lock
                     and all(row["advance"] for row in measured["summary"].values()), "Acoustic stage did not qualify")
    lock = {"method": RULE, "acoustic_sha256": acoustic.sha(acoustic_path.read_bytes()),
            "source_freeze_sha256": acoustic.sha((args.parent / "freeze.json").read_bytes()),
            "runner_sha256": acoustic.sha(Path(__file__).read_bytes()), "assets": source["assets"],
            "windows": [list(window) for window in WINDOWS]}
    if args.phase == "freeze":
        acoustic.write(lock_path, lock)
        return
    acoustic.require(acoustic.read(lock_path) == lock, "Voice freeze differs")
    frozen = acoustic.sha(lock_path.read_bytes())
    if args.phase == "score":
        score(args, roster, measured["rows"], lock, source, frozen)
    else:
        audit(args, roster, measured["rows"], frozen, source)


if __name__ == "__main__":
    main()
