"""Test both blind stereo outputs without assigning the speaker of interest."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import audio_blind_spatial_transfer as blind
from scripts.audio_blind_spatial_voice import WINDOWS, useful

SCHEMA = "chipmunk.audio-unassigned-stereo-voice/1"
ARMS = ("original", "balanced", "either", "both")
PILOT = Path("docs/evaluation-panels/audio-measured-spatial-transfer-2026-11-12.json")
TRANSFER = Path("docs/evaluation-panels/audio-blind-huddersfield-transfer-2026-11-12.json")
LOCATIONS = ("bbc", "balanced_target", "lateral_target", "second_competitor")
POLICY = {
    "case_roles": "Surrey uses genuine views only to set the points. BBC and the three Huddersfield layouts are reported separately. The exact audio was used in prior studies.",
    "arms": "Use the original stereo mean, the balance-routed output, the higher score across either anonymous output, or the lower score when both outputs score. Score both without speaker labels. Each output must pass the speech check and be unclipped.",
    "windows": "Use the original two-second grid. Each fixed point is the highest eligible genuine Surrey file response, with strict excess. Prefer the earlier interval at a tie.",
    "locations": "A useful range covers half the edited time and reaches at least 0.30 temporal IoU. Preserve all original findings in the additive comparison. A source group flags when any ordinary version flags. Report all missed and unassessed files.",
    "review": "For each layout, compare extra and lost ranges, added ordinary groups, background-only flags, and equal-false-group budgets. No variant serves or enters confidence on this reused panel.",
}


def bound_rosters(pilot, room):
    earlier = blind.read(PILOT)["commitments"]
    later = blind.read(TRANSFER)
    pilot_roster = pilot / "roster.json"
    room_roster = room / "roster.json"
    pilot_lock = pilot / "freeze.json"
    room_lock = room / "freeze.json"
    for path, expected in ((pilot_roster, earlier["roster"]), (pilot_lock, earlier["freeze"]),
                           (room_roster, later["roster_sha256"]), (room_lock, later["freeze_sha256"])):
        blind.require(blind.sha(path.read_bytes()) == expected, "Retained acoustic panel does not match publication")
    sources = [
        (pilot, {"selection": "surrey", "held": "bbc"}[row["panel"]], row["arm"], row)
        for row in blind.read(pilot_roster)
    ] + [(room, row["condition"], row["kind"], row) for row in blind.read(room_roster)]
    blind.require(Counter(location for _, location, _, _ in sources) ==
                  {"surrey": 26, "bbc": 26, **dict.fromkeys(LOCATIONS[1:], 36)}, "Unexpected source counts")
    return {
        "schema": SCHEMA,
        "policy": POLICY,
        "runner_sha256": blind.sha(Path(__file__).read_bytes()),
        "separator_sha256": blind.sha(Path(blind.__file__).read_bytes()),
        "sources": {key: expected for key, expected in (("pilot", earlier["roster"]), ("transfer", later["roster_sha256"]))},
        "assets": blind.read(pilot_lock)["assets"],
        "windows": [list(window) for window in WINDOWS],
    }, sources


def waveform_views(folder, source, *, alternate=False):
    transforms = {"fft": np.fft.rfft, "inverse": np.fft.irfft} if alternate else {}
    original = blind.load_pcm(folder, source, "original").astype("<i2")
    stereo = blind.load_pcm(folder, source, "stereo")
    blind.require(np.array_equal(original, np.rint(stereo.mean(1))), "Original downmix differs")
    separated = blind.separate(stereo, **transforms)
    if separated is None:
        return (original, None, None), None
    candidates, route, _ = separated
    rounded = [np.rint(channel) for channel in candidates.T]
    return (original, *(channel.astype("<i2") if np.all((channel >= -32768) & (channel <= 32767)) else None
                        for channel in rounded)), route


def score_rows(sources, assets, weights, output, frozen):
    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.run_audio_specialists_v15 import load_voice

    network_free()
    model = load_voice({"weights": str(weights), "assets": assets})
    speech = SpeechActivity(weights / "silero_vad.jit")
    cache = {}
    calls = 0
    started = time.monotonic()
    with output.open("x") as stream:
        stream.write(frozen + "\n")
        for index, (folder, location, kind, source) in enumerate(sources):
            signals, route = waveform_views(folder, source)
            row = metadata(index, location, kind, source, route)
            for integers in signals:
                if integers is None:
                    row["reads"].append(None)
                    continue
                pcm = integers.astype(np.float32) / 32768
                spans = speech.spans(pcm)
                entries = []
                for a, b in WINDOWS:
                    sha = blind.sha(integers[a:b].tobytes())
                    response = None
                    if window_in_speech(spans, a / blind.RATE, b / blind.RATE):
                        if sha not in cache:
                            (probability,) = model.predict([pcm[a:b]])
                            blind.require(math.isfinite(probability) and 0 <= probability <= 1, "Invalid voice output")
                            cache[sha] = float(logit(probability))
                            calls += 1
                        response = cache[sha]
                    entries.append([sha, response])
                row["reads"].append(entries)
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            if (index + 1) % 12 == 0 or index + 1 == len(sources):
                print(location, index + 1, len(sources), "cases,", calls, "voice forwards,",
                      round(time.monotonic() - started, 1), "seconds", flush=True)


def metadata(index, location, kind, source, route):
    return {"case": index, "location": location, "group": source["group"], "kind": kind,
            "family": source["family"] if source["truth"] is not None else None,
            "truth": source["truth"], "route": route, "reads": []}


def best(row, *, shared=False):
    scores = [[entry[1] for entry in view] if view is not None else [None] * len(WINDOWS) for view in row["reads"]]
    original, left, right = scores
    balanced = scores[1 + row["route"]] if row["route"] is not None else [None] * len(WINDOWS)
    channels = list(zip(left, right, strict=True))
    either = [max((value for value in pair if value is not None), default=None) for pair in channels]
    both = [min(pair) if None not in pair else None for pair in channels]
    output = {}
    for arm, values in zip(ARMS, (original, balanced, either, both), strict=True):
        eligible = [(value, -i) for i, value in enumerate(values)
                    if value is not None and (not shared or original[i] is not None and either[i] is not None)]
        score, index = max(eligible, default=(None, 0))
        output[arm] = (score, WINDOWS[-index] if score is not None else None)
    return output


def outcomes(rows, points, *, shared=False):
    by_case = {row["case"]: best(row, shared=shared) for row in rows}
    output = {}
    for arm in ARMS:
        flagged, located, false_groups, background = set(), set(), set(), set()
        assessed = wrong_location = false_clips = 0
        for row in rows:
            response, window = by_case[row["case"]][arm]
            if response is None:
                continue
            assessed += 1
            if response <= points[arm]:
                continue
            if row["truth"] is not None:
                flagged.add(row["case"])
                if useful(row["truth"], window):
                    located.add(row["case"])
                else:
                    wrong_location += 1
            else:
                false_clips += 1
                false_groups.add(row["group"])
                if row["kind"] == "background_only":
                    background.add(row["group"])
        output[arm] = {"assessed": assessed, "unassessed": len(rows) - assessed, "AI flags": len(flagged),
                       "useful_locations": len(located), "wrong_locations": wrong_location, "false_clips": false_clips,
                       "false_groups": len(false_groups), "background_flags": len(background),
                       "inconclusive": assessed - len(flagged) - false_clips,
                       "_locations": located, "_false": false_groups, "_background": background}
    incumbent = output["original"]
    for arm, summary in output.items():
        additions = summary["_locations"] - incumbent["_locations"]
        summary["new_locations"] = len(additions)
        summary["new_families"] = sorted({row["family"] for row in rows if row["case"] in additions})
        summary["lost_locations"] = len(incumbent["_locations"] - summary["_locations"])
        summary["new_false_groups"] = len(summary["_false"] - incumbent["_false"])
        summary["new_background_flags"] = len(summary["_background"] - incumbent["_background"])
    return {arm: {key: value for key, value in summary.items() if not key.startswith("_")}
            for arm, summary in output.items()}


def summarize(rows):
    selection = [row for row in rows if row["location"] == "surrey" and row["truth"] is None]
    result = {}
    for shared in (False, True):
        readings = [best(row, shared=shared) for row in selection]
        points = {arm: max(value for response in readings if (value := response[arm][0]) is not None) for arm in ARMS}
        trials = {}
        for location in LOCATIONS:
            cases = [row for row in rows if row["location"] == location]
            ceilings = {arm: {} for arm in ARMS}
            for row in cases:
                if row["truth"] is not None:
                    continue
                for arm, (score, _) in best(row, shared=shared).items():
                    if score is not None:
                        ceilings[arm][row["group"]] = max(score, ceilings[arm].get(row["group"], -math.inf))
            ranked = {arm: sorted(groups.values(), reverse=True) for arm, groups in ceilings.items()}
            trials[location] = {
                "edited": sum(row["truth"] is not None for row in cases),
                "ordinary": sum(row["truth"] is None for row in cases),
                "fixed": outcomes(cases, points, shared=shared),
                "equal_false_group": {
                    str(budget): {arm: value["useful_locations"] for arm, value in
                                  outcomes(cases, {arm: scores[budget] if len(scores) > budget else -math.inf
                                                   for arm, scores in ranked.items()}, shared=shared).items()}
                    for budget in (0, 1, 2)
                },
            }
        result["shared" if shared else "all_admitted"] = {"points": points, "trials": trials}
    return result


def audit(rows, sources, weights, assets):
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.upgraded import SpeechActivity

    blind.require(blind.sha((weights / "silero_vad.jit").read_bytes()) == assets["silero_vad.jit"], "Speech weights differ")
    speech = SpeechActivity(weights / "silero_vad.jit")
    admitted = set()
    windows = 0
    for index, (row, (folder, location, kind, source)) in enumerate(zip(rows, sources, strict=True)):
        signals, route = waveform_views(folder, source, alternate=True)
        blind.require({**row, "reads": []} == metadata(index, location, kind, source, route), "Source or route differs")
        for entries, integers in zip(row["reads"], signals, strict=True):
            blind.require((entries is None) == (integers is None), "Channel assessment differs")
            if entries is None:
                continue
            spans = speech.spans(integers.astype(np.float32) / 32768)
            for (sha, score), (a, b) in zip(entries, WINDOWS, strict=True):
                accepted = window_in_speech(spans, a / blind.RATE, b / blind.RATE)
                blind.require(sha == blind.sha(integers[a:b].tobytes()) and accepted == (score is not None)
                              and (score is None or math.isfinite(score)), "Exact wave or speech admission differs")
                windows += 1
                if accepted:
                    admitted.add(sha)
    return {"verified_windows": windows, "distinct_admitted_windows": len(admitted), "audited_cases": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "audit"))
    parser.add_argument("pilot", type=Path)
    parser.add_argument("room", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("models/weights/audio-upgrade"))
    args = parser.parse_args()
    lock, sources = bound_rosters(args.pilot, args.room)
    freeze = args.output / "freeze.json"
    ledger = args.output / "private-windows.jsonl"
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        blind.write(freeze, lock)
        return
    blind.require(lock == blind.read(freeze), "Method or source changed after freeze")
    frozen = blind.sha(freeze.read_bytes())
    if args.phase == "score":
        score_rows(sources, lock["assets"], args.weights, ledger, frozen)
    else:
        digest, *saved = ledger.read_text().splitlines()
        blind.require(digest == frozen, "The score ledger uses a different freeze")
        rows = [json.loads(row) for row in saved]
        verified = audit(rows, sources, args.weights, lock["assets"])
        report = {**lock, "freeze_sha256": frozen, "private_ledger_sha256": blind.sha(ledger.read_bytes()),
                  "summary": summarize(rows), "native_replay": verified}
        blind.write(args.output / "result.json", report)
        print(json.dumps(verified), flush=True)


if __name__ == "__main__":
    main()
