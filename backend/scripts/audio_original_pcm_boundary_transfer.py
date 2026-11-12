"""Compare exact original PCM detection with an independent paired-boundary cue."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import audio_spectra_long_room_holdout as old

RATE, ROOT, BASE = old.RATE, old.ROOT, old.BASE
SAME = BASE / "audio-same-speaker-enrollment-2026-11-12"
ALT = BASE / "audio-spectra-long-room-final-2026-11-12"
ARMS = ("current", "boundary", "combined")
SCHEMA = "chipmunk.audio-original-pcm-boundary-transfer/1"
FRAME, HOP, EDIT, STEP, CONTEXT, GAP = 400, 160, 12800, 1280, 2560, 320
read, write, sha, digest, require = old.read, old.write, old.sha, old.digest, old.require
METHOD = {
    "roles": "Use exact same-speaker clone mixtures. Set points from 12 genuine controls from six selection voices, confirm on eight other clone hosts and their two ordinary counterparts, and diagnose the six point-role clone inserts only after freezing points. Previously scored held expressive and stock trios and two isolated meeting cuts form a separate exposed transfer panel, with existing exact current model scores reused by input hash.",
    "current": "Use the already approved current voice checkpoint and original mono PCM. Score native 2 s spans starting at seconds 1, 3 and 5 when present. Apply ordinary speech activity admission. Take the maximum eligible raw log-odds and preserve the exact native range. Do not apply out-of-scope customer confidence calibration.",
    "boundary": "Twenty-four fixed mel-frequency bands from 80 to 7200 Hz. Mean Hann-windowed power from 25 ms frames with 10 ms hops across 160 ms left and right contexts around each boundary, omitting the nearest 20 ms on each side. Compare normalized band distributions with Jensen-Shannon divergence. At fixed 80 ms steps, use the lower of both endpoint divergences of an 800 ms candidate. No model or waveform replacement.",
    "fusion": "For a boundary range, pair its score with the highest admitted current log-odds from a native 2 s window covering the entire candidate. Standardize both against the median and interquartile range of respective candidates from genuine point inputs with a minimum scale of 1e-6. Score the mean of standardized values and retain the candidate range. Compare each arm alone and together.",
    "points": "Fixed points strictly exceed each arm's highest genuine point-control response. Report matched retrospective false-group budgets 0, 1 and 2 separately. A useful range needs 0.30 temporal intersection over union and 0.50 edit coverage. Flagging an ordinary control counts as a false source group, including ordinary splices. No room/generator transfer or serving claim follows from a descriptive threshold sweep.",
}


def source_rows():
    same, alt = (read(folder / "result.json") for folder in (SAME, ALT))
    require(all(sha(folder / "roster.json") == result["commitments"]["roster"] for folder, result in ((SAME, same), (ALT, alt))),
            "A prior source roster changed")
    require(alt["commitments"]["confirmation-current"] == sha(ALT / "confirmation-current.json")
            and read(ALT / "freeze.json")["current_assets"] == old.voice.ASSETS, "Exposed comparison checkpoint or ledger changed")
    rows = []
    for row in read(SAME / "roster.json"):
        role = row["role"]
        panel = "clone_hold" if role == "confirmation" else "point_positive" if row["truth"] else "point"
        rows.append({"id": row["id"], "panel": panel, "group": row["group"], "arm": row["arm"], "family": "clone",
                     "room": row["background"]["meeting"], "samples": 96000, "truth": row["truth"], "edit": row["edit"],
                     "pcm_sha256": row["original_pcm_sha256"], "asset": "same"})
    prior = read(ALT / "roster.json")
    groups = {row["group"]: row["family"] for row in prior if row["role"] == "confirmation" and row["truth"] is not None and row["family"] in {"stock", "expressive"}}
    for row in prior:
        if row["role"] == "confirmation" and (row["group"] in groups or row["arm"] == "background_only"):
            rows.append({"id": row["id"], "panel": "exposed_transfer", "group": "alt-"+row["group"], "arm": row["arm"],
                         "family": groups.get(row["group"]), "room": row["background"]["room"], "samples": row["background"]["samples"],
                         "truth": row["truth"], "edit": row.get("edit_range"), "pcm_sha256": row["pcm_sha256"], "asset": "alt"})
    require(Counter(row["panel"] for row in rows) == {"point": 12, "point_positive": 6, "clone_hold": 24, "exposed_transfer": 20},
            "Exact source-role denominator changed")
    require(all(len(raw(row)) == 2*row["samples"] for row in rows), "Exact source clock differs")
    return rows


def raw(row):
    source = SAME / "pcm" / f"{row['id']}-original.pcm" if row["asset"] == "same" else ALT / "pcm" / f"{row['id']}.pcm"
    data = source.read_bytes()
    require(digest(data) == row["pcm_sha256"], "Exact original PCM differs")
    return data


def freeze(run, voice):
    rows = source_rows()
    require(all(sha(voice / file) == pin for file, pin in old.voice.ASSETS.items()), "Licensed checkpoint bytes differ")
    run.mkdir(mode=0o700)
    write(run / "roster.json", rows)
    write(run / "freeze.json", {"schema": SCHEMA, "method": METHOD, "runner_sha256": sha(Path(__file__)), "current_assets": old.voice.ASSETS,
          "roster_sha256": sha(run / "roster.json"), "same_source": sha(SAME / "result.json"),
          "alternate_source": sha(ALT / "result.json"), "alternate_ledger": sha(ALT / "confirmation-current.json")})
    print("Frozen 62 source-bound PCM inputs and the original-PCM comparison", flush=True)


def boundary_cells(samples):
    spectrum = np.fft.rfft(np.lib.stride_tricks.sliding_window_view(samples, FRAME)[::HOP] * np.hanning(FRAME), n=512)
    power = abs(spectrum)**2
    mel = 2595*np.log10(1+np.asarray([80, 7200])/700)
    edges = 700*(10**(np.linspace(*mel, 25)/2595)-1)
    freqs = np.fft.rfftfreq(512, 1/RATE)
    band = np.column_stack([(freqs >= low) & (freqs < high) for low, high in zip(edges[:-1], edges[1:], strict=True)])
    values = power @ band
    centers = FRAME//2 + HOP*np.arange(len(values))
    output = {}
    for k in range(3*STEP, len(samples)-2*STEP, STEP):
        left = values[(centers >= k-GAP-CONTEXT) & (centers < k-GAP)].mean(axis=0)+1e-8
        right = values[(centers >= k+GAP) & (centers < k+GAP+CONTEXT)].mean(axis=0)+1e-8
        left, right = left/left.sum(), right/right.sum()
        mix = (left+right)/2
        output[k] = float(.5*np.sum(left*np.log(left/mix)+right*np.log(right/mix)))
    return [{"start": k, "end": k+EDIT, "score": min(v, output[k+EDIT]), "endpoints": [v, output[k+EDIT]]}
            for k, v in output.items() if k+EDIT in output]


def assess(rows, voice):
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    old.spectra.incumbent.network_free()
    model = load_voice({"weights": str(voice), "assets": old.voice.ASSETS})
    vad = SpeechActivity(voice / "silero_vad.jit")
    cached = {}
    for row in rows:
        audio = raw(row)
        values = np.frombuffer(audio, "<i2").astype(np.float32)/32768
        spans = vad.spans(values)
        current = []
        for start in range(RATE, len(values)-2*RATE+1, 2*RATE):
            end = start+2*RATE
            window = audio[2*start:2*end]
            pin = digest(window)
            response = None
            if old.window_in_speech(spans, start/RATE, end/RATE):
                if pin not in cached:
                    probability, = model.predict([values[start:end]])
                    require(math.isfinite(probability), "Invalid current response")
                    cached[pin] = old.calibration.logit(probability)
                response = cached[pin]
            current.append({"start": start, "end": end, "score": response, "pcm_sha256": pin})
        yield {"id": row["id"], "pcm_sha256": row["pcm_sha256"], "current": current, "boundary": boundary_cells(values)}
    print(f"Measured {len(cached)} new current-model native window forwards", flush=True)


def alternate(rows):
    ledger = {row["id"]: row for row in read(ALT / "confirmation-current.json")["rows"]}
    for row in rows:
        saved = ledger[row["id"]]
        require(saved["pcm_sha256"] == row["pcm_sha256"], "Alternate current baseline belongs to different PCM")
        values = np.frombuffer(raw(row), "<i2").astype(np.float32)/32768
        yield {"id": row["id"], "pcm_sha256": row["pcm_sha256"], "current": saved["windows"], "boundary": boundary_cells(values)}


def local_pairs(row):
    for span in row["boundary"]:
        possible = [win["score"] for win in row["current"] if win["score"] is not None and win["start"] <= span["start"] and win["end"] >= span["end"]]
        if possible:
            yield span, max(possible)


def responses(row, scalers):
    def highest(values):
        best = max(values, key=lambda x: x["score"], default=None)
        return [best["score"], [best["start"], best["end"]]] if best else [None, None]

    combined = []
    for boundary, current in local_pairs(row):
        z1 = (current-scalers[0][0])/scalers[0][1]
        z2 = (boundary["score"]-scalers[1][0])/scalers[1][1]
        combined.append(boundary | {"score": (z1+z2)/2})
    return {"current": highest(win for win in row["current"] if win["score"] is not None),
            "boundary": highest(row["boundary"]), "combined": highest(combined)}


def point_inputs(measured):
    features = np.array([[current, edge["score"]] for row in measured for edge, current in local_pairs(row)])
    limits = np.percentile(features, [25, 50, 75], axis=0)
    scalers = [[float(middle), max(1e-6, float(upper-lower))] for lower, middle, upper in limits.T]
    scores = [responses(row, scalers) for row in measured]
    points = {arm: max(row[arm][0] for row in scores if row[arm][0] is not None) for arm in ARMS}
    return {"scalers": scalers, "points": points}


def performance(rows, arm, point):
    assessed = [row for row in rows if row["responses"][arm][0] is not None]
    flags = [row for row in assessed if row["responses"][arm][0] > point]
    ai = [row for row in flags if row["truth"] is not None]
    useful = [row for row in ai if old.voice.useful_range(*row["responses"][arm][1], row["truth"])]
    return {"ai_flags": len(ai), "useful_ranges": len(useful), "false_groups": len({row["group"] for row in flags if row["truth"] is None}),
            "unassessed": len(rows)-len(assessed), "background_false_flags": sum(row["arm"] == "background_only" for row in flags)}


def equal_alarm(rows, arm, budget):
    peaks = defaultdict(list)
    for row in rows:
        if row["truth"] is None and row["responses"][arm][0] is not None:
            peaks[row["group"]].append(row["responses"][arm][0])
    ordered = sorted((max(v) for v in peaks.values()), reverse=True)
    return performance(rows, arm, ordered[budget]) if budget < len(ordered) else None


def summarize(rows, points):
    panels = {name: [row for row in rows if row["panel"] == name] for name in ("clone_hold", "exposed_transfer", "point_positive")}
    panels["all_held"] = panels["clone_hold"] + panels["exposed_transfer"]
    panels |= {family: [row for row in panels["exposed_transfer"] if row["family"] == family] for family in ("expressive", "stock")}
    return {panel: {"edits": sum(row["truth"] is not None for row in subset), "genuine_views": sum(row["truth"] is None for row in subset),
                    "fixed": {arm: performance(subset, arm, points[arm]) for arm in ARMS},
                    "useful_ranges_before_points": {arm: sum(row["truth"] is not None and row["responses"][arm][1] is not None
                                                             and old.voice.useful_range(*row["responses"][arm][1], row["truth"])
                                                             for row in subset) for arm in ARMS},
                    "equal_false_groups": {str(b): {arm: equal_alarm(subset, arm, b) for arm in ARMS} for b in (0, 1, 2)}
                    } for panel, subset in panels.items()}


def verified_rows(run):
    lock, rows = read(run / "freeze.json"), read(run / "roster.json")
    require(lock["runner_sha256"] == sha(Path(__file__)) and lock["roster_sha256"] == sha(run / "roster.json")
            and lock["same_source"] == sha(SAME / "result.json") and lock["alternate_source"] == sha(ALT / "result.json")
            and lock["alternate_ledger"] == sha(ALT / "confirmation-current.json"), "Original PCM study freeze changed")
    return rows


def report(run, rows, audit=False):
    files = ("point", "held", "alternate")
    ledgers = [read(run / f"{name}.json") for name in files]
    seal = read(run / "points.json")
    require(seal["point_sha256"] == sha(run / "point.json") and seal["freeze_sha256"] == sha(run / "freeze.json")
            and seal["values"] == point_inputs(ledgers[0]), "Genuine-only points changed")
    decision, by_key = [], {(row["asset"], row["id"]): row for row in rows}
    for asset, ledger in zip(("same", "same", "alt"), ledgers, strict=True):
        for row in ledger:
            original = by_key.pop((asset, row["id"]))
            data = raw(original)
            require(row["pcm_sha256"] == original["pcm_sha256"], "Source-bound model score changed")
            require([x["start"] for x in row["current"]] == list(range(RATE, len(data)//2-2*RATE+1, 2*RATE)), "Native model window starts changed")
            for win in row["current"]:
                require(win["end"]-win["start"] == 2*RATE and win["pcm_sha256"] == digest(data[2*win["start"]:2*win["end"]]),
                        "Native model window hash or range changed")
            original_pcm = np.frombuffer(data, "<i2").astype(np.float32)/32768
            require(row["boundary"] == boundary_cells(original_pcm), "Boundary scores cannot be reproduced on exact PCM")
            decision.append(original | {"responses": responses(row, seal["values"]["scalers"])})
    require(not by_key, "Score receipt left out a frozen input")
    result = {"schema": SCHEMA, "method": METHOD, "panels": summarize(decision, seal["values"]["points"]),
              "commitments": {name: sha(run / f"{name}.json") for name in (*files, "freeze", "roster", "points")}}
    if audit:
        require(read(run / "decisions.json") == decision and read(run / "result.json") == result, "Exact localization readback differs")
        print("Replayed every native current-model window, boundary and false-group comparison", flush=True)
    else:
        write(run / "decisions.json", decision)
        write(run / "result.json", result)
        print(result["panels"]["clone_hold"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "point", "held", "report", "audit"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--voice", type=Path, default=ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    run = args.run
    if args.phase == "freeze":
        freeze(run, args.voice)
        return
    rows = verified_rows(run)
    if args.phase == "point":
        measured = list(assess([row for row in rows if row["panel"] == "point"], args.voice))
        write(run / "point.json", measured)
        write(run / "points.json", {"freeze_sha256": sha(run / "freeze.json"), "point_sha256": sha(run / "point.json"), "values": point_inputs(measured)})
    elif args.phase == "held":
        seal = read(run / "points.json")
        require(seal["point_sha256"] == sha(run / "point.json") and seal["freeze_sha256"] == sha(run / "freeze.json"), "Freeze point-control scores first")
        chosen = [row for row in rows if row["panel"] in {"clone_hold", "point_positive"}]
        write(run / "held.json", list(assess(chosen, args.voice)))
        write(run / "alternate.json", list(alternate([row for row in rows if row["asset"] == "alt"])))
    else:
        report(run, rows, audit=args.phase == "audit")


if __name__ == "__main__":
    main()
