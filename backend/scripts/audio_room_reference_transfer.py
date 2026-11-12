"""Measure voice localization after controlled competing-speech attenuation."""

from __future__ import annotations

import argparse
import hashlib
import math
import resource
import time
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np
from scipy.signal import istft, stft

from app.detectors.audio.calibration import logit
from scripts import audio_overlap_window_transfer as previous
from scripts.prepare_audio_specialists_v15 import ASSETS

RATE = 16_000
STARTS = (RATE, 3 * RATE)
LENGTH = 2 * RATE
TRACK = 6 * RATE
NFFT = 512
HOP = 128
FLOOR = .04
VIEWS = ("stationary_reference", "aligned_reference", "residual_20db")
ARMS = ("observed", *VIEWS, "dry_reference")
ROLES = ("selection", "confirmation")
PUBLISHED = Path(__file__).resolve().parents[2] / "docs/evaluation-panels/audio-overlap-window-transfer-2026-11-12.json"
SCHEMA = "chipmunk.audio-room-reference-transfer/1"
METHOD = {
    "parent": "Reuse all 48 exact six-second tracks and the previously admitted 1–3 and 3–5 second windows. Keep host, room and generator assignments fixed.",
    "components": "The known competing component is the exact signed sample difference between each quantized mixture and its gain-matched dry source. This has at most one LSB of rounding ambiguity.",
    "stationary_reference": "Estimate mean competing-component power from the parts strictly outside each scored two-second window, using unpadded 512-point periodic Hann frames at 128-sample hops. No within-window samples enter this estimate.",
    "aligned_reference": "Use the exact within-window competing component's time-frequency power. This is a stronger aligned background-only research reference.",
    "masks": "Apply sqrt(clip(1 - background_power / max(mixture_power, 1e-12), 0.04, 1)) on the original complex spectrum with zero-padded 512-point Hann analysis and 128-sample hops. Recover the exact-length integer PCM by overlap-add and round to signed 16 bits.",
    "residual_20db": "Subtract 90% of the known competing waveform amplitude. Leave a controlled 20 dB quieter competing component without time-frequency filtering.",
    "points": "Keep the previous maximum-genuine point for observed and dry. For each new view, strictly exceed the maximum across its six processed and six dry genuine selection controls.",
    "decision": "Select the higher raw log-odds response among the two previously admitted windows. A useful location requires at least 0.30 IoU and 0.50 recall of the known insert.",
    "advance": "For either spectral reference to justify a real separate-channel trial, require at least two useful held locations from two generator families, at most one genuine false flag and at least four correctly ordered matched pairs.",
}
read, write, digest, require = previous.read, previous.write, previous.digest, previous.require


def pcm(raw):
    return np.frombuffer(raw, "<i2").astype(np.float64)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parent_inputs(folder):
    public = read(PUBLISHED)
    require(all(digest(folder / f"{key}.json") == value for key, value in public["commitments"].items()),
            "Published source commitments differ")
    frozen = read(folder / "freeze.json")
    require(digest(folder / "roster.json") == frozen["roster_sha256"], "Parent roster commitment differs")
    source = read(folder / "roster.json")
    result = read(folder / "result.json")
    pairs = {}
    observations = {}
    for role in ROLES:
        path = folder / f"{role}.json"
        require(digest(path) == result["ledgers_sha256"][role], "Parent classifier ledger differs")
        observations.update((row["id"], row) for row in read(path)["rows"])
    for row in source:
        content = (folder / "pcm" / f"{row['id']:03}.pcm").read_bytes()
        observation = observations[row["id"]]
        require(len(content) == 2 * TRACK and sha(content) == row["source_pcm_sha256"] == observation["input_sha256"],
                "Parent source audio differs")
        short = observation["short"][1::2]
        require([item["start"] for item in short] == list(STARTS)
                and all(item["score"] is not None and item["end"] - item["start"] == LENGTH for item in short),
                "Paired source windows must all be admitted on their original coordinates")
        require(all(sha(content[2*item["start"]:2*item["end"]]) == item["pcm_sha256"] for item in short),
                "Parent window hash differs")
        key = row["panel"], row["host_group"], row["kind"]
        entry = pairs.setdefault(key, {})
        require(row["condition"] not in entry, "Duplicate parent condition")
        entry[row["condition"]] = row, content, [item["score"] for item in short]
    require(len(pairs) == 24 and all(set(row) == {"clean", "room"} for row in pairs.values()), "Parent roster incomplete")
    return sorted(pairs.values(), key=lambda pair: pair["room"][0]["id"])


def spectrum(samples, *, padded):
    return stft(samples, fs=RATE, window="hann", nperseg=NFFT, noverlap=NFFT-HOP,
                boundary="zeros" if padded else None, padded=padded)[2]


def spectrally_attenuate(observed, nuisance_power):
    coeff = spectrum(observed, padded=True)
    power = np.abs(coeff)**2
    mask = np.sqrt(np.clip(1. - nuisance_power / np.maximum(power, 1e-12), FLOOR, 1.))
    return istft(coeff * mask, fs=RATE, window="hann", nperseg=NFFT, noverlap=NFFT-HOP)[1][:LENGTH]


def integer(samples):
    require(len(samples) == LENGTH and np.isfinite(samples).all(), "Invalid processed window")
    rounded = np.rint(samples)
    clipped = int(np.count_nonzero((rounded < -32768) | (rounded > 32767)))
    return np.clip(rounded, -32768, 32767).astype("<i2").tobytes(), clipped


def transformed(pair, start):
    end = start + LENGTH
    clean, mixed = (pcm(pair[condition][1]) for condition in ("clean", "room"))
    nuisance = mixed - clean
    observed = mixed[start:end]
    local = nuisance[start:end]
    context = np.concatenate([np.abs(spectrum(data, padded=False))**2 for data in (nuisance[:start], nuisance[end:])], axis=1)
    return {
        "stationary_reference": integer(spectrally_attenuate(observed, context.mean(axis=1, keepdims=True))),
        "aligned_reference": integer(spectrally_attenuate(observed, np.abs(spectrum(local, padded=True))**2)),
        "residual_20db": integer(observed - .90 * local),
    }


def reconstruction_snr(reference, observed):
    error = float(np.dot(reference - observed, reference - observed))
    return None if error == 0 else 10 * math.log10(float(np.dot(reference, reference)) / error)


def provenance(args):
    return {"schema": SCHEMA, "method": METHOD, "script_sha256": digest(Path(__file__)), "parent_public_sha256": digest(PUBLISHED),
            "parent_roster_sha256": digest(args.parent / "roster.json"), "parent_points_sha256": digest(args.parent / "points.json"),
            "assets": ASSETS, "windows": [[start, start+LENGTH] for start in STARTS]}


def observations(args, role, pairs, freeze):
    from scripts import audio_disagreement_transfer as previous_runtime
    from scripts.run_audio_specialists_v15 import load_voice

    previous_runtime.spectra.incumbent.network_free()
    model = load_voice({"weights": str(args.weights), "assets": freeze["assets"]})
    started, observed, forwards = time.monotonic(), [], 0
    for pair in pairs:
        row = pair["room"][0]
        if row["panel"] != role:
            continue
        output = {arm: [] for arm in ARMS}
        for index, start in enumerate(STARTS):
            end = start + LENGTH
            reference = pcm(pair["clean"][1][2*start:2*end])
            for condition, label in (("room", "observed"), ("clean", "dry_reference")):
                _, source, scores = pair[condition]
                samples = source[2*start:2*end]
                output[label].append({"score": scores[index], "sha256": sha(samples), "clipped": 0,
                                      "reconstruction_snr_db": reconstruction_snr(reference, pcm(samples))})
            for view, (samples, clipped) in transformed(pair, start).items():
                prediction, = model.predict([pcm(samples).astype(np.float32) / 32768])
                require(math.isfinite(prediction) and 0 <= prediction <= 1, "Invalid voice response")
                output[view].append({"score": logit(prediction), "sha256": sha(samples), "clipped": clipped,
                                     "reconstruction_snr_db": reconstruction_snr(reference, pcm(samples))})
                forwards += 1
        observed.append({"id": row["id"], "arms": output})
        if len(observed) % 6 == 0:
            print(role, len(observed), "mixtures", forwards, "voice forwards", flush=True)
    write(args.output / f"{role}.json", {"freeze_sha256": digest(args.output / "freeze.json"), "rows": observed,
           "forwards": forwards, "seconds": time.monotonic() - started, "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def ledger(args, role, pairs):
    rows = [pair["room"][0] for pair in pairs if pair["room"][0]["panel"] == role]
    raw = read(args.output / f"{role}.json")
    require(raw["freeze_sha256"] == digest(args.output / "freeze.json")
            and [row["id"] for row in rows] == [row["id"] for row in raw["rows"]], "New scoring roster differs")
    return [(row, obs["arms"]) for row, obs in zip(rows, raw["rows"], strict=True)]


def winning(records):
    index = max(range(len(STARTS)), key=lambda i: records[i]["score"])
    start = STARTS[index]
    return records[index]["score"], [start, start + LENGTH]


def points(args, selection):
    point = read(args.parent / "points.json")["points"]["short_odd"]["point"]
    controls = [arms for row, arms in selection if row["kind"] == "ordinary"]
    require(len(controls) == 6 and max(winning(row[arm])[0] for row in controls for arm in ("observed", "dry_reference")) == point,
            "Previous fixed point does not match its independent controls")
    return {"observed": point, "dry_reference": point,
            **{view: max(winning(arms[arm])[0] for arms in controls for arm in (view, "dry_reference")) for view in VIEWS}}


def summarize(rows, operating):
    totals = {arm: Counter() for arm in ARMS}
    rooms = {}
    snr = {arm: [] for arm in ARMS if arm != "dry_reference"}
    matched = {}
    for row, scores in rows:
        room = rooms.setdefault(row["room_index"], {arm: Counter() for arm in ARMS})
        for arm in ARMS:
            result = totals[arm]
            score, region = winning(scores[arm])
            family, kind = row["family"], row["kind"]
            matched.setdefault((row["host_group"], arm), {})[kind] = score
            flag = score > operating[arm]
            metric = f"{kind}_{'flag' if flag else 'inconclusive'}"
            for target in (result, room[arm]):
                target[metric] += 1
                if kind == "partial_ai" and previous.useful(region, row["truth"]):
                    target["useful_before_point"] += 1
                    if flag:
                        target["useful_flag"] += 1
                        target[f"useful_{family}"] += 1
            if arm in snr:
                snr[arm].extend(item["reconstruction_snr_db"] for item in scores[arm] if item["reconstruction_snr_db"] is not None)
    for (_, arm), by_kind in matched.items():
        true, ordinary = (by_kind[kind] for kind in ("partial_ai", "ordinary"))
        totals[arm]["matched_ai_above" if true > ordinary else "matched_tie" if true == ordinary else "matched_ai_below"] += 1
    return {"counts": totals, "rooms": rooms, "median_reconstruction_snr_db": {arm: median(values) for arm, values in snr.items()}}


def followup(counts):
    return {view: counts[view]["useful_flag"] >= 2
            and sum(counts[view][f"useful_{family}"] > 0 for family in previous.FAMILIES) >= 2
            and counts[view]["ordinary_flag"] <= 1 and counts[view]["matched_ai_above"] >= 4 for view in VIEWS[:2]}


def run(args):
    pairs = parent_inputs(args.parent)
    freeze = provenance(args)
    folder = args.output
    frozen = folder / "freeze.json"
    if args.stage == "freeze":
        folder.mkdir(mode=0o700)
        write(frozen, freeze)
        return
    require(read(frozen) == freeze, "Method or source freeze changed")
    calibration = folder / "points.json"
    bound = {"freeze_sha256": digest(frozen)}
    if args.stage == "fit" or args.panel == "confirmation" or args.stage == "report":
        selection = ledger(args, "selection", pairs)
        bound.update(selection_sha256=digest(folder / "selection.json"), points=points(args, selection))
        if args.stage == "fit":
            require(not (folder / "confirmation.json").exists(), "Selection points must precede held inference")
            write(calibration, bound)
            return
        require(read(calibration) == bound, "Held scoring requires frozen selection points")
    if args.stage == "score":
        require(not (folder / f"{args.panel}.json").exists(), "Scoring ledger already exists")
        observations(args, args.panel, pairs, freeze)
        return
    panels = {role: summarize(ledger(args, role, pairs), bound["points"]) for role in ROLES}
    write(folder / "result.json", {"schema": SCHEMA, "panels": panels, "followup": followup(panels["confirmation"]["counts"]),
          "points_sha256": digest(calibration), "ledgers_sha256": {role: digest(folder / f"{role}.json") for role in ROLES}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "score", "fit", "report"))
    for name in ("output", "parent", "weights"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--panel", choices=ROLES, default="selection")
    run(parser.parse_args())
