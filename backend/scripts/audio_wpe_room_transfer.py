"""Compare a blind dereverberated view with its matched native voice score."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import median

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import irfft, rfft
from scipy.ndimage import uniform_filter1d

from scripts import audio_room_stress_transfer as stress

SCHEMA = "chipmunk.audio-wpe-room/1"
FRAME, HOP, DELAY, TAPS, ITERATIONS = 512, 128, 3, 10, 3
METHOD = {
    "transform": "512-sample square-root periodic Hann, 128-sample hop, symmetric zero padding and weighted overlap-add at the unchanged 16 kHz clock. Use both channels to predict both current spectra.",
    "prediction": "Estimate late energy from ten previous stereo frames starting three hops (24 ms) back. Iterate three variance-normalized least-squares updates, using a five-frame mean power estimate with a one-percent per-bin floor and 0.001 diagonal loading relative to covariance mean diagonal. Return the residual stereo mean in its original clock. Decline invalid, clipped and near-empty outputs.",
    "roles": "Use the exact frozen Aachen room-stress waveform groups. Prior-exposed samples support comparison only. Before new assessment inference fit one candidate point at the strict maximum over eight separate genuine readers and all full-room and noisy stages. Also report use at the original mono point as a diagnostic. Test dry and early stages as negative processing controls and the six background-only excerpts in the assessment role.",
    "decisions": "Add the strongest admitted WPE window above its fitted point to the retained frozen mono result. Compare exact timestamps on two windows and publisher truth at overlap 0.30 and half-edit coverage. Require at least two added reader groups from two families across full-room and noisy stages, no more than one extra genuine group and no background-only additions. Dry and early controls must also have no added genuine groups. Repeated levels of the same source count once.",
    "acoustic": "For paired positive clips measure full-clip and truth-window SI-SDR with the known early-room foreground after inference input construction. The reference is used only for diagnostics, without influencing output, points, or eligibility.",
}


def transform(stereo, fft):
    analysis = np.sqrt(.5 - .5 * np.cos(2 * np.pi * np.arange(FRAME) / FRAME))
    padded = np.pad(stereo, ((FRAME // 2, FRAME // 2), (0, 0)))
    chunks = sliding_window_view(padded, FRAME, axis=0)[::HOP]
    return fft(chunks * analysis, axis=-1).transpose(2, 0, 1), analysis


def restore(spectrum, window, inverse):
    frames = inverse(spectrum, n=FRAME, axis=0).T * window
    samples = np.zeros(stress.SAMPLES + FRAME)
    weights = np.zeros_like(samples)
    for index, frame in enumerate(frames):
        span = slice(index * HOP, index * HOP + FRAME)
        samples[span] += frame
        weights[span] += window ** 2
    clip = slice(FRAME // 2, FRAME // 2 + stress.SAMPLES)
    return samples[clip] / weights[clip]


def wpe(stereo, *, fft=rfft, inverse=irfft):
    stress.require(stereo.shape == (stress.SAMPLES, 2), "Expected original six-second stereo clock")
    spectrum, window = transform(stereo, fft)
    history = np.pad(spectrum, ((0, 0), (DELAY + TAPS - 1, 0), (0, 0)))
    predictors = np.concatenate([history[:, TAPS - i - 1:TAPS - i - 1 + spectrum.shape[1]]
                                 for i in range(TAPS)], axis=2)
    active = DELAY + TAPS - 1
    regressors, observed = predictors[:, active:], spectrum[:, active:]
    reg_adj = regressors.conj().swapaxes(-1, -2)
    residual = spectrum
    floor = .01 * np.maximum(np.mean(abs(spectrum) ** 2, axis=(1, 2)), 1e-10)
    eye = np.eye(TAPS * 2)
    for _ in range(ITERATIONS):
        power = uniform_filter1d(np.mean(abs(residual) ** 2, axis=2), size=5, axis=1)
        weighted = reg_adj / np.maximum(power[:, None, active:], floor[:, None, None])
        covariance = weighted @ regressors
        diagonal = .001 * np.maximum(np.trace(covariance, axis1=1, axis2=2).real / (TAPS * 2), 1e-10)
        filters = np.linalg.solve(covariance + diagonal[:, None, None] * eye, weighted @ observed)
        residual = spectrum - predictors @ filters
    waveform = np.rint(restore(residual.mean(axis=2), window, inverse))
    if (not np.isfinite(waveform).all() or np.max(abs(waveform)) >= 32768 or
            np.mean(waveform ** 2) < .02 * np.mean(stereo.mean(axis=1) ** 2)):
        return None
    return waveform.astype("<i2")


def signal_gain(candidate, mono, reference):
    def sisdr(estimate, clean):
        estimate = estimate.astype(np.float64) - estimate.mean()
        clean = clean.astype(np.float64) - clean.mean()
        target = clean * (np.dot(estimate, clean) / max(np.dot(clean, clean), 1e-9))
        difference = estimate - target
        return 10 * math.log10(max(1e-9, np.dot(target, target)) / max(1e-9, np.dot(difference, difference)))

    return round(sisdr(candidate, reference) - sisdr(mono, reference), 6)


def prepare(args):
    prior_lock, prior_roster, stereo = stress.sources(args)
    stress.compare(args.stress_run / "freeze.json", prior_lock)
    stress.compare(args.stress_run / "roster.json", prior_roster)
    public = stress.read(args.stress_run / "public.json")
    stress.require(public["freeze_sha256"] == stress.digest(args.stress_run / "freeze.json") and
                   public["private_roster_sha256"] == stress.digest(args.stress_run / "roster.json"),
                   "Paired room score does not bind its waveform sources")
    refs = {row["group"]: stress.peak_quantize(waveform.astype(np.float64).mean(axis=1))
            for row, waveform in zip(prior_roster, stereo, strict=True) if row["stage"] == "early_room"}
    prior_reads = {row["case"]: row for row in public["assessment"]["rows"]}
    rows, signals = [], []
    fft_exact, fft_max_lsb = 0, 0
    stages = {"full_room", *[f"snr_{level}" for level in stress.LEVELS]}
    for row, waveform in zip(prior_roster, stereo, strict=True):
        if row["role"] == "development" and row["stage"] not in stages:
            continue
        reading = prior_reads.get(row["case"])
        if row["role"] == "assessment":
            stress.require(reading is not None and all(reading[key] == row[key] for key in row if key != "sha256"),
                           "Retained mono score has a different room or truth clock")
        mono = stress.peak_quantize(waveform.astype(np.float64).mean(axis=1))
        candidate = wpe(waveform)
        alternate = wpe(waveform, fft=np.fft.rfft, inverse=np.fft.irfft)
        if candidate is None:
            stress.require(alternate is None, "FFT implementation changed abstention")
            fft_exact += 1
        else:
            stress.require(alternate is not None, "FFT implementation changed candidate availability")
            error = int(np.max(np.abs(candidate.astype(np.int32) - alternate.astype(np.int32))))
            stress.require(error <= 1, "FFT implementation differs by more than one native integer unit")
            fft_max_lsb = max(fft_max_lsb, error)
            fft_exact += error == 0
        evidence = None
        if row["truth"] and candidate is not None:
            reference = refs[row["group"]]
            intervals = [slice(*truth) for truth in row["truth"]]
            evidence = {"full_db": signal_gain(candidate, mono, reference),
                        "edit_db": [signal_gain(candidate[s], mono[s], reference[s]) for s in intervals]}
        rows.append({**row, "mono": reading["responses"]["mono"] if reading else None,
                     "candidate_sha256": stress.sha(candidate.tobytes()) if candidate is not None else None,
                     "acoustic": evidence})
        signals.append(candidate)
    freeze = {"schema": SCHEMA, "method": METHOD, "assets": prior_lock["assets"], "windows": stress.WINDOWS,
              "levels_db": stress.LEVELS, "prior_points": prior_lock["prior_points"],
              "runner": stress.digest(Path(__file__)), "ancestor_public": stress.digest(args.stress_run / "public.json"),
              "ancestor_freeze": public["freeze_sha256"], "ancestor_roster": public["private_roster_sha256"],
              "alternate_fft_cases": len(rows), "alternate_fft_exact_cases": fft_exact, "alternate_fft_max_lsb": fft_max_lsb}
    return freeze, rows, signals


def score(args, lock, rows, signals, role):
    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.run_audio_specialists_v15 import load_voice

    network_free()
    voice = load_voice({"weights": str(args.weights), "assets": lock["assets"]})
    activity = SpeechActivity(args.weights / "silero_vad.jit")
    cache, records = {}, []
    started = time.monotonic()
    for row, signal in zip(rows, signals, strict=True):
        if row["role"] != role:
            continue
        windows = None
        if signal is not None:
            pcm = signal.astype(np.float32) / 32768
            speech = activity.spans(pcm)
            windows = []
            for a, b in stress.WINDOWS:
                value = None
                if window_in_speech(speech, a / stress.RATE, b / stress.RATE):
                    key = stress.sha(signal[a:b].tobytes())
                    if key not in cache:
                        probability, = voice.predict([pcm[a:b]])
                        stress.require(math.isfinite(probability) and 0 <= probability <= 1, "Invalid voice response")
                        cache[key] = float(logit(probability))
                    value = cache[key]
                windows.append(value)
        records.append({key: row[key] for key in ("case", "group", "room", "role", "kind", "stage", "truth", "family", "acoustic")}
                       | {"responses": {"mono": row["mono"], "wpe": [] if windows is None else [windows]}})
        if len(records) % 10 == 0:
            print(role, len(records), len(cache), round(time.monotonic() - started, 2), flush=True)
    print(role, len(records), len(cache), round(time.monotonic() - started, 2), flush=True)
    return {"rows": records, "forwards": len(cache)}


def fit_point(fit):
    rows = fit["rows"]
    admitted = [(row["group"], picked[0]) for row in rows if (picked := stress.chosen(row, "wpe")) is not None]
    stress.require(len(rows) == 40 and len({group for group, _ in admitted}) == 8
                   and all(row["role"] == "development" and row["kind"] == "genuine" and not row["truth"] for row in rows),
                   "Expected eight separate genuine point groups across five conditions")
    return max(value for _, value in admitted)


def summary(rows, baseline, fitted):
    conditions, advancement = {}, {}
    for candidate_point, point in (("fitted", fitted), ("prior_mono", baseline)):
        found, ordinary, background, clean = {}, set(), set(), set()
        stages = {}
        for stage in (*stress.STAGES, "background_only"):
            cohort = [row for row in rows if row["stage"] == stage]
            mono, ranges = stress.outcomes(cohort, "mono", baseline)
            candidate, locations = stress.outcomes(cohort, "wpe", point)
            gains = {group: locations[group] for group in candidate["located"] - mono["located"]}
            added = candidate["ordinary"] - mono["ordinary"]
            if stage in {"dry", "early_room"}:
                clean.update(added)
            else:
                found.update(gains)
                ordinary.update(added)
            background.update(candidate["background"] - mono["background"])
            stages[stage] = {"mono": {key: sorted(value) for key, value in mono.items()}, "mono_ranges": ranges,
                             "wpe": {key: sorted(value) for key, value in candidate.items()}, "added_locations": gains,
                             "added_ordinary": sorted(added)}
        families = {location["family"] for location in found.values()}
        advancement[candidate_point] = {"unique_edit_groups": len(found), "generator_families": sorted(families),
                                        "added_ordinary": sorted(ordinary - background), "added_background": sorted(background),
                                        "added_dry_or_early": sorted(clean),
                                        "passes": len(found) >= 2 and len(families) >= 2 and len(ordinary) <= 1
                                        and not background and not clean}
        conditions[candidate_point] = stages
    acoustics = {}
    for stage in stress.STAGES:
        assessed = [row["acoustic"] for row in rows if row["stage"] == stage and row["acoustic"] is not None]
        acoustics[stage] = {"clips": len(assessed),
                            "median_full_db": median(row["full_db"] for row in assessed) if assessed else None,
                            "median_edit_db": median(median(row["edit_db"]) for row in assessed) if assessed else None}
    return {"conditions": conditions, "advancement": advancement, "acoustics": acoustics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "score"))
    parser.add_argument("run", type=Path)
    for flag in ("stress-run", "input", "publisher", "prior-directional", "ami"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=stress.aachen.ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    lock, rows, signals = prepare(args)
    lock_path, roster_path = args.run / "freeze.json", args.run / "roster.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        stress.write(lock_path, lock)
        stress.write(roster_path, rows)
        print("Frozen", len(rows), "native waveforms, alternate FFT exact", lock["alternate_fft_exact_cases"],
              stress.digest(lock_path), flush=True)
        return
    stress.compare(lock_path, lock)
    stress.compare(roster_path, rows)
    fit_path, point_path = args.run / "development.json", args.run / "point.json"
    if args.phase == "fit":
        results = score(args, lock, rows, signals, "development")
        stress.write(fit_path, results)
        stress.write(point_path, {"point": fit_point(results), "development_sha256": stress.digest(fit_path),
                                 "freeze_sha256": stress.digest(lock_path)})
        print("Point frozen", fit_point(results), flush=True)
        return
    fit, point = stress.read(fit_path), stress.read(point_path)
    stress.require(point == {"point": fit_point(fit), "development_sha256": stress.digest(fit_path),
                             "freeze_sha256": stress.digest(lock_path)}, "Development-only fit changed")
    assessment = score(args, lock, rows, signals, "assessment")
    readout = summary(assessment["rows"], lock["prior_points"]["mono"], point["point"])
    stress.write(args.run / "public.json", {"schema": SCHEMA, "freeze": lock, "freeze_sha256": stress.digest(lock_path),
                 "private_roster_sha256": stress.digest(roster_path), "development": fit, "point": point,
                 "assessment": assessment, "summary": readout})
    print(json.dumps(readout["advancement"], indent=2), flush=True)


if __name__ == "__main__":
    main()
