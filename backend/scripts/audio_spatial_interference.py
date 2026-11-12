"""Compare cross-fitted stereo interference cancellation on a locked room panel."""
from __future__ import annotations

import argparse
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal import istft, stft

from app.detectors.audio.decode import decode_audio
from scripts import audio_mono_dereverb_transfer as parent
from scripts import audio_stereo_agreement as agreement

paired = agreement.paired
sha, read, write = parent.sha, parent.read, parent.write
RATE, COUNT = parent.RATE, parent.SAMPLES
FFT, HOP, BAND, COHERENCE = 512, 128, (150, 6500), .15
COEFFICIENT_LIMIT, DAMPING, STRENGTH = 2., .03, .8
METHOD = {
    "fft_samples": FFT, "hop_samples": HOP, "band_hz": list(BAND), "correlation_floor": COHERENCE,
    "coefficient_limit": COEFFICIENT_LIMIT, "diagonal_loading": DAMPING, "subtract_fraction": STRENGTH,
    "fit": "Fit every frequency from two other one-second blocks, then apply to the held block.",
    "target": "The fitted reference is half the channel difference, assuming equal foreground on both channels.",
    "offset_stress": "One foreground channel is delayed four samples and attenuated by three dB.",
    "suppression": "Withhold an incumbent AI flag when an admitted alternative loses AI and its raw log odds drop beyond ln(2).",
    "rescue": "Promote an admitted incumbent inconclusive case when the admitted alternative leans AI.",
    "advance": "Require correct changes on at least three transfer foreground groups across both rooms. "
               "Permit at most two lost synthetic flags for suppression or one added false flag for rescue. "
               "Rescue must improve two generator families. No changes to clean or native controls. "
               "Offset variants must lose at most two true flags and add at most one false flag.",
}


def spectrum(pcm):
    return stft(pcm, fs=RATE, window="hann", nperseg=FFT, noverlap=FFT-HOP, axis=0, boundary="zeros")[2]


def components(pcm):
    stereo = spectrum(pcm)
    return (stereo[:, 0] + stereo[:, 1]) * .5, (stereo[:, 0] - stereo[:, 1]) * .5


def fit(pcm):
    mid, side = components(pcm)
    blocks = np.minimum(np.arange(mid.shape[1]) * HOP // RATE, 2)
    freq = np.fft.rfftfreq(FFT, 1 / RATE)
    coefficients = np.zeros_like(mid)
    for block in range(3):
        train = blocks != block
        a, b = mid[:, train], side[:, train]
        power_mid = np.mean(np.abs(a)**2, axis=1)
        power_side = np.mean(np.abs(b)**2, axis=1)
        cross = np.mean(a * np.conj(b), axis=1)
        denominator = power_mid * power_side
        coherence = np.divide(np.abs(cross)**2, denominator, out=np.zeros_like(power_mid), where=denominator > 0)
        weights = cross / (power_side + DAMPING * power_mid + 1e-12)
        weights *= COEFFICIENT_LIMIT / np.maximum(COEFFICIENT_LIMIT, np.abs(weights))
        weights[(coherence < COHERENCE) | (freq < BAND[0]) | (freq > BAND[1])] = 0
        coefficients[:, blocks == block] = weights[:, None]
    if not np.isfinite(coefficients).all():
        raise ValueError("Invalid interference coefficients")
    return coefficients


def apply(pcm, coefficients):
    mid, side = components(pcm)
    cleaned = mid - STRENGTH * coefficients * side
    output = istft(cleaned, fs=RATE, window="hann", nperseg=FFT, noverlap=FFT-HOP, boundary=True)[1][:COUNT]
    if output.shape != (COUNT,) or not np.isfinite(output).all():
        raise ValueError("Conditioned signal has a different sample clock")
    return output


def pcm_bytes(values):
    integer = np.rint(values)
    if integer.min() < -32768 or integer.max() > 32767:
        raise ValueError("Conditioned audio would clip")
    return integer.astype("<i2").tobytes()


def scene(voice, backgrounds, index):
    front = np.stack((voice, voice), axis=1)
    shifted = front.copy()
    channel = index % 2
    shifted[:, channel] = np.r_[np.zeros(4), voice[:-4]] * 10**(-3/20)
    placements = [("center", j, front) for j in range(len(backgrounds))]
    placements.extend(("offset", j, shifted) for j in (index % 2, 2 + (index + 1) % 2))
    cases = []
    for placement, number, target in placements:
        native = backgrounds[number]
        gain = np.linalg.norm(target.mean(axis=1)) / np.linalg.norm(native.mean(axis=1)) * 10**.25
        cases.append((placement, number, target, native * gain))
    gain = min(1., 31_129 / max(float(np.max(np.abs(a) + np.abs(b))) for _, _, a, b in cases))
    return pcm_bytes(front * gain), [(position, j, pcm_bytes(a * gain), pcm_bytes(b * gain))
                                     for position, j, a, b in cases]


def write_stereo(folder, ordinal, stereo):
    import wave

    path = folder / "wav" / f"{ordinal:03}.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((2, 2, RATE, 0, "NONE", ""))
        stream.writeframes(stereo)
    digest = sha(path)
    decoded = decode_audio(path, source_sha256=digest, max_seconds=3.)
    if len(decoded.pcm) != COUNT or decoded.mapping["timeline"] != paired.stereo.TIMELINE:
        raise ValueError("Stereo clip differs from the serving mono clock")
    return pcm_bytes(decoded.pcm * 32768), digest


def prepare(folder, archived, mono_run, first_mics, second_mics):
    old_lock, old_rows, old_gates, old_scores, old_binding = agreement.inputs(archived)
    mono_lock, mono_rows = parent.frozen(mono_run)
    (folder / "wav").mkdir(parents=True)
    (folder / "mono").mkdir()
    (folder / "foreground").mkdir()
    rows, captures = [], {}

    def save(stereo, *, mono=None, foreground=None, **meta):
        index = len(rows)
        if mono is None:
            mono, wav_sha = write_stereo(folder, index, stereo)
        else:
            wav_sha = None
        (folder / "mono" / f"{index:03}.pcm").write_bytes(mono)
        if foreground is not None:
            (folder / "foreground" / f"{index:03}.pcm").write_bytes(foreground)
        rows.append({"id": index, "stereo_sha256": sha(stereo), "mono_sha256": sha(mono), "wav_sha256": wav_sha,
                     "foreground_sha256": sha(foreground) if foreground is not None else None, **meta})

    old_clean = {}
    for row, gate, score in zip(old_rows, old_gates, old_scores, strict=True):
        stereo, _ = paired.pcm_views(archived, row)
        mono = (archived / "pcm" / f"{row['id']:03}.pcm").read_bytes()
        assert bool(score.get("mono")) == gate["admitted"]["mono"]
        if row["treatment"] == "clean":
            old_clean[row["id"]] = stereo.tobytes()
        save(stereo.tobytes(), mono=mono, foreground=old_clean.get(row.get("clean_id")),
             role="archived", placement="center" if row["treatment"] == "mixed" else row["treatment"],
             baseline=score.get("mono"), archived_id=row["id"],
             **{k: row.get(k) for k in ("room", "mode", "treatment", "group", "family")})

    backgrounds = []
    for native in sorted((row for row in mono_rows if row["role"] == "transfer" and row["treatment"] == "native"),
                         key=lambda r: (r["room"], r["mode"])):
        room = native["room"]
        paths = [first_mics / f"{room}.Array1-01.wav", second_mics / f"{room}.Array1-05.wav"]
        digests = [sha(path) for path in paths]
        if digests[0] != mono_lock["rooms"][room]["sha256"]:
            raise ValueError("Original microphone differs from published room freeze")
        start = native["source"]["source_start_sample"]
        stereo = paired.native_cut(paths, start)
        mono_input = (mono_run / "pcm" / f"{native['id']:03}.pcm").read_bytes()
        if sha(mono_input) != native["sha256"] or stereo[:, 0].tobytes() != mono_input:
            raise ValueError("First native microphone slice differs from earlier exact clip")
        captures[room] = {"hashes": digests, "bytes": [p.stat().st_size for p in paths], "first_url": mono_lock["rooms"][room]["url"],
                          "second_url": mono_lock["rooms"][room]["url"].replace("Array1-01", "Array1-05")}
        backgrounds.append((stereo.astype(float), room, native["mode"], start))
        save(stereo.tobytes(), role="transfer", placement="native", baseline=None, archived_id=None,
             room=room, mode=native["mode"], treatment="native", group=room, family="genuine", native_start=start)

    foregrounds = [row for row in mono_rows if row["role"] == "transfer" and row["treatment"] == "clean"]
    if len(backgrounds) != 4 or len(foregrounds) != 16:
        raise ValueError("Expected four native stereo cuts and sixteen foreground groups")
    for index, row in enumerate(foregrounds):
        raw = (mono_run / "pcm" / f"{row['id']:03}.pcm").read_bytes()
        if sha(raw) != row["sha256"]:
            raise ValueError("Foreground differs from earlier exact clean recording")
        clean, cases = scene(parent.decode(raw), [case[0] for case in backgrounds], index)
        meta = {"role": "transfer", "group": row["group"], "family": row["family"], "baseline": None, "archived_id": None}
        save(clean, placement="clean", treatment="clean", room=None, mode=None, parent_clean=row["id"], **meta)
        for position, j, front, noise in cases:
            _, room, mode, start = backgrounds[j]
            mixed = np.frombuffer(front, "<i2").astype(float) + np.frombuffer(noise, "<i2").astype(float)
            save(pcm_bytes(mixed), foreground=front, placement=position, treatment="mixed", room=room, mode=mode,
                 native_start=start, parent_clean=row["id"], **meta)
    write(folder / "roster.json", rows)
    record = {"schema": "chipmunk.audio-spatial-interference/1", "method": METHOD,
              "roster_sha256": sha(folder / "roster.json"), "runner_sha256": sha(Path(__file__)),
              "archived": old_binding, "mono_parent_sha256": sha(mono_run / "freeze.json"),
              "incumbent_assets": old_lock["incumbent_assets"], "calibration_sha256": old_lock["calibration_sha256"],
              "publisher_mics": captures, "planned": dict(Counter(row["placement"] for row in rows))}
    write(folder / "freeze.json", record)
    print(record["planned"], flush=True)


def bound(folder, archived):
    lock, rows = (read(folder / name) for name in ("freeze.json", "roster.json"))
    old_lock, old_rows, _, _, old_binding = agreement.inputs(archived)
    if (lock["runner_sha256"] != sha(Path(__file__)) or lock["roster_sha256"] != sha(folder / "roster.json")
            or lock["method"] != METHOD or lock["archived"] != old_binding
            or lock["calibration_sha256"] != old_lock["calibration_sha256"]
            or lock["incumbent_assets"] != old_lock["incumbent_assets"]):
        raise ValueError("Frozen source roster or method changed")
    return lock, rows, old_rows


def audio_bytes(folder, archived, row, old_rows):
    import wave

    ordinal = row["id"]
    name = folder / "wav" / f"{ordinal:03}.wav"
    if row["role"] == "archived":
        stereo = paired.pcm_views(archived, old_rows[row["archived_id"]])[0].tobytes()
    else:
        if sha(name) != row["wav_sha256"]:
            raise ValueError("Stereo WAV differs from its source commitment")
        with wave.open(str(name), "rb") as stream:
            stereo = stream.readframes(COUNT + 1)
    mono = (folder / "mono" / f"{ordinal:03}.pcm").read_bytes()
    if len(stereo) != 4 * COUNT or sha(stereo) != row["stereo_sha256"] or sha(mono) != row["mono_sha256"]:
        raise ValueError("Committed stereo or mono input differs")
    return np.frombuffer(stereo, "<i2").astype(float).reshape(COUNT, 2), parent.decode(mono)


def component_metrics(stereo, clean, coefficients, output):
    interference = stereo - clean
    a, b = apply(clean, coefficients), apply(interference, coefficients)
    if max(abs(a + b - output)) > 1e-7:
        raise ValueError("Estimated output does not recombine from the two reference components")
    norm = np.linalg.norm
    voice, noise = norm(clean.mean(axis=1)), norm(interference.mean(axis=1))
    return {"snr_gain_db": 20 * math.log10(norm(a) * noise / (norm(b) * voice)),
            "voice_level_change_db": 20 * math.log10(norm(a) / voice),
            "voice_error_rms_db": 20 * math.log10(max(norm(a-clean.mean(axis=1)) / voice, 1e-12))}


def score(folder, archived):
    from app.detectors.audio import calibration
    from app.detectors.audio.upgraded import SpeechActivity
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from app.engine.boundary import Deadline
    from scripts.run_audio_specialists_v15 import load_voice

    lock, rows, old_rows = bound(folder, archived)
    parent.INCUMBENT.network_free()
    weights = parent.INCUMBENT.WEIGHTS
    voice = load_voice({"weights": str(weights), "assets": lock["incumbent_assets"]})
    activity = SpeechActivity(weights / "silero_vad.jit")
    activity.model = activity_count = paired.stereo.gate.CountedVAD(activity.model)
    classifier = PellaVoiceDetector()
    bundle = calibration.load()
    count = Counter()

    def track_forward(*_):
        count["classifier"] += 1

    hook = voice.register_forward_pre_hook(track_forward)

    def measure(values, original):
        waveform = (values / 32768).astype(np.float32)
        spans = activity.spans(waveform)
        cached = SimpleNamespace(obj=SimpleNamespace(spans=lambda _: spans))
        ctx = SimpleNamespace(profile="full", lease=SimpleNamespace(get=lambda _: cached),
                              outputs=SimpleNamespace(audio=AudioTrack("analysed", waveform, COUNT)))
        windows = classifier._windows(ctx)
        if not windows:
            return {"decision": "unassessed", "raw": None}
        if len(windows) != 1 or windows[0][0] != 0 or not np.array_equal(windows[0][1], waveform):
            raise ValueError("The serving admission changed the exact input")
        raw, = voice.predict([waveform])
        direction = (calibration.evaluate(raw, waveform, windows, voice, Deadline(120))["decision"] if original
                     else calibration.initial_decision(calibration.probability(raw, bundle), bundle))
        return {"decision": direction, "raw": raw}

    began, observed = time.monotonic(), []
    try:
        for row in rows:
            stereo, mono = audio_bytes(folder, archived, row, old_rows)
            left, right = stereo.T
            baseline = row["baseline"] or {"decision": "unassessed", "raw": None}
            if row["role"] == "transfer":
                baseline = measure(mono, original=True)
            result = {"id": row["id"], "mono_sha256": row["mono_sha256"], "baseline": baseline,
                      "conditioned": {"decision": "unassessed", "raw": None}, "treatment": "unchanged"}
            if not np.array_equal(left, right):
                coefficients = fit(stereo)
                filtered = apply(stereo, coefficients)
                if row["foreground_sha256"]:
                    raw = (folder / "foreground" / f"{row['id']:03}.pcm").read_bytes() if row["role"] == "transfer" else (
                        paired.pcm_views(archived, old_rows[old_rows[row["archived_id"]]["clean_id"]])[0].tobytes())
                    if sha(raw) != row["foreground_sha256"]:
                        raise ValueError("Known foreground changed")
                    clean = np.frombuffer(raw, "<i2").astype(float).reshape(COUNT, 2)
                    result["acoustic"] = component_metrics(stereo, clean, coefficients, filtered)
                if baseline["decision"] in {"ai", "inconclusive"}:
                    try:
                        raw = pcm_bytes(filtered)
                    except ValueError:
                        result["treatment"] = "clipped"
                    else:
                        result.update(treatment="conditioned", conditioned_sha256=sha(raw),
                                      conditioned=measure(parent.decode(raw), original=False))
            observed.append(result)
            if len(observed) % 24 == 0:
                print(f"Scored {len(observed)}/{len(rows)} with {count['classifier']} forwards", flush=True)
    finally:
        hook.remove()
    write(folder / "scores.json", {"freeze_sha256": sha(folder / "freeze.json"), "rows": observed,
          "classifier_forwards": count["classifier"], "activity_frames": activity_count.calls,
          "elapsed_s": round(time.monotonic()-began, 3), "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def aggregate(folder, archived):
    from statistics import median

    from app.detectors.audio import calibration

    lock, roster, _ = bound(folder, archived)
    score_record = read(folder / "scores.json")
    if score_record["freeze_sha256"] != sha(folder / "freeze.json"):
        raise ValueError("Scores do not match the frozen room panel")
    effects, metrics, gain_groups, gain_rooms, gain_families = defaultdict(Counter), defaultdict(list), defaultdict(set), defaultdict(set), defaultdict(set)
    for row, score_row in zip(roster, score_record["rows"], strict=True):
        if (row["id"], row["mono_sha256"]) != (score_row["id"], score_row["mono_sha256"]):
            raise ValueError("A score belongs to a different source interval")
        role, position = row["role"], row["placement"]
        truth = "genuine" if row["family"] == "genuine" else "synthetic"
        a, b = score_row["baseline"], score_row["conditioned"]
        suppress = a["decision"] == "ai" and b["decision"] not in {"ai", "unassessed"} and (
            calibration.logit(a["raw"]) - calibration.logit(b["raw"]) > math.log(2))
        rescue = a["decision"] == "inconclusive" and b["decision"] == "ai"
        key = f"{role}/{position}/{truth}"
        effects[key].update(planned=1, **{f"baseline_{a['decision']}": 1, f"candidate_{b['decision']}": 1},
                            suppress=suppress, rescue=rescue)
        for arm, changed, target in (("suppress", suppress, "genuine"), ("rescue", rescue, "synthetic")):
            if changed and truth == target:
                gain_groups[role, position, arm].add(row["group"])
                gain_rooms[role, position, arm].add(row["room"])
                gain_families[role, position, arm].add(row["family"])
        if "acoustic" in score_row:
            for field, value in score_row["acoustic"].items():
                metrics[f"{role}/{position}/{row['room']}/{row['mode']}/{field}"].append(value)
    advance = {}
    for arm, error_class, offset_limit in (("suppress", "synthetic", 2), ("rescue", "genuine", 1)):
        key = ("transfer", "center", arm)
        advance[arm] = (len(gain_groups[key]) >= 3 and len(gain_rooms[key]) >= 2
                        and (arm == "suppress" or len(gain_families[key]) >= 2)
                        and effects[f"transfer/center/{error_class}"][arm] <= offset_limit
                        and effects[f"transfer/offset/{error_class}"][arm] <= offset_limit
                        and not any(value[arm] for cell, value in effects.items() if "/clean/" in cell or "/native/" in cell))
    output = {"schema": lock["schema"], "method": METHOD, "freeze_sha256": sha(folder / "freeze.json"),
              "scores_sha256": sha(folder / "scores.json"), "planned": lock["planned"], "cells": effects,
              "median_acoustic": {key: median(value) for key, value in metrics.items()},
              "gain_groups": {"/".join(key): len(value) for key, value in gain_groups.items()},
              "gain_rooms": {"/".join(key): len(value) for key, value in gain_rooms.items()},
              "gain_families": {"/".join(key): len(value) for key, value in gain_families.items()}, "advance": advance,
              **{key: score_record[key] for key in ("classifier_forwards", "activity_frames", "elapsed_s", "rss_bytes")}}
    write(folder / "aggregate.json", output)
    print(advance, flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("prepare", "score", "aggregate"))
    cli.add_argument("folder", type=Path)
    cli.add_argument("--archived", type=Path, required=True)
    cli.add_argument("--mono-run", type=Path)
    cli.add_argument("--first-mics", type=Path)
    cli.add_argument("--second-mics", type=Path)
    args = cli.parse_args()
    if args.phase == "prepare":
        prepare(args.folder, args.archived, args.mono_run, args.first_mics, args.second_mics)
    else:
        (score if args.phase == "score" else aggregate)(args.folder, args.archived)


if __name__ == "__main__":
    main()
