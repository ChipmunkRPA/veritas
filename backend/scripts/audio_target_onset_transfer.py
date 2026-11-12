"""Compare onset-steered stereo target speech against mono in competing rooms."""

from __future__ import annotations

import argparse
import hashlib
import math
import resource
import time
from collections import Counter
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import numpy as np
from scipy.signal import istft, stft

from app.detectors.audio import calibration
from scripts import audio_spatial_interference as prior
from scripts.prepare_audio_specialists_v15 import ASSETS

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
PUBLISHED = ROOT / "docs/evaluation-panels/audio-spatial-interference-2026-11-11.json"
SOURCE = BASE / "audio-spatial-interference-2026-11-11/run"
ARCHIVED = BASE / "audio-paired-stereo-2026-11-11/run"
WEIGHTS = ROOT / "models/weights/audio-upgrade"
RATE, ONSET, QUERY = 16_000, 8_000, 40_000
FFT, HOP, LAG = 512, 128, 8
ARMS = ("mono", "onset_steered")
ROLES = {"selection": "archived", "confirmation": "transfer"}
SCHEMA = "chipmunk.audio-target-onset-transfer/1"
METHOD = {
    "sources": "All centered simultaneous mixtures and their exact clean foregrounds from the verified archived/transfer stereo panel. Include untouched native simultaneous cuts and one clean control per foreground. Prior project sources, newly synthesized geometry.",
    "construction": "First 0.5 s of the foreground is an isolated target onset. Use its disjoint last 2.5 s as the query with the original measured stereophonic room background. Set query mono target-to-background RMS to -5 dB. Deterministically displace foreground by 1–7 samples and 1.5–4.5 dB in one channel using a group hash. Share gains between each mixed view and its component audit.",
    "target": "Estimate signed integer delay and gain from the time-aligned onset channels only. Require onset RMS at least -45 dBFS on both channels, correlation at least .95, delay at most 8 samples and gain ratio .4–2.5. The onset and supplied target timing are required. The clean and native controls lack a supplied onset and keep the mono view.",
    "representation": "Cross-fit a distortionless two-channel frequency-domain beamformer with the onset steering vector. A periodic Hann 512-point transform uses 128-sample hops. Estimate covariance outside each of five held half-second blocks. Diagonal loading is .05 times the mean channel power per frequency. Apply from 150 to 6500 Hz. Outside the band use equally weighted, steered channels, normalized at the reference microphone.",
    "score": "Classify each mono view and its candidate view with the same pinned voice model and serving Full speech admission on the exact 2.5 s query clock. The candidate uses mono fallback without a qualified onset or admitted steered view. The unchanged mono also gets its current serving direction and stability logic.",
    "point": "For each arm use strict excess of the greater of its highest admitted clean/native genuine score and third-highest admitted noisy genuine score on selection. Never consult confirmation scores before the point ledger. Equal-false-alarm curves at 0/1/2/4 counts on held noisy cases are retrospective diagnostics.",
    "advance": "To justify fresh acquisition, require improved held noisy AI flags at the frozen points with no additional genuine false flags, gains across both rooms and at least two families, no additional clean/native false flags, and median noisy acoustic SNR improvement >= 2 dB in each held room. No app or confidence change.",
}
read, write = prior.read, prior.write


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def pcm(values):
    quantized = np.rint(values)
    require(np.isfinite(quantized).all() and np.max(np.abs(quantized)) <= 32767, "Invalid or clipped samples")
    return quantized.astype("<i2").tobytes()


def array(raw, channels=1):
    output = np.frombuffer(raw, "<i2").astype(np.float64)
    return output.reshape(-1, channels) if channels > 1 else output


def geometry(group):
    code = bytes.fromhex(digest(f"{SCHEMA}:{group}".encode()))
    return code[0] % 2, 1 + code[1] % 7, (1.5, 3.0, 4.5)[code[2] % 3]


def scene(clean, nuisance, group):
    channel, delay, attenuation = geometry(group)
    stereo = np.repeat(clean[:, None], 2, axis=1)
    stereo[:, channel] = np.r_[np.zeros(delay), clean[:-delay]] * 10 ** (-attenuation / 20)
    query, background = stereo[ONSET:], nuisance[ONSET:]
    norm = np.linalg.norm
    require(norm(query.mean(axis=1)) > 0 and norm(background.mean(axis=1)) > 0, "Silent component")
    background = background * (norm(query.mean(axis=1)) / norm(background.mean(axis=1))) * 10 ** .25
    peak = max(np.abs(query + background).max(), np.abs(stereo).max(), np.abs(background).max())
    gain = min(1., 31_129 / peak)
    return pcm(stereo[:ONSET] * gain), pcm(query * gain), pcm(background * gain)


def estimate(anchor):
    results = []
    left, right = anchor.T
    if np.any(np.sqrt(np.mean(anchor**2, axis=0)) < 32768 * 10 ** (-45 / 20)):
        return None
    for lag in range(-LAG, LAG + 1):
        a, b = left[LAG:-LAG], right[LAG + lag:len(right) - LAG + lag]
        power = float(a @ a)
        denominator = math.sqrt(power * float(b @ b))
        if denominator:
            cross = float(a @ b)
            results.append((cross / denominator, lag, cross / power))
    if not results:
        return None
    similarity, lag, ratio = max(results)
    if similarity < .95 or not .4 <= ratio <= 2.5:
        return None
    return {"lag": lag, "ratio": ratio, "correlation": similarity}


def spectrum(values):
    return stft(values, fs=RATE, nperseg=FFT, noverlap=FFT-HOP, axis=0)[2]


def beam_weights(observed, target):
    spectra = spectrum(observed)
    hz = np.fft.rfftfreq(FFT, 1 / RATE)
    steering = np.stack((np.ones_like(hz), target["ratio"] * np.exp(-2j * np.pi * hz * target["lag"] / RATE)), axis=1)
    centers = np.arange(spectra.shape[-1]) * HOP
    blocks = np.minimum(centers // ONSET, 4)
    outside_band = (hz < 150) | (hz > 6500)
    output = np.zeros_like(spectra)
    for block in range(5):
        # Exclude any FFT frame that touches the held half-second block.
        training = (centers + FFT // 2 < block * ONSET) | (centers - FFT // 2 >= (block + 1) * ONSET)
        fit = spectra[:, :, training]
        covariance = np.einsum("fct,fdt->fcd", fit, fit.conj()) / fit.shape[-1]
        load = .05 * covariance.trace(axis1=1, axis2=2).real / 2 + 1e-10
        numerator = np.linalg.solve(covariance + load[:, None, None] * np.eye(2), steering[..., None])[..., 0]
        numerator[outside_band] = steering[outside_band]
        weights = numerator / np.einsum("fc,fc->f", steering.conj(), numerator).real[:, None]
        output[:, :, blocks == block] = weights[:, :, None]
    require(np.allclose(np.einsum("fct,fc->ft", output.conj(), steering), 1., atol=1e-9), "Steered response differs from one")
    return output


def apply(values, weights):
    spectrum_out = np.einsum("fct,fct->ft", weights.conj(), spectrum(values))
    waveform = istft(spectrum_out, fs=RATE, nperseg=FFT, noverlap=FFT-HOP)[1][:QUERY]
    require(waveform.shape == (QUERY,) and np.isfinite(waveform).all(), "Steered query clock differs")
    return waveform


def db(numerator, denominator):
    norm = np.linalg.norm
    return float(20 * np.log10(max(norm(numerator), 1e-10) / max(norm(denominator), 1e-10)))


def project(mix, front, back, target):
    weights = beam_weights(mix, target)
    recovered, nuisance = apply(front, weights), apply(back, weights)
    steered = apply(mix, weights)
    require(np.max(np.abs(recovered + nuisance - steered)) < 1e-7, "Component replay does not reconstruct beam output")
    # Peak protection affects the rendering, not the component-SNR comparison.
    peak_gain = min(1., 31_129 / max(1., np.max(np.abs(steered))))
    raw_snr, steered_snr = db(front.mean(axis=1), back.mean(axis=1)), db(recovered, nuisance)
    return pcm(steered * peak_gain), {"mono_snr_db": raw_snr, "steered_snr_db": steered_snr, "snr_gain_db": steered_snr-raw_snr,
            "target_error_db": db(recovered-front[:, 0], front[:, 0]), "peak_gain": peak_gain}


def build(source, archived):
    lock, roster, old = prior.bound(source, archived)
    receipt = read(PUBLISHED)
    require(prior.sha(source / "freeze.json") == receipt["freeze_sha256"], "Published spatial parent differs")
    require(lock["calibration_sha256"] == calibration.SHA256 and lock["incumbent_assets"] == ASSETS, "Pinned scoring assets differ")
    rows, views = [], []
    role_sets = {}
    for role, old_role in ROLES.items():
        room_groups, foreground_groups, foreground_digests = set(), set(), set()
        current = [row for row in roster if row["role"] == old_role]
        for row in current:
            placement = row["placement"]
            if placement not in {"center", "clean", "native"} or (placement != "clean" and row["mode"] != "simultaneous"):
                continue
            observed, _ = prior.audio_bytes(source, archived, row, old)
            target = None
            truth = "genuine" if row["family"] == "genuine" else "synthetic"
            audit = None
            if placement == "center":
                raw = (source / "foreground" / f"{row['id']:03}.pcm").read_bytes()
                require(digest(raw) == row["foreground_sha256"], "Frozen foreground changed")
                clean = array(raw, 2)
                require(np.array_equal(clean[:, 0], clean[:, 1]), "Only exact centered foregrounds qualify")
                onset, front, back = scene(clean[:, 0], observed - clean, row["group"])
                front, back = array(front, 2), array(back, 2)
                mix = front + back
                raw_mono = pcm(mix.mean(axis=1))
                target = estimate(array(onset, 2))
                audit = {"onset_sha256": digest(onset), "query_stereo_sha256": digest(pcm(mix))}
                raw_steered = raw_mono
                if target is not None:
                    channel, delay, attenuation = geometry(row["group"])
                    require(target["lag"] == delay * (1 if channel == 1 else -1), "Estimated target lag missed controlled geometry")
                    require(abs(20 * math.log10(target["ratio"]) - attenuation * (1 if channel == 0 else -1)) < .05,
                            "Estimated target level missed controlled geometry")
                    raw_steered, metrics = project(mix, front, back, target)
                    audit.update(metrics)
                require(abs(db(front.mean(axis=1), back.mean(axis=1)) + 5) < .01, "Matched background level changed")
                foreground_digests.add(row["foreground_sha256"])
                foreground_groups.add(row["group"])
                room_groups.add(row["room"])
            else:
                raw_mono = pcm(observed[ONSET:].mean(axis=1))
                raw_steered = raw_mono
            rows.append({"id": row["id"], "role": role, "source": placement, "truth": truth, "family": row["family"],
                         "room": row["room"], "group": row["group"], "target": target, "acoustics": audit,
                         "pcm_sha256": {"mono": digest(raw_mono), "onset_steered": digest(raw_steered)}})
            views.append((raw_mono, raw_steered))
        role_sets[role] = (room_groups, foreground_groups, foreground_digests)
    require(all(not a & b for a, b in zip(*role_sets.values(), strict=True)), "Roles share exact rooms or foreground ancestry")
    expected = {"selection": {"center": 34, "clean": 17, "native": 8}, "confirmation": {"center": 32, "clean": 16, "native": 2}}
    require(all(Counter(row["source"] for row in rows if row["role"] == role) == count for role, count in expected.items()),
            "Required role roster changed")
    return rows, views, {"source_freeze_sha256": prior.sha(source / "freeze.json"), "source_public_sha256": prior.sha(PUBLISHED),
                         "calibration_sha256": calibration.SHA256, "assets": ASSETS}


def freeze(args):
    rows, views, ancestry = build(args.source, args.archived)
    args.run.mkdir(mode=0o700)
    output = args.run / "pcm"
    output.mkdir()
    for row, pair in zip(rows, views, strict=True):
        for arm, raw in zip(ARMS, pair, strict=True):
            (output / f"{row['id']:03}-{arm}.pcm").write_bytes(raw)
    write(args.run / "roster.json", rows)
    write(args.run / "freeze.json", {"schema": SCHEMA, "method": METHOD, "ancestry": ancestry,
          "script_sha256": prior.sha(Path(__file__)), "roster_sha256": prior.sha(args.run / "roster.json")})
    print(Counter((row["role"], row["source"]) for row in rows), flush=True)


def bound(args):
    lock = read(args.run / "freeze.json")
    require(lock["script_sha256"] == prior.sha(Path(__file__)) and lock["roster_sha256"] == prior.sha(args.run / "roster.json")
            and lock["ancestry"]["calibration_sha256"] == prior.sha(calibration.PATH) and lock["method"] == METHOD,
            "Frozen experiment, roster or serving calibration changed")
    return lock, read(args.run / "roster.json")


def checked_pcm(args, row, arm):
    raw = (args.run / "pcm" / f"{row['id']:03}-{arm}.pcm").read_bytes()
    require(len(raw) == 2 * QUERY and digest(raw) == row["pcm_sha256"][arm], "Frozen query samples changed")
    return raw


def score(args, role):
    from app.detectors.audio.upgraded import SpeechActivity
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from app.engine.boundary import Deadline
    from scripts.audio_noise_speech_gate_v256 import CountedVAD
    from scripts.run_audio_specialists_v15 import load_voice

    lock, rows = bound(args)
    if role == "confirmation":
        require(read(args.run / "points.json")["selection_sha256"] == prior.sha(args.run / "selection.json"), "Freeze selection points first")
    prior.parent.INCUMBENT.network_free()
    voice = load_voice({"weights": str(args.weights), "assets": lock["ancestry"]["assets"]})
    activity = SpeechActivity(args.weights / "silero_vad.jit")
    activity.model = count_gate = CountedVAD(activity.model)
    classifier = PellaVoiceDetector()
    counter = Counter()

    def count(*_):
        counter["forwards"] += 1

    def measure(raw, serving):
        waveform = array(raw).astype(np.float32) / 32768
        spans = activity.spans(waveform)
        activity_lease = SimpleNamespace(obj=SimpleNamespace(spans=lambda _: spans))
        ctx = SimpleNamespace(profile="full", lease=SimpleNamespace(get=lambda _: activity_lease),
                              outputs=SimpleNamespace(audio=AudioTrack("analysed", waveform, QUERY)))
        windows = classifier._windows(ctx)
        if not windows:
            return {"score": None, "direction": "unassessed"}
        require(len(windows) == 1 and windows[0][0] == 0 and np.array_equal(windows[0][1], waveform), "Serving gate changed query clock")
        probability, = voice.predict([waveform])
        require(math.isfinite(probability) and 0 <= probability <= 1, "Invalid voice probability")
        direction = calibration.evaluate(probability, waveform, windows, voice, Deadline(120))["decision"] if serving else None
        return {"score": calibration.logit(probability), "direction": direction}

    results = []
    start = time.monotonic()
    hook = voice.register_forward_pre_hook(count)
    try:
        for row in rows:
            if row["role"] != role:
                continue
            mono, steered = (checked_pcm(args, row, arm) for arm in ARMS)
            current = measure(mono, True)
            proposed = measure(steered, False) if steered != mono and current["score"] is not None else {"score": None}
            selected = "mono" if proposed["score"] is None else "onset_steered"
            results.append({"id": row["id"], "pcm_sha256": row["pcm_sha256"], "mono": current["score"],
                            "onset_steered": (current if selected == "mono" else proposed)["score"],
                            "candidate_input": selected, "serving": current["direction"]})
            if len(results) % 12 == 0:
                print(role, len(results), dict(counter), flush=True)
    finally:
        hook.remove()
    write(args.run / f"{role}.json", {"freeze_sha256": prior.sha(args.run / "freeze.json"),
          "points_sha256": None if role == "selection" else prior.sha(args.run / "points.json"), "rows": results,
          "forwards": counter["forwards"], "activity_frames": count_gate.calls,
          "seconds": time.monotonic() - start, "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def ledger(args, role, rows):
    scores = read(args.run / f"{role}.json")
    require(scores["freeze_sha256"] == prior.sha(args.run / "freeze.json")
            and (role == "selection" or scores["points_sha256"] == prior.sha(args.run / "points.json")), "Role ledger binding changed")
    pairs = list(zip([row for row in rows if row["role"] == role], scores["rows"], strict=True))
    require(all((row["id"], row["pcm_sha256"]) == (obs["id"], obs["pcm_sha256"]) for row, obs in pairs), "Query response clock binding changed")
    return scores, pairs


def operating_point(pairs, arm):
    noisy = sorted((obs[arm] for row, obs in pairs if row["source"] == "center" and row["truth"] == "genuine"
                    and obs[arm] is not None), reverse=True)
    controls = [obs[arm] for row, obs in pairs if row["source"] != "center" and row["truth"] == "genuine" and obs[arm] is not None]
    require(len(noisy) >= 3 and len(controls) >= 1, "Insufficient admitted controls")
    return max(noisy[2], max(controls))


def fit(args):
    _, rows = bound(args)
    _, pairs = ledger(args, "selection", rows)
    write(args.run / "points.json", {"selection_sha256": prior.sha(args.run / "selection.json"),
          "points": {arm: operating_point(pairs, arm) for arm in ARMS}})


def flag(value, point):
    return value is not None and value > point


def decisions(pairs, points):
    output = {}
    for row, obs in pairs:
        key = f"{row['source']}/{row['room'] or 'none'}/{row['truth']}"
        counts = output.setdefault(key, Counter())
        counts["planned"] += 1
        counts["steered_input"] += obs["candidate_input"] == "onset_steered"
        for arm in ARMS:
            counts[f"{arm}_flags"] += flag(obs[arm], points[arm])
            counts[f"{arm}_unassessed"] += obs[arm] is None
        counts[f"serving_{obs['serving']}"] += 1
    return output


def equal_false_alarm(pairs, budget):
    output = {}
    for arm in ARMS:
        negatives = sorted((obs[arm] for row, obs in pairs if row["truth"] == "genuine" and obs[arm] is not None), reverse=True)
        require(len(negatives) > budget, "Too few held genuine scores for equal-false-alarm readout")
        cut = negatives[budget]
        output[arm] = {"false_alarms": sum(value > cut for value in negatives),
                       "ai_flags": sum(flag(obs[arm], cut) for row, obs in pairs if row["truth"] == "synthetic")}
    return output


def report(args):
    lock, rows = bound(args)
    selection, choose = ledger(args, "selection", rows)
    held, confirm = ledger(args, "confirmation", rows)
    points = read(args.run / "points.json")["points"]
    require(all(operating_point(choose, arm) == points[arm] for arm in ARMS), "Point arithmetic differs")
    noisy = [(row, obs) for row, obs in confirm if row["source"] == "center"]
    primary = decisions(confirm, points)
    gains = [(row, obs) for row, obs in noisy if row["truth"] == "synthetic" and not flag(obs["mono"], points["mono"])
             and flag(obs["onset_steered"], points["onset_steered"])]
    rooms = sorted({row["room"] for row, _ in noisy})
    metrics = {}
    for room in rooms:
        acoustic_rows = [row["acoustics"] for row, _ in noisy if row["room"] == room and row["target"] is not None]
        metrics[room] = {"accepted_onset": len(acoustic_rows),
                         **{key: median(item[key] for item in acoustic_rows) if acoustic_rows else None
                            for key in ("mono_snr_db", "steered_snr_db", "snr_gain_db", "target_error_db")}}
    totals = {}
    for label in ("genuine", "synthetic"):
        totals[label] = {arm: sum(flag(obs[arm], points[arm]) for row, obs in noisy if row["truth"] == label) for arm in ARMS}
    advance = {"more_ai_flags": totals["synthetic"]["onset_steered"] > totals["synthetic"]["mono"],
               "no_additional_noisy_false_flags": totals["genuine"]["onset_steered"] <= totals["genuine"]["mono"],
               "gains_across_both_rooms_and_two_families": {row["room"] for row, _ in gains} == set(rooms)
               and len({row["family"] for row, _ in gains}) >= 2,
               "no_additional_clean_or_native_false_flags": all(cell["onset_steered_flags"] <= cell["mono_flags"]
                   for key, cell in primary.items() if not key.startswith("center/") and key.endswith("/genuine")),
               "median_snr_gain_at_least_2db_in_each_room": all(cell["snr_gain_db"] is not None and cell["snr_gain_db"] >= 2
                                                             for cell in metrics.values())}
    result = {"schema": SCHEMA, "method": METHOD, "ancestry": lock["ancestry"], "scope": "controlled, previously exposed full-voice clips",
              "commitments": {name: prior.sha(args.run / f"{name}.json") for name in ("freeze", "roster", "selection", "points", "confirmation")},
              "selection": decisions(choose, points), "confirmation": primary, "confirmation_noisy_flags": totals,
              "confirmation_equal_false_alarm": {str(n): equal_false_alarm(noisy, n) for n in (0, 1, 2, 4)},
              "confirmation_acoustics": metrics, "confirmation_gain_groups": len({row["group"] for row, _ in gains}),
              "confirmation_gain_families": sorted({row["family"] for row, _ in gains}), "advance": advance,
              "run_resources": {role: {name: item[name] for name in ("forwards", "activity_frames", "seconds", "rss_bytes")}
                                for role, item in (("selection", selection), ("confirmation", held))}}
    write(args.run / "result.json", result)
    print(totals, advance, flush=True)


def audit(args):
    lock, rows = bound(args)
    reproduced, views, ancestry = build(args.source, args.archived)
    require(rows == reproduced and lock["ancestry"] == ancestry, "Regenerated source roster differs")
    for row, pair in zip(rows, views, strict=True):
        for arm, raw in zip(ARMS, pair, strict=True):
            require(raw == checked_pcm(args, row, arm), "Regenerated query differs")
    _, selected = ledger(args, "selection", rows)
    _, held = ledger(args, "confirmation", rows)
    result = read(args.run / "result.json")
    points = {arm: operating_point(selected, arm) for arm in ARMS}
    require(result["selection"] == decisions(selected, points) and result["confirmation"] == decisions(held, points)
            and result["confirmation_equal_false_alarm"] == {str(n): equal_false_alarm(
                [(row, obs) for row, obs in held if row["source"] == "center"], n) for n in (0, 1, 2, 4)}, "Result replay differs")
    print(f"Verified {len(rows)} rows, {len(rows)*2} exact query hashes, both operating points and four held equal-alarm readouts.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "selection", "fit", "confirmation", "report", "audit"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--archived", type=Path, default=ARCHIVED)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    args = parser.parse_args()
    if args.phase in ROLES:
        score(args, args.phase)
    else:
        {"freeze": freeze, "fit": fit, "report": report, "audit": audit}[args.phase](args)


if __name__ == "__main__":
    main()
