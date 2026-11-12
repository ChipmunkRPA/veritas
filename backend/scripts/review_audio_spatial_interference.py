"""Replay the spatial interference inputs and outcome arithmetic without model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np
from scipy.fft import irfft
from scipy.signal import istft, stft

from scripts.review_audio_mono_dereverb_transfer import (
    CLOCK,
    PAD,
    SAMPLES,
    WIDTH,
    WINDOW,
    analyze,
    decision,
    read,
    require,
)

RATE = 16_000
BLOCK = np.minimum(CLOCK // RATE, 2)
FREQUENCY = np.arange(WIDTH // 2 + 1) * RATE / WIDTH
PLANNED = {"archived": {"center": 68, "clean": 17, "native": 16},
           "transfer": {"center": 64, "clean": 16, "native": 4, "offset": 32}}
EMPTY = {"decision": "unassessed", "raw": None}


def digest(value):
    if isinstance(value, Path):
        with value.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    return hashlib.sha256(value).hexdigest()


def wave_pcm(path, channels, start=0, *, whole=False):
    with wave.open(str(path), "rb") as stream:
        require((stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) == (channels, 2, RATE),
                f"Unexpected source clock in {path.name}")
        require(not whole or stream.getnframes() == SAMPLES, f"Unexpected duration in {path.name}")
        stream.setpos(start)
        raw = stream.readframes(SAMPLES)
    require(len(raw) == SAMPLES * 2 * channels, f"Truncated source in {path.name}")
    return raw


def pcm(values):
    rounded = np.rint(values)
    require(np.min(rounded) >= -32768 and np.max(rounded) <= 32767, "Unexpected PCM clipping")
    return rounded.astype("<i2")


def room_inputs(first, second, freeze, mono, parent):
    for room, capture in freeze["publisher_mics"].items():
        first_source = parent["rooms"][room]
        require(capture["first_url"] == first_source["url"]
                and capture["second_url"] == first_source["url"].replace("Array1-01", "Array1-05")
                and capture["hashes"][0] == first_source["sha256"], "Unexpected publisher source")
        paths = (first / f"{room}.Array1-01.wav", second / f"{room}.Array1-05.wav")
        require([digest(path) for path in paths] == capture["hashes"]
                and [path.stat().st_size for path in paths] == capture["bytes"], "Publisher capture commitment differs")
    cuts = {}
    for row in read(mono / "roster.json"):
        if (row["role"], row["treatment"]) != ("transfer", "native"):
            continue
        room, mode = row["room"], row["mode"]
        start = row["source"]["source_start_sample"]
        a = wave_pcm(first / f"{room}.Array1-01.wav", 1, start)
        b = wave_pcm(second / f"{room}.Array1-05.wav", 1, start)
        require(a == (mono / "pcm" / f"{row['id']:03}.pcm").read_bytes() and digest(a) == row["sha256"],
                "First-microphone cut differs from the prior interval")
        cuts[room, mode, start] = np.stack((np.frombuffer(a, "<i2"), np.frombuffer(b, "<i2")), axis=1)
    rooms = set(freeze["publisher_mics"])
    require(len(rooms) == 2 and len(cuts) == 4
            and {(room, mode) for room, mode, _ in cuts} == {
                (room, mode) for room in rooms for mode in ("sequential", "simultaneous")
            }, "Unbalanced native cuts")
    return dict(sorted(cuts.items()))


def transfer_inputs(mono, cuts):
    voices = [row for row in read(mono / "roster.json")
              if (row["role"], row["treatment"]) == ("transfer", "clean")]
    require(len(voices) == 16 and len({row["group"] for row in voices}) == 16, "Unbalanced foreground groups")
    output = {("native", room, mode, start): (clip.tobytes(), None)
              for (room, mode, start), clip in cuts.items()}
    for i, row in enumerate(voices):
        raw = (mono / "pcm" / f"{row['id']:03}.pcm").read_bytes()
        require(len(raw) == SAMPLES * 2 and digest(raw) == row["sha256"], "Parent foreground bytes differ")
        voice = np.frombuffer(raw, "<i2").astype(float)
        center = np.broadcast_to(voice[:, None], (SAMPLES, 2))
        shifted = center.copy()
        shifted[:, i % 2] = np.pad(voice[:-4], (4, 0)) * 10**(-3 / 20)
        foreground = row["group"], row["family"], row["id"]
        variants = {}
        for j, (cut, signal) in enumerate(cuts.items()):
            for position, target in (("center", center), ("offset", shifted)):
                if position == "offset" and j not in (i % 2, 2 + (i + 1) % 2):
                    continue
                ratio = np.linalg.norm(target.mean(axis=1)) / np.linalg.norm(signal.mean(axis=1)) * 10**.25
                variants[(position, *foreground, *cut)] = (target, ratio * signal)
        peak = max(np.max(np.abs(a) + np.abs(b)) for a, b in variants.values())
        gain = min(1., 31_129 / peak)
        output[("clean", *foreground)] = pcm(gain * center).tobytes(), None
        for key, (target, noise) in variants.items():
            a, b = pcm(gain * target), pcm(gain * noise)
            output[key] = pcm(a.astype(np.int32) + b).tobytes(), a.tobytes()
    return output


def source_pcm(folder, archived, row, old_rows, old_scores, old_gates, transfer):
    name = f"{row['id']:03}"
    baseline = row["baseline"] or EMPTY
    mono = (folder / "mono" / f"{name}.pcm").read_bytes()
    position = row["placement"]
    if row["role"] == "archived":
        old = old_rows[row["id"]]
        require(row["archived_id"] == old["id"] == row["id"] and row["wav_sha256"] is None
                and all(row[key] == old.get(key) for key in ("room", "mode", "treatment", "group", "family"))
                and position == ("center" if old["treatment"] == "mixed" else old["treatment"])
                and row["baseline"] == old_scores[row["id"]].get("mono")
                and bool(row["baseline"]) == old_gates[row["id"]]["admitted"]["mono"],
                f"Archived identity or response differs for row {name}")
        path = archived / "wav" / f"{name}.wav"
        raw = wave_pcm(path, 2, whole=True)
        channels = np.frombuffer(raw, "<i2").reshape(SAMPLES, 2)
        require(digest(path) == old["wav_sha256"] and digest(raw) == old["stereo_sha256"]
                and [digest(channel.tobytes()) for channel in channels.T] == old["channels_sha256"]
                and digest(mono) == old["pcm_sha256"] and mono == (archived / "pcm" / f"{name}.pcm").read_bytes(),
                f"Archived PCM differs for row {name}")
        target = (wave_pcm(archived / "wav" / f"{old['clean_id']:03}.wav", 2, whole=True)
                  if "clean_id" in old else None)
    else:
        cut = (row["room"], row["mode"], row.get("native_start"))
        foreground = (row["group"], row["family"], row.get("parent_clean"))
        expected = {"native": "native", "clean": "clean", "center": "mixed", "offset": "mixed"}
        require(row["archived_id"] is None and row["baseline"] is None and row["treatment"] == expected[position],
                f"Unexpected transfer labels for row {name}")
        key = (position, *cut) if position == "native" else (position, *foreground)
        if position == "native":
            require(row["group"] == row["room"] and row["family"] == "genuine", "Unexpected native labels")
        elif position == "clean":
            require(cut == (None, None, None), "A clean voice received a room")
        else:
            key += cut
        expected_raw, target = transfer.pop(key)
        path = folder / "wav" / f"{name}.wav"
        raw = wave_pcm(path, 2, whole=True)
        summed = np.frombuffer(raw, "<i2").reshape(SAMPLES, 2).astype(np.int32).sum(axis=1)
        downmix = ((summed + 1) // 2).astype("<i2").tobytes()
        require(raw == expected_raw and digest(path) == row["wav_sha256"] and mono == downmix,
                f"Transfer scene or exact integer downmix differs for row {name}")
    require(digest(raw) == row["stereo_sha256"] and digest(mono) == row["mono_sha256"]
            and len(mono) == SAMPLES * 2 and (digest(target) if target is not None else None) == row["foreground_sha256"],
            f"Frozen PCM commitments differ for row {name}")
    if target is not None:
        require(target == (folder / "foreground" / f"{name}.pcm").read_bytes(),
                f"Local foreground bytes differ for row {name}")
    return np.frombuffer(raw, "<i2").astype(float).reshape(SAMPLES, 2), target, baseline


def components(stereo):
    left, right = stereo.T
    return analyze((left + right) / 2), analyze((left - right) / 2)


def fitted(stereo):
    mid, side = components(stereo)
    weights = np.zeros_like(mid)
    for block in range(3):
        train = block != BLOCK
        a, b = mid[:, train], side[:, train]
        cross = (a * b.conjugate()).mean(axis=1)
        x, y = (np.square(np.abs(z)).mean(axis=1) for z in (a, b))
        w = cross / (y + .03 * x + 1e-12)
        w *= 2 / np.maximum(2, np.abs(w))
        w[(np.abs(cross)**2 < .15 * x * y) | (x * y == 0) | (FREQUENCY < 150) | (FREQUENCY > 6500)] = 0
        weights[:, ~train] = w[:, None]
    return weights


def condition(stereo, weights):
    mid, side = components(stereo)
    samples = np.zeros(SAMPLES + 2 * PAD)
    scale = np.zeros_like(samples)
    for start, frame in zip(CLOCK, irfft((mid - .8 * weights * side).T), strict=True):
        samples[start:start + WIDTH] += frame * WINDOW * WINDOW.sum()
        scale[start:start + WIDTH] += WINDOW**2
    return samples[PAD:-PAD] / scale[PAD:-PAD]


def verify_conditioned(stereo, weights, filtered, expected, ordinal):
    rounded = pcm(filtered)
    if digest(rounded.tobytes()) == expected:
        return False
    # At a half-integer sample, floating-point ordering can change the int16 result.
    # Check exact bytes using the frozen library convention after checking the independent replay.
    _, _, spectrum = stft(stereo, fs=RATE, window="hann", nperseg=WIDTH, noverlap=384,
                          axis=0, boundary="zeros")
    left, right = spectrum[:, 0], spectrum[:, 1]
    _, canonical = istft((left + right) / 2 - .8 * weights * ((left - right) / 2), fs=RATE,
                          window="hann", nperseg=WIDTH, noverlap=384, boundary=True)
    exact = pcm(canonical)
    changed = exact != rounded
    require(np.allclose(canonical, filtered, rtol=0, atol=1e-8)
            and np.all(np.abs(filtered[changed] - np.floor(filtered[changed]) - .5) <= 1e-8)
            and digest(exact.tobytes()) == expected, f"Conditioned waveform hash differs for row {ordinal}")
    return True


def acoustic(stereo, target, weights, filtered):
    voice = np.frombuffer(target, "<i2").astype(float).reshape(SAMPLES, 2)
    noise = stereo - voice
    clean, residual = condition(voice, weights), condition(noise, weights)
    require(np.allclose(clean + residual, filtered, rtol=0, atol=1e-7), "Acoustic components do not recombine")
    rms = np.linalg.norm
    a, b = voice.mean(axis=1), noise.mean(axis=1)
    return {"snr_gain_db": 20 * math.log10((rms(clean) / rms(a)) / (rms(residual) / rms(b))),
            "voice_level_change_db": 20 * math.log10(rms(clean) / rms(a)),
            "voice_error_rms_db": 20 * math.log10(max(rms(clean - a) / rms(a), 1e-12))}


def audit(args):
    folder, archived, mono = args.folder, args.archived, args.mono_parent
    freeze, roster, scores, result = (read(folder / f"{name}.json")
                                      for name in ("freeze", "roster", "scores", "aggregate"))
    calibration = read(args.calibration)
    runner = Path(__file__).with_name("audio_spatial_interference.py")
    require(freeze["runner_sha256"] == digest(runner) and freeze["roster_sha256"] == digest(folder / "roster.json")
            and freeze["calibration_sha256"] == digest(args.calibration)
            and freeze["mono_parent_sha256"] == digest(mono / "freeze.json")
            and scores["freeze_sha256"] == digest(folder / "freeze.json"), "Study freeze commitments differ")
    require(all(digest(archived / f"{name}.json") == freeze["archived"][f"{name}_sha256"]
                for name in ("aggregate", "freeze", "gate", "roster", "scores")), "Archived commitments differ")
    previous, parent = read(archived / "freeze.json"), read(mono / "freeze.json")
    require(digest(mono / "roster.json") == parent["roster_sha256"]
            and all(record[key] == freeze[key] for record in (previous, parent)
                    for key in ("calibration_sha256", "incumbent_assets")), "Parent roster or model commitments differ")
    old_rows = read(archived / "roster.json")
    old_scores, old_gates = (read(archived / f"{name}.json")["rows"] for name in ("scores", "gate"))
    cuts = room_inputs(args.first_mics, args.second_mics, freeze, mono, parent)
    transfer = transfer_inputs(mono, cuts)
    placements = {role: dict(Counter(row["placement"] for row in roster if row["role"] == role)) for role in PLANNED}
    require(placements == PLANNED and len(roster) == len(old_rows) + len(transfer) == 217
            and len(old_rows) == len(old_scores) == len(old_gates) == 101
            and freeze["planned"] == dict(Counter(row["placement"] for row in roster)), "Unexpected case mix")
    counts, summaries, changes = Counter(), defaultdict(list), defaultdict(list)
    cells, evidence, paired = defaultdict(Counter), defaultdict(list), defaultdict(dict)
    for i, (row, score) in enumerate(zip(roster, scores["rows"], strict=True)):
        require(i == row["id"] == score["id"] and row["mono_sha256"] == score["mono_sha256"],
                f"Score binding differs for row {i}")
        stereo, target, prior = source_pcm(folder, archived, row, old_rows, old_scores, old_gates, transfer)
        baseline, alternative = score["baseline"], score["conditioned"]
        a, logit_a = decision(baseline["raw"], calibration)
        b, logit_b = decision(alternative["raw"], calibration)
        if row["role"] == "archived":
            require(baseline == prior, f"Archived decision differs for row {i}")
        else:
            require(baseline["decision"] == a or (a, baseline["decision"]) == ("original", "inconclusive"),
                    f"Unexpected incumbent calibration result for row {i}")
            counts.update(activity_tracks=1, classifier_forwards=int(a != "unassessed") + int(a == "original"))
        require(alternative["decision"] == b, f"Unexpected conditioned calibration result for row {i}")
        treatment = "unchanged"
        if np.any(stereo[:, 0] != stereo[:, 1]):
            weights = fitted(stereo)
            filtered = condition(stereo, weights)
            if target is not None:
                values = acoustic(stereo, target, weights, filtered)
                require(score.get("acoustic", {}).keys() == values.keys(), f"Acoustic fields differ for row {i}")
                counts["acoustic"] += 1
                for key, value in values.items():
                    difference = abs(score["acoustic"][key] - value)
                    require(difference <= 1e-7, f"Acoustic {key} differs for row {i}: {difference}")
                    summaries[f"{row['role']}/{row['placement']}/{row['room']}/{row['mode']}/{key}"].append(value)
                if row["role"] == "transfer":
                    group = row["parent_clean"], row["room"], row["mode"], row["native_start"]
                    paired[group][row["placement"]] = values
            if baseline["decision"] in {"ai", "inconclusive"}:
                treatment = "clipped" if np.min(np.rint(filtered)) < -32768 or np.max(np.rint(filtered)) > 32767 else "conditioned"
                if treatment == "conditioned":
                    ties = verify_conditioned(stereo, weights, filtered, score.get("conditioned_sha256"), i)
                    counts["waveforms_with_rounding_ties"] += ties
                    counts.update(activity_tracks=1, classifier_forwards=int(b != "unassessed"))
        require(treatment == score["treatment"] and (target is not None) == ("acoustic" in score)
                and (treatment == "conditioned" or ("conditioned_sha256" not in score and alternative == EMPTY)),
                f"Conditioned measurement coverage differs for row {i}")
        counts[treatment] += 1
        suppress = baseline["decision"] == "ai" and b in {"inconclusive", "original"} and logit_a - logit_b > math.log(2)
        rescue = baseline["decision"] == "inconclusive" and b == "ai"
        role, position = row["role"], row["placement"]
        truth = "genuine" if row["family"] == "genuine" else "synthetic"
        cells[f"{role}/{position}/{truth}"].update(
            {"planned": 1, f"baseline_{baseline['decision']}": 1, f"candidate_{b}": 1,
             "suppress": suppress, "rescue": rescue})
        for arm, changed, desired in (("suppress", suppress, "genuine"), ("rescue", rescue, "synthetic")):
            if changed:
                changes[f"{role}/{position}/{truth}/{arm}"].append(i)
                if truth == desired:
                    evidence[role, position, arm].append(row)
    require(not transfer, "A frozen transfer scene was omitted")
    require(counts["classifier_forwards"] == scores["classifier_forwards"]
            and 94 * counts["activity_tracks"] == scores["activity_frames"], "Inference receipt counts differ")
    advancement = {}
    for arm, other, limit in (("suppress", "synthetic", 2), ("rescue", "genuine", 1)):
        gains = evidence["transfer", "center", arm]
        advancement[arm] = (len({row["group"] for row in gains}) >= 3 and len({row["room"] for row in gains}) >= 2
                            and (arm == "suppress" or len({row["family"] for row in gains}) >= 2)
                            and all(cells[f"transfer/{geometry}/{other}"][arm] <= limit for geometry in ("center", "offset"))
                            and not any(cell[arm] for key, cell in cells.items() if key.split("/")[1] in {"clean", "native"}))
    group_counts = {}
    for field, name in (("group", "gain_groups"), ("room", "gain_rooms"), ("family", "gain_families")):
        group_counts[name] = {"/".join(key): len({row[field] for row in rows}) for key, rows in evidence.items()
                              if rows or field == "group"}
    require(dict(cells) == result["cells"] and advancement == result["advance"]
            and all(result[name] == value for name, value in group_counts.items()), "Outcome summary differs")
    require(summaries.keys() == result["median_acoustic"].keys()
            and all(abs(median(values) - result["median_acoustic"][key]) <= 1e-7 for key, values in summaries.items()),
            "Acoustic summary differs")
    require(result["scores_sha256"] == digest(folder / "scores.json")
            and all(result[key] == freeze[key] for key in ("method", "schema", "planned"))
            and all(result[key] == scores[key] for key in ("freeze_sha256", "classifier_forwards", "activity_frames",
                                                          "elapsed_s", "rss_bytes")), "Aggregate receipt differs")
    matched = [group for group in paired.values() if "offset" in group]
    require(len(matched) == 32 and all("center" in pair for pair in matched), "Offset scenes lost matched center controls")
    matched_summary = {}
    for metric in ("snr_gain_db", "voice_level_change_db", "voice_error_rms_db"):
        center = [pair["center"][metric] for pair in matched]
        offset = [pair["offset"][metric] for pair in matched]
        matched_summary[metric] = {"center": median(center), "offset": median(offset),
                                   "paired_offset_minus_center": median(b - a for a, b in zip(center, offset, strict=True))}
    return {"status": "verified_without_model_inference", "cases": len(roster), "placements": placements,
            "counts": dict(counts), "advance": advancement, "changed_rows": dict(changes),
            "matched_geometry": {"pairs": len(matched), "median_acoustic": matched_summary},
            "freeze_sha256": scores["freeze_sha256"], "scores_sha256": result["scores_sha256"],
            "aggregate_sha256": digest(folder / "aggregate.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    for name in ("archived", "mono-parent", "first-mics", "second-mics"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=Path(__file__).parents[1]
                        / "app/detectors/audio/calibrations/local-speech-v3.json")
    print(json.dumps(audit(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
