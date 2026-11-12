"""Compare a pinned speech-enhancement view with the current local voice check."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import statistics
import subprocess
import tarfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import audio_strong_traffic_view as earlier

ROOT, RATE, COUNT = earlier.ROOT, earlier.RATE, earlier.COUNT
read, sha, write_new = earlier.read, earlier.sha, earlier.write_new
MODEL = "gtcrn_simple.onnx"
REVISION = "502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73"
MODEL_SHA = "b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87"
LICENSE_SHA = "c467165e5860b4a7494ef4a6e2788e4115d95b5b8989bb5f86287089adddf794"
SUNDAY_SHA1 = "01c8dca37e8e7aa6be9fc37851951c54cdec3553"
SUNDAY_STARTS = tuple(range(45 * RATE, 69 * RATE, COUNT))
SALT = b"chipmunk/traffic-speech-enhancement/2026-11-11/"
COUNTS = {"genuine": 10, "clone": 3, "expressive": 3, "stock": 3}
FAMILIES = ("clone", "expressive", "stock")
FFT, HOP = 512, 256
FRAMES = 1 + COUNT // HOP
INPUTS = {"mix": [1, 257, 1, 2], "conv_cache": [2, 1, 16, 16, 33],
          "tra_cache": [2, 3, 1, 1, 16], "inter_cache": [2, 1, 33, 16]}
RULE = {"minimum_suppression_db": 6, "minimum_true_additions": 6, "minimum_gained_voice_groups": 3,
        "minimum_tapes_with_gain": 2, "maximum_false_additions": 1, "maximum_clean_additions": 0,
        "maximum_isolated_additions": 0, "minimum_newly_assessed_synthetic": 3,
        "minimum_tapes_with_new_assessment": 2, "minimum_clean_median_si_sdr_db": 12,
        "minimum_median_noisy_si_sdr_gain_db": 3, "minimum_tapes_per_level_with_si_sdr_gain": 3}


def order(value):
    return sha(SALT + str(value).encode())


def decode(body, kind):
    return subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", kind, "-i", "pipe:0", "-ar", str(RATE),
                           "-ac", "1", "-f", "s16le", "pipe:1"], input=body, capture_output=True, check=True, timeout=20).stdout


def source_roster(archive, previous, older):
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "md5").hexdigest() != earlier.MD5:
            raise ValueError("The LibriSpeech archive differs from the publisher checksum")
    prior_receipt = read(earlier.PANEL / "audio-strong-traffic-view-transfer-2026-11-11.json")
    if (sha(previous / "freeze.json") != prior_receipt["commitments"]["freeze_sha256"]
            or sha(previous / "roster.json") != read(previous / "freeze.json")["roster_sha256"]
            or sha(older / "freeze.json") != read(earlier.PREVIOUS)["freeze_sha256"]
            or sha(older / "cases.json") != read(older / "freeze.json")["cases_sha256"]):
        raise ValueError("The preceding traffic or two-background source roster changed")
    prior = read(previous / "roster.json") + read(older / "cases.json")
    seen_pcm = {row["source_pcm_sha256"] for row in prior if "source_pcm_sha256" in row}
    speakers = {row["speaker"] for row in prior if row.get("speaker")}
    with gzip.open(earlier.CATALOGUE, "rt") as stream:
        catalogue = json.load(stream)["sources"]
    old_speakers = {row["canonical_speaker_sha256"] for row in catalogue}
    old_pcm = {row[key] for row in catalogue for key in ("sha256", "prefix_pcm_sha256", "production_pcm_sha256")}
    manifests, sources, commitments = [], [], {}
    receipt = read(earlier.PREP)
    for role in ("development", "holdout"):
        path = earlier.clips.sources.MEDIA / role / "private-manifest.json"
        commitments[role] = sha(path)
        if commitments[role] != receipt["roles"][role]["private_manifest_sha256"]:
            raise ValueError("The prepared foreground source manifest changed")
        for row in read(path)["foregrounds"]:
            if row["lane"] in FAMILIES:
                raw = earlier.clips.exact_source(row, role)
                manifests.append((role, row, raw))
    seen_groups = {row["group"] for row in prior if row.get("group")}
    for _, row, raw in manifests:
        if sha(raw) in seen_pcm:
            speakers.add(row["voice"])
            seen_groups.add(row["group_id"])
    for family in FAMILIES:
        current = []
        for role, row, raw in sorted(manifests, key=lambda item: (item[0] != "holdout", order(item[1]["slot"]))):
            if (row["lane"] != family or row["voice"] in speakers or row["group_id"] in seen_groups
                    or {row["original_sha256"], row["prefix_pcm_sha256"], sha(raw)} & old_pcm):
                continue
            current.append(({"family": family, "group": row["group_id"], "speaker": row["voice"], "role": role,
                             "source_sha256": row["original_sha256"], "source_pcm_sha256": sha(raw)}, raw))
            speakers.add(row["voice"])
            seen_groups.add(row["group_id"])
            if len(current) == COUNTS[family]:
                break
        if len(current) != COUNTS[family]:
            raise ValueError(f"Too few distinct {family} foreground groups")
        sources.extend(current)
    readers = []
    with tarfile.open(archive, "r:gz") as tar:
        by_reader = {}
        for member in tar:
            match = re.fullmatch(r"LibriSpeech/test-clean/(\d+)/\d+/\d+-\d+-\d+\.flac", member.name)
            if match and member.isfile() and 0 < member.size <= 2_000_000:
                speaker = f"librispeech:{match[1]}"
                if speaker not in speakers and sha(speaker.encode()) not in old_speakers:
                    by_reader.setdefault(speaker, []).append(member)
        for speaker, members in sorted(by_reader.items(), key=lambda item: order(item[0])):
            for member in sorted(members, key=lambda item: order(item.name)):
                with tar.extractfile(member) as stream:
                    encoded = stream.read()
                body = decode(encoded, "flac")
                if len(body) < 2 * (COUNT + 8000):
                    continue
                start = (len(body) // 2 - COUNT) // 2
                raw = body[2 * start:2 * (start + COUNT)]
                if {sha(encoded), sha(raw), earlier.clips.float_pcm_sha(raw)} & (old_pcm | seen_pcm):
                    continue
                readers.append(({"family": "genuine", "group": speaker, "speaker": speaker, "role": "new_reader",
                                 "source_sha256": sha(encoded), "source_pcm_sha256": sha(raw),
                                 "member": member.name, "start_sample": start}, raw))
                break
            if len(readers) == COUNTS["genuine"]:
                break
    if len(readers) != COUNTS["genuine"]:
        raise ValueError("Too few new LibriSpeech reader groups")
    return readers + sources, {"librispeech_archive": sha(archive), "prior_traffic_roster": sha(previous / "roster.json"),
                              "prior_two_background_roster": sha(older / "cases.json"),
                              "known_speaker_catalogue": sha(earlier.CATALOGUE), "synthetic_manifests": commitments}


def noise_roster(folder):
    receipts, hashes = earlier.background.result_set()
    known = earlier.traffic_sources(receipts, hashes)
    tapes = {f"archived_{key}": [(row["start_sample"], raw) for row, raw in rows] for key, rows in known.items()}
    source, metadata = folder / "sunday.ogg", folder / "sunday-source.json"
    records = [r for page in read(metadata)["query"]["pages"].values() for r in page["imageinfo"]]
    record, = records
    raw = source.read_bytes()
    if (record["sha1"] != SUNDAY_SHA1 or hashlib.sha1(raw).hexdigest() != SUNDAY_SHA1
            or record["size"] != len(raw) or record["extmetadata"]["LicenseShortName"]["value"] != "Public domain"):
        raise ValueError("The new traffic recording differs from its publisher record")
    pcm = decode(raw, "ogg")
    tapes["new_sunday"] = [(start, pcm[2 * start:2 * (start + COUNT)]) for start in SUNDAY_STARTS]
    if any(len(part) != 2 * COUNT or not any(part) for rows in tapes.values() for _, part in rows):
        raise ValueError("Incomplete traffic excerpt")
    return tapes, {**hashes, "new_traffic_media": sha(raw), "new_traffic_publisher_record": sha(metadata),
                   "new_traffic_full_pcm": sha(pcm)}


def binding(folder):
    if sha(folder / MODEL) != MODEL_SHA or sha(folder / "LICENSE") != LICENSE_SHA:
        raise ValueError("The checkpoint or publisher MIT notice differs")
    paths = (Path(__file__).resolve(), Path(earlier.__file__).resolve(), earlier.CALIBRATION,
             ROOT / "backend/app/detectors/audio/upgraded.py", ROOT / "backend/app/detectors/audio/calibration.py",
             ROOT / "backend/app/detectors/video/adapters/audio.py", ROOT / "backend/scripts/run_audio_specialists_v15.py")
    return {"code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths}, "current_assets": earlier.ASSETS,
            "enhancement": {"revision": REVISION, "model_sha256": MODEL_SHA, "license_sha256": LICENSE_SHA,
                            "license": "MIT", "runtime": "onnxruntime 1.23.2 CPU single-thread",
                            "fft": FFT, "hop": HOP, "window": "square root periodic Hann", "padding": "reflect",
                            "samples": COUNT, "recurrent_cache": "reset per input", "post_gain": "none"},
            "rule": RULE, "conditions": ["clean", "-5", "-10", "isolated"],
            "scope": "Preserve directional current results. On original inconclusive/unassessed inputs, enhanced AI "
                     "can add a flag only if the enhanced view meets the current gate and observed suppression point."}


def pcm(wave):
    return np.clip(np.rint(wave * 32768), -32768, 32767).astype("<i2").tobytes()


def float_pcm(raw):
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768


def prepare(folder, archive, previous, older):
    freeze = binding(folder)
    voices, inputs = source_roster(archive, previous, older)
    tapes, noise = noise_roster(folder)
    (folder / "pcm").mkdir()
    (folder / "noise").mkdir()
    cases = []

    def save(raw, fields):
        i = len(cases)
        (folder / "pcm" / f"{i:03}.pcm").write_bytes(raw)
        cases.append({"id": i, **fields, "pcm_sha256": sha(raw)})

    for cuts in tapes.values():
        cuts.sort(key=lambda pair: order(sha(pair[1])))
        for _, raw in cuts:
            (folder / "noise" / f"{sha(raw)}.pcm").write_bytes(raw)
    for i, (row, raw) in enumerate(voices):
        clean_id = len(cases)
        wave = float_pcm(raw).astype(np.float64)
        save(raw, {**row, "condition": "clean", "tape": None, "clean_id": clean_id})
        for tape, cuts in sorted(tapes.items()):
            start, background = cuts[i % len(cuts)]
            samples = float_pcm(background).astype(np.float64)
            for level in (-5, -10):
                mixture = wave + samples * np.linalg.norm(wave) / np.linalg.norm(samples) * 10 ** (-level / 20)
                gain = min(1., .95 / float(np.max(np.abs(mixture))))
                save(pcm(mixture * gain), {**row, "condition": str(level), "tape": tape, "clean_id": clean_id,
                                          "noise_sha256": sha(background), "noise_start_sample": start, "gain": gain})
    for tape, cuts in sorted(tapes.items()):
        for start, raw in cuts[:2]:
            save(raw, {"condition": "isolated", "family": "background", "group": tape, "tape": tape,
                       "noise_sha256": sha(raw), "noise_start_sample": start})
    write_new(folder / "roster.json", cases)
    write_new(folder / "freeze.json", {**freeze, "schema": "chipmunk.audio-traffic-enhancement-freeze/1",
                                      "sources": {**inputs, **noise}, "roster_sha256": sha(folder / "roster.json"),
                                      "planned": len(cases), "foreground_groups": len(voices), "traffic_tapes": sorted(tapes)})
    print(f"Frozen {len(cases)} exact views from {len(voices)} foreground groups and {len(tapes)} traffic recordings", flush=True)


def frozen(folder):
    lock, rows = read(folder / "freeze.json"), read(folder / "roster.json")
    if (any(lock[key] != value for key, value in binding(folder).items())
            or lock["roster_sha256"] != sha(folder / "roster.json")
            or [row["id"] for row in rows] != list(range(lock["planned"]))):
        raise ValueError("Frozen method, checkpoint or case roster changed")
    return lock, rows


def load(folder, row, *, enhanced=False):
    key = "enhanced_sha256" if enhanced else "pcm_sha256"
    raw = (folder / ("enhanced" if enhanced else "pcm") / f"{row['id']:03}.pcm").read_bytes()
    if len(raw) != 2 * COUNT or sha(raw) != row[key]:
        raise ValueError("The exact sample window changed")
    return float_pcm(raw)


def signal_spectra(wave):
    pad = np.pad(wave, (FFT // 2, FFT // 2), mode="reflect")
    window = np.sqrt(.5 - .5 * np.cos(2 * np.pi * np.arange(FFT) / FFT)).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(pad, FFT)[::HOP]
    return np.fft.rfft(frames * window, axis=1), window


def overlap_add(spectra, window):
    frames = np.fft.irfft(spectra, n=FFT, axis=1) * window
    length = (len(frames) - 1) * HOP + FFT
    out, weights = np.zeros(length), np.zeros(length)
    for index, frame in enumerate(frames):
        left = index * HOP
        out[left:left + FFT] += frame
        weights[left:left + FFT] += window**2
    keep = slice(FFT // 2, FFT // 2 + COUNT)
    return out[keep] / weights[keep]


def denoise(wave, session):
    spectra, window = signal_spectra(wave)
    states = {name: np.zeros(shape, dtype=np.float32) for name, shape in INPUTS.items() if name != "mix"}
    output = np.empty_like(spectra)
    for index, item in enumerate(spectra):
        sample = np.stack((item.real, item.imag), axis=-1).astype(np.float32)[None, :, None, :]
        out, *cache = session.run(None, {"mix": sample, **states})
        states = dict(zip(states, cache, strict=True))
        output[index] = out[0, :, 0, 0] + 1j * out[0, :, 0, 1]
    result = overlap_add(output, window)
    if len(spectra) != FRAMES or result.shape != wave.shape or not np.isfinite(result).all():
        raise ValueError("Enhanced waveform did not preserve its exact sample clock")
    return result


def db_ratio(numerator, denominator):
    return float(10 * np.log10(max(1e-18, numerator) / max(1e-18, denominator)))


def energy(wave):
    return float(np.dot(wave.astype(np.float64), wave.astype(np.float64)))


def si_sdr(estimated, truth):
    ref = truth.astype(np.float64) - np.mean(truth, dtype=np.float64)
    value = estimated.astype(np.float64) - np.mean(estimated, dtype=np.float64)
    matched = ref * (np.dot(value, ref) / energy(ref))
    return db_ratio(energy(matched), energy(value - matched))


def enhance(folder):
    import onnxruntime as ort

    _, rows = frozen(folder)
    earlier.network_free()
    if ort.__version__ != "1.23.2":
        raise ValueError("Use the frozen ONNX Runtime version")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(folder / MODEL), sess_options=options, providers=["CPUExecutionProvider"])
    if {r.name: r.shape for r in session.get_inputs()} != INPUTS:
        raise ValueError("Unexpected streaming enhancement input signature")
    (folder / "enhanced").mkdir()
    observations, repeated = [], False
    for row in rows:
        original = load(folder, row)
        enhanced = denoise(original, session)
        raw = pcm(enhanced)
        if not repeated:
            if not np.array_equal(enhanced, denoise(original, session)):
                raise ValueError("First exact enhancement repeat differs")
            repeated = True
        (folder / "enhanced" / f"{row['id']:03}.pcm").write_bytes(raw)
        wave = float_pcm(raw)
        item = {"id": row["id"], "enhanced_sha256": sha(raw), "suppression_db": db_ratio(energy(original), energy(wave)),
                "clipped_samples": int(np.count_nonzero(np.abs(enhanced) >= 1.))}
        if "clean_id" in row:
            truth = load(folder, rows[row["clean_id"]])
            item["enhanced_si_sdr_db"] = si_sdr(wave, truth)
            item["si_sdr_gain_db"] = None if row["condition"] == "clean" else item["enhanced_si_sdr_db"] - si_sdr(original, truth)
        observations.append(item)
        if len(observations) % 30 == 0:
            print(f"Enhanced {len(observations)}/{len(rows)} exact windows", flush=True)
    write_new(folder / "enhancement.json", {"freeze_sha256": sha(folder / "freeze.json"), "frames_per_view": FRAMES,
                                             "onnx_forwards": (len(rows) + 1) * FRAMES, "rows": observations})


def score(folder):
    from app.detectors.audio import calibration
    from app.detectors.audio.upgraded import SpeechActivity
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from app.engine.boundary import Deadline
    from scripts.audio_noise_speech_gate_v256 import CountedVAD
    from scripts.run_audio_specialists_v15 import load_voice

    lock, rows = frozen(folder)
    conditioned = read(folder / "enhancement.json")
    if conditioned["freeze_sha256"] != sha(folder / "freeze.json"):
        raise ValueError("Enhancement receipt differs from the freeze")
    earlier.network_free()
    voice = load_voice({"weights": str(earlier.WEIGHTS), "assets": lock["current_assets"]})
    activity = SpeechActivity(earlier.WEIGHTS / "silero_vad.jit")
    activity.model = counted_gate = CountedVAD(activity.model)
    detector = PellaVoiceDetector()
    count, observations = 0, []

    def counted(*_):
        nonlocal count
        count += 1

    def check(wave):
        ctx = SimpleNamespace(profile="full", lease=SimpleNamespace(get=lambda _: SimpleNamespace(obj=activity)),
                              outputs=SimpleNamespace(audio=AudioTrack("analysed", wave, len(wave))))
        selected = detector._windows(ctx)
        if not selected:
            return {"decision": "unassessed", "raw": None}
        if len(selected) != 1 or selected[0][0] != 0 or not np.array_equal(selected[0][1], wave):
            raise ValueError("Production speech gate selected a different input window")
        raw, = voice.predict([wave])
        result = calibration.evaluate(raw, wave, selected, voice, Deadline(120))
        return {"decision": result["decision"], "raw": float(raw), "probe": result["probe_status"],
                "stability_change": result["stability_log_odds_change"]}

    hook = voice.register_forward_pre_hook(counted)
    try:
        for row, receipt in zip(rows, conditioned["rows"], strict=True):
            if row["id"] != receipt["id"]:
                raise ValueError("Enhanced receipt order differs from the source roster")
            current = check(load(folder, row))
            alternative = check(load(folder, receipt, enhanced=True))
            observations.append({"id": row["id"], "pcm_sha256": row["pcm_sha256"],
                                 "enhanced_sha256": receipt["enhanced_sha256"], "current": current, "enhanced": alternative})
            if len(observations) % 15 == 0:
                print(f"Scored {len(observations)}/{len(rows)} exact pairs ({count} voice forwards)", flush=True)
    finally:
        hook.remove()
    write_new(folder / "scores.json", {"freeze_sha256": sha(folder / "freeze.json"),
                                       "enhancement_sha256": sha(folder / "enhancement.json"),
                                       "voice_forwards": count, "speech_gate_forwards": counted_gate.calls, "rows": observations})


def readout(folder):
    lock, rows = frozen(folder)
    enhancement, scores = read(folder / "enhancement.json"), read(folder / "scores.json")
    freeze_sha = sha(folder / "freeze.json")
    if (enhancement["freeze_sha256"] != freeze_sha or scores["freeze_sha256"] != freeze_sha
            or scores["enhancement_sha256"] != sha(folder / "enhancement.json")):
        raise ValueError("Readout stages differ from their frozen inputs")
    cells, scene_gains = {}, {}
    true, false, new_clean, new_isolated, newly_assessed, gains_clean = [], [], [], [], [], []
    for row, altered, responses in zip(rows, enhancement["rows"], scores["rows"], strict=True):
        if (row["id"], row["pcm_sha256"], altered["enhanced_sha256"]) != (responses["id"], responses["pcm_sha256"], responses["enhanced_sha256"]) or altered["id"] != row["id"]:
            raise ValueError("Recorded response does not match the frozen waveform pair")
        orig, new = (responses[key]["decision"] for key in ("current", "enhanced"))
        final = "ai" if orig in {"inconclusive", "unassessed"} and new == "ai" and altered["suppression_db"] >= RULE["minimum_suppression_db"] else orig
        kind, condition, tape = row["family"], row["condition"], row["tape"]
        for group in {(condition, "all", kind), (condition, tape, kind)}:
            cell = cells.setdefault(group, {name: Counter() for name in ("current", "enhanced", "combined")})
            for arm, value in zip(cell, (orig, new, final), strict=True):
                cell[arm][value] += 1
        if condition in {"-5", "-10"}:
            scene_gains.setdefault((condition, tape), []).append(altered["si_sdr_gain_db"])
            if orig == "unassessed" and new != "unassessed" and kind in FAMILIES:
                newly_assessed.append(row)
        if condition == "clean":
            gains_clean.append(altered["enhanced_si_sdr_db"])
        if final == "ai" and orig != "ai":
            if condition == "clean":
                new_clean.append(row)
            elif condition == "isolated":
                new_isolated.append(row)
            elif kind == "genuine":
                false.append(row)
            else:
                true.append(row)
    medians = {key: statistics.median(values) for key, values in scene_gains.items()}
    checks = {"true_additions": len(true) >= RULE["minimum_true_additions"],
              "gained_voice_groups": len({r["group"] for r in true}) >= RULE["minimum_gained_voice_groups"],
              "gained_tapes": len({r["tape"] for r in true}) >= RULE["minimum_tapes_with_gain"],
              "false_additions": len(false) <= RULE["maximum_false_additions"],
              "clean_additions": len(new_clean) <= RULE["maximum_clean_additions"],
              "isolated_additions": len(new_isolated) <= RULE["maximum_isolated_additions"],
              "newly_assessed": len(newly_assessed) >= RULE["minimum_newly_assessed_synthetic"],
              "newly_assessed_tapes": len({r["tape"] for r in newly_assessed}) >= RULE["minimum_tapes_with_new_assessment"],
              "clean_fidelity": statistics.median(gains_clean) >= RULE["minimum_clean_median_si_sdr_db"],
              "noisy_fidelity": all(sum(value >= RULE["minimum_median_noisy_si_sdr_gain_db"]
                                        for (level, _), value in medians.items() if level == condition)
                                    >= RULE["minimum_tapes_per_level_with_si_sdr_gain"] for condition in ("-5", "-10"))}
    public = {"schema": "chipmunk.audio-traffic-enhancement-evaluation/1", "freeze_sha256": freeze_sha,
              "freeze": lock, "enhancement_sha256": sha(folder / "enhancement.json"), "score_sha256": sha(folder / "scores.json"),
              "resource_counts": {"onnx_forwards": enhancement["onnx_forwards"], "voice_forwards": scores["voice_forwards"],
                                  "speech_gate_forwards": scores["speech_gate_forwards"]},
              "acoustic": {"median_clean_si_sdr_db": statistics.median(gains_clean),
                           "median_noisy_si_sdr_gains_db": {f"{level}:{tape}": value for (level, tape), value in medians.items()},
                           "clipped_samples": sum(r["clipped_samples"] for r in enhancement["rows"])},
              "conditional": {"new_true_ai": len(true), "gained_groups": len({r["group"] for r in true}),
                              "gained_tapes": len({r["tape"] for r in true}), "new_false_ai": len(false),
                              "new_clean_ai": len(new_clean), "new_isolated_ai": len(new_isolated),
                              "enhanced_newly_assessed_synthetic": len(newly_assessed),
                              "enhanced_newly_assessed_tapes": len({r["tape"] for r in newly_assessed}),
                              "checks": checks, "advance": all(checks.values())},
              "counts": [{"condition": condition, "tape": tape, "family": kind, **values}
                         for (condition, tape, kind), values in sorted(cells.items(), key=lambda item: tuple(map(str, item[0])))]}
    write_new(folder / "aggregate.json", public)
    print(json.dumps(public["conditional"], sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "enhance", "score", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--librispeech", type=Path)
    parser.add_argument("--prior-traffic", type=Path)
    parser.add_argument("--prior-background", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        if not all((args.librispeech, args.prior_traffic, args.prior_background)):
            parser.error("Freeze needs the publisher archive and both prior runs")
        prepare(args.folder, args.librispeech, args.prior_traffic, args.prior_background)
    else:
        {"enhance": enhance, "score": score, "readout": readout}[args.action](args.folder)


if __name__ == "__main__":
    main()
