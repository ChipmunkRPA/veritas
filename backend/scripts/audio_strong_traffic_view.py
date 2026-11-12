"""Test a frozen corroborating voice score on source-separated traffic mixtures."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import audio_matched_background_panel_v248 as background
from scripts import audio_paired_transfer_v230 as head
from scripts import audio_short_edit_cases_v172 as clips
from scripts.audio_waveform_fallback_transfer import sha, write_new

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "docs/evaluation-panels"
WEIGHTS = ROOT / "models/weights/audio-upgrade"
CALIBRATION = ROOT / "backend/app/detectors/audio/calibrations/local-speech-v3.json"
CATALOGUE = PANEL / "audio-v9-identity-plan.json.gz"
PREVIOUS = PANEL / "audio-wavlm-two-background-transfer-2026-11-11.json"
PREP = PANEL / "audio-short-edit-materialization-complete-v169c.json"
MD5 = "32fa31d27d2e1cad72775fee3f4849a9"
RATE, COUNT = 16_000, 48_000
N_GENUINE = 10
QUOTAS = {"clone": 3, "expressive": 3, "stock": 2}
SALT = b"chipmunk/strong-traffic-noise-view/2026-11-11/"
ASSETS = {"pellav2_detector.pt": "d21aedfb52e2f87706c2753715a29665482b0f4a034800c31579df5d458559ac",
          "silero_vad.jit": "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720",
          "w2v-bert-config.json": "f5572bd5998b68182e9c328a43127ed21fed687f6910497136b91a4e3b0e3675"}
RULE = {"raw_margin": .05, "minimum_correct_additions": 6, "minimum_gained_source_groups": 3,
        "minimum_traffic_recording_groups_with_gain": 2, "maximum_new_false_flags": 1,
        "maximum_new_clean_flags": 0, "minimum_genuine_assessed_minus5_per_traffic": 5}


def read(path):
    return json.loads(path.read_text())


def order(value):
    return sha(SALT + str(value).encode())


def freeze_context():
    _, points = head.fit_context()
    return {
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
            Path(__file__).resolve(), Path(head.__file__).resolve(), Path(head.native.__file__).resolve(),
            ROOT / "backend/app/detectors/audio/calibration.py",
            ROOT / "backend/app/detectors/audio/upgraded.py",
            ROOT / "backend/app/detectors/video/adapters/audio.py",
            ROOT / "backend/scripts/run_audio_specialists_v15.py",
        )},
        "assets": ASSETS, "calibration_sha256": sha(CALIBRATION),
        "head_sha256": sha(head.learner.TRIAL / "fitted.json"),
        "points_sha256": sha(head.learner.TRIAL / "calibration.json"),
        "encoder": head.native.checkpoint(),
        "candidate_points": points[head.METHOD], "rule": RULE,
        "transform": {"fft": 512, "hop": 128, "power_floor_quantile": .20, "subtract_multiplier": 1.5,
                      "minimum_power_ratio": .16, "phase": "original", "pcm": "same-length signed 16-bit"},
        "scope": "Only incumbent admitted and inconclusive can change to AI. The filtered candidate needs the "
                 "unchanged AI cutoff and the ordinary paired score must be within 0.05 of the cutoff or higher.",
    }


def sources(archive, earlier):
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "md5").hexdigest() != MD5:
            raise ValueError("Publisher test-clean archive checksum differs")
    prior_results = read(PREVIOUS)
    if sha(earlier / "freeze.json") != prior_results["freeze_sha256"]:
        raise ValueError("Wrong prior transfer freeze")
    if sha(earlier / "cases.json") != read(earlier / "freeze.json")["cases_sha256"]:
        raise ValueError("Wrong prior transfer case roster")
    used = {row["source_pcm_sha256"] for row in read(earlier / "cases.json")}
    with gzip.open(CATALOGUE, "rt") as stream:
        catalogue = json.load(stream)
    forbidden_speakers = {row["canonical_speaker_sha256"] for row in catalogue["sources"]}
    forbidden_hashes = {row[key] for row in catalogue["sources"]
                        for key in ("sha256", "prefix_pcm_sha256", "production_pcm_sha256")}
    inputs = clips.sources.MEDIA / "holdout/private-manifest.json"
    manifest = read(inputs)
    if sha(inputs) != read(PREP)["roles"]["holdout"]["private_manifest_sha256"]:
        raise ValueError("Wrong publisher source qualification manifest")
    forbidden_speakers |= {sha(row["voice"].encode()) for row in manifest["foregrounds"] if row["voice"]}
    by_lane = {}
    for lane in QUOTAS:
        candidates = [(row, clips.exact_source(row, "holdout")) for row in manifest["foregrounds"] if row["lane"] == lane]
        seen = {row["voice"] for row, raw in candidates if sha(raw) in used}
        seen_groups = {row["group_id"] for row, raw in candidates if sha(raw) in used}
        selected = []
        for row, raw in sorted(candidates, key=lambda value: order(value[0]["slot"])):
            if (row["voice"] in seen or row["group_id"] in seen_groups or row["original_sha256"] in forbidden_hashes
                    or row["prefix_pcm_sha256"] in forbidden_hashes or sha(raw) in used):
                continue
            selected.append(({"family": lane, "speaker": row["voice"], "group": row["group_id"],
                              "source_sha256": row["original_sha256"], "source_pcm_sha256": sha(raw)}, raw))
            seen.add(row["voice"])
            seen_groups.add(row["group_id"])
            if len(selected) == QUOTAS[lane]:
                break
        if len(selected) != QUOTAS[lane]:
            raise ValueError(f"Insufficient source-separated {lane} groups")
        by_lane[lane] = selected
    genuine = []
    with tarfile.open(archive, mode="r:gz") as members:
        groups = {}
        for member in members:
            if match := re.fullmatch(r"LibriSpeech/test-clean/(\d+)/(\d+)/(\d+-\d+-\d+).flac", member.name):
                speaker = match[1]
                if member.isfile() and 0 < member.size <= 2_000_000 and sha(f"librispeech:{speaker}".encode()) not in forbidden_speakers:
                    groups.setdefault(speaker, []).append(member)
        for speaker in sorted(groups, key=order):
            for member in sorted(groups[speaker], key=lambda item: order(item.name)):
                stream = members.extractfile(member)
                with stream:
                    encoded = stream.read()
                payload = subprocess.run(
                    ["ffmpeg", "-nostdin", "-v", "error", "-f", "flac", "-i", "pipe:0", "-ar", "16000", "-ac", "1",
                     "-f", "s16le", "pipe:1"], input=encoded, capture_output=True, check=True, timeout=10,
                ).stdout
                if len(payload) < 2 * (COUNT + 8000):
                    continue
                offset = (len(payload) // 2 - COUNT) // 2
                raw = payload[2 * offset:2 * (offset + COUNT)]
                if ({sha(encoded), sha(raw), clips.float_pcm_sha(raw)} & forbidden_hashes) or sha(raw) in used:
                    continue
                genuine.append(({"family": "genuine", "speaker": f"librispeech:{speaker}", "group": f"libri:{speaker}",
                                 "source_sha256": sha(encoded), "source_pcm_sha256": sha(raw),
                                 "native_start": offset, "original_member": member.name}, raw))
                break
            if len(genuine) == N_GENUINE:
                break
    if len(genuine) != N_GENUINE:
        raise ValueError("Insufficient new publisher reader groups")
    rows = [*genuine, *(item for values in by_lane.values() for item in values)]
    if len({row["speaker"] for row, _ in rows}) != len(rows):
        raise ValueError("Foreground speaker identities overlap")
    return rows, {"librispeech_archive_sha256": sha(archive), "short_edit_holdout_sha256": sha(inputs),
                  "previous_transfer_roster_sha256": sha(earlier / "cases.json"), "old_catalogue_sha256": sha(CATALOGUE)}


def foreground(raw):
    return np.frombuffer(raw, dtype="<i2").astype(np.float64)


def traffic_sources(results, hashes):
    commons = background.background
    sources = commons.load_sources()
    grouped = {}
    for row in background.evidence_packet(results, hashes):
        if row["environment"] == "traffic":
            grouped.setdefault(row["slot"], []).append(row)
    traffic = {}
    for slot, rows in grouped.items():
        source = sources[slot - 1]
        original, pcm = commons.asset_paths(slot, source)
        acquisition = commons.TRIAL / f"acquisition-{slot:02}.json"
        qualification = commons.TRIAL / f"qualification-{slot:02}.json"
        raw, decoded = read(acquisition), read(qualification)
        if (source.environment != "traffic" or source.license not in {"CC-BY-SA-4.0", "CC0-1.0"}
                or raw["sha256"] != sha(original) or raw["publisher_sha1"] != source.publisher_sha1
                or decoded["acquisition_receipt_sha256"] != sha(acquisition)
                or decoded["freeze_sha256"] != results["background"]["freeze_sha256"]
                or decoded["wav_sha256"] != sha(pcm) or decoded["status"] != "verified_pcm"):
            raise ValueError("Traffic recording differs from the rights and exact preparation receipt")
        with wave.open(str(pcm), "rb") as stream:
            if stream.getparams()[:4] != (1, 2, RATE, decoded["samples"]):
                raise ValueError("Traffic sample grid differs")
            body = stream.readframes(stream.getnframes())
        if sha(body) != decoded["pcm_sha256"] or any(r["qualification_receipt_sha256"] != sha(qualification) for r in rows):
            raise ValueError("Traffic window preparation differs")
        traffic[slot] = []
        for row in rows:
            start = row["start_sample"] * 2
            part = body[start:start + COUNT * 2]
            if len(part) != COUNT * 2 or sha(part) != row["pcm_sha256"]:
                raise ValueError("Provisional traffic excerpt changed")
            traffic[slot].append((row, part))
    return traffic


def prepare(folder, archive, earlier):
    binding = freeze_context()
    sources_list, parent = sources(archive, earlier)
    results, hashes = background.result_set()
    traffic = traffic_sources(results, hashes)
    if len(traffic) != 3:
        raise ValueError("Expected three independently recorded traffic groups")
    for cuts in traffic.values():
        cuts.sort(key=lambda item: order(item[0]["pcm_sha256"]))
    (folder / "pcm").mkdir()
    rows = []

    def save(signal, values):
        raw = np.rint(signal).astype("<i2").tobytes()
        index = len(rows)
        (folder / "pcm" / f"{index:03}.pcm").write_bytes(raw)
        rows.append({"id": index, **values, "pcm_sha256": sha(raw)})

    for i, (source, raw) in enumerate(sources_list):
        wave = foreground(raw)
        save(wave, {**source, "condition": "clean", "tape": None, "snr_db": None})
        for tape, cuts in sorted(traffic.items()):
            cut, samples = cuts[i % len(cuts)]
            noise = foreground(samples)
            for snr in (-5, -10):
                noise_scaled = noise * np.linalg.norm(wave) / np.linalg.norm(noise) * (10 ** (-snr / 20))
                mix = wave + noise_scaled
                gain = min(1., 31_129 / np.max(np.abs(mix)))
                save(mix * gain, {**source, "condition": f"traffic_{abs(snr)}", "tape": tape, "snr_db": snr,
                                 "noise_pcm_sha256": cut["pcm_sha256"], "noise_start_sample": cut["start_sample"],
                                 "mix_gain": float(gain)})
    for tape, cuts in sorted(traffic.items()):
        for index in (0, 1):
            cut, samples = cuts[index]
            save(foreground(samples), {"family": "background", "group": f"tape:{tape}", "condition": "isolated",
                                       "tape": tape, "snr_db": None, "noise_pcm_sha256": cut["pcm_sha256"],
                                       "noise_start_sample": cut["start_sample"]})
    write_new(folder / "roster.json", rows)
    write_new(folder / "freeze.json", {**binding, "schema": "chipmunk.audio-strong-traffic-view-freeze/1",
               "source_commitments": {**parent, **hashes}, "roster_sha256": sha(folder / "roster.json"),
               "planned": len(rows)})
    print(f"Frozen {len(rows)} exact audio views", flush=True)


def load_pcm(folder, row):
    raw = (folder / "pcm" / f"{row['id']:03}.pcm").read_bytes()
    if len(raw) != COUNT * 2 or sha(raw) != row["pcm_sha256"]:
        raise ValueError("Frozen exact input PCM changed")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768


def reduced(wave):
    from scipy.signal import istft, stft

    _, _, bins = stft(wave, RATE, nperseg=512, noverlap=384)
    power = np.abs(bins) ** 2
    floor = np.quantile(power, .20, axis=1, keepdims=True)
    gain = np.sqrt(np.maximum(.16, 1. - 1.5 * floor / np.maximum(power, 1e-12)))
    _, signal = istft(bins * gain, RATE, nperseg=512, noverlap=384)
    if len(signal) < COUNT or not np.isfinite(signal).all():
        raise ValueError("The reduced view did not preserve the exact waveform duration")
    return np.clip(np.rint(signal[:COUNT] * 32768), -32768, 32767).astype(np.float32) / 32768


def frozen(folder):
    lock, rows = read(folder / "freeze.json"), read(folder / "roster.json")
    if any(lock[k] != v for k, v in freeze_context().items()) or lock["roster_sha256"] != sha(folder / "roster.json"):
        raise ValueError("Frozen plan or local checkpoint identity changed")
    if len(rows) != lock["planned"] or [r["id"] for r in rows] != list(range(len(rows))):
        raise ValueError("The complete ordered source roster is required")
    return lock, rows


def network_free():
    def guard(event, _args):
        if event == "socket.connect":
            raise RuntimeError("Evaluation models must stay offline")
    sys.addaudithook(guard)


def baseline(folder):
    from app.detectors.audio import calibration
    from app.detectors.audio.upgraded import SpeechActivity
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from app.engine.boundary import Deadline
    from scripts.audio_noise_speech_gate_v256 import CountedVAD
    from scripts.run_audio_specialists_v15 import load_voice

    lock, rows = frozen(folder)
    network_free()
    voice = load_voice({"weights": str(WEIGHTS), "assets": lock["assets"]})
    activity = SpeechActivity(WEIGHTS / "silero_vad.jit")
    activity.model = gate = CountedVAD(activity.model)
    detector = PellaVoiceDetector()
    count, observations = 0, []

    def counted(*_):
        nonlocal count
        count += 1

    hook = voice.register_forward_pre_hook(counted)
    try:
        for row in rows:
            wave = load_pcm(folder, row)
            ctx = SimpleNamespace(profile="full", lease=SimpleNamespace(get=lambda _: SimpleNamespace(obj=activity)),
                                  outputs=SimpleNamespace(audio=AudioTrack("analysed", wave, len(wave))))
            admitted = detector._windows(ctx)
            item = {"id": row["id"], "pcm_sha256": row["pcm_sha256"], "decision": "unassessed"}
            if admitted:
                if len(admitted) != 1 or admitted[0][0] != 0 or not np.array_equal(admitted[0][1], wave):
                    raise ValueError("Normal gate admitted a different sample window")
                raw, = voice.predict([wave])
                decision = calibration.evaluate(raw, wave, admitted, voice, Deadline(120))["decision"]
                item.update(raw=float(raw), decision=decision)
                if decision == "inconclusive":
                    signal = reduced(wave)
                    transformed, = voice.predict([signal])
                    item.update(reduced_raw=float(transformed), reduced_pcm_sha256=head.native.float_sha(signal))
            observations.append(item)
            if len(observations) % 15 == 0:
                print(f"Current voice {len(observations)}/{len(rows)} ({count} forwards)", flush=True)
    finally:
        hook.remove()
    write_new(folder / "baseline.json", {"freeze_sha256": sha(folder / "freeze.json"), "voice_forwards": count,
                                         "speech_gate_forwards": gate.calls, "rows": observations})


def candidate(folder):
    import torch

    from scripts.audio_wavlm_features_v13 import load

    lock, rows = frozen(folder)
    raw = read(folder / "baseline.json")
    if raw["freeze_sha256"] != sha(folder / "freeze.json"):
        raise ValueError("Current voice journal differs from the freeze")
    network_free()
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    model = load(head.native.WEIGHTS)
    fitted, _ = head.fit_context()
    cutoff = lock["candidate_points"]["ai_above"]
    observations = []
    for row, base in zip(rows, raw["rows"], strict=True):
        if (base["id"], base["pcm_sha256"]) != (row["id"], row["pcm_sha256"]):
            raise ValueError("Current voice result binds a different PCM")
        item = {"id": row["id"], "ordinary_candidate": None, "reduced_candidate": None,
                "ordinary_rescue": base["decision"], "reduced_rescue": base["decision"], "corroborated": base["decision"]}
        if base["decision"] == "inconclusive":
            original = load_pcm(folder, row)
            filtered = reduced(original)
            if head.native.float_sha(filtered) != base["reduced_pcm_sha256"]:
                raise ValueError("The paired reduced-view input changed")
            original_score = head.paired_score(base["raw"], head.native.extract(original, model), fitted)
            filtered_score = head.paired_score(base["reduced_raw"], head.native.extract(filtered, model), fitted)
            if not (math.isfinite(original_score) and math.isfinite(filtered_score)):
                raise ValueError("Non-finite fixed voice response")
            item.update(ordinary_candidate=original_score, reduced_candidate=filtered_score,
                        ordinary_rescue="ai" if original_score >= cutoff else base["decision"],
                        reduced_rescue="ai" if filtered_score >= cutoff else base["decision"],
                        corroborated="ai" if filtered_score >= cutoff and original_score >= cutoff - RULE["raw_margin"]
                        else base["decision"])
        observations.append(item)
    write_new(folder / "candidate.json", {"freeze_sha256": sha(folder / "freeze.json"),
        "baseline_sha256": sha(folder / "baseline.json"), "encoder_forwards": 2 * sum(
            row["ordinary_candidate"] is not None for row in observations), "rows": observations})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "baseline", "candidate"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--librispeech", type=Path)
    parser.add_argument("--earlier", type=Path)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.action == "freeze":
        if not args.librispeech or not args.earlier:
            parser.error("freeze requires the local test-clean archive and prior transfer receipts")
        prepare(args.folder, args.librispeech, args.earlier)
    else:
        {"baseline": baseline, "candidate": candidate}[args.action](args.folder)


if __name__ == "__main__":
    main()
