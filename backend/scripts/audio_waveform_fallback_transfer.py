"""Freeze and replay a source-separated fallback for repeated-audio exhibits."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import wave
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path, PurePosixPath

import numpy as np

from app.detectors.audio.decode import decode_audio, source_ranges, wav_bytes
from app.evidence.audio_exact_repetition import find_repeated_spans
from app.evidence.audio_similarity import find_similar_spans as two_band
from scripts.audio_repetition_similarity import find_similar_spans as waveform

ROOT = Path(__file__).resolve().parents[2]
RATE, SAMPLES = 16_000, 48_000
WIDTH, FADE, TARGET, MIN_OVERLAP = 10_400, 320, 28_000, 2_400
DONORS = tuple(range(1600, 17601, 3200))
CONDITIONS = {"clean": (None, None), "kitchen_plus5": ("kitchen", 5), "kitchen_minus5": ("kitchen", -5),
              "traffic_plus5": ("traffic", 5), "traffic_minus5": ("traffic", -5)}
DELIVERIES = {
    "pcm": (".wav", "pcm_s16le", []), "aac_64k": (".m4a", "aac", ["-c:a", "aac", "-b:a", "64k"]),
    "mp3_64k": (".mp3", "mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]),
    "opus_48k": (".opus", "opus", ["-c:a", "libopus", "-b:a", "48k"]),
}
TAPES = {"kitchen": "DKITCHEN_16k.zip", "traffic": "STRAFFIC_16k.zip"}
LANGUAGE_COUNTS = {"et": 2, "lt": 2, "sl": 2, "fi": 3, "nl": 3}
PARENT = {
    "ami_plan": "6a35b68edec375a83db430ee8dca94abba2270e0159253aa8bd0f86ff48b457f",
    "ami_freeze": "1f6121cacb4edc256b643dd767f89c03ea52f4ca5257ccfca5300949002af679",
    "vox_manifest": "207f2084f1dd036f197c404c5b727f7740266b6eaa2037cb6488da33c954399a",
    "noise_admission": "70473811ceefa9063977400b070b49870229256180ed9f586055bb0ea73d6705",
    "previous_result": "6750ccff4eb24548259ebbe453258031333f14f04be662da95417443545c029b",
}
PANEL = ROOT / "docs" / "evaluation-panels"
CODE = (
    "backend/scripts/audio_waveform_fallback_transfer.py",
    "backend/scripts/audio_repetition_similarity.py",
    "backend/app/evidence/audio_exact_repetition.py",
    "backend/app/evidence/audio_similarity.py",
    "backend/app/evidence/audio_similarity_binding.py",
    "backend/app/detectors/audio/decode.py",
)
POLICY = {
    "minimum_added_correct_views": 24,
    "minimum_groups_with_gain": 6,
    "minimum_ami_meetings_with_gain": 2,
    "minimum_voxpopuli_speakers_with_gain": 2,
    "maximum_added_control_flags": 3,
    "maximum_groups_with_added_control_flags": 2,
    "maximum_added_wrong_only_copy_views": 3,
}
METHOD = {
    "selection": "Four annotated two-voice and four three-voice windows from each of three distinct AMI meetings; "
                 "one frozen stem per VoxPopuli speaker and distinct parliamentary session date in quota order.",
    "edit": "Copy the largest-energy 650 ms from six early ranges to 1.75 s, with 20 ms raised-cosine boundary blends.",
    "controls": "No authored copy. The native recordings may contain ordinary recurrent sounds or speech.",
    "noise": "New channel-one DEMAND excerpts at 100 + 5 * source ordinal seconds. Fixed original-based gain "
             "and shared anti-clipping attenuation for each matched pair; whole-clip nominal SNR.",
    "route": "Keep all exact and two-band pairs. Run frozen waveform cue only when both lists are empty.",
    "support": "Both half-open ranges must overlap their blend-excluded authored span by at least 2400 samples; "
               "exact lag for exact pairs, at most 32 samples of lag error for approximate pairs.",
    "policy": POLICY,
    "scope": "Short continuous mono acoustic-similarity evidence only, with zero AI confidence weight.",
}


def sha(value: bytes | Path) -> str:
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    with value.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def read(path: Path):
    return json.loads(path.read_text())


def write_new(path: Path, data) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")


def wav_samples(stream, start: int, count: int | None) -> bytes:
    stream.seek(0)
    with wave.open(stream, "rb") as wav:
        if wav.getparams()[:3] != (1, 2, RATE):
            raise ValueError("Expected native mono signed 16-bit WAV")
        wav.setpos(start)
        size = wav.getnframes() - start if count is None else count
        pcm = wav.readframes(size)
    if len(pcm) != 2 * size:
        raise ValueError("Audio source window is truncated")
    return pcm


def prepare(ami: Path, vox: Path, noise: Path):
    parent_paths = {
        "ami_plan": ami / "natural-source-plan.json", "ami_freeze": ami / "stream-presence-freeze.json",
        "vox_manifest": vox / "manifest.json", "noise_admission": PANEL / "audio-demand-kitchen-traffic-assets-v212.json",
        "previous_result": PANEL / "audio-repeat-robustness-transfer-2026-11-11.json",
    }
    if {key: sha(path) for key, path in parent_paths.items()} != PARENT:
        raise ValueError("Expected the exact prior source plans and repetition result")
    plan, freeze = read(parent_paths["ami_plan"]), read(parent_paths["ami_freeze"])
    foreground, tapes, rows, native = [], {}, [], []
    for i, room in enumerate(plan["rooms"]):
        path = ami / "natural-raw" / f"{room['meeting']}.wav"
        binding = {"group": i, "sha256": sha(path)}
        if binding != freeze["native_sources"][i]:
            raise ValueError("AMI source differs from the frozen original")
        native.append(binding)
        with path.open("rb") as stream:
            for voices in (2, 3):
                selected = sorted(c["begin_centiseconds"] for c in room["cases"] if c["expected_voices"] == voices)[:4]
                if len(selected) != 4 or len(set(selected)) != 4:
                    raise ValueError("AMI annotation plan lacks four distinct reference windows")
                for begin in selected:
                    start = begin * 160
                    raw = wav_samples(stream, start, SAMPLES)
                    rows.append({"family": "ami", "group": f"m{i:02}", "start_sample": start,
                                 "annotated_voices": voices, "pcm_sha256": sha(raw)})
                    foreground.append(raw)
    selected_speakers, sessions, counts = set(), set(), Counter()
    for group in read(parent_paths["vox_manifest"])["groups"]:
        language = group["language"]
        if counts[language] >= LANGUAGE_COUNTS[language]:
            continue
        per_triplet = 2 if language in {"et", "lt"} else 1
        taken = 0
        for slot, source in enumerate(group["sources"]):
            speaker, session = source["id"], source["utterance"].split("-", 1)[0]
            if speaker in selected_speakers or session in sessions:
                continue
            raw = (vox / "stems" / f"{group['group']:03}-{slot}.pcm").read_bytes()
            if len(raw) != 2 * SAMPLES or sha(raw) != source["pcm_sha256"]:
                raise ValueError("VoxPopuli stem differs from the frozen source")
            selected_speakers.add(speaker)
            sessions.add(session)
            counts[language] += 1
            rows.append({"family": "voxpopuli", "group": f"v{len(selected_speakers):02}", "language": language,
                         "source_triplet": group["group"], "source_slot": slot, "source_session": session,
                         "speaker": speaker, "encoded_sha256": source["encoded_sha256"], "pcm_sha256": sha(raw)})
            foreground.append(raw)
            taken += 1
            if taken == per_triplet or counts[language] == LANGUAGE_COUNTS[language]:
                break
    if counts != LANGUAGE_COUNTS or len(foreground) != 36 or len(native) != 3:
        raise ValueError("Expected three AMI meetings and 12 distinct VoxPopuli speakers in five languages")
    if len({row["pcm_sha256"] for row in rows}) != len(rows):
        raise ValueError("A frozen source window was selected twice")
    pins = {row["file"]: row for row in read(parent_paths["noise_admission"])["archives"]}
    for kind, name in TAPES.items():
        path, pin = noise / name, pins[name]
        if path.stat().st_size != pin["bytes"] or sha(path) != pin["sha256"]:
            raise ValueError("DEMAND publisher archive differs from the frozen input")
        with zipfile.ZipFile(path) as archive:
            channel, = (item for item in archive.namelist() if PurePosixPath(item).name == "ch01.wav")
            raw = wav_samples(io.BytesIO(archive.read(channel)), 0, None)
        if len(raw) != 2 * pin["channel_samples"] or sha(raw) != pin["channel_pcm_sha256"]:
            raise ValueError("DEMAND channel differs from the frozen input")
        tapes[kind] = np.frombuffer(raw, dtype="<i2")
    lock = {"schema": "chipmunk.audio-waveform-fallback-freeze/1", "method": METHOD,
            "parent_sha256": PARENT, "code_sha256": {path: sha(ROOT / path) for path in CODE},
            "codec_sha256": {name: sha(Path(shutil.which(name) or "").resolve()) for name in ("ffmpeg", "ffprobe")},
            "source_roster": rows, "ami_originals": native, "noise_roster": pins,
            "conditions": json.loads(encoded(CONDITIONS)), "deliveries": json.loads(encoded(DELIVERIES)),
            "planned_views": len(rows) * len(CONDITIONS) * len(DELIVERIES) * 2}
    return lock, foreground, tapes


def copy_edit(original: np.ndarray) -> tuple[np.ndarray, int]:
    donor = max(DONORS, key=lambda start: float(np.sum(original[start:start + WIDTH] ** 2)))
    blend = .5 - .5 * np.cos(np.linspace(0, np.pi, FADE))
    envelope = np.r_[blend, np.ones(WIDTH - 2 * FADE), blend[::-1]]
    copy = original.copy()
    copy[TARGET:TARGET + WIDTH] += envelope * (original[donor:donor + WIDTH] - copy[TARGET:TARGET + WIDTH])
    return copy, donor


def treatments(raw: bytes, index: int, tapes):
    original = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    copy, donor = copy_edit(original)
    for condition, (kind, snr) in CONDITIONS.items():
        if kind is None:
            background, offset = np.zeros_like(original), None
        else:
            offset = (100 + 5 * index) * RATE
            noise = tapes[kind][offset:offset + SAMPLES].astype(np.float64)
            if len(noise) != SAMPLES or not np.any(noise):
                raise ValueError("Missing frozen noise excerpt")
            background = noise * (np.linalg.norm(original) / np.linalg.norm(noise)) * 10 ** (-snr / 20)
        control, altered = original + background, copy + background
        gain = min(1., 31_129 / max(np.abs(control).max(), np.abs(altered).max()))
        for label, pcm in (("control", control), ("copy", altered)):
            actual = None if kind is None else 20 * np.log10(
                np.linalg.norm(original if label == "control" else copy) / np.linalg.norm(background)
            )
            yield condition, label, donor if label == "copy" else None, offset, actual, np.rint(pcm * gain).astype("<i2")


def witness(span, donor, tolerance: int) -> bool:
    if donor is None:
        return False
    return (abs(span.second[0] - span.first[0] - (TARGET - donor)) <= tolerance
            and min(span.first[1], donor + WIDTH - FADE) - max(span.first[0], donor + FADE) >= MIN_OVERLAP
            and min(span.second[1], TARGET + WIDTH - FADE) - max(span.second[0], TARGET + FADE) >= MIN_OVERLAP)


def measurements(pcm: bytes, donor) -> dict:
    exact, band = find_repeated_spans(pcm), two_band(pcm)
    added = None if exact.spans or band.spans else waveform(pcm)
    return {
        name: {"result": asdict(result), "correct": any(witness(span, donor, tolerance) for span in result.spans)}
        if result is not None else None
        for name, result, tolerance in (("exact", exact, 0), ("two_band", band, 32), ("waveform", added, 32))
    }


def deliver(pcm: np.ndarray, delivery: str, folder: Path):
    suffix, codec, options = DELIVERIES[delivery]
    with tempfile.TemporaryDirectory(dir=folder) as temp:
        source, target = Path(temp) / "input.wav", Path(temp) / f"encoded{suffix}"
        source.write_bytes(wav_bytes(pcm.astype(np.float32) / 32768))
        if delivery == "pcm":
            target = source
        else:
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(source), "-map_metadata", "-1",
                 "-fflags", "+bitexact", *options, "-threads", "1", str(target)], capture_output=True, check=True, timeout=30,
            )
        source_sha = sha(target)
        decoded = decode_audio(target, source_sha256=source_sha)
        count = len(decoded.pcm)
        mapping = decoded.mapping
        origin = int(mapping["source_start_pts"] or 0) * Fraction(mapping["source_time_base"]) * RATE
        clock = source_ranges(mapping, 0, count)
        if (len(clock) != 1 or clock[0]["source_start_pts"] != origin or mapping["source_codec"] != codec
                or (delivery == "aac_64k" and not 0 <= count - SAMPLES < 1024)
                or (delivery != "aac_64k" and count != SAMPLES)):
            raise ValueError("The app decode does not have the expected continuous codec sample clock")
        payload = wav_samples(io.BytesIO(decoded.wav), 0, count)
        if delivery == "pcm" and payload != pcm.tobytes():
            raise ValueError("Lossless delivery changed the signed PCM input")
    return payload, {"encoded_sha256": source_sha, "pcm_sha256": sha(payload), "clock": clock[0],
                     "samples": count, "padding_samples": count - SAMPLES}


def source_views(index, row, raw, tapes, folder):
    for condition, label, donor, offset, actual, signal in treatments(raw, index, tapes):
        for delivery in DELIVERIES:
            pcm, clock = deliver(signal, delivery, folder)
            yield {"source": index, "family": row["family"], "group": row["group"], "condition": condition,
                   "delivery": delivery, "label": label, "donor": donor, "noise_offset": offset,
                   "actual_snr_db": actual, **clock, **measurements(pcm, donor)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--ami", type=Path, required=True)
    parser.add_argument("--vox", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    args = parser.parse_args()
    lock, foreground, tapes = prepare(args.ami, args.vox, args.noise)
    if args.action == "freeze":
        write_new(args.folder / "freeze.json", lock)
        print(f"Frozen {lock['planned_views']} views from {len(foreground)} source windows", flush=True)
        return
    if lock != read(args.folder / "freeze.json"):
        raise ValueError("Source, configuration, or implementation changed after freeze")
    def views(item):
        index, (row, raw) = item
        return list(source_views(index, row, raw, tapes, args.folder))
    with (args.folder / "observations.jsonl").open("x") as journal, ThreadPoolExecutor(max_workers=3) as pool:
        for i, results in enumerate(pool.map(views, enumerate(zip(lock["source_roster"], foreground, strict=True))), 1):
            journal.writelines(encoded(row) + "\n" for row in results)
            journal.flush()
            print(f"Decoded and measured {i}/{len(foreground)} source windows", flush=True)


if __name__ == "__main__":
    main()
