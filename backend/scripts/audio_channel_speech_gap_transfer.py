"""Check source-clocked stereo speech-activity support in unscored mono intervals."""

from __future__ import annotations

import argparse
import json
import math
import wave
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np

from app.detectors.audio.decode import decode_audio
from app.evidence.stereo_binding import SILENCE_FLOOR_DBFS, downmix
from scripts.audio_gtcrn_traffic import ROOT, read, sha, write_new
from scripts.prepare_natural_overlap_transfer import ANNOTATION_SHA256, track_words

RATE, SAMPLES, STEP = 16_000, 48_000, 160
ROOMS = ("ES2004a", "IS1009a", "ES2005a", "IS1006a")
SELECT = {"little_overlap": 6, "overlap": 5, "quiet": 4}
CHECKPOINT = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
PANEL = ROOT / "docs/evaluation-panels"
RULE = {"minimum_range_samples": 5120, "mono_support_guard_samples": 800, "minimum_channel_dbfs": -48,
        "minimum_known_speech_samples": 3840, "minimum_known_speech_fraction": .50,
        "minimum_native_additional_valid_windows": 6, "minimum_native_meetings": 2,
        "maximum_native_unsupported_flags": 2, "maximum_quiet_flags": 0,
        "minimum_control_synthetic_added_groups": 3, "minimum_control_synthetic_families": 2,
        "maximum_clean_flags": 0}


def salt(value):
    return sha(f"chipmunk/channel-speech-gap/1:{value}".encode())


def exact_wave(path, channels):
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()) != (channels, 2, RATE, SAMPLES):
            raise ValueError("Expected exact three-second native-rate PCM WAV")
        body = stream.readframes(SAMPLES + 1)
    if len(body) != SAMPLES * channels * 2:
        raise ValueError("Source PCM is truncated")
    return np.frombuffer(body, dtype="<i2").reshape(SAMPLES, channels)


def write_wave(path, pcm):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((pcm.shape[1], 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes(pcm.astype("<i2").tobytes())


def intervals(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2).tolist()


def native_rows(folder, spatial, old_wavs, old_receipt):
    spatial_public = read(PANEL / "audio-spatial-overlap-transfer-2026-11-11.json")["commitments"]
    old_public = read(PANEL / "audio-ami-native-stereo-complete-v291.json")
    if (sha(old_receipt) != old_public["acquisition_sha256"] or sha(spatial / "receipt.json") != spatial_public["source_receipt_sha256"]
            or sha(spatial / "roster.json") != spatial_public["private_roster_sha256"]):
        raise ValueError("Prior source commitments changed")
    acquisition = read(old_receipt)["files"]
    new = {row["name"]: row for row in read(spatial / "receipt.json")["files"]}
    exclusion = {room: [(60 + 30 * i) * RATE for i in range(12)] for room in ROOMS[:2]}
    for index, room in enumerate(ROOMS[2:], 1):
        exclusion[room] = [row["original_start_sample"] for row in read(spatial / "roster.json")
                           if row["condition"] == "native" and row["role"] == "confirmation" and row["meeting"] == index]
    annotations = spatial / "native/ami_public_manual_1.6.2.zip"
    if sha(annotations) != ANNOTATION_SHA256:
        raise ValueError("Manual AMI annotation release changed")
    rows, metadata = [], {}
    with ZipFile(annotations) as zf:
        for index, room in enumerate(ROOMS):
            sources = []
            for mic in ("Array1-01", "Array1-05"):
                name = f"{room}.{mic}.wav"
                path = (old_wavs if name in acquisition else spatial / "native") / name
                expected = (acquisition | new)[name]["sha256"]
                if sha(path) != expected:
                    raise ValueError("AMI physical microphone bytes changed")
                sources.append(path)
                metadata[name] = expected
            values = [track_words(zf.read(f"words/{room}.{person}.words.xml")) for person in "ABCD"]
            with wave.open(str(sources[0]), "rb") as left, wave.open(str(sources[1]), "rb") as right:
                if any((w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, RATE) for w in (left, right)):
                    raise ValueError("Physical microphone channel sample clock differs")
                end = min(left.getnframes(), right.getnframes()) // STEP
                activity = np.zeros((4, end), dtype=bool)
                for participant, bounds in enumerate(values):
                    for a, b in bounds:
                        activity[participant, a:b] = True
                choices = defaultdict(list)
                for start in range(300, end - 299, 300):
                    source_start = start * STEP
                    if any(abs(source_start - old) < SAMPLES for old in exclusion[room]):
                        continue
                    active = activity[:, start:start + 300].sum(axis=0)
                    quiet, overlap, speaking = int((active == 0).sum()), int((active >= 2).sum()), int((active > 0).sum())
                    if quiet >= 296:
                        choices["quiet"].append(start)
                    elif overlap <= 10 and speaking >= 90:
                        choices["little_overlap"].append(start)
                    elif overlap >= 75 and speaking >= 100:
                        choices["overlap"].append(start)
                for condition, count in SELECT.items():
                    selected = sorted(choices[condition], key=lambda start: salt((room, start)))[:count]
                    if len(selected) != count:
                        raise ValueError(f"Insufficient disjoint annotation-only {condition} windows in {room}")
                    for start in selected:
                        source_start = start * STEP
                        for stream in (left, right):
                            stream.setpos(source_start)
                        pcm = np.column_stack([np.frombuffer(stream.readframes(SAMPLES), "<i2") for stream in (left, right)])
                        asset = folder / "stereo" / f"{len(rows):03}.wav"
                        write_wave(asset, pcm)
                        annotated = activity[:, start:start + 300].any(axis=0)
                        rows.append({"panel": "native", "group": f"meeting:{index + 1}", "family": "genuine", "condition": condition,
                                     "wav_sha256": sha(asset), "stereo_sha256": sha(pcm.tobytes()), "source_start_sample": source_start,
                                     "known_word_samples": [[a * STEP, b * STEP] for a, b in intervals(annotated)]})
    return rows, metadata | {"annotations": ANNOTATION_SHA256}


def controlled_rows(folder, case_journal, stereo_media, offset):
    receipt = read(PANEL / "audio-native-stereo-pair-holdout-complete-v277.json")
    if sha(case_journal) != receipt["private_cases_sha256"]:
        raise ValueError("Controlled stereo source journal changed")
    rows = []
    for item in map(json.loads, case_journal.read_text().splitlines()):
        old = stereo_media / (item["case"] + ".wav")
        if item["status"] != "prepared" or sha(old) != item["wav_sha256"]:
            raise ValueError("Controlled source differs from its qualified waveform")
        pcm = exact_wave(old, 2)
        if sha(pcm.tobytes()) != item["stereo_pcm_sha256"] or [sha(pcm[:, i].tobytes()) for i in (0, 1)] != item["channel_pcm_sha256"]:
            raise ValueError("Controlled channel samples differ")
        (folder / "stereo" / f"{offset + len(rows):03}.wav").write_bytes(old.read_bytes())
        rows.append({"panel": "controlled", "group": item["source_group"], "foreground": item["source_slot"], "family": item["lane"],
                     "condition": item["condition"], "wav_sha256": item["wav_sha256"], "stereo_sha256": item["stereo_pcm_sha256"],
                     "expected_mono_sha256": item["mixed_pcm_sha256"], "source_start_sample": 0})
    if len(rows) != receipt["planned_cases"]:
        raise ValueError("Controlled panel denominator differs")
    return rows


def binding():
    paths = (Path(__file__).resolve(), ROOT / "backend/app/detectors/audio/upgraded.py", ROOT / "backend/app/detectors/audio/decode.py",
             ROOT / "backend/app/evidence/stereo_binding.py", ROOT / "backend/app/detectors/video/adapters/audio.py")
    return {"code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in paths}, "checkpoint_sha256": CHECKPOINT,
            "rule": RULE, "sample_rate": RATE, "samples": SAMPLES, "native_selection": SELECT}


def prepare(folder, spatial, old_wavs, old_receipt, journal, media):
    (folder / "stereo").mkdir()
    (folder / "mono").mkdir()
    native, commitments = native_rows(folder, spatial, old_wavs, old_receipt)
    controlled = controlled_rows(folder, journal, media, len(native))
    rows = []
    for index, item in enumerate([*native, *controlled]):
        path = folder / "stereo" / f"{index:03}.wav"
        decoded = decode_audio(path, source_sha256=item["wav_sha256"], max_seconds=3.)
        pcm = np.rint(decoded.pcm * 32768).astype("<i2").tobytes()
        if (len(pcm) != 2 * SAMPLES or decoded.mapping["source_channels"] != 2
                or decoded.mapping["timeline"] != [{"start_sample": 0, "end_sample": SAMPLES, "source_pts": 0}]
                or ("expected_mono_sha256" in item and item["expected_mono_sha256"] != sha(pcm))):
            raise ValueError("Production mono sample clock or controlled downmix changed")
        (folder / "mono" / f"{index:03}.pcm").write_bytes(pcm)
        rows.append({"id": index, **item, "mono_sha256": sha(pcm)})
    write_new(folder / "roster.json", rows)
    write_new(folder / "freeze.json", {**binding(), "schema": "chipmunk.audio-channel-speech-gap-freeze/1", "views": len(rows),
                                       "sources": commitments, "controlled_journal_sha256": sha(journal), "roster_sha256": sha(folder / "roster.json")})
    print(f"Frozen {len(native)} fresh native intervals and {len(controlled)} controlled views", flush=True)


def frozen(folder):
    lock, rows = read(folder / "freeze.json"), read(folder / "roster.json")
    if (any(lock[key] != value for key, value in binding().items()) or sha(folder / "roster.json") != lock["roster_sha256"]
            or [item["id"] for item in rows] != list(range(lock["views"]))):
        raise ValueError("Frozen code or source roster differs")
    return lock, rows


def pcm(folder, row):
    index = row["id"]
    stereo, mono = folder / "stereo" / f"{index:03}.wav", (folder / "mono" / f"{index:03}.pcm").read_bytes()
    channels = exact_wave(stereo, 2)
    if sha(stereo) != row["wav_sha256"] or sha(channels.tobytes()) != row["stereo_sha256"] or len(mono) != 2 * SAMPLES or sha(mono) != row["mono_sha256"]:
        raise ValueError("Exact stereo or downmix input changed")
    return channels, np.frombuffer(mono, "<i2")


def rms_dbfs(samples):
    power = float(np.mean((samples.astype(np.float64) / 32768.) ** 2))
    return round(10 * math.log10(power), 3) if power else SILENCE_FLOOR_DBFS


def reports(stereo, mono, spans, admitted):
    if admitted:
        return [], []
    no_mono = np.ones(SAMPLES, dtype=bool)
    guard = RULE["mono_support_guard_samples"]
    for a, b in spans[0]:
        no_mono[max(0, a - guard):min(SAMPLES, b + guard)] = False
    marks = []
    for channel, group in enumerate(spans[1:]):
        for a, b in group:
            support = no_mono.copy()
            support[:a], support[b:] = False, False
            for start, end in intervals(support):
                if end - start < RULE["minimum_range_samples"]:
                    continue
                sample_level = rms_dbfs(stereo[start:end, channel])
                if sample_level >= RULE["minimum_channel_dbfs"]:
                    marks.append({"channel": channel, "start_sample": start, "end_sample": end, "rms_dbfs": sample_level,
                                  "clip_pcm_sha256": sha(stereo[start:end, channel].tobytes())})
    attenuation = []
    for a in range(0, SAMPLES, RATE):
        b = a + RATE
        levels = [rms_dbfs(x[a:b]) for x in (stereo[:, 0], stereo[:, 1], mono)]
        if downmix(*levels)["strong_attenuation"]:
            attenuation.append({"start_sample": a, "end_sample": b})
    return marks, attenuation


def run(folder, vad):
    import torch

    from app.detectors.audio.upgraded import SpeechActivity
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from scripts.audio_noise_speech_gate_v256 import CountedVAD
    from scripts.audio_strong_traffic_view import network_free

    _, rows = frozen(folder)
    if sha(vad) != CHECKPOINT:
        raise ValueError("The local speech activity weights changed")
    network_free()
    torch.set_num_threads(1)
    activity = SpeechActivity(vad)
    activity.model = counted = CountedVAD(activity.model)
    detector = PellaVoiceDetector()
    observations = []
    for row in rows:
        stereo, mono = pcm(folder, row)
        waves = [x.astype(np.float32) / 32768 for x in (mono, stereo[:, 0], stereo[:, 1])]
        all_spans = [activity.spans(wave) for wave in waves]
        cached = SimpleNamespace(obj=SimpleNamespace(spans=lambda _, value=all_spans[0]: value))
        ctx = SimpleNamespace(profile="full", lease=SimpleNamespace(get=lambda _, value=cached: value),
                              outputs=SimpleNamespace(audio=AudioTrack("analysed", waves[0], SAMPLES)))
        selected = detector._windows(ctx)
        if selected and (len(selected) != 1 or selected[0][0] != 0 or len(selected[0][1]) != SAMPLES):
            raise ValueError("Expected an exact full mono voice window")
        spans = [[[round(a * RATE), round(b * RATE)] for a, b in group] for group in all_spans]
        marks, attenuation = reports(stereo, mono, spans, bool(selected))
        observations.append({"id": row["id"], "source_stereo_sha256": row["stereo_sha256"], "source_mono_sha256": row["mono_sha256"],
                             "spans": spans, "mono_assessed": bool(selected), "marks": marks, "existing_attenuation": attenuation})
        if len(observations) % 40 == 0:
            print(f"Physical speech coverage {len(observations)}/{len(rows)}", flush=True)
    write_new(folder / "observations.json", {"freeze_sha256": sha(folder / "freeze.json"), "vad_forwards": counted.calls,
                                              "classifier_forwards": 0, "rows": observations})


def overlaps_speech(spans, marks):
    for mark in marks:
        a, b = mark["start_sample"], mark["end_sample"]
        overlap = sum(max(0, min(b, stop) - max(a, start)) for start, stop in spans)
        if overlap >= RULE["minimum_known_speech_samples"] and overlap / (b - a) >= RULE["minimum_known_speech_fraction"]:
            return True
    return False


def readout(folder):
    lock, rows = frozen(folder)
    run = read(folder / "observations.json")
    if run["freeze_sha256"] != sha(folder / "freeze.json"):
        raise ValueError("Observation journal binds different inputs")
    clean_reference, counts = {}, defaultdict(Counter)
    correct_native, correct_control = set(), set()
    quality = Counter()
    evidence = []
    for row, observed in zip(rows, run["rows"], strict=True):
        if (observed["id"], observed["source_stereo_sha256"], observed["source_mono_sha256"]) != (row["id"], row["stereo_sha256"], row["mono_sha256"]):
            raise ValueError("Different source bound in observed interval")
        stereo, mono = pcm(folder, row)
        marks, attenuation = reports(stereo, mono, observed["spans"], observed["mono_assessed"])
        if (marks, attenuation) != (observed["marks"], observed["existing_attenuation"]):
            raise ValueError("Source interval measurement replay differs")
        condition, panel, group = row["condition"], row["panel"], row["group"]
        if panel == "native":
            reference = row["known_word_samples"]
        else:
            key = row["foreground"]
            if condition == "matched_clean":
                if not np.array_equal(stereo[:, 0], stereo[:, 1]):
                    raise ValueError("The clean reference must contain identical dry foreground channels")
                energy = np.sqrt(np.mean((stereo[:, 0].reshape(-1, STEP).astype(np.float64) / 32768) ** 2, axis=1))
                clean_reference[key] = [[a * STEP, b * STEP] for a, b in intervals(energy >= max(.002, .15 * max(energy)))]
            reference = clean_reference[key]
        candidate = overlaps_speech(reference, marks)
        baseline = overlaps_speech(reference, attenuation)
        bucket = counts[(panel, condition, row["family"])]
        bucket.update(planned=1, mono_assessed=int(observed["mono_assessed"]), candidate_flags=int(bool(marks)),
                      baseline_flags=int(bool(attenuation)), candidate_supported=int(candidate), baseline_supported=int(baseline),
                      additional_supported=int(candidate and not baseline), unsupported_flags=int(bool(marks) and not candidate),
                      opportunities=int(not observed["mono_assessed"] and sum(b-a for a, b in reference) >= RULE["minimum_known_speech_samples"]))
        if panel == "native":
            quality.update(native_additions=int(candidate and not baseline), native_unsupported=int(bool(marks) and not candidate),
                           native_quiet=int(condition == "quiet" and bool(marks)))
            if candidate and not baseline:
                correct_native.add(group)
        else:
            quality.update(control_clean=int(condition == "matched_clean" and bool(marks)))
            if candidate and not baseline and row["family"] != "genuine":
                correct_control.add((group, row["family"]))
        for mark in marks:
            a, b = mark["start_sample"], mark["end_sample"]
            evidence.append({**mark, "id": row["id"], "source_wav_sha256": row["wav_sha256"], "time_base": "1/16000",
                             "source_start_pts": row["source_start_sample"] + a, "source_end_pts": row["source_start_sample"] + b,
                             "mono_sha256": row["mono_sha256"]})
    checks = {"native_additions": quality["native_additions"] >= RULE["minimum_native_additional_valid_windows"],
              "native_meetings": len(correct_native) >= RULE["minimum_native_meetings"],
              "native_unsupported": quality["native_unsupported"] <= RULE["maximum_native_unsupported_flags"],
              "native_quiet": quality["native_quiet"] <= RULE["maximum_quiet_flags"],
              "control_groups": len({key for key, _ in correct_control}) >= RULE["minimum_control_synthetic_added_groups"],
              "control_families": len({family for _, family in correct_control}) >= RULE["minimum_control_synthetic_families"],
              "control_clean": quality["control_clean"] <= RULE["maximum_clean_flags"]}
    write_new(folder / "private-evidence.json", evidence)
    aggregate = {"schema": "chipmunk.audio-channel-speech-gap-evaluation/1", "freeze": lock,
                 "freeze_sha256": sha(folder / "freeze.json"), "observations_sha256": sha(folder / "observations.json"),
                 "private_evidence_sha256": sha(folder / "private-evidence.json"), "vad_forwards": run["vad_forwards"], "classifier_forwards": 0,
                 "checks": checks, "advance": all(checks.values()), "native_gained_meetings": len(correct_native),
                 "controlled_synthetic_gained_groups": len({key for key, _ in correct_control}),
                 "controlled_synthetic_gained_families": len({family for _, family in correct_control}),
                 "selection_counts": dict(quality), "source_intervals": len(evidence),
                 "counts": [{"panel": panel, "condition": condition, "family": family, **bucket}
                            for (panel, condition, family), bucket in sorted(counts.items())]}
    write_new(folder / "aggregate.json", aggregate)
    print(json.dumps({"checks": checks, "counts": dict(quality), "source_intervals": len(evidence), "advance": aggregate["advance"]}, sort_keys=True), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("freeze", "run", "readout"))
    cli.add_argument("folder", type=Path)
    for argument in ("spatial", "native-wavs", "native-receipt", "controlled-journal", "controlled-media", "vad"):
        cli.add_argument("--" + argument, type=Path)
    args = cli.parse_args()
    if args.stage == "freeze":
        sources = (args.spatial, args.native_wavs, args.native_receipt, args.controlled_journal, args.controlled_media)
        if not all(sources):
            cli.error("Freeze requires the retained native and controlled source files")
        prepare(args.folder, *sources)
    elif args.stage == "run":
        if args.vad is None:
            cli.error("Run requires the pinned local speech activity checkpoint")
        run(args.folder, args.vad)
    else:
        readout(args.folder)


if __name__ == "__main__":
    main()
