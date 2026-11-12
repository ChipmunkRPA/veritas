"""Test the fixed stereo rescue on matched sequential and overlapping room speech."""

from __future__ import annotations

import argparse
import resource
import wave
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np

from app.detectors.audio.decode import decode_audio
from scripts import audio_native_stereo_gate_v279 as stereo
from scripts import audio_spectra_background_transfer as serving
from scripts.audio_harmonic_overlap_transfer import EXPOSED, MIRROR
from scripts.audio_short_edit_cases_v172 import exact_source, sources
from scripts.prepare_natural_overlap_transfer import track_words

RATE, COUNT, VIEWS = stereo.RATE, stereo.SAMPLES, stereo.VIEWS
sha, read, write = serving.sha, serving.read, serving.write_new
MODES = ("sequential", "simultaneous")
MICS = ("Array1-01", "Array1-05")
FAMILIES = ("clone", "expressive", "stock")
ROOMS_EXCLUDED = EXPOSED | {"ES2007c", "IS1004c"}
RULE = {"minimum_additional_synthetic_groups": 3, "minimum_added_synthetic_families": 2,
        "minimum_conditions_with_gain": 2, "maximum_additional_genuine_false_flags": 1,
        "maximum_additional_clean_or_native_false_flags": 0, "minimum_assessed_genuine_per_room_condition": 5}


def order(kind, label):
    return sha(f"chipmunk/paired-room-stereo/1:{kind}:{label}".encode() if kind != "room"
               else f"chipmunk/paired-room-stereo/1:{label}".encode())


def candidates(archive, room):
    tracks = [track_words(archive.read(f"words/{room}.{who}.words.xml")) for who in "ABCD"]
    extent = min(90_000, max(stop for track in tracks for _, stop in track))
    active = np.zeros((4, extent), dtype=bool)
    for speaker, words in enumerate(tracks):
        for start, stop in words:
            active[speaker, start:stop] = True
    options = {mode: defaultdict(list) for mode in MODES}
    for start in range(400, extent - 300, 400):
        view = active[:, start:start + 300]
        counts = view.sum(axis=1)
        speakers = tuple(sorted(int(i) for i in np.argsort(-counts)[:2]))
        two = counts[list(speakers)]
        if view.any(axis=0).sum() < 245 or min(two) < 35 or counts.sum() - two.sum() > 12:
            continue
        overlap = int((view.sum(axis=0) >= 2).sum())
        if overlap <= 10 or overlap >= 90:
            mode = "sequential" if overlap <= 10 else "simultaneous"
            options[mode][speakers].append((start, overlap))
    pairs = []
    for pair in sorted(options["simultaneous"]):
        low, high = (sorted(options[mode].get(pair, ()), key=lambda row: order(mode, (room, pair, row[0]))) for mode in MODES)
        pairs.extend((pair, low_view, high_view) for low_view, high_view in zip(low, high))
    return sorted(pairs, key=lambda row: order("pair", (room, row)))[:4]


def rooms(archive):
    xml = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
    groups = {node.get("observation"): {p.get("global_name") for p in node if p.get("global_name")}
              for node in xml.iter() if node.get("observation")}
    exposed = set().union(*(groups[name] for name in ROOMS_EXCLUDED))
    selected = {}
    for site in ("ES", "IS"):
        for room in sorted(groups, key=lambda name: order("room", name)):
            if not room.startswith(site) or len(groups[room]) != 4 or exposed & groups[room]:
                continue
            try:
                pairs = candidates(archive, room)
            except KeyError:
                continue
            if len(pairs) == 4:
                selected[room] = pairs
                exposed.update(groups[room])
                break
    if len(selected) != 2:
        raise ValueError("The annotation source could not supply two disjoint room pairs")
    return selected


def qualify_container(path):
    with wave.open(str(path), "rb") as audio:
        channels, width, rate = audio.getparams()[:3]
        if channels not in (1, 2) or (width, rate) != (2, RATE):
            raise ValueError("The native channel clock differs")
        if channels == 2:
            # Some publisher microphone WAVs duplicate their one physical channel.
            frames = 0
            while raw := audio.readframes(480_000):
                samples = np.frombuffer(raw, "<i2").reshape(-1, 2)
                if not np.array_equal(samples[:, 0], samples[:, 1]):
                    raise ValueError("The publisher file does not identify one exact physical microphone channel")
                frames += len(samples)
            if frames != audio.getnframes():
                raise ValueError("The publisher microphone file is truncated")
        return channels, audio.getnframes()


def native_cut(paths, start):
    channels = []
    for path in paths:
        with wave.open(str(path), "rb") as audio:
            n = audio.getnchannels()
            audio.setpos(start)
            raw = audio.readframes(COUNT)
        if len(raw) != 2 * COUNT * n:
            raise ValueError("A native source window is truncated")
        channels.append(np.frombuffer(raw, "<i2").reshape(-1, n)[:, 0])
    return np.column_stack(channels)


def foregrounds():
    path = sources.MEDIA / "holdout/private-manifest.json"
    provenance = read(serving.PANEL / "audio-short-edit-materialization-complete-v169c.json")
    if sha(path) != provenance["roles"]["holdout"]["private_manifest_sha256"]:
        raise ValueError("Held voices differ from the verified source and license manifest")
    rows = read(path)["foregrounds"]
    selected, seen = [], set()
    for family, limit in (("genuine", 8), *((family, 3) for family in FAMILIES)):
        available = sorted((r for r in rows if r["lane"] == family and r["status"] == "qualified"),
                           key=lambda r: order("foreground", r["slot"]))
        found = 0
        for row in available:
            if row["voice"] not in seen:
                selected.append((exact_source(row, "holdout"), row))
                seen.add(row["voice"])
                found += 1
                if found == limit:
                    break
        if found != limit:
            raise ValueError(f"Insufficient distinct source groups for {family}")
    return selected, sha(path)


def render(raw, backgrounds):
    target = np.frombuffer(raw, "<i2").astype(np.float64)
    speech = np.repeat(target[:, None], 2, axis=1)
    scales = [float(np.linalg.norm(target) / np.linalg.norm(cut.astype(np.float64).mean(axis=1)) * 10**.25)
              for cut in backgrounds]
    contributions = [background * scale for background, scale in zip(backgrounds, scales, strict=True)]
    peak = max(float(np.max(np.sum(np.abs(speech) + np.abs(noise), axis=1))) for noise in contributions)
    gain = min(1., 31_129 / peak)
    clean = np.rint(speech * gain).astype(np.int32)
    variants = []
    for scale, residual in zip(scales, contributions, strict=True):
        q = np.rint(residual * gain).astype(np.int32)
        snr = float(20 * np.log10(np.linalg.norm(clean[:, 0]) / np.linalg.norm(q.mean(axis=1))))
        if abs(snr + 5) > .02:
            raise ValueError("The quantized mono reference SNR differs")
        variants.append(((clean + q).astype("<i2"), {"scale": scale, "mono_snr_db": snr}))
    return clean.astype("<i2"), gain, variants


def freeze(folder, ami, annotations):
    if sha(annotations) != serving.ANNOTATION_SHA256:
        raise ValueError("Exact AMI manual annotations required")
    foreground, manifest_sha = foregrounds()
    with ZipFile(annotations) as archive:
        selected = rooms(archive)
    (folder / "wav").mkdir(parents=True)
    (folder / "pcm").mkdir()
    rows, acquisitions, backgrounds = [], {}, defaultdict(list)

    def save(pcm, **metadata):
        index = len(rows)
        path = folder / "wav" / f"{index:03}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setparams((2, 2, RATE, 0, "NONE", ""))
            wav.writeframes(pcm.tobytes())
        wav_sha = sha(path)
        mono = decode_audio(path, source_sha256=wav_sha, max_seconds=3.)
        if mono.mapping["timeline"] != stereo.TIMELINE or mono.mapping["source_channels"] != 2 or len(mono.pcm) != COUNT:
            raise ValueError("The production downmix and native stereo sample clocks differ")
        raw = np.rint(mono.pcm * 32768).astype("<i2").tobytes()
        (folder / "pcm" / f"{index:03}.pcm").write_bytes(raw)
        rows.append({"id": index, "wav_sha256": wav_sha, "pcm_sha256": sha(raw), "stereo_sha256": sha(pcm.tobytes()),
                     "channels_sha256": [sha(pcm[:, i].tobytes()) for i in (0, 1)], **metadata})

    for room, pairs in selected.items():
        paths = [ami / f"{room}.{microphone}.wav" for microphone in MICS]
        containers = [qualify_container(path) for path in paths]
        if containers[0][1] != containers[1][1]:
            raise ValueError("The two native microphones have unequal sample extents")
        acquisitions[room] = {path.name: {"sha256": sha(path), "bytes": path.stat().st_size,
                                         "publisher_container_channels": count,
                                         "url": f"{MIRROR}/{room}/audio/{path.name}"}
                              for path, (count, _) in zip(paths, containers, strict=True)}
        for pair, (_, *cuts) in enumerate(pairs):
            paired = []
            for mode, (start, overlap) in zip(MODES, cuts, strict=True):
                native = native_cut(paths, start * 160)
                meta = {"room": room, "pair": pair, "mode": mode, "native_source_start": start * 160,
                        "annotated_overlap_ms": overlap * 10}
                save(native, family="genuine", group=room, treatment="native", **meta)
                paired.append((native, meta, rows[-1]["id"]))
            backgrounds[room].append(paired)
    for index, (raw, ref) in enumerate(foreground):
        assigned = [item for offset, pairs in enumerate(backgrounds.values()) for item in pairs[(index + offset) % 4]]
        clean, gain, variants = render(raw, [native for native, _, _ in assigned])
        base = {"family": ref["lane"], "group": ref["group_id"], "source_sha256": ref["original_sha256"],
                "clean_pcm_sha256": sha(raw), "common_gain": gain}
        save(clean, treatment="clean", **base)
        clean_index = rows[-1]["id"]
        for (pcm, levels), (_, meta, native_index) in zip(variants, assigned, strict=True):
            save(pcm, treatment="mixed", clean_id=clean_index, native_id=native_index, **levels, **meta, **base)
    write(folder / "roster.json", rows)
    locked = {"schema": "chipmunk.paired-stereo-overlap/1", "roster_sha256": sha(folder / "roster.json"),
              "runner_sha256": sha(Path(__file__)), "gate_code_sha256": sha(Path(stereo.__file__)),
              "incumbent_assets": serving.incumbent.ASSETS, "calibration_sha256": sha(serving.incumbent.CALIBRATION),
              "annotations_sha256": sha(annotations), "foreground_manifest_sha256": manifest_sha,
              "rooms": acquisitions, "planned": dict(Counter(row["treatment"] for row in rows)), "rule": RULE,
              "source_grid": "16 kHz; exactly 48,000 samples; original room cuts retained at native offsets",
              "native_pair": "same annotated two main participants, in separate source ranges, at matched target mono SNR",
              "policy": "unchanged v279 physical channel selection and v283 AI-only rescue"}
    write(folder / "freeze.json", locked)
    print(locked["planned"], flush=True)


def frozen(folder):
    lock, rows = read(folder / "freeze.json"), read(folder / "roster.json")
    if (lock["runner_sha256"] != sha(Path(__file__)) or lock["gate_code_sha256"] != sha(Path(stereo.__file__))
            or lock["roster_sha256"] != sha(folder / "roster.json") or lock["rule"] != RULE
            or lock["calibration_sha256"] != sha(serving.incumbent.CALIBRATION)):
        raise ValueError("The paired room evaluation differs from its prospective freeze")
    return lock, rows


def pcm_views(folder, row):
    path = folder / "wav" / f"{row['id']:03}.wav"
    if sha(path) != row["wav_sha256"]:
        raise ValueError("The stereo WAV bytes differ")
    with wave.open(str(path), "rb") as wav:
        raw = wav.readframes(COUNT + 1)
    mono = (folder / "pcm" / f"{row['id']:03}.pcm").read_bytes()
    if (len(raw), len(mono), sha(raw), sha(mono)) != (COUNT * 4, COUNT * 2, row["stereo_sha256"], row["pcm_sha256"]):
        raise ValueError("The exact stereo or mono PCM differs")
    native = np.frombuffer(raw, "<i2").reshape(-1, 2)
    if [sha(native[:, i].tobytes()) for i in (0, 1)] != row["channels_sha256"]:
        raise ValueError("Physical source-channel identity differs")
    return native, {"mono": np.frombuffer(mono, "<i2").astype(np.float32) / 32768,
                    **{f"ch{i}": native[:, i].astype(np.float32) / 32768 for i in (0, 1)}}


def gate(folder):
    import torch

    from app.detectors.audio.upgraded import SpeechActivity
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack

    lock, rows = frozen(folder)
    serving.incumbent.network_free()
    if sha(stereo.gate.VAD) != lock["incumbent_assets"]["silero_vad.jit"]:
        raise ValueError("The speech activity checkpoint changed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    activity = SpeechActivity(stereo.gate.VAD)
    activity.model = counter = stereo.gate.CountedVAD(activity.model)
    detector, result = PellaVoiceDetector(), []
    for row in rows:
        native, views = pcm_views(folder, row)
        spans, admitted = {}, {}
        for name, values in views.items():
            seconds = activity.spans(values)
            spans[name] = [[round(start * RATE), round(stop * RATE)] for start, stop in seconds]
            cached = SimpleNamespace(obj=SimpleNamespace(spans=lambda _, measured=seconds: measured))
            context = SimpleNamespace(profile="full", lease=SimpleNamespace(get=lambda _, model=cached: model),
                outputs=SimpleNamespace(audio=AudioTrack("analysed", values, COUNT)))
            windows = detector._windows(context)
            admitted[name] = bool(windows)
            if windows and (len(windows) != 1 or windows[0][0] != 0 or not np.array_equal(windows[0][1], values)):
                raise ValueError("Serving window changed the exact physical view")
        observation = stereo.observation(native, spans)
        if observation["admitted"] != admitted:
            raise ValueError("The v279 gate and serving adapter differ")
        result.append({"id": row["id"], "spans": spans, **observation})
    if counter.calls != len(rows) * len(VIEWS) * stereo.FORWARDS_PER_VIEW:
        raise ValueError("Unexpected number of speech activity forwards")
    write(folder / "gate.json", {"freeze_sha256": sha(folder / "freeze.json"), "rows": result, "forwards": counter.calls})
    print(dict(Counter(row["reason"] for row in result)), flush=True)


def classify(folder):
    from app.detectors.audio import calibration
    from app.engine.boundary import Deadline
    from scripts.run_audio_specialists_v15 import load_voice

    lock, rows = frozen(folder)
    admitted = read(folder / "gate.json")
    if admitted["freeze_sha256"] != sha(folder / "freeze.json"):
        raise ValueError("Channel admission was not recorded on the frozen media")
    serving.incumbent.network_free()
    voice = load_voice({"weights": str(serving.incumbent.WEIGHTS), "assets": lock["incumbent_assets"]})
    count = 0

    def forward(*_):
        nonlocal count
        count += 1

    hook = voice.register_forward_pre_hook(forward)
    result = []
    try:
        for row, gate_row in zip(rows, admitted["rows"], strict=True):
            native, waves = pcm_views(folder, row)
            selected = gate_row["selected"]
            expected = stereo.observation(native, gate_row["spans"])
            if gate_row["id"] != row["id"] or {k: gate_row[k] for k in expected} != expected:
                raise ValueError("Independent acoustic channel selection differs")
            views = [("mono", "mono")]
            if selected is not None:
                views.append(("selected", f"ch{selected}"))
            outputs = {}
            for policy_name, view in views:
                if not gate_row["admitted"][view]:
                    continue
                waveform = waves[view]
                raw, = voice.predict([waveform])
                result_ = calibration.evaluate(raw, waveform, [(0., waveform)], voice, Deadline(120))
                if not result_["scope_supported"]:
                    raise ValueError("The selected view is outside the calibrated extent")
                outputs[policy_name] = {"raw": raw, "decision": result_["decision"], "probe_status": result_["probe_status"],
                                        "stability_log_odds_change": result_["stability_log_odds_change"]}
            result.append({"id": row["id"], **outputs})
            if len(result) % 20 == 0:
                print(f"Scored {len(result)} views ({count} classifier forwards)", flush=True)
    finally:
        hook.remove()
    write(folder / "scores.json", {"freeze_sha256": sha(folder / "freeze.json"), "gate_sha256": sha(folder / "gate.json"),
                                  "rows": result, "forwards": count, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def readout(folder, ami):
    lock, rows = frozen(folder)
    gates, scores = (read(folder / name) for name in ("gate.json", "scores.json"))
    if (scores["gate_sha256"] != sha(folder / "gate.json")
            or any(receipt["freeze_sha256"] != sha(folder / "freeze.json") for receipt in (gates, scores))):
        raise ValueError("Score and gate artifacts are not tied to the frozen inputs")
    for acquisition in lock["rooms"].values():
        if any(sha(ami / name) != item["sha256"] or qualify_container(ami / name)[0] != item["publisher_container_channels"]
               for name, item in acquisition.items()):
            raise ValueError("Original physical stereo recording differs")
    cases = {row["id"]: pcm_views(folder, row)[0] for row in rows}
    counts, evidence, gains = defaultdict(Counter), [], []
    wrong = Counter()
    for row, gate_row, score in zip(rows, gates["rows"], scores["rows"], strict=True):
        index, treatment = row["id"], row["treatment"]
        if (gate_row["id"], score["id"]) != (index, index):
            raise ValueError("Measured decisions do not follow the frozen roster")
        if treatment == "native":
            paths = [ami / f"{row['room']}.{mic}.wav" for mic in MICS]
            if not np.array_equal(native_cut(paths, row["native_source_start"]), cases[index]):
                raise ValueError("Source microphone cut differs")
        if treatment == "mixed":
            clean, background = cases[row["clean_id"]].astype(np.int32), cases[row["native_id"]]
            quantized = np.rint(background * row["scale"] * row["common_gain"]).astype(np.int32)
            if not np.array_equal(clean + quantized, cases[index]):
                raise ValueError("Independent exact paired residual reconstruction failed")
        selected = gate_row["selected"]
        baseline = score.get("mono", {}).get("decision", "unassessed")
        candidate = score.get("selected", {}).get("decision", "unassessed")
        added = baseline in {"inconclusive", "unassessed"} and candidate == "ai"
        key = (treatment, row.get("room"), row.get("mode"), row["family"])
        counts[key].update(planned=1, selected=selected is not None, assessed=baseline != "unassessed",
                           baseline_ai=baseline == "ai", baseline_original=baseline == "original",
                           baseline_inconclusive=baseline == "inconclusive", added_ai=added, channel_admitted=candidate != "unassessed")
        if added:
            if row["family"] == "genuine":
                wrong[treatment] += 1
            else:
                gains.append(row)
            evidence.append({"id": index, "view": selected, "start_sample": 0, "end_sample": COUNT, "time_base": "1/16000",
                             "stereo_wav_sha256": row["wav_sha256"], "selected_pcm_sha256": row["channels_sha256"][selected]})
    write(folder / "evidence-private.json", evidence)
    groups, families = ({row[field] for row in gains} for field in ("group", "family"))
    modes = {row.get("mode") for row in gains if row["treatment"] == "mixed"}
    tests = {"gains": len(groups) >= RULE["minimum_additional_synthetic_groups"],
             "families": len(families) >= RULE["minimum_added_synthetic_families"],
             "conditions": len(modes) >= RULE["minimum_conditions_with_gain"],
             "false": sum(wrong.values()) <= RULE["maximum_additional_genuine_false_flags"],
             "clean_and_native": wrong["clean"] + wrong["native"] <= RULE["maximum_additional_clean_or_native_false_flags"],
             "coverage": all(counts[("mixed", room, mode, "genuine")]["assessed"] >= RULE["minimum_assessed_genuine_per_room_condition"]
                             for room in lock["rooms"] for mode in MODES)}
    result = {"schema": "chipmunk.paired-stereo-overlap-result/1", "freeze": lock, "freeze_sha256": sha(folder / "freeze.json"),
              "gate_sha256": sha(folder / "gate.json"), "scores_sha256": sha(folder / "scores.json"), "checks": tests,
              "advance": all(tests.values()), "gains": len(gains), "gained_groups": len(groups), "gained_families": len(families),
              "additional_false": sum(wrong.values()), "source_intervals": len(evidence),
              "evidence_sha256": sha(folder / "evidence-private.json"), "speech_gate_forwards": gates["forwards"],
              "selection_reasons": dict(Counter(row["reason"] for row in gates["rows"])),
              "classifier_forwards": scores["forwards"], "peak_rss_bytes": scores["peak_rss_bytes"],
              "cells": [{"treatment": t, "room": room, "mode": mode, "family": family, **value}
                        for (t, room, mode, family), value in counts.items()]}
    write(folder / "aggregate.json", result)
    print({k: result[k] for k in ("gains", "additional_false", "checks", "advance")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "gate", "classify", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--ami", type=Path)
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.folder, args.ami, args.annotations)
    elif args.stage == "gate":
        gate(args.folder)
    elif args.stage == "classify":
        classify(args.folder)
    else:
        readout(args.folder, args.ami)
