"""Test whether adjacent ambience can identify limits in the original voice input."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import audio_gtcrn_traffic as prior

ROOT, RATE, COUNT = prior.ROOT, prior.RATE, prior.COUNT
read, sha, write_new = prior.read, prior.sha, prior.write_new
ASSETS = prior.earlier.ASSETS
ROSTER = {"genuine": 8, "clone": 2, "expressive": 2, "stock": 2}
SALT = b"chipmunk/original-audio-background-limited/2026-11-11/"
TAPES = {
    "lyngby": ("lyngby.ogg", "cadfaba6820e47967d0f296c9fad2214913458b0", "CC BY-SA 4.0", 9),
    "gransee": ("gransee.oga", "a25e47a41dbf00479f08e3c2bdf34c383a26ae2c", "CC BY-SA 4.0", 0),
    "sunday": ("sunday.ogg", prior.SUNDAY_SHA1, "Public domain", 84),
}
REF = ((0, 5600), (42400, COUNT))
FOCUS = tuple(zip(range(5600, 42400, 9200), range(14800, 42401, 9200), strict=True))
FREQUENCIES = np.fft.rfftfreq(512, 1 / RATE)
BANDS = (250, 500, 1000, 2000, 4000)
LEVELS = (-6, 6)
RULE = {
    "required_center_speech_samples": 3584, "maximum_reference_speech_samples": 1280,
    "minimum_reference_dbfs": -54, "maximum_flank_band_difference_db": 5, "minimum_agreeing_bands": 3,
    "maximum_mixed_power_excess_db": 2.5, "maximum_band_shape_difference_db": 4,
    "ground_truth_masking_snr_db": -3, "ground_truth_audible_snr_db": 3,
    "minimum_masked_recall_per_scene": .40, "minimum_scenes_with_recall": 2,
    "maximum_false_audible_rate": .10, "maximum_false_absent_rate": .05,
    "minimum_genuine_groups_with_hits": 4, "minimum_synthetic_groups_with_hits": 4,
}


def order(key):
    return sha(SALT + str(key).encode())


def source_groups(archive, old_paths):
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "md5").hexdigest() != prior.earlier.MD5:
            raise ValueError("Unexpected LibriSpeech archive")
    reference = read(prior.earlier.PANEL / "audio-traffic-enhancement-transfer-2026-11-11.json")
    if (sha(old_paths[2]) != reference["freeze"]["roster_sha256"]
            or sha(old_paths[0]) != reference["freeze"]["sources"]["prior_traffic_roster"]
            or sha(old_paths[1]) != reference["freeze"]["sources"]["prior_two_background_roster"]):
        raise ValueError("Earlier foreground group comparison manifests differ")
    old_rows = [row for path in old_paths for row in read(path)]
    old_sources = {row["source_pcm_sha256"] for row in old_rows if row.get("source_pcm_sha256")}
    speakers = {row["speaker"] for row in old_rows if row.get("speaker")}
    groups = {row["group"] for row in old_rows if row.get("group")}
    receipt = read(prior.earlier.PREP)
    synthetic, manifests = [], {}
    for role in ("development", "holdout"):
        path = prior.earlier.clips.sources.MEDIA / role / "private-manifest.json"
        manifests[role] = sha(path)
        if manifests[role] != receipt["roles"][role]["private_manifest_sha256"]:
            raise ValueError("Unexpected qualified synthetic source manifest")
        for row in read(path)["foregrounds"]:
            if row["lane"] in prior.FAMILIES:
                pcm = prior.earlier.clips.exact_source(row, role)
                synthetic.append((row, pcm))
                if sha(pcm) in old_sources:
                    speakers.add(row["voice"])
                    groups.add(row["group_id"])
    with gzip.open(prior.earlier.CATALOGUE, "rt") as stream:
        catalogue = json.load(stream)["sources"]
    speaker_hashes = {row["canonical_speaker_sha256"] for row in catalogue}
    old_hashes = old_sources | {row[k] for row in catalogue for k in ("sha256", "prefix_pcm_sha256", "production_pcm_sha256")}
    selected = []
    for family in prior.FAMILIES:
        count = 0
        for row, raw in sorted(synthetic, key=lambda item: order(item[0]["slot"])):
            if (row["lane"] != family or row["voice"] in speakers or row["group_id"] in groups
                    or {sha(raw), row["original_sha256"], row["prefix_pcm_sha256"]} & old_hashes):
                continue
            selected.append((family, row["group_id"], sha(raw), raw))
            speakers.add(row["voice"])
            groups.add(row["group_id"])
            count += 1
            if count == ROSTER[family]:
                break
        if count != ROSTER[family]:
            raise ValueError(f"Insufficient distinct prepared voices in {family}")
    genuine = []
    with tarfile.open(archive, "r:gz") as tar:
        candidates = defaultdict(list)
        for member in tar:
            match = re.fullmatch(r"LibriSpeech/test-clean/(\d+)/\d+/\d+-\d+-\d+\.flac", member.name)
            if match and member.isfile() and 0 < member.size <= 2_000_000:
                speaker = f"librispeech:{match[1]}"
                if speaker not in speakers and sha(speaker.encode()) not in speaker_hashes:
                    candidates[speaker].append(member)
        for speaker, members in sorted(candidates.items(), key=lambda item: order(item[0])):
            for member in sorted(members, key=lambda m: order(m.name)):
                with tar.extractfile(member) as stream:
                    encoded = stream.read()
                body = prior.decode(encoded, "flac")
                start = (len(body) // 2 - COUNT) // 2
                if start < 4000:
                    continue
                raw = body[start * 2:(start + COUNT) * 2]
                if {sha(encoded), sha(raw), prior.earlier.clips.float_pcm_sha(raw)} & old_hashes:
                    continue
                genuine.append(("genuine", speaker, sha(raw), raw))
                break
            if len(genuine) == ROSTER["genuine"]:
                break
    if len(genuine) != ROSTER["genuine"]:
        raise ValueError("Insufficient new known LibriSpeech reader groups")
    return genuine + selected, {"librispeech": sha(archive), "qualified_synthetic": manifests,
                                "prior_rosters": [sha(path) for path in old_paths]}


def background(folder):
    result, commitments = {}, {}
    for tape, (filename, expected, license_name, start_seconds) in TAPES.items():
        metadata = folder / f"{tape}.json"
        record, = [row for page in read(metadata)["query"]["pages"].values() for row in page["imageinfo"]]
        encoded = (folder / filename).read_bytes()
        if (len(encoded) != record["size"] or hashlib.sha1(encoded).hexdigest() != expected
                or record["sha1"] != expected or record["extmetadata"]["LicenseShortName"]["value"] != license_name):
            raise ValueError("Traffic recording differs from its source checksum and terms")
        pcm = prior.decode(encoded, "ogg")
        clips = [(offset, pcm[offset * 2:(offset + COUNT) * 2])
                 for offset in range(start_seconds * RATE, (start_seconds + 24) * RATE, COUNT)]
        if any(len(raw) != 2 * COUNT or not any(raw) for _, raw in clips):
            raise ValueError("The frozen traffic ranges must each be exact three-second PCM")
        result[tape] = sorted(clips, key=lambda item: order(sha(item[1])))
        commitments[tape] = {"original_sha256": sha(encoded), "publisher_metadata_sha256": sha(metadata),
                             "pcm_sha256": sha(pcm), "license": license_name}
    return result, commitments


def spec_powers(wave):
    frames = np.lib.stride_tricks.sliding_window_view(wave, 512)[::256]
    window = np.hanning(512)
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2 * (2 / (512 * np.sum(window**2)))
    return np.stack([power[:, (a <= FREQUENCIES) & (b > FREQUENCIES)].sum(axis=1)
                     for a, b in zip(BANDS[:-1], BANDS[1:], strict=True)], axis=1)


def db(value):
    return 10 * np.log10(np.maximum(value, 1e-12))


def support_spans(spans):
    return [(round(a * RATE), round(b * RATE)) for a, b in spans]


def coverage(wave, spans):
    def speech(a, b):
        return sum(max(0, min(b, end) - max(a, start)) for start, end in spans)

    flanks = np.array([np.median(spec_powers(wave[a:b]), axis=0) for a, b in REF])
    reference = flanks.mean(axis=0)
    if speech(FOCUS[0][0], FOCUS[-1][1]) < RULE["required_center_speech_samples"]:
        status = "unassessed_no_speech_anchor"
    elif any(speech(a, b) > RULE["maximum_reference_speech_samples"] for a, b in REF):
        status = "unassessed_reference_speech"
    elif min(db(flanks.sum(axis=1))) < RULE["minimum_reference_dbfs"]:
        status = "unassessed_reference_quiet"
    elif np.count_nonzero(np.abs(db(flanks[0]) - db(flanks[1])) <= RULE["maximum_flank_band_difference_db"]) < RULE["minimum_agreeing_bands"]:
        status = "unassessed_reference_changed"
    else:
        status = "assessed"
    rows = []
    for a, b in FOCUS:
        band = np.median(spec_powers(wave[a:b]), axis=0)
        excess = float(db(band.sum()) - db(reference.sum()))
        agreement = int(np.count_nonzero(np.abs((db(band) - db(reference)) - excess) <= RULE["maximum_band_shape_difference_db"]))
        flag = status == "assessed" and excess <= RULE["maximum_mixed_power_excess_db"] and agreement >= RULE["minimum_agreeing_bands"]
        rows.append({"start_sample": a, "end_sample": b, "excess_db": excess, "matching_bands": agreement, "flag": flag,
                     "detected_speech_samples": speech(a, b)})
    return {"scope": status, "ref_power_db": db(flanks).tolist(), "speech_spans": spans, "intervals": rows}


def embed(raw):
    start, end = FOCUS[0][0], FOCUS[-1][1]
    result = np.zeros(COUNT)
    piece = prior.float_pcm(raw)[start:end].astype(np.float64)
    ramp = .5 - .5 * np.cos(np.linspace(0, np.pi, 400))
    piece[:len(ramp)] *= ramp
    piece[-len(ramp):] *= ramp[::-1]
    result[start:end] = piece
    return prior.pcm(result)


def mix(foreground, noise, level, *, mix_band):
    target = slice(FOCUS[0][0], FOCUS[-1][1])
    if mix_band == "speech_band":
        foreground_energy, noise_energy = (float(spec_powers(signal[target]).sum()) for signal in (foreground, noise))
        ratio = math.sqrt(foreground_energy / noise_energy)
    else:
        ratio = np.linalg.norm(foreground[target]) / np.linalg.norm(noise[target])
    multiplier = ratio * 10 ** (-level / 20)
    total = foreground + noise * multiplier
    gain = min(1., .95 / np.max(np.abs(total)))
    return prior.pcm(total * gain), float(multiplier), float(gain)


def ground_truth(foreground, noise):
    p = [float(spec_powers(foreground[a:b]).mean(axis=0).sum()) for a, b in FOCUS]
    n = [float(spec_powers(noise[a:b]).mean(axis=0).sum()) for a, b in FOCUS]
    levels = [float(db(value) - db(other)) for value, other in zip(p, n, strict=True)]
    truth = ["absent" if value < max(1e-6, .016 * max(p)) else
             "masked" if level <= RULE["ground_truth_masking_snr_db"] else
             "audible" if level >= RULE["ground_truth_audible_snr_db"] else "transition"
             for value, level in zip(p, levels, strict=True)]
    return truth, levels


def binding():
    dependencies = (Path(__file__).resolve(), Path(prior.__file__).resolve(), Path(prior.earlier.__file__).resolve(),
                    prior.earlier.CALIBRATION, ROOT / "backend/app/detectors/audio/calibration.py",
                    ROOT / "backend/app/detectors/audio/upgraded.py", ROOT / "backend/app/detectors/video/adapters/audio.py",
                    ROOT / "backend/scripts/run_audio_specialists_v15.py")
    return {"code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in dependencies}, "assets": ASSETS,
            "rule": RULE, "foreground_counts": ROSTER, "ref": [list(row) for row in REF],
            "focus": [list(row) for row in FOCUS], "bands_hz": list(BANDS), "fft": 512, "hop": 256,
            "sample_rate": RATE, "samples": COUNT, "levels_db": list(LEVELS)}


def prepare(folder, archive, previous, older, enhancement, mix_band):
    lock = binding()
    voices, sources = source_groups(archive, (previous, older, enhancement))
    tapes, environments = background(folder)
    (folder / "pcm").mkdir()
    (folder / "noise").mkdir()
    rows = []

    def save(raw, metadata):
        index = len(rows)
        (folder / "pcm" / f"{index:03}.pcm").write_bytes(raw)
        rows.append({"id": index, "pcm_sha256": sha(raw), **metadata})

    for cuts in tapes.values():
        for _, raw in cuts:
            (folder / "noise" / f"{sha(raw)}.pcm").write_bytes(raw)
    for index, (family, group, source_sha, raw) in enumerate(voices):
        clean = embed(raw)
        wave = prior.float_pcm(clean).astype(np.float64)
        fields = {"clean_id": len(rows), "family": family, "group": group, "source_pcm_sha256": source_sha}
        save(clean, {**fields, "condition": "clean", "tape": None})
        for tape, cuts in sorted(tapes.items()):
            offset, body = cuts[index % len(cuts)]
            noise = prior.float_pcm(body).astype(np.float64)
            for level in LEVELS:
                mixed, scale, gain = mix(wave, noise, level, mix_band=mix_band)
                save(mixed, {**fields, "condition": str(level), "tape": tape, "noise_sha256": sha(body),
                             "noise_start_sample": offset, "noise_gain": scale, "mix_gain": gain})
    for tape, cuts in sorted(tapes.items()):
        for offset, raw in cuts[:2]:
            save(raw, {"family": "background", "group": tape, "condition": "isolated", "tape": tape,
                       "noise_start_sample": offset, "noise_sha256": sha(raw)})
    write_new(folder / "roster.json", rows)
    write_new(folder / "freeze.json", {**lock, "schema": "chipmunk.audio-background-limited-freeze/1", "views": len(rows), "mix_band": mix_band,
                                       "source_commitments": sources, "background_commitments": environments,
                                       "roster_sha256": sha(folder / "roster.json")})
    print(f"Frozen {len(rows)} views, {len(voices)} source groups and {len(tapes)} street recordings", flush=True)


def frozen(folder):
    lock, rows = read(folder / "freeze.json"), read(folder / "roster.json")
    if (lock["mix_band"] not in {"full_band", "speech_band"}
            or any(lock[key] != value for key, value in binding().items()) or sha(folder / "roster.json") != lock["roster_sha256"]
            or [row["id"] for row in rows] != list(range(lock["views"]))):
        raise ValueError("Frozen inputs or method differ")
    return lock, rows


def run(folder):
    from app.detectors.audio import calibration
    from app.detectors.audio.upgraded import SpeechActivity
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from app.engine.boundary import Deadline
    from scripts.audio_noise_speech_gate_v256 import CountedVAD
    from scripts.run_audio_specialists_v15 import load_voice

    lock, roster = frozen(folder)
    prior.earlier.network_free()
    weights = prior.earlier.WEIGHTS
    voice = load_voice({"weights": str(weights), "assets": lock["assets"]})
    activity = SpeechActivity(weights / "silero_vad.jit")
    activity.model = gate = CountedVAD(activity.model)
    detector = PellaVoiceDetector()
    results, forwards = [], 0

    def track(*_):
        nonlocal forwards
        forwards += 1

    hook = voice.register_forward_pre_hook(track)
    try:
        for row in roster:
            wave = prior.load(folder, row)
            spans = activity.spans(wave)
            state = coverage(wave, support_spans(spans))
            cached = SimpleNamespace(obj=SimpleNamespace(spans=lambda _, samples=spans: samples))
            ctx = SimpleNamespace(profile="full", lease=SimpleNamespace(get=lambda _, model=cached: model),
                                  outputs=SimpleNamespace(audio=AudioTrack("analysed", wave, len(wave))))
            selected = detector._windows(ctx)
            state.update(id=row["id"], pcm_sha256=row["pcm_sha256"], current="unassessed", raw=None, stability=None)
            if selected:
                if len(selected) != 1 or selected[0][0] != 0 or not np.array_equal(wave, selected[0][1]):
                    raise ValueError("The current app gate returned a different window")
                raw, = voice.predict([wave])
                score = calibration.evaluate(raw, wave, selected, voice, Deadline(120))
                state.update(current=score["decision"], raw=float(raw), stability=score["stability_log_odds_change"])
            results.append(state)
            if len(results) % 15 == 0:
                print(f"Original voice coverage {len(results)}/{len(roster)} ({forwards} classifier forwards)", flush=True)
    finally:
        hook.remove()
    write_new(folder / "measurements.json", {"freeze_sha256": sha(folder / "freeze.json"), "classifier_forwards": forwards,
                                            "activity_forwards": gate.calls, "rows": results})


def readout(folder):
    lock, roster = frozen(folder)
    measures = read(folder / "measurements.json")
    if measures["freeze_sha256"] != sha(folder / "freeze.json"):
        raise ValueError("Current measurements bind another freeze")
    cells, outcomes, scopes = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    hit_groups = defaultdict(set)
    matched, review_cases, confidence_effects = 0, [], defaultdict(Counter)
    for row, observed in zip(roster, measures["rows"], strict=True):
        if (row["id"], row["pcm_sha256"]) != (observed["id"], observed["pcm_sha256"]):
            raise ValueError("Measurement binding differs")
        original = prior.load(folder, row)
        if row["condition"] == "isolated":
            truth = ["absent"] * len(FOCUS)
        else:
            fg = prior.load(folder, roster[row["clean_id"]]).astype(np.float64)
            noise = np.zeros(COUNT)
            if row["condition"] != "clean":
                raw = (folder / "noise" / f"{row['noise_sha256']}.pcm").read_bytes()
                if sha(raw) != row["noise_sha256"]:
                    raise ValueError("Exact isolated background differs")
                noise = prior.float_pcm(raw).astype(np.float64)
                reconstructed, scale, gain = mix(fg, noise, int(row["condition"]), mix_band=lock["mix_band"])
                if sha(reconstructed) != row["pcm_sha256"] or (scale, gain) != (row["noise_gain"], row["mix_gain"]):
                    raise ValueError("Original mixed sample reconstruction differs")
                noise *= scale
                matched += 1
            truth, _ = ground_truth(fg, noise)
        kind = "synthetic" if row["family"] in prior.FAMILIES else row["family"]
        scopes[(kind, row["condition"])][observed["scope"]] += 1
        outcomes[(kind, row["condition"])][observed["current"]] += 1
        if any(item["flag"] for item in observed["intervals"]):
            confidence_effects[(kind, row["condition"])][observed["current"]] += 1
        replay = coverage(original, observed["speech_spans"])
        if any(observed[key] != value for key, value in replay.items()):
            raise ValueError("Acoustic measurement replay differs from the original samples")
        for label, item, interval in zip(truth, observed["intervals"], FOCUS, strict=True):
            if (item["start_sample"], item["end_sample"]) != interval:
                raise ValueError("Measured interval differs from the source sample clock")
            key = (kind, row["condition"], row["tape"] or "clean", label)
            bucket = cells[key]
            bucket["planned"] += 1
            bucket["assessed"] += observed["scope"] == "assessed"
            bucket["flagged"] += item["flag"]
            if label == "masked" and item["flag"]:
                hit_groups[kind].add(row["group"])
                a, b = interval
                review_cases.append({"id": row["id"], "family": kind, "tape": row["tape"], "source_pcm_sha256": row["pcm_sha256"],
                                     "clip_pcm_sha256": sha(prior.pcm(original[a:b])), "source_sample_span": [a, b],
                                     "ref_sample_spans": [list(x) for x in REF], "time_base": "1/16000"})
    sums = defaultdict(Counter)
    for (_, _, tape, label), value in cells.items():
        sums[("all", label)].update(value)
        sums[(tape, label)].update(value)
    scene_passes = sum(sums[(tape, "masked")]["planned"] > 0 and
                       sums[(tape, "masked")]["flagged"] / sums[(tape, "masked")]["planned"] >= RULE["minimum_masked_recall_per_scene"]
                       for tape in TAPES)
    checks = {"masked_recall": scene_passes >= RULE["minimum_scenes_with_recall"],
              "audible_false_rate": sums[("all", "audible")]["flagged"] / sums[("all", "audible")]["planned"] <= RULE["maximum_false_audible_rate"],
              "absent_false_rate": sums[("all", "absent")]["flagged"] / sums[("all", "absent")]["planned"] <= RULE["maximum_false_absent_rate"],
              "genuine_hit_groups": len(hit_groups["genuine"]) >= RULE["minimum_genuine_groups_with_hits"],
              "synthetic_hit_groups": len(hit_groups["synthetic"]) >= RULE["minimum_synthetic_groups_with_hits"]}
    write_new(folder / "private-evidence.json", review_cases)
    pack = {"schema": "chipmunk.audio-background-limited-evaluation/1", "freeze_sha256": sha(folder / "freeze.json"),
            "freeze": lock, "measurements_sha256": sha(folder / "measurements.json"),
            "private_evidence_sha256": sha(folder / "private-evidence.json"), "matched_mixtures_reconstructed": matched,
            "classifier_forwards": measures["classifier_forwards"], "activity_forwards": measures["activity_forwards"],
            "advance": all(checks.values()), "checks": checks, "scene_passes": scene_passes,
            "hit_group_counts": {k: len(v) for k, v in hit_groups.items()},
            "scopes": [{"family": kind, "condition": level, **value} for (kind, level), value in sorted(scopes.items())],
            "interval_summary": [{"tape": tape, "truth": label, **value} for (tape, label), value in sorted(sums.items())],
            "cells": [{"family": kind, "condition": level, "tape": tape, "truth": label, **value}
                      for (kind, level, tape, label), value in sorted(cells.items())],
            "current_results": [{"family": kind, "condition": level, "current": value,
                                 "overlap_any_coverage_warning": confidence_effects[(kind, level)]}
                                for (kind, level), value in sorted(outcomes.items())]}
    write_new(folder / "aggregate.json", pack)
    print(json.dumps({"advance": pack["advance"], "checks": checks, "groups": pack["hit_group_counts"],
                      "scene_passes": scene_passes, "warnings": len(review_cases)}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--librispeech", type=Path)
    parser.add_argument("--prior-traffic", type=Path)
    parser.add_argument("--prior-background", type=Path)
    parser.add_argument("--prior-enhancement", type=Path)
    parser.add_argument("--mix-band", choices=("full_band", "speech_band"), default="full_band")
    args = parser.parse_args()
    if args.action == "freeze":
        if not all((args.librispeech, args.prior_traffic, args.prior_background, args.prior_enhancement)):
            parser.error("Freeze requires the publisher archive and three previous case manifests")
        prepare(args.folder, args.librispeech, args.prior_traffic, args.prior_background, args.prior_enhancement, args.mix_band)
    else:
        {"run": run, "readout": readout}[args.action](args.folder)


if __name__ == "__main__":
    main()
