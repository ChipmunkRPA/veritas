# SPDX-License-Identifier: Apache-2.0
"""Test safeguards for false speech alerts after background reduction."""
from __future__ import annotations

import argparse
import math
import tarfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import audio_background_rate_diagnostic as prior

ROOT, sha, load, save, require = prior.ROOT, prior.sha, prior.load, prior.save, prior.require
PUBLIC = ROOT / "docs/evaluation-panels"
SCHEMA = "chipmunk.audio-background-confirmation/1"
POINT = prior.METHOD["point"]
RATE = 16_000
SPAN = 3 * RATE
PREFIXES = {
    "radar": ("radar-96.gz", 96 * 1024**2, "a272b90b10710572aca7f9dcae2eafd474597388a2382ea5a3d81aff0e4d5447"),
    "llama": ("llama-320.tar", 320 * 1024**2, "ffeafa9e3432b70cbf6a6a7daf69ef88d2d47d884d62d7ffab5b09fa70a8cab5"),
}
BACKGROUNDS = {
    "rain": {"sha256": "e0233652d8173650149adcf3d6e5ddc01b73e8a93ce61b8d7809abfe582251a8", "offsets": [3, 15, 29, 43]},
    "train": {"sha256": "06746a387a2995595b99da5c3f43b2008681fc98fadf6ad513d9a4226973f179", "offsets": [32, 58, 84, 110]},
    "traffic": {"sha256": "9702be26fec457ac8c54df61fbe62f120189119b5734377cda9e5ae442566efd", "offsets": [2, 6, 10, 14]},
}
PANEL_FILE = "audio-nemotron-noise-{}-2026-11-12.json"
CALIBRATION = ("transfer", "confirmation")
PROBES = ("source_band", "enhanced_inverted")
METHOD = {
    "selection": "18 RADAR English sources per label and all available unused Llama speakers, one utterance each, preferring synthetic families with fewer speakers. Sort on SHA-256 of schema and publisher ID. Require a single label for 3–14 seconds. Crop the exact center three seconds.",
    "controls": "Common peak-safe foreground gain, clean plus rain, train and traffic at -5 dB foreground-to-added-sound RMS. Four exact background-only excerpts per recording. Each upload is a new 16 kHz PCM16 WAV.",
    "eligible": "Use the existing voice result. Only primary-inconclusive or primary-unassessed clips enter the optional check. Use the incumbent speech activity gate and processed voice operating point.",
    "probes": {"source_band": "An additional voice response on the original waveform band-limited to 100–3800 Hz, using the incumbent 4th-order zero-phase filter. This test evaluation forwards the same exact samples without requiring a second speech-activity gate.",
               "enhanced_inverted": "An additional voice response on the exact background-reduced PCM with reversed polarity. This view preserves audible content and tests score stability. No second enhancement or speech gate."},
    "point": POINT,
    "calibration": "Fit each one-sided point separately using only the 19 existing supplemental synthetic flags. Point is the second-lowest reading, inclusive, to retain 18 of 19. Incumbent stays fixed. No holdout-based refitting or use of oracle noise inputs.",
    "advance": {"minimum_removed_genuine": 2, "minimum_removed_backgrounds": 2, "minimum_retained_synthetic_fraction": .85,
                "maximum_lost_in_any_scene_fraction": .25, "minimum_noisy_incumbent_flags": 6},
}


def known():
    partial = load(PUBLIC / "audio-partial-publisher-transfer-2026-11-12.json")["roster"]
    radar = load(PUBLIC / "audio-radar-transfer-2026-11-12.json")["intake"]
    finished = {name: load(PUBLIC / PANEL_FILE.format(name))["held"] for name in CALIBRATION}
    return {"llama": {str(row["source"]["speaker"]) for row in partial if row.get("source")}
                     | {row["group"] for row in finished["transfer"] if row["group"]},
            "radar": {row["id"] for row in radar} | {row["group"] for row in finished["confirmation"] if row["group"]}}


def raw_sources(folder, radar, llama):
    from scripts import audio_partial_publisher_transfer as partial_lib
    from scripts import audio_radar_transfer as radar_lib

    excludes = known()
    for publisher, settings in {"radar": (radar, radar_lib.MEMBER, "labels.txt", radar_lib.SOURCE_FILES["labels.txt"]),
                                 "llama": (llama, partial_lib.MEMBER, "part-a.labels.txt", partial_lib.PINS["part-a.labels.txt"])}.items():
        original, members, label_path, label_hash = settings
        fname, size, archive_hash = PREFIXES[publisher]
        path = folder / fname
        require(path.stat().st_size == size and sha(path.read_bytes()) == archive_hash, "Publisher archive prefix differs")
        label_bytes = (original / label_path).read_bytes()
        require(sha(label_bytes) == label_hash, "Publisher labels differ")
        labels = {name: values for name, *values in map(str.split, label_bytes.decode().splitlines())}
        languages = {}
        if publisher == "radar":
            languages_bytes = (original / "languages.txt").read_bytes()
            require(sha(languages_bytes) == radar_lib.SOURCE_FILES["languages.txt"], "Publisher languages differ")
            languages = dict(line.split() for line in languages_bytes.decode().splitlines())
        available = []
        with tarfile.open(path, "r|gz" if publisher == "radar" else "r|") as archive:
            try:
                for member in archive:
                    match = members.fullmatch(member.name)
                    if not member.isfile() or not match:
                        continue
                    name = match[1]
                    speaker = name if publisher == "radar" else match[2]
                    if speaker in excludes[publisher] or (publisher == "radar" and languages.get(name) != "EN"):
                        continue
                    duration, label, *segments = labels[name]
                    if not 3 <= Decimal(duration) <= 14 or len(segments) != 1:
                        continue
                    start, end, segment_label = segments[0].split("-")
                    if Decimal(start) != 0 or Decimal(end) != Decimal(duration) or segment_label != label:
                        continue
                    with archive.extractfile(member) as entry:
                        media = entry.read()
                    require(len(media) == member.size and label in {"bonafide", "spoof"}, "Incomplete publisher member")
                    family = name.split("-")[1] if publisher == "llama" and label == "spoof" else None
                    available.append(({"publisher": publisher, "id": name, "speaker_or_file": speaker, "label": label,
                                       "family": family, "source_sha256": sha(media), "duration": duration}, media))
            except (EOFError, tarfile.ReadError) as error:
                require(isinstance(error, EOFError) or "end of data" in str(error), "Malformed bounded archive")
        seen = set()
        strata = Counter((r["label"], r["family"]) for r, _ in available)
        taken = Counter()
        for row, media in sorted(available, key=lambda pair: (strata[pair[0]["label"], pair[0]["family"]],
                                                sha(f"{SCHEMA}/{pair[0]['id']}".encode()))):
            speaker = row["speaker_or_file"]
            if speaker in seen or (publisher == "radar" and taken[row["label"]] == 18):
                continue
            taken[row["label"]] += 1
            seen.add(speaker)
            if publisher == "radar":
                raw = prior.previous.ffmpeg("-f", "flac", "-i", "pipe:0", "-ac", "1", "-ar", str(RATE), "-c:a",
                                            "pcm_s16le", "-f", "s16le", "pipe:1", pcm=media)
            else:
                raw = partial_lib.native(media)
                require(partial_lib.labels_for(labels[row["id"]], len(raw) // 2) == [[0, len(raw) // 2, row["label"]]],
                        "Publisher duration or label clock differs")
            require(len(raw) >= 2 * SPAN and abs(round(Decimal(row["duration"]) * RATE) - len(raw) // 2) <= 16,
                    "Foreground sample clock differs")
            start = (len(raw) // 2 - SPAN) // 2
            clip = raw[2 * start:2 * (start + SPAN)]
            yield row | {"source_pcm_sha256": sha(raw), "start_sample": start, "clip_sha256": sha(clip)}, clip
        if publisher == "radar":
            require(taken == {"bonafide": 18, "spoof": 18}, "Not enough eligible distinct RADAR files")
        else:
            require(taken["bonafide"] >= 8 and taken["spoof"] >= 3, "Not enough eligible distinct Llama speakers")


def freeze(run, source, radar, llama, sounds):
    import numpy as np

    from scripts import audio_vocal_event_mixtures as audio

    def numeric(raw):
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0

    rows, masks = [], {}
    run.mkdir(mode=0o700)
    clips = run / "wav"
    clips.mkdir()
    for scene, manifest in BACKGROUNDS.items():
        path = sounds / f"{scene}.ogg" if scene != "traffic" else radar / "traffic.ogg"
        require(sha(path.read_bytes()) == manifest["sha256"], "Background recording changed")
        masks[scene] = numeric(audio.decode(path))

    def add(source, scene, start, signal, gain):
        raw = np.rint(signal * gain * 32768).astype("<i2").tobytes()
        path = clips / f"{len(rows):03}.wav"
        prior.wav(path, raw, RATE)
        _, commitment = prior.decode(path, RATE, sha(path.read_bytes()))
        rows.append({"id": len(rows), "source": source, "scene": scene, "background_start": start,
                     "wav_sha256": sha(path.read_bytes()), "analysis_pcm_sha256": commitment})

    strata = Counter()
    for row, clip in raw_sources(source, radar, llama):
        values = numeric(clip)
        rank = strata[row["publisher"], row["label"]]
        strata[row["publisher"], row["label"]] += 1
        views = {"clean": (None, values)}
        for scene, manifest in BACKGROUNDS.items():
            start = RATE * manifest["offsets"][rank % len(manifest["offsets"])]
            noise = masks[scene][start:start+SPAN]
            require(len(noise) == SPAN and np.linalg.norm(noise) > 0 and np.linalg.norm(values) > 0, "Missing noise or speech")
            views[scene] = (start, values + noise * (np.linalg.norm(values) / np.linalg.norm(noise)) * 10**.25)
        gain = min(1., .98 / max(float(np.max(np.abs(view))) for _, view in views.values()))
        for scene, (start, view) in views.items():
            add(row, scene, start, view, gain)
    for scene, manifest in BACKGROUNDS.items():
        for second in manifest["offsets"]:
            start = second * RATE
            add(None, scene, start, masks[scene][start:start+SPAN], .98)
    require(len({row["analysis_pcm_sha256"] for row in rows}) == len(rows), "Holdout PCM duplicates")
    save(run / "freeze.json", plan() | {"rows": rows})
    print(len(rows), "unscored, source-bound WAV files")


def plan():
    return {"schema": SCHEMA, "method": METHOD, "runner_sha256": sha(Path(__file__).read_bytes()),
            "archive_sha256": {publisher: pin for publisher, (_, _, pin) in PREFIXES.items()}, "backgrounds": BACKGROUNDS,
            "parents": {name: sha((PUBLIC / PANEL_FILE.format(name)).read_bytes()) for name in CALIBRATION}}


def voices(probes, original, enhanced, voice):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    from app.detectors.audio import calibration

    band = sosfiltfilt(butter(4, [100., 3800.], btype="bandpass", fs=RATE, output="sos"), original).astype(np.float32)
    samples = {"source_band": band, "enhanced_inverted": -enhanced}
    scores = voice.predict([samples[name] for name in probes])
    require(all(0 <= score <= 1 for score in scores) and len(scores) == len(probes), "Malformed probe response")
    return {key: float(calibration.logit(probability)) for key, probability in zip(probes, scores, strict=True)}


def calibrated(old, voice):
    import numpy as np

    rows = []
    for panel in CALIBRATION:
        folder = old / ("frozen" if panel == "transfer" else panel)
        panel_data = load(PUBLIC / PANEL_FILE.format(panel))["held"]
        for row in panel_data:
            current = row["candidates"]["new_enhancer"]
            if row["incumbent"] not in prior.ELIGIBLE or not current["admitted"] or current["logit"] <= POINT:
                continue
            name = f"{row['id']:03}"
            original = (folder / "pcm" / f"{name}.pcm").read_bytes()
            enhanced = (folder / "enhanced" / f"{name}-new_enhancer.pcm").read_bytes()
            require(sha(original) == row["pcm_sha256"] and sha(enhanced) == current["pcm_sha256"]
                    and len(original) == len(enhanced) == 2*SPAN, "Calibration PCM changed")
            waves = [np.frombuffer(raw, "<i2").astype(np.float32) / 32768 for raw in (original, enhanced)]
            rows.append({"panel": panel, "id": row["id"], "label": row["label"], "scene": row["scene"],
                         "pcm_sha256": sha(original), "analysis_pcm_sha256": sha(waves[0].astype("<f4").tobytes()),
                         "probes": voices(PROBES, *waves, voice)})
    return rows


def fit(rows):
    usable = [row["probes"] for row in rows if row["label"] == "spoof" and row["scene"] != "gain_control"]
    require(len(usable) == 19, "Calibration cohort drifted")
    return {probe: sorted(row[probe] for row in usable)[1] for probe in PROBES}


def measured(run, rows, voice, activity, enhancer):
    import numpy as np

    from app.detectors.audio import calibration
    from app.detectors.audio.content import window_in_speech
    from scripts.audio_radar_transfer import adapter_result

    for row in rows:
        path = run / "wav" / f"{row['id']:03}.wav"
        digest = sha(path.read_bytes())
        pcm, pcm_sha = prior.decode(path, RATE, digest)
        require((digest, pcm_sha) == (row["wav_sha256"], row["analysis_pcm_sha256"]), "Frozen WAV or PCM varies")
        primary = adapter_result(pcm, voice, activity)
        output = None
        enhanced_sha = None
        probes = None
        if primary["decision"] in prior.ELIGIBLE:
            result = np.asarray(enhancer.generate(pcm, sample_rate=RATE, chunk_seconds=None).audio, dtype=np.float32)
            require(result.shape == pcm.shape and np.isfinite(result).all(), "Enhanced return altered sample clock")
            raw = np.clip(np.rint(result * 32768), -32768, 32767).astype("<i2").tobytes()
            enhanced_sha = sha(raw)
            enhanced = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
            if window_in_speech(activity.spans(enhanced), 0., 3.):
                score, = voice.predict([enhanced])
                require(0 <= score <= 1, "Malformed enhanced response")
                output = float(calibration.logit(score))
                if output > POINT:
                    probes = voices(PROBES, pcm, enhanced, voice)
        yield {"id": row["id"], "primary": primary["decision"], "enhanced_pcm_sha256": enhanced_sha,
               "enhanced": output, "probes": probes}
        if (row["id"]+1) % 24 == 0:
            print(f"Scored {row['id']+1}/{len(rows)}", flush=True)


def review(frozen, case_measurements, points):
    counts = defaultdict(Counter)
    removed_backgrounds = {probe: set() for probe in PROBES}
    for case, row in zip(frozen, case_measurements, strict=True):
        require(case["id"] == row["id"] and row["primary"] in {"ai", "original", *prior.ELIGIBLE}, "Measured identity drifted")
        admitted = row["enhanced"] is not None
        require(not admitted or (math.isfinite(row["enhanced"]) and row["primary"] in prior.ELIGIBLE), "Invalid enhancement response")
        require((row["enhanced_pcm_sha256"] is not None) == (row["primary"] in prior.ELIGIBLE), "Incomplete enhancement path")
        incumbent = admitted and row["enhanced"] > POINT
        require(incumbent == (row["probes"] is not None), "Probe scope differs")
        label = case["source"]["label"] if case["source"] else "isolated"
        scene = case["scene"]
        counter = counts[scene, label]
        counter.update(files=1, primary_ai=int(row["primary"] == "ai"), primary_inconclusive=int(row["primary"] == "inconclusive"),
                       primary_unassessed=int(row["primary"] == "unassessed"), assessed=int(admitted), incumbent=int(incumbent))
        for probe in PROBES:
            value = row["probes"][probe] if incumbent else None
            require(value is None or math.isfinite(value), "Invalid probe response")
            keep = incumbent and value >= points[probe]
            counter[probe] += int(keep)
            if incumbent and not keep and scene != "clean" and label == "bonafide":
                removed_backgrounds[probe].add(scene)
    decisions = {}
    gate = METHOD["advance"]
    synth = sum(counts[scene, "spoof"]["incumbent"] for scene in BACKGROUNDS)
    for probe in PROBES:
        kept = sum(counts[scene, "spoof"][probe] for scene in BACKGROUNDS)
        removed = sum(counts[scene, "bonafide"]["incumbent"] - counts[scene, "bonafide"][probe] for scene in BACKGROUNDS)
        checks = {"enough_synthetic_flags": synth >= gate["minimum_noisy_incumbent_flags"],
                  "synthetic_retained": kept >= synth * gate["minimum_retained_synthetic_fraction"],
                  "retained_per_scene": all(counts[scene, "spoof"][probe] >= counts[scene, "spoof"]["incumbent"]
                                            * (1-gate["maximum_lost_in_any_scene_fraction"]) for scene in BACKGROUNDS),
                  "clean_synthetic_retained": counts["clean", "spoof"][probe] == counts["clean", "spoof"]["incumbent"],
                  "removed_genuine": removed >= gate["minimum_removed_genuine"],
                  "removed_across_scenes": len(removed_backgrounds[probe]) >= gate["minimum_removed_backgrounds"]}
        decisions[probe] = {"checks": checks, "passes": all(checks.values()), "kept_synthetic": kept,
                            "removed_genuine": removed, "total_incumbent_synthetic": synth}
    return {"counts": {f"{scene}/{label}": count for (scene, label), count in counts.items()}, "decisions": decisions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "verify-inputs", "fit", "score", "review"))
    parser.add_argument("run", type=Path)
    for name in ("source", "radar", "llama", "sounds", "old"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        record = load(args.run)
        points = fit(record["calibration"])
        require(record["points"] == points, "Operating points do not recount")
        require(record["result"] == review(record["holdout"]["rows"], record["measurements"], points), "Result does not recount")
        print(prior.previous.serialized(record["result"]).decode(), end="")
    elif args.phase in {"freeze", "verify-inputs"}:
        paths = (args.source, args.radar, args.llama, args.sounds)
        require(all(paths), "Supply archive prefixes, publisher labels and noise recordings")
        if args.phase == "freeze":
            freeze(args.run, *paths)
        else:
            with TemporaryDirectory() as temp:
                fresh = Path(temp) / "replayed"
                freeze(fresh, *paths)
                require(load(fresh / "freeze.json") == load(args.run / "freeze.json"), "Native input freeze did not reproduce")
    else:
        holdout = load(args.run / "freeze.json")
        require(all(holdout[key] == value for key, value in plan().items()), "Frozen procedure changed")
        point_path = args.run / "points.json"
        if args.phase == "fit":
            require(args.old and not point_path.exists() and not (args.run / "receipt.json").exists(), "Fit must precede inference")
            voice, _, _ = prior.restored_weights()
            previous_rows = calibrated(args.old, voice)
            require({row["analysis_pcm_sha256"] for row in previous_rows}.isdisjoint(row["analysis_pcm_sha256"] for row in holdout["rows"]),
                    "Calibration and held inputs overlap")
            save(point_path, {"calibration": previous_rows, "points": fit(previous_rows), "freeze_sha256": sha((args.run / "freeze.json").read_bytes())})
        else:
            points = load(point_path)
            require(points["freeze_sha256"] == sha((args.run / "freeze.json").read_bytes())
                    and points["points"] == fit(points["calibration"]), "Point selection differs")
            measurements = list(measured(args.run, holdout["rows"], *prior.restored_weights()))
            result = review(holdout["rows"], measurements, points["points"])
            save(args.run / "receipt.json", points | {"holdout": holdout, "measurements": measurements, "result": result})
            print(prior.previous.serialized(result["decisions"]).decode(), end="")


if __name__ == "__main__":
    main()
