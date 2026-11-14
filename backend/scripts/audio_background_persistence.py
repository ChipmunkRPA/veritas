# SPDX-License-Identifier: Apache-2.0
"""Compare temporal persistence and voice-band agreement after background reduction."""
from __future__ import annotations

import argparse
import io
import math
import tarfile
import wave
from collections import Counter, defaultdict
from decimal import Decimal
from hashlib import file_digest
from pathlib import Path
from zipfile import ZipFile

from scripts import audio_background_confirmation as predecessor

prior = predecessor.prior
require, sha, load, save = prior.require, prior.sha, prior.load, prior.save
RATE, SPAN, POINT = predecessor.RATE, predecessor.SPAN, predecessor.POINT
PANEL = predecessor.PUBLIC / "audio-background-safeguard-2026-11-13.json"
SCHEMA = "chipmunk.audio-background-persistence/1"
REPLAY_LOGIT_TOLERANCE = .001
CROPS = ((0, 24_000), (12_000, 36_000), (24_000, 48_000))
TAPES = {"street": ("STRAFFIC", "2efa87262f272bbf9ba578088e81939c"),
         "washroom": ("DWASHING", "7e5ee9437ce9409c5f9a779b6212a240")}
ARMS = ("persistence", "disjoint_edges", "processed_band")
METHOD = {
    "input": "Exact prior source-bound 16 kHz WAVs for development. Holdout is 18 additional English RADAR files per class with one whole-file label, 3–14 seconds, fixed hash order, and no known file or PCM overlap. Save center three seconds, matched gain-only and separate -5 dB RMS mixtures from DEMAND channel 1. Keep four isolated excerpts from each tape.",
    "development_replay": "Require the same primary assessment, speech admission, and enhanced whole-window flag as in the historical run. The whole-window logit must be within 0.001 of the prior measurement. Retain each run's enhanced PCM digest to expose any numeric variation.",
    "base_point": POINT,
    "persistence": "Median of at least two estimated scores among the three 1.5-s enhanced subwindows. When exactly two are admitted use the smaller reading. The 1.5-s scores are ordinal and receive a separate empirical point.",
    "disjoint_edges": "Minimum of the two non-overlapping 1.5-s portions (left/right). Both portions must pass speech admission. No sample overlap, but the same source and classifier are shared.",
    "processed_band": "One response on enhanced PCM after a 250–3400 Hz order-4 zero-phase band-pass. Speech admission uses the full unfiltered enhanced view, as for the incumbent.",
    "only_if": "Probe only supplemental positive responses at the unchanged point. No new flag is allowed and no primary result may change. Audition and score the processed waveform using the existing local assets.",
    "development": "No more than four of the 24 synthetic noisy additions may be lost. For each arm select the highest point retaining 20 additions, given admitted scores. When fewer than 20 are admitted the arm is unqualified. A qualified point must remove at least two genuine development flags. All feasible arms are measured descriptively on the holdout without refitting, including arms that did not qualify for adoption on development. Clean behavior is measured separately and used in confirmation admission.",
    "advance": "Retain at least 85% of the noisy synthetic additions overall and 75% per background. Retain all clean synthetic additions. Require at least six available noisy synthetic additions, at least two fewer genuine noisy flags and at least one removal in each tape. These are evaluation conditions, not public thresholds.",
}


def pin():
    return {"schema": SCHEMA, "method": METHOD, "runner_sha256": sha(Path(__file__).read_bytes()),
            "calibration_sha256": sha(PANEL.read_bytes()), "noise": {k: list(value) for k, value in TAPES.items()}}


def saved_flags(panel):
    return {row["id"]: row for row in panel["measurements"] if row["probes"] is not None}


def verify_development(panel, rows):
    flags = saved_flags(panel)
    require(len(rows) == len(flags) and {row["id"] for row in rows} == flags.keys(),
            "Development response identities differ")
    for row in rows:
        baseline = flags[row["id"]]
        require(row["primary"] == baseline["primary"] and row["enhanced_sha256"] is not None
                and row["probes"] is not None and row["enhanced"] > POINT
                and abs(row["enhanced"] - baseline["enhanced"]) <= REPLAY_LOGIT_TOLERANCE,
                "Previously flagged assessment did not reproduce")
    return {
        "rows": len(rows),
        "enhanced_exact_pcm_matches": sum(row["enhanced_sha256"] == flags[row["id"]]["enhanced_pcm_sha256"] for row in rows),
        "max_logit_drift": max(abs(row["enhanced"] - flags[row["id"]]["enhanced"]) for row in rows),
    }


def known():
    dev = load(PANEL)["holdout"]["rows"]
    records = [case["source"] for case in dev if case["source"]]
    return ({row["id"] for row in records if row["publisher"] == "radar"} | predecessor.known()["radar"],
            {row["source_pcm_sha256"] for row in records}, {row["clip_sha256"] for row in records},
            {case["analysis_pcm_sha256"] for case in dev})


def sources(path, radar):
    from scripts import audio_radar_transfer as radar_lib

    _, size, expected = predecessor.PREFIXES["radar"]
    require(path.stat().st_size == size and sha(path.read_bytes()) == expected, "Publisher prefix changed")
    metadata = (radar / "labels.txt").read_bytes()
    language = (radar / "languages.txt").read_bytes()
    require(sha(metadata) == radar_lib.SOURCE_FILES["labels.txt"] and sha(language) == radar_lib.SOURCE_FILES["languages.txt"],
            "Publisher labels changed")
    labels = {name: values for name, *values in map(str.split, metadata.decode().splitlines())}
    languages = dict(line.split() for line in language.decode().splitlines())
    excluded, pcm_excluded, clip_excluded, _ = known()
    candidates = defaultdict(list)
    with tarfile.open(path, "r|gz") as archive:
        try:
            for member in archive:
                match = radar_lib.MEMBER.fullmatch(member.name)
                if not member.isfile() or not match or match[1] in excluded or languages.get(match[1]) != "EN":
                    continue
                name = match[1]
                duration, label, *segments = labels[name]
                if not 3 <= Decimal(duration) <= 14 or segments != [f"0.0000-{duration}-{label}"]:
                    continue
                with archive.extractfile(member) as entry:
                    data = entry.read()
                require(len(data) == member.size and data[:4] == b"fLaC" and label in {"bonafide", "spoof"}, "Invalid audio member")
                candidates[label].append((name, duration, data))
        except (EOFError, tarfile.ReadError) as error:
            require(isinstance(error, EOFError) or "end of data" in str(error), "Malformed prefix")
    for label, choices in sorted(candidates.items()):
        used = 0
        for name, duration, data in sorted(choices, key=lambda row: sha(f"{SCHEMA}/{row[0]}".encode())):
            raw = prior.previous.ffmpeg("-f", "flac", "-i", "pipe:0", "-ac", "1", "-ar", str(RATE), "-c:a",
                                        "pcm_s16le", "-f", "s16le", "pipe:1", pcm=data)
            length = len(raw) // 2
            require(length >= SPAN and abs(round(Decimal(duration) * RATE) - length) <= 16, "Publisher extent differs")
            start = (length - SPAN) // 2
            clip = raw[2*start:2*(start+SPAN)]
            if sha(raw) in pcm_excluded or sha(clip) in clip_excluded:
                continue
            pcm_excluded.add(sha(raw))
            clip_excluded.add(sha(clip))
            yield {"id": name, "label": label, "source_sha256": sha(data), "source_pcm_sha256": sha(raw),
                   "start": start, "clip_sha256": sha(clip), "seconds": duration}, clip, used
            used += 1
            if used == 18:
                break
        require(used == 18, "Insufficient unseen whole-label files")


def freeze(run, archive, radar, background):
    import numpy as np

    def view(data):
        return np.frombuffer(data, "<i2").astype(np.float64) / 32768

    masks = {}
    for scene, (stem, digest) in TAPES.items():
        path = background[scene]
        with path.open("rb") as source:
            require(file_digest(source, "md5").hexdigest() == digest, "Official DEMAND archive differs")
        with ZipFile(path) as source:
            raw = source.read(f"{stem}/ch01.wav")
        with wave.open(io.BytesIO(raw)) as stream:
            require((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) == (1, 2, RATE, "NONE"),
                    "DEMAND WAV format differs")
            masks[scene] = view(stream.readframes(stream.getnframes()))
    rows = []
    run.mkdir(mode=0o700)
    folder = run / "wav"
    folder.mkdir()

    def store(scene, source, start, signal):
        path = folder / f"{len(rows):03}.wav"
        prior.wav(path, np.rint(signal * 32768).astype("<i2").tobytes(), RATE)
        media = sha(path.read_bytes())
        _, pcm = prior.decode(path, RATE, media)
        rows.append({"id": len(rows), "scene": scene, "source": source, "noise_start": start, "wav_sha256": media,
                     "analysis_pcm_sha256": pcm})

    for source, clip, rank in sources(archive, radar):
        voice = view(clip)
        cases = {"clean": (None, voice)}
        for scene, mask in masks.items():
            start = (12 + 13*rank) * RATE
            excerpt = mask[start:start+SPAN]
            require(len(excerpt) == SPAN and np.linalg.norm(excerpt) > 0 and np.linalg.norm(voice) > 0, "Invalid input extent or level")
            cases[scene] = (start, voice + excerpt * np.linalg.norm(voice) / np.linalg.norm(excerpt) * 10**.25)
        gain = min(1., .98 / max(float(np.max(abs(signal))) for _, signal in cases.values()))
        for scene, (start, signal) in cases.items():
            store(scene, source, start, signal * gain)
    for scene, mask in masks.items():
        for second in (12, 77, 142, 207):
            start = second * RATE
            store(scene, None, start, mask[start:start + SPAN] * .98)
    require(len(rows) == 116 == len({row["analysis_pcm_sha256"] for row in rows})
            and known()[-1].isdisjoint(row["analysis_pcm_sha256"] for row in rows), "Source or fixture overlap")
    save(run / "freeze.json", pin() | {"rows": rows})
    print("Froze 116 distinct WAV files before inference", flush=True)


def measurements(waveform, spans, voice):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    from app.detectors.audio import calibration
    from app.detectors.audio.content import window_in_speech

    starts = [i for i, (a, b) in enumerate(CROPS) if window_in_speech(spans, a / RATE, b / RATE)]
    selected = [waveform[CROPS[i][0]:CROPS[i][1]] for i in starts]
    filtered = sosfiltfilt(butter(4, [250., 3400.], btype="bandpass", fs=RATE, output="sos"), waveform).astype(np.float32)
    probabilities = voice.predict([filtered, *selected])
    require(len(probabilities) == len(starts) + 1 and all(0 <= value <= 1 for value in probabilities),
            "Invalid probability from voice classifier")
    scores = [float(calibration.logit(value)) for value in probabilities]
    return {"crops": [scores[starts.index(i) + 1] if i in starts else None for i in range(len(CROPS))], "band": scores[0]}


def values(record):
    if record is None:
        return dict.fromkeys(ARMS)
    a, b, c = record["crops"]
    admitted = sorted(v for v in (a, b, c) if v is not None)
    return {"persistence": admitted[-2] if len(admitted) >= 2 else None,
            "disjoint_edges": min(a, c) if a is not None and c is not None else None, "processed_band": record["band"]}


def score(folder, cases, voice, activity, enhancer, *, known_flags=None):
    import numpy as np

    from app.detectors.audio import calibration
    from app.detectors.audio.content import window_in_speech
    from scripts.audio_radar_transfer import adapter_result

    for index, case in enumerate(cases, 1):
        if known_flags is not None and case["id"] not in known_flags:
            continue
        path = folder / "wav" / f"{case['id']:03}.wav"
        actual = sha(path.read_bytes())
        raw, pcm = prior.decode(path, RATE, actual)
        require((actual, pcm) == (case["wav_sha256"], case["analysis_pcm_sha256"]), "Unsealed delivery")
        primary = adapter_result(raw, voice, activity)["decision"]
        enhanced, probes, processed_sha = None, None, None
        if primary in prior.ELIGIBLE:
            values_ = np.asarray(enhancer.generate(raw, sample_rate=RATE, chunk_seconds=None).audio, dtype=np.float32)
            require(values_.shape == raw.shape and np.isfinite(values_).all(), "Enhanced waveform instead changed clock")
            processed = np.clip(np.rint(values_ * 32768), -32768, 32767).astype("<i2").tobytes()
            waveform = np.frombuffer(processed, "<i2").astype(np.float32) / 32768
            processed_sha = sha(processed)
            spans = activity.spans(waveform)
            if window_in_speech(spans, 0., 3.):
                probability, = voice.predict([waveform])
                require(0 <= probability <= 1, "Invalid probability from voice classifier")
                enhanced = float(calibration.logit(probability))
                if enhanced > POINT:
                    probes = measurements(waveform, spans, voice)
        yield {"id": case["id"], "primary": primary, "enhanced": enhanced, "enhanced_sha256": processed_sha, "probes": probes}
        if known_flags is not None or index % 20 == 0:
            print(f"Measured source row {case['id']} ({index}/{len(cases)})", flush=True)


def operating_points(cases, records):
    indexed = {case["id"]: case for case in cases}
    synth, genuine = [], []
    for row in records:
        case = indexed[row["id"]]
        if case["scene"] == "clean":
            continue
        (synth if case["source"]["label"] == "spoof" else genuine).append(values(row["probes"]))
    require(len(synth) == 24 and len(genuine) == 11, "Prior noisy baseline labels differed")
    points = {}
    for arm in ARMS:
        seq = sorted((row[arm] for row in synth if row[arm] is not None), reverse=True)
        point = seq[19] if len(seq) >= 20 else None
        removed = sum(point is None or row[arm] is None or row[arm] < point for row in genuine)
        points[arm] = {"admitted_synthetic": len(seq), "point": point, "removed_genuine": removed if point is not None else None,
                       "qualified": point is not None and removed >= 2}
    return points


def summary(cases, results, points):
    counts = defaultdict(Counter)
    for case, row in zip(cases, results, strict=True):
        require(case["id"] == row["id"] and row["primary"] in {"original", "ai", *prior.ELIGIBLE}, "Invalid inference identity")
        admitted = row["enhanced"] is not None
        require(not admitted or (math.isfinite(row["enhanced"]) and row["primary"] in prior.ELIGIBLE), "Invalid enhanced response")
        incumbent = admitted and row["enhanced"] > POINT
        require((row["probes"] is not None) == incumbent and (row["enhanced_sha256"] is not None) == (row["primary"] in prior.ELIGIBLE),
                "Unexpected analysis branch")
        label = case["source"]["label"] if case["source"] else "isolated"
        count = counts[case["scene"], label]
        count.update(files=1, primary_ai=int(row["primary"] == "ai"), primary_inconclusive=int(row["primary"] == "inconclusive"),
                     primary_unassessed=int(row["primary"] == "unassessed"), incumbent=int(incumbent), admitted=int(admitted))
        for arm, value in values(row["probes"]).items():
            point = points[arm]["point"]
            require(value is None or math.isfinite(value), "Invalid probe response")
            count[arm] += int(value is not None and point is not None and value >= point)
    total_synth = sum(counts[scene, "spoof"]["incumbent"] for scene in TAPES)
    decisions = {}
    for arm in ARMS:
        kept = sum(counts[scene, "spoof"][arm] for scene in TAPES)
        removals = {scene: counts[scene, "bonafide"]["incumbent"] - counts[scene, "bonafide"][arm] for scene in TAPES}
        checks = {"qualified_in_development": points[arm]["qualified"], "six_synthetic_flags": total_synth >= 6,
                  "synthetic_retained": kept >= total_synth * .85,
                  "each_scene_retained": all(counts[scene, "spoof"][arm] >= counts[scene, "spoof"]["incumbent"] * .75 for scene in TAPES),
                  "clean_retained": counts["clean", "spoof"][arm] == counts["clean", "spoof"]["incumbent"],
                  "two_removals": sum(removals.values()) >= 2, "both_scenes_helped": all(removals.values())}
        decisions[arm] = {"checks": checks, "passes": all(checks.values()), "retained_noisy_synthetic": [kept, total_synth], "removals": removals}
    return {"counts": {f"{key[0]}/{key[1]}": row for key, row in counts.items()}, "decisions": decisions}


def show(value):
    print(prior.previous.serialized(value).decode(), end="")


def audited_rows(frozen):
    require(all(frozen[key] == value for key, value in pin().items()), "Frozen implementation changed")
    rows = frozen["rows"]
    require([row["id"] for row in rows] == list(range(116)) and len({row["wav_sha256"] for row in rows}) == 116
            and len({row["analysis_pcm_sha256"] for row in rows}) == 116, "Missing or duplicated frozen inputs")
    exclusions = known()
    rosters = defaultdict(list)
    isolated = defaultdict(list)
    for row in rows:
        require(row["analysis_pcm_sha256"] not in exclusions[-1] and row["scene"] in {"clean", *TAPES}, "Unexpected frozen input")
        source = row["source"]
        if source is None:
            isolated[row["scene"]].append(row["noise_start"])
        else:
            require(source["id"] not in exclusions[0] and source["source_pcm_sha256"] not in exclusions[1]
                    and source["clip_sha256"] not in exclusions[2], "Development and held source overlap")
            rosters[source["id"]].append(row)
    require(dict(isolated) == {scene: [second * RATE for second in (12, 77, 142, 207)] for scene in TAPES},
            "Isolated controls differ")
    sources = []
    ranks = Counter()
    for matched in rosters.values():
        first, *mixed = matched
        require([row["scene"] for row in matched] == ["clean", *TAPES] and first["noise_start"] is None
                and all(row["source"] == first["source"] for row in matched), "Matched source variants differ")
        label = first["source"]["label"]
        require(all(row["noise_start"] == (12 + 13 * ranks[label]) * RATE for row in mixed), "Noise time offsets differ")
        ranks[label] += 1
        sources.append(first["source"])
    require(Counter(source["label"] for source in sources) == {"bonafide": 18, "spoof": 18}
            and all(len({source[key] for source in sources}) == 36 for key in ("source_pcm_sha256", "clip_sha256")),
            "Held source labels or identity differ")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "score", "review"))
    parser.add_argument("run", type=Path)
    for flag in ("archive", "radar", "street", "washroom", "development"):
        parser.add_argument(f"--{flag}", type=Path)
    args = parser.parse_args()
    panel = load(PANEL)
    cases = panel["holdout"]["rows"]
    if args.phase == "freeze":
        require(all((args.archive, args.radar, args.street, args.washroom)), "Supply sources")
        freeze(args.run, args.archive, args.radar, {"street": args.street, "washroom": args.washroom})
    elif args.phase == "review":
        saved = load(args.run)
        require(saved["development_replay"] == verify_development(panel, saved["development"]),
                "Development replay summary differs")
        require(saved["holdout_sha256"] == sha(prior.previous.serialized(saved["holdout"])), "Holdout commitment differs")
        held = audited_rows(saved["holdout"])
        points = operating_points(cases, saved["development"])
        require(saved["points"] == points and saved["result"] == summary(held, saved["holdout_results"], points),
                "Measurements do not recount")
        show(saved["result"])
    else:
        frozen = load(args.run / "freeze.json")
        held = audited_rows(frozen)
        op = args.run / "selection.json"
        if args.phase == "fit":
            require(args.development and not op.exists(), "Only one development pass for the held run")
            flags = saved_flags(panel)
            rows = list(score(args.development, cases, *prior.restored_weights(), known_flags=flags))
            replay = verify_development(panel, rows)
            points = operating_points(cases, rows)
            save(op, {"holdout_sha256": sha((args.run / "freeze.json").read_bytes()), "development": rows,
                      "development_replay": replay, "points": points})
            show(points)
        else:
            fitted = load(op)
            require(fitted["holdout_sha256"] == sha((args.run / "freeze.json").read_bytes())
                    and fitted["points"] == operating_points(cases, fitted["development"])
                    and fitted["development_replay"] == verify_development(panel, fitted["development"]),
                    "Development selection changed")
            rows = list(score(args.run, held, *prior.restored_weights()))
            result = summary(held, rows, fitted["points"])
            save(args.run / "receipt.json", fitted | {"holdout": frozen, "holdout_results": rows, "result": result})
            show(result["decisions"])


if __name__ == "__main__":
    main()
