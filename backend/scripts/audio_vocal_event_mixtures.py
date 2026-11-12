# SPDX-License-Identifier: Apache-2.0
"""Compare audio findings when the same speech receives a different vocal event."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import wave
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
PUBLIC = ROOT / "docs/evaluation-panels"
EVENT_RECEIPT = PUBLIC / "audio-vocal-event-source-roster-2026-11-12.json"
SPEECH_RECEIPT = PUBLIC / "audio-remote-neighbor-replication-2026-11-12.json"
SCHEMA = "chipmunk.audio-vocal-event-mixtures/1"
WEIGHTS = ROOT / "models/weights/audio-upgrade"
RATE, SAMPLES, EVENT_SAMPLES = 16000, 64000, 24000
CARRIERS = {"wav": 0, "mp3": 1105}
WINDOWS = ((0, 64000), (0, 32000), (16000, 48000), (32000, 64000))
POINT = .50
FAMILIES = ("clone", "expressive", "stock")
PAIRS = (("0323", "529818"), ("0715", "823548"), ("0376", "516473"),
         ("0323", "823548"), ("0715", "529818"), ("0376", "569568"))
EVENT_CLASSES = {"laughter": set(range(16, 22)), "breath": {26, *range(41, 47), 50}, "cough_sentinel": {47, 48}}
SPEECH_CLASSES = {*range(10), 15}
SOUND_ASSETS = {
    "panns-Cnn10.pth": "5240bdc47444e331bbeb54fd741d2b3f933a4526e84b0c17bf3593df94b13962",
    "class_labels_indices.csv": "cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429",
}
APP_CODE = ("app/detectors/audio/calibration.py", "app/detectors/audio/calibrations/local-speech-v3.json",
            "app/detectors/audio/decode.py", "app/detectors/audio/upgraded.py", "app/detectors/video/adapters/audio.py",
            "app/detectors/video/audio.py", "app/engine/profiles.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, data):
    path.write_bytes(encoded(data))


def encoded(data):
    return (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def commitment(data):
    return digest(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def decode(path, *, floating=False):
    format_ = "f32le" if floating else "s16le"
    return subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-protocol_whitelist", "file",
                           "-i", f"file:{path.resolve()}", "-map", "0:a:0", "-c:a", f"pcm_{format_}",
                           "-ar", str(RATE), "-ac", "1", "-f", format_, "pipe:1"],
                          capture_output=True, check=True, timeout=12).stdout


def rms(values):
    import numpy as np

    return float(np.sqrt(np.mean(values * values)))


def event_inputs(directory):
    import numpy as np

    record = read(EVENT_RECEIPT)
    events = {}
    for role in ("synthetic", "ordinary"):
        for item in record[role]:
            if role == "synthetic":
                name = item["key"]
                path = directory / "moss" / f"{name}.wav"
                pin = item["members"]["wav"]["sha256"]
            else:
                name = item["item_id"]
                path = directory / "ordinary" / f"{name}.mp3"
                pin = item["preview"]["sha256"]
            require(sha(path) == pin, "The exact publisher recording differs")
            samples = np.frombuffer(decode(path, floating=True), dtype="<f4").astype(np.float64)
            require(len(samples) >= EVENT_SAMPLES and np.isfinite(samples).all(), "Insufficient or invalid event audio")
            starts = range(0, len(samples) - EVENT_SAMPLES + 1, 1600)
            begin = max(starts, key=lambda start: rms(samples[start:start + EVENT_SAMPLES]))
            raw = samples[begin:begin + EVENT_SAMPLES].copy()
            fade = np.linspace(0., 1., 480)
            raw[:len(fade)] *= fade
            raw[-len(fade):] *= fade[::-1]
            raw_rms = rms(raw)
            require(raw_rms > .0005, "Event selection is too quiet")
            normalized = raw / raw_rms
            events[name] = (normalized, {
                "source": name, "role": role, "family": item["family"], "file_sha256": pin,
                "source_samples": [begin, begin + EVENT_SAMPLES], "sample_rate": RATE,
                "source_rms_before_normalization": raw_rms, "source_peak_before_normalization": float(np.max(np.abs(raw))),
                "normalized_faded_sha256": digest(normalized.astype("<f8").tobytes()),
            })
    return events


def speech_inputs(directory):
    import numpy as np

    roster = read(directory / "roster.json")
    published = read(SPEECH_RECEIPT)
    require(sha(directory / "roster.json") == published["commitments"]["roster"], "The earlier speech roster differs")
    by_id = {(row["group"], row["arm"]): row for row in roster}
    eligible = [sorted((row for row in roster if row["arm"] == "synthetic" and row["family"] == family
                        and row["truth"][1] - row["truth"][0] >= 12800), key=lambda row: row["id"])[:2]
                for family in FAMILIES]
    selection = [row for round_ in zip(*eligible, strict=True) for row in round_]
    require(len(selection) == 6 and len({row["host"]["canonical_speaker"] for row in selection}) == 6,
            "Speech donors are not six distinct host groups")
    for index, row in enumerate(selection):
        lower, upper = row["truth"]
        start = max(0, min(96000 - SAMPLES, (lower + upper - SAMPLES) // 2))
        require(start <= lower < upper <= start + SAMPLES, "The test crop excludes an edited sample")
        data = {}
        for role, arm in (("genuine", "untouched"), ("partial_ai", "synthetic")):
            item = by_id[(row["group"], arm)]
            raw = (directory / "pcm" / f"{item['id']}-foreground.pcm").read_bytes()
            require(len(raw) == 192000 and digest(raw) == item["foreground_sha256"], "The retained speech stem differs")
            values = np.frombuffer(raw, dtype="<i2")[start:start + SAMPLES].astype(np.float64) / 32768
            data[role] = (values, {"source": item["id"], "sha256": item["foreground_sha256"], "sample_start": start,
                                   "truth": [lower - start, upper - start] if role == "partial_ai" else None})
        yield f"voice-{index + 1:02d}", row["family"], data


def variants(events, speech):
    import numpy as np

    for index, (group, generator, voices) in enumerate(speech):
        start = (16000, 24000)[index % 2]
        end = start + EVENT_SAMPLES
        plans = []
        for voice_role, (voice, witness) in voices.items():
            plans.append((voice_role, witness, voice, None, None, np.zeros(SAMPLES)))
            for donor in PAIRS[index]:
                event, label = events[donor]
                voice_rms = rms(voice[start:end])
                require(voice_rms > .002, "Insufficient speech stem energy in the overlay range")
                for snr in (-5, 5):
                    added = np.zeros(SAMPLES)
                    added[start:end] = event * voice_rms * 10 ** (-snr / 20)
                    plans.append((voice_role, witness, voice, label, snr, added))
        peak = max(float(np.max(np.abs(source + added))) for _, _, source, _, _, added in plans)
        gain = min(.80 / peak, .08 / rms(voices["genuine"][0]))
        for voice_role, witness, source, label, snr, added in plans:
            yield ({"group": group, "voice_role": voice_role, "voice_generator": generator, "speech_source": witness,
                    "event": label, "event_span": [start, end] if label else None, "speech_to_event_db": snr, "gain": gain},
                   source * gain, added * gain)
    for index, (event, witness) in enumerate(events.values()):
        start = (16000, 24000)[index % 2]
        end = start + EVENT_SAMPLES
        added = np.zeros(SAMPLES)
        gain = min(.8 / float(np.max(np.abs(event))), .08)
        added[start:end] = event * gain
        yield ({"group": f"event-{index + 1:02d}", "voice_role": "no_foreground", "voice_generator": None,
                "speech_source": None, "event": witness, "event_span": [start, end], "speech_to_event_db": None, "gain": gain},
               np.zeros(SAMPLES), added)


def render(output, sources):
    import numpy as np

    rows = []
    for index, (record, voice, event) in enumerate(sources):
        key = f"case-{index + 1:03d}"
        row = {"id": key, **record, "carriers": {}}
        voice_i = np.rint(voice * 32768).astype("<i2")
        event_i = np.rint(event * 32768).astype("<i2")
        mixed = voice_i.astype(np.int32) + event_i.astype(np.int32)
        require(np.max(np.abs(mixed)) < 32767, "An integer mixture clipped")
        row["components"] = {"speech": digest(voice_i.tobytes()), "event": digest(event_i.tobytes())}
        path = output / f"{key}.wav"
        with wave.open(str(path), "wb") as stream:
            stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            stream.writeframes(mixed.astype("<i2").tobytes())
        other = output / f"{key}.mp3"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file",
                        "-i", f"file:{path.resolve()}", "-map_metadata", "-1", "-c:a", "libmp3lame", "-b:a", "48k",
                        str(other)], check=True, timeout=12)
        for source in (path, other):
            pcm = decode(source)
            require(len(pcm) == 2 * SAMPLES, "A carrier differs from the planned decoded length")
            if source == path:
                require(pcm == mixed.astype("<i2").tobytes(), "Lossless delivery differs from the exact integer mix")
            row["carriers"][source.suffix[1:]] = {"bytes": sha(source), "pcm": digest(pcm)}
        if row["speech_to_event_db"] is not None:
            a, b = row["event_span"]
            ratio = 20 * math.log10(rms(voice_i[a:b].astype(float)) / rms(event_i[a:b].astype(float)))
            require(abs(ratio - row["speech_to_event_db"]) <= .03, "A mixture deviated from its relative level")
            row["rendered_ratio_db"] = ratio
        rows.append(row)
    return rows


def assets():
    from scripts.prepare_audio_specialists_v15 import ASSETS

    return ASSETS | SOUND_ASSETS


def prepare(output, event_dir, speech_dir):
    output.mkdir(mode=0o700)
    rows = render(output, variants(event_inputs(event_dir), speech_inputs(speech_dir)))
    require(len(rows) == 69, "The fixed pilot requires 60 voice views and nine isolated events")
    save(output / "inputs.json", rows)
    lock = {"schema": SCHEMA, "runner_sha256": sha(Path(__file__)), "inputs_sha256": sha(output / "inputs.json"),
            "event_receipt_sha256": sha(EVENT_RECEIPT), "speech_receipt_sha256": sha(SPEECH_RECEIPT),
            "assets": assets(), "cutoff": POINT, "windows": WINDOWS, "foreground_groups": 6, "event_sources": 9,
            "app_code": {path: sha(ROOT / "backend" / path) for path in APP_CODE},
            "arms": ["four-second sound tags", "three two-second sound tags", "current voice decision",
                     "voice decision with non-speech-dominant review"],
            "purpose": "Small held mechanism test for crossed component origin; no weight fitting or population claim."}
    save(output / "freeze.json", lock)
    print(f"Froze {len(rows)} cases and {2 * len(rows)} exact WAV/MP3 carriers.", flush=True)


def voice_result(pcm, voice, vad):
    from types import SimpleNamespace

    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from app.engine.detector import Profile

    lease = {PellaVoiceDetector.MODEL_ID: SimpleNamespace(obj=voice, revision="pinned"),
             "silero.speech_activity": SimpleNamespace(obj=vad, revision="pinned")}
    ctx = SimpleNamespace(profile=Profile.FULL, outputs=SimpleNamespace(audio=AudioTrack("analysed", pcm, samples=len(pcm))),
                          lease=SimpleNamespace(get=lease.__getitem__), exhibits=SimpleNamespace(series=lambda *a, **k: "offline"),
                          deadline=SimpleNamespace(check=lambda *a: None))
    result = PellaVoiceDetector().run(ctx)
    if result.status == "not_applicable":
        return {"decision": "unassessed", "raw": None, "reference_estimate": None}
    screen = result.details["audio_screening"]
    require(result.status == "ok" and screen["scope_supported"] and len(result.unit_scores) == 1,
            "The production voice adapter used an unexpected input scope")
    return {"decision": screen["decision"], "raw": result.aggregate.raw, "reference_estimate": screen["probability_ai"]}


def best_event(tags):
    candidates = [(i, max((label for label in labels if any(label[0] in values for values in EVENT_CLASSES.values())),
                          key=lambda label: label[1], default=(None, -1))) for i, labels in enumerate(tags)]
    index, (class_id, value) = max(candidates, key=lambda row: (row[1][1], -row[0]))
    if value < POINT:
        return None
    return index, next(name for name, classes in EVENT_CLASSES.items() if class_id in classes)


def located(selection, windows, expected, truth):
    if selection is None or truth is None or selection[1] != expected:
        return False
    start, end = windows[selection[0]]
    overlap = max(0, min(end, truth[1]) - max(start, truth[0]))
    return 2 * overlap >= (end - start + truth[1] - truth[0] - overlap)


def summarize(rows, cases):
    require(len(rows) == 2 * len(cases) and len(cases) == 69, "Missing measured pilot cases")
    metrics, speech = {}, {}
    for carrier in CARRIERS:
        delivered = [row for row in rows if row["carrier"] == carrier]
        require([row["id"] for row in delivered] == [case["id"] for case in cases], "Carrier roster differs")
        groups, voices = {}, {}
        clean = {(case["group"], case["voice_role"]): row["voice"]["decision"]
                 for case, row in zip(cases, delivered, strict=True) if case["event"] is None}
        for case, row in zip(cases, delivered, strict=True):
            event, span, voice_role = case["event"], case["event_span"], case["voice_role"]
            role = "speech_only" if event is None else "isolated" if voice_role == "no_foreground" else str(case["speech_to_event_db"])
            expected = event["family"] if event else None
            tags = row["sound"]
            require(row["pcm_sha256"] == case["carriers"][carrier]["pcm"] and len(tags) == 4
                    and all(len(labels) == 3 and len({i for i, _ in labels}) == 3
                            and all(type(i) is int and 0 <= i < 527 and math.isfinite(p) and 0 <= p <= 1 for i, p in labels)
                            and sorted((p for _, p in labels), reverse=True) == [p for _, p in labels] for labels in tags),
                    "Measurement source or top-category output differs")
            broad, focused = best_event(tags[:1]), best_event(tags[1:])
            counter = groups.setdefault(role, Counter())
            counter["cases"] += 1
            for key, picked, windows in (("broad", broad, WINDOWS[:1]), ("focused", focused, WINDOWS[1:])):
                counter[f"{key}_flag"] += picked is not None
                counter[f"{key}_correct_category"] += picked is not None and picked[1] == expected
                counter[f"{key}_useful_location"] += located(picked, windows, expected, span)
            outcome = row["voice"]["decision"]
            gate = (broad is not None and tags[0][0][0] in EVENT_CLASSES[broad[1]]
                    and not any(index in SPEECH_CLASSES for index, _ in tags[0]))
            after = "inconclusive" if gate and outcome in {"ai", "original"} else outcome
            label = f"{voice_role}/{event['role'] if event else 'none'}/{role}"
            voice_counter = voices.setdefault(label, Counter())
            voice_counter[f"current_{outcome}"] += 1
            voice_counter[f"proposed_{after}"] += 1
            voice_counter["would_withhold"] += after != outcome
            if event is not None and voice_role != "no_foreground":
                baseline = clean[(case["group"], voice_role)]
                voice_counter["changes_from_clean"] += outcome != baseline
                voice_counter["new_ai_from_clean"] += outcome == "ai" and baseline != "ai"
        metrics[carrier], speech[carrier] = groups, voices
    return {"event_context": metrics, "voice": speech}


def score(output, weights):
    from app.core.config import settings
    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.upgraded import SoundEventsModel, SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    lock, cases = read(output / "freeze.json"), read(output / "inputs.json")
    require(settings.audio_detector_mode == "candidate" and lock["runner_sha256"] == sha(Path(__file__))
            and lock["inputs_sha256"] == sha(output / "inputs.json") and lock["assets"] == assets()
            and all(sha(ROOT / "backend" / name) == pin for name, pin in lock["app_code"].items())
            and all(sha(weights / name) == pin for name, pin in lock["assets"].items()), "Model, settings or study freeze differs")
    voice = load_voice({"weights": str(weights), "assets": {k: v for k, v in lock["assets"].items() if k not in SOUND_ASSETS}})
    vad = SpeechActivity(weights / "silero_vad.jit")
    sound = SoundEventsModel(weights / "panns-Cnn10.pth", weights / "class_labels_indices.csv", "cpu")
    rows = []
    for carrier, origin in CARRIERS.items():
        for index, case in enumerate(cases, 1):
            path = output / f"{case['id']}.{carrier}"
            pin = case["carriers"][carrier]
            require(sha(path) == pin["bytes"], "Scored audio bytes differ")
            decoded = decode_audio(path, source_sha256=pin["bytes"], max_seconds=5)
            pcm = decoded.pcm
            require(decoded.mapping["timeline_time_base"] == "1/16000" and decoded.mapping["timeline"] ==
                    [{"start_sample": 0, "end_sample": SAMPLES, "source_pts": origin}]
                    and digest((pcm * 32768).astype("<i2").tobytes()) == pin["pcm"], "Decoded source binding differs")
            tags = [[(item["class_index"], item["score"]) for item in sound.predict(pcm[a:b])] for a, b in WINDOWS]
            rows.append({"id": case["id"], "carrier": carrier, "pcm_sha256": pin["pcm"], "sound": tags,
                         "voice": voice_result(pcm, voice, vad)})
            if index % 12 == 0:
                print(f"{carrier}: measured {index}/69 source-bound files", flush=True)
    receipt = {"freeze": lock, "cases": cases, "measurements": rows,
               "cases_sha256": commitment(cases), "measurements_sha256": commitment(rows),
               "summary": summarize(rows, cases)}
    save(output / "result.json", receipt)


def review(path):
    result = read(path)
    lock, cases, measurements = (result[key] for key in ("freeze", "cases", "measurements"))
    require(lock["schema"] == SCHEMA and lock["runner_sha256"] == sha(Path(__file__)) and lock["cutoff"] == POINT
            and lock["windows"] == [list(window) for window in WINDOWS]
            and lock["event_receipt_sha256"] == sha(EVENT_RECEIPT) and lock["speech_receipt_sha256"] == sha(SPEECH_RECEIPT)
            and lock["inputs_sha256"] == digest(encoded(cases))
            and result["cases_sha256"] == commitment(cases) and result["measurements_sha256"] == commitment(measurements),
            "The case, response or prior source commitment differs")
    require(summarize(measurements, cases) == result["summary"], "Published counts differ from paired replay")
    print(f"Verified {len(cases)} case recipes and {len(measurements)} measured carriers.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score", "review"))
    parser.add_argument("target", type=Path)
    parser.add_argument("--events", type=Path, default=BASE / "audio-moss-vocal-event-sources-2026-11-12")
    parser.add_argument("--speech", type=Path, default=BASE / "audio-remote-neighbor-replication-2026-11-12")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.target, args.events, args.speech)
    elif args.phase == "score":
        score(args.target, args.weights)
    else:
        review(args.target)


if __name__ == "__main__":
    main()
