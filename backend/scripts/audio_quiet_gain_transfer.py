"""Test the fixed speech gain retry on quiet clips with known foreground origins."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from scripts import audio_paired_loud_background as io

SCHEMA = "chipmunk.audio-quiet-gain-transfer/1"
SOURCE = io.PANELS / "audio-short-edit-materialization-complete-v169c.json"
ROOMS = io.PANELS / "audio-speech-admission-transfer-2026-11-13.json"
CALIBRATION = io.ROOT / "backend/app/detectors/audio/calibrations/local-speech-v3.json"
METHOD = {
    "inputs": "One hash-ordered qualified three-second holdout delivery per voice. Mix with the first zero-word archived AMI excerpt from each of four rooms, selected by case number and assigned round-robin by family.",
    "conditions": "Normalize the foreground to -36 dBFS RMS. Set each room background to -31 dBFS RMS for a matched -5 dB signal-to-background ratio and PCM16 quantize. Include both quiet unoverlaid speech and the assigned quiet mixture, plus one isolated room control per room.",
    "comparison": "Use the unchanged speech activity gate, then retry with up to +6 dB without clipping on originally unassessed windows only. Feed the original samples to the calibrated classifier when the gain view admits them.",
    "advance": "At least two added AI detections in the noisy condition across two generator families, at most one additional noisy genuine AI flag, and zero new clean genuine or isolated background AI flags. Retain all existing decisions.",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make(folder, manifest, cache):
    import numpy as np

    from scripts.audio_short_edit_cases_v172 import float_pcm_sha, read_pcm

    anchor = io.read(SOURCE)["roles"]["holdout"]["private_manifest_sha256"]
    io.check(io.digest(manifest) == anchor, "Holdout speech manifest differs")
    rooms = io.read(ROOMS)
    noise = {}
    for room in rooms["freeze"]["rooms"]:
        item = min((row for row in rooms["rows"] if row["room"] == room and row["word_cells"] == 0),
                   key=lambda row: row["id"])
        data = (cache / "pcm" / f"{item['id']:03}.pcm").read_bytes()
        io.check(len(data) == 96000 and sha(data) == item["pcm_sha256"], "Room recording differs")
        noise[room] = (item["id"], level(data, -31))
    names = list(noise)
    foreground = sorted((row for row in io.read(manifest)["foregrounds"] if row["status"] == "qualified"),
                        key=lambda row: sha(f"{SCHEMA}/{row['slot']}".encode()))
    folder.mkdir(mode=0o700)
    (folder / "pcm").mkdir()
    groups, per_family, rows = set(), Counter(), []

    def save(signal, label, condition, room, source_hash, group=None):
        io.check(np.max(np.abs(signal)) < 32767 / 32768, "The quiet input would clip")
        data = np.rint(signal * 32768).astype("<i2").tobytes()
        index = len(rows)
        (folder / "pcm" / f"{index:03}.pcm").write_bytes(data)
        rows.append({"id": index, "label": label, "condition": condition, "group": group,
                     "room": room, "noise_case": noise[room][0], "source_pcm_sha256": source_hash, "pcm_sha256": sha(data)})

    for item in foreground:
        family, voice = item["lane"], item["voice"]
        if (family, voice) in groups:
            continue
        groups.add((family, voice))
        room = names[per_family[family] % len(names)]
        per_family[family] += 1
        path = Path(item["file"])
        io.check(io.digest(path) == item["wav_sha256"], "The retained speech file differs")
        data = read_pcm(path)
        io.check(float_pcm_sha(data) == item["prefix_pcm_sha256"], "The retained speech prefix differs")
        voice = level(data, -36)
        source_hash = item["prefix_pcm_sha256"]
        save(voice, family, "quiet", room, source_hash, len(groups))
        save(voice + noise[room][1], family, "quiet_mixed", room, source_hash, len(groups))
    for room in names:
        save(noise[room][1], "background", "isolated", room, None)
    io.check(per_family == {"genuine": 8, "clone": 6, "expressive": 12, "stock": 6}, "The captured voice population differs")
    io.write(folder / "freeze.json", {"schema": SCHEMA, "method": METHOD, "rows": rows, "source_manifest_sha256": anchor,
        "room_panel_sha256": io.digest(ROOMS), "calibration_sha256": io.digest(CALIBRATION),
        "runner_sha256": io.digest(Path(__file__))})
    print(f"Prepared {len(rows)} quiet matched deliveries", flush=True)


def level(data, db):
    import numpy as np

    pcm = np.frombuffer(data, "<i2").astype(np.float64) / 32768
    rms = np.sqrt(np.mean(pcm**2))
    io.check(rms > 0.0001, "RMS is too close to digital silence for a useful test")
    return pcm * (10**(db / 20) / rms)


def measure(folder):
    import numpy as np

    from app.core.config import settings
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_gain_gate_voice_transfer import WEIGHTS, count_samples
    from scripts.audio_radar_transfer import adapter_result
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    frozen = io.read(folder / "freeze.json")
    io.check(frozen["runner_sha256"] == io.digest(Path(__file__)) and frozen["calibration_sha256"] == io.digest(CALIBRATION)
             and frozen["room_panel_sha256"] == io.digest(ROOMS), "The quiet input freeze differs")
    network_free()
    settings.audio_detector_mode = "candidate"
    activity = SpeechActivity(WEIGHTS / "silero_vad.jit")
    voice = load_voice({"weights": str(WEIGHTS), "assets": ASSETS})
    observations = []
    for row in frozen["rows"]:
        data = (folder / "pcm" / f"{row['id']:03}.pcm").read_bytes()
        io.check(len(data) == 96000 and sha(data) == row["pcm_sha256"], "Quiet delivery differs")
        pcm = np.frombuffer(data, "<i2").astype(np.float32) / 32768
        before = adapter_result(pcm, voice, activity)["decision"]
        after, speech, gain = before, None, None
        if before == "unassessed":
            speech = count_samples(activity.spans(pcm))
            multiplier = min(10**(6/20), (32767 / 32768) / max(float(np.max(np.abs(pcm))), 1e-9))
            spans = activity.spans(pcm * multiplier)
            gain = count_samples(spans)
            if gain >= 24000:
                def admitted(values, expected=pcm, saved=spans):
                    io.check(np.array_equal(values, expected), "The detector must receive the quiet original")
                    return saved

                after = adapter_result(pcm, voice, SimpleNamespace(spans=admitted))["decision"]
        observations.append({"before": before, "after": after, "original_speech": speech, "gain_speech": gain})
        if len(observations) % 16 == 0:
            print(f"Measured {len(observations)} quiet deliveries", flush=True)
    receipt = {"freeze": frozen, "observations": observations, "model_files": ASSETS}
    io.write(folder / "receipt.json", receipt | {"summary": summarize(receipt)})


def summarize(receipt):
    from scripts.audio_gain_gate_voice_transfer import DIRECTIONS

    frozen, observations = (receipt[key] for key in ("freeze", "observations"))
    rows = frozen["rows"]
    io.check(frozen["schema"] == SCHEMA and frozen["method"] == METHOD and frozen["calibration_sha256"] == io.digest(CALIBRATION)
             and frozen["room_panel_sha256"] == io.digest(ROOMS) and len(rows) == len(observations) == 68, "The quiet study differs")
    counts, families = defaultdict(Counter), set()
    for row, observation in zip(rows, observations, strict=True):
        condition, label = row["condition"], row["label"]
        before, after, speech, gain = (observation[key] for key in ("before", "after", "original_speech", "gain_speech"))
        io.check(before in DIRECTIONS and after in DIRECTIONS, "Invalid direction")
        if before == "unassessed":
            io.check(0 <= speech < 24000 and 0 <= gain <= 48000 and (after != "unassessed") == (gain >= 24000),
                     "The quiet gate status differs")
        else:
            io.check(before == after and speech is None and gain is None, "Existing direction changed")
        count = counts[f"{condition}/{label}"]
        count.update({"files": 1, f"before_{before}": 1, f"after_{after}": 1,
                      "newly_scored": int(before == "unassessed" and after != "unassessed"),
                      "new_ai": int(before != "ai" and after == "ai")})
        if condition == "quiet_mixed" and label != "genuine" and before != "ai" and after == "ai":
            families.add(label)
    advance = (len(families) >= 2 and counts["quiet_mixed/genuine"]["new_ai"] <= 1
               and counts["quiet/genuine"]["new_ai"] == counts["isolated/background"]["new_ai"] == 0)
    return {"by_condition": dict(counts), "gained_noisy_families": sorted(families), "advance": advance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "measure", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        make(args.path, args.manifest, args.cache)
    elif args.phase == "measure":
        measure(args.path)
    else:
        receipt = io.read(args.path)
        summary = summarize(receipt)
        io.check(summary == receipt["summary"], "Quiet comparison does not replay")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
