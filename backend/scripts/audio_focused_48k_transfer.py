# SPDX-License-Identifier: Apache-2.0
"""Replay focused speech localization on the same clips delivered as 48 kHz WAV."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts import audio_focused_window_transfer as parent
from scripts.audio_vocal_event_mixtures import PUBLIC, read, save, sha

SCHEMA = "chipmunk.audio-focused-48k-transfer/1"
NATIVE_RATE = 48_000
PARENT = PUBLIC / "audio-focused-window-transfer-2026-11-12.json"
METHOD = {
    "delivery": "Resample each exact 16 kHz signed 16-bit mixture to mono 48 kHz signed 16-bit WAV with FFmpeg.",
    "measurement": "Use the serving resampled PCM and continuous clock. Keep the original windows, classifier and calibration.",
    "selection": "Keep every source in the parent roster, including matched genuine and isolated controls.",
}


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(output, original, capsule):
    parent.require(sha(original / "roster.json") == capsule["freeze"]["roster_sha256"], "Original roster differs")
    output.mkdir(mode=0o700)
    (output / "wav").mkdir(mode=0o700)
    (output / "responses").mkdir(mode=0o700)
    cases = []
    for row in capsule["roster"]:
        raw = parent.native_wave(original, row)
        file = output / "wav" / f"{row['case']:03}.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-filter_threads", "1", "-f", "s16le",
                        "-ar", str(parent.RATE), "-ac", "1", "-i", "pipe:0", "-ar", str(NATIVE_RATE),
                        "-c:a", "pcm_s16le", "-fflags", "+bitexact", str(file)],
                       input=raw, check=True, timeout=15)
        pcm, mapping = delivery(file, sha(file), row["samples"])
        cases.append({"case": row["case"], "wav_sha256": mapping["source_sha256"],
                      "pcm_sha256": mapping["pcm_sha256"], "samples": len(pcm)})
    save(output / "freeze.json", header() | {"cases": cases})
    print(f"Bound {len(cases)} 48 kHz WAV deliveries", flush=True)


def delivery(file, pin, samples):
    from app.detectors.audio.decode import decode_audio

    decoded = decode_audio(file, source_sha256=pin)
    mapping = decoded.mapping
    parent.require(len(decoded.pcm) == samples == mapping["samples"]
                   and (mapping["source_codec"], mapping["source_container"], mapping["source_bit_depth"],
                        mapping["source_channels"], mapping["source_sample_rate"]) == ("pcm_s16le", "wav", 16, 1, NATIVE_RATE)
                   and mapping["timeline_time_base"] == "1/16000"
                   and mapping["timeline"] == [{"start_sample": 0, "end_sample": samples, "source_pts": 0}],
                   "48 kHz delivery differs in encoding, duration or source clock")
    return decoded.pcm, mapping


def header():
    return {"schema": SCHEMA, "method": METHOD, "parent_sha256": sha(PARENT)}


def tally(capsule, responses):
    rows = capsule["roster"]
    parent.require([entry["case"] for entry in responses] == [row["case"] for row in rows],
                   "Comparison must cover the complete original roster")
    for row, response in zip(rows, responses, strict=True):
        parent.verify_windows(row, response)
    analysis = parent.compare(rows, responses)
    baseline = capsule["result"]["cells"]
    same, same_ranges = 0, 0
    for row, old, new in zip(rows, capsule["responses"], responses, strict=True):
        previous = {tuple(item["span"]): item["direction"] for item in old["windows"]}
        delivered = {tuple(item["span"]): item["direction"] for item in new["windows"]}
        same += previous == delivered
        same_ranges += parent.ranges(row["samples"], previous)["focused"] == parent.ranges(row["samples"], delivered)["focused"]
    return {"identical_decisions_files": same, "identical_selected_ranges_files": same_ranges,
            "native_16k": baseline, "delivered_48k": analysis}


def score(output, capsule):
    from app.detectors.audio import calibration
    from scripts.audio_radar_transfer import adapter_result
    from scripts.audio_reduced_background_localization import assets_for_scores

    parent.require(parent.METHOD["calibration_sha256"] == calibration.SHA256, "Voice calibration differs")
    lock = read(output / "freeze.json")
    parent.require(all(lock[key] == value for key, value in header().items()), "Delivery freeze differs")
    voice, activity = assets_for_scores(capsule["freeze"])
    responses = []
    for row, case in zip(capsule["roster"], lock["cases"], strict=True):
        parent.require(row["case"] == case["case"], "Wrong frozen case")
        file = output / "wav" / f"{row['case']:03}.wav"
        pcm, mapping = delivery(file, case["wav_sha256"], row["samples"])
        parent.require(mapping["pcm_sha256"] == case["pcm_sha256"], "Decoded delivery changed")
        response_path = output / "responses" / f"{row['case']:03}.json"
        raw = (pcm * 32768).astype("<i2").tobytes()
        if response_path.is_file():
            response = read(response_path)
        else:
            windows = []
            for start, end in parent.schedule(len(pcm)):
                entry = adapter_result(pcm[start:end], voice, activity)
                parent.require((entry["scope"] and entry["windows"] == [[0, parent.SPAN]])
                               or entry["decision"] == "unassessed", "Voice result outside its frozen scope")
                windows.append({"span": [start, end], "direction": entry["decision"],
                                "pcm_sha256": sha_bytes(raw[2 * start:2 * end])})
            response = {"case": row["case"], "windows": windows}
            temp = response_path.with_suffix(".tmp")
            save(temp, response)
            temp.replace(response_path)
        parent.verify_windows(row, response, raw)
        responses.append(response)
        if len(responses) % 10 == 0:
            print(f"Scored {len(responses)} of {len(lock['cases'])} deliveries", flush=True)
    save(output / "receipt.json", lock | {"responses": responses, "results": tally(capsule, responses)})


def review(receipt, capsule, native=None):
    proof = read(receipt)
    parent.require(all(proof[key] == value for key, value in header().items())
                   and proof["results"] == tally(capsule, proof["responses"]), "Comparison differs")
    checked = 0
    for row, case, response in zip(capsule["roster"], proof["cases"], proof["responses"], strict=True):
        parent.require(case["case"] == row["case"] and case["samples"] == row["samples"]
                       and all(parent.clock.SHA.fullmatch(case[key]) for key in ("wav_sha256", "pcm_sha256")),
                       "Input case commitment differs")
        if native:
            pcm, mapping = delivery(native / "wav" / f"{row['case']:03}.wav", case["wav_sha256"], row["samples"])
            parent.require(mapping["pcm_sha256"] == case["pcm_sha256"], "Original decoding differs")
            parent.verify_windows(row, response, (pcm * 32768).astype("<i2").tobytes())
            checked += len(response["windows"])
    print(json.dumps(proof["results"] | {"native_windows_verified": checked}, sort_keys=True, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "score", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--native", type=Path)
    args = parser.parse_args()
    capsule = read(PARENT)
    if args.action == "prepare":
        if args.native is None:
            parser.error("prepare requires the reconstructed parent bank as --native")
        prepare(args.path, args.native, capsule)
    elif args.action == "score":
        score(args.path, capsule)
    else:
        review(args.path, capsule, args.native)


if __name__ == "__main__":
    main()
