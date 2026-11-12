"""Test short speech-window localization across exact six-second audio carriers."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from app.detectors.audio import decode
from scripts import audio_edge_window_grouped_sources as grouped
from scripts import audio_edge_window_transfer as windows

ROOT, RATE, SAMPLES = windows.ROOT, windows.RATE, windows.SAMPLES
read, write, sha, digest, require = windows.read, windows.write, windows.sha, windows.digest, windows.require
SCHEMA = "chipmunk.audio-short-codec-transfer/1"
RECIPES = {
    "flac": (".flac", ("-c:a", "flac"), "flac", 16000, 0),
    "mp3": (".mp3", ("-c:a", "libmp3lame", "-b:a", "48k"), "mp3", 16000, 1105),
    "opus": (".ogg", ("-c:a", "libopus", "-b:a", "24k", "-application", "voip"), "opus", 48000, 0),
}
ARMS = ("full", "combined")
LOSSES = ("mp3", "opus")
STAGES = ("calibration", "held")
BUDGETS = (2, 1)
EVAL_FILES = (40, 50)
RELEASE = ROOT / "docs/evaluation-panels/audio-short-codec-transfer-2026-11-12.json"


def inputs(held):
    for role, cases in (
        ("calibration", ((row, pcm) for row, pcm, _ in windows.calibration_cases())),
        ("held", grouped.cases(held)),
    ):
        for idx, (row, pcm) in enumerate(cases):
            yield role, f"{role}-{idx:02d}", row, pcm


def decode_delivered(path, name):
    _, _, codec, sample_rate, origin = RECIPES[name]
    decoded = decode.decode_audio(path, source_sha256=sha(path), max_seconds=7)
    mapping = decoded.mapping
    require(mapping["source_codec"] == codec and mapping["source_sample_rate"] == sample_rate
            and mapping["source_channels"] == mapping["channels"] == 1
            and mapping["sample_rate"] == RATE and mapping["timeline_time_base"] == "1/16000"
            and mapping["timeline"] == [{"start_sample": 0, "end_sample": SAMPLES, "source_pts": origin}]
            and len(mapping.get("audio_stream_indices", [0])) == 1 and len(decoded.pcm) == SAMPLES,
            f"{name} changed the exact decoded sample count, carrier profile, or clock")
    return (decoded.pcm * 32768).astype("<i2").tobytes(), mapping["timeline"][0]


def recipe(name, path, pcm):
    _, options, _, _, _ = RECIPES[name]
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "s16le", "-ar", str(RATE), "-ac", "1",
                    "-i", "pipe:0", "-map_metadata", "-1", *options, str(path)], input=pcm, check=True)


def prepare(run, held):
    run.mkdir(mode=0o700)
    records = []
    groups = {stage: {} for stage in STAGES}
    for role, key, row, pcm in inputs(held):
        slots = groups[role]
        group = slots.setdefault(row["group"], f"{role}-group-{len(slots):02d}")
        deliveries = {}
        for name, (ext, *_details) in RECIPES.items():
            path = run / (key + "-" + name + ext)
            recipe(name, path, pcm)
            output, clock = decode_delivered(path, name)
            if name == "flac":
                require(output == pcm, "Lossless input differed from the original PCM")
            else:
                require(output != pcm, "A lossy delivery did not alter the original samples")
            deliveries[name] = {"file": path.name, "bytes": {"sha256": sha(path)},
                                "pcm": {"sha256": digest(output)}, "clock": clock}
        records.append({"id": key, "role": role, "group": group, "truth": row["truth"],
                        "family": row.get("family"), "condition": row["arm"],
                        "original": {"sha256": digest(pcm)}, "deliveries": deliveries})
    require(tuple(sum(row["role"] == s for row in records) for s in STAGES) == EVAL_FILES
            and tuple(map(len, groups.values())) == (16, 14)
            and sum(row["truth"] is not None for row in records) == 12,
            "Study inputs do not match the locked denominators")
    write(run / "inputs.json", records)
    write(run / "freeze.json", {
        "schema": SCHEMA,
        "inputs": {"sha256": sha(run / "inputs.json")},
        "held": {"sha256": sha(held / "sources.json")},
        "script": {"sha256": sha(Path(__file__))},
        "inference": {"sha256": sha(Path(windows.__file__))},
        "assets": windows.native.transfer.OLD.voice.ASSETS,
        "selection_groups": 16,
        "held_groups": 14,
        "points": {"primary": BUDGETS[0], "sensitivity": BUDGETS[1]},
        "useful_minimum_iou": .30,
        "useful_minimum_recall": .50,
        "admission": {"minimum_new_useful_groups": 2, "maximum_new_false_groups": 1},
        "recipes": RECIPES,
    })
    print(f"Froze {len(records)} paired source recordings and {len(records) * len(RECIPES)} deliveries.", flush=True)


def strongest(record, arm):
    responses = [(item["score"], item["start"], item["end"]) for item in record
                 if item["score"] is not None and (item["start"], item["end"]) in windows.GRIDS[arm]]
    return max(responses, key=lambda value: (value[0], -value[1]), default=None)


def point(rows, arm, codec, budget):
    maxima = {}
    for row in rows:
        best = strongest(row["windows"][codec], arm)
        if best:
            maxima[row["group"]] = max(maxima.get(row["group"], -float("inf")), best[0])
    ordered = sorted(maxima.values(), reverse=True)
    require(len(ordered) >= 12 and sum(score > ordered[budget] for score in ordered) == budget,
            "Calibration has too few scored groups or tied cutoffs")
    return ordered[budget]


def score(run, weights):
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    freeze = read(run / "freeze.json")
    require(freeze["inputs"]["sha256"] == sha(run / "inputs.json")
            and freeze["script"]["sha256"] == sha(Path(__file__))
            and freeze["inference"]["sha256"] == sha(Path(windows.__file__)),
            "Study definitions changed after the freeze")
    windows.native.transfer.OLD.spectra.incumbent.network_free()
    require(all(sha(weights / name) == pin for name, pin in freeze["assets"].items()), "Pinned weights changed")
    activity = SpeechActivity(weights / "silero_vad.jit")
    model = load_voice({"weights": str(weights), "assets": freeze["assets"]})
    rows = read(run / "inputs.json")
    for stage in STAGES:
        assessed = []
        for idx, row in enumerate((row for row in rows if row["role"] == stage), 1):
            row["windows"] = {}
            for codec in LOSSES:
                delivery = row["deliveries"][codec]
                path = run / delivery["file"]
                require(sha(path) == delivery["bytes"]["sha256"], "Exact carrier changed")
                pcm, clock = decode_delivered(path, codec)
                require(digest(pcm) == delivery["pcm"]["sha256"] and clock == delivery["clock"],
                        "Exact decoded samples or clock changed")
                row["windows"][codec] = windows.measure(pcm, model, activity, {})
            assessed.append(row)
            if idx % 10 == 0:
                print(f"{stage}: scored {idx} paired files", flush=True)
        write(run / f"{stage}.json", assessed)
        if stage == "calibration":
            points = {codec: {arm: {str(k): point(assessed, arm, codec, k) for k in BUDGETS} for arm in ARMS}
                      for codec in LOSSES}
            write(run / "points.json", points)
            print("Wrote both codec points before scoring the held deliveries.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--held", type=Path, default=grouped.HELD)
    parser.add_argument("--weights", type=Path, default=windows.native.WEIGHTS)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.run, args.held)
    else:
        score(args.run, args.weights)


if __name__ == "__main__":
    main()
