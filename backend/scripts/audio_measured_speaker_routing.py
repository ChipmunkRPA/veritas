"""Replay pinned diarization and assess an audio supplement's measured routing."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tarfile
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path

RATE = 16000
SAMPLES = 3 * RATE
TICK = RATE // 100
MODEL_SHA = "08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1"
RUNTIME_ARCHIVE_SHA = "2c3a7a719b778daeb14f147e011891f16e4b2ac62388d8b8d8b365cc2eebdf15"
RUNTIME_BINARY_MEMBER = "nemo-speech-0.3.0-macos-aarch64-cpu/bin/nemo-speech"
POINTS = {"nominal": ".50", "sensitive": ".30"}
ROUTING = {
    "model": "nvidia/Nemotron-3-Diarization q8_0, v0.3.0 official Apple Silicon CPU runtime",
    "license": "OpenMDW-1.1 weights; upstream release runtime",
    "mode": "One full-attention pass on each exact three-second mono PCM input per point",
    "speaker_points": POINTS,
    "single": "Exactly two tracks lasting at least 160 ms, with two co-active for at least 240 ms total and 180 ms continuously",
    "overlap": "Three or more simultaneous tracks for at least 160 ms total",
    "corroborated": "Both points must support the single-interferer route; either point can show overlap",
    "supplement": "Only an incumbent inconclusive voice window may be promoted. Uses the previously frozen raw or unpaired relation room point.",
    "advance": "On the separate room panel recover at least two additional distinct synthetic source groups over the same routed raw arm, add at most one genuine false flag across the two noisy scenes, and route at least half the assessed single-interferer views. On the controlled count panel no three-track mixture may route as two.",
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path):
    return json.loads(path.read_bytes())


def write(path: Path, data) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")


def pcm(folder: Path, row: dict) -> bytes:
    path = folder / "pcm" / f"{row['id']:03}.pcm"
    data = path.read_bytes()
    if len(data) != SAMPLES * 2 or hashlib.sha256(data).hexdigest() != row["pcm_sha256"]:
        raise ValueError("Exact 48,000 sample mono source binding required")
    return data


def _clock(value) -> int:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Diarization timestamp must be finite")
    sample = round(value * RATE)
    if not 0 <= sample <= SAMPLES + TICK or abs(value * RATE - sample) > .001 or sample % TICK:
        raise ValueError("Unexpected diarization source clock")
    return min(sample, SAMPLES)  # Upstream can close the final 10 ms cell one tick past the file.


def measurement(segments: list[dict]) -> dict:
    """Measure simultaneous tracks, counting each track at most once per cell."""
    events = defaultdict(list)
    if len(segments) > 8 * (SAMPLES // TICK):
        raise ValueError("Unexpected number of speaker intervals")
    for segment in segments:
        a, b = _clock(segment["start"]), _clock(segment["end"])
        speaker = segment["speaker"]
        if type(speaker) is not int or not 1 <= speaker <= 8 or a > b:
            raise ValueError("Invalid speaker interval")
        if a < b:
            events[a].append((speaker, 1))
            events[b].append((speaker, -1))
    events[SAMPLES]
    active = Counter()
    duration_by_speaker = Counter()
    two = three = run = longest = 0
    previous = 0
    for at, changes in sorted(events.items()):
        delta = at - previous
        simultaneous = sum(count > 0 for count in active.values())
        if simultaneous == 2:
            two += delta
            run += delta
            longest = max(longest, run)
        else:
            run = 0
        if simultaneous >= 3:
            three += delta
        for speaker, count in active.items():
            if count > 0:
                duration_by_speaker[speaker] += delta
        for speaker, change in changes:
            active[speaker] += change
        previous = at
    substantive = sum(length >= 16 * TICK for length in duration_by_speaker.values())
    route = "unassessed"
    if three >= 16 * TICK:
        route = "overlap_detected"
    elif substantive == 2 and two >= 24 * TICK and longest >= 18 * TICK:
        route = "single_interferer"
    return {"route": route, "substantive_tracks": substantive, "two_samples": two, "three_or_more_samples": three,
            "longest_two_samples": longest}


def routes(measurements: dict) -> dict:
    nominal, sensitive = (measurements[name]["route"] for name in POINTS)
    if "overlap_detected" in (nominal, sensitive):
        corroborated = "overlap_detected"
    elif nominal == sensitive == "single_interferer":
        corroborated = "single_interferer"
    else:
        corroborated = "unassessed"
    return {"nominal": nominal, "corroborated": corroborated}


def manifest_and_lock(folder: Path, binary: Path):
    manifest_path = folder / "manifest.json"
    manifest = read(manifest_path)
    archive_path = folder / "runtime.tar.gz"
    if sha(archive_path) != RUNTIME_ARCHIVE_SHA or sha(folder / "model.gguf") != MODEL_SHA:
        raise ValueError("Exact publisher archive and GGUF bytes required")
    binary_sha = sha(binary)
    with tarfile.open(archive_path, "r:gz") as archive:
        with archive.extractfile(RUNTIME_BINARY_MEMBER) as published_binary:
            if hashlib.file_digest(published_binary, "sha256").hexdigest() != binary_sha:
                raise ValueError("The executable must match the exact pinned release archive member")
    return manifest, {"schema": "chipmunk.audio-measured-speaker-routing/1",
                      "method": ROUTING, "code_sha256": sha(Path(__file__)), "model_sha256": MODEL_SHA,
                      "runtime_archive_sha256": RUNTIME_ARCHIVE_SHA, "binary_sha256": binary_sha,
                      "manifest_sha256": sha(manifest_path), "planned_cases": len(manifest["rows"])}


def run(folder: Path, binary: Path, manifest: dict):
    input_dir = folder / "waves"
    input_dir.mkdir()
    for row in manifest["rows"]:
        with wave.open(str(input_dir / f"{row['id']:03}.wav"), "wb") as output:
            output.setparams((1, 2, RATE, SAMPLES, "NONE", "not compressed"))
            output.writeframes(pcm(folder, row))
    environment = {key: os.environ[key] for key in ("PATH", "TMPDIR") if key in os.environ}
    environment.update(NEMO_SPEECH_MODEL_DIR=str(folder / "unused-cache"), NO_COLOR="1")
    measured = {row["id"]: {} for row in manifest["rows"]}
    started = time.monotonic()
    for name, point in POINTS.items():
        destination = folder / name
        destination.mkdir()
        command = [str(binary), "diarize", str(input_dir), "--model", str(folder / "model.gguf"),
                   "--offline", "--device", "cpu", "--format", "json", "--output-dir", str(destination),
                   "--concurrency", "1", "--onset", point, "--offset", point, "--pad-onset", "0",
                   "--pad-offset", "0", "--min-duration-on", "0", "--min-duration-off", "0"]
        with (folder / f"{name}.log").open("xb") as log:
            subprocess.run(command, check=True, timeout=240, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if {path.stem for path in destination.glob("*.json")} != {f"{key:03}" for key in measured}:
            raise ValueError("One complete diarization result per frozen PCM input required")
        for key in measured:
            path = destination / f"{key:03}.json"
            result = read(path)
            if Path(result["file"]).resolve() != (input_dir / f"{key:03}.wav").resolve():
                raise ValueError("Diarization output points to another input")
            measured[key][name] = {**measurement(result["segments"]), "segments_sha256": sha(path)}
    write(folder / "measured.json", {"freeze_sha256": sha(folder / "freeze.json"), "calls": len(measured)*len(POINTS),
          "elapsed_s": time.monotonic()-started, "rows": [{"id": i, "measurements": item} for i, item in measured.items()]})


def summarize(manifest, measured, folder=None):
    results = defaultdict(lambda: defaultdict(Counter))
    effects = defaultdict(lambda: defaultdict(Counter))
    gained_groups = defaultdict(set)
    for row, m in zip(manifest["rows"], measured["rows"], strict=True):
        if m["id"] != row["id"]:
            raise ValueError("Diarization rows are reordered")
        if folder is not None:
            pcm(folder, row)
            for name, expected in m["measurements"].items():
                source = folder / name / f"{row['id']:03}.json"
                measured_again = measurement(read(source)["segments"])
                if {**measured_again, "segments_sha256": sha(source)} != expected:
                    raise ValueError("Diarization measurement changed since scoring")
        detected = routes(m["measurements"])
        panel, condition = row["panel"], row["condition"]
        for gate, route in detected.items():
            results[panel][condition][f"{gate}:{route}"] += 1
        if row["decision"] is None:  # The controlled panel measures routing alone.
            continue
        truth = "synthetic" if row["label"] else "genuine"
        effects[panel][condition][f"incumbent_{truth}_{row['decision']}"] += 1
        for arm, crossed in row["crossed_room_point"].items():
            if crossed and row["decision"] == "inconclusive":
                for gate, route in detected.items():
                    if route == "single_interferer":
                        effects[panel][condition][f"{gate}:{arm}:added_{truth}"] += 1
                        if row["label"]:
                            gained_groups[panel, gate, arm].add(row["group"])
                if condition == "mostly_single_interferer":
                    effects[panel][condition][f"oracle:{arm}:added_{truth}"] += 1
    return {"routes": results, "decisions": effects,
            "synthetic_groups_added": {":".join(key): len(groups) for key, groups in gained_groups.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    folder, binary = args.folder.resolve(), args.binary.resolve()
    manifest, lock = manifest_and_lock(folder, binary)
    freeze = folder / "freeze.json"
    if args.action == "freeze":
        rows = manifest["rows"]
        if not rows or [row["id"] for row in rows] != list(range(len(rows))):
            raise ValueError("Ordered unique case indices required")
        for row in rows:
            pcm(folder, row)
        write(freeze, lock)
    elif read(freeze) != lock:
        raise ValueError("Original source, implementation and runtime freeze required")
    elif args.action == "run":
        run(folder, binary, manifest)
    else:
        measured = read(folder / "measured.json")
        if measured["freeze_sha256"] != sha(freeze) or measured["calls"] != len(manifest["rows"])*len(POINTS):
            raise ValueError("Diarization output is missing measured forwards")
        result = summarize(manifest, measured, folder)
        write(folder / "aggregate.json", {**result, "freeze_sha256": sha(freeze),
              "measurement_sha256": sha(folder / "measured.json"), "calls": measured["calls"], "elapsed_s": measured["elapsed_s"]})


if __name__ == "__main__":
    main()
