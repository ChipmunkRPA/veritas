# SPDX-License-Identifier: Apache-2.0
"""Replay localized identical-channel evidence on matched AMI recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import wave
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOST = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
DEVELOPMENT = ("ES2004a", "IS1009a")
HOLDOUTS = {"first": ("ES2006a", "IS1006a"), "revision": ("ES2010a", "IS1005a")}
FLOORS = {"first": (-50.0, -50.0), "revision": (-75.0, -70.0)}
RATE = 16000
CONDITIONS = {
    "local_left_copy": (3840, "wav"), "local_average": (7680, "wav"),
    "local_left_copy_aac": (3840, "m4a"), "local_average_aac": (7680, "m4a"),
    "untouched": (0, "wav"), "untouched_aac": (0, "m4a"), "gain": (0, "wav"),
    "muted_both": (0, "wav"), "whole_dual_mono": (0, "wav"), "crossfade_without_copy": (0, "wav"),
}
CODE = ("backend/app/detectors/audio/stereo_collapse.py", "backend/app/evidence/stereo_collapse_binding.py")


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_paths(meeting):
    return [f"{meeting}.Array1-{mic}.wav" for mic in ("01", "05")]


def freeze(development, held, receipt, protocol):
    meetings_by_panel = {"development": DEVELOPMENT, "held": HOLDOUTS[protocol]}
    sources = {}
    for panel, bank in (("development", development), ("held", held)):
        for meeting in meetings_by_panel[panel]:
            for name in source_paths(meeting):
                path = bank / name
                with wave.open(str(path), "rb") as audio:
                    if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, RATE):
                        raise ValueError(f"{name} is outside the 16 kHz mono PCM scope")
                    frames = audio.getnframes()
                if frames < 393 * RATE or path.stat().st_size < 2 * frames + 44:
                    raise ValueError(f"{name} is incomplete for the declared source grid")
                sources[name] = {"meeting": meeting, "panel": panel, "frames": frames, "sha256": sha(path),
                                 "bytes": path.stat().st_size, "url": f"{HOST}/{meeting}/audio/{name}"}
    record = {"schema": "chipmunk.audio-identical-stereo-study/2", "sources": sources, "protocol": protocol,
              "meetings": meetings_by_panel, "minimum_dbfs": list(FLOORS[protocol]),
              "conditions": {name: list(condition) for name, condition in CONDITIONS.items()},
              "excerpt_start_s": list(range(60, 391, 30)), "excerpt_seconds": 3,
              "code": {name: sha(ROOT / name) for name in (*CODE, str(Path(__file__).resolve().relative_to(ROOT)))},
              "decision": "Both held meetings need exact lossless recoveries and at least 40 of 48 exact "
                          "lossless intervals combined. At most one false interval on the 144 held controls. "
                          "Assess AAC separately. Qualifies a confidence-neutral editing observation."}
    with receipt.open("x") as handle:
        print(json.dumps(record, indent=2, sort_keys=True), file=handle)


def _recount(rows, protocol):
    expected = [(meeting, start, name) for meeting in (*DEVELOPMENT, *HOLDOUTS[protocol]) for start in range(60, 391, 30)
                for name in CONDITIONS]
    if [(r["meeting"], r["excerpt_start_s"], r["condition"]) for r in rows] != expected:
        raise ValueError("Delivery roster differs")
    totals = {}
    gained = set()
    for row in rows:
        meeting, clock, name = (row[k] for k in ("meeting", "excerpt_start_s", "condition"))
        ordinal = (clock - 60) // 30
        start = 11357 + 233 * ordinal
        length, _ = CONDITIONS[name]
        target = [start, start + length] if length else None
        panel = "development" if meeting in DEVELOPMENT else "held"
        if row["truth"] != target:
            raise ValueError("Authored interval differs from the frozen clock")
        range_count = len(row["measured_ranges"])
        if range_count > 3 or any(not 0 <= a < b <= 3 * RATE for a, b in row["measured_ranges"]):
            raise ValueError("Invalid reported interval")
        counter = totals.setdefault(panel, {}).setdefault(name, Counter())
        exact = target in row["measured_ranges"] if target else False
        useful = any(max(0, min(b, target[1]) - max(a, target[0])) * 2 >= max(b, target[1]) - min(a, target[0])
                     for a, b in row["measured_ranges"]) if target else False
        counter.update({"files": 1, "exact": exact, "useful": useful, "flagged": range_count > 0,
                        "false_intervals": range_count - int(useful), "prior_one_second_flags": row["prior_flag"]})
        if exact and panel == "held" and name in ("local_left_copy", "local_average"):
            gained.add(meeting)
    held = totals["held"]
    exact = sum(held[name]["exact"] for name in ("local_left_copy", "local_average"))
    false_intervals = sum(held[name]["false_intervals"] for name, (length, _) in CONDITIONS.items() if not length)
    return {"panels": totals, "held_exact_lossless": exact, "held_groups_with_exact": len(gained),
            "held_false_control_intervals": false_intervals,
            "advance": exact >= 40 and len(gained) == 2 and false_intervals <= 1}


def run(development, held, frozen):
    import numpy as np

    from app.detectors.audio.decode import decode_audio, source_ranges
    from app.detectors.audio.stereo_collapse import measure

    protocol = frozen["protocol"]
    minimum_difference, minimum_signal = FLOORS[protocol]
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        wav, aac = (Path(temp) / name for name in ("trial.wav", "trial.m4a"))
        for meeting in (*DEVELOPMENT, *HOLDOUTS[protocol]):
            bank = development if meeting in DEVELOPMENT else held
            names = source_paths(meeting)
            for name in names:
                if sha(bank / name) != frozen["sources"][name]["sha256"]:
                    raise ValueError(f"Input does not match frozen publisher bytes for {name}")
            with ExitStack() as contexts:
                tracks = [contexts.enter_context(wave.open(str(bank / name), "rb")) for name in names]
                if tracks[0].getnframes() != tracks[1].getnframes():
                    raise ValueError("Source microphones do not cover the same output clock")
                for index, position in enumerate(frozen["excerpt_start_s"]):
                    for track in tracks:
                        track.setpos(position * RATE)
                    samples = np.column_stack([np.frombuffer(track.readframes(3 * RATE), dtype="<i2") for track in tracks])
                    start = 11357 + 233 * index
                    for name, (length, carrier) in CONDITIONS.items():
                        changed = samples.copy()
                        target = [start, start + length] if length else None
                        if name.startswith("local_left"):
                            changed[start:start+length, 1] = changed[start:start+length, 0]
                        elif name.startswith("local_average"):
                            block = changed[start:start+length]
                            block[:] = (block.astype(np.int32).sum(axis=1) // 2)[:, None]
                        elif name == "muted_both":
                            changed[start:start+3840] = 0
                        elif name == "whole_dual_mono":
                            changed[:, 1] = changed[:, 0]
                        elif name == "gain":
                            changed[start:start+3840, 1] //= 4
                        elif name == "crossfade_without_copy":
                            block = changed[start:start+7680, 1]
                            weight = np.sin(np.linspace(0, np.pi, len(block))) * .8
                            block[:] = np.rint(block * (1 - weight) + changed[start:start+7680, 0] * weight)
                        with wave.open(str(wav), "wb") as output:
                            output.setparams((2, 2, RATE, 0, "NONE", ""))
                            output.writeframes(changed.astype("<i2", copy=False).tobytes())
                        file = wav
                        if carrier == "m4a":
                            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(wav), "-threads", "1",
                                            "-c:a", "aac", "-b:a", "96k", str(aac)], check=True, timeout=20)
                            file = aac
                        digest = sha(file)
                        decoded = decode_audio(file, source_sha256=digest, inspect_stereo=True)
                        stereo = decoded.mapping["stereo_context"]
                        if decoded.stereo_pcm is None or len(decoded.stereo_pcm) < 3 * RATE:
                            raise ValueError("Production decode did not retain the matched stereo prefix")
                        replay = measure(decoded.stereo_pcm[:3*RATE], decoded.mapping["timeline"],
                                         lambda a, b, mapping=decoded.mapping: source_ranges(mapping, a, b),
                                         minimum_difference=minimum_difference, minimum_signal=minimum_signal)
                        rows.append({"meeting": meeting, "excerpt_start_s": position, "condition": name, "truth": target,
                                     "source_sha256": digest, "decoded_stereo_sha256": stereo["stereo_pcm_sha256"],
                                     "measured_ranges": [[r["start_sample"], r["end_sample"]] for r in replay["selected"]],
                                     "prior_flag": any(w["status"] == "strong_imbalance" or w["downmix"]["strong_attenuation"]
                                                       for w in stereo["windows"]) or bool(stereo["channel_silence"]["selected"])})
            print(f"Scored {meeting}", flush=True)
    return {"freeze": frozen, "rows": rows, "results": _recount(rows, protocol)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "review"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--held", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--protocol", choices=tuple(HOLDOUTS), default="revision")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.development, args.held, args.output, args.protocol)
        return
    if args.action == "run":
        frozen = json.loads(args.freeze.read_bytes())
        if not all(sha(ROOT / name) == digest for name, digest in frozen["code"].items()):
            raise ValueError("Frozen evaluator differs")
        result = run(args.development, args.held, frozen)
        with args.output.open("x") as handle:
            print(json.dumps(result, indent=2, sort_keys=True), file=handle)
    else:
        result = json.loads(args.output.read_bytes())
        protocol = result["freeze"].get("protocol", "first")
        if result["results"] != _recount(result["rows"], protocol):
            raise ValueError("Results differ from the recorded intervals")
    print(json.dumps(result["results"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
