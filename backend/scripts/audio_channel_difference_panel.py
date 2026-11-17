"""Reproduce a frozen stereo-background repetition comparison from public AMI WAVs."""
from __future__ import annotations

import argparse
import hashlib
import json
import wave
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from app.evidence import (
    audio_channel_difference,
    audio_exact_repetition,
    audio_similarity,
    audio_waveform_similarity,
)
from scripts import audio_gain_repetition_transfer as prior

ROOT = Path(__file__).resolve().parents[2]
PINS = ROOT / "docs/evaluation-panels/audio-gain-public-replay-2026-11-13.json"
MEETINGS = ("ES2004a", "IS1009a", "ES2005a", "ES2008a")
STARTS = (101, 123, 157, 183, 214, 246, 277, 318)
SAMPLES = 60000
VARIANTS = ("recorded", "ordinary_mix", "other_scene", "dual_mono", "center_10db", "center_18db", "offset_10db")
BOUND_TOLERANCE = 4


def sha(data):
    return hashlib.sha256(data).hexdigest()


def plan():
    pins = json.loads(PINS.read_text())["sources"]
    return {
        "schema": "chipmunk.audio-channel-difference-plan/1",
        "meetings": MEETINGS, "source_starts_s": STARTS, "samples": SAMPLES,
        "sources": [{"file": name, **pin} for name, pin in sorted(pins.items())],
        "foreground": "Next meeting in the roster, same scheduled start plus 12 seconds.",
        "install_region": "Both room channels are copied to the second range before speech is mixed.",
        "source_start_frame": "6893 + 239 * index", "destination_start_frame": "37811 + 631 * index",
        "copy_frames": "Rotate 3840, 6400 and 10240 samples by index.",
        "ordinary_replacement": "Same duration and destination, from 5 seconds after the scheduled room start.",
        "offset_control": "The right foreground is delayed four samples and attenuated by one eighth.",
        "variants": VARIANTS, "projection": audio_channel_difference.PROJECTION,
        "max_boundary_error_samples": BOUND_TOLERANCE,
    }


def rms(samples):
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def mixtures(stereo, other, center, index):
    destination = 37811 + 631 * index
    source = 6893 + 239 * index
    count = (3840, 6400, 10240)[index % 3]
    ranges = ((source, source + count), (destination, destination + count))

    room = np.rint(stereo / 2).astype(np.int32)
    copied = room.copy()
    copied[slice(*ranges[1])] = room[slice(*ranges[0])]
    ordinary = room.copy()
    ordinary[slice(*ranges[1])] = np.rint(other[:count] / 2).astype(np.int32)

    def mixed(bed, level, *, offset=False):
        if not rms(center) or not rms(room):
            raise ValueError("Scheduled source has no active audio")
        scale = rms(room) / rms(center) * 10 ** (level / 20)
        # Use the same unchanged room and headroom for every variant in this scene.
        ceiling = np.max(np.abs(center)) * rms(room) / rms(center) * 10 ** (18 / 20)
        headroom = min(1.0, 30000 / (ceiling + max(np.abs(room).max(), np.abs(ordinary).max()) + 2))
        voice = np.rint(center * scale * headroom).astype(np.int32)
        right = np.rint(np.pad(voice[:-4], (4, 0)) * 0.875).astype(np.int32) if offset else voice
        result = np.rint(bed * headroom).astype(np.int32) + np.column_stack((voice, right))
        if np.max(np.abs(result)) > 32767:
            raise ValueError("Constructed input clipped")
        return result.astype("<i2")

    yield "recorded", stereo, None
    yield "ordinary_mix", mixed(room, 10), None
    yield "other_scene", mixed(ordinary, 10), None
    yield "dual_mono", np.column_stack((center, center)), None
    yield "center_10db", mixed(copied, 10), ranges
    yield "center_18db", mixed(copied, 18), ranges
    yield "offset_10db", mixed(copied, 10, offset=True), ranges


def read_room(readers, meeting, start, count):
    return np.column_stack([prior.decode_wav(reader, start, count)[:, 0] for reader in readers[meeting]])


def measure(directory, frozen, examples):
    by_meeting = {}
    with ExitStack() as opened:
        for pin in frozen["sources"]:
            path = directory / pin["file"]
            if path.stat().st_size != pin["bytes"] or prior.digest(path) != pin["sha256"]:
                raise ValueError(f"Publisher WAV differs from its pin: {path.name}")
            reader = opened.enter_context(wave.open(str(path), "rb"))
            prior.check_profile(reader, 1)
            by_meeting.setdefault(path.name.split(".")[0], []).append(reader)
        rows = []
        for position, meeting in enumerate(MEETINGS):
            foreground = MEETINGS[(position + 1) % len(MEETINGS)]
            for index, start in enumerate(STARTS):
                sample = start * prior.RATE
                stereo = read_room(by_meeting, meeting, sample, SAMPLES)
                other = read_room(by_meeting, meeting, sample + 5 * prior.RATE, 10240)
                center = read_room(by_meeting, foreground, sample + 12 * prior.RATE, SAMPLES)[:, 1]
                for variant, delivered, truth in mixtures(stereo, other, center, index):
                    row, _ = prior.score(delivered, None, "ffmpeg", production_check=index == 0)
                    mono = prior.pcm_mono(delivered, "ffmpeg")
                    assert sha(mono) == row["mono_pcm_sha256"]
                    bands = audio_similarity.find_similar_spans(mono)
                    similar = bands.spans
                    if not similar and not row["exact_mono"]:
                        similar = audio_waveform_similarity.find_similar_spans(mono).spans
                    base = [*row["exact_mono"], *row["exact_channel"][0], *row["exact_channel"][1],
                            *row["gain"][0], *row["gain"][1], *pack(similar)]
                    projection = audio_channel_difference.project(delivered.tobytes())
                    projected = np.frombuffer(projection, "<i2")
                    reference = (delivered[:, 0].astype(np.int32) - delivered[:, 1]) // 2
                    if not np.array_equal(projected, reference):
                        raise ValueError("Calculation differs from independent int32 projection")
                    result = audio_exact_repetition.find_repeated_spans(projection)
                    difference = pack(result.spans)
                    if any(not np.array_equal(reference[slice(*span.first)], reference[slice(*span.second)]) for span in result.spans):
                        raise ValueError("Returned coordinates do not reproduce the repeated PCM")
                    rows.append({
                        "meeting": meeting, "start_s": start, "variant": variant, "truth": truth,
                        "input_sha256": row["native_pcm_sha256"], "mono_sha256": row["mono_pcm_sha256"],
                        "difference_sha256": sha(projection), "production_decode_checked": index == 0,
                        "base_spans": sorted({(tuple(pair["first"]), tuple(pair["second"])) for pair in base}),
                        "difference_spans": difference, "difference_status": result.status,
                        "difference_reason": result.reason,
                    })
                    if examples and position == index == 0:
                        (examples / f"{variant}.wav").write_bytes(prior.pcm_wav(delivered))
            print(f"Replayed eight source windows from {meeting}", flush=True)
    return rows


def pack(spans):
    return [{"first": span.first, "second": span.second} for span in spans]


def matches(pair, truth):
    return truth is not None and all(abs(a - b) <= BOUND_TOLERANCE for span, target in zip(pair, truth, strict=True)
                                    for a, b in zip(span, target, strict=True))


def totals(rows):
    summary = {}
    for variant in VARIANTS:
        cases = [row for row in rows if row["variant"] == variant]
        summary[variant] = {
            "deliveries": len(cases),
            "baseline_alerts": sum(bool(row["base_spans"]) for row in cases),
            "baseline_useful": sum(any(matches(pair, row["truth"]) for pair in row["base_spans"]) for row in cases),
            "difference_alerts": sum(bool(row["difference_spans"]) for row in cases),
            "difference_useful": sum(any(matches((pair["first"], pair["second"]), row["truth"])
                                         for pair in row["difference_spans"]) for row in cases),
            "difference_wrong_pairs": sum(not matches((pair["first"], pair["second"]), row["truth"])
                                          for row in cases for pair in row["difference_spans"]),
            "difference_unassessed": sum(row["difference_status"] != "measured" for row in cases),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "replay"))
    parser.add_argument("input", type=Path)
    parser.add_argument("--examples", type=Path)
    args = parser.parse_args()
    freeze = args.input / "difference-plan.json"
    current = json.loads(json.dumps(plan()))
    if args.action == "freeze":
        with freeze.open("x") as stream:
            json.dump(current, stream, sort_keys=True, indent=2)
        return
    if json.loads(freeze.read_text()) != current:
        raise ValueError("Comparison plan differs from the registered source schedule")
    if args.examples:
        args.examples.mkdir(exist_ok=False)
    rows = measure(args.input, current, args.examples)
    results = {"plan": current, "totals": totals(rows), "rows": rows}
    with (args.input / "difference-results.json").open("x") as stream:
        json.dump(results, stream, sort_keys=True, indent=2)
    print(json.dumps(results["totals"], indent=2))


if __name__ == "__main__":
    main()
