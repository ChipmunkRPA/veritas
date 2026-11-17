"""Compare stereo repeat checks on newly selected, licensed meeting recordings."""
from __future__ import annotations

import argparse
import json
import wave
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from app.evidence import audio_channel_alignment as aligned
from app.evidence import (
    audio_channel_alignment_binding,
    audio_channel_difference,
    audio_exact_repetition,
    audio_similarity,
    audio_waveform_similarity,
)
from scripts import audio_channel_difference_panel as previous
from scripts import audio_gain_repetition_transfer as base

PUBLISHER = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
MEETINGS = ("ES2014a", "IS1003a", "IB4001", "TS3003a")
STARTS = (89, 116, 163, 198, 227, 254, 291, 333)
OFFSETS = (3, -5, 8, -7, -3, 6, -8, 5)
GAINS = (.78, 1.14, .89, 1.06, .84, 1.2, .92, .81)
SAMPLES = 60_000


def plan(directory):
    sources = []
    for meeting in MEETINGS:
        for channel in ("01", "05"):
            file = f"{meeting}.Array1-{channel}.wav"
            path = directory / file
            sources.append({"file": file, "bytes": path.stat().st_size, "sha256": base.digest(path),
                            "url": f"{PUBLISHER}/{meeting}/audio/{file}"})
    return {"schema": "chipmunk.audio-channel-alignment-trial/1", "sources": sources, "meetings": list(MEETINGS),
            "source_starts_s": list(STARTS), "right_offsets": list(OFFSETS), "right_gains": list(GAINS),
            "clip_samples": SAMPLES, "minimum_paired_iou": .50,
            "implementation_sha256": base.digest(Path(aligned.__file__)),
            "projection_sha256": base.digest(Path(audio_channel_alignment_binding.__file__)),
            "recipe_sha256": base.digest(Path(__file__))}


def delayed(values, shift):
    if shift > 0:
        return np.pad(values[:-shift], (shift, 0))
    return np.pad(values[-shift:], (0, -shift))


def deliveries(stereo, other, speech, index):
    start = 6_447 + 197 * index
    dest = 35_379 + 797 * index
    count = (3_840, 6_400, 10_240)[index % 3]
    ranges = ((start, start + count), (dest, dest + count))
    copied, replaced = stereo.copy(), stereo.copy()
    copied[dest:dest + count] = stereo[start:start + count]
    replaced[dest:dest + count] = other[:count]
    shift, gain = OFFSETS[index], GAINS[index]
    room_rms, foreground_rms = previous.rms(stereo), previous.rms(speech)
    if not room_rms or not foreground_rms:
        raise ValueError("Scheduled source has no active audio")
    scale = room_rms / foreground_rms
    voice_peak = np.max(np.abs(speech.astype(np.int32))) * scale * 10 ** (18 / 20) * 1.2
    room_peak = max(np.max(np.abs(values.astype(np.int32))) for values in (stereo, replaced))
    headroom = min(1.0, 30_000 / (voice_peak + room_peak + 4))

    def mixed(bed, level, mode):
        left = np.rint(speech * (scale * headroom * 10 ** (level / 20)))
        if mode == "centered":
            right = left
        else:
            right = np.rint(gain * delayed(left, shift))
            if mode == "moving":
                right[SAMPLES // 2:] = np.rint(delayed(left, -shift)[SAMPLES // 2:] / gain)
        mixed = np.rint(bed * headroom) + np.column_stack((left, right))
        if np.max(np.abs(mixed)) > 32767:
            raise ValueError("Constructed input clipped")
        return mixed.astype("<i2")

    yield "recorded", stereo, None
    yield "ordinary_panned", mixed(stereo, 10, "panned"), None
    yield "different_sound_panned", mixed(replaced, 10, "panned"), None
    yield "changing_pan", mixed(stereo, 10, "moving"), None
    yield "centered_10db", mixed(copied, 10, "centered"), ranges
    yield "panned_10db", mixed(copied, 10, "panned"), ranges
    yield "panned_18db", mixed(copied, 18, "panned"), ranges
    yield "panned_6db", mixed(copied, 6, "panned"), ranges


def pair_ranges(spans):
    return [(pair.first, pair.second) for pair in spans]


def outcome(rows):
    def match(pair, truth):
        return bool(truth) and all(
            2 * max(0, min(b, d) - max(a, c)) >= max(b, d) - min(a, c)
            for (a, b), (c, d) in zip(pair, truth, strict=True)
        )

    output = {}
    for variant in sorted({row["variant"] for row in rows}):
        cohort = [row for row in rows if row["variant"] == variant]
        result = {"deliveries": len(cohort)}
        for name in ("baseline", "with_alignment"):
            result[name] = {"flagged": sum(bool(row[name]) for row in cohort),
                            "correct": sum(any(match(pair, row["truth"]) for pair in row[name]) for row in cohort)}
        states = [audio_channel_alignment_binding.STATUSES[row["reason"]] for row in cohort]
        result["alignment_skipped"] = states.count("skipped")
        result["alignment_unassessed"] = states.count("unavailable") + states.count("ambiguous")
        result["alignment_new_wrong_pairs"] = sum(
            not match(pair, row["truth"]) for row in cohort for pair in row["with_alignment"] if pair not in row["baseline"]
        )
        output[variant] = result
    return output


def measurement(stereo, *, check_decoder):
    original, _ = base.score(stereo, None, "ffmpeg", production_check=check_decoder)
    pcm = base.pcm_mono(stereo, "ffmpeg")
    similar = audio_similarity.find_similar_spans(pcm).spans
    if not similar and not original["exact_mono"]:
        similar = audio_waveform_similarity.find_similar_spans(pcm).spans
    exact_difference = audio_exact_repetition.find_repeated_spans(audio_channel_difference.project(stereo.tobytes()))
    observed = [*original["exact_mono"], *original["exact_channel"][0], *original["exact_channel"][1],
                *original["gain"][0], *original["gain"][1]]
    baseline = sorted({(tuple(row["first"]), tuple(row["second"])) for row in observed}
                      | set(pair_ranges(similar)) | set(pair_ranges(exact_difference.spans)))
    if exact_difference.spans:
        alignment, reason, spans = None, "exact_channel_difference_found", ()
    else:
        alignment, reason, spans = aligned.search(stereo)
    return {"sha256": original["native_pcm_sha256"], "decoder_checked": check_decoder,
            "baseline": baseline, "with_alignment": sorted(set(baseline) | set(pair_ranges(spans))), "reason": reason,
            "alignment": vars(alignment) if alignment else None}


def run(directory, examples):
    frozen = json.loads((directory / "alignment-plan.json").read_bytes())
    if frozen != plan(directory):
        raise ValueError("Input files or frozen comparison recipe differ from the plan")
    if examples:
        examples.mkdir()
    rows = []
    with ExitStack() as stack:
        readers = {m: [stack.enter_context(wave.open(str(directory / f"{m}.Array1-{channel}.wav")))
                       for channel in ("01", "05")] for m in MEETINGS}
        for channels in readers.values():
            for reader in channels:
                base.check_profile(reader, 1)
        for j, name in enumerate(MEETINGS):
            voice = MEETINGS[(j + 1) % len(MEETINGS)]
            for i, begin in enumerate(STARTS):
                sample = begin * base.RATE
                stereo = previous.read_room(readers, name, sample, SAMPLES)
                other = previous.read_room(readers, name, sample + 5 * base.RATE, 10_240)
                speech = previous.read_room(readers, voice, sample + 12 * base.RATE, SAMPLES)[:, 1]
                for variant, data, truth in deliveries(stereo, other, speech, i):
                    measured = measurement(data, check_decoder=i == 0)
                    row = {"source": name, "start_s": begin, "variant": variant, "truth": truth, **measured}
                    rows.append(row)
                    if examples and ((truth and not row["baseline"] and row["with_alignment"])
                                     or (i == 0 and variant in {"recorded", "ordinary_panned"})):
                        (examples / f"{name}-{begin}-{variant}.wav").write_bytes(base.pcm_wav(data))
            print(f"Checked eight source positions from {name}", flush=True)
    report = {"plan": frozen, "totals": outcome(rows), "rows": rows}
    write(directory / "alignment-results.json", report)
    print(json.dumps(report["totals"], indent=2))


def write(path, document):
    with path.open("x") as out:
        json.dump(document, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "replay"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--examples", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        write(args.directory / "alignment-plan.json", plan(args.directory))
    else:
        run(args.directory, args.examples)


if __name__ == "__main__":
    main()
