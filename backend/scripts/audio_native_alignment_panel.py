# SPDX-License-Identifier: Apache-2.0
"""Replay native 48 kHz background repetition against the serving mono checks."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.audio-native-alignment-panel/1"
DURATION = 180_000
SCENE_STARTS = (37, 66, 94, 123, 155, 182, 220, 247)
SPEECH_STARTS = (52, 75, 104, 136, 171, 206, 232, 281)
SHIFTS = (4, -7, 14, -17, 23, -28, 34, -35)
GAINS = (.78, 1.14, .89, 1.06, .84, 1.20, .92, .81)
VOICES = {"development": ("ES2014a", "IS1003a"), "confirmation": ("IB4001", "TS3003a")}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("xb") as fd:
        fd.write(gzip.compress(text.encode(), mtime=0) if path.suffix == ".gz" else (text + "\n").encode())


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def segment(path, second, samples, rate):
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, rate, "NONE"):
            raise ValueError("Unknown native source format")
        wav.setpos(second * rate)
        data = wav.readframes(samples)
    if len(data) != samples * 2:
        raise ValueError("Native source sample span is incomplete")
    return np.frombuffer(data, dtype="<i2")


def save_wav(path, stereo):
    import wave

    with wave.open(str(path), "wb") as wav:
        wav.setparams((2, 2, 48_000, 0, "NONE", "not compressed"))
        wav.writeframes(stereo.tobytes())


def deliveries(room, replacement, voice, variant, *, offset):
    import numpy as np

    from scripts.audio_channel_alignment_panel import delayed
    from scripts.audio_channel_difference_panel import rms

    first = 19_341 + variant * 599
    second = 106_137 + variant * 2339
    count = (11520, 19200, 30720)[variant % 3]
    truth = ((first, first + count), (second, second + count))
    copy, different = room.copy(), room.copy()
    copy[second:second + count] = room[first:first + count]
    different[second:second + count] = replacement[:count]
    scale = rms(room) / rms(voice)
    gain = GAINS[variant]
    peak = max(np.max(np.abs(view.astype(np.int32))) for view in (room, different))
    headroom = min(1., 30000 / (np.max(np.abs(voice)) * scale * 10 ** .9 * 1.2 + peak + 4))

    def mixed(background, level, *, moving=False):
        left = np.rint(voice * (scale * headroom * 10 ** (level / 20)))
        right = np.rint(gain * delayed(left, offset))
        if moving:
            right[len(right) // 2:] = np.rint(delayed(left, -offset)[len(right) // 2:] / gain)
        rendered = np.rint(background * headroom) + np.column_stack((left, right))
        if np.max(np.abs(rendered)) > 32767:
            raise ValueError("The constructed stereo mix clipped")
        return rendered.astype("<i2")

    yield "recorded", room, None
    yield "ordinary_panned", mixed(room, 10), None
    yield "different_background", mixed(different, 10), None
    yield "moving_voice", mixed(room, 10, moving=True), None
    yield "repeat_10db", mixed(copy, 10), truth
    yield "repeat_18db", mixed(copy, 18), truth


def iou(a, b):
    shared = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return shared / (max(a[1], b[1]) - min(a[0], b[0])) if shared else 0.


def counts(rows):
    output = defaultdict(lambda: defaultdict(Counter))
    contributions = defaultdict(set)
    for row in rows:
        for name, response in row["responses"].items():
            spans = response["spans"]
            correct = bool(row["truth"] and any(all(iou(actual, labeled) >= .5
                                                   for actual, labeled in zip(pair, row["truth"], strict=True))
                                                 for pair in spans))
            output[row["role"]][name].update(files=1, assessed=response["assessed"], flagged=bool(spans), correct=correct)
            if correct:
                contributions[row["role"], name].add(row["scene"])
    for (role, name), scenes in contributions.items():
        output[role][name]["correct_source_groups"] = len(scenes)
    return output


def measure(root, ami, split, output):
    import numpy as np
    from scipy.signal import resample_poly

    from app.detectors.audio.decode import decode_audio
    from app.evidence import (
        audio_channel_alignment,
        audio_exact_repetition,
        audio_native_alignment,
        audio_similarity,
        audio_waveform_similarity,
    )

    output.mkdir(mode=0o700)
    inputs = [row for row in read(root / "recorded-sources.json") if row["split"] == split]
    rows, voices = [], {}
    for position, scene in enumerate(inputs):
        name = scene["scene"]
        files = [root / name / Path(item["member"]).name for item in scene["members"]]
        if any(digest(file) != record["sha256"] for file, record in zip(files, scene["members"], strict=True)):
            raise ValueError("The publisher background channels changed")
        meeting = VOICES[split][position % 2]
        voice_source = ami / f"{meeting}.Array1-05.wav"
        if meeting not in voices:
            voices[meeting] = digest(voice_source)
        for index, start in enumerate(SCENE_STARTS):
            stereo = np.column_stack([segment(file, start, DURATION, 48_000) for file in files])
            # The control replaces sound from outside the clip so it has no authored repeat.
            replacement = np.column_stack([segment(file, start + 5, 30720, 48_000) for file in files])
            speech_start = SPEECH_STARTS[index] + 11 * (position // 2)
            voice = resample_poly(segment(voice_source, speech_start, DURATION // 3, 16_000).astype(np.float64), 3, 1)
            for role, authored, truth in deliveries(stereo, replacement, voice, index, offset=SHIFTS[index]):
                media = output / f"{name}-{index}-{role}.wav"
                save_wav(media, authored)
                sha = digest(media)
                decoded = decode_audio(media, source_sha256=sha, retain_stereo=True)
                if (len(decoded.mapping["timeline"]) != 1 or decoded.stereo_pcm is None
                        or len(decoded.stereo_pcm) != DURATION // 3 or decoded.mapping["source_sample_rate"] != 48000):
                    raise ValueError("The delivered file clock differs from the scheduled source")
                payload = np.rint(decoded.pcm * 32768).astype("<i2").tobytes()
                mono_checks = (audio_exact_repetition.find_repeated_spans, audio_similarity.find_similar_spans,
                               audio_waveform_similarity.find_similar_spans)
                for check in mono_checks:
                    used = check(payload)
                    if used.spans:
                        break
                coarse_fit, coarse_reason, coarse = audio_channel_alignment.search(decoded.stereo_pcm)
                fit, reason, spans = audio_native_alignment.search(authored)

                def native(pairs):
                    return [tuple((3 * low, 3 * high) for low, high in (item.first, item.second)) for item in pairs]

                rows.append({"scene": name, "start_seconds": start, "meeting": meeting, "speech_start_seconds": speech_start,
                             "delivery": media.name, "input_sha256": sha, "role": role, "truth": truth,
                             "actual_offset": SHIFTS[index] if role != "recorded" else None,
                             "responses": {"serving_mono": {"spans": native(used.spans), "assessed": used.status == "measured"},
                                           "downsample_then_align": {"spans": native(coarse), "assessed": coarse_reason is None,
                                                                      "shift": 3 * coarse_fit.right_offset_samples if coarse_fit else None},
                                           "native_48k": {"spans": spans, "assessed": reason is None, "reason": reason,
                                                          "shift": fit.right_offset_samples if fit else None}}})
        print(name, len(rows), flush=True)
    save(output / "result.json.gz", {"schema": SCHEMA, "split": split, "rows": rows, "counts": counts(rows), "scenes": inputs,
                                    "recipe_sha256": digest(Path(__file__)),
                                    "foreground_sha256": voices, "implementation_sha256": digest(Path(audio_native_alignment.__file__))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    trial = actions.add_parser("measure")
    trial.add_argument("root", type=Path)
    trial.add_argument("ami", type=Path)
    trial.add_argument("split", choices=VOICES)
    trial.add_argument("output", type=Path)
    actions.add_parser("review").add_argument("receipt", type=Path)
    args = vars(parser.parse_args())
    action = args.pop("action")
    if action == "measure":
        measure(**args)
    else:
        record = read(args["receipt"])
        output = counts(record["rows"])
        if record["schema"] != SCHEMA or output != record["counts"]:
            raise ValueError("The measured results do not recount")
        print(json.dumps(output, sort_keys=True))
