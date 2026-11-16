# SPDX-License-Identifier: MIT
"""Replay frozen separate-channel speech points on new paired voices and rooms."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_stereo_regression as prior

SOURCE, voices = prior.SOURCE, prior.voices
RATE, SAMPLES, GRID = prior.RATE, prior.SAMPLES, prior.GRID
PARENT = prior.PANELS / "audio-stereo-regression-2026-11-15.json.gz"
MLS_REVISION = "faf6604dcf0bb9ec0d9c280a140ad74da9c931b0"
MLS_SHA = "3dfbd075e54516c1e012db24a111ad2e3f18ae7ef2fce115a74ed7b435ea269b"
MLS_SIZE = 248644329
SCHEMA = "chipmunk.audio-stereo-channel-replication/1"
FAMILIES = ("piper", "kitten")
ROOMS = ("aachen", "huddersfield")
PAIR_COUNT = 12
WIDTH = 15360
FADE = 320
LEVELS = ("strict", "wider")
METHOD = {
    "source": "Select 24 declared MLS development-shard speakers outside the two earlier stock experiments and their recorded donors. "
    "Use distinct book and source IDs, with one six-second lossless crop from each. Order by fixed hashes "
    "and enforce audible speech around the edit using waveform amplitude only.",
    "cases": "Pair two readers in each of 12 distinct scenes. Assign six scenes to the Aachen meeting room and six "
    "to Huddersfield. Alternate directions, edit clocks and the quieter voice's generation family. "
    "Render measured stereo responses with the second voice 5 dB louder in the original mono mix. "
    "Apply fixed half-amplitude headroom before splicing. Compare untouched, ordinary splices, "
    "generated replacements of each voice and replacements of both.",
    "decision": "Replay the existing mono check and both separate-channel fallbacks from the preceding study. "
    "Keep the mono flag when present. Otherwise select the highest admitted response across the two "
    "untouched source channels at the same five two-second window starts and fixed prior operating points. "
    "Report which staged edit was localized by sample overlap. A flag does not establish speaker identity.",
}


def rank(*parts):
    return SOURCE.digest("/".join((SCHEMA, *map(str, parts))).encode())


def contract():
    earlier = prior.union.duration.read_gz(PARENT)
    return {
        "schema": SCHEMA,
        "method": METHOD,
        "code_sha256": SOURCE.path_sha(Path(__file__)),
        "earlier_sha256": SOURCE.path_sha(PARENT),
        "prior_code_sha256": SOURCE.path_sha(Path(prior.__file__)),
        "source_revision": MLS_REVISION,
        "source_sha256": MLS_SHA,
        "pairs": PAIR_COUNT,
        "assets": earlier["held"]["contract"]["assets"],
        "mono_point": earlier["held"]["contract"]["mono_point"],
        "channel_points": {name: earlier["held"]["points"]["channels"][name] for name in LEVELS},
    }


def excluded(stock, placement):
    a, b = stock / "freeze.json", placement / "placements-freeze.json"
    first, second = SOURCE.read(a), SOURCE.read(b)
    speakers = {
        int(item["sources"][key])
        for item in first["cases"]
        if item["group"] in range(8)
        for key in ("host_reader", "donor_reader")
    }
    speakers.update(
        int(item[key]) for item in second["source_commitments"] for key in ("reader", "donor_reader")
    )
    return speakers, {a.name: SOURCE.path_sha(a), b.name: SOURCE.path_sha(b)}


def rms(values):
    import numpy as np

    return float(np.linalg.norm(values.astype(np.float64)) / math.sqrt(len(values)))


def read_native(body):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    waveform, rate = sf.read(io.BytesIO(body), dtype="int16")
    SOURCE.check(waveform.ndim == 1 and rate > 0, "MLS source was not originally mono")
    if rate == RATE:
        return waveform.astype("<i2", copy=False)
    divisor = math.gcd(rate, RATE)
    values = resample_poly(waveform.astype(np.float64), RATE // divisor, rate // divisor)
    return np.clip(np.rint(values), -32768, 32767).astype("<i2")


def select_crop(pcm, clock, source):
    a, b = clock, clock + WIDTH
    choices = list(range(0, len(pcm) - SAMPLES + 1, 4000))
    choices.sort(key=lambda start: rank("crop", source, start))
    alternatives = [
        point for point in range(8000, SAMPLES - WIDTH, 16000) if point + WIDTH <= a or point >= b
    ]
    for start in choices:
        crop = pcm[start : start + SAMPLES]
        if min(rms(crop[i : i + RATE]) for i in range(0, SAMPLES, RATE)) < 175 or rms(crop[a:b]) < 265:
            continue
        eligible = [(rms(crop[point : point + WIDTH]), point) for point in alternatives]
        level, alternative = max(eligible, default=(0, 0))
        if level >= 265:
            return start, crop, alternative
    return None


def reader_sources(path, excluded_ids, clocks):
    import pyarrow.parquet as pq

    SOURCE.check(
        path.stat().st_size == MLS_SIZE and SOURCE.path_sha(path) == MLS_SHA,
        "The pinned full MLS file differs",
    )
    table = pq.ParquetFile(path).read(
        columns=["speaker_id", "book_id", "audio", "audio_duration", "original_path"]
    )
    order = defaultdict(list)
    for index, row in enumerate(
        table.select(["speaker_id", "book_id", "audio_duration", "original_path"]).to_pylist()
    ):
        if not 7.0 <= row["audio_duration"] <= 20 or row["speaker_id"] in excluded_ids:
            continue
        order[row["speaker_id"]].append((index, row))
    readers = sorted(order, key=lambda speaker: rank("reader", speaker))
    books, seen, found = set(), set(), []
    for reader in readers:
        clock = clocks[len(found)]
        for index, metadata in sorted(
            order[reader], key=lambda item: rank("record", reader, item[1]["original_path"])
        ):
            book, key = metadata["book_id"], metadata["original_path"]
            if book in books or key in seen:
                continue
            raw = table["audio"][index].as_py()["bytes"]
            pcm = read_native(raw)
            planned = select_crop(pcm, clock, key)
            if planned is None:
                continue
            start, crop, alternate = planned
            found.append(
                (
                    {
                        "reader": rank("privacy-reader", reader),
                        "book": rank("privacy-book", book),
                        "source": rank("native-path", key),
                        "source_sha256": SOURCE.digest(raw),
                        "original_samples": len(pcm),
                        "crop_start": start,
                        "splice_source": alternate,
                    },
                    crop,
                )
            )
            books.add(book)
            seen.add(key)
            break
        if len(found) == len(clocks):
            break
    SOURCE.check(len(found) == len(clocks), f"The pinned shard supplied {len(found)} qualifying speakers")
    return found


def synths(assets, runtime):
    from importlib.metadata import version

    import numpy as np

    sys.path.extend([str(path) for path in runtime])
    pinned = SOURCE.read(assets / "freeze.json")["generation_assets_sha256"]
    SOURCE.check(
        all(SOURCE.path_sha(assets / name) == pin for name, pin in pinned.items()),
        "Pinned test voice assets differ",
    )
    sys.path.append(str(assets / "kittentts-0.8.0-py3-none-any.whl"))
    import espeakng_loader
    import onnxruntime
    from kittentts.onnx_model import KittenTTS_1_Onnx
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    from piper import PiperVoice, SynthesisConfig

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    piper = PiperVoice.load(assets / "en_US-ljspeech-medium.onnx")
    config = SOURCE.read(assets / "kitten-config.json")

    def generate(family, ordinal, prompt):
        seed = 9101 + ordinal
        onnxruntime.set_seed(seed)
        np.random.seed(seed)
        if family == "piper":
            options = SynthesisConfig(noise_scale=0, noise_w_scale=0)
            return np.concatenate(
                [part.audio_float_array for part in piper.synthesize(prompt, syn_config=options)]
            )
        kitten = KittenTTS_1_Onnx(
            str(assets / config["model_file"]),
            str(assets / config["voices"]),
            config["speed_priors"],
            config["voice_aliases"],
        )
        return kitten.generate_single_chunk(prompt, "Jasper")

    versions = {
        name: version(name)
        for name in (
            "piper-tts",
            "onnxruntime",
            "phonemizer-fork",
            "espeakng-loader",
            "misaki",
            "numpy",
            "scipy",
        )
    }
    return generate, versions, pinned


def fragment(samples, family):
    import numpy as np
    from scipy.signal import resample_poly

    voiced = np.flatnonzero(abs(samples) > 0.035 * np.max(abs(samples)))
    SOURCE.check(len(voiced) >= 2, "Stock generator returned no audible speech")
    raw_rate = {"piper": 22050, "kitten": 24000}[family]
    divisor = math.gcd(RATE, raw_rate)
    converted = resample_poly(
        samples[voiced[0] : voiced[-1] + 1].astype(np.float64), RATE // divisor, raw_rate // divisor
    )
    SOURCE.check(len(converted) >= WIDTH, "The generated utterance has insufficient audible duration")
    indexes = range(0, len(converted) - WIDTH + 1, 160)
    best = max(indexes, key=lambda i: rms(converted[i : i + WIDTH]))
    return converted[best : best + WIDTH], best


def splice(pcm, donor, clock):
    import numpy as np

    start, end = clock, clock + WIDTH
    window = np.ones(WIDTH, dtype=np.float64)
    window[:FADE] = np.linspace(0, 1, FADE)
    window[-FADE:] = window[:FADE][::-1]
    gain = rms(pcm[start:end]) / rms(donor)
    mixed = (1 - window) * pcm[start:end] + window * gain * donor
    SOURCE.check(np.max(abs(mixed)) < 32767, "Matched insertion clipped")
    output = pcm.copy()
    output[start:end] = np.rint(mixed).astype("<i2")
    return output


def room_inputs(hudd, air):
    from scripts.audio_blind_huddersfield_transfer import impulses
    from scripts.audio_flank_background_transfer import aachen_room

    aachen, aachen_pins, aachen_grant = aachen_room(air)
    paths, hudd_pins = impulses(hudd)
    return (
        {"aachen": aachen, "huddersfield": {"left": paths["S07"], "right": paths["S03"]}},
        {"air": aachen_pins, "air_grant_sha256": aachen_grant, "huddersfield": hudd_pins},
    )


def scene_cards():
    for i in range(PAIR_COUNT):
        room = ROOMS[i % 2]
        slot = i // 2
        schedule = ("staggered", "overlapping")[slot % 2]
        directions = ("left", "right") if (slot + i % 2) % 2 == 0 else ("right", "left")
        families = FAMILIES if (slot // 2 + i % 2) % 2 == 0 else FAMILIES[::-1]
        yield i, room, schedule, directions, families


def freeze(args):
    import numpy as np

    pairs = list(scene_cards())
    clocks = [clock for _, _, schedule, _, _ in pairs for clock in voices.LOCATIONS[schedule]]
    locked = contract()
    exclusions, pins = excluded(args.stock, args.placement)
    inputs = reader_sources(args.mls, exclusions, clocks)
    rooms, room_pins = room_inputs(args.huddersfield, args.aachen)
    generate, runtime, generation_pins = synths(args.stock, args.runtime)
    args.output.mkdir(mode=0o700)
    pcm_root = args.output / "pcm"
    pcm_root.mkdir()
    cases, metadata = [], []
    for scene, room, schedule, directions, families in pairs:
        clips, donors = [], []
        for lane, (family, clock) in enumerate(zip(families, voices.LOCATIONS[schedule], strict=True)):
            ordinal = 2 * scene + lane
            native, pcm = inputs[ordinal]
            pcm = np.rint(0.5 * pcm).astype("<i2")
            prompt = f"The lantern glowed beside the stone bridge at number {ordinal + 11}."
            raw = generate(family, ordinal, prompt)
            piece, sample = fragment(raw, family)
            control = pcm[native["splice_source"] : native["splice_source"] + WIDTH]
            variants = {
                "untouched": pcm,
                "genuine": splice(pcm, control, clock),
                "synthetic": splice(pcm, piece, clock),
            }
            clips.append(
                {
                    **native,
                    "family": family,
                    "original_span": [clock, clock + WIDTH],
                    "samples": variants,
                    "variants": {name: prior.pcm_hash(values) for name, values in variants.items()},
                }
            )
            donors.append(
                {
                    "family": family,
                    "prompt_sha256": SOURCE.digest(prompt.encode()),
                    "native_sha256": prior.audio_hash(raw),
                    "fragment_start": sample,
                }
            )
        bound, stems = voices.render(clips, schedule, rooms[room], directions)
        metadata.append(
            {"scene": scene, "room": room, "schedule": schedule, "voices": bound, "donors": donors}
        )
        for kind, choices in voices.CONDITIONS.items():
            stereo = stems[0][choices[0]].astype(np.int32) + stems[1][choices[1]].astype(np.int32)
            SOURCE.check(np.max(abs(stereo)) < 32768, "Clipping in a matched room delivery")
            digest = prior.pcm_hash(stereo)
            (pcm_root / f"{digest}.pcm").write_bytes(stereo.astype("<i2").tobytes())
            labels = {
                name: clip["span"]
                for name, clip, choice in zip(("quieter", "louder"), bound, choices, strict=True)
                if choice == "synthetic"
            }
            cases.append({"scene": scene, "room": room, "kind": kind, "pcm_sha256": digest, "labels": labels})
        print(f"Sealed fresh matched stereo scene {scene + 1}/{PAIR_COUNT}", flush=True)
    plan = {
        "contract": locked,
        "excluded_reader_count": len(exclusions),
        "exclusion_sha256": pins,
        "rooms": room_pins,
        "synthesis": {"assets": generation_pins, "runtime": runtime},
        "scenes": metadata,
        "cases": cases,
    }
    validate_plan(plan)
    SOURCE.write(args.output / "freeze.json", plan)


def validate_plan(plan):
    SOURCE.check(
        plan["contract"] == contract() and len(plan["scenes"]) == PAIR_COUNT,
        "Frozen rules or source roster differ",
    )
    expected, identities = [], defaultdict(set)
    for (scene, room, schedule, directions, families), item in zip(
        scene_cards(), plan["scenes"], strict=True
    ):
        SOURCE.check(
            (item["scene"], item["room"], item["schedule"]) == (scene, room, schedule),
            "Room assignment differs",
        )
        for direction, family, clock, voice in zip(
            directions, families, voices.LOCATIONS[schedule], item["voices"], strict=True
        ):
            SOURCE.check(
                voice["family"] == family
                and voice["direction"] == direction
                and voice["shift"] == 0
                and voice["original_span"] == [clock, clock + WIDTH]
                and voice["span"]
                == [clock + voice["direct_arrival"], clock + voice["direct_arrival"] + WIDTH],
                "Source voice clock or layout differs",
            )
            for name in ("reader", "book", "source", "source_sha256"):
                SOURCE.check(
                    voice[name] not in identities[name], "A reader, work or source recording was repeated"
                )
                identities[name].add(voice[name])
        for kind, choices in voices.CONDITIONS.items():
            labels = {
                name: clip["span"]
                for name, clip, choice in zip(("quieter", "louder"), item["voices"], choices, strict=True)
                if choice == "synthetic"
            }
            expected.append((scene, room, kind, labels))
    SOURCE.check(
        expected == [(r["scene"], r["room"], r["kind"], r["labels"]) for r in plan["cases"]]
        and len({r["pcm_sha256"] for r in plan["cases"]}) == len(expected),
        "Held controls, edit clocks or audio inputs differ",
    )


def readout(anchor):
    if anchor is None:
        return None
    return list(GRID[-anchor[1]])


def score(args):
    import numpy as np
    import torch

    from scripts.run_audio_specialists_v15 import load_voice

    plan = SOURCE.read(args.input / "freeze.json")
    validate_plan(plan)
    locked = plan["contract"]
    model = load_voice({"weights": args.weights, "assets": locked["assets"]})
    torch.set_num_threads(4)
    activity = prior.union.ROOM.runtime(args.weights / "silero_vad.jit")
    measured = []
    for case in plan["cases"]:
        stereo = voices.load_stereo(args.input, case["pcm_sha256"])
        mono = prior.arithmetic_mono(stereo)
        counts, logits = prior.scan(mono, model, activity)
        readings = {
            "mono": {"pcm_sha256": prior.audio_hash(mono), "speech_samples": counts, "logits": logits}
        }
        if prior.choose(prior.best(logits), locked["mono_point"]) is None:
            for name, channel in zip(("left", "right"), stereo.T, strict=True):
                values = channel.astype(np.float32) / 32768
                counts, logits = prior.scan(values, model, activity)
                readings[name] = {
                    "pcm_sha256": prior.audio_hash(values),
                    "speech_samples": counts,
                    "logits": logits,
                }
        measured.append(readings)
        if len(measured) % 5 == 0:
            print(f"Scored {len(measured)} of {len(plan['cases'])} fresh stereo deliveries", flush=True)
    record = {"plan": plan, "observed": measured}
    record["results"] = review(record)
    data = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.write_bytes(gzip.compress(data, mtime=0))


def valid_reading(row):
    values = row["logits"]
    SOURCE.check(
        len(values) == len(row["speech_samples"]) == len(GRID)
        and all(
            type(n) is int
            and 0 <= n <= 2 * RATE
            and ((value is None and n < RATE) or (n >= RATE and value is not None and math.isfinite(value)))
            for n, value in zip(row["speech_samples"], values, strict=True)
        ),
        "A score contradicts the speech gate",
    )
    return values


def decisions(readings, locked):
    mono = prior.best(valid_reading(readings["mono"]))
    best = mono if mono and mono[0] > locked["mono_point"] else None
    target = {"mono": (readout(best), mono is not None)}
    if best:
        SOURCE.check(set(readings) == {"mono"}, "A candidate ran before the primary non-flag")
        return {**target, **{name: target["mono"] for name in LEVELS}}
    SOURCE.check(set(readings) == {"mono", "left", "right"}, "Missing independent channel readouts")
    channels = [prior.best(valid_reading(readings[name])) for name in ("left", "right")]
    best = max((value for value in channels if value is not None), default=None)
    for name in LEVELS:
        chosen = readout(best) if best and best[0] > locked["channel_points"][name] else None
        target[name] = (chosen, best is not None or mono is not None)
    return target


def review(record):
    plan = record["plan"]
    validate_plan(plan)
    results, scene_gains = defaultdict(Counter), defaultdict(set)
    for case, scores in zip(plan["cases"], record["observed"], strict=True):
        labels = case["labels"]
        outcomes = decisions(scores, plan["contract"])
        for name, (window, assessed) in outcomes.items():
            hit = {lane for lane, span in labels.items() if prior.useful(window, [span])[0]}
            candidate = outcomes["mono"][0] is None and bool(window)
            if candidate and hit:
                scene_gains[case["room"], name].add(case["scene"])
            for group in (case["room"], "all"):
                tally = results[group, name]
                tally.update(
                    edited=bool(labels),
                    useful=bool(hit),
                    ordinary=not labels,
                    ordinary_flags=bool(window and not labels),
                    unassessed_edited=bool(labels) and not assessed,
                    unassessed_ordinary=not labels and not assessed,
                    new_useful=candidate and bool(hit),
                    new_ordinary=candidate and not labels,
                    wrong_interval=bool(window and labels and not hit),
                    quieter=int("quieter" in hit),
                    louder=int("louder" in hit),
                    both=len(hit) == 2,
                )
    out = {
        f"{room}/{method}": dict(
            counts,
            distinct_gained_scenes=(
                sum(len(scene_gains[r, method]) for r in ROOMS)
                if room == "all"
                else len(scene_gains[room, method])
            ),
        )
        for (room, method), counts in sorted(results.items())
    }
    SOURCE.check(record.get("results", out) == out, "Held findings differ from the frozen operating points")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("freeze")
    for name in ("mls", "stock", "placement", "huddersfield", "aachen", "output"):
        prepare.add_argument(name, type=Path)
    prepare.add_argument("--runtime", nargs="+", type=Path, required=True)
    run = modes.add_parser("score")
    for name in ("input", "weights", "output"):
        run.add_argument(name, type=Path)
    verify = modes.add_parser("review")
    verify.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args)
    elif args.mode == "score":
        score(args)
    else:
        record = prior.union.duration.read_gz(args.receipt)
        print(json.dumps(review(record), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
