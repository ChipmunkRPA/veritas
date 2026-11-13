"""Compare word changes in two recordings with a pinned offline speech recognizer."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from scripts import audio_aligned_speech_difference as acoustic
from scripts.audio_source_isolation_diagnostic import PANELS, check, digest, path_sha, read, useful, write

SCHEMA = "chipmunk.audio-paired-word-change/1"
RATE = acoustic.RATE
HOP = acoustic.HOP
SCENES = ("clean", "construction", "leicester")
WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
ASSET_REPO = "https://huggingface.co/openai/whisper-tiny/resolve/169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
ASSETS = {
    "config.json": "ffdccec4f3211f4c63310f2b7098f309fe70f3952cedc5e4d11e43f5b2379b98",
    "preprocessor_config.json": "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666",
    "model.safetensors": "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395",
    "generation_config.json": "a5d5325911f16e74001a72fa13d6e208eee51548f994646de1f4b4cc8b35b512",
    "tokenizer.json": "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566",
}
GENERATION = {"language": "en", "task": "transcribe", "temperature": 0.0,
              "do_sample": False, "num_beams": 1, "max_new_tokens": 96}
METHOD = {
    "source": "All matched SINE preview cases and paired ordinary controls. All PartialEdit originals "
              "and both named generators on their clean, +5 dB market, and -5 dB construction files. "
              "Both AMI assessment meetings and all five ordinary controls per excerpt.",
    "speech": "Deterministic CPU English decode, no prompts and no reference text. Use 20 ms word times "
              "on each file's 16 kHz analysis clock. Casefold and remove punctuation except apostrophes "
              "inside words. Skip a pair if either decoded word sequence lacks verified times.",
    "witness": "Require at least six matching words and common coverage of at least 65% of the longer "
               "sequence. Select only replacement blocks of one to four words, bounded immediately on "
               "both sides by two matching words whose sequences occur exactly once in both copies. "
               "The changed submitted words must occupy 80 ms to three seconds.",
    "agreement": "Report the acoustic and transcript candidates separately. An agreed interval uses the "
                 "submitted word times when both channels cover at least one third of the smaller interval "
                 "and reach at least 0.20 intersection over union on both recordings' clocks.",
    "assessment": "A useful range covers at least half a labeled edit with at least 0.30 intersection "
                  "over union on the submitted clock. Benchmarks measure differences against an earlier "
                  "copy. No method establishes AI origin, and no new candidate enters overall confidence.",
    "decoding": GENERATION,
}
ACOUSTIC_RECEIPT = PANELS / "audio-aligned-speech-difference-2026-11-13.json"


def asset_paths(speech, language):
    paths = {name: (language if name in {"generation_config.json", "tokenizer.json"} else speech) / name
             for name in ASSETS}
    for name, path in paths.items():
        check(path.is_file() and path_sha(path) == ASSETS[name], f"Speech asset does not match publisher pin: {name}")
    return paths


def recognizer(speech, language):
    import numpy as np
    import torch
    from transformers import (
        GenerationConfig,
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperTokenizerFast,
        pipeline,
    )

    paths = asset_paths(speech, language)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration.from_pretrained(
        speech, local_files_only=True, use_safetensors=True, attn_implementation="eager"
    ).eval().requires_grad_(False)
    generation = GenerationConfig.from_dict(read(paths["generation_config.json"]))
    generation.forced_decoder_ids = None
    model.generation_config = generation
    tokenizer = WhisperTokenizerFast(tokenizer_file=str(paths["tokenizer.json"]))
    asr = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer,
                   feature_extractor=WhisperFeatureExtractor.from_pretrained(speech, local_files_only=True),
                   device="cpu")

    def decode(signal):
        with torch.inference_mode():
            value = asr(signal.astype(np.float32) / 32768, return_timestamps="word", generate_kwargs=GENERATION)
        return word_positions(value["chunks"], len(signal))

    return decode


def word_positions(chunks, samples):
    words = []
    for chunk in chunks:
        tokens = WORD.findall(chunk["text"].casefold().replace("’", "'"))
        if not tokens:
            continue
        a, b = chunk["timestamp"]
        if a is None or b is None:
            return None
        start, stop = max(0, round(a * RATE / HOP) * HOP), min(samples, round(b * RATE / HOP) * HOP)
        if start > stop or words and start < words[-1][2]:
            return None
        for token in tokens:
            words.append((token, start, stop))
    return words or None


def unique_run(tokens, flank):
    return sum(tokens[i:i+len(flank)] == flank for i in range(len(tokens) - len(flank) + 1)) == 1


def anchored(first, second):
    if first is None or second is None:
        return {"status": "unavailable", "counts": None, "common": None, "ranges": []}
    a, b = [[item[0] for item in words] for words in (first, second)]
    alignment = SequenceMatcher(None, a, b, autojunk=False)
    opcodes = alignment.get_opcodes()
    count = sum(stop-start for kind, start, stop, _, _ in opcodes if kind == "equal")
    result = {"status": "insufficient_match", "counts": [len(a), len(b)], "common": count, "ranges": []}
    if count < 6 or 100 * count < 65 * max(len(a), len(b)):
        return result
    result["status"] = "assessed"
    for previous, operation, following in zip(opcodes, opcodes[1:], opcodes[2:]):
        kind, a0, a1, b0, b1 = operation
        if kind != "replace" or not (1 <= a1-a0 <= 4 and 1 <= b1-b0 <= 4):
            continue
        if not (previous[0] == following[0] == "equal" and previous[2]-previous[1] >= 2
                and following[2]-following[1] >= 2):
            continue
        if any(not unique_run(tokens, flank) for tokens in (a, b) for flank in (a[a0-2:a0], a[a1:a1+2])):
            continue
        first_span = [first[a0][1], first[a1-1][2]]
        second_span = [second[b0][1], second[b1-1][2]]
        if first_span[0] >= first_span[1] or not 1280 <= second_span[1]-second_span[0] <= 3*RATE:
            continue
        result["ranges"].append({"earlier": first_span, "submitted": second_span,
                                 "earlier_words": a1-a0, "submitted_words": b1-b0})
    return result


def compatible(a, b):
    shared = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    lengths = a[1]-a[0], b[1]-b[0]
    return 3*shared >= min(lengths) and 5*shared >= sum(lengths)-shared


def agreement(candidate, word_ranges):
    if candidate is None:
        return []
    return [item for item in word_ranges if compatible(item["submitted"], candidate["submitted"])
            and compatible(item["earlier"], candidate["earlier"])]


def cases(sine, partialedit, ami):
    yield from acoustic.sine_cases(sine)
    yield from acoustic.partial_cases(partialedit, acoustic.bank("partialedit"), scenes=SCENES)
    yield from acoustic.ordinary_cases(ami, assessment=True)


def roster():
    expected = acoustic.expected_roster()
    for row in acoustic.bank("partialedit")["roster"]:
        if row["scene"] == "leicester":
            expected[str(row["id"])] = (row["utterance"], f"leicester/{row['family']}", row["truth"])
    return expected


def count(rows):
    summary = {}
    for row in rows:
        cell = summary.setdefault(row["role"], Counter())
        exact = row["truth"]
        lexical = row["word"]["ranges"]
        physical = row["acoustic"]
        both = agreement(physical, lexical)
        cell.update({"pairs": 1, "word_assessed": row["word"]["status"] == "assessed", "word_flags": bool(lexical),
                     "word_useful": any(useful(x["submitted"], t) for x in lexical for t in exact),
                     "acoustic_flags": physical is not None,
                     "acoustic_useful": bool(physical and any(useful(physical["submitted"], t) for t in exact)),
                     "agree_flags": bool(both), "agree_useful": any(useful(x["submitted"], t) for x in both for t in exact)})
    return summary


def compare(first, second, operating_point):
    measurement = acoustic.compare(first, second)
    selected = measurement["selected"]
    if selected is None or selected["score"] <= operating_point:
        return None
    residual = acoustic.processing_residual(first, second)
    if residual is not None and residual <= 0.10:
        return None
    return {"earlier": selected["earlier_span"], "submitted": selected["span"]}


def score(args):
    decode = recognizer(args.speech, args.language)
    cache = {}
    rows = []
    previous = read(ACOUSTIC_RECEIPT)
    acoustic.review(previous)
    for ident, group, role, truth, first, second in cases(args.sine, args.partialedit, args.ami):
        sizes = [len(first), len(second)]
        hashes = [digest(signal.tobytes()) for signal in (first, second)]
        for key, signal in zip(hashes, (first, second), strict=True):
            if key not in cache:
                cache[key] = decode(signal)
        first_words, second_words = [cache[key] for key in hashes]
        word = anchored(first_words, second_words)
        if args.private_words:
            with args.private_words.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"id": ident, "earlier": first_words, "submitted": second_words}) + "\n")
        rows.append({"id": ident, "group": group, "role": role, "truth": truth, "samples": sizes, "pcm_sha256": hashes,
                     "word": word, "acoustic": compare(first, second, previous["point"])})
        print(f"{len(rows)} pairs / {len(cache)} transcriptions", flush=True)
    return {"schema": SCHEMA, "method": METHOD, "assets": ASSETS, "parent_sha256": path_sha(ACOUSTIC_RECEIPT),
            "runtime": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "transformers")},
            "forwards": len(cache), "rows": rows, "summary": count(rows)}


def review(value):
    check(value["schema"] == SCHEMA and value["method"] == METHOD and value["assets"] == ASSETS
          and value["parent_sha256"] == path_sha(ACOUSTIC_RECEIPT), "Different method, assets or comparator")
    expected = roster()
    prior = {row["id"]: row for row in read(ACOUSTIC_RECEIPT)["assessment"]}
    partial = {str(row["id"]): row for row in acoustic.bank("partialedit")["roster"]}
    keys = set()
    for row in value["rows"]:
        ident = row["id"]
        check(expected.pop(ident) == (row["group"], row["role"], row["truth"]), "Different input role or label")
        if ident in prior:
            check(row["samples"] == prior[ident]["samples"] and row["pcm_sha256"] == prior[ident]["input_sha256"],
                  "Different committed input")
        else:
            check(row["pcm_sha256"][1] == partial[ident]["pcm_sha256"] and row["samples"][1] == partial[ident]["samples"],
                  "Different market input")
        keys.update(row["pcm_sha256"])
        word = row["word"]
        status, counts, common = (word[key] for key in ("status", "counts", "common"))
        if status == "unavailable":
            check(counts is None and common is None and not word["ranges"], "Unavailable word sequence reports a difference")
        else:
            check(status in {"assessed", "insufficient_match"} and 0 <= common <= min(counts)
                  and (status == "assessed") == (common >= 6 and 100*common >= 65*max(counts))
                  and (status == "assessed" or not word["ranges"]), "Invalid word-match summary")
        for item in word["ranges"] + ([row["acoustic"]] if row["acoustic"] else []):
            for side, samples in zip(("earlier", "submitted"), row["samples"], strict=True):
                start, stop = item[side]
                check(0 <= start < stop <= samples and start % HOP == 0 and (stop % HOP == 0 or stop == samples),
                      "Evidence timing exceeds one source clock")
        for item in word["ranges"]:
            check(1280 <= item["submitted"][1] - item["submitted"][0] <= 3*RATE
                  and all(1 <= item[k] <= 4 for k in ("earlier_words", "submitted_words")), "Unexpected word replacement")
    result = count(value["rows"])
    check(not expected and value["forwards"] == len(keys) and result == value["summary"], "Incomplete report or incorrect counts")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("score", "review"))
    parser.add_argument("receipt", type=Path)
    for option in ("speech", "language", "sine", "partialedit", "ami", "private-words"):
        parser.add_argument(f"--{option}", type=Path)
    args = parser.parse_args()
    if args.phase == "score":
        if not all((args.speech, args.language, args.sine, args.partialedit, args.ami)):
            parser.error("scoring requires pinned model files and all three verified source caches")
        if args.receipt.exists() or args.private_words and args.private_words.exists():
            parser.error("choose new output paths")
        if args.private_words:
            args.private_words.touch(mode=0o600, exist_ok=False)
        value = score(args)
        review(value)
        write(args.receipt, value)
    else:
        value = read(args.receipt)
    print(json.dumps(review(value), sort_keys=True))


if __name__ == "__main__":
    main()
