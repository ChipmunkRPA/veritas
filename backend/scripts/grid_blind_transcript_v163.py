"""Compare blind English transcription on the exact previously matched GRID originals."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import types
import wave
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts import av_speech_native_v110 as native
from scripts import grid_phone_alignment_v162 as baseline
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = native.ROOT, native.BASE, native.PANEL
TRIAL = BASE / "grid-blind-transcript-v163"
PUBLIC = PANEL / "grid-blind-transcript-freeze-v163.json"
ASSETS = BASE / "grid-blind-transcript-assets-v163"
SPEECH = native.ASSETS / "speech"
SCRIPT = "scripts.grid_blind_transcript_v163"
RESULT_SCHEMA = "chipmunk.grid-blind-transcript-result/163"
REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
REPO = "https://huggingface.co/openai/whisper-tiny"
TREE_URL = f"https://huggingface.co/api/models/openai/whisper-tiny/tree/{REVISION}"
TREE_SHA = "7d6ebe3bf45e6b27d1fa2fc44fef2eb690716d4480335445ce5978a3e92c7ce2"
PUBLISHER_ASSETS = {
    "generation_config.json": (
        3747,
        "4b26dd66b8f7bca37d851d259fdc118315cacc62",
        "a5d5325911f16e74001a72fa13d6e208eee51548f994646de1f4b4cc8b35b512",
    ),
    "tokenizer.json": (
        2480466,
        "1e95340ff836fad1b5932e800fb7b8c5e6d78a74",
        "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566",
    ),
}
SPECIAL = {
    "<|endoftext|>": 50257,
    "<|startoftranscript|>": 50258,
    "<|en|>": 50259,
    "<|transcribe|>": 50359,
    "<|notimestamps|>": 50363,
}
END = SPECIAL["<|endoftext|>"]
PREFIX = (50258, 50259, 50359, 50363)
DECODING = {
    "language": "en",
    "task": "transcribe",
    "return_timestamps": False,
    "return_token_timestamps": False,
    "return_dict_in_generate": True,
    "force_unique_generate_call": True,
    "condition_on_prev_tokens": False,
    "temperature": 0.0,
    "prompt_ids": None,
    "compression_ratio_threshold": None,
    "logprob_threshold": None,
    "no_speech_threshold": None,
}
GENERATION = {"num_beams": 1, "num_return_sequences": 1, "do_sample": False, "max_new_tokens": 64}
DIGITS = dict(
    zip(
        "0123456789",
        ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
        strict=True,
    )
)
UNMATCHED = "publisher_word_reference_unmatched"
CODE = (
    "backend/scripts/grid_blind_transcript_v163.py",
    "backend/tests/test_grid_blind_transcript_v163.py",
    "docs/grid-blind-transcript-v163-method.md",
    "backend/scripts/grid_phone_alignment_v162.py",
    "backend/scripts/grid_phone_alignment_v161.py",
    "backend/scripts/av_speech_native_v110.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/video_appearance_corrected_v112.py",
    "backend/scripts/local_probe_limits_v24.py",
    "backend/scripts/audio_tree_sampler_v64.py",
)
read, write, sha, require = native.read, native.write, native.sha, native.require


def file_bytes(path, size, oid, digest):
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size == size,
        "Publisher file size differs",
    )
    body = path.read_bytes()
    git_payload = f"blob {size}\0".encode("ascii") + body
    require(
        hashlib.sha1(git_payload, usedforsecurity=False).hexdigest() == oid
        and hashlib.sha256(body).hexdigest() == digest,
        "Publisher asset bytes differ",
    )
    return body


def publisher_assets():
    tree = ASSETS / "publisher-tree.json"
    require(
        tree.is_file() and not tree.is_symlink() and sha(tree) == TREE_SHA, "Pinned publisher tree required"
    )
    listed = read(tree)
    records = []
    for name, (size, oid, digest) in PUBLISHER_ASSETS.items():
        rows = [r for r in listed if r["path"] == name]
        require(
            rows == [{"type": "file", "oid": oid, "size": size, "path": name}], "Publisher tree row differs"
        )
        file_bytes(ASSETS / name, size, oid, digest)
        records.append(
            {"url": f"{REPO}/resolve/{REVISION}/{name}", "bytes": size, "git_blob": oid, "sha256": digest}
        )
    return {
        "revision": REVISION,
        "tree_url": TREE_URL,
        "tree_sha256": TREE_SHA,
        "files": records,
        "license": "apache-2.0",
    }


def baseline_paths():
    return (
        baseline.PUBLIC,
        baseline.TRIAL / "freeze.json",
        baseline.TRIAL / "private-inputs.json",
        baseline.TRIAL / "run/result.json",
        baseline.TRIAL / "run/observations.jsonl",
        PANEL / "grid-phone-alignment-complete-v162.json",
    )


def check_roster(inputs, previous, result):
    require(
        len(inputs) == len(previous) == result["planned_originals"] == 16, "Exact 16-row v162 roster required"
    )
    matched = []
    for row, observed in zip(inputs, previous, strict=True):
        if "unavailable" in row:
            require(row == observed == {"unavailable": UNMATCHED}, "Unmatched publisher row differs")
            continue
        require(
            len(row["words"]) == observed["words"] == 6
            and type(observed["blind_exact_words"]) is bool
            and type(observed["word_edits"]) is int
            and observed["word_edits"] >= 0
            and observed["blind_exact_words"] == (observed["word_edits"] == 0),
            "Exact matched v162 word measurements required",
        )
        matched.append(observed)
    require(
        len(matched) == result["blind_prompts_assessed"] == 8
        and result["publisher_word_reference_unmatched"] == 8
        and result["other_unavailable"] == 0
        and sum(r["blind_exact_words"] for r in matched) == result["blind_exact_prompts"]
        and sum(r["word_edits"] for r in matched) == result["blind_word_edits"]
        and sum(r["words"] for r in matched) == result["blind_reference_words"] == 48,
        "v162 result does not reconcile with its whole roster",
    )


def source_roster():
    public_freeze, local_freeze, input_path, result_path, journal, public_result = baseline_paths()
    frozen, result = read(public_freeze), read(result_path)
    require(
        frozen == read(local_freeze)
        and frozen["schema"] == "chipmunk.grid-phone-alignment-freeze/162"
        and frozen["implementation_sha256"] == sha(Path(baseline.__file__))
        and frozen["private_inputs_sha256"] == sha(input_path)
        and result == read(public_result)
        and result["schema"] == "chipmunk.grid-phone-alignment-result/162"
        and result["freeze_sha256"] == sha(public_freeze)
        and result["private_observations_sha256"] == sha(journal),
        "Actual matching v162 freeze, roster and result required",
    )
    inputs = read(input_path)
    previous = [json.loads(line) for line in journal.read_text().splitlines()]
    check_roster(inputs, previous, result)
    return inputs, previous


def identity():
    native.verify()
    source_roster()
    transformer = Path(importlib.metadata.distribution("transformers").locate_file("transformers"))
    library = (
        "generation/utils.py",
        "generation/configuration_utils.py",
        "generation/logits_process.py",
        "models/whisper/generation_whisper.py",
        "models/whisper/modeling_whisper.py",
        "models/whisper/feature_extraction_whisper.py",
    )
    return {
        "schema": "chipmunk.grid-blind-transcript-freeze/163",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "source_artifact_sha256": {str(p.relative_to(ROOT)): sha(p) for p in baseline_paths()},
        "native_preprocessor_freeze_sha256": sha(native.PUBLIC),
        "publisher_assets": publisher_assets(),
        "runtime_packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "tokenizers", "numpy", "safetensors", "psutil")
        },
        "runtime_source_sha256": {p: sha(transformer / p) for p in library},
        "planned_originals": 16,
        "matched_word_references": 8,
        "maximum_encoder_forwards": 8,
        "decode_options": DECODING,
        "generation_options": GENERATION,
        "decoder_prefix_ids": list(PREFIX),
        "word_rules": "Lowercase word tokens, remove punctuation, preserve numeric tokens in primary; isolated ASCII digits to English words in secondary.",
        "scope": "Offline same-roster blind ASR word comparison only. No phone timing, new detector or default change.",
    }


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve earlier trials")
    body = identity()
    TRIAL.mkdir(mode=0o700, exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)


def verify():
    require(read(TRIAL / "freeze.json") == read(PUBLIC) == identity(), "Exact published v163 freeze required")


def waveform(row):
    import numpy as np

    source = Path(row["wav_path"])
    archive = baseline.earlier.planning.ARCHIVE.resolve()
    require(
        source.resolve().is_relative_to(archive) and sha(source) == row["wav_sha256"], "Original WAV differs"
    )
    with wave.open(str(source), "rb") as reader:
        require(
            (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getcomptype())
            == (1, 2, 16000, "NONE"),
            "Expected exact whole mono signed16 16 kHz PCM",
        )
        count = reader.getnframes()
        raw = reader.readframes(count + 1)
    require(
        640 <= count <= 480000
        and len(raw) == count * 2
        and abs(Fraction(count, 16000) - Fraction(row["publisher_end_sample"], 25000)) <= Fraction(1, 25),
        "Original PCM or its published extent differs",
    )
    pcm = np.frombuffer(raw, dtype="<i2").astype("<f4") / np.float32(32768)
    native.validate_pcm(pcm)
    return pcm


def generated_content(ids):
    require(
        ids[: len(PREFIX)] == list(PREFIX)
        and len(PREFIX) < len(ids) <= len(PREFIX) + GENERATION["max_new_tokens"],
        "Frozen English prefix or generation length differs",
    )
    require(ids[-1] == END, "Blind generation did not finish within the frozen token limit")
    content = ids[len(PREFIX) : -1]
    require(all(type(i) is int and 0 <= i < END for i in content), "Unexpected control or timestamp token")
    return content


class BlindDecoder:
    def __init__(self):
        import torch
        from safetensors.torch import load_file
        from tokenizers import Tokenizer
        from transformers import (
            GenerationConfig,
            WhisperConfig,
            WhisperFeatureExtractor,
            WhisperForConditionalGeneration,
        )

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        self.tokenizer = Tokenizer.from_file(str(ASSETS / "tokenizer.json"))
        require(
            all(self.tokenizer.token_to_id(token) == index for token, index in SPECIAL.items()),
            "Publisher tokenizer control mapping changed",
        )
        self.generation = GenerationConfig.from_dict(read(ASSETS / "generation_config.json"))
        self.generation.forced_decoder_ids = None
        for key, value in GENERATION.items():
            setattr(self.generation, key, value)
        config = WhisperConfig.from_dict(read(SPEECH / "config.json"))
        config._attn_implementation = "eager"
        self.model = WhisperForConditionalGeneration(config).to(device="cpu", dtype=torch.float32)
        state = load_file(SPEECH / "model.safetensors", device="cpu")
        projection = "proj_out.weight"
        embedding = "model.decoder.embed_tokens.weight"
        require(
            projection not in state
            and self.model.proj_out.weight is self.model.model.decoder.embed_tokens.weight,
            "Expected the publisher's shared output projection",
        )
        state[projection] = state[embedding]
        self.model.load_state_dict(state, strict=True)
        require(
            all(
                value.dtype == torch.float32 and torch.equal(value, self.model.state_dict()[key])
                for key, value in state.items()
            ),
            "Full strict CPU model load differs from publisher tensor bytes",
        )
        self.model.eval().requires_grad_(False)
        self.extractor = WhisperFeatureExtractor.from_pretrained(SPEECH, local_files_only=True)
        self.encoder_forwards = self.decoder_forwards = 0
        self.model.model.encoder.register_forward_pre_hook(self.count_encoder)
        self.model.model.decoder.register_forward_pre_hook(self.count_decoder)

    def count_encoder(self, _module, _args):
        self.encoder_forwards += 1

    def count_decoder(self, _module, _args):
        self.decoder_forwards += 1

    def transcribe(self, pcm):
        import torch

        native.validate_pcm(pcm)
        with torch.inference_mode():
            mel = self.extractor(
                pcm,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                max_length=480000,
                truncation=True,
                do_normalize=False,
            ).input_features
            require(
                mel.shape == (1, 80, 3000)
                and mel.dtype == torch.float32
                and mel.device.type == "cpu"
                and torch.isfinite(mel).all(),
                "Expected the v110 CPU log-mel input",
            )
            encoder_before = self.encoder_forwards
            result = self.model.generate(input_features=mel, generation_config=self.generation, **DECODING)
            require(
                self.encoder_forwards == encoder_before + 1 and result.sequences.shape[0] == 1,
                "Expected one full clip decode",
            )
            ids = result.sequences[0].tolist()
            return self.tokenizer.decode(generated_content(ids), skip_special_tokens=True)


def tokenize(text):
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text.lower())


def digit_words(tokens):
    return [DIGITS.get(token, token) for token in tokens]


def score(row, old, hypothesis):
    reference = [word for word, _, _ in row["words"]]
    tokens = tokenize(hypothesis)
    edits = baseline.earlier.word_edits
    return {
        "hypothesis": hypothesis,
        "exact": tokens == reference,
        "edits": edits(reference, tokens),
        "digit_equivalent_exact": digit_words(tokens) == digit_words(reference),
        "digit_equivalent_edits": edits(digit_words(reference), digit_words(tokens)),
        "baseline_edits": old["word_edits"],
        "baseline_exact": old["blind_exact_words"],
        "reference_words": len(reference),
    }


def observe(inputs, previous, transcribe, publish):
    rows = []
    for ordinal, (source, old) in enumerate(zip(inputs, previous, strict=True)):
        row = {"roster_ordinal": ordinal}
        if "unavailable" in source:
            row["unavailable"] = UNMATCHED
        else:
            try:
                pcm = waveform(source)
                before = native.fingerprint(pcm)
                hypothesis = transcribe(pcm)
                require(native.fingerprint(pcm) == before, "Blind decoder altered original samples")
                row.update(score(source, old, hypothesis))
            except (OSError, ValueError, RuntimeError, EOFError, wave.Error) as error:
                row["unavailable"] = type(error).__name__
        rows.append(row)
        publish(row)
    return rows


def summary(rows):
    counts = Counter()
    for row in rows:
        if "unavailable" in row:
            counts["unmatched" if row["unavailable"] == UNMATCHED else "other_unavailable"] += 1
            continue
        counts["assessed"] += 1
        for key in (
            "exact",
            "edits",
            "digit_equivalent_exact",
            "digit_equivalent_edits",
            "baseline_exact",
            "baseline_edits",
            "reference_words",
        ):
            counts[key] += row[key]
        difference = row["edits"] - row["baseline_edits"]
        counts["fewer" if difference < 0 else "more" if difference > 0 else "tied"] += 1
    require(len(rows) == 16 and counts["unmatched"] == 8, "Every original roster row must remain represented")
    return {
        "planned_originals": len(rows),
        "publisher_word_reference_unmatched": counts["unmatched"],
        "other_unavailable": counts["other_unavailable"],
        "blind_prompts_assessed": counts["assessed"],
        "blind_reference_words": counts["reference_words"],
        "blind_exact_prompts": counts["exact"],
        "blind_word_edits": counts["edits"],
        "digit_equivalent_exact_prompts": counts["digit_equivalent_exact"],
        "digit_equivalent_word_edits": counts["digit_equivalent_edits"],
        "paired_v162_exact_prompts": counts["baseline_exact"],
        "paired_v162_word_edits": counts["baseline_edits"],
        "paired_fewer_edits": counts["fewer"],
        "paired_equal_edits": counts["tied"],
        "paired_more_edits": counts["more"],
    }


def worker(output):
    folder = TRIAL / "authored"
    require(
        output == folder / "output"
        and (folder / "resources.jsonl").is_file()
        and (folder / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact offline guarded worker required",
    )
    verify()
    output.mkdir(mode=0o700, exist_ok=False)
    inputs, previous = source_roster()
    decoder = BlindDecoder()
    journal = output / "private-observations.jsonl"
    with journal.open("x") as stream:

        def record(row):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()

        rows = observe(inputs, previous, decoder.transcribe, record)
    aggregate = summary(rows)
    require(
        aggregate["blind_prompts_assessed"] <= decoder.encoder_forwards <= 8
        and decoder.decoder_forwards <= 8 * GENERATION["max_new_tokens"],
        "Unexpected blind decoder forward budget",
    )
    result = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "private_observations_sha256": sha(journal),
        **aggregate,
        "encoder_forwards": decoder.encoder_forwards,
        "decoder_forwards": decoder.decoder_forwards,
        "new_neural_calls": decoder.encoder_forwards + decoder.decoder_forwards,
        "defaults_changed": False,
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def routed_popen(argv, **kwargs):
    output = TRIAL / "authored/output"
    require(
        argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored", str(output)],
        "Exact authored guard required",
    )
    return subprocess.Popen([*argv[:3], SCRIPT, "worker", str(output)], **kwargs)


def run():
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "subprocess": proxy,
            "prerequisites": lambda _command, _batch: True,
            "__file__": __file__,
        },
    )("authored")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "worker"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    require(
        (args.command == "worker") == (args.output is not None), "Only the worker takes an output argument"
    )
    if args.command == "worker":
        worker(args.output)
    else:
        (freeze if args.command == "freeze" else run)()
