"""Bounded language-token diagnostic; no transcript, policy or authenticity change."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import subprocess
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import read_wav
from scripts import audio_alania_transfer_v179 as turkish
from scripts import audio_short_edit_materialize_v169c as english
from scripts import av_speech_native_v110 as native
from scripts import grid_blind_transcript_v163 as whisper
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT = Path(__file__).resolve().parents[2]
TRIAL = ROOT / "backend/data/independent-evaluation/audio-language-scope-v184"
PUBLIC = ROOT / "docs/evaluation-panels/audio-language-scope-freeze-v184.json"
METHOD = ROOT / "docs/audio-language-scope-v184-method.md"
MODULE = "scripts.audio_language_scope_v184"
ENGLISH_RECEIPT = ROOT / "docs/evaluation-panels/audio-short-edit-materialization-complete-v169c.json"
TURKISH_RECEIPT = ROOT / "docs/evaluation-panels/audio-alania-preparation-result-v179.json"
SPEECH_FREEZE = ROOT / "docs/evaluation-panels/av-speech-native-freeze-v110.json"
WHISPER_FREEZE = ROOT / "docs/evaluation-panels/grid-blind-transcript-freeze-v163.json"
SPEECH = native.ASSETS / "speech"
ASSETS = whisper.ASSETS
CODE = ("backend/scripts/audio_language_scope_v184.py", "backend/tests/test_audio_language_scope_v184.py",
        "docs/audio-language-scope-v184-method.md")
MARGIN_BINS = (0.0, 0.1, 0.25, 0.5, 1.0)
require, sha, read, write = native.require, native.sha, native.read, native.write


def model_assets():
    native_frozen, blind_frozen = read(SPEECH_FREEZE), read(WHISPER_FREEZE)
    commitments = {}
    for row in native_frozen["speech_assets"]:
        path = native.ASSETS / row["path"]
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == row["bytes"]
                and sha(path) == row["sha256"], "Frozen Whisper tensor or preprocessing asset differs")
        commitments[path.name] = row["sha256"]
    for row in blind_frozen["publisher_assets"]["files"]:
        path = ASSETS / Path(row["url"]).name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == row["bytes"]
                and sha(path) == row["sha256"], "Frozen Whisper language configuration differs")
        commitments[path.name] = row["sha256"]
    config = read(ASSETS / "generation_config.json")
    languages = config.get("lang_to_id") or {}
    require(config.get("is_multilingual") is True and config.get("decoder_start_token_id") == 50258
            and languages.get("<|en|>") == 50259 and languages.get("<|tr|>") == 50268
            and len(languages) >= 90 and len(set(languages.values())) == len(languages)
            and all(isinstance(key, str) and type(value) is int for key, value in languages.items()),
            "Expected the pinned multilingual language-token table")
    return commitments


def english_rows():
    """Verify only development metadata and the 14 genuine WAVs; never open holdout."""
    stage = english.TRIAL / "bind-00"
    profile_path, result_path = stage / "profile.json", stage / "output/result.json"
    profile, result, published = read(profile_path), read(result_path), read(ENGLISH_RECEIPT)
    require(result["status"] == "complete" and result["holdout_scored"] is False and result["new_neural_calls"] == 0
            and result["freeze_sha256"] == sha(english.PUBLIC)
            and profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0
            and profile["new_neural_calls"] == 0 and profile["worker_receipt_sha256"] == sha(result_path)
            and profile["resource_journal_sha256"] == sha(stage / "resources.jsonl")
            and published["resource_profile_sha256"] == sha(profile_path)
            and published["roles"]["development"] == result["roles"]["development"],
            "Exact guarded v169c development preparation required")
    folder = english.MEDIA / "development"
    manifest = folder / "private-manifest.json"
    require(not folder.is_symlink() and manifest.is_file() and not manifest.is_symlink()
            and sha(manifest) == result["roles"]["development"]["private_manifest_sha256"],
            "Exact v169c development manifest required")
    body = read(manifest)
    selected = [row for row in body["foregrounds"] if row.get("lane") == "genuine"]
    require(body["role"] == "development" and body["scored"] is False and len(selected) == 14
            and len({row["group_id"] for row in selected}) == len({row["voice"] for row in selected}) == 7
            and all(row["status"] == "qualified" and row["license"] == "CC-BY-4.0"
                    and row["candidate"].startswith("libri:") and row["voice"].startswith("librispeech:")
                    for row in selected), "Fourteen qualified LibriSpeech development clips from seven groups required")
    out = []
    for row in selected:
        path = Path(row["file"])
        require(path.parent == folder and path.is_file() and not path.is_symlink() and sha(path) == row["wav_sha256"]
                and row["wav_range"] == turkish.source_range(48000), "Exact development WAV and sample range required")
        pcm = read_wav(path.read_bytes())
        require(len(pcm) == 48000 and turkish.sha256(pcm.astype("<f4").tobytes()).hexdigest() == row["prefix_pcm_sha256"],
                "Exact development PCM required")
        out.append({"slot": row["slot"], "group": row["group_id"], "language": "en", "path": str(path),
                    "wav_sha256": row["wav_sha256"], "pcm_sha256": row["prefix_pcm_sha256"], "samples": len(pcm)})
    return out


def turkish_rows():
    selected = turkish.verify()
    rows, receipt = turkish.verified_preparation(selected)
    require(receipt == read(TURKISH_RECEIPT) and len(rows) == 24 and all(r["status"] == "qualified" for r in rows),
            "Exact full v179 Turkish preparation required")
    return [{"slot": row["slot"], "group": row["slot"], "language": "tr", "render_mode": row["render_mode"],
             "path": str(turkish.prepared_dir() / f"{row['slot']}.wav"), "wav_sha256": row["wav_sha256"],
             "pcm_sha256": row["pcm_sha256"], "samples": row["samples"]} for row in rows]


def context():
    commitments = model_assets()
    rows = [*turkish_rows(), *english_rows()]
    packages = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "transformers", "safetensors", "psutil")}
    return {"schema": "chipmunk.audio-language-scope-freeze/184", "planned": {"tr_synthetic": 24, "en_genuine": 14},
            "independent_groups": {"tr_synthetic": 24, "en_genuine": 7}, "assets": commitments,
            "source_receipt_sha256": {"v169c": sha(ENGLISH_RECEIPT), "v179": sha(TURKISH_RECEIPT)},
            "code_sha256": {name: sha(ROOT / name) for name in CODE}, "runtime_packages": packages,
            "method": "first decoder position over all publisher language tokens; no transcription or generation",
            "margin_bins": list(MARGIN_BINS), "fitting_permitted": False, "holdout_access_permitted": False}, rows


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve all previous attempts")
    record, rows = context()
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "private-inputs.json", rows)
    record["private_inputs_sha256"] = sha(TRIAL / "private-inputs.json")
    write(TRIAL / "freeze.json", record)
    write(PUBLIC, record)


def verify():
    record, rows = context()
    record["private_inputs_sha256"] = sha(TRIAL / "private-inputs.json")
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == record and read(TRIAL / "private-inputs.json") == rows,
            "Exact frozen diagnostic and inputs required")
    return rows


def summarise_logits(logits, language_ids):
    """Normalize only the publisher's language tokens, preserving a descriptive margin."""
    require(len(language_ids) >= 3 and len(logits) == len(language_ids) and len(set(language_ids)) == len(language_ids)
            and all(type(value) is int for value in language_ids) and all(math.isfinite(float(value)) for value in logits),
            "Complete finite language-token logits required")
    values = np.asarray(logits, dtype=np.float64)
    probs = np.exp(values - np.max(values))
    probs /= np.sum(probs)
    order = sorted(range(len(language_ids)), key=lambda i: (-float(probs[i]), language_ids[i]))
    winner = language_ids[order[0]]
    return {"label": "english" if winner == 50259 else "turkish" if winner == 50268 else "other",
            "margin": float(probs[order[0]] - probs[order[1]])}


def summarise(rows):
    require(len(rows) == 38 and Counter(row["language"] for row in rows) == {"tr": 24, "en": 14},
            "Every planned language case needs a result")
    require(all(row.get("label") in {"english", "turkish", "other", "unavailable"}
                and (row.get("label") != "unavailable" or row.get("margin") is None) for row in rows),
            "Only known diagnostic outcomes are permitted")
    result = {}
    for language in ("tr", "en"):
        items = [row for row in rows if row["language"] == language]
        labels = Counter(row["label"] for row in items)
        bins = Counter()
        grouped = defaultdict(list)
        for row in items:
            grouped[row["group"]].append(row["label"])
            if row["label"] != "unavailable":
                margin = row["margin"]
                require(isinstance(margin, (int, float)) and math.isfinite(margin) and 0 <= margin <= 1,
                        "A scored result needs a finite margin")
                label = "0-.1" if margin < .1 else ".1-.25" if margin < .25 else ".25-.5" if margin < .5 else ".5-1"
                bins[label] += 1
        require(len(grouped) == (24 if language == "tr" else 7), "Frozen source groups differ")
        result[language] = {"planned": len(items), "groups": len(grouped),
                            "labels": {key: labels[key] for key in ("english", "turkish", "other", "unavailable")},
                            "margin_bins": {key: bins[key] for key in ("0-.1", ".1-.25", ".25-.5", ".5-1")},
                            "groups_with_any_english": sum("english" in values for values in grouped.values()),
                            "groups_with_any_non_english": sum(any(x in {"turkish", "other"} for x in values) for values in grouped.values()),
                            "groups_with_unavailable": sum("unavailable" in values for values in grouped.values())}
    return result


class LanguageModel:
    def __init__(self):
        import torch
        from safetensors.torch import load_file
        from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        config = WhisperConfig.from_dict(read(SPEECH / "config.json"))
        config._attn_implementation = "eager"
        self.model = WhisperForConditionalGeneration(config).to(device="cpu", dtype=torch.float32)
        state = load_file(SPEECH / "model.safetensors", device="cpu")
        require("proj_out.weight" not in state and self.model.proj_out.weight is self.model.model.decoder.embed_tokens.weight,
                "Expected tied Whisper output projection")
        state["proj_out.weight"] = state["model.decoder.embed_tokens.weight"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False)
        self.extractor = WhisperFeatureExtractor.from_pretrained(SPEECH, local_files_only=True)
        self.ids = sorted(read(ASSETS / "generation_config.json")["lang_to_id"].values())
        self.forwards = 0
        self.model.register_forward_pre_hook(self._count)

    def _count(self, _module, _args):
        self.forwards += 1

    def score(self, pcm):
        import torch

        native.validate_pcm(pcm)
        with torch.inference_mode():
            mel = self.extractor(pcm, sampling_rate=16000, return_tensors="pt", padding="max_length",
                                 max_length=480000, truncation=True, do_normalize=False).input_features
            require(mel.shape == (1, 80, 3000) and mel.dtype == torch.float32 and torch.isfinite(mel).all(),
                    "Expected one finite CPU log-mel input")
            before = self.forwards
            logits = self.model(input_features=mel, decoder_input_ids=torch.tensor([[50258]], dtype=torch.long),
                                use_cache=False).logits
            require(self.forwards == before + 1 and logits.ndim == 3 and logits.shape[:2] == (1, 1),
                    "Exactly one native first-position forward required")
            return summarise_logits(logits[0, 0, self.ids].tolist(), self.ids)


def worker(output):
    folder = TRIAL / "authored"
    require(output == folder / "output" and (folder / "resources.jsonl").is_file()
            and (folder / "worker.log").is_file() and os.environ.get("HF_HUB_OFFLINE") == "1"
            and os.environ.get("TRANSFORMERS_OFFLINE") == "1", "Only the guarded offline worker is permitted")
    inputs = verify()
    output.mkdir(mode=0o700)
    model, rows = LanguageModel(), []
    for row in inputs:
        before = model.forwards
        path = Path(row["path"])
        require(path.is_file() and not path.is_symlink() and sha(path) == row["wav_sha256"], "Input WAV changed")
        pcm = read_wav(path.read_bytes())
        require(len(pcm) == row["samples"] and turkish.sha256(pcm.astype("<f4").tobytes()).hexdigest() == row["pcm_sha256"],
                "Input PCM changed")
        try:
            response = model.score(pcm)
        except (ValueError, RuntimeError) as error:
            response = {"label": "unavailable", "margin": None, "error_type": type(error).__name__}
        rows.append({**{key: row[key] for key in ("slot", "group", "language")}, **response,
                     "forwards": model.forwards - before})
    journal = output / "private-observations.json"
    write(journal, rows)
    write(output / "result.json", {"schema": "chipmunk.audio-language-scope-result/184", "status": "complete",
          "freeze_sha256": sha(PUBLIC), "private_observations_sha256": sha(journal), "summary": summarise(rows),
          "new_neural_calls": model.forwards, "holdout_scored": False, "defaults_changed": False,
          "scope": "Turkish synthetic clips and English genuine clips differ in language, source and origin; "
                   "differences cannot be attributed to language alone. No service policy is validated."})


def run():
    output = TRIAL / "authored/output"

    def launch(argv, **kwargs):
        require(argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored", str(output)],
                "Exact monitored diagnostic command required")
        return subprocess.Popen([*argv[:3], MODULE, "worker", str(output)], **kwargs)

    proxy = types.SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired)
    scoped(runtime.run, {"TRIAL": TRIAL, "verify": verify, "subprocess": proxy,
                         "prerequisites": lambda *_: True, "__file__": __file__})("authored")


def readout():
    inputs = verify()
    stage = TRIAL / "authored"
    profile, result = read(stage / "profile.json"), read(stage / "output/result.json")
    journal = stage / "output/private-observations.json"
    rows = read(journal)
    require(profile["child_started"] is True and profile["command"] == "authored"
            and profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0
            and profile["error_types"] == profile["resource_limit_observations"] == []
            and profile["new_neural_calls"] == result["new_neural_calls"]
            and profile["runtime_sha256"] == sha(Path(__file__))
            and profile["resource_journal_sha256"] == sha(stage / "resources.jsonl")
            and profile["worker_receipt_sha256"] == sha(stage / "output/result.json")
            and result["freeze_sha256"] == sha(PUBLIC) and result["private_observations_sha256"] == sha(journal)
            and [(r["slot"], r["group"], r["language"]) for r in rows] ==
                [(r["slot"], r["group"], r["language"]) for r in inputs]
            and result["summary"] == summarise(rows) and result["new_neural_calls"] == sum(r["forwards"] for r in rows)
            and all(type(r["forwards"]) is int and r["forwards"] in {0, 1} for r in rows)
            and result["holdout_scored"] is False and result["defaults_changed"] is False,
            "Exact completed guarded diagnostic required")
    return {**result, "resource_profile_sha256": sha(stage / "profile.json"),
            "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
            "worker_duration_seconds": profile["duration_s"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "worker", "readout"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    require((args.command == "worker") == (args.output is not None), "Only the guarded worker accepts an output")
    if args.command == "worker":
        worker(args.output)
    elif args.command == "readout":
        print(json.dumps(readout(), indent=2, allow_nan=False))
    else:
        {"freeze": freeze, "run": run}[args.command]()
