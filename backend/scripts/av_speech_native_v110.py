"""Guarded CPU speech-encoder contract for native-clock mouth conditioning.

Research features only. No authenticity prediction, fitting or serving change.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
ASSETS = BASE / "lipsync-v7/assets"
TRIAL = BASE / "av-speech-native-v110"
PANEL = ROOT / "docs/evaluation-panels"
PUBLIC = PANEL / "av-speech-native-freeze-v110.json"
CODE = (
    "backend/scripts/av_speech_native_v110.py",
    "backend/scripts/av_speech_runtime_v110.py",
    "backend/tests/test_av_speech_native_v110.py",
    "docs/av-speech-native-v110-method.md",
    "backend/scripts/av_conditioning_clock_v109.py",
    "backend/scripts/musetalk_benchmark_v7.py",
    "backend/scripts/fetch_mouth_generator_v7.py",
    "backend/scripts/video_appearance_native_v108.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/local_probe_limits_v24.py",
    "backend/scripts/audio_tree_sampler_v64.py",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def fingerprint(array):
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def validate_states(states, last, layer_inputs):
    """Check the native API against four independently captured layer inputs."""
    import numpy as np

    values = list(states)
    require(
        len(values) == 5 and len(layer_inputs) == 4,
        "Five native states and four captured layer inputs required",
    )
    require(
        all(
            isinstance(x, np.ndarray)
            and x.shape == (1, 1500, 384)
            and x.dtype == np.float32
            and np.isfinite(x).all()
            for x in values + list(layer_inputs) + [last]
        ),
        "Complete finite CPUfloat32 encoder arrays required",
    )
    require(
        all(np.array_equal(a, b) for a, b in zip(values[:4], layer_inputs, strict=True))
        and np.array_equal(values[-1], last),
        "Native state capture/order differs from layer-input/final readout",
    )
    return np.ascontiguousarray(np.stack([x[0] for x in values], axis=1))


def validate_pcm(pcm):
    import numpy as np

    require(
        isinstance(pcm, np.ndarray)
        and pcm.ndim == 1
        and pcm.dtype == np.float32
        and 640 <= len(pcm) <= 480000
        and np.isfinite(pcm).all()
        and np.max(np.abs(pcm)) <= 1.0,
        "Finite mono16k float32 source with fixed extent required",
    )


class NativeEncoder:
    def __init__(self):
        import torch
        from safetensors import safe_open
        from transformers import WhisperConfig, WhisperFeatureExtractor
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        verify()
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        config = WhisperConfig.from_dict(read(ASSETS / "speech/config.json"))
        config._attn_implementation = "eager"
        self.encoder = WhisperEncoder(config).to(device="cpu", dtype=torch.float32)
        with safe_open(ASSETS / "speech/model.safetensors", framework="pt", device="cpu") as checkpoint:
            keys = list(checkpoint.keys())
            require(
                all(k.startswith(("model.encoder.", "model.decoder.", "proj_out.")) for k in keys),
                "Unexpected publisher tensor family",
            )
            state = {
                k.removeprefix("model.encoder."): checkpoint.get_tensor(k)
                for k in keys
                if k.startswith("model.encoder.")
            }
        self.encoder.load_state_dict(state, strict=True)
        require(
            all(torch.equal(v, self.encoder.state_dict()[k]) for k, v in state.items()),
            "Strict encoder load changed publisher values",
        )
        self.encoder.eval().requires_grad_(False)
        del state
        self.extractor = WhisperFeatureExtractor.from_pretrained(ASSETS / "speech", local_files_only=True)
        self.forward_count = 0

    def encode(self, pcm):
        import numpy as np
        import torch

        validate_pcm(pcm)
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
                "Exact finite fixed-length CPU log-mel input required",
            )
            captured = []

            def before(layer, args):
                captured.append(args[0].detach().cpu().numpy().copy())

            handles = [layer.register_forward_pre_hook(before) for layer in self.encoder.layers]
            try:
                self.forward_count += 1
                result = self.encoder(mel, output_hidden_states=True)
            finally:
                for handle in handles:
                    handle.remove()
            states = [x.detach().cpu().numpy().copy() for x in result.hidden_states]
            hidden = validate_states(states, result.last_hidden_state.detach().cpu().numpy(), captured)
        require(
            hidden.shape == (1500, 5, 384) and hidden.dtype == np.float32, "Native five-state packing changed"
        )
        return hidden, {
            "mel_sha256": fingerprint(mel.numpy()),
            "full_hidden_sha256": fingerprint(hidden),
            "layer_readout_matches": True,
        }


def identity():
    from scripts.av_conditioning_clock_v109 import graph_sources
    from scripts.fetch_mouth_generator_v7 import verify as verify_asset

    clock = read(PANEL / "av-conditioning-clock-freeze-v109.json")
    require(
        all(sha(ROOT / p) == h for p, h in clock["code_sha256"].items())
        and graph_sources() == clock["library_source"],
        "Reviewed clock/library implementation changed",
    )
    manifest = read(PANEL / "mouth-v7-generator-assets.json")
    speech = [r for r in manifest["assets"] if r["role"] == "speech"]
    require(
        len(speech) == 3 and all(r["license"] == "apache-2.0" for r in speech),
        "Exactly three permitted pinned speech assets required",
    )
    actual = [verify_asset(ASSETS / r["path"], r) for r in speech]
    return {
        "schema": "chipmunk.av-speech-native-freeze/110",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "speech_assets": actual,
        "clock_freeze_sha256": sha(PANEL / "av-conditioning-clock-freeze-v109.json"),
        "clock_result_sha256": sha(PANEL / "av-conditioning-clock-result-v109.json"),
        "clock_plan_sha256": sha(BASE / "av-conditioning-clock-v109/preparation/clock-plan.jsonl"),
        "library_source": graph_sources(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "torch", "transformers", "safetensors", "psutil")
        },
        "authored_forwards": 3,
        "source_files_used": 0,
        "scope": "Authored CPU speech encoder/state/routing/repeat and suffix-context contract. No source inference, generation, detector score, accuracy or customer-confidence qualification.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "authored_forwards_planned": 3, "new_neural_calls": 0}), flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Published unchanged native speech contract required",
    )


def authored(output):
    import numpy as np
    import torch

    from scripts.av_conditioning_clock_v109 import frame_plan, select_chunk
    from scripts.musetalk_benchmark_v7 import audio_chunks

    native = NativeEncoder()
    t = np.arange(32000, dtype=np.float64) / 16000
    original = (0.2 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    first, proof = native.encode(original)
    repeat, repeated_proof = native.encode(original.copy())
    require(
        np.array_equal(first, repeat) and proof == repeated_proof, "Native authored repeat changed bitwise"
    )
    legacy = audio_chunks(torch.from_numpy(first[:100].copy()), 50).numpy()
    routed = np.stack(
        [select_chunk(first, frame_plan(i, f"{i}/25", "0", "0", len(original), "0")) for i in range(50)]
    )
    require(np.array_equal(legacy, routed), "Native 25-Hz compatibility routing changed")
    suffix = np.zeros(64000, np.float32)
    suffix[: len(original)] = original
    tail = np.arange(25600, dtype=np.float64) / 16000
    suffix[38400:] = (0.02 * np.sin(2 * np.pi * 733 * tail)).astype(np.float32)
    require(np.array_equal(original, suffix[: len(original)]), "Authored prefix changed")
    counterfactual, suffix_proof = native.encode(suffix)
    deltas = []
    for layer in range(5):
        difference = counterfactual[:100, layer] - first[:100, layer]
        deltas.append(
            {
                "layer": layer,
                "different_values": int(np.count_nonzero(difference)),
                "maximum_abs_difference": float(np.max(np.abs(difference))),
            }
        )
    result = {
        "schema": "chipmunk.av-speech-native-authored/110",
        "status": "complete",
        "native_forward_count": native.forward_count,
        "hidden_shape": list(first.shape),
        "native_layer_readout_matches": True,
        "native_repeat_bitwise": True,
        "legacy_25hz_chunks_equal": True,
        "routed_canonical_frames": 50,
        "authored_prefix_samples": len(original),
        "counterfactual_tail_start_sample": 38400,
        "prefix_deltas_after_suffix_change": deltas,
        "original_proof": proof,
        "suffix_proof": suffix_proof,
        "new_neural_calls": native.forward_count,
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "scope": "Authored execution/routing only. Suffix probe measures context dependence, not AI evidence, source timing, achieved mouth synchronization or detector accuracy.",
    }
    require(native.forward_count == 3, "Exactly the predeclared authored forwards required")
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "monitor", "authored"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        require(args.output is None, "Freeze has no stage output")
        freeze()
    else:
        verify()
        require(
            args.output is not None and args.output.resolve().is_relative_to(TRIAL.resolve()),
            "Private guarded output required",
        )
        args.output.mkdir(exist_ok=False)
        if args.command == "monitor":
            from scripts.video_appearance_native_v108 import monitor

            monitor(args.output)
        else:
            authored(args.output)
