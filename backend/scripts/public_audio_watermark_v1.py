"""Pinned public AudioSeal research adapter; no provider attribution or AI verdict.

Uses a private reviewed source/dependency directory. The serving environment is
unchanged. Unrestricted upstream pickle/network loaders are never called.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

RATE = 16000
CODE_REVISION = "1496f7662d6aa625602bb0dcd1b4997b63b47609"
MODEL_REVISION = "3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba"
WEIGHTS = {
    "detector_base.pth": "8a78e8a83584113523e161fc599fcab10fd0e94c04d2eb9d2fa1e9ec91ab69d9",
    "generator_base.pth": "7a845b5fbe9364a63a3909d8ab3fe064d13a76ae4c2e983573e08c69b7b51748",
}
GLOBALS = {
    "omegaconf.dictconfig.DictConfig",
    "omegaconf.listconfig.ListConfig",
    "omegaconf.nodes.AnyNode",
    "omegaconf.base.ContainerMetadata",
    "omegaconf.base.Metadata",
    "typing.Any",
    "collections.defaultdict",
    "builtins.list",
    "builtins.dict",
    "builtins.int",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_source(root):
    source = read(root / "source-review.json")
    require(
        source["code_revision"] == CODE_REVISION and source["model_revision"] == MODEL_REVISION,
        "Pinned public revisions required",
    )
    for row in source["files"]:
        require(
            digest(root / "reviewed-source" / row["path"]) == row["sha256"], "Reviewed public source changed"
        )
    dependencies = read(root / "isolated-python-files.json")
    for row in dependencies["files"]:
        require(
            digest(root / "isolated-python" / row["path"]) == row["sha256"],
            "Private dependency source changed",
        )
    for name, sha in WEIGHTS.items():
        require(digest(root / name) == sha, "Pinned public checkpoint changed")


def plain_metadata_dictionary(*unused):
    # Restricted Torch rejects legacy defaultdict SETITEM. Only replace its
    # unused training-metadata default factory; preserve all explicit items.
    return {}


def load_models(root):
    verify_source(root)
    sys.path[:0] = [str(root / "isolated-python"), str(root / "reviewed-source/src")]
    import typing

    import torch
    import yaml
    from audioseal import builder, loader
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode

    def forbidden(*args, **kwargs):
        raise AssertionError("No upstream unrestricted pickle, download or provider call permitted")

    loader._safe_load_checkpoint = forbidden
    loader.load_model_checkpoint = forbidden
    allow = [
        DictConfig,
        ListConfig,
        AnyNode,
        ContainerMetadata,
        Metadata,
        typing.Any,
        (plain_metadata_dictionary, "collections.defaultdict"),
        list,
        dict,
        int,
    ]
    models, receipts = {}, {}
    for kind, name, card in [
        ("detector", "detector_base.pth", "audioseal_detector_16bits"),
        ("generator", "generator_base.pth", "audioseal_wm_16bits"),
    ]:
        path = root / name
        require(
            set(torch.serialization.get_unsafe_globals_in_checkpoint(path)) == GLOBALS,
            "Checkpoint data-global inventory changed",
        )
        with torch.serialization.safe_globals(allow):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        require(set(checkpoint) == {"xp.cfg", "model"}, "Expected data checkpoint required")
        state = checkpoint["model"]
        require(
            all(
                isinstance(k, str)
                and isinstance(v, torch.Tensor)
                and v.dtype == torch.float32
                and torch.isfinite(v).all()
                for k, v in state.items()
            ),
            "Finite float32 state tensors required",
        )
        config = yaml.safe_load((root / "reviewed-source/src/audioseal/cards" / (card + ".yaml")).read_text())
        config["seanet"]["norm"] = "none"
        seanet = builder.SEANetConfig(**config["seanet"])
        if kind == "detector":
            model = builder.create_detector(
                builder.AudioSealDetectorConfig(nbits=16, seanet=seanet), device="cpu", dtype=torch.float32
            )
        else:
            model = builder.create_generator(
                builder.AudioSealWMConfig(
                    nbits=16, seanet=seanet, decoder=builder.DecoderConfig(**config["decoder"])
                ),
                device="cpu",
                dtype=torch.float32,
            )
        state = loader.convert_state_dict_for_scriptable_model(state)
        loader.load_state_dict(model, state)
        expected = {}
        for key, value in state.items():
            if key.endswith(".weight_g"):
                prefix = key[: -len(".weight_g")]
                expected[prefix + ".weight"] = torch._weight_norm(state[prefix + ".weight_v"], value, 0)
            elif not key.endswith(".weight_v"):
                expected[key] = value
        actual = model.state_dict()
        require(
            set(actual) == set(expected) and all(torch.equal(v, expected[k]) for k, v in actual.items()),
            "Every loaded tensor must equal its source or native weight-normalization conversion",
        )
        model.eval().requires_grad_(False)
        models[kind] = model
        receipts[kind] = {
            "checkpoint_sha256": WEIGHTS[name],
            "source_state_tensors": len(state),
            "loaded_state_tensors": len(actual),
            "all_loaded_tensors_verified": True,
            "parameters": sum(p.numel() for p in model.parameters()),
            "weights_only": True,
            "unused_training_metadata_resolved": False,
            "unrestricted_pickle_used": False,
        }
    return models, receipts


def summarize(probabilities, *, sample_rate=RATE):
    require(
        sample_rate == RATE
        and probabilities.dtype == np.float32
        and probabilities.ndim == 1
        and len(probabilities) > 0
        and np.isfinite(probabilities).all()
        and probabilities.min() >= 0
        and probabilities.max() <= 1,
        "Finite native 16kHz sample responses required",
    )
    bins, intervals = [], []
    for start in range(0, len(probabilities), RATE // 4):
        end = min(start + RATE // 4, len(probabilities))
        values = probabilities[start:end]
        fraction = float(np.count_nonzero(values > 0.5) / len(values))
        positive = fraction > 0.5
        bins.append(
            {
                "start_sample": start,
                "end_sample": end,
                "start_s": start / RATE,
                "end_s": end / RATE,
                "mean_watermark_response": float(values.mean(dtype=np.float64)),
                "positive_sample_fraction": fraction,
                "selected": positive,
            }
        )
        if positive:
            if intervals and intervals[-1]["end_sample"] == start:
                intervals[-1].update(end_sample=end, end_s=end / RATE)
            else:
                intervals.append(
                    {"start_sample": start, "end_sample": end, "start_s": start / RATE, "end_s": end / RATE}
                )
    return {
        "sample_rate": RATE,
        "samples": len(probabilities),
        "bins": bins,
        "intervals": intervals,
        "positive_sample_fraction": float(np.count_nonzero(probabilities > 0.5) / len(probabilities)),
        "score_meaning": "Fraction of native sample responses strictly above0.5; not AI-authorship confidence.",
        "provider_authenticated": False,
        "used_in_confidence": False,
        "early_stop_eligible": False,
    }
