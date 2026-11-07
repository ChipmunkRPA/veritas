"""Synthetic graph parity against inspected upstream mask/filterbank code.

No real research recording, classifier fitting or serving inference is run.
This is a restricted upstream reference, not an installed full Asteroid runtime:
original masker/normalization definitions and native filterbanks are loaded from
publisher-checksummed wheels. Inactive tracing decorators are identity wrappers;
optional training, publishing and audio-file loading code is never imported.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from scripts import audio_separation_v14 as local

WHEELS = {
    "asteroid-0.7.0-py3-none-any.whl": "ea97a24901d9d9851b4a594171bd7c6dd900fee2c132b9ce045aa09926d489c7",
    "asteroid_filterbanks-0.4.0-py3-none-any.whl": "4932ac8b6acc6e08fb87cbe8ece84215b5a74eee284fe83acf3540a72a02eaf5",
}


def definitions(wheel, name, selected, namespace):
    with zipfile.ZipFile(wheel) as archive:
        source = archive.read(name).decode()
    module = ast.parse(source)
    nodes = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected]
    if {node.name for node in nodes} != set(selected):
        raise ValueError("Expected inspected upstream definitions absent")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name, "exec"), namespace)
    return hashlib.sha256(source.encode()).hexdigest()


def reference(package_root, state, sample_rate):
    for name, expected in WHEELS.items():
        if local.digest(package_root / name) != expected:
            raise ValueError("Pinned upstream package changed")
    namespace = {"torch": torch, "nn": nn, "List": list, "EPS": 1e-8, "script_if_tracing": lambda f: f}
    asteroid = package_root / "asteroid-0.7.0-py3-none-any.whl"
    hashes = {"norms": definitions(asteroid, "asteroid/masknn/norms.py",
        {"z_norm", "_glob_norm", "_LayerNorm", "GlobLN"}, namespace)}
    namespace.update(norms=SimpleNamespace(get=lambda name: {"gLN": namespace["GlobLN"]}[name]),
                     activations=SimpleNamespace(get=lambda name: {"relu": nn.ReLU}[name]),
                     has_arg=lambda cls, name: name in inspect.signature(cls).parameters)
    hashes["convolutional"] = definitions(asteroid, "asteroid/masknn/convolutional.py",
                                          {"Conv1DBlock", "TDConvNet"}, namespace)
    sys.path.insert(0, str(package_root / "asteroid_filterbanks-0.4.0-py3-none-any.whl"))
    try:
        from asteroid_filterbanks import make_enc_dec
    finally:
        sys.path.pop(0)
    encoder, decoder = make_enc_dec("FreeFB", n_filters=512, kernel_size=32, stride=16, sample_rate=sample_rate)
    masker = namespace["TDConvNet"](in_chan=512, n_src=2, out_chan=512, n_blocks=8, n_repeats=3,
                                    bn_chan=128, hid_chan=512, skip_chan=128, conv_kernel_size=3,
                                    norm_type="gLN", mask_act="relu", causal=False)

    class Reference(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.masker, self.decoder = encoder, masker, decoder

        def forward(self, pcm):
            encoded = self.encoder(pcm)
            return self.decoder(self.masker(encoded) * encoded[:, None])

    result = Reference().eval().requires_grad_(False)
    result.load_state_dict(state, strict=True)
    return result, hashes


def audit(weights_root, package_root, output):
    if output.exists():
        raise ValueError("New audit path required")
    torch.set_num_threads(4)
    model = local.load(weights_root)
    release = torch.load(weights_root / "pytorch_model.bin", map_location="cpu", weights_only=True)
    upstream, source_hashes = reference(package_root, release["state_dict"], 8000)
    rate_metadata_only, _ = reference(package_root, release["state_dict"], 16000)
    rng = np.random.default_rng(20261107)
    clock = np.arange(64000, dtype=np.float32) / 16000
    samples = {
        "two_sines": np.asarray(.12 * np.sin(2 * np.pi * 193 * clock) + .08 * np.sin(2 * np.pi * 347 * clock), np.float32),
        "seeded_noise": np.asarray(rng.standard_normal(64000) * .05, np.float32),
        "impulse": np.r_[np.float32(.3), np.zeros(63999, np.float32)],
    }
    probes = []
    for name, pcm in samples.items():
        source = pcm.copy()
        tensor = torch.from_numpy(pcm.copy())[None, None]
        start = time.monotonic()
        actual = local.separate(pcm, model)
        with torch.inference_mode():
            expected = upstream(tensor).numpy()[0]
            metadata_changed = rate_metadata_only(tensor).numpy()[0]
        probes.append({"id": name, "input_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
                       "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                       "output_shape": list(actual.shape), "source_unchanged": np.array_equal(pcm, source),
                       "upstream_max_absolute_difference": float(np.max(np.abs(expected - actual))),
                       "sample_rate_metadata_max_difference": float(np.max(np.abs(expected - metadata_changed))),
                       "all_outputs_exact": np.array_equal(actual, expected) and np.array_equal(expected, metadata_changed),
                       "seconds_three_forwards": time.monotonic() - start})
    repeat = local.separate(samples["two_sines"], model)
    passed = all(r["all_outputs_exact"] and r["source_unchanged"] and r["output_shape"] == [2, 64000] for r in probes)
    repeated_exact = hashlib.sha256(repeat.tobytes()).hexdigest() == probes[0]["output_sha256"]
    result = {"schema": "chipmunk.audio-separation-structural-audit/14", "spec": local.SPEC,
              "model_args_retained": release["model_args"], "strict_tensor_load": True,
              "all345_tensor_entries_loaded": True, "unsafe_pickle_globals": [],
              "upstream_packages": WHEELS, "upstream_code_sha256": source_hashes,
              "local_code_sha256": local.digest(Path(local.__file__)), "audit_code_sha256": local.digest(Path(__file__)),
              "synthetic_probes": probes, "repeated_synthetic_exact": repeated_exact, "passed": passed and repeated_exact,
              "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "dtype": "float32", "threads": 4},
              "project_recordings_scored": 0, "fitting_started": False, "serving_changed": False,
              "scope": "Inspected upstream mask/filterbank graph parity on synthetic inputs only; no physical separation quality, AI detection gain, license-version resolution or outside review established."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": result["passed"], "synthetic_probes": len(probes),
                      "maximum_difference": max(r["upstream_max_absolute_difference"] for r in probes),
                      "project_recordings_scored": 0}), flush=True)
    if not result["passed"]:
        raise ValueError("Synthetic upstream graph/clock/repeat audit failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights_root", "package_root", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    audit(args.weights_root, args.package_root, args.output)
