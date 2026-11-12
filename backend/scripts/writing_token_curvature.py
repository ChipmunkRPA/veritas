"""Measure local token predictability on exact, paired writing revisions."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic

import numpy as np

SCHEMA = "chipmunk.writing-token-curvature/1"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
RUNTIME = {"mlx": "0.32.4", "mlx-lm": "0.31.3", "transformers": "5.22.1"}
MODEL_ASSETS = {
    "config.json": "0e8c8aa86468aba09c9d32157ff4bc2301c7e6c50e4398960425b2ea71e66f77",
    "generation_config.json": "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model.safetensors": "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
METRICS = ("native_context", "mean_log_probability", "curvature")


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def target_token_indices(text: str, offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    left, right = span
    if not 0 <= left < right <= len(text):
        raise ValueError("Invalid focus source coordinates")
    required = {i for i in range(left, right) if not text[i].isspace()}
    covered = set()
    indices = []
    for i, (start, end) in enumerate(offsets):
        if not 0 <= start <= end <= len(text):
            raise ValueError("Invalid tokenizer source coordinates")
        positions = {j for j in range(start, end) if not text[j].isspace()}
        if positions.intersection(required):
            if i == 0 or not positions.issubset(required):
                raise ValueError("Scored token includes unpredicted or unrelated source text")
            indices.append(i)
            covered.update(positions)
    if not indices or covered != required:
        raise ValueError("The focus does not have complete causal token coverage")
    return indices


def conditional_moments(logits: np.ndarray, labels: list[int]) -> list[tuple[float, float, float]]:
    """For each position, return observed log probability, expected log probability and variance."""
    if logits.ndim != 2 or logits.shape[0] != len(labels) or not labels or logits.shape[1] < 2:
        raise ValueError("Expected aligned, nonempty vocabulary logits and observed token IDs")
    if any(type(label) is not int or not 0 <= label < logits.shape[1] for label in labels):
        raise ValueError("Observed token is outside the model vocabulary")
    result = []
    for start in range(0, len(labels), 8):
        scores = np.asarray(logits[start:start + 8], dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("The scorer returned nonfinite logits")
        shifted = scores - scores.max(axis=1, keepdims=True)
        weights = np.exp(shifted)
        normalizer = weights.sum(axis=1, keepdims=True)
        probability = weights / normalizer
        log_probability = shifted - np.log(normalizer)
        mean = (probability * log_probability).sum(axis=1)
        var = (probability * (log_probability - mean[:, None]) ** 2).sum(axis=1)
        observed = log_probability[np.arange(len(scores)), labels[start:start + len(scores)]]
        result.extend(zip(observed.tolist(), mean.tolist(), var.tolist(), strict=True))
    return result


def focus_values(moments: list[tuple[float, float, float]]) -> dict[str, float]:
    observed, expected, variance = zip(*moments, strict=True)
    total_variance = math.fsum(variance)
    if not all(math.isfinite(value) for triple in moments for value in triple) or total_variance <= 0 or min(variance) < 0:
        raise ValueError("Focus moments do not give a valid conditional variance")
    return {
        "mean_log_probability": math.fsum(observed) / len(observed),
        "curvature": math.fsum(x - mean for x, mean in zip(observed, expected, strict=True)) / math.sqrt(total_variance),
    }


def freeze(directory: Path, model: Path) -> None:
    from scripts.writing_shared_anchor_ranking import source_windows
    from scripts.writing_two_sided_context import BASELINE_ASSETS, read_cases

    inputs = ("source.json", "protocol.json", "private-holdout.jsonl", "generation-freeze.json", "admission.json")
    verified_model(model)
    geometry = {}
    for case in read_cases(directory / "private-holdout.jsonl"):
        if case["status"] != "constructed":
            continue
        windows = source_windows(case["text"])[:2]
        if len(windows) != 2:
            raise ValueError("Admitted case is outside the frozen two-focus scope")
        focuses = [list(window[1]) for window in windows]
        if any(span not in focuses for span in case["ai_spans"] + case["ordinary_spans"]):
            raise ValueError("A labelled edit is outside the exact scored foci")
        geometry[case["case"]] = {"windows": windows, "focuses": focuses}
    write(directory / "frozen.json", {
        "schema": SCHEMA, "inputs": {name: file_hash(directory / name) for name in inputs},
        "scorer": file_hash(Path(__file__)), "baseline_assets": BASELINE_ASSETS,
        "model_revision": REVISION, "model_assets": MODEL_ASSETS, "runtime": RUNTIME,
        "quantization": {"mode": "affine", "group_size": 64, "bits": 4}, "geometry": geometry,
    })


def verified_model(model: Path) -> None:
    if model.name != REVISION or any(file_hash(model / name) != value for name, value in MODEL_ASSETS.items()):
        raise ValueError("Local causal snapshot differs from the frozen model")


def locked_inputs(directory: Path) -> tuple[dict, list[dict]]:
    frozen = read(directory / "frozen.json")
    if (frozen["scorer"] != file_hash(Path(__file__))
            or any(file_hash(directory / name) != value for name, value in frozen["inputs"].items())):
        raise ValueError("Frozen study code or inputs changed")
    cases = [json.loads(line) for line in (directory / "private-holdout.jsonl").read_text().splitlines()]
    return frozen, [case for case in cases if case["status"] == "constructed"]


def causal_run(directory: Path, snapshot: Path) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from transformers import AutoTokenizer

    if os.getenv("HF_HUB_OFFLINE") != "1" or os.getenv("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("Local causal evaluation requires offline model settings")
    if any(importlib.metadata.version(package) != version for package, version in RUNTIME.items()):
        raise ValueError("Local causal runtime differs from the frozen packages")
    frozen, cases = locked_inputs(directory)
    verified_model(snapshot)
    mx.set_default_device(mx.gpu)
    model, _ = load(str(snapshot), tokenizer_config={"trust_remote_code": False})
    nn.quantize(model, **frozen["quantization"])
    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    result = {}
    logit_samples = {}
    sample_labels = {}
    start = monotonic()
    for case in cases:
        body = case["text"]
        focuses = frozen["geometry"][case["case"]]["focuses"]
        encoded = tokenizer(body, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        tokens, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if len(tokens) != len(offsets) or not 1 < len(tokens) <= 512 or text_hash(body) != case["text_sha256"]:
            raise ValueError("Input does not meet complete native-token and source requirements")
        indices = [target_token_indices(body, offsets, span) for span in focuses]
        combined = sorted(set().union(*map(set, indices)))
        logits = model(mx.array([tokens]))[0]
        target_logits = np.asarray(logits[mx.array([i - 1 for i in combined])].astype(mx.float32))
        samples = conditional_moments(target_logits, [tokens[i] for i in combined])
        arm = case["arm"]
        if arm not in logit_samples:
            logit_samples[arm] = target_logits[0].copy()
            sample_labels[arm] = {"case": case["case"], "token_index": combined[0], "observed_id": tokens[combined[0]]}
        table = dict(zip(combined, samples, strict=True))
        result[case["case"]] = {
            "input_sha256": text_hash(body), "token_ids": tokens, "offsets": offsets,
            "focuses": [{"span": span, "token_indices": focus,
                         "moments": [table[i] for i in focus], "values": focus_values([table[i] for i in focus])}
                        for span, focus in zip(focuses, indices, strict=True)],
        }
        print(case["case"], len(tokens), [len(focus) for focus in indices], flush=True)
    np.savez(directory / "private-logit-samples.npz", **logit_samples)
    write(directory / "private-causal.json", {"cases": result, "forwards": len(result),
          "elapsed_seconds": round(monotonic() - start, 3), "audit_logit_labels": sample_labels})


def baseline_run(directory: Path, snapshot: Path) -> None:
    from scripts import writing_two_sided_context as native

    frozen, cases = locked_inputs(directory)
    windows = {case["case"]: [native.native_input(case["text"], spans) for spans in frozen["geometry"][case["case"]]["windows"]]
               for case in cases}
    classifier = native.open_classifier(snapshot)
    calls = [0]
    handle = classifier.model.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    try:
        inputs = list(dict.fromkeys(value for pair in windows.values() for value in pair))
        measured = native.native_responses(classifier, inputs, lambda: None)
    finally:
        handle.remove()
    result = {case: [{"response": measured[text][0], "token_sha256": measured[text][1],
                      "input_sha256": text_hash(text)} for text in pair] for case, pair in windows.items()}
    write(directory / "private-baseline.json", {"cases": result, "forwards": calls[0], "unique_inputs": len(inputs)})


def compare(a: float, b: float) -> str:
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Comparisons require finite responses")
    return "higher" if a > b else "lower" if a < b else "tied"


def aggregate(directory: Path) -> None:
    frozen, cases = locked_inputs(directory)
    causal, baseline = (read(directory / f"private-{name}.json") for name in ("causal", "baseline"))
    target_indexes = {
        case["group"]: frozen["geometry"][case["case"]]["focuses"].index(case["ordinary_spans"][0])
        for case in cases if case["ordinary_spans"]
    }
    details = defaultdict(dict)
    for case in cases:
        group, key, arm = case["group"], case["case"], case["arm"]
        spans = frozen["geometry"][key]["focuses"]
        focus = target_indexes[group]
        target = case["ai_spans"] or case["ordinary_spans"]
        if target and target[0] != spans[focus]:
            raise ValueError("A paired revision moved to a different focus position")
        values = [{**row["values"], "native_context": native["response"]} for row, native in
                  zip(causal["cases"][key]["focuses"], baseline["cases"][key], strict=True)]
        details[group][arm] = {metric: (values[focus][metric], values[1-focus][metric]) for metric in METRICS}
    results = defaultdict(lambda: defaultdict(Counter))
    public = defaultdict(dict)
    for group, arms in details.items():
        for arm, metrics in arms.items():
            outcomes = {}
            for metric, (focus, other) in metrics.items():
                outcome = {"target_rank": compare(focus, other)}
                if arm != "original":
                    outcome["against_original"] = compare(focus, arms["original"][metric][0])
                if arm not in {"original", "human_sentence"}:
                    outcome["against_genuine_edit"] = compare(focus, arms["human_sentence"][metric][0])
                    outcome["against_both_human_controls"] = compare(
                        focus, max(arms["human_sentence"][metric][0], arms["original"][metric][0])
                    )
                results[arm][metric].update({f"{kind}_{order}": 1 for kind, order in outcome.items()})
                outcomes[metric] = outcome
            public[group][arm] = outcomes
    write(directory / "aggregate.json", {
        "schema": SCHEMA, "counts": results, "source_outcomes": public, "assessed": len(cases),
        "causal": {key: causal[key] for key in ("forwards", "elapsed_seconds")},
        "baseline": {key: value for key, value in baseline.items() if key != "cases"},
        "commitments": {key: file_hash(directory / name) for key, name in
                        (("freeze", "frozen.json"), ("causal", "private-causal.json"), ("baseline", "private-baseline.json"),
                         ("audit_logits", "private-logit-samples.npz"))},
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "causal", "baseline", "aggregate"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.stage == "aggregate":
        aggregate(args.directory)
    elif args.snapshot is None:
        parser.error("This stage requires a pinned local model snapshot")
    else:
        {"freeze": freeze, "causal": causal_run, "baseline": baseline_run}[args.stage](args.directory, args.snapshot)


if __name__ == "__main__":
    main()
