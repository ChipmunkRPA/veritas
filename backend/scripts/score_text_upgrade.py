"""Score a frozen local text manifest with one detector per process."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from time import monotonic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.detectors.text.classifier import (
    DesklibAIDetectionModel,
    DesklibClassifier,
    FourClassTextClassifier,
    _validated_logits,
)
from app.detectors.text.coverage import classify_complete
from app.detectors.text.extract import from_string
from app.detectors.text.pipeline import PRIMARY_CHUNK_MAX_LEN, chunk_average, chunk_words, chunks_of
from app.detectors.text.segment import segment
from app.engine.registry import get_registry


def strict_load(cls, path: Path, **kwargs):
    model, info = cls.from_pretrained(
        path, local_files_only=True, use_safetensors=True, output_loading_info=True, **kwargs
    )
    errors = {key: info[key] for key in ("missing_keys", "mismatched_keys", "error_msgs", "unexpected_keys") if info.get(key)}
    if errors:
        raise ValueError(f"Checkpoint did not load exactly: {errors}")
    return model


def snapshot(model_id: str) -> Path:
    from app.detectors.image.adapters.loaders import _file

    entry = get_registry().manifest.model(model_id)
    return _file(entry, "config.json").parent


def sequence_predictor(name: str, path: Path, device: str, batch: int,
                       max_len: int = PRIMARY_CHUNK_MAX_LEN) -> Callable[[list[str]], list[float]]:
    if name == "tropa":
        classifier = FourClassTextClassifier.from_snapshot(path, device)
        return lambda texts: classifier.predict(texts, max_len=max_len, batch=batch)

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if name == "desklib":
        model = strict_load(DesklibAIDetectionModel, path).to(device).eval()
        classifier = DesklibClassifier.from_parts(tokenizer, model, str(path), device)
        return lambda texts: classifier.predict(texts, max_len=max_len, batch=batch)

    metadata = json.loads((path / "detector-config.json").read_text())
    if metadata["label_mapping"] != {"human": 0, "ai": 1}:
        raise ValueError("Unexpected candidate label order")
    model = strict_load(
        AutoModelForSequenceClassification, path, attn_implementation="sdpa", dtype=torch.float32
    ).to(device).eval()
    temperature = float(metadata["temperature"])

    def score(encoded):
        ids = encoded["input_ids"]
        logits = _validated_logits(model(ids, attention_mask=encoded["attention_mask"]).logits, len(ids), width=2)
        return (logits / temperature).softmax(dim=-1)[:, 1]

    @torch.inference_mode()
    def predict(texts):
        scores, _ = classify_complete(
            texts, tokenizer, device, score, max_len=max_len, batch=batch, check=lambda _: None
        )
        return scores

    return predict


def sequence_scores(cases: list[dict], predict, batch: int):
    for start in range(0, len(cases), batch):
        batch_cases = cases[start:start + batch]
        chunk_texts = []
        inputs = []
        for case in batch_cases:
            sentences = segment(from_string(case["text"]))
            chunks = chunks_of(sentences)
            weights = chunk_words(chunks, sentences)
            inputs.append((len(chunk_texts), len(chunks), weights))
            chunk_texts.extend(text for text, _, _ in chunks)
        scores = predict(chunk_texts)
        for case, (offset, count, weights) in zip(batch_cases, inputs, strict=True):
            yield case, chunk_average(scores[offset:offset + count], weights)


def ratio_scores(cases: list[dict], device: str):
    from app.detectors.text.adapters.learned import BinocularsDetector
    from app.detectors.text.adapters.loaders import load_causal_lm
    from app.detectors.text.binoculars import Binoculars, document_score

    dtype = "float16" if device == "mps" else "float32"
    tokenizer, observer = load_causal_lm(BinocularsDetector.OBSERVER, device, dtype).obj
    other_tokenizer, performer = load_causal_lm(BinocularsDetector.PERFORMER, device, dtype).obj
    if tokenizer.get_vocab() != other_tokenizer.get_vocab():
        raise ValueError("The paired tokenizers disagree on vocabulary IDs")
    model = Binoculars.from_parts(tokenizer, observer, performer, device)
    for case in cases:
        tokens = model.token_scores(case["text"])
        other = other_tokenizer(case["text"], add_special_tokens=False, return_offsets_mapping=True)
        if tokens.source_token_ids != other["input_ids"] or tokens.offsets != [tuple(o) for o in other["offset_mapping"]]:
            raise ValueError("The paired tokenizers disagree on the source text")
        raw, _ = document_score(tokens)
        yield case, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("desklib", "tropa", "modernbert", "binoculars"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("Metal inference is unavailable")
    payload = args.manifest.read_bytes()
    cases = json.loads(payload)["cases"][:args.limit]
    start = monotonic()
    if args.model == "binoculars":
        results = ratio_scores(cases, args.device)
    else:
        path = args.checkpoint
        if args.model == "desklib":
            path = snapshot("hf.desklib.ai_text_detector_v1_01")
        if path is None:
            parser.error("This model requires a local checkpoint")
        results = sequence_scores(cases, sequence_predictor(args.model, path, args.device, args.batch), args.batch)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps({"model": args.model, "manifest_sha256": hashlib.sha256(payload).hexdigest(),
                                 "device": args.device, "count": len(cases)}) + "\n")
        for i, (case, raw) in enumerate(results, 1):
            output.write(json.dumps({"id": case["id"], "text_sha256": case["text_sha256"], "raw": raw}) + "\n")
            if i % 20 == 0 or i == len(cases):
                output.flush()
                print(f"{args.model}: {i}/{len(cases)} in {monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
