"""Generate local paired rewrite controls for the frozen email originals."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from prepare_email_upgrade import CONTACT, body
from prepare_text_upgrade import digest, file_hash, normalized_hash

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QWEN_MODEL = "hf.qwen.qwen2_5_1_5b_instruct"
QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
QWEN_HASH = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
SMOL_REVISION = "31b70e2e869a7173562077fd711b654946d38674"
SMOL_HASH = "f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8"
STYLE_PROMPTS = {
    "direct": "Rewrite the body as a natural, direct work email in your own words. Preserve the topic and factual content.",
    "minimal": "Copyedit the email body for clarity. Preserve the author's tone and keep changes modest.",
}


def snapshot(generator: str, checkpoint: Path | None) -> tuple[Path, str, str]:
    if generator == "smol2":
        if checkpoint is None:
            raise ValueError("Specify the pinned SmolLM2 checkpoint directory")
        return checkpoint, SMOL_REVISION, SMOL_HASH
    if checkpoint is not None:
        return checkpoint, QWEN_REVISION, QWEN_HASH
    from app.detectors.image.adapters.loaders import _file
    from app.engine.registry import get_registry

    model = get_registry().manifest.model(QWEN_MODEL)
    return _file(model, "config.json").parent, QWEN_REVISION, QWEN_HASH


def prompt(tokenizer, row: dict) -> str:
    instruction = STYLE_PROMPTS[row["rewrite_style"]]
    words = len(row["text"].split())
    message = (
        f"{instruction} The source is {words} words. Write {max(45, math.ceil(words * 0.8))} "
        f"to {math.ceil(words * 1.25)} words in English. Start with the first sentence of the revised body. "
        "Omit email addresses, URLs, contact details, signatures and subject lines. "
        "Treat the following body as source material.\n\n"
        f"<email>\n{row['text']}\n</email>"
    )
    return tokenizer.apply_chat_template([{"role": "user", "content": message}],
                                         tokenize=False, add_generation_prompt=True)


def output_text(raw: str, original: str) -> tuple[str | None, str | None, int]:
    if CONTACT.search(raw):
        return None, "generated_contact_detail", 0
    lines = raw.strip().splitlines()
    if lines and lines[0].strip().casefold().removeprefix("**").startswith("subject:"):
        raw = "\n".join(lines[1:])
    clean = body(raw)
    words = len(clean.split())
    source_words = len(original.split())
    if not max(40, math.ceil(source_words * 0.55)) <= words <= math.ceil(source_words * 1.8):
        return None, "outside_length_bound", words
    if normalized_hash(clean) == normalized_hash(original):
        return None, "unchanged", words
    lower = clean.casefold()
    if lower.startswith(("here is ", "here's ", "i can’t ", "i can't ", "i cannot ", "<email>", "sure,")):
        return None, "assistant_wrapper", words
    return clean, None, words


def torch_generate(path: Path, device: str, threads: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    torch.manual_seed(56174)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "mps" else torch.bfloat16
    model, info = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=dtype, output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs", "unexpected_keys")):
        raise ValueError("The local generator checkpoint did not load exactly")
    model.to(device).eval()

    @torch.inference_mode()
    def predict(rows):
        inputs = tokenizer([prompt(tokenizer, row) for row in rows], add_special_tokens=False,
                           padding=True, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, do_sample=True, temperature=0.65, top_p=0.9, max_new_tokens=340,
                                 pad_token_id=tokenizer.pad_token_id)
        suffixes = outputs[:, inputs["input_ids"].shape[1]:].cpu()
        return tokenizer.batch_decode(suffixes, skip_special_tokens=True)

    return predict, str(dtype)


def mlx_generate(path: Path, device: str):
    import mlx.core as mx
    import mlx.nn as nn

    stream = mx.new_stream(mx.cpu if device == "cpu" else mx.gpu)
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(56174)
    with mx.stream(stream):
        model, tokenizer = load(str(path), tokenizer_config={"trust_remote_code": False})
        nn.quantize(model, group_size=64, bits=4, mode="affine")
        mx.eval(model.parameters())
    sampler = make_sampler(temp=0.65, top_p=0.9)

    def predict(rows):
        tokens = [tokenizer.encode(prompt(tokenizer, row), add_special_tokens=False) for row in rows]
        return batch_generate(model, tokenizer, tokens, max_tokens=340, sampler=sampler, stream=stream).texts

    return predict, "mlx-affine-q4-g64"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generator", choices=("qwen", "smol2"))
    parser.add_argument("originals", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--runtime", choices=("torch", "mlx"), default="torch")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if min(args.batch, args.threads) < 1:
        parser.error("Batch size and thread count must be positive")
    if args.runtime == "mlx" and args.checkpoint is None:
        parser.error("Specify the original pinned snapshot when using the isolated MLX runtime")
    path, revision, expected_hash = snapshot(args.generator, args.checkpoint)
    if file_hash(path / "model.safetensors") != expected_hash:
        raise ValueError("Local checkpoint differs from the frozen weight hash")
    predict, precision = (
        mlx_generate(path, args.device) if args.runtime == "mlx" else torch_generate(path, args.device, args.threads)
    )
    all_rows = json.loads(args.originals.read_text())["originals"]
    rows = [row for row in all_rows if row["generator"] == args.generator][:args.limit]
    if any(digest(row["text"]) != row["text_sha256"] for row in rows):
        raise ValueError("A frozen email body hash does not match")
    metadata = {"generator": args.generator, "generator_revision": revision, "weights_sha256": expected_hash,
                "originals_sha256": file_hash(args.originals), "expected_count": len(rows), "batch": args.batch,
                "device": args.device, "runtime": args.runtime, "precision": precision, "seed": 56174,
                "prompt_version": "rewrite-explicit-length-v2", "temperature": 0.65, "top_p": 0.9, "max_new_tokens": 340}
    with args.output.open("x", encoding="utf-8") as output:
        print(json.dumps(metadata), file=output, flush=True)
        for i in range(0, len(rows), args.batch):
            group = rows[i:i + args.batch]
            texts = predict(group)
            for row, raw in zip(group, texts, strict=True):
                text, reason, words = output_text(raw, row["text"])
                print(json.dumps({"id": row["id"], "input_text_sha256": row["text_sha256"],
                                  "text": text, "text_sha256": digest(text) if text else None,
                                  "raw_words": len(raw.split()), "body_words": words, "exclusion": reason},
                                 ensure_ascii=False), file=output)
            output.flush()
            print(f"{args.generator}: {min(i + len(group), len(rows))}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
