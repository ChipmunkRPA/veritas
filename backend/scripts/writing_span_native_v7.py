"""Pinned local CPU encoders with actual input binding and complete native offsets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from app.detectors.text.extract import from_string
from app.detectors.text.pipeline import windows
from app.detectors.text.segment import segment
from scripts import run_writing_modernbert_v4 as old
from scripts import writing_modernbert_v4 as identities
from scripts.writing_span_data_v7 import OLD, sha
from scripts.writing_span_geometry_v7 import native_coverage, require, token_targets


def tensor_sha(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


class Native:
    def __init__(self, with_current=False):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        self.paths = json.loads((OLD / "freeze.json").read_text())["model_paths"]
        old.assets(Path(self.paths["candidate"]), identities.CANDIDATE_ASSETS)
        self.encoder = old.load_model("candidate", self.paths)
        self.current = None
        self.calls = {"encoder": 0, "current_classifier": 0}
        self.handles = []

        def count_encoder(_module, _args, _kwargs):
            self.calls["encoder"] += 1

        self.handles.append(
            self.encoder.model.model.register_forward_pre_hook(count_encoder, with_kwargs=True)
        )
        if with_current:
            old.assets(Path(self.paths["baseline"]), identities.BASELINE_ASSETS)
            self.current = old.load_model("baseline", self.paths)

            def count_current(_module, _args, _kwargs):
                self.calls["current_classifier"] += 1

            self.handles.append(self.current.model.register_forward_pre_hook(count_current, with_kwargs=True))

    @torch.inference_mode()
    def features(self, case, audit=False):
        text, tok = case["text"], self.encoder.tok
        plain = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        encoded = tok(
            text,
            truncation=True,
            max_length=512,
            stride=0,
            return_overflowing_tokens=True,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
        expected = plain["input_ids"]
        offsets = [tuple(map(int, v)) for v in plain["offset_mapping"]]
        require(
            0 < len(expected) <= 4096, "Complete bounded token sequence required; never silently truncate"
        )
        native_coverage(text, offsets)
        recovered, locations, vectors, proofs = [], [], [], []
        for ids, mask, specials, positions in zip(
            encoded["input_ids"],
            encoded["attention_mask"],
            encoded["special_tokens_mask"],
            encoded["offset_mapping"],
            strict=True,
        ):
            keep = [i for i, (a, b) in enumerate(zip(mask, specials, strict=True)) if a and not b]
            recovered.extend(ids[i] for i in keep)
            locations.extend(tuple(map(int, positions[i])) for i in keep)
            mini = {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}
            captured = []

            def bind(_module, args, kwargs, wanted=mini, proof=captured):
                actual_ids = kwargs.get("input_ids", args[0] if args else None)
                actual_mask = kwargs.get("attention_mask")
                require(
                    torch.equal(actual_ids, wanted["input_ids"])
                    and torch.equal(actual_mask, wanted["attention_mask"]),
                    "Actual encoder call differs from the planned native window",
                )
                proof.append(tensor_sha(actual_ids) + ":" + tensor_sha(actual_mask))

            hook = self.encoder.model.model.register_forward_pre_hook(bind, with_kwargs=True)
            try:
                output = self.encoder.model.model(**mini, return_dict=True).last_hidden_state
            finally:
                hook.remove()
            require(
                len(captured) == 1
                and output.shape == (1, len(ids), 768)
                and output.dtype == torch.float32
                and bool(torch.isfinite(output).all()),
                "Every finite native hidden token required",
            )
            selected = output[0, keep].detach().cpu().contiguous()
            if audit:
                whole = self.encoder.model(**mini, output_hidden_states=True, return_dict=True)
                require(
                    torch.equal(output, whole.hidden_states[-1]),
                    "Whole-classifier/native-encoder feature tap differs",
                )
            vectors.append(selected.numpy().copy())
            proofs.append(
                {
                    "content_tokens": len(keep),
                    "model_input_sha256": captured[0],
                    "hidden_float32_sha256": tensor_sha(selected),
                    "full_model_feature_tap_exact": audit,
                }
            )
        require(
            recovered == expected and locations == offsets,
            "Actual windows must preserve every original token and offset once",
        )
        values = np.concatenate(vectors)
        result = {k: case[k] for k in ("id", "group", "role", "text")}
        result.update(
            features=values,
            offsets=offsets,
            positions=np.asarray([(a + b) / (2 * len(text)) for a, b in offsets], np.float32),
            native={
                "tokens": len(expected),
                "windows": len(proofs),
                "proofs": proofs,
                "complete_features_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                "native_offsets_sha256": hashlib.sha256(
                    json.dumps(offsets, separators=(",", ":")).encode()
                ).hexdigest(),
            },
        )
        if case["role"] == "fit":
            result["targets"], result["weights"] = token_targets(text, case["labels"], offsets)
            result["truth_spans"] = case["truth_spans"]
        return result

    @torch.inference_mode()
    def current_proposals(self, case, audit=False):
        require(self.current is not None, "Explicit current detector required")
        doc = from_string(case["text"])
        require(doc.text == case["text"], "Prepared extraction must be idempotent")
        foci = segment(doc)
        contexts = windows(foci, 3)
        coverage = np.zeros(len(doc.text), bool)
        wanted = []
        for context in contexts:
            enc = self.current.tok(
                context, truncation=True, max_length=512, stride=0, return_overflowing_tokens=True
            )
            wanted.extend(enc["input_ids"])
        actual = []

        def bind(_module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            attention = kwargs["attention_mask"]
            for row, valid in zip(ids, attention, strict=True):
                actual.append(row[valid.bool()].cpu().tolist())

        hook = self.current.model.register_forward_pre_hook(bind, with_kwargs=True)
        try:
            scores = self.current.predict(contexts, max_len=512, batch=4, check=lambda _: None)
        finally:
            hook.remove()
        require(
            actual == wanted and len(scores) == len(foci), "Actual current detector inputs or foci differ"
        )
        proposals = []
        audits = 0
        maximum = 0.0
        for focus, context, q, record in zip(foci, contexts, scores, self.current.last_coverage, strict=True):
            coverage[focus.start : focus.end] = True
            require(np.isfinite(q) and 0 <= q <= 1, "Current raw response is invalid")
            if audit or record["token_windows"] > 1:
                delta = abs(old.direct_response(self.current, "baseline", context) - q)
                require(delta <= 1e-5, "Independent current window response differs")
                audits, maximum = audits + 1, max(maximum, delta)
            proposals.append({"start": focus.start, "end": focus.end, "raw": float(q)})
        required = np.asarray([not c.isspace() for c in doc.text])
        require(required.any() and coverage[required].all(), "Current foci omit source characters")
        return proposals, {
            "foci": len(foci),
            "direct_audits": audits,
            "max_direct_difference": maximum,
            "actual_input_windows": len(actual),
            "actual_input_ids_sha256": hashlib.sha256(
                json.dumps(actual, separators=(",", ":")).encode()
            ).hexdigest(),
        }

    def unchanged(self):
        for clf in (self.encoder, self.current):
            if clf is not None:
                require(
                    all(not p.requires_grad and p.grad is None for p in clf.model.parameters()),
                    "A pretrained parameter entered fitting",
                )
        old.assets(Path(self.paths["candidate"]), identities.CANDIDATE_ASSETS)
        if self.current is not None:
            old.assets(Path(self.paths["baseline"]), identities.BASELINE_ASSETS)
        return True


def smoke(output):
    native = Native(with_current=True)
    cases = [
        "An authored runtime probe checks positions and shapes, without measuring detector accuracy.",
        "An authored Unicode probe: café, e\u0301 and 🙂 preserve their original character positions.",
        "An authored overflow probe retains every word and source position " * 120 + ".",
    ]
    records = []
    for i, text in enumerate(cases):
        case = {"id": str(i), "group": "authored", "role": "smoke", "text": text}
        if i == 1:
            before = dict(native.calls)
            try:
                native.features(case, audit=True)
            except ValueError as error:
                require(
                    "coverage omits" in str(error) and native.calls == before,
                    "Unsupported Unicode must be refused before a model forward",
                )
                records.append({"status": "unsupported_coverage_refused", "new_model_forwards": 0})
                continue
            raise ValueError("The explicit combining-mark coverage refusal changed")
        f = native.features(case, audit=True)
        _, current = native.current_proposals(case, audit=True)
        records.append(
            {
                "status": "supported",
                "tokens": f["native"]["tokens"],
                "windows": f["native"]["windows"],
                "feature_width": f["features"].shape[1],
                "native_feature_tap_exact": True,
                "current_direct_audits": current["direct_audits"],
                "current_maximum_difference": current["max_direct_difference"],
            }
        )
    native.unchanged()
    value = {
        "schema": "chipmunk.writing-span-native-smoke/7",
        "authored_cases": 3,
        "records": records,
        "actual_pretrained_forwards": native.calls,
        "encoder_parameters": sum(p.numel() for p in native.encoder.model.parameters()),
        "project_source_forwards": 0,
        "fitted_heads": 0,
        "detection_quality": False,
        "implementation_sha256": sha(Path(__file__)),
    }
    with output.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    smoke(parser.parse_args().output)
