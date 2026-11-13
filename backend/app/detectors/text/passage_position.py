"""Optional local evidence from the frozen context-based writing head."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

from app.core.config import settings
from app.detectors.text.extract import text_hash
from app.engine.boundary import DetectorTimeout
from app.evidence import document_matched_context, document_passage_position

HEAD = Path(__file__).parent / "calibrations/passage-position-head-v1.json"
HEAD_SHA256 = "d44984c5c137a4c8653a226932491b950764e989507756320428d993cff8b498"
EXTRA_SECONDS = 6.0


@lru_cache(maxsize=1)
def _head():
    raw = HEAD.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HEAD_SHA256:
        raise ValueError("The pinned writing position weights differ")
    row = json.loads(raw)
    if (set(row) != {"coef", "mean", "scale", "intercept"} or not math.isfinite(row["intercept"])
            or any(len(row[name]) != 1024 or not all(math.isfinite(value) for value in row[name])
                   for name in ("coef", "mean", "scale"))
            or any(value < .2 for value in row["scale"])):
        raise ValueError("Invalid writing position weights")
    return row


def _scores(classifier, inputs, deadline):
    import numpy as np
    import torch

    head = _head()
    deadline.check("writing position inputs")
    encoded = classifier.tok([text for text, _, _ in inputs], padding=True, truncation=False, return_tensors="pt",
                             return_offsets_mapping=True, return_special_tokens_mask=True)
    if encoded["input_ids"].shape[1] > 512:
        raise ValueError("The complete writing input exceeds the tokenizer limit")
    active = encoded["attention_mask"].bool()
    ordinary = active & ~encoded["special_tokens_mask"].bool()
    with torch.inference_mode():
        hidden = classifier.model.model(
            input_ids=encoded["input_ids"].to(classifier.device),
            attention_mask=encoded["attention_mask"].to(classifier.device),
            return_dict=True,
        ).last_hidden_state.cpu()
        deadline.check("writing position response")
        if hidden.shape[:2] != active.shape or hidden.shape[-1] != 1024 or not torch.isfinite(hidden).all():
            raise ValueError("Unexpected native writing response")
        output = []
        for i, (text, start, length) in enumerate(inputs):
            ids = encoded["input_ids"][i][active[i]].tolist()
            if ids != classifier.tok(text, truncation=False, verbose=False)["input_ids"]:
                raise ValueError("The tokenizer did not preserve the complete input")
            end = start + length
            focus, context, covered = [], [], set()
            for j, (a, b) in enumerate(encoded["offset_mapping"][i].tolist()):
                if not ordinary[i][j]:
                    continue
                points = {k for k in range(a, b) if not text[k].isspace()}
                inside = {k for k in points if start <= k < end}
                if inside and inside != points:
                    raise ValueError("A native token crosses the writing focus boundary")
                if inside:
                    focus.append(j)
                    covered.update(inside)
                elif points:
                    context.append(j)
            if not focus or not context or any(k not in covered and not text[k].isspace() for k in range(start, end)):
                raise ValueError("The native writing focus has incomplete coverage")
            vector = (hidden[i, focus].mean(0) - hidden[i, context].mean(0)).numpy()
            transformed = (vector - head["mean"]) / head["scale"]
            value = float(transformed @ np.asarray(head["coef"]) + head["intercept"])
            if not math.isfinite(value):
                raise ValueError("Non-finite writing position response")
            output.append({"input_sha256": text_hash(text), "response_hex": value.hex(),
                           "focus_tokens": len(focus), "neighbor_tokens": len(context)})
        deadline.check("writing position evidence")
    return output


def capture(classifier, doc, sentences, ctx, spec, revisions):
    if (not settings.text_passage_position_review or settings.text_detector_mode != "candidate"
            or settings.text_window_sentences != 3 or str(ctx.device) not in {"cpu", "mps"}
            or not document_matched_context.matches(spec.id, spec.version, revisions)):
        return None
    spans = document_passage_position.geometry(doc.text, doc.source_map, doc.meta.get("method"), sentences)
    if spans is None:
        return None
    # Leave headroom for committing the normal result. The small optional
    # review never replaces or changes the existing classifier output.
    remaining = ctx.deadline.remaining()
    if remaining is not None and remaining < EXTRA_SECONDS + 1:
        return document_passage_position.unassessed(spans, sentences[1].text)
    try:
        readings = _scores(classifier, document_passage_position.inputs(doc.text, spans), ctx.deadline.nested(EXTRA_SECONDS))
    except (DetectorTimeout, KeyError, OSError, RuntimeError, ValueError):
        return document_passage_position.unassessed(spans, sentences[1].text)
    return document_passage_position.record(spans, readings, sentences[1].text)
