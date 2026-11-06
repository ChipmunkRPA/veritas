"""Supervised AI-text classifiers.

Primary: desklib/ai-text-detector-v1.01 — DeBERTa-v3-large with mean pooling, trained on the RAID
benchmark family (robust to paraphrase/adversarial attacks; top of the RAID leaderboard at release).
Secondary: Hello-SimpleAI/chatgpt-detector-roberta — HC3-trained RoBERTa (ChatGPT-era baseline).
"""
from __future__ import annotations

import json
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)

from app.core import deadline
from app.core.config import settings, text_classifier_source
from app.core.device import DEVICE
from app.detectors.common.registry import registry
from app.detectors.text.coverage import classify_complete

PRIMARY_BATCH = 8
SECONDARY_BATCH = 16


def _validated_logits(logits: Any, batch_size: int, *, width: int | None = None) -> torch.Tensor:
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != batch_size
            or logits.shape[1] < 1 or (width is not None and logits.shape[1] != width)
            or not logits.is_floating_point()):
        raise ValueError("classifier logits have an invalid shape or dtype")
    logits = logits.float()
    if not torch.isfinite(logits).all():
        raise ValueError("classifier returned non-finite raw logits")
    return logits


class DesklibAIDetectionModel(PreTrainedModel):
    """DeBERTa encoder + mean pooling + linear head (architecture published with the desklib checkpoint)."""

    config_class = AutoConfig
    base_model_prefix = "model"
    _tied_weights_keys: dict = {}

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()  # transformers >= 5 requires post_init (sets tied-weight metadata); init_weights() is legacy

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.model(input_ids, attention_mask=attention_mask)
        h = out[0]
        m = attention_mask.unsqueeze(-1).expand(h.size()).float()
        pooled = (h * m).sum(1) / torch.clamp(m.sum(1), min=1e-9)
        return {"logits": self.classifier(pooled)}


class DesklibClassifier:
    name = "desklib-deberta-v3-large"

    def __init__(self) -> None:
        self.source = text_classifier_source()
        self.device: Any = DEVICE
        self.tok = AutoTokenizer.from_pretrained(settings.text_classifier_id)  # tokenizer is unchanged by fine-tuning
        self.model = DesklibAIDetectionModel.from_pretrained(self.source).to(DEVICE).eval()

    @classmethod
    def from_parts(cls, tokenizer: Any, model: Any, source: str, device: Any) -> DesklibClassifier:
        """A view over an already-loaded tokenizer and model (v2 adapters pass the leased model and its device)."""
        clf = cls.__new__(cls)
        clf.source, clf.device, clf.tok, clf.model = source, device, tokenizer, model
        return clf

    @torch.inference_mode()
    def predict(self, texts: list[str], max_len: int = 512, batch: int = PRIMARY_BATCH, *, check=deadline.check) -> list[float]:
        def score(encoded):
            logits = self.model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])["logits"]
            logits = _validated_logits(logits, len(encoded["input_ids"]), width=1)
            return torch.sigmoid(logits).squeeze(-1)

        scores, self.last_coverage = classify_complete(texts, self.tok, self.device, score, max_len=max_len, batch=batch, check=check)
        return scores


class FourClassTextModel(DesklibAIDetectionModel):
    """Mean-pooled classifier with the four classes specified by the pinned release."""

    def __init__(self, config):
        super().__init__(config)
        if config.detector_num_labels != 4:
            raise ValueError("Unexpected text classification head")
        self.classifier = nn.Linear(config.hidden_size, 4)


class FourClassTextClassifier:
    def __init__(self, tokenizer: Any, model: FourClassTextModel, device: Any) -> None:
        self.tok, self.model, self.device = tokenizer, model, device

    @classmethod
    def from_snapshot(cls, path: Path, device: Any) -> FourClassTextClassifier:
        head = json.loads((path / "serving_head.json").read_text(encoding="utf-8"))
        if head["labels"] != ["human", "ai", "ai_edited", "humanized"] or head["human_index"] != 0:
            raise ValueError("Unexpected text classification labels")
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        model, info = FourClassTextModel.from_pretrained(
            path, local_files_only=True, use_safetensors=True, output_loading_info=True,
        )
        if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs", "unexpected_keys")):
            raise ValueError("The text classifier checkpoint did not load exactly")
        return cls(tokenizer, model.to(device).eval(), device)

    @torch.inference_mode()
    def predict(self, texts: list[str], max_len: int = 512, batch: int = PRIMARY_BATCH, *, check=deadline.check) -> list[float]:
        def score(encoded):
            ids = encoded["input_ids"]
            output = self.model(ids, attention_mask=encoded["attention_mask"])
            logits = _validated_logits(output["logits"], len(ids), width=4)
            return 1 - logits.softmax(dim=-1)[:, 0]

        scores, self.last_coverage = classify_complete(
            texts, self.tok, self.device, score, max_len=max_len, batch=batch, check=check,
        )
        return scores


class SecondaryClassifier:
    name = "hello-simpleai-roberta"

    def __init__(self) -> None:
        mid = settings.text_secondary_classifier_id
        self.device: Any = DEVICE
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.model = AutoModelForSequenceClassification.from_pretrained(mid).to(DEVICE).eval()
        self._bind()

    @classmethod
    def from_parts(cls, tokenizer: Any, model: Any, device: Any) -> SecondaryClassifier:
        clf = cls.__new__(cls)
        clf.device, clf.tok, clf.model = device, tokenizer, model
        clf._bind()
        return clf

    def _bind(self) -> None:
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.pos = next((i for i, l in id2label.items() if "chatgpt" in l or "ai" in l or "fake" in l), 1)

    @torch.inference_mode()
    def predict(self, texts: list[str], max_len: int = 512, batch: int = SECONDARY_BATCH, *, check=deadline.check) -> list[float]:
        def score(encoded):
            logits = _validated_logits(self.model(**encoded).logits, len(encoded["input_ids"]))
            if not 0 <= self.pos < logits.shape[1]:
                raise ValueError("classifier AI class is outside the returned logits")
            return torch.softmax(logits, dim=-1)[:, self.pos]

        scores, self.last_coverage = classify_complete(texts, self.tok, self.device, score, max_len=max_len, batch=batch, check=check)
        return scores


def get_primary() -> DesklibClassifier:
    """Load-only compatibility getter; inference must use the scoped borrow."""
    return registry.get("text:desklib", DesklibClassifier)


def get_secondary() -> SecondaryClassifier | None:
    """Load-only compatibility getter; inference must use the scoped borrow."""
    if not settings.text_secondary_classifier_id:
        return None
    return registry.get("text:secondary", SecondaryClassifier)


def borrow_primary() -> AbstractContextManager[DesklibClassifier]:
    """Keep the legacy primary bundle leased through its complete inference."""
    return registry.borrow("text:desklib", DesklibClassifier)


def borrow_secondary() -> AbstractContextManager[SecondaryClassifier | None]:
    """A disabled secondary yields None without acquiring or loading a bundle."""
    if not settings.text_secondary_classifier_id:
        return nullcontext(None)
    return registry.borrow("text:secondary", SecondaryClassifier)
