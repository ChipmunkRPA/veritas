"""Generic wrapper around HuggingFace image-classification checkpoints used as AI/deepfake detectors.

Each entry in settings.image_classifiers is "repo_id|positive_label_substring|weight". The positive
label is the class meaning *AI-generated / fake*; its softmax probability is the signal value.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from app.core.config import settings
from app.core.device import DEVICE
from app.detectors.common.registry import registry


@dataclass
class ClassifierSpec:
    repo_id: str
    positive: str
    weight: float
    face_oriented: bool

    @property
    def short(self) -> str:
        return self.repo_id.split("/")[-1]


FACE_ORIENTED = {"prithivMLmods/Deep-Fake-Detector-v2-Model", "dima806/deepfake_vs_real_image_detection", "Wvolf/ViT_Deepfake_Detection"}


def specs() -> list[ClassifierSpec]:
    out = []
    for raw in settings.image_classifiers:
        repo, pos, w = (raw.split("|") + ["", "1.0"])[:3]
        out.append(ClassifierSpec(repo.strip(), pos.strip().lower(), float(w), repo.strip() in FACE_ORIENTED))
    return out


class HFImageClassifier:
    def __init__(self, spec: ClassifierSpec):
        self.spec: ClassifierSpec | None = spec
        self.device: Any = DEVICE
        self.processor = AutoImageProcessor.from_pretrained(spec.repo_id)
        self.model = AutoModelForImageClassification.from_pretrained(spec.repo_id).to(DEVICE).eval()
        self._bind(spec.positive)

    @classmethod
    def from_parts(cls, processor: Any, model: Any, positive: str, device: Any) -> HFImageClassifier:
        """A view over an already-loaded processor and model (v2 adapters pass the leased model and its device)."""
        clf = cls.__new__(cls)
        clf.spec, clf.device, clf.processor, clf.model = None, device, processor, model
        clf._bind(positive)
        return clf

    def _bind(self, positive: str) -> None:
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.pos_idx = next(i for i, l in self.id2label.items() if positive in l.lower())

    @torch.inference_mode()
    def predict(self, images: list[Image.Image]) -> list[dict]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits.float()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        out = []
        for p in probs:
            out.append(
                {
                    "p_ai": float(p[self.pos_idx]),
                    "labels": {self.id2label[i]: round(float(v), 4) for i, v in enumerate(p)},
                }
            )
        return out


def get_classifier(spec: ClassifierSpec) -> HFImageClassifier:
    """Load-only compatibility getter; inference must use the scoped borrow."""
    return registry.get(f"imgclf:{spec.repo_id}", lambda: HFImageClassifier(spec))


def borrow_classifier(spec: ClassifierSpec) -> AbstractContextManager[HFImageClassifier]:
    """Keep the legacy classifier pinned throughout prediction and attribution."""
    return registry.borrow(f"imgclf:{spec.repo_id}", lambda: HFImageClassifier(spec))


def to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb)
