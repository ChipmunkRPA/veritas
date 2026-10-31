"""`adapter_loader` functions (models.yaml `load.adapter_loader`) for the v0.3 document models.

The ModelManager calls `loader(model_id, device, dtype)` only after `ensure_verified` has checked every pinned file of
that model. These functions read the verified local files (HF snapshots at the pinned revision, `data/weights/...` for
the local fine-tune). Nothing here downloads.
"""
from __future__ import annotations

from app.detectors.image.adapters.loaders import LoadedModel, _entry, _file, _revision
from app.engine.fetchers import ensure_verified

DESKLIB_BASE = "hf.desklib.ai_text_detector_v1_01"


def _snapshot(model_id: str):
    return _file(_entry(model_id), "config.json").parent


def load_desklib(model_id: str, device: str, dtype: str) -> LoadedModel:
    from transformers import AutoTokenizer

    from app.detectors.text.classifier import DesklibAIDetectionModel

    entry = _entry(model_id)
    snapshot = _snapshot(model_id)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = DesklibAIDetectionModel.from_pretrained(snapshot).to(device).eval()
    return LoadedModel((tokenizer, model, entry.source.repo or model_id), _revision(entry))


def load_desklib_finetune(model_id: str, device: str, dtype: str) -> LoadedModel:
    """The fine-tuned weights with the base checkpoint's tokenizer, as v0.3 loads them (the fine-tune kept the vocabulary)."""
    from transformers import AutoTokenizer

    from app.detectors.text.classifier import DesklibAIDetectionModel

    base = _entry(DESKLIB_BASE)
    ensure_verified(base)
    tokenizer = AutoTokenizer.from_pretrained(_snapshot(DESKLIB_BASE))
    entry = _entry(model_id)
    model = DesklibAIDetectionModel.from_pretrained(_file(entry, "config.json").parent).to(device).eval()
    weights = next(f.sha256 for f in entry.source.files if f.path == "model.safetensors")
    return LoadedModel((tokenizer, model, entry.source.path or model_id), str(weights))


def load_sequence_classifier(model_id: str, device: str, dtype: str) -> LoadedModel:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    entry = _entry(model_id)
    snapshot = _snapshot(model_id)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = AutoModelForSequenceClassification.from_pretrained(snapshot).to(device).eval()
    return LoadedModel((tokenizer, model), _revision(entry))


def load_causal_lm(model_id: str, device: str, dtype: str) -> LoadedModel:
    """Binoculars LLMs: fp16 when the ModelManager grants it (accelerators, `fp16_safe`), else fp32."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    entry = _entry(model_id)
    snapshot = _snapshot(model_id)
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch_dtype).to(device).eval()
    return LoadedModel((tokenizer, model), _revision(entry))
