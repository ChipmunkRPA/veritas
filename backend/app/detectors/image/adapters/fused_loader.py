"""Dedicated, local-only composite loader for the fixed image inspection model."""
from __future__ import annotations

from app.detectors.image.adapters.loaders import LoadedModel, _entry, _file
from app.engine.models import ModelUnavailable


def load_fused(model_id: str, device: str, dtype: str) -> LoadedModel:
    from app.detectors.image import fused
    from app.engine.fetchers import ensure_verified

    if model_id != fused.MODEL_ID:
        raise ModelUnavailable("The fixed image inspection loader received an unexpected model id")
    entry = _entry(model_id)
    if entry.depends_on != [fused.BACKBONE_MODEL_ID]:
        raise ModelUnavailable("The fixed image inspection dependency differs from its manifest contract")
    backbone = _entry(fused.BACKBONE_MODEL_ID)
    for model, filename, digest in (
        (entry, fused.CHECKPOINT_FILE, fused.CHECKPOINT_SHA256),
        (backbone, fused.BACKBONE_FILE, fused.BACKBONE_SHA256),
    ):
        if (model.source.type != "local" or len(model.source.files) != 1
                or model.source.files[0].path != filename or model.source.files[0].sha256 != digest):
            raise ModelUnavailable("The fixed image inspection manifest contains an unexpected checkpoint")
        ensure_verified(model)
    # The manager leases the primary only; its declared footprint includes both
    # branches. The dependency is verified here, never separately loaded/leased.
    obj = fused.load_model(_file(entry, fused.CHECKPOINT_FILE), _file(backbone, fused.BACKBONE_FILE), device, dtype)
    return LoadedModel(obj, fused.CHECKPOINT_REVISION)
