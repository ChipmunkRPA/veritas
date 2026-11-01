"""Offline GenD engineering validation; publisher examples are not an accuracy benchmark.

Run after downloading the pinned official reference and its example-crop manifest to
data/model-upgrade-review/gend. No model is enabled and no customer result is changed.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import CLIPConfig, CLIPModel
from transformers.models.clip.image_processing_pil_clip import CLIPImageProcessorPil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.detectors.image import gend
from app.detectors.image.faces import detect_faces, load_yunet
from app.engine.fetchers import ensure_verified, file_path
from app.engine.registry import get_registry

REFERENCE_SHA256 = "d2bdc7d57ea208def628f16064c7d22317ec92878b982ac6ac5ab4bcda4991ac"


def main() -> None:
    root = Path("data/model-upgrade-review/gend")
    manifest = json.loads((root / "fixtures.json").read_text())
    reference_path = root / "modeling_gend.py"
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == REFERENCE_SHA256
    rows = manifest["items"]
    images = []
    for row in rows:
        path = root / row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        images.append(np.asarray(Image.open(path).convert("RGB")))
    torch.set_num_threads(4)
    entry = get_registry().manifest.model("gend.clip_l14")
    assert entry is not None
    ensure_verified(entry, force=True)
    weights = file_path(entry, next(f for f in entry.source.files if f.path == "model.safetensors"))
    processor = CLIPImageProcessorPil(**json.loads((root / "clip-preprocessor_config.json").read_text()))
    reference_inputs = processor(images=[Image.fromarray(x) for x in images], return_tensors="pt")["pixel_values"]
    native_inputs = gend.to_batch(images)
    preprocess_delta = float((reference_inputs - native_inputs).abs().max())
    assert preprocess_delta <= 1e-6, preprocess_delta

    spec = importlib.util.spec_from_file_location("gend_reviewed_reference", reference_path)
    assert spec and spec.loader
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    config = CLIPConfig.from_dict(json.loads((root / "clip-config.json").read_text()))

    class OfflineEncoder(upstream.CLIPEncoder):
        """Only replace downloads with config construction; retain the publisher's forward."""

        def __init__(self, model_name: str):
            torch.nn.Module.__init__(self)
            clip = CLIPModel(config)
            self.vision_model = clip.vision_model
            self.visual_projection = clip.visual_projection
            self.model_name = model_name
            self.features_dim = self.vision_model.config.hidden_size
            self._preprocess = processor

    upstream.CLIPEncoder = OfflineEncoder
    reference = upstream.GenD(upstream.GenDConfig(backbone="openai/clip-vit-large-patch14", head="LinearNorm"))
    reference.load_state_dict(load_file(str(weights)), strict=True)
    reference.eval()
    started = time.monotonic()
    reference_logits = []
    with torch.inference_mode():
        for chunk in reference_inputs.split(4):
            logits = reference(chunk).double()
            reference_logits.extend((logits[:, 1] - logits[:, 0]).tolist())
    reference_seconds = time.monotonic() - started
    del reference
    gc.collect()
    model = gend.load_model(weights, "cpu")
    native_logits = []
    started = time.monotonic()
    for chunk in native_inputs.split(4):
        native_logits.extend(gend.logits(model, chunk, "cpu"))
    cpu_seconds = time.monotonic() - started
    logit_delta = float(np.max(np.abs(np.asarray(native_logits) - reference_logits)))
    assert logit_delta < 1e-4, logit_delta
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    accelerated_logits = native_logits
    mps_seconds = None
    if device == "mps":
        model.to("mps")
        started = time.monotonic()
        accelerated_logits = []
        for chunk in native_inputs.split(4):
            accelerated_logits.extend(gend.logits(model, chunk, "mps"))
        torch.mps.synchronize()
        mps_seconds = time.monotonic() - started
        assert np.max(np.abs(np.asarray(accelerated_logits) - native_logits)) < 1e-3

    # Reapplying a different face crop to pre-cropped examples is a sensitivity probe,
    # not a claim that the app's YuNet pipeline reproduces upstream RetinaFace crops.
    face_entry = get_registry().manifest.model("yunet.onnx")
    if face_entry is None:
        face_entry = next(e for e in get_registry().manifest.models if any("face_detection_yunet" in f.path for f in e.source.files))
    ensure_verified(face_entry)
    detector = load_yunet(file_path(face_entry, face_entry.source.files[0]))
    recropped, recrop_indices = [], []
    for i, pixels in enumerate(images):
        faces = detect_faces(pixels[:, :, ::-1].copy(), detector=detector, min_size=48)
        if faces:
            crop = gend.align(pixels, faces[0].landmarks)
            if crop is not None:
                recropped.append(crop)
                recrop_indices.append(i)
    recrop_logits = []
    if recropped:
        for chunk in gend.to_batch(recropped).split(4):
            recrop_logits.extend(gend.logits(model, chunk, device))
    recrop_scores = dict(zip(recrop_indices, recrop_logits, strict=True))
    results = []
    for i, row in enumerate(rows):
        fake = "/real/" not in row["path"] and "/Celeb-real/" not in row["path"] and "/YouTube-real/" not in row["path"]
        results.append({**row, "publisher_label": "fake" if fake else "real", "reference_logit": reference_logits[i],
                        "local_logit": native_logits[i], "local_fake_response": float(torch.sigmoid(torch.tensor(native_logits[i]))),
                        "recrop_logit": recrop_scores.get(i), "decision_at_logit_zero": "fake" if native_logits[i] >= 0 else "real"})
    counts = {label: {"n": sum(r["publisher_label"] == label for r in results),
                      "flagged": sum(r["publisher_label"] == label and r["local_logit"] >= 0 for r in results)}
              for label in ("real", "fake")}
    output = {"purpose": manifest["purpose"], "weights_revision": entry.source.revision,
              "weight_sha256": next(f.sha256 for f in entry.source.files if f.path == "model.safetensors"),
              "reference_sha256": REFERENCE_SHA256, "preprocessing_max_abs_difference": preprocess_delta,
              "forward_max_abs_logit_difference": logit_delta, "cpu_mps_max_abs_logit_difference":
              float(np.max(np.abs(np.asarray(accelerated_logits) - native_logits))) if device == "mps" else None,
              "timing_seconds": {"reference_cpu": reference_seconds, "local_cpu": cpu_seconds, "local_mps": mps_seconds},
              "publisher_example_counts": counts, "recrop_probe": {"scored": len(recrop_scores),
              "boundary_changes": sum((native_logits[i] >= 0) != (z >= 0) for i, z in recrop_scores.items())},
              "production_enabled": False, "decision": "Keep disabled pending independent validation and commercial data-rights review.",
              "limits": ["Examples include related frames and possible training overlap; counts are not accuracy estimates.",
                         "No new independent source-disjoint benchmark is present in this archive.",
                         "YuNet landmark crops differ from the publisher's RetinaFace pipeline; numerical forward parity does not validate this replacement.",
                         "Historical admission records report failed incremental fusion value and elevated real-video false positives."],
              "items": results}
    (root / "verification.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({k: v for k, v in output.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
