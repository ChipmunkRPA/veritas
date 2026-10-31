"""GenD with a CLIP ViT-L/14 encoder (Yermakov, Cech, Matas, Fritz, WACV 2026): the inference path, written from the upstream
code.

- **Model.** transformers `CLIPVisionModel` (ViT-L/14 at 224 px, QuickGELU). Its post-LayerNorm CLS token (`pooler_output`)
  is L2-normalised and mapped by a 1024 → 2 linear head to logits for real and fake. This is upstream `GenD` with `backbone
  openai/clip-vit-large-patch14` and `head LinearNorm` (`modeling_gend.py` in HF `yermandy/GenD_CLIP_L_14@891ce014`, MIT).
  The pinned safetensors keys carry the upstream `feature_extractor.vision_model.` and `model.linear.` prefixes. The CLIP
  `visual_projection.weight`, which the forward pass never uses, is dropped by name; the rest load strictly. No upstream code
  is vendored.
- **Face crop (ML N5).** Upstream `detector.align_face` (github.com/yermandy/GenD @ 387a4226, MIT) with `target_size None`
  and scale 1.3: a similarity transform (`cv2.estimateAffinePartial2D`, LMEDS) of the five landmarks (image-left eye,
  image-right eye, nose tip, image-left and image-right mouth corners) onto a fixed template, an output side of 1.3 × the
  mean ratio of landmark distances to template distances, and a bilinear warp. Upstream takes the landmarks from RetinaFace
  (insightface buffalo_l det_10g); Veritas takes YuNet's from the faces stage, mapped to native pixels.
- **Preprocessing.** The CLIP image processor of upstream's pinned transformers 4.56.2 (`CLIPImageProcessor`, PIL path;
  `CLIPImageProcessorPil` in transformers 5): bicubic resize of the short side to 224 px (long side truncated), centre crop
  224 px, scale to [0, 1], CLIP mean and std.
- **Output.** z = logit_fake − logit_real, so sigmoid(z) equals upstream's softmax p(fake).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

CROP = 224
SCALE = 1.3
WHOLE_IMAGE_MAX_PX = 256  # ML N5: portraits of at most 256 px are fed whole
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
# upstream detector.align_face destination points, as fractions of the output side
TEMPLATE = np.array([[0.34, 0.46], [0.66, 0.46], [0.5, 0.64], [0.37, 0.82], [0.63, 0.82]], dtype=np.float32)
UNUSED_KEYS = ("feature_extractor.visual_projection.weight",)


def vision_config(**overrides: Any) -> Any:
    """openai/clip-vit-large-patch14 vision tower; tests pass smaller overrides."""
    from transformers import CLIPVisionConfig

    params = {"hidden_size": 1024, "intermediate_size": 4096, "projection_dim": 768, "num_hidden_layers": 24,
              "num_attention_heads": 16, "image_size": CROP, "patch_size": 14, "hidden_act": "quick_gelu",
              "layer_norm_eps": 1e-5}
    return CLIPVisionConfig(**(params | overrides))


class GenD(torch.nn.Module):
    def __init__(self, config: Any = None) -> None:
        from transformers import CLIPVisionModel

        super().__init__()
        self.backbone = CLIPVisionModel(config if config is not None else vision_config())
        self.head = torch.nn.Linear(self.backbone.config.hidden_size, 2)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixel_values=pixel_values).pooler_output
        return self.head(torch.nn.functional.normalize(features, p=2, dim=1))


def read_state(path: Path) -> dict[str, torch.Tensor]:
    """Upstream key names → this module's. Unknown keys pass through unchanged, so strict loading refuses them."""
    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = {}
    for key, value in load_file(str(path)).items():
        if key in UNUSED_KEYS:
            continue
        if key.startswith("feature_extractor.vision_model."):  # transformers 5 CLIPVisionModel has no nested vision_model
            key = "backbone." + key.removeprefix("feature_extractor.vision_model.")
        elif key.startswith("model.linear."):
            key = "head." + key.removeprefix("model.linear.")
        state[key] = value
    return state


def load_model(path: Path, device: Any, config: Any = None) -> torch.nn.Module:
    model = GenD(config)
    model.load_state_dict(read_state(path), strict=True)
    return model.to(device).eval()


def align_transform(landmarks: Any, scale: float = SCALE) -> tuple[np.ndarray, int] | None:
    """Upstream `align_face`'s similarity transform (2 × 3, source px → crop px) and output side; None when none exists."""
    src = np.asarray(landmarks, dtype=np.float32).reshape(len(TEMPLATE), 2)
    upper = np.triu_indices(len(TEMPLATE), k=1)
    desired = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=-1)[upper]
    reference = np.linalg.norm(TEMPLATE[:, None, :] - TEMPLATE[None, :, :], axis=-1)[upper]
    side = np.round(np.mean(desired / reference) * scale).astype(int)  # np.int64, as upstream
    if side < 1:
        return None
    # upstream's in-place steps: each is computed in float64 (int64 and float64 operands) and stored in the float32 template
    dst = TEMPLATE.copy()
    margin = side * (scale - 1) / 2.0
    dst[:, :] = dst * side
    dst += margin
    dst *= side / (side + 2 * margin)
    matrix = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
    if matrix is None:
        return None
    return matrix, int(side)


def warp(pixels: np.ndarray, matrix: np.ndarray, side: int) -> np.ndarray:
    return cv2.warpAffine(np.ascontiguousarray(pixels), matrix, (side, side), flags=cv2.INTER_LINEAR)


def align(pixels: np.ndarray, landmarks: Any, scale: float = SCALE) -> np.ndarray | None:
    """Upstream `align_face(img, landmarks, target_size=None, scale)` on 8-bit RGB pixels; None when no transform exists."""
    transform = align_transform(landmarks, scale)
    return None if transform is None else warp(pixels, *transform)


def to_batch(faces: list[np.ndarray], crop: int = CROP) -> torch.Tensor:
    """CLIP preprocessing of 8-bit RGB inputs → float32 NCHW."""
    arrays = []
    for face in faces:
        height, width = face.shape[:2]
        new_long = int(crop * max(width, height) / min(width, height))
        size = (crop, new_long) if width <= height else (new_long, crop)  # PIL (width, height)
        resized = np.asarray(Image.fromarray(np.ascontiguousarray(face)).resize(size, Image.Resampling.BICUBIC))
        top, left = (resized.shape[0] - crop) // 2, (resized.shape[1] - crop) // 2
        arrays.append(resized[top:top + crop, left:left + crop])
    x = np.stack(arrays).astype(np.float32) / 255.0
    x = (x - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


def logits(model: Any, batch: torch.Tensor, device: Any) -> list[float]:
    """z = logit_fake − logit_real per input (upstream class order: 0 real, 1 fake)."""
    with torch.inference_mode():
        out = model(batch.to(device)).detach().to("cpu", torch.float32).double()  # MPS has no float64
    return [float(z) for z in out[:, 1] - out[:, 0]]


def gradcam(model: Any, batch: torch.Tensor, device: Any) -> np.ndarray | None:
    """Grad-CAM of z for the first input of a preprocessed batch, on the patch grid (16 × 16 for ViT-L/14 at 224 px), in
    [0, 1]; None when the model has no CLIP encoder or the map is empty.

    Hooks the output of the last encoder block's pre-attention LayerNorm: its patch tokens reach the class token (the only
    token the head reads) through that block's attention, so they receive a gradient (as `image/gradcam.vit_gradcam_map`).
    Channel weights are the mean gradient over patches; the map is the positive part of the weighted sum.
    `torch.autograd.grad` leaves no gradient buffers on the shared model."""
    encoder = getattr(getattr(model, "backbone", None), "encoder", None)
    if encoder is None:
        return None
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["tokens"] = output[0] if isinstance(output, tuple) else output

    handle = encoder.layers[-1].layer_norm1.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            x = batch[:1].to(device).clone().requires_grad_(True)
            out = model(x)
            z = out[0, 1] - out[0, 0]
            (grads,) = torch.autograd.grad(z, captured["tokens"])
    finally:
        handle.remove()
    tokens, grads = captured["tokens"][0, 1:].detach(), grads[0, 1:]
    side = round(tokens.shape[0] ** 0.5)
    if side * side != tokens.shape[0]:
        return None
    cam = torch.relu((tokens * grads.mean(dim=0, keepdim=True)).sum(dim=-1)).reshape(side, side)
    values = cam.detach().to("cpu", torch.float32).numpy()
    if not np.isfinite(values).all() or values.max() <= 0:
        return None
    return values / values.max()
