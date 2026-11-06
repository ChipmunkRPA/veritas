"""Pinned portrait inference for local benchmark edits, never a customer detector.

Only the publisher's MIT human animation weights are loaded. Face placement and
mouth measurements are supplied by the caller; InsightFace and its checkpoints
are neither imported nor downloaded. Rendered copies must retain their AI label.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

SOURCE_REVISION = "9b294b3d0536135442ea73cb01e6cb3ca7029dd3"
WEIGHT_REVISION = "82a4fa6735ca58432b6ce39301b4b9ee066dea47"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(root: Path, device="cpu"):
    receipt = json.loads((root / "permissions/liveportrait-verified-assets.json").read_text())
    if receipt["source_revision"] != SOURCE_REVISION or receipt["checkpoint_revision"] != WEIGHT_REVISION:
        raise ValueError("Unexpected portrait-model source or weight revision")
    for row in receipt["code"]:
        if digest(root / "liveportrait-source" / row["path"]) != row["sha256"]:
            raise ValueError("Portrait inference source differs from its reviewed hash")
    for row in receipt["weights"]:
        if digest(root / "liveportrait-weights" / row["path"]) != row["sha256"]:
            raise ValueError("Portrait checkpoint differs from its publisher LFS hash")
    package = "chipmunk_benchmark_portrait"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(root / "liveportrait-source/src")]
        sys.modules[package] = module

    def cls(module, name):
        return getattr(importlib.import_module(package + ".modules." + module), name)

    config = yaml.safe_load((root / "liveportrait-source/src/config/models.yaml").read_text())["model_params"]
    names = {
        "appearance_feature_extractor": ("appearance_feature_extractor", "AppearanceFeatureExtractor"),
        "motion_extractor": ("motion_extractor", "MotionExtractor"),
        "warping_module": ("warping_network", "WarpingNetwork"),
        "spade_generator": ("spade_generator", "SPADEDecoder"),
    }
    models = {}
    for name, (module, class_name) in names.items():
        path = root / "liveportrait-weights/liveportrait/base_models" / (name + ".pth")
        if torch.serialization.get_unsafe_globals_in_checkpoint(path):
            raise ValueError("Checkpoint requires an unreviewed pickle global")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(module, class_name)(**config[name + "_params"])
        model.load_state_dict(state, strict=True)
        models[name] = model.to(device).eval()
    path = root / "liveportrait-weights/liveportrait/retargeting_models/stitching_retargeting_module.pth"
    if torch.serialization.get_unsafe_globals_in_checkpoint(path):
        raise ValueError("Retargeting checkpoint requires an unreviewed pickle global")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = cls("stitching_retargeting_network", "StitchingRetargetingNetwork")(
        **config["stitching_retargeting_module_params"]["lip"]
    )
    model.load_state_dict(
        {k.removeprefix("module."): v for k, v in state["retarget_mouth"].items()}, strict=True
    )
    models["lip"] = model.to(device).eval()
    camera = importlib.import_module(package + ".utils.camera")
    return Renderer(models, camera, device)


class Renderer:
    def __init__(self, models, camera, device):
        self.models, self.camera, self.device = models, camera, device

    @torch.inference_mode()
    def render(self, crop_rgb, source_ratio, target_ratio):
        if crop_rgb.shape != (256, 256, 3) or crop_rgb.dtype != np.uint8:
            raise ValueError("Benchmark renderer needs a 256x256 uint8 RGB face crop")
        if (
            not np.isfinite([source_ratio, target_ratio]).all()
            or not 0 <= min(source_ratio, target_ratio) <= max(source_ratio, target_ratio) <= 1
        ):
            raise ValueError("Invalid source or target mouth ratio")
        x = torch.from_numpy(crop_rgb.copy()).permute(2, 0, 1)[None].float().to(self.device) / 255
        f = self.models["appearance_feature_extractor"](x)
        motion = self.models["motion_extractor"](x)
        angles = [self.camera.headpose_pred_to_degree(motion[k]) for k in ("pitch", "yaw", "roll")]
        rotation = self.camera.get_rotation_matrix(*angles)
        canonical, expression = motion["kp"].reshape(-1, 21, 3), motion["exp"].reshape(-1, 21, 3)
        source = motion["scale"][:, None] * (canonical @ rotation + expression)
        source[:, :, :2] += motion["t"][:, None, :2]
        ratios = torch.tensor([[source_ratio, target_ratio]], device=self.device, dtype=torch.float32)
        delta = self.models["lip"](torch.cat([source.reshape(1, -1), ratios], dim=1)).reshape(1, 21, 3)
        warped = self.models["warping_module"](f, kp_driving=source + delta, kp_source=source)
        result = self.models["spade_generator"](warped["out"])
        if result.shape != (1, 3, 512, 512) or not bool(torch.isfinite(result).all()):
            raise ValueError("Unexpected or nonfinite portrait output")
        return (result[0].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)


def crop_from_mesh(rgb, landmarks):
    """Recorded affine crop based on permitted mesh output, no restricted detector."""
    h, w = rgb.shape[:2]
    p = np.array([(x.x * w, x.y * h) for x in landmarks], dtype=np.float64)
    if p.shape != (478, 2) or not np.isfinite(p).all():
        raise ValueError("Unsupported face landmarks")
    lo, hi = np.quantile(p[:468], [0.01, 0.99], axis=0)
    side = float(np.max(hi - lo) * 2.2)
    if side < 80:
        raise ValueError("The source face is too small for the benchmark crop")
    center = (lo + hi) / 2 + np.array([0, -0.1 * (hi[1] - lo[1])])
    angle = np.degrees(np.arctan2(*(p[263] - p[33])[::-1]))
    matrix = cv2.getRotationMatrix2D(tuple(center), float(angle), 256 / side)
    matrix[:, 2] += np.array([128, 128]) - center
    crop = cv2.warpAffine(rgb, matrix, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mouth = float(np.linalg.norm(p[13] - p[14]) / max(np.linalg.norm(p[78] - p[308]), 1))
    if not 0 <= mouth <= 1:
        raise ValueError("Invalid measured source aperture")
    return crop, matrix, mouth
