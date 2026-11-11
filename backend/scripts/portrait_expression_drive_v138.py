"""Research-only same-person expression driving; not a qualified detector/generator.

Uses the retained MIT human animation modules. No additional weights or face
recognition are loaded. Native cropping/compositing and actual neural response
qualification remain the caller's separately frozen responsibilities.
"""

from __future__ import annotations

import numpy as np


def expression_points(source_points, source_expression, driving_expression, source_scale):
    """Keep source identity/pose/translation; change its expression displacement.

    This independent arithmetic helper uses NumPy and no model. It is also the
    scalar/reference definition for the CPU renderer's native tensor equation.
    """
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (source_points, source_expression, driving_expression)
    ]
    if any(value.shape != (1, 21, 3) or not np.isfinite(value).all() for value in arrays):
        raise ValueError("Three finite corresponding 21-point arrays are required")
    scale = np.asarray(source_scale, dtype=np.float64)
    if scale.shape != (1, 1, 1) or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("A finite positive source scale is required")
    return arrays[0] + scale * (arrays[2] - arrays[1])


class ExpressionRenderer:
    """Unqualified CPU float32 benchmark bridge over the existing human loader."""

    def __init__(self, assets):
        import torch

        from scripts.liveportrait_mouth_benchmark import load

        self.renderer = load(assets, "cpu")
        if any(
            p.device.type != "cpu" or p.dtype != torch.float32
            for module in self.renderer.models.values()
            for p in module.parameters()
        ):
            raise ValueError("Explicit CPU float32 parameters are required")

    def render(self, source_crop, driving_crop):
        import torch

        if any(crop.shape != (256, 256, 3) or crop.dtype != np.uint8 for crop in (source_crop, driving_crop)):
            raise ValueError("Two corresponding 256-pixel uint8 RGB face crops are required")
        models, camera = self.renderer.models, self.renderer.camera
        with torch.inference_mode():
            images = [
                torch.from_numpy(crop.copy()).permute(2, 0, 1)[None].float() / 255
                for crop in (source_crop, driving_crop)
            ]
            appearance = models["appearance_feature_extractor"](images[0])
            source = models["motion_extractor"](images[0])
            driver = models["motion_extractor"](images[1])
            angles = [camera.headpose_pred_to_degree(source[k]) for k in ("pitch", "yaw", "roll")]
            rotation = camera.get_rotation_matrix(*angles)
            canonical = source["kp"].reshape(1, 21, 3)
            expression = source["exp"].reshape(1, 21, 3)
            driving_expression = driver["exp"].reshape(1, 21, 3)
            scale = source["scale"][:, None]
            source_points = scale * (canonical @ rotation + expression)
            source_points[:, :, :2] += source["t"][:, None, :2]
            driving_points = source_points + scale * (driving_expression - expression)
            reference = expression_points(
                source_points.numpy(), expression.numpy(), driving_expression.numpy(), scale.numpy()
            )
            if not np.allclose(driving_points.numpy(), reference, rtol=1e-6, atol=1e-6):
                raise ValueError("Tensor expression displacement differs from the NumPy reference")
            warped = models["warping_module"](appearance, kp_driving=driving_points, kp_source=source_points)
            result = models["spade_generator"](warped["out"])
            if result.shape != (1, 3, 512, 512) or not bool(torch.isfinite(result).all()):
                raise ValueError("Unexpected or nonfinite expression-driven output")
            return (result[0].permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
