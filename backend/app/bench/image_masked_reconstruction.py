# SPDX-License-Identifier: Apache-2.0
"""Contextual JPEG measurements using two masked reconstructions at every pixel."""

from __future__ import annotations

import hashlib
import io

MODEL = "microsoft/swin-base-simmim-window6-192"
REVISION = "6c7c8dea55b6bc1fb8e20c21dae660cc2a45cada"
ASSETS = {
    "README.md": "a3d6c80b1242a31f7134c15a2e27414c9facb40dba72f77ab857c6ad7c2bdc5a",
    "config.json": "9895750c1a4ab4de222685821a351e931e9609fa535538cd939bda8338023f95",
    "preprocessor_config.json": "c2bb2c1dedbc98b293a197477abdcd7d6a1d2ae053b3fe5fb8394c7d962dba61",
    "model.safetensors": "f90e6f096a935b50cc267eb5b4f91fc3f35eaf3a5faa8a9c368cbdcce2ced44b",
}
SIZE, BLOCK, PATCH, CELL = 192, 32, 4, 8
MEAN, STD = (.485, .456, .406), (.229, .224, .225)
ARMS = ("excess", "deficit", "texture_only", "paired_context")


class Reconstruction:
    def __init__(self, assets, device):
        import cv2
        import numpy as np
        import torch
        from transformers import SwinForMaskedImageModeling

        for name, expected in ASSETS.items():
            with (assets / name).open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                    raise ValueError(f"The reconstruction asset differs from the pinned Microsoft release: {name}")
        torch.set_num_threads(3)
        cv2.setNumThreads(1)
        self.device = device
        model, loaded = SwinForMaskedImageModeling.from_pretrained(
            assets, local_files_only=True, use_safetensors=True, attn_implementation="eager", output_loading_info=True,
        )
        if any(loaded.values()):
            raise ValueError("The reconstruction model did not load the complete publisher checkpoint")
        self.model = model.to(device).eval()
        self.mean = np.asarray(MEAN, dtype=np.float32)
        self.std = np.asarray(STD, dtype=np.float32)
        y, x = np.indices((SIZE // BLOCK, SIZE // BLOCK))
        # Each location is hidden in two different neighborhoods.
        a, b = (x + y) % 2 == 0, (x + y // 2) % 2 == 0
        masks = np.stack((a, ~a, b, ~b))
        self.pixels = masks.repeat(BLOCK, axis=1).repeat(BLOCK, axis=2)
        patches = masks.repeat(BLOCK // PATCH, axis=1).repeat(BLOCK // PATCH, axis=2)
        self.hidden_patches = torch.from_numpy(patches.reshape(4, -1)).to(device)

    def _decode(self, body):
        import cv2
        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(body), formats=["JPEG"]) as image:
            if image.mode != "RGB" or not 192 <= min(image.size) <= max(image.size) <= 768:
                raise ValueError("Reconstruction requires a bounded, native RGB JPEG")
            height, width = image.height, image.width
            source = np.asarray(image)
        scale = SIZE / max(height, width)
        w, h = round(scale * width), round(scale * height)
        resized = cv2.resize(source, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        y, x = (SIZE - h) // 2, (SIZE - w) // 2
        padded = cv2.copyMakeBorder(resized, y, SIZE - h - y, x, SIZE - w - x, cv2.BORDER_CONSTANT, value=MEAN)
        return padded, np.s_[y:y+h, x:x+w], (height, width)

    def observe(self, body):
        import numpy as np
        import torch

        image, crop, source_shape = self._decode(body)
        normalized = (image - self.mean) / self.std
        tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).to(self.device)
        with torch.inference_mode():
            predictions = self.model(tensor[None].repeat(4, 1, 1, 1), bool_masked_pos=self.hidden_patches).reconstruction
            rendered = predictions.permute(0, 2, 3, 1).cpu().numpy() * self.std + self.mean
        if rendered.shape != (4, SIZE, SIZE, 3) or not np.isfinite(rendered).all():
            raise ValueError("The four reconstructions do not match the expected pixel grid")
        first = np.where(self.pixels[0, ..., None], rendered[0], rendered[1])
        second = np.where(self.pixels[2, ..., None], rendered[2], rendered[3])
        return features(image[crop], first[crop], second[crop], source_shape)


def _pooled(field, shape):
    import cv2
    import numpy as np

    # Preserve the source's right and bottom partial cells. Interpolate in
    # source geometry first when the input edge is not on the native grid.
    h, w = shape
    if h % CELL or w % CELL:
        native = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)
        native = np.pad(native, ((0, -h % CELL), (0, -w % CELL)), mode="edge")
        return cv2.resize(native, (native.shape[1] // CELL, native.shape[0] // CELL), interpolation=cv2.INTER_AREA)
    return cv2.resize(field, (w // CELL, h // CELL), interpolation=cv2.INTER_AREA)


def features(image, first, second, shape):
    import cv2
    import numpy as np

    average = (first + second) / 2
    disagreement = np.mean(np.abs(first - second), axis=2)
    error = (np.abs(first - image) + np.abs(second - image)) / 2
    error = error.mean(axis=2)
    def blur(value, sigma):
        return cv2.GaussianBlur(value, (0, 0), sigma)

    def energy(value, sigma):
        return np.abs(value - blur(value, sigma)).mean(axis=2)

    detail, coarse = energy(image, .85), energy(image, 2.6)
    prediction_detail = energy(average, .85)
    intensity = image.mean(axis=2)
    color = image.max(axis=2) - image.min(axis=2)
    native = (intensity, color, np.log1p(32 * detail), np.log1p(32 * coarse))
    log_error = np.log(.025 + error)
    contextual = (log_error, np.log(.025 + disagreement), log_error - np.log(.025 + disagreement + .6 * detail),
                  np.log(.01 + detail) - np.log(.01 + prediction_detail),
                  np.mean(np.abs(blur(image - average, 2.)), axis=2))
    cells = [_pooled(value, shape) for value in (*native, *contextual)]
    # Use only surrounding cells for the comparative response.
    ring = np.ones((9, 9), np.float32)
    ring[2:7, 2:7] = 0
    ring /= ring.sum()
    relative = [cell - cv2.filter2D(cell, -1, ring, borderType=cv2.BORDER_REFLECT_101) for cell in cells]
    return np.stack(cells[:len(native)] + relative[:len(native)]), np.stack(cells + relative), relative[len(native) + 2]


def proposals(fields, heads, shape):
    import numpy as np

    from app.detectors.image.jpeg_boundary_localizer import proposals as region

    texture, context, response = fields
    maps = {"excess": response, "deficit": -response}
    for name, measurements in (("texture_only", texture), ("paired_context", context)):
        head = heads[name]
        scaled = (measurements - np.asarray(head["center"])[:, None, None]) / np.asarray(head["scale"])[:, None, None]
        logits = np.einsum("c,chw->hw", np.asarray(head["weights"]), scaled) + head["bias"]
        maps[name] = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    output = {}
    for name, heat in maps.items():
        score, mask = region(heat)
        output[name] = score, mask.repeat(CELL, axis=0).repeat(CELL, axis=1)[:shape[0], :shape[1]]
    return output
