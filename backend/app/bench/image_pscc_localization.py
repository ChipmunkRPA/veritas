# SPDX-License-Identifier: Apache-2.0
"""Run the pinned public PSCC-Net localizer and measure native JPEG regions."""

from __future__ import annotations

import hashlib
import io
from copy import deepcopy
from pathlib import Path

REVISION = "53e5ff77d8dc5feddda060cd085f9b765761f816"
PUBLISHER = f"https://raw.githubusercontent.com/proteus1991/PSCC-Net/{REVISION}"
CELL = 8
ARMS = ("fixed_mask", "finest_map", "scale_agreement")
SOURCE_FILES = {
    "LICENSE": "6644ee3539b171a7ebd39193786e0654456c57df6ba0b6f389fdb9a7b09685c9",
    "test.py": "07d9911da874fb8dfe86658b0fb22f8736f23ae60997532c05b3415a3ab07458",
    "utils/config.py": "24a52c41f3f227256be74fd6dec17bec6907fb4007321dbeae0ed926ee1423ed",
    "utils/load_vdata.py": "eecc6bec8920e30367ddcf63f72e396c492750fd65ce6ccd833ada301af893d7",
    "models/seg_hrnet_config.py": "a933009348e8394a96c0c89d930d18ffb0ec10f693e58f4787a2ed3cb939a7d5",
    "models/seg_hrnet.py": "ccb71be379e591eb84329c34500ecc015a157c92ee73327e392b6187672d78e9",
    "models/NLCDetection.py": "325d4a9402e4fe640ecfb0abd24a9bc4b490bc567fc80d3e93ad64936b9757a6",
    "checkpoint/HRNet_checkpoint/HRNet.pth": "d3b21edc4930187a6801cc818bd7b999fb5d8078d8f2e2193572e91ea5160096",
    "checkpoint/NLCDetection_checkpoint/NLCDetection.pth": "11ea3461253cf059b299ad4b6b89008485f94a2d2b2da83ec28c2282a095b00b",
}


def _asset(root, name):
    body = (root / name).read_bytes()
    if hashlib.sha256(body).hexdigest() != SOURCE_FILES[name]:
        raise ValueError(f"The publisher file differs from the pinned release: {name}")
    return body


def _definitions(root, name, *, remove=b"", imports=None):
    body = _asset(root, name)
    if remove:
        if body.count(remove) != 1:
            raise ValueError(f"The compatibility change no longer matches: {name}")
        body = body.replace(remove, b"")
    namespace = {"__name__": Path(name).stem, **(imports or {})}
    exec(compile(body, str(root / name), "exec"), namespace)
    return namespace


def _parameters(model, data):
    import torch

    stored = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not stored or any(not key.startswith("module.") or not torch.is_tensor(value) for key, value in stored.items()):
        raise ValueError("The official checkpoint uses an unexpected parameter format")
    model.load_state_dict({key.removeprefix("module."): value for key, value in stored.items()}, strict=True)


class _Config(dict):
    """Read the checkpoint's fixed configuration without an extra runtime package."""

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

    def clone(self):
        return {key: value.clone() if isinstance(value, _Config) else deepcopy(value) for key, value in self.items()}


class Localizer:
    def __init__(self, assets, *, device="cpu", threads=3):
        import cv2
        import torch

        torch.set_num_threads(threads)
        cv2.setNumThreads(1)
        self.device = device
        for name in ("LICENSE", "test.py", "utils/config.py", "utils/load_vdata.py"):
            _asset(assets, name)
        config = _definitions(
            assets, "models/seg_hrnet_config.py", remove=b"from yacs.config import CfgNode as CN\n",
            imports={"CN": _Config},
        )["get_hrnet_cfg"]
        network = _definitions(
            assets, "models/seg_hrnet.py", remove=b"        last_inp_channels = np.int(np.sum(pre_stage_channels))\n",
        )["HighResolutionNet"]
        head = _definitions(
            assets, "models/NLCDetection.py", remove=b"from models.seg_hrnet_config import get_hrnet_cfg\n",
            imports={"get_hrnet_cfg": config},
        )["NLCDetection"]
        self.backbone = network(config())
        self.head = head({"crop_size": (256, 256)})
        _parameters(self.backbone, _asset(assets, "checkpoint/HRNet_checkpoint/HRNet.pth"))
        _parameters(self.head, _asset(assets, "checkpoint/NLCDetection_checkpoint/NLCDetection.pth"))
        self.backbone.to(device).eval()
        self.head.to(device).eval()

    def maps(self, body):
        import numpy as np
        import torch
        import torch.nn.functional as functional
        from PIL import Image

        with Image.open(io.BytesIO(body), formats=["JPEG"]) as original:
            if original.mode != "RGB" or not 128 <= min(original.size) <= max(original.size) <= 768:
                raise ValueError("The benchmark needs a bounded RGB JPEG")
            rgb = np.array(original, dtype=np.float32) / 255
        height, width, _ = rgb.shape
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(self.device)
        padding = (0, -width % CELL, 0, -height % CELL)
        tensor = functional.pad(tensor, padding, mode="reflect") if any(padding) else tensor
        with torch.inference_mode():
            masks = self.head(self.backbone(tensor))
            native = [functional.interpolate(mask, tensor.shape[-2:], mode="bilinear", align_corners=True)
                      [0, 0, :height, :width].cpu().numpy() for mask in masks]
        if len(native) != 4 or any(mask.shape != rgb.shape[:2] or not np.isfinite(mask).all() for mask in native):
            raise ValueError("The localizer did not return four finite source-aligned maps")
        return native


def selected_regions(maps):
    import numpy as np

    from app.detectors.image.jpeg_boundary_localizer import proposals

    fine = maps[0]
    height, width = fine.shape
    rows, columns = height // CELL, width // CELL
    agreement = np.sqrt(fine * np.minimum.reduce(maps[1:]))
    fixed = fine >= .5
    output = {"fixed_mask": (1. if fixed.any() else None, fixed)}
    for name, response in (("finest_map", fine), ("scale_agreement", agreement)):
        cells = response[:rows * CELL, :columns * CELL].reshape(rows, CELL, columns, CELL).mean(axis=(1, 3))
        score, region = proposals(cells)
        full = np.zeros_like(fixed)
        full[:rows * CELL, :columns * CELL] = region.repeat(CELL, axis=0).repeat(CELL, axis=1)
        output[name] = (score, full)
    return output
