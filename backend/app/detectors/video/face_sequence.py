"""Consecutive face-crop inspection statistic, promoted from the frozen v6 experiment.

Responses are feature distances, never authenticity probabilities. The frozen
benchmark scripts remain unchanged so promotion can be compared independently.
"""
from __future__ import annotations

import numpy as np
import torch

from app.detectors.video.decode import FrameTable
from app.detectors.video.tracking import DinoEmbedder

REVIEW_POINT = 0.007326911389827726
METHOD = "face-sequence-change/1"
SIZE = 112

INNER = np.zeros((8, 8), dtype=bool)

INNER[2:6, 2:6] = True

RING = ~INNER.flatten()

CORE = INNER.flatten()

def patch_features(crops: dict[int, np.ndarray], dino: DinoEmbedder, *, check=None) -> dict[int, np.ndarray]:
    keys = sorted(crops)
    features = {}
    dtype = next(dino.model.parameters()).dtype
    for start in range(0, len(keys), 12):
        if check is not None:
            check("consecutive face features")
        indices = keys[start:start+12]
        pixels = np.stack([crops[i] for i in indices]).astype(np.float32) / 255
        values = torch.from_numpy(((pixels - dino.MEAN) / dino.STD).transpose(0, 3, 1, 2).copy()).to(dino.device, dtype)
        with torch.inference_mode():
            tokens = dino.model(pixel_values=values).last_hidden_state[:, 1:].float()
            tokens = torch.nn.functional.normalize(tokens, dim=-1).cpu().numpy()
        if tokens.shape[1] != 64:
            raise ValueError("Unexpected face-feature grid")
        features.update(zip(indices, tokens, strict=True))
    return features

def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    scale = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return 1 - np.sum(a * b, axis=-1) / np.maximum(scale, 1e-8)

def sequence_windows(centers: set[int], tracks: dict, patches: dict, table: FrameTable) -> list[dict]:
    windows = []
    for track_id, samples in tracks.items():
        index = {sample.frame_index for sample in samples}
        blocks = []
        for i in sorted(centers):
            group = (i-1, i, i+1)
            if all(k in index and k in patches for k in group) and table.seconds(i+1) - table.seconds(i-1) <= .13:
                blocks.append((i, np.mean([patches[k] for k in group], axis=0)))
        if len(blocks) < 7:
            continue
        times = np.array([table.seconds(i) for i, _ in blocks])
        values = np.stack([value for _, value in blocks])
        for start in range(2, len(blocks)-4):
            for end in range(start+3, len(blocks)-1):
                span = times[end-1] - times[start]
                if span > 2.6:
                    break
                if span < .52 or np.max(np.diff(times[start-2:end+2])) > .55:
                    continue
                left, core, right = (np.mean(part, axis=0) for part in (values[start-2:start], values[start:end], values[end:end+2]))
                left_shift, right_shift, normal = distance(core, left), distance(core, right), distance(left, right)
                inside = min(float(left_shift[CORE].mean()), float(right_shift[CORE].mean()))
                outside = max(float(left_shift[RING].mean()), float(right_shift[RING].mean()))
                response = inside - .60 * outside - .35 * float(normal[CORE].mean())
                windows.append({"track_id": track_id, "start_s": float(times[start]), "end_s": float(times[end-1]),
                                "response": response, "inner_change": inside, "outer_change": outside,
                                "flank_change": float(normal[CORE].mean()),
                                "centers": [i for i, _ in blocks[start-2:end+2]]})
    return sorted(windows, key=lambda w: (-w["response"], w["start_s"], w["track_id"]))[:12]
