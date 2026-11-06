"""Pinned offline CakeLens model and bounded source-frame sampling."""
from __future__ import annotations

import math
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
from prepare_video_upgrade_eval import sha

CHECKPOINT = "27d8a8ea23dad701d26c1d59486fbd48a553b968421ddd30e9a505fd000c231d"
SOURCE_FILES = {
    "constants.py": "29a141ab26d75097eff384eb969f1bebef09072b033c086b718e845c8fd30901",
    "data_types.py": "17407d0fb987be914a4330972087c53f28e5a1835320574372395d5bf7ad6618",
    "utils.py": "490bbb652abd7963612449e5254b30a61a01a5a9621dea127a23d4f296e20aa9",
    "model.py": "25db1dba8f5a065108a2dda883b0abe3fe3996177e313d6022f5525535ab10e1",
}
WINDOW = 9
SIZE = 512


def pinned_model(source: Path) -> types.ModuleType:
    for filename, digest in SOURCE_FILES.items():
        if sha(source / filename) != digest:
            raise ValueError(f"unexpected upstream file: {filename}")
    package = types.ModuleType("_pinned_cakelens")
    package.__path__ = []
    sys.modules[package.__name__] = package
    for filename in SOURCE_FILES:
        path = source / filename
        name = f"{package.__name__}.{path.stem}"
        module = types.ModuleType(name)
        module.__package__ = package.__name__
        sys.modules[name] = module
        setattr(package, path.stem, module)
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return package.model


def read_windows(path: Path):
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            raise ValueError("video has no readable frame count")
        groups = math.ceil(total / WINDOW)
        # Use the first, middle and last of the same nine-frame groups as the
        # publisher's full scan. This bounds cost on long sources.
        for group in np.linspace(0, groups - 1, min(groups, 3), dtype=int):
            start = int(group) * WINDOW
            if start and not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
                raise ValueError(f"could not seek to frame {start}")
            frames = []
            for index in range(start, min(start + WINDOW, total)):
                success, frame = capture.read()
                if not success:
                    raise ValueError(f"could not decode frame {index}")
                frames.append(cv2.cvtColor(frame[:SIZE, :SIZE], cv2.COLOR_BGR2RGB))
            yield start, np.stack(frames)
    finally:
        capture.release()


class CakeLensScorer:
    def __init__(self, source: Path, checkpoint: Path, device: str):
        import torch

        if sha(checkpoint) != CHECKPOINT:
            raise ValueError("unexpected CakeLens checkpoint")
        model = pinned_model(source)
        with torch.device("meta"):
            self.model = model.Model()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        self.model.load_state_dict(state["model_state_dict"], strict=True, assign=True)
        self.torch = torch
        self.transform = model.make_transformer()
        self.device = device
        self.model.to(device).eval()
        cv2.setNumThreads(1)

    def score(self, path: Path) -> dict:
        started = time.perf_counter()
        windows = []
        with self.torch.inference_mode():
            for start, frames in read_windows(path):
                frame_count, height, width, _ = frames.shape
                missing = (WINDOW - frame_count, SIZE - height, SIZE - width)
                n, h, w = missing
                clip = self.torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
                # Publisher centers only the zero padding around a cropped
                # input. Pixels are still cropped at the top left.
                clip = self.torch.nn.functional.pad(
                    clip, (w // 2, w - w // 2, h // 2, h - h // 2, 0, 0, n // 2, n - n // 2),
                )
                tensor = self.transform(clip).unsqueeze(0).to(self.device)
                score = float(self.model(tensor).sigmoid()[0, 0].item())
                if not math.isfinite(score):
                    raise ValueError("non-finite response")
                windows.append({"start_frame": start, "frames": frame_count, "score": score})
        mean = float(np.mean([window["score"] for window in windows]))
        p = min(max(mean, 1e-7), 1 - 1e-7)
        return {
            "status": "ok", "score": mean, "logit": math.log(p / (1 - p)),
            "runtime_s": round(time.perf_counter() - started, 4), "windows": windows,
        }
