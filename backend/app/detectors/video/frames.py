"""Video probing (ffprobe) and uniform frame sampling (OpenCV)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from app.core.logging import get_logger
from app.detectors.video.decode import _input

log = get_logger(__name__)


def probe(path: Path) -> dict:
    if not shutil.which("ffprobe"):
        log.warning("ffprobe.missing")
        return {}
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-print_format", "json", "-show_format", "-show_streams",
             _input(path)],
            capture_output=True, text=True, timeout=60, check=True, stdin=subprocess.DEVNULL,
        )
        j = json.loads(r.stdout)
    except Exception as e:
        log.warning("ffprobe.failed", error=str(e))
        return {}
    v = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in j.get("streams", []) if s.get("codec_type") == "audio"), None)
    fmt = j.get("format", {})
    num, den = (v.get("r_frame_rate", "0/1").split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den) else 0.0
    tags = {**{k.lower(): str(val)[:200] for k, val in (fmt.get("tags") or {}).items()},
            **{k.lower(): str(val)[:200] for k, val in (v.get("tags") or {}).items()}}
    return {
        "container": fmt.get("format_name"),
        "duration": float(fmt.get("duration", 0) or 0),
        "bit_rate": int(fmt.get("bit_rate", 0) or 0),
        "codec": v.get("codec_name"),
        "profile": v.get("profile"),
        "width": v.get("width"),
        "height": v.get("height"),
        "fps": round(fps, 3),
        "nb_frames": int(v.get("nb_frames", 0) or 0),
        "pix_fmt": v.get("pix_fmt"),
        "has_audio": a is not None,
        "audio_codec": a.get("codec_name") if a else None,
        "tags": tags,
    }


def sample_frames(path: Path, n: int) -> list[tuple[int, float, np.ndarray]]:
    """Return n uniformly spaced (frame_index, timestamp_s, rgb) tuples."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Video could not be decoded")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not (1.0 <= fps <= 120.0):  # WebM/MKV with variable frame rate often report 1000 fps and bogus counts
        fps = 25.0
        total = 0
    out: list[tuple[int, float, np.ndarray]] = []
    if total <= 0:  # unknown length: decode sequentially, keep an evenly thinned reservoir, then subsample
        keep: list[tuple[int, float, np.ndarray]] = []
        step, idx = 1, 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if idx % step == 0:
                keep.append((idx, idx / fps, cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
                if len(keep) > 4 * n:  # thin the reservoir and halve the sampling rate
                    keep = keep[::2]
                    step *= 2
            idx += 1
            if idx > 200_000:
                break
        cap.release()
        if not keep:
            raise ValueError("No frames could be decoded")
        picks = np.linspace(0, len(keep) - 1, num=min(n, len(keep)), dtype=int)
        return [keep[i] for i in picks]
    picks = np.linspace(0, max(total - 1, 0), num=min(n, total), dtype=int)
    for i in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        out.append((int(i), float(i / fps), cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if len(out) < min(4, n):  # seeking failed (broken index): fall back to sequential decoding
        cap = cv2.VideoCapture(str(path))
        keep: list[tuple[int, float, np.ndarray]] = []
        idx = 0
        while True:
            ok, fr = cap.read()
            if not ok or idx > 20_000:
                break
            if idx % 5 == 0:
                keep.append((idx, idx / fps, cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            idx += 1
        cap.release()
        if len(keep) > len(out):
            picks = np.linspace(0, len(keep) - 1, num=min(n, len(keep)), dtype=int)
            out = [keep[i] for i in picks]
    if not out:
        raise ValueError("No frames could be decoded")
    return out
