"""Temporal-consistency analysis across sampled frames.

Face-swap pipelines process frames independently, so the replaced face tends to:
- jitter: landmark positions (relative to the face box) move erratically frame to frame;
- flicker: the face region's appearance changes more between frames than the rest of the scene;
- fluctuate: per-frame detector scores are volatile where a real face is consistently low.
These are heuristics; they are weighted modestly and stated with reference ranges.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from matplotlib.figure import Figure

  # OO API: no pyplot global state, safe in worker threads
from PIL import Image

from app.detectors.image.faces import Face, iou

# Folio palette for figures (matches frontend/design/direction-b-stripe-notion.md)
INK, MUTED, RULE, PAPER = "#1B1A17", "#625D54", "#E4E0D7", "#FCFBF9"
SCALE = ["#0B6B54", "#2A7460", "#6E685D", "#AE4617", "#9B2915"]  # authentic → AI


def _style(ax, fig=None) -> None:
    if fig is not None:
        fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.grid(axis="y", color=RULE, linewidth=0.6)
    ax.set_axisbelow(True)



def track_primary_face(frame_faces: list[list[dict]], iou_thresh: float = 0.3) -> list[int | None]:
    """Greedy IoU linking of the primary (largest) face across frames. Returns per-frame face index or None."""
    track: list[int | None] = []
    prev: Face | None = None
    for faces in frame_faces:
        if not faces:
            track.append(None)
            continue
        objs = [Face(f["x"], f["y"], f["w"], f["h"], f["score"], f.get("landmarks", [])) for f in faces]
        if prev is None:
            j = 0
        else:
            ious = [iou(prev, o) for o in objs]
            j = int(np.argmax(ious)) if max(ious) >= iou_thresh else 0
        prev = objs[j]
        track.append(j)
    return track


def landmark_jitter(frame_faces: list[list[dict]], track: list[int | None]) -> dict:
    """Std of frame-to-frame change in box-normalised landmark positions, and box-size volatility."""
    rel: list[np.ndarray] = []
    sizes: list[float] = []
    for faces, j in zip(frame_faces, track):
        if j is None or not faces[j].get("landmarks"):
            rel.append(None)  # type: ignore[arg-type]
            continue
        f = faces[j]
        lm = np.array(f["landmarks"], dtype=np.float32)
        rel.append(np.stack([(lm[:, 0] - f["x"]) / max(f["w"], 1), (lm[:, 1] - f["y"]) / max(f["h"], 1)], axis=1))
        sizes.append(float(f["w"]))
    deltas = [np.abs(b - a).mean() for a, b in zip(rel[:-1], rel[1:]) if a is not None and b is not None]
    if len(deltas) < 3:
        return {"frames_tracked": int(sum(r is not None for r in rel)), "landmark_jitter": None}
    d = np.array(deltas)
    # eye-distance consistency (inter-ocular distance relative to box width should be stable)
    return {
        "frames_tracked": int(sum(r is not None for r in rel)),
        "landmark_jitter": round(float(np.median(d)), 4),
        "landmark_jitter_p90": round(float(np.percentile(d, 90)), 4),
        "box_size_cv": round(float(np.std(sizes) / (np.mean(sizes) + 1e-6)), 4),
    }


def ring_box(f: dict, shape: tuple[int, ...], expand: float = 1.7) -> tuple[int, int, int, int]:
    """Box expanded around the face (hair, neck, shoulders) that moves with the head."""
    cx, cy = f["x"] + f["w"] / 2, f["y"] + f["h"] / 2
    w, h = f["w"] * expand, f["h"] * expand
    x0, y0 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
    x1, y1 = int(min(shape[1], cx + w / 2)), int(min(shape[0], cy + h / 2))
    return x0, y0, x1, y1


def face_flicker(frames: list[np.ndarray], frame_faces: list[list[dict]], track: list[int | None]) -> dict:
    """Inter-frame change inside the face vs in the surrounding ring (hair/neck) that moves with the head.

    Both are measured on 64px grey thumbnails of the respective region, so head motion cancels out;
    a blended face that is re-synthesised every frame changes more than the ring it sits in.
    """
    fvals, bvals = [], []
    prev_face = prev_bg = None
    for rgb, faces, j in zip(frames, frame_faces, track):
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if j is None:
            prev_face = prev_bg = None
            continue
        f = faces[j]
        face = cv2.resize(g[f["y"] : f["y"] + f["h"], f["x"] : f["x"] + f["w"]], (64, 64)).astype(np.float32)
        x0, y0, x1, y1 = ring_box(f, g.shape)
        ring = g[y0:y1, x0:x1].copy()
        ring[f["y"] - y0 : f["y"] - y0 + f["h"], f["x"] - x0 : f["x"] - x0 + f["w"]] = 0
        bg = cv2.resize(ring, (64, 64)).astype(np.float32)
        if prev_face is not None:
            fvals.append(float(np.abs(face - prev_face).mean()))
            bvals.append(float(np.abs(bg - prev_bg).mean()))
        prev_face, prev_bg = face, bg
    if len(fvals) < 3:
        return {"face_flicker_ratio": None}
    return {
        "face_change": round(float(np.median(fvals)), 3),
        "background_change": round(float(np.median(bvals)), 3),
        "face_flicker_ratio": round(float(np.median(fvals) / (np.median(bvals) + 1e-6)), 3),
    }


def score_volatility(probs: list[float | None]) -> dict:
    p = np.array([x for x in probs if x is not None], dtype=np.float32)
    if len(p) < 3:
        return {"score_std": None}
    return {"score_std": round(float(p.std()), 4), "score_mean": round(float(p.mean()), 4), "score_max": round(float(p.max()), 4),
            "flagged_fraction": round(float((p > 0.5).mean()), 4), "n": len(p)}


def timeline_plot(times: list[float], per_frame: list[dict]) -> np.ndarray:
    fig = Figure(figsize=(8, 2.8), dpi=110)
    ax = fig.subplots()
    pf = [f["p_frame"] for f in per_frame]
    _style(ax, fig)
    ax.plot(times, pf, lw=1.6, color=INK, label="fused per-frame p(AI)")
    face = [(t, f["p_face"]) for t, f in zip(times, per_frame) if f.get("p_face") is not None]
    if face:
        ax.plot([t for t, _ in face], [p for _, p in face], lw=1.1, ls="--", color=SCALE[4], label="face-swap detectors")
    ax.axhline(0.5, color=MUTED, ls=":", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("p(AI)")
    ax.legend(fontsize=8, loc="upper right", frameon=False, labelcolor=MUTED)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def contact_sheet(frames: list[np.ndarray], per_frame: list[dict], times: list[float], k: int = 6, tile: int = 256) -> np.ndarray:
    order = sorted(range(len(frames)), key=lambda i: per_frame[i]["p_frame"], reverse=True)[:k]
    order.sort()
    tiles = []
    for i in order:
        rgb = frames[i]
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        for f in per_frame[i]["faces"]:
            cv2.rectangle(vis, (f["x"], f["y"]), (f["x"] + f["w"], f["y"] + f["h"]), (0, 200, 255), max(2, rgb.shape[1] // 400))
        h, w = vis.shape[:2]
        s = tile / max(h, w)
        vis = cv2.resize(vis, (int(w * s), int(h * s)))
        canvas = np.zeros((tile + 28, tile, 3), np.uint8)
        canvas[: vis.shape[0], : vis.shape[1]] = vis
        cv2.putText(canvas, f"t={times[i]:.1f}s p={per_frame[i]['p_frame']:.2f}", (4, tile + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(canvas)
    if not tiles:
        return np.zeros((64, 64, 3), np.uint8)
    return cv2.cvtColor(np.concatenate(tiles, axis=1), cv2.COLOR_BGR2RGB)


def key_frame_crops(frames: list[np.ndarray], per_frame: list[dict], times: list[float], k: int = 6, width: int = 480) -> list[tuple[str, np.ndarray, float]]:
    """Individual key-frame images (highest fused score first, then chronological) for the UI filmstrip."""
    order = sorted(range(len(frames)), key=lambda i: per_frame[i]["p_frame"], reverse=True)[:k]
    order.sort()
    out = []
    for n, i in enumerate(order, 1):
        rgb = frames[i]
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        for f in per_frame[i]["faces"]:
            cv2.rectangle(vis, (f["x"], f["y"]), (f["x"] + f["w"], f["y"] + f["h"]), (245, 245, 240), max(1, rgb.shape[1] // 640))
        h, w = vis.shape[:2]
        s_ = width / w
        vis = cv2.resize(vis, (width, int(h * s_)))
        out.append((f"keyframe_{n}.jpg", cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), times[i]))
    return out
