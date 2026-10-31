"""Batched, artifact-free scoring of many frames (used by the video pipeline and the eval harness).

Per frame: faces, per-model probabilities, face/background forensic ratios, and fused
general / face / overall probabilities. Models are run in batches across frames.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.core import deadline
from app.core.config import settings
from app.core.logging import get_logger
from app.detectors.common.fusion import fuse_probability
from app.detectors.image import forensics
from app.detectors.image.classifiers import borrow_classifier, specs
from app.detectors.image.dfdc import borrow_dfdc
from app.detectors.image.faces import Face, detect_faces, face_input
from app.detectors.image.univfd import borrow_univfd

log = get_logger(__name__)
BATCH = 8


def score_frames(frames: list[np.ndarray], *, with_forensics: bool = True, video: bool = False) -> list[dict]:
    n = len(frames)
    out: list[dict] = [{"faces": [], "scores": {}, "forensics": {}} for _ in range(n)]
    faces_per_frame: list[list[Face]] = []
    for i, rgb in enumerate(frames):
        f = detect_faces(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        faces_per_frame.append(f)
        out[i]["faces"] = [x.to_dict() for x in f]
    pils = [Image.fromarray(f) for f in frames]
    general: list[tuple[str, float]] = []  # (model, weight) for general detectors
    face_models: list[tuple[str, float]] = []

    if settings.univfd_enabled:
        try:
            with borrow_univfd() as model:
                try:
                    probs = model.predict_batch(pils)
                finally:
                    del model
            for i, p in enumerate(probs):
                out[i]["scores"]["univfd"] = round(float(p), 4)
            general.append(("univfd", settings.univfd_weight))
        except Exception as e:
            log.warning("univfd.batch_failed", error=str(e))

    face_idx = [i for i in range(n) if faces_per_frame[i]]
    face_inputs = [face_input(frames[i], faces_per_frame[i][0]) for i in face_idx]
    face_crops = [Image.fromarray(a) for a, _, _ in face_inputs]
    upscaled_frac = float(np.mean([u for _, _, u in face_inputs])) if face_inputs else 0.0

    for spec in specs():
        deadline.check("frame classification")
        try:
            with borrow_classifier(spec) as clf:
                try:
                    if spec.face_oriented:
                        if not face_crops:
                            continue
                        preds = []
                        for b in range(0, len(face_crops), BATCH):
                            preds += clf.predict(face_crops[b : b + BATCH])
                        for i, pr in zip(face_idx, preds):
                            out[i]["scores"][spec.short] = round(pr["p_ai"], 4)
                        face_models.append((spec.short, face_frame_weight(spec.short, spec.weight, video=video, upscaled_frac=upscaled_frac)))
                    else:
                        preds = []
                        for b in range(0, n, BATCH):
                            preds += clf.predict(pils[b : b + BATCH])
                        for i, pr in enumerate(preds):
                            out[i]["scores"][spec.short] = round(pr["p_ai"], 4)
                        general.append((spec.short, frame_weight(spec.short, spec.weight, video=video)))
                finally:
                    del clf
        except Exception as e:
            log.warning("classifier.batch_failed", model=spec.repo_id, error=str(e))

    if settings.dfdc_enabled and face_idx:
        try:
            with borrow_dfdc() as model:
                try:
                    preds = model.predict([(frames[i], faces_per_frame[i][0]) for i in face_idx])
                finally:
                    del model
            for i, p in zip(face_idx, preds):
                out[i]["scores"]["dfdc_effnet_b7"] = round(float(p), 4)
            small = float(np.mean([faces_per_frame[i][0].w < settings.dfdc_min_face_px for i in face_idx]))
            face_models.append(("dfdc_effnet_b7", dfdc_frame_weight(small)))
        except Exception as e:
            log.warning("dfdc.batch_failed", error=str(e))

    for i in range(n):
        fb = faces_per_frame[i][0].box() if faces_per_frame[i] else None
        if with_forensics:
            try:
                m, _ = forensics.noise_residual(frames[i], face_box=fb)
                out[i]["forensics"].update({k: m[k] for k in ("noise_face_bg_ratio", "noise_flat_fraction") if k in m})
                m2, _ = forensics.ela(frames[i], face_box=fb)
                if "ela_face_bg_ratio" in m2:
                    out[i]["forensics"]["ela_face_bg_ratio"] = m2["ela_face_bg_ratio"]
            except Exception as e:  # pragma: no cover
                log.warning("forensics.frame_failed", error=str(e))
    fuse_frame_scores(out, general, face_models)
    return out


def frame_weight(short: str, base: float, *, video: bool) -> float:
    """A whole-frame model's weight, times the v0.3 per-model video override."""
    return base * (settings.video_weight_overrides if video else {}).get(short, 1.0)


def face_frame_weight(short: str, base: float, *, video: bool, upscaled_frac: float) -> float:
    """A face-oriented model's weight: video override, and × face_upscaled_weight when most face crops were upscaled."""
    w = frame_weight(short, base, video=video)
    return w * settings.face_upscaled_weight if upscaled_frac > 0.5 else w


def dfdc_frame_weight(small_frac: float) -> float:
    return settings.dfdc_weight * (0.4 if small_frac > 0.5 else 1.0)


def fuse_frame_scores(out: list[dict], general: list[tuple[str, float]], face_models: list[tuple[str, float]]) -> None:
    """Per-frame fused general, face and overall probabilities from the per-model frame scores (rounded to 4 places)."""
    for f in out:
        sc = f["scores"]
        pg = fuse_probability([(sc[m], w) for m, w in general if m in sc])
        pf = fuse_probability([(sc[m], w) for m, w in face_models if m in sc]) if f["faces"] else None
        f["p_general"] = round(pg, 4)
        f["p_face"] = round(pf, 4) if pf is not None else None
        f["p_frame"] = round(fuse_probability([(sc[m], w) for m, w in general + face_models if m in sc]), 4)
