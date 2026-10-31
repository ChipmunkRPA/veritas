"""Image analysis pipeline: provenance → faces → learned detectors → forensics → fusion → evidence."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.core import deadline
from app.core.config import settings
from app.core.evidence import Signal
from app.core.logging import get_logger
from app.detectors.common.fusion import fuse, prob_to_signal_fields
from app.detectors.image import forensics
from app.detectors.image.classifiers import borrow_classifier, specs, to_pil
from app.detectors.image.dfdc import borrow_dfdc
from app.detectors.image.faces import Face, detect_faces, face_input
from app.detectors.image.gradcam import vit_gradcam
from app.detectors.image.metadata import metadata_signals
from app.detectors.image.univfd import borrow_univfd

try:  # HEIC/HEIF support
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # pragma: no cover
    pass

log = get_logger(__name__)


MAX_PIXELS = 80_000_000  # hard refuse above this (decompression-bomb guard)
MAX_SIDE = 4096  # analyse a downscaled copy above this; detectors resize to ≤ 380 px anyway

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def load_image(path: Path) -> tuple[Image.Image, np.ndarray, dict]:
    """Open safely. Returns (pil, rgb array, info) where info notes any downscaling applied for analysis."""
    with Image.open(path) as src:
        src.load()  # raises DecompressionBombError above MAX_PIXELS
        fmt, quant = src.format, getattr(src, "quantization", None)
        im = ImageOps.exif_transpose(src)
        if im is src:
            im = src.copy()
    im.format = fmt  # exif_transpose returns a copy without format/quantization; keep them for the report + JPEG signal
    if quant is not None:
        im.quantization = quant  # type: ignore[attr-defined]
    info = {"original_width": im.width, "original_height": im.height, "downscaled": False}
    work = im
    if max(im.size) > MAX_SIDE:
        work = im.copy()
        work.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        info["downscaled"] = True
    rgb = np.asarray(work.convert("RGB"))
    return im, rgb, info


def _save(art_dir: Path | None, name: str, rgb: np.ndarray) -> list[str]:
    if art_dir is None:
        return []
    Image.fromarray(rgb).save(art_dir / name, quality=88)
    return [name]


def classifier_signals(rgb: np.ndarray, faces: list[Face], art_dir: Path | None, *, fast: bool) -> tuple[list[Signal], list[dict]]:
    signals: list[Signal] = []
    face_results: list[dict] = [{"face": f.to_dict(), "scores": {}} for f in faces]
    whole = to_pil(rgb)
    face_inputs = [face_input(rgb, f) for f in faces[:6]]
    face_crops = [to_pil(a) for a, _, _ in face_inputs]
    any_whole = any(w for _, w, _ in face_inputs)
    all_upscaled = bool(face_inputs) and all(u for _, _, u in face_inputs)

    # --- UnivFD (general-purpose, generator-agnostic) ---
    if settings.univfd_enabled:
        try:
            with borrow_univfd() as model:
                try:
                    u = model.predict([whole])[0]
                finally:
                    del model
            p = u["p_ai"]
            signals.append(
                Signal(
                    id="univfd_clip", name="UnivFD — CLIP ViT-L/14 universal fake detector", category="model",
                    observed=f"p(fake)={p:.3f} (mean of {u['crops']} native-resolution crops" + (", image upscaled — unreliable)" if u["upscaled"] else ")"),
                    reference="Trained on ProGAN; generalises to GAN and diffusion families; 0.5 decision boundary",
                    weight=settings.univfd_weight * (0.3 if u["upscaled"] else 1.0), **prob_to_signal_fields(p),
                    reasoning=("CLIP feature-space probe places this image on the synthetic side of the boundary; generator-agnostic feature statistics resemble training-set fakes." if p >= 0.5 else "CLIP feature-space probe places this image with real photographs; its semantic-feature statistics match natural images."),
                    details={"model": "WisconsinAIVision/UniversalFakeDetect ViT-L/14 fc", **u},
                )
            )
        except Exception as e:
            log.warning("univfd.failed", error=str(e))

    # --- HuggingFace classifiers ---
    for spec in specs():
        try:
            with borrow_classifier(spec) as clf:
                try:
                    if spec.face_oriented:
                        if face_crops:
                            preds = clf.predict(face_crops)
                            for fr, pr in zip(face_results, preds):
                                fr["scores"][spec.short] = round(pr["p_ai"], 4)
                            p = max(pr["p_ai"] for pr in preds)
                            target = f"{len(face_crops)} face input(s){' (portrait, whole image)' if any_whole else ''}, max taken"
                            weight = spec.weight * (settings.face_upscaled_weight if all_upscaled else 1.0)
                            if all_upscaled:
                                target += "; crop upscaled — reduced weight"
                        else:
                            p = clf.predict([whole])[0]["p_ai"]
                            target = "whole image (no face found — out of the model's training distribution)"
                            weight = spec.weight * 0.4
                    else:
                        p = clf.predict([whole])[0]["p_ai"]
                        target = "whole image"
                        weight = spec.weight
                    arts: list[str] = []
                    if settings.gradcam_enabled and not fast and art_dir is not None and p >= 0.5 and hasattr(clf.model, "vit"):
                        src = np.asarray(face_crops[0]) if (spec.face_oriented and face_crops) else rgb
                        cam = vit_gradcam(clf, src)
                        if cam is not None:
                            arts = _save(art_dir, f"gradcam_{spec.short}.jpg", cam)
                    signals.append(
                        Signal(
                            id=f"clf_{spec.short}", name=f"Classifier — {spec.repo_id}", category="face" if spec.face_oriented else "model",
                            observed=f"p(ai)={p:.3f} on {target}", reference="Fine-tuned binary classifier; 0.5 decision boundary",
                            weight=weight, artifacts=arts, **prob_to_signal_fields(p),
                            reasoning=(f"The {'face-oriented' if spec.face_oriented else 'AI-image'} classifier assigns {p:.0%} probability to the synthetic class" + ("; the Grad-CAM overlay shows the regions that drove this decision." if arts else ".")),
                            details={"architecture": type(clf.model).__name__, "positive_label": clf.id2label[clf.pos_idx]},
                        )
                    )
                except Exception as e:
                    log.warning("classifier.failed", model=spec.repo_id, error=str(e))
                finally:
                    del clf
        except Exception as e:
            log.warning("classifier.load_failed", model=spec.repo_id, error=str(e))
            continue
    # --- DFDC EfficientNet-B7 (face-swap, video-grade) ---
    if settings.dfdc_enabled and faces:
        try:
            with borrow_dfdc() as model:
                try:
                    preds = model.predict([(rgb, f) for f in faces[:6]])
                finally:
                    del model
            for fr, pr in zip(face_results, preds):
                fr["scores"]["dfdc_effnet_b7"] = round(pr, 4)
            p = max(preds)
            small_face = max(f.w for f in faces[:6]) < settings.dfdc_min_face_px
            signals.append(
                Signal(
                    id="clf_dfdc_effnet_b7", name="DFDC-winner EfficientNet-B7 face-swap classifier", category="face",
                    observed=f"p(fake)={p:.3f} (max over {len(preds)} face(s))",
                    reference="Trained on the 100k-video Deepfake Detection Challenge corpus; 0.5 decision boundary",
                    weight=settings.dfdc_weight * (0.4 if small_face else 1.0), **prob_to_signal_fields(p),
                    reasoning=("The face exhibits blending/warping statistics that this detector associates with face replacement." if p >= 0.5 else "The strongest open face-swap detector finds no face-replacement statistics on the face.")
                    + (" The face is small and was upscaled for the detector, so this reading is down-weighted." if small_face else ""),
                    details={"model": f"selimsef/dfdc_deepfake_challenge ensemble of {len(settings.dfdc_urls)} B7 checkpoints", "per_face": [round(x, 4) for x in preds]},
                )
            )
        except Exception as e:
            log.warning("dfdc.failed", error=str(e))
    return signals, face_results


def forensic_signals(rgb: np.ndarray, pil: Image.Image, faces: list[Face], art_dir: Path | None, *, fast: bool) -> list[Signal]:
    sig: list[Signal] = []
    fb = faces[0].box() if faces else None

    # --- spectrum ---
    try:
        m, art = forensics.spectrum_analysis(rgb)
        # generated images: periodic peaks and/or abnormal high-frequency tail; the heuristic is capped to [0.40, 0.75]
        p, _p_peaks, p_slope = forensics.spectrum_score(m)
        arts = [] if fast else _save(art_dir, "spectrum.png", art)
        sig.append(
            Signal(
                id="fft_spectrum", name="Frequency-domain spectrum analysis", category="forensic",
                observed=f"slope={m['spectral_slope']}, periodic peaks={m['periodic_peaks']} (max {m['peak_strength_sigma']}σ), HF ratio={m['hf_ratio']}",
                reference="Camera images: smooth ~1/f spectrum (slope ≈ −0.8…−1.4), no peaks >3σ in the high band",
                weight=0.45, artifacts=arts, **prob_to_signal_fields(p),
                reasoning=("Periodic spectral peaks consistent with generator up-sampling grids were detected" if m["periodic_peaks"] > 0 else "No periodic up-sampling peaks were detected")
                + (" and the spectral slope departs from the natural-image range." if p_slope > 0.6 else "; the spectral slope is within the natural-image range."),
                details=m,
            )
        )
    except Exception as e:
        log.warning("forensics.spectrum_failed", error=str(e))

    # --- ELA ---
    try:
        m, art = forensics.ela(rgb, face_box=fb)
        ratio = m.get("ela_face_bg_ratio")
        p, raw = forensics.ela_score(m)  # face/background ratio when a face was measured, else block inconsistency
        if ratio is not None:
            obs = f"face/background error ratio={ratio}, block CV={m['ela_block_cv']}"
            reason = ("The face region shows a compression error level distinctly different from its surroundings, which is characteristic of a pasted or regenerated face." if raw > 0.6 else "Compression error is uniform between the face and the background, as expected for a single-generation image.")
        else:
            obs = f"mean error={m['ela_mean']}, block CV={m['ela_block_cv']}"
            reason = ("Error levels vary strongly between regions, suggesting composited content with different compression histories." if raw > 0.6 else "Error levels are spatially uniform, consistent with a single compression history.")
        arts = [] if fast else _save(art_dir, "ela.jpg", art)
        sig.append(
            Signal(
                id="ela", name="Error Level Analysis (JPEG recompression)", category="forensic", observed=obs,
                reference="Single-generation image: uniform error; face vs surrounding-ring ratio ≈ 1.0 (0.7–1.4); block CV < 0.9",
                weight=0.35, artifacts=arts, **prob_to_signal_fields(p), reasoning=reason, details=m,
            )
        )
    except Exception as e:
        log.warning("forensics.ela_failed", error=str(e))

    # --- noise residual ---
    try:
        m, art = forensics.noise_residual(rgb, face_box=fb)
        ratio = m.get("noise_face_bg_ratio")
        p, p_flat, p_ratio = forensics.noise_score(m)  # over-smooth residual; face/background inconsistency when measured
        parts = ["Residual noise is unusually flat, typical of rendered rather than sensor-captured pixels" if p_flat > 0.6 else "Sensor-like residual noise is present across the image"]
        if p_ratio is not None:
            parts.append("the face region's noise level is inconsistent with the background, a hallmark of face replacement" if p_ratio > 0.6 else "face and background noise levels are consistent")
        arts = [] if fast else _save(art_dir, "noise_map.jpg", art)
        sig.append(
            Signal(
                id="noise_residual", name="Noise-residual consistency", category="forensic",
                observed=f"σ={m['noise_std']}, local CV={m['noise_local_cv']}, flat fraction={m['noise_flat_fraction']}" + (f", face/bg ratio={ratio}" if ratio is not None else ""),
                reference="Camera: flat fraction < 0.6, face vs surrounding-ring noise ratio 0.6–1.6",
                weight=0.35, artifacts=arts, **prob_to_signal_fields(p), reasoning="; ".join(parts) + ".", details=m,
            )
        )
    except Exception as e:
        log.warning("forensics.noise_failed", error=str(e))

    # --- JPEG facts (informational) ---
    try:
        j = forensics.jpeg_info(pil)
        if j.get("jpeg"):
            sig.append(
                Signal(
                    id="jpeg_quality", name="JPEG quantisation", category="metadata",
                    observed=f"estimated quality ≈ {j['estimated_quality']}", reference="Informational: heavy recompression degrades all detectors",
                    ai_probability=None, weight=0.0, direction="inconclusive", strength="none",
                    reasoning="Quality estimate from the luminance quantisation table; low quality (<70) reduces the reliability of pixel-level signals.",
                    details=j,
                )
            )
    except Exception:
        pass
    return sig


def analyze_image_array(rgb: np.ndarray, pil: Image.Image | None, art_dir: Path | None, *, fast: bool = False, path: Path | None = None) -> dict:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    signals: list[Signal] = []
    meta: dict = {}

    if path is not None and pil is not None and not fast:
        t = time.perf_counter()
        ms, meta = metadata_signals(path, pil)
        signals += ms
        timings["metadata"] = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    faces = detect_faces(bgr)
    timings["faces"] = round(time.perf_counter() - t, 3)
    if not fast and art_dir is not None and faces:
        vis = bgr.copy()
        for i, f in enumerate(faces):
            cv2.rectangle(vis, (f.x, f.y), (f.x + f.w, f.y + f.h), (0, 200, 255), max(2, rgb.shape[1] // 400))
            cv2.putText(vis, f"face {i+1}", (f.x, max(12, f.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, rgb.shape[1] / 1600), (0, 200, 255), 2)
        _save(art_dir, "faces.jpg", vis[:, :, ::-1])

    t = time.perf_counter()
    deadline.check("face detection")
    cs, face_results = classifier_signals(rgb, faces, art_dir, fast=fast)
    signals += cs
    timings["classifiers"] = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    deadline.check("classification")
    signals += forensic_signals(rgb, pil if pil is not None else Image.fromarray(rgb), faces, art_dir, fast=fast)
    timings["forensics"] = round(time.perf_counter() - t, 3)

    # per-face fused probability (from face-oriented classifiers only)
    for fr in face_results:
        if fr["scores"]:
            fr["p_ai"] = round(float(np.mean(list(fr["scores"].values()))), 4)

    # specialist asymmetry: a face-swap detector saying "no swap" is weak evidence about a *fully synthetic*
    # image, so negatives from face specialists count at half weight on stills — and at a quarter when two or
    # more whole-image detectors agree strongly that the image is synthetic (a generated image is not a swap).
    general_strong = sum(1 for x in signals if x.category == "model" and x.ai_probability is not None and x.ai_probability >= 0.9)
    for x in signals:
        if x.category == "face" and x.ai_probability is not None and x.ai_probability < 0.5:
            factor = 0.25 if general_strong >= 2 else 0.5
            x.weight = round(x.weight * factor, 3)
            x.reasoning += " (Negative result from a face-oriented detector; reduced weight — it does not rule out a fully synthetic image" + (", and whole-image detectors agree the image is generated.)" if general_strong >= 2 else ".)")

    # hypothesis separation: a positive face-swap finding is not refuted by whole-image "real scene" evidence
    face_sigs = [x for x in signals if x.category == "face" and x.ai_probability is not None]
    if faces and face_sigs:
        top_face = max(face_sigs, key=lambda x: x.weight)
        if top_face.ai_probability >= 0.6:
            for x in signals:
                if x.category == "model" and x.ai_probability is not None and x.ai_probability < 0.5:
                    x.weight = round(x.weight * 0.4, 3)
                    x.reasoning += " Down-weighted: general scene evidence does not resolve the differing face-oriented score."

    small = min(rgb.shape[0], rgb.shape[1]) < 224
    if small:
        for s_ in signals:
            if s_.category in ("model", "face", "forensic"):
                s_.weight = round(s_.weight * 0.5, 3)
        signals.append(
            Signal(
                id="resolution_gate", name="Low resolution", category="statistical",
                observed=f"{rgb.shape[1]}x{rgb.shape[0]} px", reference="Detectors are trained on ≥224 px inputs",
                ai_probability=None, weight=0.0, direction="inconclusive", strength="none",
                reasoning="The image is smaller than the detectors' native input size; it was upscaled, which itself introduces synthetic-looking statistics. All pixel-level signals were halved in weight.",
            )
        )
    verdict = fuse(signals, subject="image")
    if small:
        verdict.caveats.insert(0, "Low-resolution input: pixel-level detector reliability is reduced.")
    if faces:
        verdict.reasoning.insert(0, f"{len(faces)} face(s) detected; each detector observation records whether it scored face inputs or the whole image.")
    else:
        verdict.caveats.insert(0, "No face was detected; any whole-image scores from face-oriented classifiers were down-weighted.")
    timings["total"] = round(time.perf_counter() - t0, 3)
    return {
        "verdict": verdict.model_dump(),
        "signals": [s.model_dump() for s in signals],
        "faces": face_results,
        "metadata": meta,
        "artifacts": sorted({a for s in signals for a in s.artifacts} | ({"faces.jpg"} if faces and not fast and art_dir else set())),
        "timings": timings,
    }


def analyze_image(path: Path, art_dir: Path) -> dict:
    pil, rgb, info = load_image(path)
    res = analyze_image_array(rgb, pil, art_dir, fast=False, path=path)
    res["image"] = {"width": int(rgb.shape[1]), "height": int(rgb.shape[0]), "mode": pil.mode, "format": pil.format, **info}
    if info["downscaled"]:
        res["verdict"]["caveats"].insert(0, f"The image ({info['original_width']}×{info['original_height']}) was analysed at ≤ {MAX_SIDE} px on the long side; face boxes are in analysed-image coordinates.")
    return res
