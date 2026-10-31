"""v0.3 frame scoring (`image/scoring.score_frames`) as v2 plugins: faces on every sampled frame, then each whole-frame
or face model on its own.

- **Faces.** YuNet on each sampled frame (`detect_faces`, ≥ 48 px). One regions exhibit carries every face with
  `frame_pts` and its five landmarks; the faces stage turns it into per-frame face lists.
- **Whole-frame models** score every sampled frame (UnivFD on one native centre crop, HF classifiers on the whole frame).
- **Face models** score the largest face of each face frame (`face_input` crops, DFDC 380 px crops) and report the
  fraction of upscaled crops or small faces. The legacy fuser applies the v0.3 video weight overrides and multipliers.
"""
from __future__ import annotations

from typing import ClassVar

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings
from app.detectors.image import dfdc, univfd
from app.detectors.image.adapters.common import revisions
from app.detectors.image.classifiers import HFImageClassifier
from app.detectors.image.dfdc import DFDCClassifier
from app.detectors.image.faces import detect_faces, face_input
from app.detectors.image.univfd import UnivFD
from app.detectors.video.adapters.common import (
    ALL_PROFILES,
    CPU_ONLY,
    FULL_DEEP,
    H1,
    H2,
    frame_faces,
    has_faces,
    has_frames,
    sampled_frames,
    series_exhibit,
    to_face,
    video_spec,
)
from app.engine.detector import (
    ActivationUse,
    Applicability,
    Detector,
    DetectorResult,
    InputUnit,
    RunContext,
    UnitScore,
)

BATCH = 8  # v0.3 classifier batch size over frames


class FrameFacesDetector(Detector):
    MODEL_ID = "yunet.2023mar"
    spec = video_spec("vid.faces.yunet", "YuNet face detector on sampled frames (OpenCV Zoo 2023mar)", InputUnit.FRAME_BATCH, (),
                      ("regions",), family="face-detection", model_ids=(MODEL_ID,), lane="cpu", devices=CPU_ONLY, timeout_s=60.0)

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(self.MODEL_ID, 1, "frame", "One sampled frame, detector long side at most 1280 px"),)

    def applicable(self, ctx: RunContext) -> Applicability:
        return has_frames(ctx)

    def run(self, ctx: RunContext) -> DetectorResult:
        yunet = ctx.lease.get(self.MODEL_ID).obj
        regions, units = [], []
        for f in sampled_frames(ctx):
            faces = detect_faces(cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR), detector=yunet, min_size=settings.face_min_size)
            units.append(UnitScore(f"frame:{f.index}", float(len(faces)), "score"))
            for j, face in enumerate(faces, 1):
                regions.append({"region": {"kind": "bbox", "bbox": {"x": face.x, "y": face.y, "w": face.w, "h": face.h},
                                           "frame_pts": f.pts_ms, "landmarks": [list(p) for p in face.landmarks]},
                                "score": float(face.score), "label": f"frame {f.index} face {j}"})
        exhibit_ids = []
        if regions:
            exhibit_ids.append(ctx.exhibits.regions(regions, title="Faces in sampled frames", media_ref="frame",
                                                    caption="YuNet boxes and five landmarks per sampled frame; frame_pts in ms."))
        face_frames = sum(1 for u in units if u.raw > 0)
        return DetectorResult(
            status="ok", unit_scores=units, observed=f"{len(regions)} face(s) in {face_frames} of {len(units)} sampled frames",
            reference="Faces of at least 48 px; score threshold 0.7", reasoning="Selects the inputs of the face detectors.",
            metrics={"faces": float(len(regions)), "face_frames": float(face_frames), "frames": float(len(units))},
            exhibit_ids=exhibit_ids, model_revisions=revisions(ctx, self.MODEL_ID))


def _frames_result(ctx: RunContext, spec_name: str, units: list[UnitScore], quality: dict[str, float], model_ids: tuple[str, ...],
                   target: str) -> DetectorResult:
    mean = float(np.mean([u.raw for u in units]))
    exhibit = series_exhibit(ctx, "p", [(u.quality["time_s"], u.raw) for u in units], f"{spec_name} per sampled frame",
                             "Raw detector output per frame; uncalibrated.")
    side = "synthetic" if mean >= 0.5 else "real"
    return DetectorResult(
        status="ok", unit_scores=units, aggregate=UnitScore("global", mean, "probability", quality=quality),
        observed=f"mean output {mean:.3f} over {len(units)} {target} (max {max(u.raw for u in units):.3f})",
        reference="Binary detector; decision boundary at 0.5 per frame",
        reasoning=f"On average the detector places the {target} on the {side} side of its decision boundary.",
        metrics={"mean": mean, "max": float(max(u.raw for u in units)), **quality}, exhibit_ids=[exhibit],
        model_revisions=revisions(ctx, *model_ids))


class FrameUnivFDDetector(Detector):
    CLIP, PROBE = "clip.vit_l14_openai", "univfd.fc_probe"
    spec = video_spec("vid.frames.univfd", "UnivFD per sampled frame (CLIP ViT-L/14 linear probe)", InputUnit.FRAME_BATCH, (H1,),
                      ("series",), family="learned-whole-image", model_ids=(CLIP, PROBE), profiles=FULL_DEEP,
                      version="0.4.0")  # CLIP with QuickGELU (B-4a)

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return tuple(ActivationUse(m, univfd.FRAME_BATCH, "224 px frame crop", "One centre crop per frame, batches of sixteen")
                     for m in (self.CLIP, self.PROBE))

    def applicable(self, ctx: RunContext) -> Applicability:
        return has_frames(ctx)

    def run(self, ctx: RunContext) -> DetectorResult:
        univfd = UnivFD(model=ctx.lease.get(self.CLIP).obj, fc=ctx.lease.get(self.PROBE).obj, device=ctx.device)
        frames = sampled_frames(ctx)
        probs = univfd.predict_batch([Image.fromarray(f.rgb) for f in frames])
        units = [UnitScore(f"frame:{f.index}", float(p), "probability", quality={"time_s": f.time_s}) for f, p in zip(frames, probs, strict=True)]
        return _frames_result(ctx, self.spec.name, units, {"frames": float(len(units))}, (self.CLIP, self.PROBE), "sampled frames")


class _FrameClassifier(Detector):
    MODEL_ID: ClassVar[str]
    POSITIVE: ClassVar[str]

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(self.MODEL_ID, BATCH, "classifier input", "Pinned processor resolution, batches of eight frame/face inputs"),)

    def applicable(self, ctx: RunContext) -> Applicability:
        return has_frames(ctx)

    def classifier(self, ctx: RunContext) -> HFImageClassifier:
        processor, model = ctx.lease.get(self.MODEL_ID).obj
        return HFImageClassifier.from_parts(processor, model, self.POSITIVE, ctx.device)

    def run(self, ctx: RunContext) -> DetectorResult:
        clf, frames = self.classifier(ctx), sampled_frames(ctx)
        images = [Image.fromarray(f.rgb) for f in frames]
        preds: list[dict] = []
        for b in range(0, len(images), BATCH):
            ctx.deadline.check("frame classification")
            preds += clf.predict(images[b:b + BATCH])
        units = [UnitScore(f"frame:{f.index}", float(p["p_ai"]), "probability", quality={"time_s": f.time_s})
                 for f, p in zip(frames, preds, strict=True)]
        return _frames_result(ctx, self.spec.name, units, {"frames": float(len(units))}, (self.MODEL_ID,), "sampled frames")


def _face_input_target(units: list[UnitScore]) -> str:
    """Describe only the input scope recorded by each face-oriented frame score."""
    flags = [unit.quality.get("whole_image_input") for unit in units]
    if not flags or any(isinstance(flag, bool) or not isinstance(flag, int | float) or flag not in (0.0, 1.0)
                        for flag in flags):
        return "face-selected inputs (crop or whole-frame scope not recorded)"
    if all(flag == 0.0 for flag in flags):
        return "face crops of the largest detected face"
    if all(flag == 1.0 for flag in flags):
        return "whole portrait frames selected by face detection"
    return "face-selected inputs (face crops and whole portrait frames)"


class _FaceClassifier(_FrameClassifier):
    def applicable(self, ctx: RunContext) -> Applicability:
        return has_faces(ctx)

    def run(self, ctx: RunContext) -> DetectorResult:
        clf, frames, faces = self.classifier(ctx), sampled_frames(ctx), frame_faces(ctx)
        face_frames = [f for f in frames if faces[f.index]]
        inputs = [face_input(f.rgb, to_face(faces[f.index][0])) for f in face_frames]
        crops = [Image.fromarray(a) for a, _, _ in inputs]
        preds: list[dict] = []
        for b in range(0, len(crops), BATCH):
            ctx.deadline.check("face classification")
            preds += clf.predict(crops[b:b + BATCH])
        units = [UnitScore(f"frame:{f.index}", float(p["p_ai"]), "probability",
                           quality={"time_s": f.time_s, "whole_image_input": float(whole), "upscaled": float(up)})
                 for f, p, (_, whole, up) in zip(face_frames, preds, inputs, strict=True)]
        upscaled = float(np.mean([up for _, _, up in inputs]))
        return _frames_result(ctx, self.spec.name, units, {"face_frames": float(len(units)), "upscaled_fraction": upscaled},
                              (self.MODEL_ID,), _face_input_target(units))


class FrameHaywoodsloanDetector(_FrameClassifier):
    MODEL_ID, POSITIVE = "hf.haywoodsloan.ai_image_detector_deploy", "artificial"
    spec = video_spec("vid.frames.hf.haywoodsloan", "haywoodsloan ai-image-detector-deploy per sampled frame (Swin-v2)",
                      InputUnit.FRAME_BATCH, (H1,), ("series",), family="learned-whole-image", model_ids=(MODEL_ID,),
                      profiles=FULL_DEEP)


class FrameAteeqqDetector(_FrameClassifier):
    MODEL_ID, POSITIVE = "hf.ateeqq.ai_vs_human_image_detector", "ai"
    spec = video_spec("vid.frames.hf.ateeqq", "Ateeqq ai-vs-human-image-detector per sampled frame (SigLIP)", InputUnit.FRAME_BATCH,
                      (H1,), ("series",), family="learned-whole-image", model_ids=(MODEL_ID,), profiles=FULL_DEEP)


class FrameDima806Detector(_FaceClassifier):
    MODEL_ID, POSITIVE = "hf.dima806.deepfake_vs_real_image_detection", "fake"
    spec = video_spec("vid.frames.hf.dima806", "dima806 deepfake_vs_real_image_detection on face frames (ViT-B/16)",
                      InputUnit.FACE_CROP, (H1,), ("series",), family="learned-face", model_ids=(MODEL_ID,),
                      needs=("frames", "faces"), profiles=FULL_DEEP)


class FrameWvolfDetector(_FaceClassifier):
    MODEL_ID, POSITIVE = "hf.wvolf.vit_deepfake_detection", "fake"
    spec = video_spec("vid.frames.hf.wvolf", "Wvolf ViT_Deepfake_Detection on face frames (ViT-B/16)", InputUnit.FACE_CROP, (H1,),
                      ("series",), family="learned-face", model_ids=(MODEL_ID,), needs=("frames", "faces"), profiles=FULL_DEEP)


class FrameDFDCB7Detector(Detector):
    MODEL_IDS = ("dfdc.effnet_b7_555", "dfdc.effnet_b7_999")
    spec = video_spec("vid.frames.dfdc_b7", "DFDC EfficientNet-B7 on face frames (selimsef, two-checkpoint ensemble)",
                      InputUnit.FACE_CROP, (H2,), ("series",), family="learned-face", model_ids=MODEL_IDS, needs=("frames", "faces"),
                      profiles=ALL_PROFILES)

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return tuple(ActivationUse(m, dfdc.BATCH, "380 px face crop", "Backend batches eight faces; ensemble members execute sequentially")
                     for m in self.MODEL_IDS)

    def applicable(self, ctx: RunContext) -> Applicability:
        return has_faces(ctx)

    def run(self, ctx: RunContext) -> DetectorResult:
        clf = DFDCClassifier(members=[ctx.lease.get(m).obj for m in self.MODEL_IDS], device=ctx.device)
        frames, faces = sampled_frames(ctx), frame_faces(ctx)
        face_frames = [f for f in frames if faces[f.index]]
        largest = [to_face(faces[f.index][0]) for f in face_frames]
        preds = [float(p) for p in clf.predict([(f.rgb, face) for f, face in zip(face_frames, largest, strict=True)])]
        units = [UnitScore(f"frame:{f.index}", p, "probability", quality={"time_s": f.time_s, "face_px": float(face.w)})
                 for f, p, face in zip(face_frames, preds, largest, strict=True)]
        small = float(np.mean([face.w < settings.dfdc_min_face_px for face in largest]))
        return _frames_result(ctx, self.spec.name, units, {"face_frames": float(len(units)), "small_face_fraction": small},
                              self.MODEL_IDS, "face crops of the largest face")
