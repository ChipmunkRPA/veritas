"""Learned detectors as v2 plugins: UnivFD, the Hugging Face classifiers, the DFDC EfficientNet-B7 ensemble (v0.3),
Community Forensics (I-6), DDA (I-7), SPAI (I-8) and GenD on faces (I-9).

The adapters report raw scores and quality facts only. The legacy fuser (`stages/image.fuse`) applies the v0.3 weights
and multipliers: upscaled input, no face, small face, face-specialist asymmetry, hypothesis separation, resolution gate.
Community Forensics, DDA, SPAI and GenD report logits, which the legacy fuser does not average: they reach a verdict only
through a calibrated fuser after admission (ML §3.2).
"""
from __future__ import annotations

from typing import Any, ClassVar

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import commfor, dda, dfdc, gend, native_crops, spai, univfd
from app.detectors.image.adapters.common import (
    ALL_PROFILES,
    H1,
    H2,
    analysis_rgb,
    coordinate_space,
    face_boxes,
    image_spec,
    revisions,
)
from app.detectors.image.classifiers import HFImageClassifier
from app.detectors.image.dfdc import DFDCClassifier
from app.detectors.image.faces import crop_box, face_input
from app.detectors.image.gradcam import vit_gradcam_map
from app.detectors.image.univfd import UnivFD
from app.engine.detector import (
    ActivationUse,
    Applicability,
    Detector,
    DetectorResult,
    InputUnit,
    Profile,
    RunContext,
    UnitScore,
)

MAX_FACES = 6  # v0.3 scores at most six faces per image
ATTRIBUTION_GRID_PX = 256
ATTRIBUTION_CAPTION = "Where a classifier looked. Not a map of manipulation."


class UnivFDDetector(Detector):
    CLIP, PROBE = "clip.vit_l14_openai", "univfd.fc_probe"
    spec = image_spec("img.univfd", "UnivFD (CLIP ViT-L/14 linear probe)", "learned-whole-image", InputUnit.IMAGE, (H1,),
                      model_ids=(CLIP, PROBE), version="0.4.0")  # CLIP with QuickGELU (B-4a)

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return tuple(ActivationUse(m, univfd.CROPS_PER_SCALE * univfd.MAX_SCALES, "224 px crop",
                                   "Up to five crops at each of two existing image scales") for m in (self.CLIP, self.PROBE))

    def run(self, ctx: RunContext) -> DetectorResult:
        univfd = UnivFD(model=ctx.lease.get(self.CLIP).obj, fc=ctx.lease.get(self.PROBE).obj, device=ctx.device)
        u = univfd.predict([Image.fromarray(analysis_rgb(ctx))])[0]
        p = float(u["p_ai"])
        aggregate = UnitScore("global", p, "probability",
                              quality={"input_upscaled": float(u["upscaled"]), "crops": float(u["crops"]), "scales": float(u["scales"])})
        side = "synthetic" if p >= 0.5 else "natural-image"
        return DetectorResult(
            status="ok", unit_scores=[aggregate], aggregate=aggregate,
            observed=f"sigmoid output {p:.3f} (mean of {u['crops']} native-resolution 224 px crops"
                     + (" from an upscaled image)" if u["upscaled"] else ")"),
            reference="Linear probe trained on ProGAN real and fake images; decision boundary at 0.5",
            reasoning=f"The CLIP feature-space probe places this image on the {side} side of its decision boundary.",
            metrics={"p_max": float(u["p_max"]), "crops": float(u["crops"]), "scales": float(u["scales"])},
            model_revisions=revisions(ctx, self.CLIP, self.PROBE))


def _attribution(ctx: RunContext, cam: np.ndarray, shape: tuple[int, ...], box: tuple[int, int, int, int] | None,
                 name: str) -> str:
    """Store a Grad-CAM map on a ≤ 256 px grid over the analysis copy; a face-crop map is masked to its crop box."""
    h, w = shape[:2]
    scale = ATTRIBUTION_GRID_PX / max(h, w)
    gw, gh = max(1, round(w * scale)), max(1, round(h * scale))
    cam = cam.astype(np.float32)
    mask = None
    if box is None:
        values = cv2.resize(cam, (gw, gh), interpolation=cv2.INTER_CUBIC)
    else:
        x0, y0, x1, y1 = box
        gx0, gy0 = min(gw - 1, int(x0 * scale)), min(gh - 1, int(y0 * scale))
        gx1, gy1 = min(gw, max(gx0 + 1, round(x1 * scale))), min(gh, max(gy0 + 1, round(y1 * scale)))
        values = np.zeros((gh, gw), np.float32)
        values[gy0:gy1, gx0:gx1] = cv2.resize(cam, (gx1 - gx0, gy1 - gy0), interpolation=cv2.INTER_CUBIC)
        mask = np.zeros((gh, gw), bool)
        mask[gy0:gy1, gx0:gx1] = True
    return ctx.exhibits.heatmap(np.clip(values, 0.0, 1.0), semantics="attribution", title=f"Where {name} looked",
                                caption=ATTRIBUTION_CAPTION, value_range=(0.0, 1.0), mask=mask,
                                coordinate_space=coordinate_space(ctx))


class _HFClassifier(Detector):
    MODEL_ID: ClassVar[str]
    POSITIVE: ClassVar[str]  # substring of the synthetic-class label
    FACE_ORIENTED: ClassVar[bool] = False

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(self.MODEL_ID, MAX_FACES if self.FACE_ORIENTED else 1, "classifier input",
                              "Pinned processor resolution; at most six faces or one whole image",
                              ("Optional single-input Grad-CAM backward/retention has no separately validated peak estimate.",)),)

    def run(self, ctx: RunContext) -> DetectorResult:
        processor, model = ctx.lease.get(self.MODEL_ID).obj
        clf = HFImageClassifier.from_parts(processor, model, self.POSITIVE, ctx.device)
        rgb = analysis_rgb(ctx)
        faces = face_boxes(ctx)[:MAX_FACES] if self.FACE_ORIENTED else []
        units: list[UnitScore] = []
        if faces:
            inputs = [face_input(rgb, f) for f in faces]
            preds = clf.predict([Image.fromarray(a) for a, _, _ in inputs])
            units = [UnitScore(f"face:{i}", float(pr["p_ai"]), "probability",
                               quality={"whole_image_input": float(whole), "upscaled": float(up)})
                     for i, (pr, (_, whole, up)) in enumerate(zip(preds, inputs, strict=True))]
            # The attribution must explain the exact input whose score becomes the aggregate.
            # With several faces, the first crop need not be the highest-scoring crop.
            selected = max(range(len(units)), key=lambda i: units[i].raw)
            p = units[selected].raw
            all_upscaled, any_whole = all(up for _, _, up in inputs), any(whole for _, whole, _ in inputs)
            quality = {"n_faces": float(len(faces)), "face_inputs_upscaled": float(all_upscaled), "whole_image_input": float(any_whole)}
            target = (f"{len(faces)} face input(s){' (portrait, whole image)' if any_whole else ''}, maximum taken"
                      + ("; every face input was upscaled to the model input" if all_upscaled else ""))
            cam_input = inputs[selected][0]
            cam_box = None if inputs[selected][1] else crop_box(rgb.shape, faces[selected])
            cam_scope = units[selected].unit_ref
        else:
            p = float(clf.predict([Image.fromarray(rgb)])[0]["p_ai"])
            quality = {"no_face": 1.0} if self.FACE_ORIENTED else {}
            target = ("whole image (no face found; outside the model's training distribution)" if self.FACE_ORIENTED
                      else "whole image")
            cam_input, cam_box = rgb, None
            cam_scope = "global"
        aggregate = UnitScore("global", p, "probability", quality=quality)
        exhibit_ids = []
        details = {}
        if ctx.profile != Profile.QUICK and p >= 0.5:
            ctx.deadline.check("grad-cam")
            cam = vit_gradcam_map(clf, cam_input)
            if cam is not None:
                exhibit_ids.append(_attribution(ctx, cam, rgb.shape, cam_box, self.spec.name))
                details["attribution"] = {
                    "version": "input-binding/1", "input_scope_ref": cam_scope, "input_score": p,
                    "semantics": "classifier_attribution", "coverage_kind": "input_footprint",
                    "coordinate_space": coordinate_space(ctx),
                    "crop_box_xyxy": list(cam_box) if cam_box is not None else None,
                }
        kind = "face-oriented" if self.FACE_ORIENTED else "AI-image"
        side = "synthetic" if p >= 0.5 else "real"
        return DetectorResult(
            status="ok", unit_scores=units or [aggregate], aggregate=aggregate,
            observed=f"softmax output {p:.3f} for the '{clf.id2label[clf.pos_idx]}' class on {target}",
            reference="Fine-tuned binary classifier; decision boundary at 0.5",
            reasoning=f"The {kind} classifier places its input on the {side} side of its decision boundary"
                      + ("; the attribution map shows where it looked." if exhibit_ids else "."),
            metrics={"p_ai": p, **quality}, details=details, exhibit_ids=exhibit_ids,
            model_revisions=revisions(ctx, self.MODEL_ID))


class AteeqqDetector(_HFClassifier):
    MODEL_ID, POSITIVE = "hf.ateeqq.ai_vs_human_image_detector", "ai"
    spec = image_spec("img.hf.ateeqq", "Ateeqq ai-vs-human-image-detector (SigLIP)", "learned-whole-image", InputUnit.IMAGE,
                      (H1,), ("heatmap",), model_ids=(MODEL_ID,))


class HaywoodsloanDetector(_HFClassifier):
    MODEL_ID, POSITIVE = "hf.haywoodsloan.ai_image_detector_deploy", "artificial"
    spec = image_spec("img.hf.haywoodsloan", "haywoodsloan ai-image-detector-deploy (Swin-v2)", "learned-whole-image",
                      InputUnit.IMAGE, (H1,), ("heatmap",), model_ids=(MODEL_ID,))


class Dima806Detector(_HFClassifier):
    MODEL_ID, POSITIVE, FACE_ORIENTED = "hf.dima806.deepfake_vs_real_image_detection", "fake", True
    spec = image_spec("img.hf.dima806", "dima806 deepfake_vs_real_image_detection (ViT-B/16)", "learned-face",
                      InputUnit.FACE_CROP, (H1,), ("heatmap",), model_ids=(MODEL_ID,), needs=("faces",))


class WvolfDetector(_HFClassifier):
    MODEL_ID, POSITIVE, FACE_ORIENTED = "hf.wvolf.vit_deepfake_detection", "fake", True
    spec = image_spec("img.hf.wvolf", "Wvolf ViT_Deepfake_Detection (ViT-B/16)", "learned-face", InputUnit.FACE_CROP, (H1,),
                      ("heatmap",), model_ids=(MODEL_ID,), needs=("faces",))


class DFDCB7Detector(Detector):
    MODEL_IDS = ("dfdc.effnet_b7_555", "dfdc.effnet_b7_999")
    SMALL_FACE_PX = 120  # v0.3 dfdc_min_face_px: smaller faces are heavily upscaled to the 380 px input
    spec = image_spec("img.dfdc_b7", "DFDC EfficientNet-B7 (selimsef, two-checkpoint ensemble)", "learned-face",
                      InputUnit.FACE_CROP, (H2,), model_ids=MODEL_IDS, needs=("faces",), profiles=ALL_PROFILES)

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return tuple(ActivationUse(m, min(MAX_FACES, dfdc.BATCH), "380 px face crop",
                                   "At most six faces, backend microbatch eight; ensemble members execute sequentially")
                     for m in self.MODEL_IDS)

    def applicable(self, ctx: RunContext) -> Applicability:
        return Applicability(True) if ctx.outputs.faces else Applicability(False, "no face detected")

    def run(self, ctx: RunContext) -> DetectorResult:
        clf = DFDCClassifier(members=[ctx.lease.get(m).obj for m in self.MODEL_IDS], device=ctx.device)
        rgb, faces = analysis_rgb(ctx), face_boxes(ctx)[:MAX_FACES]
        preds = [float(p) for p in clf.predict([(rgb, f) for f in faces])]
        units = [UnitScore(f"face:{i}", p, "probability", quality={"face_px": float(f.w)})
                 for i, (p, f) in enumerate(zip(preds, faces, strict=True))]
        p, max_face = max(preds), max(f.w for f in faces)
        small = max_face < self.SMALL_FACE_PX
        aggregate = UnitScore("global", p, "probability",
                              quality={"n_faces": float(len(faces)), "max_face_px": float(max_face), "small_face": float(small)})
        return DetectorResult(
            status="ok", unit_scores=units, aggregate=aggregate,
            observed=f"sigmoid output {p:.3f} (maximum over {len(preds)} face(s))",
            reference="Trained on the Deepfake Detection Challenge corpus; decision boundary at 0.5",
            reasoning=("The face shows blending or warping statistics that this detector associates with face replacement."
                       if p >= 0.5 else "The face-swap detector finds no face-replacement statistics on the face.")
                      + (f" The largest face is below {self.SMALL_FACE_PX} px and was upscaled for the detector." if small else ""),
            metrics={"n_faces": float(len(faces)), "max_face_px": float(max_face)},
            model_revisions=revisions(ctx, *self.MODEL_IDS))


class NativeCropLogitDetector(Detector):
    """ML N2 for a learned whole-image classifier: native-resolution crops (centre and corners, never resampled), one logit
    per crop and the mean logit as the aggregate; the extremes go to the explanation, the crops to a `regions` exhibit."""

    impl: ClassVar[Any]  # module exposing CROP, crop_origins(width, height), to_batch(pixels, origins), logits(model, batch, device)
    MODEL: ClassVar[str]
    LABEL: ClassVar[str]
    REFERENCE: ClassVar[str]

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(self.MODEL, len(native_crops.CROP_NAMES), "native crop",
                              f"Centre and four corners, at most five {self.impl.CROP} px crops in one call"),)

    def _small(self) -> str:
        return f"image short side below {self.impl.CROP} px (native crops only, no upscaling)"

    def applicable(self, ctx: RunContext) -> Applicability:
        """The size rule needs no model, so it is asked before the lease (BUILD_SPEC N-14c, E-10)."""
        height, width = ctx.media.native_pixels().shape[:2]
        return Applicability(True) if self.impl.crop_origins(int(width), int(height)) else Applicability(False, self._small())

    def run(self, ctx: RunContext) -> DetectorResult:
        crop = self.impl.CROP
        pixels = ctx.media.native_pixels()
        height, width = int(pixels.shape[0]), int(pixels.shape[1])
        origins = self.impl.crop_origins(width, height)
        if not origins:
            return DetectorResult.not_applicable(self._small())
        crop_logits = self.impl.logits(ctx.lease.get(self.MODEL).obj, self.impl.to_batch(pixels, origins), ctx.device)
        mean, high, low = float(np.mean(crop_logits)), max(crop_logits), min(crop_logits)
        units = [UnitScore(f"crop:{i}", z, "logit", quality={"x": float(x), "y": float(y)})
                 for i, (z, (x, y, _)) in enumerate(zip(crop_logits, origins, strict=True))]
        aggregate = UnitScore("global", mean, "logit", quality={"crops": float(len(origins)), "short_side_px": float(min(width, height))})
        regions = [{"region": {"kind": "bbox", "bbox": {"x": float(x), "y": float(y), "w": float(crop), "h": float(crop)}},
                    "score": round(z, 4), "label": f"{name} crop"} for z, (x, y, name) in zip(crop_logits, origins, strict=True)]
        exhibit_id = ctx.exhibits.regions(
            regions, title=f"Crops scored by {self.LABEL}", media_ref="original",
            caption=f"Native-resolution {crop} px crops with the classifier's logit for each. Not a map of manipulation.",
            coordinate_space={"width": width, "height": height, "scale_to_original": 1.0})
        side = "synthetic" if mean >= 0 else "natural-image"
        top = origins[crop_logits.index(high)][2]
        return DetectorResult(
            status="ok", unit_scores=units, aggregate=aggregate,
            observed=f"mean logit {mean:+.3f} over {len(origins)} native-resolution {crop} px crop(s); "
                     f"highest {high:+.3f} ({top} crop), lowest {low:+.3f}",
            reference=self.REFERENCE,
            reasoning=f"The classifier places the mean of its crop scores on the {side} side of its decision boundary."
                      + (" Its crops fall on both sides of the boundary." if low < 0 <= high else ""),
            metrics={"logit_mean": mean, "logit_max": high, "logit_min": low, "crops": float(len(origins))},
            exhibit_ids=[exhibit_id], model_revisions=revisions(ctx, self.MODEL))


class CommFor384Detector(NativeCropLogitDetector):
    """Community Forensics ViT-S/16 on native 384 px crops (ML N2): mean logit over the centre and corner crops."""

    impl = commfor
    MODEL = "commfor.vit_s16_384"
    LABEL = "Community Forensics"
    REFERENCE = ("ViT-S/16 trained on camera photographs and generator outputs of the Community Forensics training data; "
                 "decision boundary at logit 0")
    spec = image_spec("img.commfor384", "Community Forensics (ViT-S/16, native 384 px crops)", "learned-whole-image",
                      InputUnit.IMAGE, (H1,), ("regions",), model_ids=(MODEL,), profiles=ALL_PROFILES, version="0.4.0")


class DDADetector(NativeCropLogitDetector):
    """DDA, DINOv2 ViT-L/14 with LoRA, on native 336 px crops (ML N1, N2): mean logit over the centre and corner crops."""

    impl = dda
    MODEL = "dda.dinov2_vitl14_lora"
    LABEL = "DDA"
    REFERENCE = ("DINOv2 ViT-L/14 with LoRA trained to separate MSCOCO photographs from their VAE reconstructions (Dual Data "
                 "Alignment); decision boundary at logit 0")
    spec = image_spec("img.dda", "DDA (DINOv2 ViT-L/14 + LoRA, native 336 px crops)", "learned-whole-image",
                      InputUnit.IMAGE, (H1,), ("regions",), model_ids=(MODEL,), version="0.4.0")


class SPAIDetector(Detector):
    """SPAI on native-resolution 224 px patches (ML N1): one logit from spectral context attention over the patches; each
    patch's attention share goes to a `regions` exhibit."""

    MODEL = "spai.vitb16_mfm"
    spec = image_spec("img.spai", "SPAI (ViT-B/16 MFM, spectral, native resolution)", "learned-whole-image", InputUnit.IMAGE,
                      (H1,), ("regions",), model_ids=(MODEL,), profiles=frozenset({Profile.DEEP}), version="0.4.0")

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(self.MODEL, 1, "native image",
                              f"Native patch plan capped at {spai.MAX_PATCHES} patches; encoder microbatch {spai.FEATURE_BATCH}",
                              ("Actual native-image memory demand varies with resolution; this allowance is fixed. Retained patch embeddings and "
                               "aggregation do not have a validated peak estimate.",)),)

    SMALL: ClassVar[str] = f"image short side below {spai.PATCH} px (native patches only, no padding or upscaling)"

    def applicable(self, ctx: RunContext) -> Applicability:
        """The size rule needs no model, so it is asked before the lease (BUILD_SPEC N-14c, E-10)."""
        height, width = ctx.media.native_pixels().shape[:2]
        return Applicability(True) if spai.patch_plan(int(width), int(height)).origins else Applicability(False, self.SMALL)

    def run(self, ctx: RunContext) -> DetectorResult:
        pixels = ctx.media.native_pixels()
        height, width = int(pixels.shape[0]), int(pixels.shape[1])
        plan = spai.patch_plan(width, height)
        if not plan.origins:
            return DetectorResult.not_applicable(self.SMALL)
        logit, attention = spai.score(ctx.lease.get(self.MODEL).obj, pixels, plan.origins, ctx.device, check=ctx.deadline.check)
        shares: dict[tuple[int, int], list[Any]] = {}
        for share, (x, y, name) in zip(attention, plan.origins, strict=True):  # five_crop repeats origins on small images
            shares.setdefault((x, y), [0.0, name])[0] += share
        regions = [{"region": {"kind": "bbox", "bbox": {"x": float(x), "y": float(y), "w": float(spai.PATCH), "h": float(spai.PATCH)}},
                    "score": round(share, 4), "label": f"{name} patch"} for (x, y), (share, name) in shares.items()]
        exhibit_id = ctx.exhibits.regions(
            regions, title="Patches weighed by SPAI", media_ref="original",
            caption=f"Native-resolution {spai.PATCH} px patches with the share of SPAI's spectral context attention each "
                    "received (the shares sum to 1). Not a map of manipulation.",
            coordinate_space={"width": width, "height": height, "scale_to_original": 1.0})
        n, five_crop = len(plan.origins), plan.mode == "five_crop"
        top_share, top_name = max(shares.values(), key=lambda item: item[0])
        _, _, window_w, window_h = plan.window
        aggregate = UnitScore("global", logit, "logit",
                              quality={"patches": float(n), "short_side_px": float(min(width, height)),
                                       "five_crop": float(five_crop), "window_capped": float(plan.capped)})
        side = "synthetic" if logit >= 0 else "natural-image"
        return DetectorResult(
            status="ok", unit_scores=[aggregate], aggregate=aggregate,
            observed=f"logit {logit:+.3f} from spectral context attention over {n} native-resolution {spai.PATCH} px patch(es)"
                     + ("; the image has fewer than four whole patches, so the four corner crops and the centre crop were "
                        "scored" if five_crop else "")
                     + (f"; the image exceeds {spai.MAX_PIXELS // 1_000_000} MP, so only a centred {window_w}×{window_h} px "
                        "native window was scored" if plan.capped else "")
                     + f"; largest attention share {top_share:.2f} ({top_name} patch)",
            reference=("ViT-B/16 with masked-frequency pre-training that scores how consistently each patch's features survive "
                       "the removal of low or high spatial frequencies, trained on latent-diffusion images against COCO and "
                       "LSUN photographs; decision boundary at logit 0"),
            reasoning=f"The spectral reconstruction score places the image on the {side} side of the detector's decision "
                      "boundary.",
            metrics={"logit": logit, "patches": float(n), "attention_max": float(top_share),
                     "window_px": float(window_w * window_h)},
            exhibit_ids=[exhibit_id], model_revisions=revisions(ctx, self.MODEL))


class GenDDetector(Detector):
    """GenD, CLIP ViT-L/14, on face crops aligned from the YuNet landmarks (ML N5): one logit per face (at most six), the
    maximum as the aggregate; the face boxes with their logits go to a `regions` exhibit. Images of at most 256 px are fed
    whole (N5 portrait rule)."""

    MODEL = "gend.clip_l14"
    SMALL_FACE_PX = 120  # VF size gate: a smaller face is upscaled into the 224 px input
    spec = image_spec("face.gend_clip_l14", "GenD (CLIP ViT-L/14, aligned face crops)", "learned-face", InputUnit.FACE_CROP,
                      (H2,), ("regions",), model_ids=(MODEL,), needs=("faces",), version="0.4.0")

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(self.MODEL, MAX_FACES, "224 px face crop", "At most six aligned faces; small portrait uses one input"),)

    NO_LANDMARKS: ClassVar[str] = "no face with five landmarks to align"

    def applicable(self, ctx: RunContext) -> Applicability:
        """A face to score needs no model to be found, so it is asked before the lease (BUILD_SPEC N-14c, E-10): an image fed
        whole, or a face whose five landmarks give `run` an alignment."""
        if not ctx.outputs.faces:
            return Applicability(False, "no face detected")
        height, width = ctx.media.native_pixels().shape[:2]
        if max(int(width), int(height)) <= gend.WHOLE_IMAGE_MAX_PX:
            return Applicability(True)
        _, scale = ctx.media.analysis_copy()
        aligned = any(len(face.landmarks) == 5 and gend.align_transform(np.asarray(face.landmarks, np.float32) * scale) is not None
                      for face in face_boxes(ctx)[:MAX_FACES])
        return Applicability(True) if aligned else Applicability(False, self.NO_LANDMARKS)

    def run(self, ctx: RunContext) -> DetectorResult:
        pixels = ctx.media.native_pixels()
        _, scale = ctx.media.analysis_copy()
        height, width = int(pixels.shape[0]), int(pixels.shape[1])
        faces = face_boxes(ctx)[:MAX_FACES]
        whole = max(width, height) <= gend.WHOLE_IMAGE_MAX_PX
        crops, kept = ([pixels], faces[:1]) if whole else ([], [])
        if not whole:
            for face in faces:
                crop = gend.align(pixels, np.asarray(face.landmarks, np.float32) * scale) if len(face.landmarks) == 5 else None
                if crop is not None:
                    crops.append(crop)
                    kept.append(face)
        if not crops:
            return DetectorResult.not_applicable(self.NO_LANDMARKS)
        face_logits = gend.logits(ctx.lease.get(self.MODEL).obj, gend.to_batch(crops), ctx.device)
        units, regions = [], []
        for i, (z, face, crop) in enumerate(zip(face_logits, kept, crops, strict=True)):
            box = {"x": round(face.x * scale, 1), "y": round(face.y * scale, 1), "w": round(face.w * scale, 1),
                   "h": round(face.h * scale, 1)}
            units.append(UnitScore(f"face:{i}", z, "logit",
                                   quality={"face_px": box["w"], "crop_px": float(min(crop.shape[:2])),
                                            "upscaled": float(min(crop.shape[:2]) < gend.CROP)}))
            landmarks = [(round(x * scale, 1), round(y * scale, 1)) for x, y in face.landmarks] or None
            regions.append({"region": {"kind": "bbox", "bbox": box, "landmarks": landmarks}, "score": round(z, 4),
                            "label": f"face {i + 1}"})
        high, low = max(face_logits), min(face_logits)
        max_face = max(u.quality["face_px"] for u in units)
        small, upscaled = max_face < self.SMALL_FACE_PX, sum(int(u.quality["upscaled"]) for u in units)
        exhibit_id = ctx.exhibits.regions(
            regions, title="Faces scored by GenD", media_ref="original",
            caption="YuNet face boxes with the logit GenD gave each aligned face crop. Not a map of manipulation.",
            coordinate_space={"width": width, "height": height, "scale_to_original": 1.0})
        aggregate = UnitScore("global", high, "logit", quality={"n_faces": float(len(units)), "max_face_px": max_face,
                                                               "small_face": float(small), "whole_image_input": float(whole)})
        target = (f"the whole {width}×{height} px image (at most {gend.WHOLE_IMAGE_MAX_PX} px, fed whole)" if whole
                  else f"{len(units)} aligned face crop(s)")
        side = "manipulated-face" if high >= 0 else "unmanipulated-face"
        return DetectorResult(
            status="ok", unit_scores=units, aggregate=aggregate,
            observed=f"maximum logit {high:+.3f} over {target}" + (f"; lowest {low:+.3f}" if len(units) > 1 else "")
                     + (f"; {upscaled} input(s) below {gend.CROP} px were upscaled" if upscaled else "")
                     + (f"; {len(faces) - len(kept)} face(s) without five landmarks not scored" if len(kept) < len(faces) and not whole else ""),
            reference=("CLIP ViT-L/14 with fine-tuned LayerNorms and an L2-normalised linear head, trained on aligned face "
                       "crops of face-forgery video datasets; decision boundary at logit 0"),
            reasoning=f"The face-forgery detector places its highest-scoring face on the {side} side of its decision boundary."
                      + (" The faces fall on both sides of the boundary." if low < 0 <= high else "")
                      + (f" The largest face is below {self.SMALL_FACE_PX} px wide, so its crop was upscaled for the detector."
                         if small else ""),
            metrics={"logit_max": high, "logit_min": low, "n_faces": float(len(units)), "max_face_px": max_face},
            exhibit_ids=[exhibit_id], model_revisions=revisions(ctx, self.MODEL))
