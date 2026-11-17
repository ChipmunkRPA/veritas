"""Additional face-swap inspection on supported recompressed video profiles."""

from __future__ import annotations

import hashlib
import math

import numpy as np

from app.core.config import settings
from app.detectors.image.adapters.common import revisions
from app.detectors.video import face_detail_inputs, face_frequency
from app.detectors.video.adapters.common import ALL_PROFILES, video_spec
from app.detectors.video.adapters.whole_face import same_frame_enabled
from app.detectors.video.face_return import source_carrier
from app.engine.detector import ActivationUse, Applicability, Detector, DetectorResult, InputUnit, UnitScore
from app.evidence import face_detail_binding as detail
from app.evidence import whole_face_binding as whole

DETECTOR = "vid.review.face_detail"
FAMILY = "face-detail-review"


def detail_enabled() -> bool:
    return same_frame_enabled() and settings.video_whole_face_detail_review


class FaceDetailReviewDetector(Detector):
    LOCATOR, EMBEDDER = "yunet.2023mar", "dinov2.small"
    MODELS = (LOCATOR, EMBEDDER)
    spec = video_spec(DETECTOR, "Face detail after recompression", InputUnit.FACE_CROP, (), family=FAMILY, model_ids=MODELS,
                      needs=("frames",), profiles=ALL_PROFILES, timeout_s=30.0, version="0.1.0")

    def activation_plan(self, ctx):
        return (ActivationUse(self.LOCATOR, 1, "native frame", "One native face input at a time"),
                ActivationUse(self.EMBEDDER, 8, "168 px face crop", "Up to eight aligned crops per batch"))

    def applicable(self, ctx):
        inputs = ctx.outputs.face_review
        if not detail_enabled() or inputs is None or not inputs.baseline:
            return Applicability(False, "the optional face detail check requires a completed face review")
        if inputs.baseline["flagged"] or not whole.same_frame_source(
                inputs.table, (*inputs.encoding, inputs.codec_tag), source_carrier(inputs.store.path)):
            return Applicability(False, "the earlier faces or the video delivery profile do not need this fallback")
        first, second = detail.base_scores(inputs.baseline)
        if first is None or second is None or first > whole.REVIEW_POINT or second > whole.SAME_FRAME_POINT:
            return Applicability(False, "the earlier checks need usable responses without a review flag")
        return Applicability(True)

    def run(self, ctx):
        inputs = ctx.outputs.face_review
        locator = ctx.lease.get(self.LOCATOR).obj
        trace = face_detail_inputs.tracked_faces(inputs.store, locator, check=lambda: ctx.deadline.check("native face detail"))
        if len(trace) < detail.MINIMUM_TRACK:
            return DetectorResult.not_applicable("too few consecutive sampled faces with aligned source pixels")
        crops = np.stack([row[3] for row in trace])
        encoder = face_detail_inputs.PatchEncoder(ctx.lease.get(self.EMBEDDER).obj.model, ctx.device)
        values = face_detail_inputs.appearance(encoder(crops))
        inner, ring, outer = (values[i:i+48] for i in (0, 48, 96))
        relative = [a-b for a, b in zip(inner, ring, strict=True)] + [a-b for a, b in zip(ring, outer, strict=True)]
        terms = detail.contributions(detail.base_scores(inputs.baseline), relative, face_frequency.frequency_detail(crops))
        score = math.fsum(terms)
        if not math.isfinite(score):
            return DetectorResult.failed("face detail response was not finite")
        flagged = score > detail.POINT
        samples = [{"frame_index": index, "pts": pts, "box": dict(zip("xywh", box, strict=True)),
                    "decoded_rgb_sha256": digest, "aligned_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
                    "eyes_hex": [[float(value).hex() for value in eye] for eye in eyes]}
                   for index, pts, box, crop, digest, eyes in trace]
        record = {"schema": detail.METHOD, "flagged": flagged, "score_hex": score.hex(), "time_base": inputs.table.time_base,
                  "calibration_sha256": detail.CALIBRATION_SHA256, "samples": samples, "prior": inputs.baseline,
                  "terms_hex": [value.hex() for value in terms]}
        count = len(trace)
        observed = (f"Local face detail and the earlier readings together flag a face-swap pattern across {count} sampled faces."
                    if flagged else f"The face detail comparison across {count} sampled faces found no additional pattern to flag.")
        return DetectorResult(status="ok", aggregate=UnitScore("global", score, "score"), observed=observed,
                              reference="Frequency detail and visual appearance inside the face were compared with the nearby picture.",
                              reasoning="Compression, lighting and ordinary camera processing can produce similar differences.",
                              details={"review_only": True, "whole_face_review": record}, model_revisions=revisions(ctx, *self.MODELS))
