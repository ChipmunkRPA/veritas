"""Paired native-face review of videos with no assumption of original flanks."""

from __future__ import annotations

from math import isfinite

import cv2

from app.detectors.image import dfdc, gend
from app.detectors.image.adapters.common import revisions
from app.detectors.image.faces import detect_faces
from app.detectors.video.adapters.common import ALL_PROFILES, video_spec
from app.detectors.video.adapters.face_sequence import SequenceInputs
from app.engine.detector import ActivationUse, Applicability, Detector, DetectorResult, InputUnit, UnitScore
from app.evidence import whole_face_binding as whole_face

DETECTOR = "vid.review.whole_face"
FAMILY = "whole-face-review"


class WholeFaceReviewDetector(Detector):
    YUNET = "yunet.2023mar"
    REFERENCE = ("dfdc.effnet_b7_555", "dfdc.effnet_b7_999")
    COMPLEMENT = "gend.pe_large"
    MODELS = (YUNET, *REFERENCE, COMPLEMENT)
    spec = video_spec(DETECTOR, "Face-swap review across sampled frames", InputUnit.FACE_CROP, (),
                      family=FAMILY, model_ids=MODELS, needs=("frames",), profiles=ALL_PROFILES,
                      timeout_s=100.0, version="0.1.0")

    def activation_plan(self, ctx):
        return (
            ActivationUse(self.YUNET, 1, "native frame", "One face detection per source frame"),
            *(ActivationUse(key, 1, "380 px face crop", "One crop at a time; members run sequentially")
              for key in self.REFERENCE),
            ActivationUse(self.COMPLEMENT, 1, "224 px aligned face", "One aligned crop at a time"),
        )

    def applicable(self, ctx):
        inputs: SequenceInputs | None = ctx.outputs.face_review
        if inputs is None:
            return Applicability(False, "native source frames are unavailable")
        table = inputs.table
        if (table.truncated or table.interpolated or table.rotation or len(table) <= 30
                or min(table.width, table.height) < 360 or max(table.width, table.height) > 1920
                or not whole_face.sample_indices(table.pts)):
            return Applicability(False, "the source clock or frame geometry does not support this face review")
        return Applicability(True)

    def run(self, ctx):
        inputs: SequenceInputs = ctx.outputs.face_review
        table, store = inputs.table, inputs.store
        indices = whole_face.sample_indices(table.pts)
        yunet = ctx.lease.get(self.YUNET).obj
        reference = dfdc.DFDCClassifier([ctx.lease.get(key).obj for key in self.REFERENCE], ctx.device)
        complement = ctx.lease.get(self.COMPLEMENT).obj
        rows = {}
        for index, rgb in store.iter_many(indices):
            ctx.deadline.check("paired native face review")
            if not store.native_pts_verified(index):
                return DetectorResult.not_applicable("a scheduled frame lacks a verified decoder timestamp")
            faces = detect_faces(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detector=yunet, min_size=whole_face.MIN_FACE_PX)
            row = {"frame_index": index, "pts": table.pts[index], "box": None, "responses_hex": [None, None]}
            if faces:
                face = faces[0]
                row["box"] = dict(zip(("x", "y", "w", "h"), face.box(), strict=True))
                first, = reference.predict([(rgb, face)], batch=1)
                crop = gend.align(rgb, face.landmarks)
                second = None
                if crop is not None:
                    second, = gend.logits(complement, complement.to_batch([crop]), ctx.device)
                if not isfinite(first) or second is not None and not isfinite(second):
                    return DetectorResult.failed("a face review response was not finite")
                row["responses_hex"] = [float(first).hex(), float(second).hex() if second is not None else None]
            rows[index] = row
        if set(rows) != set(indices):
            return DetectorResult.failed("a scheduled native face input could not be decoded")
        samples = [rows[index] for index in indices]
        arms = [[float.fromhex(row["responses_hex"][arm]) if row["responses_hex"][arm] is not None else None
                 for row in samples] for arm in range(2)]
        score = whole_face.paired_score(*arms)
        if score is None:
            return DetectorResult.not_applicable("each face check needs at least four usable samples")
        flagged = score > whole_face.REVIEW_POINT
        observed = ("The paired face checks found a face-swap pattern worth reviewing in the sampled frames." if flagged else
                    "The paired face checks did not flag a face-swap pattern in the sampled frames.")
        return DetectorResult(status="ok", aggregate=UnitScore("global", score, "score"), observed=observed,
                              reference="Six interior source frames, with at least four usable inputs per check.",
                              reasoning="The response describes sampled faces and carries no weight in the overall AI assessment.",
                              details={"review_only": True, "whole_face_review": {"schema": whole_face.METHOD, "flagged": flagged,
                                       "score_hex": score.hex(), "time_base": table.time_base, "samples": samples}},
                              model_revisions=revisions(ctx, *self.MODELS))
