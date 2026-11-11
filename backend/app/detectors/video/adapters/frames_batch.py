"""Memory-bounded video face-frame scoring with the original two-member ensemble."""

from dataclasses import replace

import numpy as np

from app.core.config import settings
from app.detectors.image.dfdc import DFDCClassifier
from app.detectors.video.adapters.common import frame_faces, sampled_frames, to_face
from app.detectors.video.adapters.frames import FrameDFDCB7Detector, _frames_result
from app.engine.detector import ActivationUse, DetectorResult, RunContext, UnitScore

_BASE_SPEC = FrameDFDCB7Detector.spec


class FrameDFDCB7BatchOneDetector(FrameDFDCB7Detector):
    """Use the existing score and evidence path with one face per forward."""

    spec = replace(
        _BASE_SPEC,
        version="0.3.1",
        requirements=replace(_BASE_SPEC.requirements, max_batch=1),
    )

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return tuple(
            ActivationUse(
                model_id,
                self.spec.requirements.max_batch,
                "380 px face crop",
                "One face per forward with the two ensemble members executed sequentially",
            )
            for model_id in self.MODEL_IDS
        )

    def run(self, ctx: RunContext) -> DetectorResult:
        classifier = DFDCClassifier(
            members=[ctx.lease.get(model_id).obj for model_id in self.MODEL_IDS],
            device=ctx.device,
        )
        faces = frame_faces(ctx)
        face_frames = [
            (frame, to_face(faces[frame.index][0])) for frame in sampled_frames(ctx) if faces[frame.index]
        ]
        batch = self.spec.requirements.max_batch
        units = []
        for start in range(0, len(face_frames), batch):
            ctx.deadline.check("face-frame classification")
            chunk = face_frames[start : start + batch]
            predictions = classifier.predict([(frame.rgb, face) for frame, face in chunk], batch=batch)
            units.extend(
                UnitScore(
                    f"frame:{frame.index}",
                    float(score),
                    "probability",
                    quality={"time_s": frame.time_s, "face_px": float(face.w)},
                )
                for (frame, face), score in zip(chunk, predictions, strict=True)
            )
        ctx.deadline.check("face-frame evidence")
        small = float(np.mean([face.w < settings.dfdc_min_face_px for _, face in face_frames]))
        return _frames_result(
            ctx,
            self.spec.name,
            units,
            {"face_frames": float(len(units)), "small_face_fraction": small},
            self.MODEL_IDS,
            "face crops of the largest face",
        )
