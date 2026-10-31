"""YuNet face detector as a v2 detector. The faces stage turns its regions exhibit into boxes for the face detectors."""
from __future__ import annotations

import cv2

from app.detectors.image.adapters.common import (
    ALL_PROFILES,
    CPU_ONLY,
    analysis_rgb,
    coordinate_space,
    image_spec,
    revisions,
)
from app.detectors.image.faces import detect_faces
from app.engine.detector import ActivationUse, Detector, DetectorResult, InputUnit, RunContext, UnitScore

MODEL_ID = "yunet.2023mar"


class YuNetFaces(Detector):
    MIN_FACE_PX = 48  # v0.3 face_min_size
    spec = image_spec("img.faces.yunet", "YuNet face detector (OpenCV Zoo 2023mar)", "face-detection", InputUnit.IMAGE, (),
                      ("regions",), model_ids=(MODEL_ID,), lane="cpu", devices=CPU_ONLY, timeout_s=20.0,
                      profiles=ALL_PROFILES)

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(MODEL_ID, 1, "image", "One detector image, downscaled to at most 1280 px on the long side"),)

    def run(self, ctx: RunContext) -> DetectorResult:
        rgb = analysis_rgb(ctx)
        faces = detect_faces(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detector=ctx.lease.get(MODEL_ID).obj,
                             min_size=self.MIN_FACE_PX)
        exhibit_ids = []
        if faces:
            regions = [{"region": {"kind": "bbox", "bbox": {"x": float(f.x), "y": float(f.y), "w": float(f.w), "h": float(f.h)},
                                   "landmarks": [(round(x, 2), round(y, 2)) for x, y in f.landmarks] or None},
                        "score": round(f.score, 4), "label": f"face {i}"} for i, f in enumerate(faces, 1)]
            exhibit_ids.append(ctx.exhibits.regions(regions, title="Detected faces",
                                                    caption=f"YuNet boxes of at least {self.MIN_FACE_PX} px, largest first, "
                                                            "with five landmarks (eyes, nose tip, mouth corners)",
                                                    coordinate_space=coordinate_space(ctx)))
        scores = [UnitScore(f"face:{i}", round(f.score, 4), "score", quality={"face_px": float(min(f.w, f.h))})
                  for i, f in enumerate(faces)]
        return DetectorResult(
            status="ok", unit_scores=scores, aggregate=None,
            observed=f"{len(faces)} face(s) of at least {self.MIN_FACE_PX} px",
            reference="YuNet 2023mar: score threshold 0.7, NMS IoU 0.3, detection on a copy of at most 1280 px",
            reasoning="Face boxes decide which inputs the face-oriented detectors score; they are not evidence on their own.",
            metrics={"n_faces": float(len(faces))}, exhibit_ids=exhibit_ids, model_revisions=revisions(ctx, MODEL_ID))
