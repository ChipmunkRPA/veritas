"""Bounded sequence review using the exact raw-crop v6 inspection method."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import cv2

from app.core.config import settings
from app.detectors.image.adapters.common import revisions
from app.detectors.image.faces import detect_faces
from app.detectors.image.gend import align_transform, warp
from app.detectors.video.adapters.common import FULL_DEEP, video_spec
from app.detectors.video.decode import FrameTable
from app.detectors.video.face_sequence import METHOD, REVIEW_POINT, SIZE, patch_features, sequence_windows
from app.engine.detector import (
    ActivationUse,
    Applicability,
    Detector,
    DetectorResult,
    InputUnit,
    RunContext,
    UnitScore,
)
from app.evidence.face_intervals import bind_tracks

DETECTOR = "vid.review.face_sequence"
FAMILY = "face-sequence-review"
MAX_CENTERS = 64


@dataclass(frozen=True)
class SequenceInputs:
    table: FrameTable
    store: Any
    centers: tuple[int, ...]
    exhibits: tuple[dict, ...]


def record(window: dict, table: FrameTable, inputs: dict[int, dict]) -> dict:
    centers = window["centers"]
    roles = {"before": centers[:2], "during": centers[2:-2], "after": centers[-2:]}
    indices = sorted({i+k for i in centers for k in (-1, 0, 1)})
    return {"method": METHOD, "track_id": window["track_id"], "time_base": table.time_base,
            "start_pts": table.pts[roles["during"][0]], "end_pts": table.pts[roles["during"][-1]],
            "start_stream_s": window["start_s"], "end_stream_s": window["end_s"],
            "response": window["response"], "inner_change": window["inner_change"],
            "outer_change": window["outer_change"], "flank_change": window["flank_change"],
            "measurement_hex": {key: float(window[key]).hex() for key in ("response", "inner_change", "outer_change", "flank_change")},
            "centers": roles, "frame_inputs": [inputs[i] for i in indices]}


class FaceSequenceReviewDetector(Detector):
    YUNET, DINO = "yunet.2023mar", "dinov2.small"
    spec = video_spec(DETECTOR, "Consecutive face-change review", InputUnit.FRAME_BATCH, (), ("series",),
                      family=FAMILY, model_ids=(YUNET, DINO), needs=("frames", "faces"),
                      profiles=FULL_DEEP, timeout_s=120.0, version="0.1.0")

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(self.YUNET, 1, "frame", "One native frame for face detection"),
                ActivationUse(self.DINO, 12, "112 px aligned crop", "At most twelve face crops per feature batch",
                              ("At most 192 retained crops and their patch tokens; native frames use the bounded frame cache.",)))

    def applicable(self, ctx: RunContext) -> Applicability:
        inputs = ctx.outputs.face_review
        if not isinstance(inputs, SequenceInputs) or not inputs.exhibits or len(inputs.centers) < 7:
            return Applicability(False, "insufficient detected face samples and tracked context for sequence review")
        if len(inputs.centers) > MAX_CENTERS:
            return Applicability(False, "sequence review exceeds its bounded sample budget")
        ordered = sorted(inputs.centers)
        longest = current = 1
        for a, b in zip(ordered, ordered[1:]):
            current = current+1 if 0 < inputs.table.seconds(b)-inputs.table.seconds(a) <= .55 else 1
            longest = max(longest, current)
        return Applicability(longest >= 7, "insufficient closely spaced source samples" if longest < 7 else None)

    def run(self, ctx: RunContext) -> DetectorResult:
        inputs: SequenceInputs = ctx.outputs.face_review
        table = inputs.table
        centers = {i for i in inputs.centers if 0 < i < len(table)-1}
        scheduled = {i+k for i in centers for k in (-1, 0, 1)}
        crops, faces, bindings, decoded = {}, {}, {}, set()
        yunet, dino = ctx.lease.get(self.YUNET).obj, ctx.lease.get(self.DINO).obj
        check = ctx.deadline.check if ctx.deadline is not None else lambda *_: None
        for i, rgb in inputs.store.iter_many(sorted(scheduled)):
            check("consecutive face crops")
            decoded.add(i)
            detected = detect_faces(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detector=yunet, min_size=settings.face_min_size)
            if not detected or (transform := align_transform(detected[0].landmarks)) is None:
                continue
            face = detected[0]
            crop = cv2.resize(warp(rgb, *transform), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            crops[i], faces[i] = crop, face
            bindings[i] = {"frame_index": i, "pts": table.pts[i], "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                           "bbox": {key: float(getattr(face, key)) for key in ("x", "y", "w", "h")},
                           "landmarks": [[float(x), float(y)] for x, y in face.landmarks],
                           "alignment_matrix": transform[0].tolist(), "alignment_side": int(transform[1]),
                           "alignment_matrix_hex": [[float(value).hex() for value in row] for row in transform[0]],
                           "aligned_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest(), "aligned_size": SIZE}
        if decoded != scheduled:
            return DetectorResult.failed("a scheduled consecutive source frame could not be decoded")
        observed = {"kind": "regions", "created_by": "vid.faces.yunet", "regions": [
            {"region": {"kind": "bbox", "frame_pts": round(table.seconds(i)*1000), "bbox": binding["bbox"]}}
            for i, binding in bindings.items()]}
        tracks = bind_tracks([{"frame_index": i, "score": 0.} for i in crops], [observed, *inputs.exhibits], table)
        windows = sequence_windows(centers, tracks, patch_features(crops, dino, check=check), table)
        if not windows:
            return DetectorResult.not_applicable("no source-bound sequence has sufficient unambiguous tracked face context")
        highest = windows[0]
        selected = record(highest, table, bindings) if highest["response"] >= REVIEW_POINT else None
        sid = f"{self.spec.id}:global"
        exhibit = ctx.exhibits.series([{"name": "sequence_change", "points": [
            ((w["start_s"]+w["end_s"])/2, w["response"]) for w in windows]}], x_unit="s",
            title="Face-sequence change measurements", media_ref="frame", source_signal_ids=[sid],
            caption="Each point describes one overlapping sampled window, not an individual-frame probability.")
        return DetectorResult(status="ok", aggregate=UnitScore("global", highest["response"], "score"),
            observed=(f"A tracked inner-face change merits review from {highest['start_s']:.3f} to {highest['end_s']:.3f} s."
                      if selected else "The sampled face sequences did not produce an additional change range to review."),
            reference="Compared with both surrounding time groups, discounting changes outside the inner face and between the surrounding groups.",
            reasoning="This is an uncalibrated inspection cue. Motion, lighting, expression and compression may produce similar changes. It does not establish AI editing and does not change overall confidence.",
            metrics={"scheduled_frames": float(len(scheduled)), "face_frames": float(len(crops)), "windows": float(len(windows))},
            details={"face_sequence_binding_version": METHOD, "face_sequence": selected,
                     "face_sequence_windows": [record(window, table, bindings) for window in windows],
                     "input_frame_indices": sorted(crops), "review_only": True},
            exhibit_ids=[exhibit], model_revisions=revisions(ctx, self.YUNET, self.DINO))
