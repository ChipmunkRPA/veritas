"""Multi-face tracks as a v2 plugin (V-2; ARCHITECTURE §3.1 `tracks`; PRD DC-03).

The tracks stage passes a `LowResVideo` (source path, frame table, shot ranges) in `outputs.frames`. The detector decodes
the ≤ 640 px stream once and, on every 2nd frame, runs YuNet and embeds each face crop with DINOv2-small. It then tracks
within shots and clusters identities across shots (`video/tracking.py`). One `face_track` exhibit per track carries the
detected and interpolated frames in original pixels with stream pts. The detector produces no score: it selects the
per-track inputs of the face detectors (V-3).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.adapters.common import revisions
from app.detectors.image.faces import YUNET_TOP_K, crop_face, detect_faces
from app.detectors.video.adapters.common import FULL_DEEP, video_spec
from app.detectors.video.decode import LOW_RES_PX, DecodedFrame, FrameTable, decode_stream, scaled_size
from app.detectors.video.tracking import (
    IDENTITY_THRESHOLD,
    KEEP_RATIO,
    MIN_TRACK_PX,
    Detection,
    Tracker,
    cluster_identities,
    track_frames,
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

DETECT_STRIDE = 2
CROP_MARGIN = 0.2


@dataclass(frozen=True)
class LowResVideo:
    path: Path
    table: FrameTable
    shots: tuple[tuple[int, int], ...]
    max_side: int = LOW_RES_PX

    @property
    def scale_to_original(self) -> float:
        return self.table.width / scaled_size(self.table.width, self.table.height, self.max_side)[0]

    def frames(self, check: Callable[[], None] | None = None) -> Iterator[DecodedFrame]:
        return decode_stream(self.path, self.table, max_side=self.max_side, check=check)


def _clock(seconds: float) -> str:
    whole = max(0, int(seconds))
    return f"{whole // 60:02d}:{whole % 60:02d}"


class FaceTrackDetector(Detector):
    YUNET, EMBEDDER = "yunet.2023mar", "dinov2.small"
    spec = video_spec("vid.tracks.yunet_dinov2", "Face tracks: YuNet on every 2nd low-resolution frame, DINOv2-small appearance",
                      InputUnit.FRAME_BATCH, (), ("face_track",), family="face-tracking", model_ids=(YUNET, EMBEDDER),
                      timeout_s=300.0, profiles=FULL_DEEP, version="0.4.0")

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(self.YUNET, 1, "frame", "One low-resolution detection frame"),
                ActivationUse(self.EMBEDDER, YUNET_TOP_K, "224 px face crop",
                              "All faces of one detection frame; refuse any count above the YuNet factory top_k=50",
                              ("Retained tracks, frame decoding and CPU clustering are outside the estimated model allowance.",)))

    def applicable(self, ctx: RunContext) -> Applicability:
        video = ctx.outputs.frames
        if not isinstance(video, LowResVideo) or not len(video.table):
            return Applicability(False, "no decodable video frames")
        return Applicability(True)

    def run(self, ctx: RunContext) -> DetectorResult:
        video: LowResVideo = ctx.outputs.frames
        yunet = ctx.lease.get(self.YUNET).obj
        embedder = ctx.lease.get(self.EMBEDDER).obj
        table, scale = video.table, video.scale_to_original
        shot_of = np.zeros(len(table), dtype=np.int64)
        for s, (start, end) in enumerate(video.shots):
            shot_of[start:end] = s
        tracker = Tracker()
        min_low_px = math.floor(KEEP_RATIO * MIN_TRACK_PX / scale)
        check = (lambda: ctx.deadline.check("face tracking")) if ctx.deadline is not None else None
        shot, detection_frames, detections = 0, 0, 0
        for frame in video.frames(check=check):
            if int(shot_of[frame.index]) != shot:
                shot = int(shot_of[frame.index])
                tracker.new_shot(shot)
            if frame.index % DETECT_STRIDE:
                continue
            faces = detect_faces(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), max_side=video.max_side, detector=yunet,
                                 min_size=min_low_px)
            if len(faces) > YUNET_TOP_K:
                raise ValueError(f"face count exceeds the declared YuNet limit of {YUNET_TOP_K}")
            embeddings = embedder.embed([crop_face(frame.rgb, f, CROP_MARGIN) for f in faces]) if faces else []
            tracker.step([Detection(frame.index, (f.x * scale, f.y * scale, f.w * scale, f.h * scale), float(f.score),
                                    tuple((px * scale, py * scale) for px, py in f.landmarks), embeddings[k])
                          for k, f in enumerate(faces)])
            detection_frames += 1
            detections += len(faces)
        tracks = tracker.finish()
        labels = cluster_identities(tracks)
        units, exhibit_ids = [], []
        for k, (track, label) in enumerate(zip(tracks, labels, strict=True)):
            frames = track_frames(track, table)
            span = f"{_clock(table.seconds(track.first_frame))}–{_clock(table.seconds(track.last_frame))}"
            name = f"Face {label}" if label else f"Face track {k + 1}"
            exhibit_ids.append(ctx.exhibits.add(
                "face_track", title=f"{name}, {span}", media_ref="frame",
                caption=("Detected boxes on every 2nd frame at ≤ 640 px, interpolated between detections (det_score empty); "
                         "original pixels, stream pts. The letter groups tracks by appearance similarity, not face recognition."),
                track={"track_id": f"track:{k}", "first_pts": frames[0]["pts"], "last_pts": frames[-1]["pts"],
                       "n_frames": len(frames), "identity": label},
                frames=frames))
            sizes = track.face_px()
            units.append(UnitScore(f"track:{k}", float(len(track.detections)), "score",
                                   quality={"shot": float(track.shot), "median_face_px": float(np.median(sizes)),
                                            "max_face_px": float(max(sizes))}))
        identities = len({label for label in labels if label})
        return DetectorResult(
            status="ok", unit_scores=units,
            observed=f"{len(tracks)} face track(s) with {identities} appearance cluster(s) in {len(video.shots)} shot(s)",
            reference=(f"Faces of at least {MIN_TRACK_PX:.0f} px on every 2nd frame; tracks end at shot boundaries; appearance "
                       f"clusters at cosine distance ≤ {IDENTITY_THRESHOLD} (DINOv2-small), not face recognition"),
            reasoning="Selects the per-track inputs of the face detectors; produces no score.",
            metrics={"tracks": float(len(tracks)), "identities": float(identities), "shots": float(len(video.shots)),
                     "detection_frames": float(detection_frames), "detections": float(detections),
                     "dropped_tracks": float(tracker.dropped)},
            exhibit_ids=exhibit_ids, model_revisions=revisions(ctx, self.YUNET, self.EMBEDDER), device="", dtype="", timings={},
            peak_memory_mb=None)
