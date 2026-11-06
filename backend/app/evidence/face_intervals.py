"""Local changes in a single face track's recorded frame responses."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean

from app.detectors.image.faces import Face, iou
from app.detectors.video.decode import FrameTable
from app.evidence.face_interval_binding import (
    CONTEXT_COUNT,
    MAX_GAP_S,
    MAX_INTERVAL_S,
    METHOD,
    MIN_CORE_COUNT,
)

FRAME_CHECK = "vid.frames.dfdc_b7"


@dataclass(frozen=True)
class IntervalRule:
    minimum_response: float = .70
    minimum_lift: float = .25
    minimum_sample_response: float = .40


RULE = IntervalRule()


@dataclass(frozen=True)
class Sample:
    frame_index: int
    time_s: float
    score: float


@dataclass(frozen=True)
class Interval:
    track_id: str
    before: tuple[Sample, ...]
    during: tuple[Sample, ...]
    after: tuple[Sample, ...]
    lift: float

    @property
    def peak(self) -> Sample:
        return max(self.during, key=lambda sample: sample.score)


def _box(row: dict) -> Face:
    return Face(*(row[key] for key in ("x", "y", "w", "h")), score=1.0)


def bind_tracks(points: list[dict], exhibits: list[dict], table: FrameTable) -> dict[str, list[Sample]]:
    """Link the scored largest face to exactly one overlapping native-time track."""
    candidates: dict[int, list[tuple[str, Face]]] = defaultdict(list)
    largest: dict[int, Face] = {}
    millis = [round(value * 1000) for value in table.times()]
    counts = Counter(millis)
    for exhibit in exhibits:
        if exhibit.get("kind") == "face_track":
            track = exhibit["track"]["track_id"]
            for frame in exhibit["frames"]:
                index = frame["frame_index"]
                if 0 <= index < len(table) and frame["pts"] == table.pts[index]:
                    candidates[index].append((track, _box(frame["bbox"])))
        if exhibit.get("kind") == "regions" and exhibit.get("created_by") == "vid.faces.yunet":
            for scored in exhibit["regions"]:
                region = scored["region"]
                if region.get("kind") == "bbox":
                    # The frame scorer takes the first face in the exhibit's largest-first ordering.
                    largest.setdefault(region.get("frame_pts"), _box(region["bbox"]))
    tracks: dict[str, list[Sample]] = defaultdict(list)
    for row in sorted(points, key=lambda row: row["frame_index"]):
        index, score = row["frame_index"], row["score"]
        if not 0 <= index < len(table) or not math.isfinite(score) or not 0 <= score <= 1:
            continue
        ms = millis[index]
        if counts[ms] != 1 or ms not in largest or index in table.interpolated:
            continue
        overlaps = sorted(((iou(largest[ms], box), track) for track, box in candidates.get(index, ())), reverse=True)
        if overlaps and overlaps[0][0] >= .55 and (len(overlaps) == 1 or overlaps[0][0] - overlaps[1][0] >= .12):
            tracks[overlaps[0][1]].append(Sample(index, table.seconds(index), score))
    return dict(tracks)


def _intervals(track: str, samples: list[Sample], rule: IntervalRule) -> list[Interval]:
    result = []
    for start in range(CONTEXT_COUNT, len(samples) - CONTEXT_COUNT - MIN_CORE_COUNT + 1):
        before = tuple(samples[start-CONTEXT_COUNT:start])
        for end in range(start + MIN_CORE_COUNT, len(samples) - CONTEXT_COUNT + 1):
            if samples[end-1].time_s - samples[start].time_s > MAX_INTERVAL_S:
                break
            during = tuple(samples[start:end])
            if min(s.score for s in during) < rule.minimum_sample_response:
                break
            after = tuple(samples[end:end+CONTEXT_COUNT])
            response = mean(s.score for s in during)
            lift = response - max(mean(s.score for s in before), mean(s.score for s in after))
            if response >= rule.minimum_response and lift >= rule.minimum_lift:
                result.append(Interval(track, before, during, after, lift))
    return result


def localize(tracks: dict[str, list[Sample]], rule: IntervalRule = RULE) -> list[Interval]:
    candidates = []
    for track, samples in tracks.items():
        start = 0
        for end in range(1, len(samples) + 1):
            if end == len(samples) or not 0 < samples[end].time_s - samples[end-1].time_s <= MAX_GAP_S:
                candidates.extend(_intervals(track, samples[start:end], rule))
                start = end
    chosen = []
    for item in sorted(candidates, key=lambda x: (-x.lift, -len(x.during), x.during[0].frame_index, x.track_id)):
        if all(item.during[-1].time_s < other.during[0].time_s or item.during[0].time_s > other.during[-1].time_s
               or item.track_id != other.track_id for other in chosen):
            chosen.append(item)
        if len(chosen) == 2:
            break
    return chosen


def record(interval: Interval, table: FrameTable) -> dict:
    return {"method": METHOD, "track_id": interval.track_id, "time_base": table.time_base,
            "start_pts": table.pts[interval.during[0].frame_index], "end_pts": table.pts[interval.during[-1].frame_index],
            "start_stream_s": interval.during[0].time_s, "end_stream_s": interval.during[-1].time_s,
            "lift": interval.lift, "samples": {role: [
                {"frame_index": s.frame_index, "pts": table.pts[s.frame_index], "score": s.score} for s in samples]
                for role, samples in (("before", interval.before), ("during", interval.during), ("after", interval.after))}}
