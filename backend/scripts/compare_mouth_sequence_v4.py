"""Paired mouth/inner-face sequence development using identical production inputs."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from app.detectors.video.face_sequence import CORE, REVIEW_POINT, distance
from scripts.liveportrait_mouth_benchmark import digest

MOUTH = np.zeros((8, 8), bool)
MOUTH[5:8, 2:6] = True
MOUTH = MOUTH.flatten()
METHOD = "mouth-token-sequence-development/4.0"


def region_windows(centers, tracks, features, table, core=MOUTH):
    if core.shape != (64,) or core.dtype != bool or not core.any() or core.all():
        raise ValueError("A nonempty proper subset of the 8x8 token grid is required")
    result = []
    for track, samples in tracks.items():
        indices = {r.frame_index for r in samples}
        blocks = []
        for i in sorted(centers):
            group = (i - 1, i, i + 1)
            if (
                all(k in indices and k in features for k in group)
                and table.seconds(i + 1) - table.seconds(i - 1) <= 0.13
            ):
                blocks.append((i, np.mean([features[k] for k in group], axis=0)))
        if len(blocks) < 7:
            continue
        times = np.array([table.seconds(i) for i, _ in blocks])
        values = np.stack([value for _, value in blocks])
        for start in range(2, len(blocks) - 4):
            for end in range(start + 3, len(blocks) - 1):
                span = times[end - 1] - times[start]
                if span > 2.6:
                    break
                if span < 0.52 or np.max(np.diff(times[start - 2 : end + 2])) > 0.55:
                    continue
                left, middle, right = (
                    np.mean(part, axis=0)
                    for part in (values[start - 2 : start], values[start:end], values[end : end + 2])
                )
                a, b, normal = distance(middle, left), distance(middle, right), distance(left, right)
                inside, outside = (
                    min(float(a[core].mean()), float(b[core].mean())),
                    max(float(a[~core].mean()), float(b[~core].mean())),
                )
                flank = float(normal[core].mean())
                result.append(
                    {
                        "track_id": track,
                        "start_s": float(times[start]),
                        "end_s": float(times[end - 1]),
                        "response": inside - 0.60 * outside - 0.35 * flank,
                        "inner_change": inside,
                        "outer_change": outside,
                        "flank_change": flank,
                        "centers": [i for i, _ in blocks[start - 2 : end + 2]],
                    }
                )
    return sorted(result, key=lambda w: (-w["response"], w["start_s"], w["track_id"]))[:12]


def choose_point(rows):
    assessed = [r for r in rows if not r["label"] and r.get("mouth_windows")]
    if not assessed or {r["condition"] for r in assessed} != {
        "original",
        "local-light",
        "local-resample",
        "constant-audio-delay",
    }:
        return None
    return math.nextafter(max(0.0, max(r["mouth_windows"][0]["response"] for r in assessed)), math.inf)


def temporal_iou(a, b):
    if not a or not b:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) / union if union > 0 else 0.0


def proposal(row, arm, point):
    windows = row.get(arm + "_windows") or []
    return windows[0] if point is not None and windows and windows[0]["response"] >= point else None


class Exhibits:
    def __init__(self):
        self.items = []

    def add(self, kind, **kwargs):
        identifier = "research-" + str(len(self.items))
        self.items.append({"id": identifier, "kind": kind, **kwargs})
        return identifier

    def series(self, series, **kwargs):
        return self.add("series", series=series, **kwargs)

    def segments(self, segments, **kwargs):
        return self.add("segments", segments=segments, **kwargs)


def score_case(path, case, models):
    from app.core.config import settings
    from app.detectors.audio.decode import decode_audio
    from app.detectors.image.faces import detect_faces
    from app.detectors.video import sampling
    from app.detectors.video.adapters import face_sequence as adapter
    from app.detectors.video.adapters.common import AudioTrack
    from app.detectors.video.adapters.lipsync import AVInputs
    from app.detectors.video.adapters.lipsync_xcorr import LandmarkXcorrDetector
    from app.detectors.video.adapters.tracks import FaceTrackDetector, LowResVideo
    from app.detectors.video.decode import probe_frames
    from app.detectors.video.frame_store import FrameStore
    from app.detectors.video.pipeline import frame_count

    table = probe_frames(path)
    store = FrameStore(path, table, max_bytes=128 * 1024 * 1024)
    collector = Exhibits()
    ctx = SimpleNamespace(
        lease=SimpleNamespace(get=lambda key: models[key]),
        exhibits=collector,
        deadline=None,
        outputs=SimpleNamespace(frames=LowResVideo(path, table, ((0, len(table)),))),
    )
    tracking = FaceTrackDetector().run(ctx)
    if tracking.status != "ok":
        raise ValueError("Production face tracking failed")
    track_exhibits = tuple(collector.items)
    scheduled = sampling.uniform_indices(len(table), frame_count(sampling.duration_s(table)))
    centers = tuple(
        i
        for i, rgb in store.iter_many(scheduled)
        if detect_faces(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            detector=models["yunet.2023mar"].obj,
            min_size=settings.face_min_size,
        )
    )
    ctx.outputs = SimpleNamespace(face_review=adapter.SequenceInputs(table, store, centers, track_exhibits))
    baseline = adapter.FaceSequenceReviewDetector()
    captured = {}
    original = adapter.sequence_windows

    def capture(*args):
        captured["inputs"] = args
        return original(*args)

    applicability = baseline.applicable(ctx)
    if applicability.ok:
        with patch.object(adapter, "sequence_windows", side_effect=capture):
            actual = baseline.run(ctx)
        if "inputs" in captured:
            base_windows = original(*captured["inputs"])
            own_parity = region_windows(*captured["inputs"], core=CORE)
            if own_parity != base_windows:
                raise ValueError("Generalized region statistic no longer equals the production baseline")
            if actual.status == "ok" and actual.aggregate.raw != base_windows[0]["response"]:
                raise ValueError("Captured baseline differs from its production reading")
            mouth_windows = region_windows(*captured["inputs"])
        else:
            base_windows, mouth_windows = [], []
        status, reason = actual.status, actual.reason
    else:
        base_windows, mouth_windows, status, reason = [], [], "not_applicable", applicability.reason
    audio = decode_audio(path, source_sha256=case["sha256"])
    ctx.outputs = SimpleNamespace(
        av=AVInputs(
            table,
            tuple(e["track"] for e in track_exhibits if e["kind"] == "face_track"),
            {e["track"]["track_id"]: e["frames"] for e in track_exhibits if e["kind"] == "face_track"},
            AudioTrack("analysed", audio.pcm, len(audio.pcm)),
            store,
            audio.mapping,
        )
    )
    timing = LandmarkXcorrDetector().run(ctx)
    return {
        "status": status,
        "reason": reason,
        "base_windows": base_windows,
        "mouth_windows": mouth_windows,
        "sampling_centers": list(centers),
        "time_base": table.time_base,
        "baseline_capture_parity": bool(captured),
        "input_sha256": case["sha256"],
        "timing": {
            "status": timing.status,
            "reason": timing.reason,
            "metrics": timing.metrics,
            "track_quality": [u.quality for u in timing.unit_scores if "@" not in u.unit_ref],
        },
        "roi_bindings": roi_bindings(path, table, base_windows, mouth_windows, models),
    }


def roi_bindings(path, table, base_windows, mouth_windows, models):
    from app.core.config import settings
    from app.detectors.image.faces import detect_faces
    from app.detectors.image.gend import align_transform
    from app.detectors.video.decode import decode_frames

    selected = {}
    for arm, windows in (("base", base_windows), ("mouth", mouth_windows)):
        if windows:
            middle = windows[0]["centers"][2:-2]
            selected[arm] = middle[(len(middle) - 1) // 2]
    decoded = {f.index: f.rgb for f in decode_frames(path, table, selected.values())}
    result = {}
    for arm, i in selected.items():
        rgb = decoded[i]
        faces = detect_faces(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            detector=models["yunet.2023mar"].obj,
            min_size=settings.face_min_size,
        )
        transform = align_transform(faces[0].landmarks) if faces else None
        if transform is None:
            result[arm] = None
            continue
        matrix, side = transform
        inverse = cv2.invertAffineTransform(matrix * (112 / side))
        box = np.array(
            [
                [28, 28 if arm == "base" else 70],
                [84, 28 if arm == "base" else 70],
                [84, 84 if arm == "base" else 112],
                [28, 84 if arm == "base" else 112],
            ],
            np.float64,
        )
        polygon = np.c_[box, np.ones(4)] @ inverse.T
        result[arm] = {
            "frame_index": i,
            "pts": table.pts[i],
            "time_s": table.seconds(i),
            "polygon": polygon.tolist(),
            "width": rgb.shape[1],
            "height": rgb.shape[0],
        }
    return result


def freeze(manifest_path, output):
    root = Path(__file__).resolve().parents[2]
    pins = [
        "backend/scripts/compare_mouth_sequence_v4.py",
        "docs/mouth-sequence-v4-comparison.md",
        "backend/app/detectors/video/face_sequence.py",
        "backend/app/detectors/video/adapters/face_sequence.py",
        "backend/app/detectors/video/adapters/tracks.py",
        "backend/app/detectors/video/tracking.py",
        "backend/app/detectors/video/adapters/lipsync_xcorr.py",
        "backend/app/detectors/video/xcorr.py",
        "backend/app/models.yaml",
    ]
    body = {
        "method": METHOD,
        "manifest_sha256": digest(manifest_path),
        "implementation_sha256": {p: digest(root / p) for p in pins},
        "mouth_grid": MOUTH.reshape(8, 8).astype(int).tolist(),
        "baseline_review_point": REVIEW_POINT,
        "role": "consumed-source development; not fresh confirmation",
        "deployment_eligible": False,
    }
    with output.open("x") as stream:
        json.dump(body, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"frozen_sha256": digest(output), "manifest_sha256": body["manifest_sha256"]}))


def score(manifest_path, lock_path, output):
    import torch

    from app.detectors.image.adapters.loaders import load_yunet
    from app.detectors.video.adapters.loaders import load_dinov2_small
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    lock = json.loads(lock_path.read_text())
    root = Path(__file__).resolve().parents[2]
    if digest(manifest_path) != lock["manifest_sha256"] or any(
        digest(root / p) != sha for p, sha in lock["implementation_sha256"].items()
    ):
        raise ValueError("Inputs or implementation changed after comparison freeze")
    m = json.loads(manifest_path.read_text())
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    models, pins = {}, {}
    for mid, loader in (("yunet.2023mar", load_yunet), ("dinov2.small", load_dinov2_small)):
        spec = get_registry().manifest.model(mid)
        if spec.licence.class_ != "commercial":
            raise ValueError("Missing commercial checkpoint permission")
        files = ensure_verified(spec, force=True)
        pins[mid] = [{"name": p.name, "sha256": digest(p)} for p in files]
        models[mid] = loader(mid, "cpu", "float32")
    header = {
        "schema": "chipmunk.mouth-sequence-development/4",
        "manifest_sha256": digest(manifest_path),
        "lock_sha256": digest(lock_path),
        "method": METHOD,
        "weights": pins,
        "device": "cpu",
        "torch_version": torch.__version__,
    }
    complete = []
    if output.exists():
        old, *complete = [json.loads(l) for l in output.read_text().splitlines()]
        if old != header or [r["id"] for r in complete] != [r["id"] for r in m["items"][: len(complete)]]:
            raise ValueError("Cannot resume different inputs, weights or outcome order")
    else:
        with output.open("x") as stream:
            stream.write(json.dumps(header, allow_nan=False) + "\n")
    with output.open("a") as stream:
        for case in m["items"][len(complete) :]:
            started = time.monotonic()
            row = {**case, "status": "not_applicable", "base_windows": [], "mouth_windows": []}
            try:
                if case["status"] == "ready":
                    path = manifest_path.parent / case["path"]
                    if digest(path) != case["sha256"]:
                        raise ValueError("Prepared input bytes changed")
                    row.update(score_case(path, case, models))
                else:
                    row["reason"] = case.get("reason", "Preparation unavailable")
            except Exception as error:
                row.update(status="failed", reason=f"{type(error).__name__}: {error}")
            row["runtime_s"] = time.monotonic() - started
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "status": row["status"],
                        "seconds": row["runtime_s"],
                        "reason": row.get("reason"),
                    }
                ),
                flush=True,
            )


def evaluate(rows, arm, point, root):
    outcomes = []
    for row in rows:
        window = proposal(row, arm, point)
        overlap = temporal_iou(
            [window["start_s"], window["end_s"]] if window else None, row.get("edit_time_range_s")
        )
        spatial = 0.0
        binding = (row.get("roi_bindings") or {}).get(arm) if window else None
        if binding and row["label"]:
            ref = next((m for m in row.get("masks", []) if m["frame_index"] == binding["frame_index"]), None)
            if ref:
                if digest(root / ref["path"]) != ref["sha256"]:
                    raise ValueError("Ground-truth mask changed")
                truth = cv2.imread(str(root / ref["path"]), cv2.IMREAD_GRAYSCALE) > 0
                predicted = np.zeros((binding["height"], binding["width"]), np.uint8)
                cv2.fillConvexPoly(predicted, np.rint(binding["polygon"]).astype(np.int32), 1)
                intersection, union = (
                    np.count_nonzero(truth & (predicted > 0)),
                    np.count_nonzero(truth | (predicted > 0)),
                )
                spatial = intersection / union if union else 0.0
        outcomes.append(
            {
                "id": row["id"],
                "group": row["group"],
                "condition": row["condition"],
                "label": row["label"],
                "assessed": bool(row.get(arm + "_windows")),
                "flagged": bool(window),
                "temporal_iou": overlap,
                "localized": bool(row["label"] and overlap >= 0.2),
                "spatial_iou": spatial,
            }
        )
    return outcomes


def summary(rows):
    ai, genuine = [r for r in rows if r["label"]], [r for r in rows if not r["label"]]
    return {
        "planned": len(rows),
        "ai_edits": len(ai),
        "non_ai_controls": len(genuine),
        "assessed": sum(r["assessed"] for r in rows),
        "localized_ai": sum(r["localized"] for r in ai),
        "ai_without_flag": sum(not r["flagged"] for r in ai),
        "non_ai_flags": sum(r["flagged"] for r in genuine),
        "mean_ai_temporal_iou": sum(r["temporal_iou"] for r in ai) / len(ai) if ai else None,
        "mean_ai_spatial_iou": sum(r["spatial_iou"] for r in ai) / len(ai) if ai else None,
    }


def review(manifest_path, receipt, output):
    header, *rows = [json.loads(l) for l in receipt.read_text().splitlines()]
    m = json.loads(manifest_path.read_text())
    if header["manifest_sha256"] != digest(manifest_path) or [r["id"] for r in rows] != [
        r["id"] for r in m["items"]
    ]:
        raise ValueError("Every planned source requires one terminal score receipt")
    point = choose_point(rows)
    a, b = (
        evaluate(rows, "base", REVIEW_POINT, manifest_path.parent),
        evaluate(rows, "mouth", point, manifest_path.parent),
    )
    base, candidate = summary(a), summary(b)
    gains = sum(not x["localized"] and y["localized"] for x, y in zip(a, b, strict=True))
    losses = sum(x["localized"] and not y["localized"] for x, y in zip(a, b, strict=True))
    gates = {
        "additional_ai_localizations": candidate["localized_ai"] >= base["localized_ai"] + 3,
        "control_flags": all(
            sum(r["flagged"] for r in b if r["condition"] == c)
            <= sum(r["flagged"] for r in a if r["condition"] == c) + 1
            for c in ("original", "local-light", "local-resample", "constant-audio-delay")
        ),
        "retained_localizations": losses <= 1,
        "same_assessed_coverage": all(x["assessed"] == y["assessed"] for x, y in zip(a, b, strict=True)),
        "spatial_pointing_gain": candidate["mean_ai_spatial_iou"] >= base["mean_ai_spatial_iou"] + 0.01,
        "all_cases_completed": not any(r["status"] == "failed" for r in rows),
        "point_defined": point is not None,
    }
    report = {
        "schema": "chipmunk.mouth-sequence-development-summary/4",
        "manifest_sha256": digest(manifest_path),
        "score_receipt_sha256": digest(receipt),
        "candidate_review_point": point,
        "baseline": base,
        "candidate": candidate,
        "gains": gains,
        "losses": losses,
        "gates": gates,
        "primary_for_confirmation": "mouth" if all(gates.values()) else None,
        "deployment_eligible": False,
        "conditions": {
            c: {
                "baseline": summary([r for r in a if r["condition"] == c]),
                "candidate": summary([r for r in b if r["condition"] == c]),
            }
            for c in sorted({r["condition"] for r in rows})
        },
        "cases": {"baseline": a, "candidate": b},
        "limitations": [
            "Consumed actor/source development; one AI editing family, 8 connected source groups.",
            "Token position is a measurement region, not a pixel attribution or proof of AI editing.",
            "No outside independent review. Current timing measurements are separately retained in the score receipt.",
        ],
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("conditions", "cases", "limitations")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "score", "review"])
    p.add_argument("manifest", type=Path)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path, nargs="?")
    a = p.parse_args()
    if a.stage == "freeze":
        freeze(a.manifest, a.input)
    elif a.output is None:
        p.error("score/review requires an output path")
    elif a.stage == "score":
        score(a.manifest, a.input, a.output)
    else:
        review(a.manifest, a.input, a.output)
