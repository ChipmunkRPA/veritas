"""Replay every ordinary calibration pixel, common clock, operation and PCM sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings
from app.detectors.audio.decode import decode_audio
from app.detectors.image.adapters.loaders import load_yunet
from app.detectors.image.faces import detect_faces
from app.detectors.video.decode import decode_frames, probe_frames
from app.engine.fetchers import ensure_verified
from app.engine.registry import get_registry
from scripts.liveportrait_mouth_benchmark import digest
from scripts.prepare_mouth_context_v10 import TASK_SHA, clock_span, code_pins, geometry_frames, operations
from scripts.prepare_mouth_edit_v4 import composite
from scripts.run_mouth_learning_v8 import write


def audit(plan_path, manifest_path, output):
    recipe = json.loads(plan_path.read_text())
    data = json.loads(manifest_path.read_text())
    if (
        recipe["implementation_sha256"] != code_pins()
        or data["plan_sha256"] != digest(plan_path)
        or digest(Path(recipe["task"])) != TASK_SHA
    ):
        raise ValueError("Frozen preparation/task/manifest bindings changed")
    ensure_verified(get_registry().manifest.model("yunet.2023mar"), force=True)
    detector = load_yunet("yunet.2023mar", "cpu", "float32").obj
    source_root = Path(recipe["source_root"])
    records = []
    total_frames = total_masks = total_pcm_samples = 0
    for source in recipe["sources"]:
        path = source_root / source["path"]
        if digest(path) != source["sha256"]:
            raise ValueError("Reserved original bytes changed")
        table = probe_frames(path)
        audio = decode_audio(path, source_sha256=source["sha256"])
        span = clock_span(table, audio)
        if span != source["clock_span"]:
            raise ValueError("Common source clock changed")
        n = span["frames"]
        original = [f.rgb for f in decode_frames(path, table, range(n))]
        mesh = geometry_frames(original, table, Path(recipe["task"]))
        start, end = (n - 20) // 2, (n - 20) // 2 + 20
        fresh = {}
        for i in range(start, end):
            faces = detect_faces(
                cv2.cvtColor(original[i], cv2.COLOR_RGB2BGR),
                detector=detector,
                min_size=settings.face_min_size,
            )
            if len(faces) != 1:
                raise ValueError("Fresh ordinary geometry is not unique")
            fresh[i] = operations(original[i], mesh[i], faces[0])
        a, b = span["source_sample_range"]
        for case in [r for r in data["items"] if r["group"] == source["group"]]:
            record = {"id": case["id"], "status": "failed", "frames": 0, "masks": 0}
            try:
                movie = manifest_path.parent / case["path"]
                if (
                    case["status"] != "ready"
                    or case["label"] != 0
                    or case["role"] != "point"
                    or digest(movie) != case["sha256"]
                    or case["source_clock_span"] != span
                    or case["original_audio_mapping"] != audio.mapping
                ):
                    raise ValueError("Calibration role/bytes/source clocks differ")
                clock = probe_frames(movie)
                if len(clock) != n or any(abs(clock.seconds(i) - i / 25) > 1e-9 for i in range(n)):
                    raise ValueError("Calibration encoded frame clock differs")
                decoded = list(decode_frames(movie, clock, range(n)))
                if len(decoded) != n:
                    raise ValueError("Calibration frame missing")
                masks = {r["frame_index"]: r for r in case["masks"]}
                if set(masks) != set(range(start, end)):
                    raise ValueError("Planned ordinary support/mask positions changed")
                for i, frame in enumerate(decoded):
                    expected = original[i]
                    if i in masks:
                        binding = masks[i]
                        values, geometry = fresh[i]
                        opacity = min(1.0, (i - start + 1) / 2, (end - i) / 2)
                        if case["condition"] != "original":
                            transformed, alpha = values[case["condition"]]
                            expected = composite(original[i], transformed, alpha, opacity)
                        if (
                            any(binding[k] != v for k, v in geometry.items())
                            or binding["opacity"] != opacity
                            or binding["pts"] != table.pts[i]
                            or binding["source_time_s"] != table.seconds(i)
                            or binding["source_rgb_sha256"]
                            != hashlib.sha256(original[i].tobytes()).hexdigest()
                        ):
                            raise ValueError("Retained operation geometry/support/source time differs")
                        mask = manifest_path.parent / binding["path"]
                        changed = np.any(frame.rgb != original[i], axis=2)
                        if (
                            digest(mask) != binding["sha256"]
                            or not np.array_equal(changed, cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE) > 0)
                            or int(changed.sum()) != binding["changed_pixels"]
                        ):
                            raise ValueError("Exact ordinary changed-pixel mask differs")
                        record["masks"] += 1
                    if not np.array_equal(frame.rgb, expected):
                        raise ValueError("Native ordinary operation or surrounding pixels do not reproduce")
                    record["frames"] += 1
                sound = decode_audio(movie, source_sha256=case["sha256"])
                if not np.array_equal(sound.pcm, audio.pcm[a:b]) or len(sound.pcm) != n * 640:
                    raise ValueError("Common PCM span differs from the selected original samples")
                total_pcm_samples += len(sound.pcm)
                record["status"] = "ok"
            except Exception as error:
                record["reason"] = f"{type(error).__name__}: {error}"
            total_frames += record["frames"]
            total_masks += record["masks"]
            records.append(record)
        print(
            json.dumps(
                {
                    "source": source["id"],
                    "checked": len(records),
                    "passed": sum(r["status"] == "ok" for r in records),
                }
            ),
            flush=True,
        )
    body = {
        "schema": "chipmunk.mouth-context-calibration-source-audit/10",
        "plan_sha256": digest(plan_path),
        "manifest_sha256": digest(manifest_path),
        "audit_implementation_sha256": digest(Path(__file__)),
        "planned": 24,
        "passed": sum(r["status"] == "ok" for r in records),
        "native_frames": total_frames,
        "exact_masks": total_masks,
        "common_pcm_samples": total_pcm_samples,
        "cases": records,
        "scope": "All native calibration frames, exact masks, fresh ordinary operation geometry, source/encoded clocks and selected original PCM; no head inference or outside validation.",
    }
    write(output, body)
    print(json.dumps({k: v for k, v in body.items() if k != "cases"}))
    if body["passed"] != 24:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    audit(a.plan, a.manifest, a.output)
