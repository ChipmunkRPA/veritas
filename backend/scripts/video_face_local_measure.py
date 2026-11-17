"""Measure frozen face-change proposals on exactly decoded delivered pictures."""

from __future__ import annotations

import argparse
import json
from contextlib import closing
from pathlib import Path

from scripts import video_common_sources_v89 as io

ENCODER = "opencv.sface.2021dec"
LOCATOR = "yunet.2023mar"
SCHEMA = "chipmunk.video-face-local-measurements/1"


def assets():
    from app.detectors.image.adapters import loaders
    from app.engine.fetchers import ensure_verified

    output = {}
    for name in (LOCATOR, ENCODER):
        asset = loaders._entry(name)
        if asset.licence.class_ != "commercial":
            raise ValueError("Require commercially permitted face measurements")
        ensure_verified(asset, force=True)
        output[name] = [entry.sha256 for entry in asset.source.files]
    return output


def as_article(candidate, table, anchors):
    return {"range": [candidate.start, candidate.end],
            "ticks": [table.pts[candidate.start], table.pts[candidate.end]],
            "time_base": table.time_base,
            "anchors": {role: anchors[i] for role, i in candidate.anchors().items()}}


def observations(path, locator, encoder):
    from app.detectors.video import face_local_detail as study
    from app.detectors.video import face_return
    from app.detectors.video.decode import decode_frames
    from scripts.video_face_temporal_readout import table_for

    table = table_for(path)
    if not study.supported(table, ("h264", "yuv420p")):
        raise ValueError("Delivered test source has unexpected scope")
    times, sparse = face_return.schedule(table)
    schedule = set(study.SCHEDULE) | set(sparse)
    samples, details, anchors, multiple = {}, {}, {}, set()
    with closing(decode_frames(path, table, sorted(schedule), max_side=None, timeout_s=30)) as frames:
        for frame in frames:
            frame_id = frame.index
            if not frame.pts_verified or frame.pts != table.pts[frame_id]:
                raise ValueError("Native observation differs from the scheduled timestamp")
            sample, detail, anchor, many = study.observe(frame.rgb, locator, encoder, with_texture=frame_id in study.SCHEDULE)
            samples[frame_id] = sample
            if many:
                multiple.add(frame_id)
            if detail is not None:
                details[frame_id] = detail
                anchors[frame_id] = anchor | {"index": frame_id, "pts": frame.pts}
    if samples.keys() != schedule:
        raise ValueError("Native readout did not cover the planned source pictures")
    base = face_return.highest(times, sparse, samples) if not multiple.intersection(sparse) else None
    existing = [base.start, base.end] if base and face_return.distinct(base, table, None) else None
    # The production 100-picture profile currently runs only the sparse brief-return pass.
    proposals = study.proposals(samples, details)
    packed = [[pair.start, pair.end, *[round(value, 9) for value in pair.features]] for pair in proposals]
    return {"existing": existing, "existing_assessed": base is not None, "proposals": packed,
            "anchors": list(anchors.values()), "time_base": table.time_base,
            "first_tick": table.pts[0], "step_tick": table.pts[1] - table.pts[0]}


def measure(folder, output):
    import cv2

    from app.detectors.image.adapters.loaders import load_yunet
    from app.detectors.video import face_frequency, face_local_detail
    from app.detectors.video.adapters.loaders import load_face_appearance

    cv2.setNumThreads(1)
    manifest = folder / "manifest.json"
    frozen = io.read(manifest)
    weights = assets()
    locator = load_yunet(LOCATOR, "cpu", "float32").obj
    encoder = load_face_appearance(ENCODER, "cpu", "float32").obj
    header = {"schema": SCHEMA, "manifest_sha256": io.sha(manifest), "weights": weights,
              "implementation_sha256": io.sha(Path(face_local_detail.__file__)),
              "frequency_sha256": io.sha(Path(face_frequency.__file__)),
              "readout_sha256": io.sha(Path(__file__)), "purpose": frozen["purpose"]}
    index = {entry["file"]: entry for entry in frozen["rows"]}
    current = set()
    if output.exists():
        with output.open() as stream:
            if json.loads(next(stream)) != header:
                raise ValueError("The measured inputs or code changed")
            for line in stream:
                recorded = json.loads(line)
                if recorded["file"] in current or index[recorded["file"]]["input_sha256"] != recorded["input_sha256"]:
                    raise ValueError("Saved measurement commitments differ")
                current.add(recorded["file"])
    with output.open("a" if current else "x") as stream:
        if not current:
            stream.write(json.dumps(header, sort_keys=True) + "\n")
        for i, (file, case) in enumerate(index.items(), 1):
            if file in current:
                continue
            path = folder / file
            if io.sha(path) != case["input_sha256"]:
                raise ValueError("Video differs from the prepared source receipt")
            result = case | observations(path, locator, encoder)
            stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            if i % 14 == 0 or i == len(index):
                print(f"Measured {i}/{len(index)} {frozen['purpose']} files", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("delivery", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    measure(args.delivery, args.output)
