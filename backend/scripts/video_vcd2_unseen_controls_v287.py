"""Qualify fresh VCD2 camera controls from the pinned publisher catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from scripts import video_vcd2_genuine_v195 as catalog
from scripts import video_vcd2_genuine_v197 as transfer
from scripts.video_appearance_corrected_v112 import scoped

ROOT = catalog.ROOT
TRIAL = catalog.ARCHIVE / "video-vcd2-unseen-controls-v287"
PANEL = ROOT / "docs/evaluation-panels"
PUBLIC = PANEL / "video-vcd2-unseen-controls-freeze-v287.json"
PRIOR = PANEL / "video-vcd2-genuine-qualification-complete-v197.json"
SEED = "chipmunk-vcd2-unseen-controls-v287"
PER_STRATUM = 2
CODE = (
    "backend/scripts/video_vcd2_unseen_controls_v287.py",
    "backend/scripts/video_vcd2_genuine_v195.py",
    "backend/scripts/video_vcd2_genuine_v197.py",
    "backend/app/detectors/video/decode.py",
    "backend/tests/test_video_vcd2_unseen_controls_v287.py",
    "docs/video-vcd2-unseen-controls-v287-method.md",
)
require, sha, write = catalog.require, catalog.sha, catalog.write


def select(raw, benchmark, old):
    used = {row["recording_stem"] for row in old}
    cameras = {row["camera_type"] for row in old}
    ranked = sorted(benchmark, key=lambda stem: hashlib.sha256(f"{SEED}:{stem}".encode()).digest())
    rows = []
    for stratum in catalog.STRATA:
        count = 0
        for stem in ranked:
            source, bench = raw[stem], benchmark[stem]
            camera = source["log_recording_camera_code"].lower()
            resolution = source["video_resolution"]
            if (
                stem in used
                or bench["group"] != "TH"
                or source["si_ti_group"] != stratum
                or resolution not in {"1280x720", "1920x1080"}
                or not catalog.CAMERA_CODE.fullmatch(camera)
                or camera in cameras
            ):
                continue
            require(
                bench["video_resolution"] == resolution
                and bench["log_recording_camera_code"].lower() == camera,
                "Publisher benchmark and camera catalog differ",
            )
            cameras.add(camera)
            rows.append(
                {
                    "slot": len(rows) + 1,
                    "recording_stem": stem,
                    "camera_type": camera,
                    "stratum": stratum,
                    "resolution": resolution,
                    "benchmark_path": bench["relative_path"],
                }
            )
            count += 1
            if count == PER_STRATUM:
                break
        require(count == PER_STRATUM, "Insufficient additional distinct camera types in a stratum")
    return rows


def read(path):
    return json.loads(path.read_bytes())


def context(metadata):
    historic, old = transfer.context(metadata)
    result = read(PRIOR)
    require(
        result["recording_components_qualified"] == 12
        and result["freeze_sha256"] == sha(transfer.TRIAL / "freeze.json")
        and read(transfer.TRIAL / "freeze.json") == historic,
        "Exact completed previous source cohort required",
    )
    raw = catalog.checked_index(metadata, "vcd2_metadata.csv", "raw_mkv_list.txt", catalog.RAW_PATH, 838)
    bench = catalog.checked_index(
        metadata, "benchmarking-set_metadata.csv", "benchmarking_mkv_list.txt", catalog.BENCH_PATH, 120
    )
    rows = select(raw, bench, old)
    identity = {
        "schema": "chipmunk.video-vcd2-unseen-controls-private/287",
        "prior_result_sha256": sha(PRIOR),
        "publisher_revision": catalog.REVISION,
        "publisher_pins": {name: digest for name, (_, digest) in catalog.PINS.items()},
        "code_sha256": {path: sha(ROOT / path) for path in CODE},
        "decoder_sha256": {name: sha(Path(shutil.which(name)).resolve()) for name in ("ffmpeg", "ffprobe")},
        "relative_ordinary_interval_s": list(catalog.WINDOW),
        "cases": rows,
    }
    return identity, rows


def public_record(identity, rows):
    return {
        **{key: value for key, value in identity.items() if key not in {"cases", "schema"}},
        "schema": "chipmunk.video-vcd2-unseen-controls-freeze/287",
        "private_freeze_sha256": sha(TRIAL / "freeze.json"),
        "planned_groups": len(rows),
        "recordings_per_stratum": PER_STRATUM,
        "selection_seed": SEED,
        "camera_types_excluded_from_prior": 12,
        "new_neural_calls": 0,
    }


def verify(identity, rows):
    require(
        read(TRIAL / "freeze.json") == identity and read(PUBLIC) == public_record(identity, rows),
        "Exact prospective webcam roster required",
    )


def readout(identity, rows):
    verify(identity, rows)
    counts, resolutions, verified = Counter(), Counter(), []
    for row in rows:
        slot = row["slot"]
        folder = TRIAL / f"slot-{slot:02d}-attempt-01"
        qualification = folder / "qualification.json"
        if not qualification.is_file():
            counts["attempt_unqualified" if folder.is_dir() else "unattempted"] += 1
            continue
        acquired, receipt = read(folder / "acquisition.json"), read(qualification)
        media = folder / "source.mkv"
        require(
            acquired["source_url"] == catalog.AZURE + row["benchmark_path"]
            and media.stat().st_size == acquired["bytes"]
            and media.resolve(strict=True) == media
            and sha(media) == acquired["sha256"] == receipt["sha256"],
            "Exact acquired source required",
        )
        clock = catalog.media_record(media, row["resolution"])
        require(
            receipt == {"slot": slot, "attempt": 1, "sha256": acquired["sha256"], **clock},
            "Independent native clock readback differs",
        )
        counts["qualified"] += 1
        resolutions[row["resolution"]] += 1
        verified.append((acquired["bytes"], clock["clock"]["frame_count"], sha(qualification)))
    return {
        "schema": "chipmunk.video-vcd2-unseen-controls-result/287",
        "status": "complete" if counts["qualified"] == len(rows) else "partial",
        "freeze_sha256": sha(PUBLIC),
        "planned_groups": len(rows),
        "outcomes": dict(counts),
        "qualified_by_resolution": dict(resolutions),
        "verified_bytes": sum(row[0] for row in verified),
        "verified_frames": sum(row[1] for row in verified),
        "qualification_bundle_sha256": hashlib.sha256(json.dumps(verified).encode()).hexdigest(),
        "new_neural_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "qualify", "readout"))
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--slot", type=int)
    args = parser.parse_args()
    if (args.command == "qualify") != (args.slot is not None):
        parser.error("--slot is required only for qualification")
    identity, rows = context(args.metadata_dir)
    if args.command == "preflight":
        print(json.dumps({"planned_groups": len(rows), "recordings_per_stratum": PER_STRATUM}))
    elif args.command == "freeze":
        require(not PUBLIC.exists(), "Preserve the prior public freeze")
        catalog.archive_ready()
        catalog.private_directory(TRIAL)
        write(TRIAL / "freeze.json", identity)
        write(PUBLIC, public_record(identity, rows))
    elif args.command == "qualify":
        verify(identity, rows)
        scoped(transfer.qualify, {"TRIAL": TRIAL})(identity, rows, args.slot, 1)
    else:
        print(json.dumps(readout(identity, rows), indent=2, allow_nan=False))
