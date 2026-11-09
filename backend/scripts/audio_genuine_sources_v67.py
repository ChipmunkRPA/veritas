"""Acquire four pinned genuine-speech corpus files for a new calibration panel.

No model imports, dataset-server substitutions, credentials or retries. Original
publisher checksums bind complete files before any source selection or decoding.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
TRIAL = BASE / "audio-genuine-sources-v67"
DESTINATION = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify/audio-calibration-sourcebank-v67/libri360")
METADATA = BASE / "audio-v8/source-planning/libri-pinned-file-metadata.json"
METADATA_SHA = "b1ca3b912f90c112f7df124b98b7970914ca467c230fb4a300b0ccc66195e4c6"
REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
RESERVE = 1536 * 1024**2


def write(path, body):
    with path.open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")


def assets(metadata):
    if metadata["sha"] != REVISION or metadata["cardData"]["license"] != ["cc-by-4.0"]:
        raise ValueError("Pinned original CC BY 4.0 corpus metadata required")
    selected = []
    for index in range(4):
        name = f"clean/train.360/{index:04d}.parquet"
        row = next(row for row in metadata["siblings"] if row["rfilename"] == name)
        if row["size"] != row["lfs"]["size"] or row["size"] > 600_000_000:
            raise ValueError("Exact bounded publisher size required")
        selected.append({"source_path": name, "name": f"libri360-{index:04d}.parquet",
                         "bytes": row["size"], "sha256": row["lfs"]["sha256"],
                         "url": f"https://huggingface.co/datasets/openslr/librispeech_asr/resolve/{REVISION}/{name}"})
    return selected


def acquire():
    raw = METADATA.read_bytes()
    if hashlib.sha256(raw).hexdigest() != METADATA_SHA:
        raise ValueError("Previously recorded publisher metadata changed")
    planned = assets(json.loads(raw))
    if not Path("/Volumes/SEAGATE").is_mount():
        raise ValueError("Authorized external development drive must be mounted")
    if TRIAL.exists() or DESTINATION.exists():
        raise FileExistsError("Preserve existing or failed acquisition; explicit new attempt required")
    if shutil.disk_usage(DESTINATION.parent.parent).free < sum(row["bytes"] for row in planned) + RESERVE:
        raise ValueError("External acquisition space and reserve required")
    TRIAL.mkdir()
    DESTINATION.mkdir(parents=True)
    write(TRIAL / "plan.json", {"schema": "chipmunk.audio-genuine-corpus-acquisition/67",
                               "metadata_sha256": METADATA_SHA, "revision": REVISION,
                               "assets": planned, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "source_roles_assigned": False, "neural_calls": 0})
    receipt, started = [], time.monotonic()
    try:
        for row in planned:
            destination = DESTINATION / row["name"]
            partial = destination.with_suffix(".partial")
            checksum, count, file_started = hashlib.sha256(), 0, time.monotonic()
            request = urllib.request.Request(row["url"], headers={"User-Agent": "Chipmunk-Verify-public-evaluation/1"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as stream:
                while chunk := response.read(4 * 1024**2):
                    if time.monotonic() - file_started > 900 or count + len(chunk) > row["bytes"]:
                        raise ValueError("Bounded original download exceeded time or size")
                    if shutil.disk_usage(DESTINATION).free < RESERVE or shutil.disk_usage(TRIAL).free < RESERVE:
                        raise ValueError("Preserve download: disk reserve reached")
                    stream.write(chunk)
                    checksum.update(chunk)
                    count += len(chunk)
                    if count % (64 * 1024**2) == 0:
                        print(json.dumps({"asset": row["name"], "downloaded_bytes": count, "planned_bytes": row["bytes"]}), flush=True)
            if count != row["bytes"] or checksum.hexdigest() != row["sha256"]:
                raise ValueError("Complete publisher SHA256/size verification failed")
            partial.rename(destination)
            receipt.append({**row, "status": "publisher-verified", "duration_s": time.monotonic() - file_started})
            write(TRIAL / f"asset-{len(receipt):02d}.json", receipt[-1])
            print(json.dumps({"asset": row["name"], "status": "publisher-verified", "bytes": count}), flush=True)
        result = {"schema": "chipmunk.audio-genuine-corpus-receipt/67", "revision": REVISION,
                  "metadata_sha256": METADATA_SHA, "archives": receipt,
                  "total_bytes": sum(row["bytes"] for row in receipt), "duration_s": time.monotonic() - started,
                  "neural_calls": 0, "audio_decodes": 0,
                  "scope": "Complete original corpus files verified by publisher SHA256. Not selected new speakers, decoded source qualification or model performance."}
        write(TRIAL / "receipt.json", result)
        write(DESTINATION / "receipt.json", result)
        print(json.dumps({"completed": len(receipt), "total_bytes": result["total_bytes"], "neural_calls": 0}), flush=True)
    except BaseException as error:
        write(TRIAL / "failure.json", {"type": type(error).__name__, "detail": str(error)[:300],
                                     "completed_assets": len(receipt), "partial_files_preserved": True,
                                     "neural_calls": 0})
        raise


if __name__ == "__main__":
    acquire()
