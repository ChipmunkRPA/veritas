"""Verify the fixed new stock-profile files; no generation, inference or retries."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
TRIAL = BASE / "audio-matched-assets-v78"
DESTINATION = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify/audio-calibration-sourcebank-v67/stock-profiles-v78")
MANIFEST = BASE / "audio-source-ancestry-v75/manifest.json"
MANIFEST_SHA = "217190341f0b49ed76a38d4282a3760e271db284031fdece6056d59bbda7e519"
METADATA = BASE / "audio-stock-voice-preparation-v69/publisher-metadata.json"
METADATA_SHA = "ad8a022ebed0ec64da12cbe1f2527f8e4240fbd813e03feefbda7443fbc57d39"
REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, body):
    with path.open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")


def planned(manifest, metadata):
    if metadata["sha"] != REVISION or metadata["cardData"]["license"] != "apache-2.0":
        raise ValueError("Pinned Apache-2.0 stock model metadata required")
    siblings = {row["rfilename"]: row for row in metadata["siblings"]}
    result = {}
    for prompt in manifest["pending_matched_generation"]:
        name = prompt["voice_asset"]
        if not re.fullmatch(r"voices/[ab][fm]_[a-z0-9_]+\.pt", name):
            raise ValueError("Safe fixed English stock-profile path required")
        row = siblings[name]
        if (row["lfs"]["sha256"] != prompt["voice_asset_sha256"] or row["size"] != row["lfs"]["size"]
                or not 0 < row["size"] <= 1_000_000):
            raise ValueError("Bounded exact profile publisher binding required")
        value = {"file": name, "bytes": row["size"], "sha256": row["lfs"]["sha256"],
                 "url": f"https://huggingface.co/hexgrad/Kokoro-82M/resolve/{REVISION}/{name}"}
        if name in result and result[name] != value:
            raise ValueError("Conflicting stock-profile commitments")
        result[name] = value
    if len(manifest["pending_matched_generation"]) != 64 or len(result) != 16:
        raise ValueError("All64 fixed prompts and16 stock profiles required")
    return [result[key] for key in sorted(result)]


def acquire():
    if sha(MANIFEST) != MANIFEST_SHA or sha(METADATA) != METADATA_SHA:
        raise ValueError("Corrected source or publisher metadata changed")
    assets = planned(json.loads(MANIFEST.read_text()), json.loads(METADATA.read_text()))
    if TRIAL.exists() or DESTINATION.exists():
        raise FileExistsError("Never downloaded destination required; preserve every failed attempt")
    if not Path("/Volumes/SEAGATE").is_mount() or shutil.disk_usage(DESTINATION.parent).free < 3 * 1024**3:
        raise ValueError("Authorized mounted drive with fixed reserve required")
    TRIAL.mkdir()
    (DESTINATION / "voices").mkdir(parents=True)
    write(TRIAL / "plan.json", {"schema": "chipmunk.matched-stock-profile-plan/78", "assets": assets,
                               "manifest_sha256": MANIFEST_SHA, "metadata_sha256": METADATA_SHA,
                               "revision": REVISION, "code_sha256": sha(Path(__file__)), "new_model_calls": 0})
    completed = []
    try:
        for asset in assets:
            path = DESTINATION / asset["file"]
            request = urllib.request.Request(asset["url"], headers={"User-Agent": "Chipmunk-Verify-public-evaluation/1"})
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(asset["bytes"] + 1)
            if len(raw) != asset["bytes"] or hashlib.sha256(raw).hexdigest() != asset["sha256"]:
                raise ValueError("Downloaded stock profile failed publisher size/SHA256")
            with path.open("xb") as stream:
                stream.write(raw)
            completed.append(asset)
            print(json.dumps({"verified_profiles": len(completed), "planned": 16}), flush=True)
    except BaseException as error:
        write(TRIAL / "failure.json", {"type": type(error).__name__, "detail": str(error)[:250],
                                     "completed_profiles": len(completed), "new_model_calls": 0})
        raise
    write(TRIAL / "receipt.json", {"schema": "chipmunk.matched-stock-profile-receipt/78", "assets": completed,
                                  "revision": REVISION, "manifest_sha256": MANIFEST_SHA, "new_model_calls": 0})
    summary = {"schema": "chipmunk.matched-stock-profile-summary/78", "verified_profiles": len(completed),
               "total_bytes": sum(row["bytes"] for row in completed), "revision": REVISION,
               "publisher_metadata_sha256": METADATA_SHA, "source_manifest_sha256": MANIFEST_SHA,
               "receipt_sha256": sha(TRIAL / "receipt.json"), "code_sha256": sha(Path(__file__)),
               "new_model_calls": 0, "scope": "Whole stock-profile file integrity only. No tensor/runtime/generation qualification or detector result."}
    write(TRIAL / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    acquire()
