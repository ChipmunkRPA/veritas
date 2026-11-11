"""Qualify exact local room-impulse assets without rendering or model calls."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
from scipy.io import loadmat, whosmat

from scripts import audio_room_transfer_preflight_v296 as room

stereo = room.stereo
require = stereo.require
PREVIOUS = stereo.PANEL / "audio-room-transfer-preflight-v296-authored.json"
PREVIOUS_SHA = "32c31d286c99f33fa25933452e9d21411ec2e457f5f55f807ded67b6924f2215"
CODE = (
    "backend/scripts/audio_room_assets_v299.py",
    "backend/tests/test_audio_room_assets_v299.py",
    "docs/audio-room-assets-v299-method.md",
)
SURREY = "IoSR-Surrey/IoSR_ListeningRoom_BRIRs/d36af1bbd0f4083a9f9ee678ff8f499044e851a7"
HUDDERSFIELD = "APL-Huddersfield/MAIR-Library-and-Renderer/ef37a42552e6b17d94611856cbd3404e308e10f5"
RAW = "https://raw.githubusercontent.com/"
SURREY_RATE, HUDDERSFIELD_RATE = 48_000, 44_100
SURREY_POSITIONS = {"FC": (3, 0), "SiL": (11, 90), "SiR": (12, -90)}
HUDDERSFIELD_BLOBS = {
    "S01": ("1a6469467a7b4fc54ca51a8479471443e6a5808e", "011790072b4634f1a816041bd3a2333cc9a606e5"),
    "S03": ("57c432e5cde33664f018a90548cebd159d4a3466", "c26c16605bb14eb73ec3c6375f2885c4b3fa29d9"),
    "S07": ("788ed202dc80cc6fa8c3facb713a9d488c7b1adf", "c4d1b91517284dd6e5ca34cffac1cb0f91a5b1e7"),
}
PUBLISHER_GUIDES = {
    "surrey": {
        "url": RAW + SURREY + "/IoSR_ListeningRoom_BRIR.pdf",
        "bytes": 3_161_503,
        "sha256": "43322c46336dc3720e24603f9031e75b5a3e9c451251ef6d0af05c7f3d623754",
    },
    "huddersfield": {
        "url": RAW + HUDDERSFIELD + "/MAIR%20Technical%20Descriptions%20and%20User%20Guide.pdf",
        "bytes": 1_460_525,
        "sha256": "105e20e1400d632366be0195a9b4c0337d2092fd37c56e1d257a0681a5b04a1d",
    },
    "huddersfield_circular_layout": {
        "url": "https://pure.hud.ac.uk/ws/portalfiles/portal/14022860/AAM_AES2018_Millns_and_Lee.pdf",
        "bytes": 827_317,
        "sha256": "f829a5f2df9a8592d108193fdd362f97a0a88eccd4a7e7410cb9d57394f92621",
    },
}


@dataclass(frozen=True)
class Asset:
    url: str
    bytes: int
    git_blob_sha1: str

    @property
    def filename(self):
        return self.url.rsplit("/", 1)[1]


SURREY_ASSET = Asset(
    RAW + SURREY + "/IoSR_ListeningRoom_BRIRs_MATLAB/angle_0.0.mat",
    14_282_074,
    "9214e19a94dca7609c9e8aad4805bc277f5b33a6",
)
HUDDERSFIELD_ASSETS = {
    speaker: tuple(
        Asset(
            f"{RAW}{HUDDERSFIELD}/360StageArrays/KU100-0deg/VR_0DegKU100_M0{index}_{speaker}_{ear}.wav",
            529_280,
            blob,
        )
        for index, (ear, blob) in enumerate(zip(("L", "R"), blobs, strict=True), start=1)
    )
    for speaker, blobs in HUDDERSFIELD_BLOBS.items()
}


def publisher_plan():
    return {
        "publisher_guides": PUBLISHER_GUIDES,
        "development": {
            "release": "https://doi.org/10.15126/surreydata.00813511",
            "license": "CC-BY-4.0",
            "asset": asdict(SURREY_ASSET),
            "sample_rate_hz": SURREY_RATE,
            "ear_order": ["left", "right"],
            "positions": {
                speaker: {
                    "channel_one_based": index,
                    "azimuth_degrees_left_positive": angle,
                    "elevation_degrees": 0,
                }
                for speaker, (index, angle) in SURREY_POSITIONS.items()
            },
        },
        "holdout": {
            "release": "https://zenodo.org/records/2639677",
            "license": "CC-BY-4.0",
            "array": "360StageArrays/KU100-0deg",
            "sample_rate_hz": HUDDERSFIELD_RATE,
            "ear_order": ["left", "right"],
            "assets": {
                speaker: [asdict(asset) for asset in pair] for speaker, pair in HUDDERSFIELD_ASSETS.items()
            },
            "speaker_id_angle_mapping": None,
        },
    }


def pinned_bytes(directory, asset):
    path = directory / asset.filename
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        info = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(info.st_mode) and info.st_size == asset.bytes,
            f"Unexpected file type or size: {asset.filename}",
        )
        raw = stream.read(asset.bytes + 1)
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    require(
        len(raw) == asset.bytes and blob == asset.git_blob_sha1, f"Publisher bytes differ: {asset.filename}"
    )
    return raw, {"filename": asset.filename, **asdict(asset), "sha256": stereo.digest(raw)}


def surrey_impulses(raw):
    """Map the publisher's one MATLAB array without trimming or normalizing ears."""
    require(raw.startswith(b"MATLAB 5.0 MAT-file"), "Unsupported Surrey MATLAB container")
    variables = whosmat(io.BytesIO(raw))
    require(len(variables) == 1 and variables[0][0] == "brir", "One documented Surrey brir variable required")
    _, shape, dtype = variables[0]
    require(
        len(shape) == 3
        and shape[1:] == (2, 24)
        and 0 < shape[0] <= 4 * SURREY_RATE
        and dtype in ("double", "single"),
        "Expected bounded samples × ears × loudspeakers in Surrey brir",
    )
    values = loadmat(io.BytesIO(raw), variable_names=("brir",), verify_compressed_data_integrity=True)["brir"]
    require(
        values.shape == shape and values.dtype.kind == "f" and np.isfinite(values).all(),
        "Surrey coefficients must be finite real floating point with the documented geometry",
    )
    return {
        speaker: room.Impulse(
            "surrey-iosr-angle-0.0",
            speaker,
            SURREY_RATE,
            room.IMPULSE_UNIT,
            (0, 0),
            np.ascontiguousarray(values[:, :, index - 1], dtype="<f8"),
        )
        for speaker, (index, _) in SURREY_POSITIONS.items()
    }


def check_surrey(directory):
    raw, asset = pinned_bytes(directory, SURREY_ASSET)
    if raw.startswith(b"MATLAB 7.3 MAT-file"):
        return {"status": "bytes_verified_mat73_unsupported", "asset": asset, "positions": {}}
    impulses = surrey_impulses(raw)
    positions = {speaker: impulse.identity() for speaker, impulse in impulses.items()}
    require(
        len({row["coefficients_sha256"] for row in positions.values()}) == 3,
        "Surrey front and lateral positions must have distinct coefficients",
    )
    return {"status": "documented_positions_qualified", "asset": asset, "positions": positions}


def huddersfield_channel(raw):
    """Check the native 32-bit mono file and expose only coefficient metadata."""
    with sf.SoundFile(io.BytesIO(raw)) as wav:
        require(
            wav.format in ("WAV", "WAVEX")
            and wav.channels == 1
            and wav.samplerate == HUDDERSFIELD_RATE
            and wav.subtype in ("PCM_32", "FLOAT")
            and 0 < wav.frames <= 4 * HUDDERSFIELD_RATE,
            "Expected bounded mono 44.1 kHz 32-bit Huddersfield WAV",
        )
        record = {"sample_rate_hz": wav.samplerate, "encoding": wav.subtype, "samples": wav.frames}
        values = wav.read(wav.frames, dtype="float64")
    require(
        values.shape == (record["samples"],) and np.isfinite(values).all() and np.any(values),
        "Huddersfield channel is truncated, nonfinite or silent",
    )
    return {**record, "coefficients_sha256": stereo.digest(values.astype("<f8").tobytes())}


def check_huddersfield(directory):
    pairs = {}
    for speaker, files in HUDDERSFIELD_ASSETS.items():
        channels = []
        for ear, asset in zip(("left", "right"), files, strict=True):
            raw, record = pinned_bytes(directory, asset)
            channels.append({"ear": ear, **record, **huddersfield_channel(raw)})
        left, right = channels
        require(
            all(left[key] == right[key] for key in ("sample_rate_hz", "encoding", "samples"))
            and left["coefficients_sha256"] != right["coefficients_sha256"],
            "Huddersfield pair has different native headers or identical ears",
        )
        pairs[speaker] = channels
    require(
        len({row["coefficients_sha256"] for channels in pairs.values() for row in channels}) == 6,
        "Distinct Huddersfield responses required for all three candidate source IDs",
    )
    return {
        "status": "bytes_and_format_verified_angles_unassigned",
        "positions": {},
        "candidate_pairs": pairs,
    }


def verify_preflight():
    require(stereo.sha(PREVIOUS) == PREVIOUS_SHA, "Exact v296 authored receipt required")
    previous = json.loads(PREVIOUS.read_text())
    require(
        previous["status"] == "complete"
        and previous["model_calls"] == previous["measured_responses_opened"] == 0
        and all(stereo.sha(stereo.ROOT / path) == digest for path, digest in previous["code_sha256"].items()),
        "v296 implementation or authored qualification differs",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("plan", "development", "holdout"))
    parser.add_argument("--asset-dir", type=Path)
    args = parser.parse_args()
    if (args.role == "plan") != (args.asset_dir is None):
        parser.error("Specify --asset-dir for exactly one local qualification role")
    stereo.holdout.model_free()
    verify_preflight()
    plan = publisher_plan()
    result = None
    if args.role != "plan":
        directory = args.asset_dir.resolve(strict=True)
        require(directory.is_dir(), "An existing directory of exact local assets is required")
        check = check_surrey if args.role == "development" else check_huddersfield
        result = check(directory)
    stereo.holdout.model_free()
    print(
        json.dumps(
            {
                "schema": "chipmunk.audio-room-assets/299",
                "role": args.role,
                "publisher_plan": plan,
                "result": result,
                "v296_authored_receipt_sha256": PREVIOUS_SHA,
                "code_sha256": {name: stereo.sha(stereo.ROOT / name) for name in CODE},
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
                "libsndfile_version": sf.__libsndfile_version__,
                "foreground_files_opened": 0,
                "model_calls": 0,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
