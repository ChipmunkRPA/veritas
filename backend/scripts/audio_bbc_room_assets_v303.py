"""Qualify three local BBC R&D stereo room responses without rendering audio."""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts import audio_room_assets_v299 as prior

room, stereo, require = prior.room, prior.stereo, prior.require
REVISION = "c88a293ed81f4aa820bcbc4138f951552819e001"
REPOSITORY = "bbc/bbcrd-brirs"
RIG = "bbc-rd-salford-ku100-frontal"
RATE, SAMPLES, SIZE = 48_000, 16_384, 131_160
DEVELOPMENT = stereo.PANEL / "audio-room-assets-v299-development.json"
DEVELOPMENT_SHA = "e25e34aa57f80e38a552067d57abf7cff2b1a066de7da3e65aea456115942008"
CODE = (
    "backend/scripts/audio_bbc_room_assets_v303.py",
    "backend/scripts/audio_room_assets_v299.py",
    "backend/tests/test_audio_bbc_room_assets_v303.py",
    "docs/audio-bbc-room-assets-v303-method.md",
)
POSITIONS = {
    "front": ("M+000", 0, 1.99, "0fc8a92eb335e372d91eeb28f00e23dd5eac5d43"),
    "left": ("M+090", 90, 2.28, "a5307acc7f13da3c24a85487262e59adcb58995b"),
    "right": ("M-090", -90, 2.28, "650f383bfac67c7bdf93c5454c4aaf1d005dbb2b"),
}
ASSETS = {
    position: prior.Asset(
        f"{prior.RAW}{REPOSITORY}/{REVISION}/ambix-configs/irs/BBCRD_LR_{label}.wav", SIZE, blob
    )
    for position, (label, _, _, blob) in POSITIONS.items()
}
DOCUMENTS = {
    "licence_correction": f"https://github.com/{REPOSITORY}/commit/{REVISION}",
    "licence": {"path": "LICENSE", "git_blob_sha1": "3b7b82d0da2db857eda1a798dbd908ea136f07b5"},
    "static_wav_description": {
        "path": "README.md",
        "git_blob_sha1": "25bf32dc960acd5d7665725307faf96f479b283d",
    },
    "geometry": "https://downloads.bbc.co.uk/rd/pubs/whp/whp-pdf-files/WHP382.pdf",
    "geometry_scene": {
        "path": "ssr-scenes/BBCRDLR_BRS_SystemH.asd",
        "git_blob_sha1": "f45bbd350b71953d0e69378eda31440a3a611356",
    },
    "static_wav_routing": {
        "path": "ambix-configs/BBCRD_LR_FULL.config",
        "git_blob_sha1": "e1b2ac439d9b1b121cd89110ebb31a42f8779918",
    },
}


def plan():
    return {
        "room": "BBC Research & Development listening room, Salford",
        "source_repository": REPOSITORY,
        "source_commit": REVISION,
        "license": "CC-BY-SA-4.0",
        "license_scope": "Pinned static WAV files in the BBC repository",
        "receiver": "Neumann KU100, centred, frontal orientation",
        "ear_order": ["left", "right"],
        "sample_rate_hz": RATE,
        "samples_per_channel": SAMPLES,
        "native_range": [0, SAMPLES],
        "delay_samples_added": [0, 0],
        "publisher_sources": DOCUMENTS,
        "positions": {
            position: {
                "speaker": label,
                "azimuth_degrees_left_positive": angle,
                "elevation_degrees": 0,
                "physical_distance_m": distance,
                "asset": asdict(ASSETS[position]),
            }
            for position, (label, angle, distance, _) in POSITIONS.items()
        },
    }


def parse_impulse(raw, position):
    require(position in POSITIONS, "Only the three documented BBC positions are eligible")
    with sf.SoundFile(io.BytesIO(raw)) as wav:
        require(
            wav.format in ("WAV", "WAVEX")
            and wav.samplerate == RATE
            and wav.channels == 2
            and wav.frames == SAMPLES
            and wav.subtype in ("FLOAT", "PCM_32"),
            "Exact native BBC stereo 48 kHz, 16384-sample, 32-bit WAV required",
        )
        encoding = wav.subtype
        values = wav.read(SAMPLES, dtype="float64", always_2d=True)
    require(
        values.shape == (SAMPLES, 2) and not np.array_equal(values[:, 0], values[:, 1]),
        "BBC response must contain two complete, distinct ear channels",
    )
    impulse = room.Impulse(RIG, position, RATE, room.IMPULSE_UNIT, (0, 0), values)
    return impulse, encoding


def qualify(directory):
    positions = {}
    for position, asset in ASSETS.items():
        raw, identity = prior.pinned_bytes(directory, asset)
        impulse, encoding = parse_impulse(raw, position)
        record = impulse.identity()
        # Diagnostic on the preserved sample clock. A peak need not be the first arrival.
        left, right = np.argmax(np.abs(impulse.coefficients), axis=0).tolist()
        positions[position] = {
            "asset": identity,
            "impulse": record,
            "encoding": encoding,
            "peak_sample_left_right": [left, right],
        }
    require(
        len({row["impulse"]["coefficients_sha256"] for row in positions.values()}) == 3,
        "BBC front and lateral impulses must be distinct",
    )
    return {"status": "documented_positions_qualified", "positions": positions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-dir", type=Path, help="Read three exact local WAVs. Omit to show publisher pins."
    )
    args = parser.parse_args()
    stereo.holdout.model_free()
    prior.verify_preflight()
    require(stereo.sha(DEVELOPMENT) == DEVELOPMENT_SHA, "Exact v299 Surrey development receipt required")
    result = None
    if args.asset_dir is not None:
        directory = args.asset_dir.resolve(strict=True)
        require(directory.is_dir(), "An existing directory of exact local BBC assets is required")
        result = qualify(directory)
    stereo.holdout.model_free()
    print(
        json.dumps(
            {
                "schema": "chipmunk.audio-bbc-room-assets/303",
                "role": "holdout",
                "publisher_plan": plan(),
                "result": result,
                "v299_development_receipt_sha256": DEVELOPMENT_SHA,
                "v296_authored_receipt_sha256": prior.PREVIOUS_SHA,
                "code_sha256": {name: stereo.sha(stereo.ROOT / name) for name in CODE},
                "numpy_version": np.__version__,
                "libsndfile_version": sf.__libsndfile_version__,
                "speech_files_opened": 0,
                "model_calls": 0,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
