"""Verify stereo room rendering and production downmix with authored inputs."""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import fftconvolve, resample_poly

from scripts import audio_native_stereo_pairs_v277 as pairs
from scripts import audio_native_stereo_preflight_v274 as stereo
from scripts.audio_shared_noise_v216 import rms

RATE, SAMPLES = stereo.RATE, stereo.SAMPLES
IMPULSE_UNIT = "linear_amplitude_per_unit_input"
JOINT_SNR_DB = -10
NATIVE_RATES = (RATE, 44_100, 48_000, 96_000)
CONDITIONS = ("matched_room_clean", "noise_left", "noise_right")
ROOM_PLAN = {
    "development": {
        "room": "University of Surrey IoSR Listening Room",
        "doi": "10.15126/surreydata.00813511",
        "member": "IoSR_ListeningRoom_BRIRs_MATLAB/angle_0.0.mat",
    },
    "holdout": {
        "room": "St Paul's Hall, University of Huddersfield",
        "doi": "10.5281/zenodo.2639677",
        "member": "One documented two-channel stage-audience microphone pair",
    },
    "license": "CC-BY-4.0",
    "positions": {
        "foreground": "nearest horizontal 0 degrees",
        "noise_left": "nearest horizontal 90 degrees left",
        "noise_right": "nearest horizontal 90 degrees right",
    },
    "joint_noise_snr_db": JOINT_SNR_DB,
}
CODE = (
    "backend/app/detectors/audio/decode.py",
    "backend/app/detectors/video/decode.py",
    "backend/scripts/audio_native_stereo_preflight_v274.py",
    "backend/scripts/audio_native_stereo_pairs_v277.py",
    "backend/scripts/audio_room_transfer_preflight_v296.py",
    "backend/scripts/audio_shared_noise_v216.py",
    "backend/tests/test_audio_room_transfer_preflight_v296.py",
    "docs/audio-room-transfer-preflight-v296-method.md",
)


@dataclass(frozen=True)
class Impulse:
    rig: str
    position: str
    rate: int
    unit: str
    delay: tuple[int, int]
    coefficients: np.ndarray

    def identity(self):
        values = self.coefficients
        stereo.require(
            self.rig and self.position and self.rate in NATIVE_RATES and self.unit == IMPULSE_UNIT,
            "An identified rig, native clock and linear impulse unit are required",
        )
        stereo.require(
            len(self.delay) == 2
            and all(type(value) is int and 0 <= value < self.rate for value in self.delay),
            "Both native channel delays must be explicit, nonnegative integer samples",
        )
        stereo.require(
            values.ndim == 2
            and 0 < values.shape[0] <= 4 * self.rate
            and values.shape[1] == 2
            and values.dtype.kind == "f"
            and values.dtype.itemsize <= 8
            and np.isfinite(values).all()
            and np.any(values, axis=0).all(),
            "A bounded nonzero finite linear impulse pair is required",
        )
        return {
            "rig": self.rig,
            "position": self.position,
            "sample_rate_hz": self.rate,
            "unit": self.unit,
            "samples_per_channel": len(values),
            "delay_samples": list(self.delay),
            "coefficients_sha256": stereo.digest(values.astype("<f8").tobytes()),
        }


def resample(values, source, target):
    if source == target:
        return values
    common = math.gcd(source, target)
    return resample_poly(values, target // common, source // common, axis=0, padtype="constant")


def render(pcm, impulse):
    record = impulse.identity()
    stereo.require(
        type(pcm) is bytes and len(pcm) == 2 * SAMPLES, "Exact three-second mono 16 kHz PCM is required"
    )
    source = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    native = resample(source, RATE, impulse.rate)
    # Preserve measured L/R arrival offsets. Both outputs start at input clock zero.
    output = np.zeros((len(native), 2), dtype=np.float64)
    for channel, delay in enumerate(impulse.delay):
        filtered = fftconvolve(native, impulse.coefficients[:, channel])
        output[delay:, channel] = filtered[: len(native) - delay]
    rendered = resample(output, impulse.rate, RATE)
    stereo.require(
        rendered.shape == (SAMPLES, 2) and np.isfinite(rendered).all(), "Rendered stereo sample clock differs"
    )
    return rendered, {"source_pcm_sha256": stereo.digest(pcm), "impulse": record}


def compose(foreground_pcm, background_pcm, foreground, noise_left, noise_right):
    impulses = (foreground, noise_left, noise_right)
    stereo.require(
        len({item.position for item in impulses}) == 3
        and len({(item.rig, item.rate) for item in impulses}) == 1,
        "Use distinct source positions recorded on the same native receiver pair",
    )
    stereo.require(foreground_pcm != background_pcm, "Foreground and background source content must differ")
    speech, speech_record = render(foreground_pcm, foreground)
    stereo.require(
        all(rms(channel) > 0 for channel in speech.T), "Both rendered foreground channels must be nonzero"
    )
    residuals = {CONDITIONS[0]: (np.zeros_like(speech), None, 0.0)}
    for condition, impulse in zip(CONDITIONS[1:], impulses[1:], strict=True):
        noise, record = render(background_pcm, impulse)
        stereo.require(
            all(rms(channel) > 0 for channel in noise.T), "Both rendered background channels must be nonzero"
        )
        gain = rms(speech) / rms(noise) * 10 ** (-JOINT_SNR_DB / 20)
        residuals[condition] = (noise * gain, record, gain)

    peak = max(
        float(np.max(np.sum(np.abs(speech) + np.abs(noise), axis=1))) for noise, _, _ in residuals.values()
    )
    stereo.require(math.isfinite(peak) and peak > 0, "Finite nonzero room mix amplitude is required")
    common = min(1.0, (pairs.PEAK_LIMIT - 2) / peak)
    clean = np.rint(speech * common).astype(np.int32)
    cases = {}
    for condition, (noise, noise_record, gain) in residuals.items():
        residual = np.rint(noise * common).astype(np.int32)
        mixed = clean + residual
        stereo.require(
            np.max(np.sum(np.abs(mixed), axis=1)) <= pairs.PEAK_LIMIT, "Conservative stereo headroom exceeded"
        )
        levels = None
        if noise_record is not None:
            levels = {
                "joint": pairs.snr(clean, residual),
                "channels": [pairs.snr(clean[:, i], residual[:, i]) for i in (0, 1)],
            }
            stereo.require(
                abs(levels["joint"] - JOINT_SNR_DB) <= pairs.SNR_ERROR_DB,
                "Joint SNR differs after quantization",
            )
        payload = mixed.astype("<i2")
        stereo.require(
            np.array_equal(payload.astype(np.int32) - clean, residual), "Signed PCM residual differs"
        )
        cases[condition] = (
            payload,
            {
                "foreground": speech_record,
                "background": noise_record,
                "foreground_source_gain": 1.0,
                "background_source_gain": gain,
                "common_headroom_gain": common,
                "foreground_stereo_pcm_sha256": stereo.digest(clean.astype("<i2").tobytes()),
                "background_stereo_pcm_sha256": stereo.digest(residual.astype("<i2").tobytes()),
                "snr_db": levels,
            },
        )
    return cases


def fixtures():
    speech, noise = np.zeros((2, SAMPLES), dtype="<i2")
    speech[[31, 119, 24001, SAMPLES - 1]] = [1024, -2048, 4096, 512]
    noise[[97, 621, 24011, SAMPLES - 2]] = [1024, 2048, -1024, 512]
    front = Impulse("authored", "front", RATE, IMPULSE_UNIT, (0, 2), np.array([[0.5, 0.25], [0.0, 0.25]]))
    left = Impulse("authored", "left", RATE, IMPULSE_UNIT, (1, 4), np.array([[0.75, 0.125], [-0.125, 0.0]]))
    right = Impulse("authored", "right", RATE, IMPULSE_UNIT, (4, 1), left.coefficients[:, ::-1])
    return speech.tobytes(), noise.tobytes(), front, left, right


def authored_checks(directory):
    speech, noise, front, left, right = fixtures()
    cases = compose(speech, noise, front, left, right)
    rows = {
        name: {**record, **pairs.save_case(directory, name, pcm)} for name, (pcm, record) in cases.items()
    }
    front_pcm = cases[CONDITIONS[0]][0].astype(np.int32)
    left_pcm, right_pcm = (cases[name][0].astype(np.int32) - front_pcm for name in CONDITIONS[1:])
    stereo.require(
        np.array_equal(np.flatnonzero(front_pcm[:, 0])[:2], [31, 119])
        and np.array_equal(np.flatnonzero(front_pcm[:, 1])[:2], [33, 34])
        and np.array_equal(np.flatnonzero(left_pcm[:, 0])[:2], [98, 99])
        and np.array_equal(np.flatnonzero(left_pcm[:, 1])[:2], [101, 625])
        and np.array_equal(left_pcm, right_pcm[:, ::-1]),
        "Authored stereo delays or reflection geometry differ",
    )
    stereo.require(
        rows["noise_left"]["mixed_pcm_sha256"] == rows["noise_right"]["mixed_pcm_sha256"]
        and rows["noise_left"]["stereo_pcm_sha256"] != rows["noise_right"]["stereo_pcm_sha256"],
        "Mirrored authored background must preserve mono PCM while changing stereo PCM",
    )
    return rows


def main():
    stereo.holdout.model_free()
    with tempfile.TemporaryDirectory(prefix="chipmunk-audio-room-transfer-v296-") as directory:
        authored = authored_checks(Path(directory))
    stereo.holdout.model_free()
    print(
        json.dumps(
            {
                "schema": "chipmunk.audio-room-transfer-preflight/296",
                "status": "complete",
                "sample_rate_hz": RATE,
                "output_sample_range": [0, SAMPLES],
                "room_delay_units": "whole samples on the native measured clock, including zero if embedded",
                "future_room_plan": ROOM_PLAN,
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
                "code_sha256": {name: stereo.sha(stereo.ROOT / name) for name in CODE},
                "codec_binaries_sha256": {name: stereo.tool_hash(name) for name in ("ffmpeg", "ffprobe")},
                "authored_conditions": authored,
                "measured_responses_opened": 0,
                "external_audio_opened": 0,
                "model_calls": 0,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
