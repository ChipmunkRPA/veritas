"""Pure sample-clock construction for three-second speech edit controls.

Inputs are already qualified mono 16 kHz little-endian signed16 PCM. Source
clocks, identities, labels and split membership belong to the caller.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
SAMPLES = 3 * SAMPLE_RATE
TARGET_SAMPLES = (17280, 30080)
DONOR_SAMPLES = (12160, 24960)
FADE_SAMPLES = {0: 0, 20: 320, 50: 800}
SNR_LEVELS_DB = (0, 5)
PEAK_TARGET = 0.98 * 32768


class UnsupportedControl(ValueError):
    """A requested control has no usable sample support under its recipe."""


@dataclass(frozen=True)
class Edit:
    pcm_s16le: bytes
    changed_samples: int
    fade_samples: int
    target_samples: tuple[int, int] = TARGET_SAMPLES
    donor_samples: tuple[int, int] = DONOR_SAMPLES

    @property
    def sample_count(self) -> int:
        return len(self.pcm_s16le) // 2

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / SAMPLE_RATE


@dataclass(frozen=True)
class NoiseVariant:
    clean_s16le: bytes
    mixed_s16le: dict[int, bytes]
    measured_snr_db: dict[int, float]


@dataclass(frozen=True)
class PairedNoise:
    common_gain: float
    noise_s16le: dict[int, bytes]
    variants: dict[str, NoiseVariant]


def _check_pcm(pcm_s16le: bytes) -> None:
    if not isinstance(pcm_s16le, bytes) or len(pcm_s16le) % 2:
        raise ValueError("Expected whole little-endian signed16 PCM bytes")


def _samples(pcm_s16le: bytes) -> np.ndarray:
    _check_pcm(pcm_s16le)
    if len(pcm_s16le) != 2 * SAMPLES:
        raise UnsupportedControl("expected_three_seconds")
    return np.frombuffer(pcm_s16le, dtype="<i2")


def _quantize(values: np.ndarray) -> np.ndarray:
    rounded = np.rint(values)
    if rounded.min() < -32768 or rounded.max() > 32767:
        raise UnsupportedControl("pcm16_clipping")
    return rounded.astype("<i2")


def crop_pcm(pcm_s16le: bytes, start_sample: int = 0) -> bytes:
    """Return exactly 48,000 native samples from a caller-qualified extent."""
    _check_pcm(pcm_s16le)
    if type(start_sample) is not int or start_sample < 0:
        raise ValueError("Expected a nonnegative integer sample start")
    end_sample = start_sample + SAMPLES
    if 2 * end_sample > len(pcm_s16le):
        raise UnsupportedControl("insufficient_source_samples")
    return pcm_s16le[2 * start_sample : 2 * end_sample]


def splice_pcm(host: bytes, donor: bytes | None = None, *, fade_ms: int = 0) -> Edit:
    """Copy the fixed donor span into the fixed target with contained fades.

    Passing no donor uses the original host as the donor. Only the target
    interval can change. A raised-cosine fade runs from pure host to pure
    donor at the start and in reverse at the end, entirely inside the target.
    """
    if type(fade_ms) is not int or fade_ms not in FADE_SAMPLES:
        raise ValueError("Expected a hard edit or a 20 ms or 50 ms fade")
    original = _samples(host)
    donor_values = original if donor is None else _samples(donor)
    start, end = TARGET_SAMPLES
    source_start, source_end = DONOR_SAMPLES
    carrier = original[start:end]
    replacement = donor_values[source_start:source_end].copy()
    fade = FADE_SAMPLES[fade_ms]
    if fade:
        ramp = (1 - np.cos(np.linspace(0, np.pi, fade, dtype=np.float64))) / 2
        for interval, alpha in ((slice(0, fade), ramp), (slice(-fade, None), ramp[::-1])):
            before = carrier[interval].astype(np.float64)
            replacement[interval] = _quantize(before + alpha * (replacement[interval] - before))
    changed = int(np.count_nonzero(carrier != replacement))
    if not changed:
        raise UnsupportedControl("zero_changed_samples")
    output = host[: 2 * start] + replacement.tobytes() + host[2 * end :]
    return Edit(output, changed, fade)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def mix_paired_noise(
    variants: Mapping[str, bytes],
    *,
    reference: str,
    noise_pcm_s16le: bytes,
    common_gain: float | None = None,
) -> PairedNoise:
    """Apply the same 0/+5 dB treatment to all arms of a comparison.

    The named reference sets the noise level over the full three seconds.
    A shared gain covers every clean arm and both levels. If omitted, one
    attenuation targets a 0.98 peak across the complete batch. Clean and
    noise are quantized separately so each noisy arm has exactly the same
    signed sample residual at a given level. SNR is measured after that
    quantization and can differ by arm when the edit changes foreground RMS.
    """
    if reference not in variants:
        raise ValueError("Expected the named clean reference among the paired variants")
    sources = {name: _samples(pcm).astype(np.float64) for name, pcm in variants.items()}
    noise = _samples(noise_pcm_s16le).astype(np.float64)
    if not np.ptp(noise):
        raise UnsupportedControl("constant_noise")
    reference_rms = _rms(sources[reference])
    if not reference_rms:
        raise UnsupportedControl("silent_reference")
    noise_rms = _rms(noise)
    scaled = {level: noise * (reference_rms / noise_rms / 10 ** (level / 20)) for level in SNR_LEVELS_DB}

    if common_gain is None:
        peak = max(float(np.max(np.abs(background))) for background in scaled.values())
        for source in sources.values():
            peak = max(peak, float(np.max(np.abs(source))))
            for background in scaled.values():
                peak = max(peak, float(np.max(np.abs(source + background))))
        common_gain = min(1.0, PEAK_TARGET / peak)
    elif not math.isfinite(common_gain) or not 0 < common_gain <= 1:
        raise ValueError("Expected a finite shared gain in (0, 1]")

    backgrounds = {level: _quantize(values * common_gain) for level, values in scaled.items()}
    background_rms = {level: _rms(values) for level, values in backgrounds.items()}
    if not all(background_rms.values()):
        raise UnsupportedControl("quantized_silent_noise")

    treatments = {}
    for name, source in sources.items():
        clean = _quantize(source * common_gain)
        clean_rms = _rms(clean)
        if not clean_rms:
            raise UnsupportedControl("quantized_silent_foreground")
        mixed = {
            level: _quantize(clean.astype(np.int32) + background).tobytes()
            for level, background in backgrounds.items()
        }
        measured = {level: 20 * math.log10(clean_rms / rms) for level, rms in background_rms.items()}
        treatments[name] = NoiseVariant(clean.tobytes(), mixed, measured)

    return PairedNoise(
        common_gain, {level: values.tobytes() for level, values in backgrounds.items()}, treatments
    )
