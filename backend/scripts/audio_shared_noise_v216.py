"""Construct level-matched three-second speech views with shared recorded noise."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from scripts.audio_short_edit_controls import (
    SAMPLE_RATE,
    SAMPLES,
    SNR_LEVELS_DB,
    UnsupportedControl,
    crop_pcm,
    mix_paired_noise,
)

ARMS = ("genuine", "alania", "drifting")
TAPES = ("kitchen", "traffic")
SAMPLE_RANGE = (0, SAMPLES)
MAX_SNR_ERROR_DB = 0.1
Condition = tuple[str, int]


@dataclass(frozen=True)
class ArmMix:
    source_rms: float
    level_match_gain: float
    clean_s16le: bytes
    mixed_s16le: dict[Condition, bytes]
    measured_snr_db: dict[Condition, float]


@dataclass(frozen=True)
class SharedNoise:
    common_gain: float
    reference_arm: str
    noise_s16le: dict[Condition, bytes]
    variants: dict[str, ArmMix]
    sample_range: tuple[int, int] = SAMPLE_RANGE
    sample_rate: int = SAMPLE_RATE


def samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2")


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def mix_shared_noise(foregrounds: Mapping[str, bytes], tapes: Mapping[str, bytes]) -> SharedNoise:
    """Apply one clock, level match, and common gain to all available arms.

    The caller supplies qualified mono 16 kHz signed16 little-endian PCM and
    retains planned rows for absent arms. Tape excerpts already cover exactly
    the fixed output extent. No original speech labels are derived here.
    """
    if set(foregrounds) - set(ARMS) or set(tapes) != set(TAPES):
        raise ValueError("Expected known foreground arms and both fixed tapes")
    if not foregrounds:
        raise UnsupportedControl("no_available_foregrounds")
    if any(not isinstance(tapes[name], bytes) or len(tapes[name]) != 2 * SAMPLES for name in TAPES):
        raise UnsupportedControl("expected_exact_three_second_tape")

    source = {name: samples(crop_pcm(foregrounds[name])) for name in ARMS if name in foregrounds}
    levels = {name: rms(values) for name, values in source.items()}
    reference = min(levels, key=levels.__getitem__)
    target = levels[reference]
    if not target:
        raise UnsupportedControl("silent_foreground")
    match = {name: target / level for name, level in levels.items()}
    matched = {name: np.rint(values * match[name]).astype("<i2").tobytes() for name, values in source.items()}

    preliminary = {
        name: mix_paired_noise(matched, reference=reference, noise_pcm_s16le=tapes[name]) for name in TAPES
    }
    common_gain = min(treatment.common_gain for treatment in preliminary.values())
    treatments = {
        name: treatment
        if treatment.common_gain == common_gain
        else mix_paired_noise(
            matched, reference=reference, noise_pcm_s16le=tapes[name], common_gain=common_gain
        )
        for name, treatment in preliminary.items()
    }
    residuals = {
        (tape, db): treatment.noise_s16le[db]
        for tape, treatment in treatments.items()
        for db in SNR_LEVELS_DB
    }
    variants = {}
    for name in source:
        clean = treatments[TAPES[0]].variants[name].clean_s16le
        clean_samples = samples(clean).astype(np.int32)
        clean_rms = rms(clean_samples)
        mixed, measured = {}, {}
        for tape, treatment in treatments.items():
            arm = treatment.variants[name]
            if clean != arm.clean_s16le:
                raise ValueError("Clean view differs across tapes")
            for db in SNR_LEVELS_DB:
                condition = (tape, db)
                payload = arm.mixed_s16le[db]
                observed = samples(payload).astype(np.int32) - clean_samples
                if not np.array_equal(observed, samples(residuals[condition])):
                    raise ValueError("Mixed PCM differs from the shared quantized residual")
                snr = 20 * math.log10(clean_rms / rms(observed))
                if abs(snr - db) > MAX_SNR_ERROR_DB:
                    raise UnsupportedControl("quantized_snr_out_of_tolerance")
                mixed[condition], measured[condition] = payload, snr
        variants[name] = ArmMix(levels[name], match[name], clean, mixed, measured)
    return SharedNoise(common_gain, reference, residuals, variants)
