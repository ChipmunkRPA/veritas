"""Analysis profiles `quick | full | deep` (ARCHITECTURE §2.6–2.7, D-08).

Which detectors run in a profile is declared per detector in `models.yaml` (`profiles:`). This module holds the
profile-level parameters: timeout multipliers, the default-profile rule and the sampling parameters the video and
audio stages read.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.engine.detector import Profile

REDUCED = "reduced"  # effective profile when any detector of the requested profile was omitted (PL-03)
PROFILE_TIMEOUT_MULTIPLIER: dict[Profile, float] = {Profile.QUICK: 1.0, Profile.FULL: 1.5, Profile.DEEP: 3.0}
DEVICE_TIMEOUT_MULTIPLIER: dict[str, float] = {"cuda": 1.0, "mps": 1.0, "cpu": 4.0}


@dataclass(frozen=True)
class SamplingParams:
    """Video and audio sampling per profile (§2.7 table, VF §8.1)."""

    uniform_seek_frames: int | None  # cap on the uniform (legacy) frame set; quick samples no per-track frames or clips
    track_fps: tuple[float, float] | None  # sparse per-track scoring rate range
    clips_per_shot: int
    clip_frames: int
    speaking_window_s: float | None  # windows per speaking track (lip-sync)
    voice_hop_s: float | None
    audio_windows: int | None  # quick scores one chunk; None = the whole track
    second_pass_cap_s: float | None  # 0 = no second pass; None = bounded only by the analysis budget


SAMPLING: dict[Profile, SamplingParams] = {
    Profile.QUICK: SamplingParams(uniform_seek_frames=16, track_fps=None, clips_per_shot=0, clip_frames=0,
                                  speaking_window_s=None, voice_hop_s=None, audio_windows=1, second_pass_cap_s=0.0),
    Profile.FULL: SamplingParams(uniform_seek_frames=None, track_fps=(2.0, 4.0), clips_per_shot=3, clip_frames=16,
                                 speaking_window_s=1.0, voice_hop_s=2.0, audio_windows=None, second_pass_cap_s=20.0),
    Profile.DEEP: SamplingParams(uniform_seek_frames=None, track_fps=(2.0, 4.0), clips_per_shot=3, clip_frames=16,
                                 speaking_window_s=1.0, voice_hop_s=2.0, audio_windows=None, second_pass_cap_s=None),
}


def resolve_profile(requested: str | None, device: str) -> Profile:
    """`auto` (the VERITAS_DEFAULT_PROFILE default) is `full` on accelerator hosts and `quick` on CPU hosts."""
    if requested in (None, "", "auto"):
        return Profile.QUICK if _device_class(device) == "cpu" else Profile.FULL
    return Profile(requested)


def detector_timeout(timeout_s: float, profile: Profile | str, device: str, *, batches: int = 1,
                     remaining_s: float | None = None) -> float:
    """spec.timeout_s × profile multiplier × device multiplier × batches, capped by the remaining analysis budget."""
    t = timeout_s * PROFILE_TIMEOUT_MULTIPLIER[Profile(profile)] * DEVICE_TIMEOUT_MULTIPLIER.get(
        _device_class(device), DEVICE_TIMEOUT_MULTIPLIER["cpu"]) * max(1, batches)
    return t if remaining_s is None else min(t, max(0.0, remaining_s))


def effective_profile(requested: Profile | str, omitted: list[str]) -> str:
    return REDUCED if omitted else Profile(requested).value


def _device_class(device: str) -> str:
    return str(device).split(":", 1)[0]
