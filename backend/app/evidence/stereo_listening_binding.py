"""Public structure and selection rules for stereo listening evidence."""
from __future__ import annotations

import re

from app.evidence.audio_silence_binding import FLANK
from app.evidence.stereo_binding import CHANNEL_SILENCE, SAMPLE_RATE
from app.evidence.stereo_binding import SCHEMA as STEREO_SCHEMA

SCHEMA = "chipmunk.stereo-listening/1"
DIMENSIONS = (960, 290)
CHANNELS = ("left", "right", "mono")
MAX_WINDOWS = 3


def select_windows(stereo: dict | None) -> list[int]:
    if not stereo or stereo.get("schema") != STEREO_SCHEMA:
        return []
    candidates = []
    for index, window in enumerate(stereo["windows"]):
        downmix = window["downmix"]
        strong = downmix["strong_attenuation"]
        if strong or window["status"] == "strong_imbalance":
            # Silent mono ranks ahead of every finite attenuation.
            attenuation = downmix["attenuation_db"]
            loss = 0
            if strong:
                loss = float("inf") if attenuation is None else attenuation
            candidates.append((-loss, -window["level_gap_db"], index))
    return sorted(item[2] for item in sorted(candidates)[:MAX_WINDOWS])


def channel_silence_intervals(stereo: dict | None) -> list[tuple[int, int]]:
    return [(row["start_sample"] - FLANK, row["end_sample"] + FLANK)
            for row in (stereo or {}).get(CHANNEL_SILENCE, {}).get("selected", [])]


def all_artifacts(listening: dict):
    for rows in (listening["windows"], listening.get(CHANNEL_SILENCE, [])):
        yield from (row["artifacts"] for row in rows)


def validate(listening: dict, stereo: dict | None) -> None:
    try:
        selected = select_windows(stereo)
        rows = listening["windows"]
        silence = channel_silence_intervals(stereo)
        keys = {"schema", "affects_overall_confidence", "windows"}
        if silence:
            keys.add(CHANNEL_SILENCE)
        if (set(listening) != keys
                or listening["schema"] != SCHEMA or listening["affects_overall_confidence"] is not False
                or (not selected and not silence) or not isinstance(rows, list)
                or [row["stereo_window_index"] for row in rows] != selected):
            raise ValueError("Stereo listening windows differ from the measured channel comparison")
        for row in rows:
            if (set(row) != {"stereo_window_index", "artifacts"} or type(row["stereo_window_index"]) is not int
                    or set(row["artifacts"]) != {*CHANNELS, "waveform"}):
                raise ValueError("Stereo listening assets are incomplete")
            _validate_assets(row["artifacts"], SAMPLE_RATE)
        excerpts = listening.get(CHANNEL_SILENCE, [])
        if not isinstance(excerpts, list) or len(excerpts) != len(silence):
            raise ValueError("Stereo listening clips do not cover the selected silence intervals")
        for row, (start, end) in zip(excerpts, silence, strict=True):
            if set(row) != {"artifacts"}:
                raise ValueError("Stereo channel silence assets are incomplete")
            _validate_assets(row["artifacts"], end - start)
    except (KeyError, AttributeError, IndexError, TypeError) as error:
        raise ValueError("Stereo listening evidence has invalid fields") from error


def _validate_assets(assets, samples):
    if set(assets) != {*CHANNELS, "waveform"}:
        raise ValueError("Stereo listening assets are incomplete")
    for name, asset in assets.items():
        expected = ({"media_type": "image/png", "width": DIMENSIONS[0], "height": DIMENSIONS[1]} if name == "waveform"
                    else {"media_type": "audio/wav", "sample_rate": SAMPLE_RATE, "samples": samples, "channels": 1})
        if (set(asset) != {*expected, "sha256", "url"} or any(asset[key] != value for key, value in expected.items())
                or not re.fullmatch(r"[a-f0-9]{64}", asset["sha256"])):
            raise ValueError("Stereo listening asset dimensions or digest are invalid")
