"""Public structure and selection rules for stereo listening evidence."""
from __future__ import annotations

import re

from app.evidence.stereo_binding import SAMPLE_RATE
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


def validate(listening: dict, stereo: dict | None) -> None:
    try:
        selected = select_windows(stereo)
        rows = listening["windows"]
        if (set(listening) != {"schema", "affects_overall_confidence", "windows"}
                or listening["schema"] != SCHEMA or listening["affects_overall_confidence"] is not False
                or not selected or not isinstance(rows, list)
                or [row["stereo_window_index"] for row in rows] != selected):
            raise ValueError("Stereo listening windows differ from the measured channel comparison")
        for row in rows:
            assets = row["artifacts"]
            if (set(row) != {"stereo_window_index", "artifacts"} or type(row["stereo_window_index"]) is not int
                    or set(assets) != {*CHANNELS, "waveform"}):
                raise ValueError("Stereo listening assets are incomplete")
            for name, asset in assets.items():
                expected = ({"media_type": "image/png", "width": DIMENSIONS[0], "height": DIMENSIONS[1]} if name == "waveform"
                            else {"media_type": "audio/wav", "sample_rate": SAMPLE_RATE, "samples": SAMPLE_RATE, "channels": 1})
                if (set(asset) != {*expected, "sha256", "url"} or any(asset[key] != value for key, value in expected.items())
                        or not re.fullmatch(r"[a-f0-9]{64}", asset["sha256"])):
                    raise ValueError("Stereo listening asset dimensions or digest are invalid")
    except (KeyError, AttributeError, IndexError, TypeError) as error:
        raise ValueError("Stereo listening evidence has invalid fields") from error
