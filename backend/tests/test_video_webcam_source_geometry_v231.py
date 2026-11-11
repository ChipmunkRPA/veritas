from dataclasses import dataclass

import pytest

from scripts.video_webcam_source_geometry_v231 import verified_geometry


@dataclass
class Table:
    pts: tuple = (0, 33, 67)
    time_base: str = "1/1000"
    width: int = 1280
    height: int = 720
    rotation: int = 0
    interpolated: bool = False
    truncated: bool = False

    def __len__(self):
        return len(self.pts)


def test_sequence_validator_receives_bound_width_and_height():
    table = Table()
    source = {"pts": list(table.pts), "time_base": table.time_base, "size": (1280, 720)}
    assert verified_geometry(table, source) == {
        "time_base": "1/1000",
        "frame_count": 3,
        "width": 1280,
        "height": 720,
    }
    with pytest.raises(ValueError, match="geometry"):
        verified_geometry(table, {**source, "size": (720, 1280)})
