"""Verify the packet clock of a short AAC track with an explicit gapless table."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from app.detectors.video.decode import _input
from app.evidence.audio_short_review_binding import AAC_CONTAINER, AAC_FRAME

RATE = 48000
VALID_SAMPLES = 6 * RATE
PRIMING = 2112
PACKETS = (PRIMING + VALID_SAMPLES + AAC_FRAME - 1) // AAC_FRAME
_TABLE = re.compile(r"\s*00000000 ([0-9a-fA-F]{8}) ([0-9a-fA-F]{8}) ([0-9a-fA-F]{16})(?: 00000000){8}\s*")


def _clock(probe: dict) -> dict | None:
    stream, = probe["streams"]
    media = probe["format"]
    tags = media.get("tags", {})
    if (media["format_name"], tags.get("major_brand"), stream["codec_name"], stream["profile"],
        stream["channels"], stream["channel_layout"], stream["sample_rate"], stream["time_base"]) != (
        AAC_CONTAINER, "M4A ", "aac", "LC", 1, "mono", str(RATE), f"1/{RATE}",
    ):
        return None
    table = _TABLE.fullmatch(tags.get("iTunSMPB", ""))
    if not table:
        return None
    priming, remainder, valid = (int(value, 16) for value in table.groups())
    coded = priming + valid + remainder
    bitrate = int(stream["bit_rate"])
    packets = probe["packets"]
    if (priming != PRIMING or valid != VALID_SAMPLES or not 0 <= remainder < AAC_FRAME
            or stream["start_pts"] != priming or coded != stream["duration_ts"]
            or coded != AAC_FRAME * PACKETS or int(stream["nb_frames"]) != PACKETS or len(packets) != PACKETS
            or not 0 < bitrate <= 512_000):
        return None
    first_skip = [{"side_data_type": "Skip Samples", "skip_samples": priming, "discard_padding": 0}]
    for index, packet in enumerate(packets):
        if (packet["pts"] != index * AAC_FRAME or packet["dts"] != packet["pts"] or packet["duration"] != AAC_FRAME
                or packet.get("side_data_list", []) != (first_skip if index == 0 else [])):
            return None
    return {"sample_rate": RATE, "time_base": f"1/{RATE}", "channels": 1, "method": "explicit_gapless_table",
            "presentation_start_pts": priming, "presentation_end_exclusive_pts": priming + valid,
            "coded_samples": coded, "packets": PACKETS, "priming_samples": priming,
            "remainder_samples": remainder, "bit_rate": bitrate}


def read(path: Path) -> dict | None:
    """Require the stated padding to agree with the exact packet timestamps and decoder skip."""
    try:
        process = subprocess.run(
            ["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "a",
             "-read_intervals", f"%+#{PACKETS + 1}", "-show_streams", "-show_packets", "-show_entries",
             "stream=codec_name,profile,time_base,sample_rate,channels,channel_layout,duration_ts,start_pts,nb_frames,bit_rate:"
             "format=format_name:format_tags=iTunSMPB,major_brand:packet=pts,dts,duration:"
             "packet_side_data=side_data_type,skip_samples,discard_padding", "-of", "json", _input(path)],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=15, check=True,
        )
        return _clock(json.loads(process.stdout))
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, OverflowError):
        return None
