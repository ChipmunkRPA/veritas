"""Check the existing picture-change rule on qualified native carriers."""

from dataclasses import replace

from app.detectors.video import scene_return
from app.evidence import scene_return_binding as binding


def profile(table, encoding, carrier):
    if carrier == "mp4" and encoding in binding.ENCODINGS and scene_return.supported(table):
        return binding.SCHEMA
    if (carrier == "webm" and encoding == binding.WEBM_ENCODING
            and (table.width, table.height, table.coded_width, table.coded_height) == (1280, 720, 1280, 720)
            and not (table.rotation or table.truncated or table.interpolated) and table.pts
            and binding.valid_webm_clock(table.time_base, table.pts, table.pts[0])):
        return binding.WEBM_SCHEMA
    return None


def detect(frames, table, schema):
    if schema == binding.SCHEMA:
        return scene_return.detect(frames, table)
    if schema != binding.WEBM_SCHEMA:
        raise ValueError("Unknown picture-change source profile")
    # The existing detector uses picture order. A synthetic internal clock
    # must never replace the verified native WebM ticks in its evidence.
    normalized = replace(table, time_base="1/30", pts=tuple(range(len(table))))

    def at_verified_ticks():
        for frame in frames:
            if not (0 <= frame.index < len(table) and frame.pts_verified and frame.pts == table.pts[frame.index]):
                return
            yield replace(frame, pts=normalized.pts[frame.index])

    return scene_return.detect(at_verified_ticks(), normalized)
