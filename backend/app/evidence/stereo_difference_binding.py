# SPDX-License-Identifier: Apache-2.0
"""Public source-clock contract for a brief reduction in the stereo difference."""

from app.evidence.audio_silence_binding import continuous_spans

KEY = "stereo_difference_reduction"
SCHEMA = "chipmunk.stereo-difference-reduction/1"
CODECS = {"pcm_s16le", "aac"}
FLANK = 3200
HOP = 320
FRAME = 640
WIDTHS = (9, 12, 15, 19, 24, 31, 39, 49, 60)
BANDS = [[180, 580], [580, 1460], [1460, 3200], [3200, 6800]]
MINIMUM_SAMPLES = FRAME + (WIDTHS[0] + 2 * (FLANK // HOP + 1) - 1) * HOP
OPERATING_POINT_DB = 10.0
METHOD = {
    "frame_samples": FRAME,
    "hop_samples": HOP,
    "duration_candidates_samples": [width * HOP for width in WIDTHS],
    "comparison_flank_samples": FLANK,
    "frequency_bands_hz": BANDS,
    "minimum_two_band_margin_db_exclusive": OPERATING_POINT_DB,
}


def spans(timeline, limit):
    return [(start, min(end, limit)) for start, end in continuous_spans(timeline)
            if min(end, limit) - start >= MINIMUM_SAMPLES]


def _levels(values, count, lower, upper):
    return isinstance(values, list) and len(values) == count and all(
        type(value) in (float, int) and lower <= value <= upper for value in values
    )


def validate(record, timeline, limit, ranges):
    checked = spans(timeline, limit)
    coverage = sum(end - start for start, end in checked)
    selected = record["selected"]
    if (set(record) != {"schema", "method", "affects_overall_confidence", "checked_samples", "status", "selected"}
            or record["schema"] != SCHEMA or record["method"] != METHOD
            or record["affects_overall_confidence"] is not False
            or type(record["checked_samples"]) is not int or record["checked_samples"] != coverage
            or record["status"] not in ("unassessed", "no_qualifying_change", "measured_change")
            or not isinstance(selected, list) or len(selected) != int(record["status"] == "measured_change")
            or record["status"] != "unassessed" and not checked):
        raise ValueError("Stereo difference reduction has invalid coverage")
    for row in selected:
        start, end = row["start_sample"], row["end_sample"]
        margin, bands = row["score"], row["band_drop_db"]
        if (set(row) != {"start_sample", "end_sample", "score", "band_drop_db", "side_dbfs", "common_dbfs",
                         "source_timestamp_ranges"}
                or type(start) is not int or type(end) is not int
                or end - start not in METHOD["duration_candidates_samples"]
                or not any(a <= start - FLANK and end + FLANK <= b and (start - a) % HOP == HOP // 2
                           for a, b in checked)
                or row["source_timestamp_ranges"] != ranges(start, end)
                or type(margin) not in (float, int) or not OPERATING_POINT_DB < margin <= 140.
                or not _levels(bands, len(BANDS), -140., 140.)
                or sum(value >= margin - .001 for value in bands) < 2
                or not _levels(row["side_dbfs"], 3, -140., 0.)
                or not _levels(row["common_dbfs"], 3, -140., 0.)
                or row["common_dbfs"][1] < -49.):
            raise ValueError("Stereo difference reduction differs from its measured scope or clock")
