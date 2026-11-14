"""Source clock and measurement contract for a local loss of stereo differences."""

from __future__ import annotations

from app.evidence.audio_silence_binding import continuous_spans

KEY = "identical_channel_interval"
SCHEMA = "chipmunk.stereo-identical-interval/1"
FLANK = 1280
MINIMUM = 2560
MAXIMUM = 32000
MIN_DIFFERENCE = -75.0
MIN_SIGNAL = -70.0
MAX_EVENTS = 3
METHOD = {"minimum_identical_samples": MINIMUM, "maximum_identical_samples": MAXIMUM,
          "flank_samples": FLANK, "minimum_flank_difference_dbfs": MIN_DIFFERENCE,
          "minimum_signal_dbfs": MIN_SIGNAL, "maximum_events": MAX_EVENTS}


def spans(timeline, limit):
    return [(start, min(end, limit)) for start, end in continuous_spans(timeline)
            if min(end, limit) - start >= MINIMUM + 2 * FLANK]


def validate(record, timeline, limit, ranges):
    checked = spans(timeline, limit)
    coverage = sum(end - start for start, end in checked)
    selected = record["selected"]
    count = record["qualifying_intervals"]
    if (set(record) != {"schema", "method", "affects_overall_confidence", "checked_samples",
                        "qualifying_intervals", "selected"}
            or record["schema"] != SCHEMA or record["method"] != METHOD
            or record["affects_overall_confidence"] is not False
            or type(record["checked_samples"]) is not int or record["checked_samples"] != coverage
            or type(count) is not int or not 0 <= count <= coverage // MINIMUM
            or not isinstance(selected, list) or len(selected) != min(count, MAX_EVENTS)):
        raise ValueError("Identical-channel measurements have invalid coverage")
    previous_end = 0
    for row in selected:
        start, end = row["start_sample"], row["end_sample"]
        difference = row["flank_difference_dbfs"]
        level = row["minimum_signal_dbfs"]
        if (set(row) != {"start_sample", "end_sample", "flank_difference_dbfs", "minimum_signal_dbfs",
                         "source_timestamp_ranges"}
                or type(start) is not int or type(end) is not int or start < previous_end
                or not MINIMUM <= end - start <= MAXIMUM
                or not any(a <= start - FLANK and end + FLANK <= b for a, b in checked)
                or row["source_timestamp_ranges"] != ranges(start, end)
                or not isinstance(difference, list) or len(difference) != 2
                or any(type(value) not in (float, int) or not MIN_DIFFERENCE <= value <= 0 for value in difference)
                or type(level) not in (float, int) or not MIN_SIGNAL <= level <= 0):
            raise ValueError("An identical-channel interval differs from its measured scope or clock")
        previous_end = end
