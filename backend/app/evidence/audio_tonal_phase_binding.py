"""Portable validation for narrowband phase observations in short native audio."""

from __future__ import annotations

import math
from itertools import combinations

KEY = "tonal_phase"
SCHEMA = "chipmunk.acoustic-context/6"
SUFFIX = "+tonal-phase"
TITLE = "Temporary change in tone phase"
FAMILIES = (50, 60)
PARTIALS = (1, 2, 3, 4)
POINT = .60
PHASE_FLOOR = .70
MIN_TONE_DBFS = -58.
MIN_ABOVE_NEIGHBORS_DB = 8.
MAX_INSIDE_DIFFERENCE = .60
MAX_OUTSIDE_OFFSET = 1.
MAX_HARMONIC_RESIDUAL = 1.10
STEP, WIDTHS, SEPARATION = 640, range(13, 30), 3840
METHOD = {
    "tone_families_hz": list(FAMILIES), "harmonics": list(PARTIALS), "analysis_rate": 1000,
    "projection_ms": 320, "step_samples": STEP, "support_offset_samples": SEPARATION,
    "interval_duration_steps": [WIDTHS.start, WIDTHS.stop - 1], "minimum_tone_dbfs": MIN_TONE_DBFS,
    "minimum_tone_above_neighbors_db": MIN_ABOVE_NEIGHBORS_DB, "point_rad": POINT,
}
LIMITATION = ("The phase comparison uses two related narrowband tones in the mixed recording. Ordinary changes in "
              "recording speed or background sound can cause this pattern. It carries zero AI-confidence weight.")


def native_input(samples, source):
    duration = source.get("source_duration_s")
    return (48000 <= samples <= 192000 and isinstance(duration, (float, int))
            and abs(duration * 16000 - samples) <= 1
            and (source.get("source_container"), source.get("source_codec")) in {("wav", "pcm_s16le"), ("flac", "flac")}
            and source.get("source_bit_depth") == 16 and source.get("source_channels") == 1
            and source.get("source_sample_rate") == 16000 and source.get("timeline_time_base") == "1/16000"
            and len(source.get("audio_stream_indices", [0])) == 1
            and source.get("timeline") == [{"start_sample": 0, "end_sample": samples, "source_pts": 0}])


def maximum_comparisons(samples):
    frames = (((samples + 15) // 16) - 320) // 40 - 1
    return len(FAMILIES) * math.comb(len(PARTIALS), 2) * sum(max(0, frames - width - 12) for width in WIDTHS)


def _wrapped(value):
    return math.remainder(value, math.tau)


def validate(detail, samples):
    count, status = detail["comparisons"], detail["status"]
    if (type(count) is not int or not 0 <= count <= maximum_comparisons(samples)
            or status not in {"unassessed", "no_supported_change", "measured_change"}
            or (count == 0) != (status == "unassessed")
            or set(detail) != ({"status", "comparisons", "measurement"} if status == "measured_change" else {"status", "comparisons"})):
        raise ValueError("Invalid tone phase coverage")
    if status != "measured_change":
        return None
    measurement = detail["measurement"]
    tones = measurement["tones"]
    first, second = tones
    start, end = first["samples"]
    centers = [start - SEPARATION, start + SEPARATION, end - SEPARATION, end + SEPARATION]
    permitted = {tuple(family * harmonic for harmonic in pair)
                 for family in FAMILIES for pair in combinations(PARTIALS, 2)}
    frequencies = (first["frequency_hz"], second["frequency_hz"])
    if (frequencies not in permitted or type(start) is not int or type(end) is not int
            or start % STEP or end % STEP or (end - start) // STEP not in WIDTHS
            or centers[0] < 3200 or centers[-1] + 3200 - 16 >= samples
            or set(measurement) != {"score", "tones"}):
        raise ValueError("Invalid tone phase locations")
    scores = []
    for tone in tones:
        a, b = tone["inside_offsets_rad"]
        outside = tone["outside_offset_rad"]
        change, variation = min(abs(a), abs(b)), abs(_wrapped(a - b))
        numeric = (a, b, outside, tone["tone_above_neighbors_db"], tone["tone_level_dbfs"])
        if (not all(type(value) in (int, float) and math.isfinite(value) for value in numeric)
                or tone["samples"] != [start, end] or tone["support_centers"] != centers
                or max(abs(a), abs(b)) > math.pi + 1e-6
                or change < PHASE_FLOOR - 1e-6 or variation > MAX_INSIDE_DIFFERENCE + 2e-6
                or abs(outside) > MAX_OUTSIDE_OFFSET + 1e-6
                or tone["tone_level_dbfs"] < MIN_TONE_DBFS
                or tone["tone_above_neighbors_db"] < MIN_ABOVE_NEIGHBORS_DB):
            raise ValueError("Invalid tone phase readings")
        scores.append(change - .45 * (variation + abs(outside)))
    # A pair can belong to only one nominal family within the supported harmonics.
    family = next(family for family in FAMILIES if all(value / family in PARTIALS for value in frequencies))
    i, j = (value // family for value in frequencies)
    disagreement = max(abs(_wrapped(j * a - i * b)) for a, b in
                       zip(first["inside_offsets_rad"], second["inside_offsets_rad"], strict=True))
    value = measurement["score"]
    if not (type(value) in (int, float) and math.isfinite(value) and value > POINT and abs(value - min(scores)) <= 3e-6
            and disagreement <= MAX_HARMONIC_RESIDUAL + 5e-6):
        raise ValueError("Tone phase score differs from its source measurements")
    return measurement


def observations(measurement):
    values = []
    for tone in measurement["tones"]:
        frequency = tone["frequency_hz"]
        for side, phase in zip(("start", "end"), tone["inside_offsets_rad"], strict=True):
            values.append({"label": f"{frequency} Hz phase offset near the {side}", "value": phase, "unit": "rad"})
        values.append({"label": f"{frequency} Hz minimum level above nearby frequencies",
                       "value": tone["tone_above_neighbors_db"], "unit": "dB"})
    return values
