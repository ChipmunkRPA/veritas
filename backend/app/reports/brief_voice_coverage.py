"""Present signed brief-window admissions on their unchanged native source clock."""

from app.evidence.audio_brief_voice_binding import GRID, IDENTICAL_STEREO_SCHEMA, RATE, SAMPLES, SCHEMA, WIDTH


def coverage(review: dict, source: dict) -> dict | None:
    """Return display-only coverage, declining missing or inconsistent historical data."""
    if (review.get("schema") not in {SCHEMA, IDENTICAL_STEREO_SCHEMA} or review.get("status") == "skipped"
            or source.get("samples") != SAMPLES or source.get("sample_rate") != RATE
            or source.get("timeline_time_base") != "1/16000"
            or source.get("timeline") != [{"start_sample": 0, "end_sample": SAMPLES, "source_pts": 0}]
            or review.get("scheduled_windows") != len(GRID)):
        return None
    checks = review.get("window_checks")
    if not isinstance(checks, list) or len(checks) != len(GRID):
        return None
    windows = []
    for check, start in zip(checks, GRID, strict=True):
        if not isinstance(check, dict):
            return None
        speech, elevated = check.get("speech_samples"), check.get("elevated")
        if (type(check.get("start_sample")) is not int or check["start_sample"] != start
                or type(speech) is not int or not 0 <= speech <= WIDTH
                or "elevated" not in check or elevated is not None and type(elevated) is not bool
                or (elevated is not None) != (speech * 2 >= WIDTH)):
            return None
        windows.append({"start_sample": start, "end_sample": start + WIDTH,
                        "speech_samples": speech, "checked": elevated is not None})
    if sum(window["checked"] for window in windows) != review.get("speech_windows_checked"):
        return None
    boundaries = sorted({0, SAMPLES, *(window[key] for window in windows for key in ("start_sample", "end_sample"))})
    intervals = []
    for start, end in zip(boundaries, boundaries[1:]):
        covering = [window for window in windows if window["start_sample"] <= start < end <= window["end_sample"]]
        status = "checked" if any(window["checked"] for window in covering) else "insufficient_speech" if covering else "outside_schedule"
        if intervals and intervals[-1]["status"] == status and intervals[-1]["end_sample"] == start:
            intervals[-1]["end_sample"] = end
        else:
            intervals.append({"start_sample": start, "end_sample": end, "status": status})
    return {"checked_samples": sum(row["end_sample"] - row["start_sample"] for row in intervals if row["status"] == "checked"),
            "intervals": intervals, "windows": windows}
