"""Replay the strongest substantially different offset in a recorded lag search."""
from __future__ import annotations

KEY = "competing_lag"
METHOD = "strongest-at-least-three-ticks-away/1"
MIN_TICK_DISTANCE = 3
TICK_MS = 40


def compare(profile, selected_tick):
    """Report separation in correlation units on the same measured window."""
    selected = next(row["correlation"] for row in profile if row["offset_ticks"] == selected_tick)
    competitor = max((row for row in profile if abs(row["offset_ticks"] - selected_tick) >= MIN_TICK_DISTANCE),
                     key=lambda row: (row["correlation"], -abs(row["offset_ticks"] - selected_tick), -row["offset_ticks"]))
    return {"method": METHOD, "offset_ms": competitor["offset_ticks"] * TICK_MS,
            "correlation": competitor["correlation"], "correlation_difference": round(selected - competitor["correlation"], 6)}
