"""Standard-library validation for signed acoustic-context derivatives.

This validates coordinates and declared measurements, not their correspondence
to source sound. Report capture additionally recomputes them from frozen PCM.
"""
from __future__ import annotations

import math
import re

from app.evidence.audio_bandwidth_binding import BANDWIDTH_METHOD
from app.evidence.audio_bandwidth_binding import validate as validate_bandwidth

CONTINUITY_METHOD = {
    "side_samples": 8000,
    "step_samples": 4000,
    "fft_samples": 2048,
    "fft_hop_samples": 512,
    "frequency_bands": 24,
    "frequency_range_hz": [60, 7800],
    "band_spacing": "geometric",
    "profile": "median Hann-window power per band, normalized across bands",
    "distance": "square-root Jensen-Shannon distance, base 2",
    "maximum_within_side_instability": 0.20,
    "minimum_side_rms_dbfs": -60.0,
    "selection_threshold": 0.534556,
    "selection_comparison": "strictly greater than",
    "scope": "Mixed-track frequency content; no separated background or AI-origin classification.",
    "fft_support": "Full 2048-sample frames at 512-sample hops inside each half-second side; final 320 samples of each side are outside FFT support.",
}


def validate_acoustic_context(context: dict, *, source: dict, voice_packets: list[dict]) -> None:
    try:
        _validate(context, source, voice_packets)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("Acoustic context contains missing fields or invalid types") from error


def _validate(context, source, voice_packets):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def integer(value):
        return isinstance(value, int) and not isinstance(value, bool)

    def numeric(value, low, high):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and low <= value <= high

    def seconds(value, sample):
        return not isinstance(value, bool) and value in (sample / 16000, round(sample / 16000, 6))

    def same(value, expected):
        return not isinstance(value, bool) and value in (expected, round(expected, 6))

    def sha(value):
        return bool(re.fullmatch(r"[a-f0-9]{64}", str(value)))

    versions = {"chipmunk.acoustic-context/1": "mixed-acoustics-1", "chipmunk.acoustic-context/2": "mixed-acoustics-2",
                "chipmunk.acoustic-context/3": "mixed-acoustics-3"}
    require(context["schema"] in versions and context["status"] == "ready",
            "Unsupported acoustic-context schema or status")
    current = context["schema"] != "chipmunk.acoustic-context/1"
    bandwidth = context["schema"] == "chipmunk.acoustic-context/3"
    binding = context["source"]
    require(binding == {key: source[key] for key in ("sha256", "pcm_sha256", "samples", "sample_rate", "coordinate_space")}
            and all(sha(binding[key]) for key in ("sha256", "pcm_sha256")), "Acoustic context uses a different source input")
    samples = source["samples"]
    require(integer(samples) and 0 < samples <= 600 * 16000 and source["sample_rate"] == 16000
            and source["coordinate_space"] == "analysed-mono-pcm", "Invalid acoustic PCM coordinates")
    method = context["method"]
    require(method["version"] == versions[context["schema"]] and method["frame_samples"] == 8000
            and method["thresholds"] == {"energy_change_db": 12.0, "near_silence_dbfs": -60.0,
                "clipping_absolute_amplitude": 0.999, "clipping_fraction": 0.001}
            and method["spectral_bands_hz"] == {"low": [0, 250], "high": [4000, 8000]}
            and method["dbfs_floor"] == -120.0, "Unsupported acoustic measurement method")
    require(method["spectrogram"] == {"window_samples": 512, "hop_samples": 256, "window": "Hann",
                "time_pooling": "maximum magnitude across consecutive STFT frames; all frames retained in pooling",
                "maximum_time_columns": 880, "edge_padding": "zero", "dbfs_range": [-100, 0],
                "render_interpolation": "nearest"}, "Unsupported acoustic frequency-plot derivation")
    cursor = 0
    for block in source["timeline"]:
        require(all(integer(block[k]) for k in ("start_sample", "end_sample", "source_pts"))
                and block["start_sample"] == cursor < block["end_sample"] <= samples, "Invalid acoustic source clock")
        cursor = block["end_sample"]
    require(cursor == samples and source["timeline_time_base"] == "1/16000", "Acoustic source clock omits samples")

    def interval(span):
        a, b = span["start_sample"], span["end_sample"]
        require(integer(a) and integer(b) and 0 <= a < b <= samples, "Acoustic interval is outside its source")
        return a, b

    def ranges(a, b):
        return [{"start_sample": max(a, row["start_sample"]), "end_sample": min(b, row["end_sample"]),
                 "source_start_pts": row["source_pts"] + max(a, row["start_sample"]) - row["start_sample"],
                 "source_end_pts": row["source_pts"] + min(b, row["end_sample"]) - row["start_sample"], "time_base": "1/16000"}
                for row in source["timeline"] if max(a, row["start_sample"]) < min(b, row["end_sample"])]

    bins = context["bins"]
    require(isinstance(bins, list) and len(bins) == math.ceil(samples / 8000), "Acoustic bins do not cover the decoded input")
    cursor = 0
    for row in bins:
        a, b = interval(row)
        require(a == cursor and b == min(a + 8000, samples) and seconds(row["start_s"], a) and seconds(row["end_s"], b),
                "Acoustic bins have a gap or mismatched time axis")
        require(all(numeric(row[k], -120, 0) for k in ("rms_dbfs", "peak_dbfs"))
                and row["peak_dbfs"] >= row["rms_dbfs"]
                and all(numeric(row[k], 0, 1) for k in ("clipped_fraction", "low_band_fraction", "high_band_fraction", "spectral_flatness"))
                and row["low_band_fraction"] + row["high_band_fraction"] <= 1.000001,
                "Acoustic bin contains invalid energy or frequency measurements")
        cursor = b
    summary = context["summary"]
    require(summary["measured_samples"] == samples and summary["frame_count"] == len(bins)
            and numeric(summary["low_energy_reference_dbfs"], -120, 0) and numeric(summary["peak_dbfs"], -120, 0)
            and integer(summary["clipped_samples"]) and 0 <= summary["clipped_samples"] <= samples
            and integer(summary["near_silence_samples"])
            and sum(r["end_sample"] - r["start_sample"] for r in bins if r["rms_dbfs"] < -60.000001)
                <= summary["near_silence_samples"]
                <= sum(r["end_sample"] - r["start_sample"] for r in bins if r["rms_dbfs"] <= -59.999999),
            "Acoustic summary differs from measured coverage")
    require(integer(context["omitted_events"]) and context["omitted_events"] >= 0, "Invalid acoustic omission count")
    transitions = {}
    if current:
        require(method["continuity"] == CONTINUITY_METHOD, "Unsupported continuity derivation")
        rows = context["continuity_windows"]
        require(isinstance(rows, list) and len(rows) <= max(0, (samples - 16000) // 4000 + 1), "Invalid continuity window count")
        previous_boundary = -1
        eligible = []
        for row in rows:
            boundary = row["boundary_sample"]
            a, b = interval(row)
            require(integer(boundary) and boundary % 4000 == 0 and previous_boundary < boundary
                    and a == boundary - 8000 and b == boundary + 8000 and seconds(row["boundary_s"], boundary),
                    "Continuity comparison differs from source coordinates")
            previous_boundary = boundary
            require(numeric(row["energy_change_db"], 0, 120)
                    and all(numeric(row[k], 0, 1) for k in ("spectral_distance", "within_side_instability", "stationary_spectral_distance")),
                    "Invalid continuity measurements")
            gated = row["stationary_spectral_distance"]
            require((gated == 0 and (row["within_side_instability"] >= .199999 or row["spectral_distance"] == 0))
                    or (same(gated, row["spectral_distance"]) and row["within_side_instability"] <= .200001),
                    "Continuity stability condition differs")
            if gated > CONTINUITY_METHOD["selection_threshold"]:
                eligible.append(row)
        for row in sorted(eligible, key=lambda r: (-r["stationary_spectral_distance"], r["boundary_sample"])):
            if not any(abs(row["boundary_sample"] - other) < 16000 for other in transitions):
                transitions[row["boundary_sample"]] = row
            if len(transitions) >= 4:
                break
        require(integer(context["omitted_continuity_windows"])
                and context["omitted_continuity_windows"] == len(eligible) - len(transitions),
                "Continuity omission count differs")
    else:
        require("continuity_windows" not in context and "continuity" not in method, "Legacy acoustics cannot contain new continuity fields")

    measured_bandwidth = None
    if bandwidth:
        require(method["upper_frequency_drop"] == BANDWIDTH_METHOD and samples == 96000
                and source.get("source_channels") == 1 and source.get("source_sample_rate") == 16000
                and source.get("source_duration_s") == 6
                and source["timeline"] == [{"start_sample": 0, "end_sample": 96000, "source_pts": 0}],
                "Unsupported temporary upper-frequency review input or method")
        detail = context["upper_frequency_drop"]
        count, status = detail["assessed_ranges"], detail["status"]
        require(integer(count) and 0 <= count <= 275 and status in {"unassessed", "no_supported_drop", "measured_drop"}
                and (count == 0) == (status == "unassessed") and ("measurement" in detail) == (status == "measured_drop"),
                "Invalid temporary upper-frequency review coverage")
        measured_bandwidth = detail.get("measurement")
    else:
        require("upper_frequency_drop" not in context and "upper_frequency_drop" not in method,
                "Legacy acoustics cannot contain upper-frequency review")

    def image(asset, height):
        require(asset["media_type"] == "image/png" and sha(asset["sha256"])
                and asset["width"] == 960 and asset["height"] == height, "Invalid acoustic image dimensions")

    require(set(context["overview"]) == {"energy", "spectrogram"}, "Acoustic overview lacks required source plots")
    image(context["overview"]["energy"], 280)
    image(context["overview"]["spectrogram"], 320)
    categories = {"near_silence", "energy_change", "clipping"}
    if current:
        categories.add("spectral_transition")
    if bandwidth:
        categories.add("upper_frequency_drop")
    require(isinstance(context["events"], list) and len(context["events"]) <= 8 + 4 * current + bandwidth,
            "Too many acoustic evidence events")
    previous = None
    seen_transitions = set()
    seen_bandwidth = False
    for index, event in enumerate(context["events"], 1):
        require(event["id"] == f"B{index:02d}" and event["evidence_type"] == "descriptive"
                and event["category"] in categories, "Invalid acoustic event identity or semantics")
        a, b = interval(event["interval"])
        ca, cb = interval(event["context_interval"])
        order = (a, b, event["category"])
        require((previous is None or previous < order) and ca == max(0, a - 32000) and cb == min(samples, b + 32000)
                and seconds(event["start_s"], a) and seconds(event["end_s"], b), "Acoustic event context or ordering differs")
        previous = order
        require(event["source_timestamp_ranges"] == ranges(a, b), "Acoustic event source timestamps differ")
        require(event["related_voice_packet_ids"] == [p["id"] for p in voice_packets
                if p["interval"]["start_sample"] < b and p["interval"]["end_sample"] > a], "Acoustic event voice references differ")
        selected = [row for row in bins if a <= row["start_sample"] < b]
        require(event["category"] in {"spectral_transition", "upper_frequency_drop"}
                or (bool(selected) and selected[0]["start_sample"] == a and selected[-1]["end_sample"] == b),
                "Acoustic event does not align to measured bins")
        value = event["observations"][0]["value"]
        require(numeric(value, -120, 240 if event["category"] == "upper_frequency_drop" else 120), "Invalid acoustic event measurement")
        if event["category"] == "upper_frequency_drop":
            require(measured_bandwidth is not None and not seen_bandwidth and len(event["observations"]) == 3,
                    "Temporary upper-frequency event has no unique support")
            require(same(value, validate_bandwidth(measured_bandwidth, start=a, end=b, samples=samples)),
                    "Temporary upper-frequency observation differs from its support")
            before, inside, after = measured_bandwidth["relative_band_db"]
            require(all(same(event["observations"][i]["value"], round(min(before[i], after[i]) - inside[i], 6)) for i in (1, 2)),
                    "Upper-frequency observations differ from their band measurements")
            seen_bandwidth = True
        elif event["category"] == "spectral_transition":
            boundary = event["change_sample"]
            row = transitions.get(boundary)
            require(row is not None and boundary not in seen_transitions and a == row["start_sample"] and b == row["end_sample"]
                    and same(value, row["spectral_distance"]) and len(event["observations"]) == 3
                    and same(event["observations"][1]["value"], row["within_side_instability"])
                    and same(event["observations"][2]["value"], row["energy_change_db"]),
                    "Sound-content transition differs from its comparison windows")
            seen_transitions.add(boundary)
        elif event["category"] == "near_silence":
            require(b - a >= 8000 and all(row["rms_dbfs"] <= -59.999999 for row in selected)
                    and same(value, min(row["rms_dbfs"] for row in selected)), "Quiet event differs from measured energy")
        elif event["category"] == "clipping":
            require(all(row["clipped_fraction"] >= .000999 for row in selected)
                    and same(value, max(row["clipped_fraction"] for row in selected)), "Clipping event differs from measured samples")
        else:
            require(len(selected) == 2 and b - a == 16000 and event["change_sample"] == a + 8000
                    and min(row["rms_dbfs"] for row in selected) >= -60.000001
                    and abs(value) >= 12 and abs(value - (selected[1]["rms_dbfs"] - selected[0]["rms_dbfs"])) <= .000002,
                    "Energy-change event differs from adjacent measured bins")
        require(set(event["artifacts"]) == {"selected_clip", "context_clip", "waveform", "spectrogram"},
                "Acoustic event lacks required source assets")
        for role, start, end in (("selected_clip", a, b), ("context_clip", ca, cb)):
            asset = event["artifacts"][role]
            require(asset["media_type"] == "audio/wav" and sha(asset["sha256"]) and asset["sample_rate"] == 16000
                    and asset["channels"] == 1 and asset["samples"] == end - start, "Acoustic clip differs from its sample interval")
        image(event["artifacts"]["waveform"], 280)
        image(event["artifacts"]["spectrogram"], 320)
    require(seen_transitions == set(transitions), "Selected continuity leads are missing")
    require(seen_bandwidth == (measured_bandwidth is not None), "Selected temporary upper-frequency observation is missing")
