"""Reproducible mixed-track acoustic context, never a synthetic-voice verdict.

All measurements use the same frozen PCM as the voice checks. They describe
recording energy and frequency content; no source separation or speech/background
classification is claimed. Rendering and selection happen only at publication.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageDraw

from app.core.canonical import canonical_hash, canonical_json
from app.detectors.audio.decode import SAMPLE_RATE, source_ranges, wav_bytes
from app.evidence import audio_silence
from app.evidence import audio_silence_binding as silence
from app.evidence.audio_bandwidth_binding import (
    BANDWIDTH_METHOD,
    EXPANDED_SCHEMA,
    EXPANDED_VERSION,
    POINT_DB,
    native_input,
)
from app.evidence.audio_bandwidth_binding import (
    LEGACY_SCHEMA as BANDWIDTH_SCHEMA,
)
from app.evidence.audio_bandwidth_binding import (
    LEGACY_VERSION as BANDWIDTH_VERSION,
)
from app.evidence.audio_continuity import METHOD as CONTINUITY_METHOD
from app.evidence.audio_continuity import candidates as continuity_windows
from app.evidence.audio_continuity import select as select_continuity

SCHEMA = "chipmunk.acoustic-context/2"
VERSION = "mixed-acoustics-2"
FRAME_SAMPLES = 8000
MAX_EVENTS = 8
THRESHOLDS = {"energy_change_db": 12.0, "near_silence_dbfs": -60.0,
              "clipping_absolute_amplitude": 0.999, "clipping_fraction": 0.001}
LIMITATIONS = [
    "These measurements describe the mixed mono track. Speech, music and background sound have not been separated or identified.",
    "Energy changes, quiet intervals and clipping can occur in ordinary recordings, speech pauses, edits, compression or automatic gain control; none establishes AI generation.",
    "The lowest-energy frames are only a recording-level reference, not an isolated background-noise measurement.",
    "Half-second measurement windows do not locate an exact splice or voice-replacement boundary. Frequency plots are inspection context, not a synthesis score.",
    "Only the frozen decoded input was inspected; original stereo separation and audio beyond the decode limit are not assessed here.",
]


def _db(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-6))


def measurements(pcm: np.ndarray) -> list[dict]:
    """Non-overlapping bins partition all decoded samples, including the tail."""
    pcm = np.asarray(pcm)
    if pcm.ndim != 1 or not 0 < len(pcm) <= 600 * SAMPLE_RATE or not np.isfinite(pcm).all() or np.max(np.abs(pcm)) > 1:
        raise ValueError("Invalid PCM for acoustic context")
    bins = []
    for a in range(0, len(pcm), FRAME_SAMPLES):
        b = min(len(pcm), a + FRAME_SAMPLES)
        values = pcm[a:b].astype(np.float64)
        centered = values - values.mean()
        power = np.abs(np.fft.rfft(centered * np.hanning(len(values)))) ** 2
        frequencies = np.fft.rfftfreq(len(values), 1 / SAMPLE_RATE)
        total = float(power.sum())
        flatness = float(np.exp(np.log(np.maximum(power, 1e-20)).mean()) / max(float(power.mean()), 1e-20)) if total > 1e-16 else 0.0
        bins.append({"start_sample": a, "end_sample": b, "start_s": a / SAMPLE_RATE, "end_s": b / SAMPLE_RATE,
                     "rms_dbfs": _db(float(np.sqrt(np.mean(values ** 2)))), "peak_dbfs": _db(float(np.max(np.abs(values)))),
                     "clipped_fraction": float(np.mean(np.abs(values) >= THRESHOLDS["clipping_absolute_amplitude"])),
                     "low_band_fraction": float(power[frequencies < 250].sum() / total) if total > 1e-16 else 0.0,
                     "high_band_fraction": float(power[frequencies >= 4000].sum() / total) if total > 1e-16 else 0.0,
                     "spectral_flatness": min(1.0, max(0.0, flatness))})
    return bins


def _candidates(bins: list[dict]) -> list[dict]:
    candidates = []
    for category, key, predicate in (
        ("near_silence", "rms_dbfs", lambda x: x <= THRESHOLDS["near_silence_dbfs"]),
        ("clipping", "clipped_fraction", lambda x: x >= THRESHOLDS["clipping_fraction"]),
    ):
        group = []
        for row in [*bins, None]:
            if row is not None and predicate(row[key]):
                group.append(row)
            elif group:
                a, b = group[0]["start_sample"], group[-1]["end_sample"]
                # A fractional last bin must not turn one quiet sample into an interval claim.
                if b - a >= FRAME_SAMPLES or category == "clipping":
                    value = min(v[key] for v in group) if category == "near_silence" else max(v[key] for v in group)
                    candidates.append({"category": category, "start_sample": a, "end_sample": b, "value": value,
                                       "priority": 1 if category == "near_silence" else 3,
                                       "strength": (b-a)/SAMPLE_RATE if category == "near_silence" else value})
                group = []
    for previous, current in zip(bins, bins[1:]):
        change = current["rms_dbfs"] - previous["rms_dbfs"]
        # Quiet intervals have their own category. Excluding them avoids counting
        # a single pause twice and claiming a room-tone discontinuity.
        if (min(previous["rms_dbfs"], current["rms_dbfs"]) > THRESHOLDS["near_silence_dbfs"]
                and abs(change) >= THRESHOLDS["energy_change_db"]
                and current["end_sample"] - current["start_sample"] == FRAME_SAMPLES):
            candidates.append({"category": "energy_change", "start_sample": previous["start_sample"],
                               "end_sample": current["end_sample"], "change_sample": current["start_sample"],
                               "value": change, "priority": 2, "strength": abs(change)})
    return candidates


def _base(pcm: np.ndarray, mapping: dict, source: dict, voice_packets: list[dict], *, version: str = VERSION) -> dict:
    if not isinstance(version, str):
        raise ValueError("Unsupported acoustic measurement version")
    with_silence = version.endswith(silence.VERSION_SUFFIX)
    base_version = version.removesuffix(silence.VERSION_SUFFIX)
    schemas = {"mixed-acoustics-1": "chipmunk.acoustic-context/1", VERSION: SCHEMA,
               BANDWIDTH_VERSION: BANDWIDTH_SCHEMA, EXPANDED_VERSION: EXPANDED_SCHEMA}
    if base_version not in schemas or (with_silence and base_version == "mixed-acoustics-1"):
        raise ValueError("Unsupported acoustic measurement version")
    current = base_version != "mixed-acoustics-1"
    bandwidth = base_version in {BANDWIDTH_VERSION, EXPANDED_VERSION}
    bins = measurements(pcm)
    candidates = _candidates(bins)
    selected = sorted(candidates, key=lambda c: (-c["priority"], -c["strength"], c["start_sample"], c["category"]))[:MAX_EVENTS]
    rows = json.loads(canonical_json(continuity_windows(pcm))) if current else []
    extra = select_continuity(rows)
    selected += [{**row, "category": "spectral_transition", "value": row["spectral_distance"],
                  "change_sample": row["boundary_sample"]} for row in extra]
    bandwidth_detail = None
    if bandwidth:
        from app.evidence.audio_bandwidth_confirmation import locate

        if not native_input(len(pcm), mapping, base_version):
            raise ValueError("Temporary upper-frequency review requires a supported lossless mono source and clock")
        count, best = locate(pcm)
        bandwidth_detail = {"assessed_ranges": count, "status": "unassessed" if best is None else "no_supported_drop"}
        if best and best["drop_db"] > POINT_DB:
            a, b = best["range"]
            bandwidth_detail.update(status="measured_drop", measurement=best)
            selected.append({"category": "upper_frequency_drop", "start_sample": a, "end_sample": b, "value": best["drop_db"]})
    silence_detail = audio_silence.measure(pcm, mapping["timeline"]) if with_silence else None
    if silence_detail is not None:
        selected.extend({**row, "category": silence.KEY, "value": 0} for row in silence_detail["selected"])
    selected.sort(key=lambda c: (c["start_sample"], c["end_sample"], c["category"]))
    document = {"schema": silence.CONTEXT_SCHEMA if with_silence else schemas[base_version], "status": "ready", "source": source,
                "method": {"version": version, "frame_samples": FRAME_SAMPLES,
                           "description": "Deterministic measurements of frozen mixed mono PCM; no sound-source separation or AI classification.",
                           "thresholds": THRESHOLDS, "selection": "Clipping, then energy changes, then longest quiet intervals; up to eight, shown in time order.",
                           "spectral_bands_hz": {"low": [0, 250], "high": [4000, 8000]},
                           "spectrogram": {"window_samples": 512, "hop_samples": 256, "window": "Hann",
                               "time_pooling": "maximum magnitude across consecutive STFT frames; all frames retained in pooling",
                               "maximum_time_columns": 880, "edge_padding": "zero",
                               "dbfs_range": [-100, 0], "render_interpolation": "nearest"},
                           "dbfs_floor": -120.0},
                "summary": {"measured_samples": len(pcm), "frame_count": len(bins),
                            "low_energy_reference_dbfs": float(np.percentile([x["rms_dbfs"] for x in bins], 10)),
                            "peak_dbfs": _db(float(np.max(np.abs(pcm)))),
                            "clipped_samples": int(np.sum(np.abs(pcm) >= THRESHOLDS["clipping_absolute_amplitude"])),
                            "near_silence_samples": sum(x["end_sample"]-x["start_sample"] for x in bins if x["rms_dbfs"] <= THRESHOLDS["near_silence_dbfs"])},
                "bins": bins, "events": [], "omitted_events": max(0, len(candidates) - MAX_EVENTS), "limitations": list(LIMITATIONS)}
    if current:
        document["method"]["continuity"] = CONTINUITY_METHOD
        document["method"]["selection"] += " Up to four distinct frequency-content transitions are added without replacing the original leads."
        document["continuity_windows"] = rows
        document["omitted_continuity_windows"] = sum(row["stationary_spectral_distance"] > CONTINUITY_METHOD["selection_threshold"] for row in rows) - len(extra)
        document["limitations"].append("A possible sound-content transition may be ordinary music, a new sound, microphone movement or recording processing. Steady sound helped this check in local testing; it missed many other edits and does not identify AI generation.")
    if bandwidth:
        document["method"]["upper_frequency_drop"] = BANDWIDTH_METHOD
        document["upper_frequency_drop"] = bandwidth_detail
        document["limitations"].append("The short upper-frequency check compares the mixed track only when surrounding upper-frequency sound is measurable. A supported dip can result from ordinary sound changes, filtering, or a recording-path change and carries zero weight in the AI estimate.")
    if silence_detail is not None:
        document["method"][silence.KEY] = silence.METHOD
        document[silence.KEY] = silence_detail
        document["limitations"].append(silence.LIMITATION)
    for i, candidate in enumerate(selected, 1):
        a, b, category, value = (candidate[k] for k in ("start_sample", "end_sample", "category", "value"))
        ca, cb = max(0, a-2*SAMPLE_RATE), min(len(pcm), b+2*SAMPLE_RATE)
        if category == silence.KEY:
            ca, cb = a - silence.FLANK, b + silence.FLANK
            label, unit = "Nonzero samples in the selected interval", "samples"
            observation = f"All {b-a:,} samples are zero from {a/SAMPLE_RATE:.3f} to {b/SAMPLE_RATE:.3f} s, with measurable sound on both sides."
            interpretation = silence.LIMITATION + " This observation carries zero AI-confidence weight."
        elif category == "upper_frequency_drop":
            label, unit = "Minimum relative drop across both upper frequency bands and both flanks", "dB"
            observation = f"The mixed track has at least {value:.1f} dB less upper-frequency energy relative to the lower band across {a/SAMPLE_RATE:.3f} to {b/SAMPLE_RATE:.3f} s. Comparable upper-frequency sound returns afterward."
            interpretation = "Listen to the selected interval and both surrounding portions. Ordinary sound changes or filtering can produce this pattern. This observation carries zero weight in the AI estimate."
        elif category == "spectral_transition":
            label, unit = "Frequency-content distance between adjacent half-second windows", "distance (0–1)"
            observation = f"The mixed-track frequency profile changes around the {candidate['change_sample']/SAMPLE_RATE:.3f} s comparison position (distance {value:.3f}; review threshold {CONTINUITY_METHOD['selection_threshold']:.3f})."
            interpretation = "Possible sound-content transition: listen to the selected interval and surrounding sound. Ordinary music, a new sound or recording processing can cause this pattern. This is a listening lead, not evidence of AI origin or an exact splice boundary."
        elif category == "energy_change":
            label, unit = "Adjacent half-second RMS change", "dB"
            observation = f"Mixed-track RMS changes by {value:+.1f} dB across the {candidate['change_sample']/SAMPLE_RATE:.3f} s bin boundary."
            interpretation = "Listen for a recording-level or ambience transition. Speech onset, music and automatic gain changes can produce the same pattern."
        elif category == "clipping":
            label, unit = "Highest measured-bin near-full-scale fraction", "fraction"
            observation = f"At least one measured bin contains {value*100:.2f}% near-full-scale samples."
            interpretation = "Inspect possible recording distortion. This threshold marks samples near full scale; it does not prove clipping at capture or synthetic audio."
        else:
            label, unit = "Lowest measured-bin RMS", "dBFS"
            observation = f"The mixed track stays at or below -60 dBFS RMS for {(b-a)/SAMPLE_RATE:.3f} s."
            interpretation = "Listen for a pause or very quiet recording interval. Low energy does not establish missing ambience, a splice or authentic audio."
        event = {"id": f"B{i:02d}", "category": category, "evidence_type": "descriptive",
                 "interval": {"start_sample": a, "end_sample": b}, "context_interval": {"start_sample": ca, "end_sample": cb},
                 "start_s": a/SAMPLE_RATE, "end_s": b/SAMPLE_RATE,
                 "source_timestamp_ranges": source_ranges(mapping, a, b),
                 "observation": observation, "interpretation": interpretation,
                 "observations": [{"label": label, "value": value, "unit": unit}],
                 "related_voice_packet_ids": [p["id"] for p in voice_packets if p["interval"]["start_sample"] < b and p["interval"]["end_sample"] > a],
                 "limitations": ["This interval is descriptive acoustic context, not a synthetic-voice finding or exact edit boundary."]}
        if "change_sample" in candidate:
            event["change_sample"] = candidate["change_sample"]
        if category == silence.KEY:
            event["title"] = silence.TITLE
            event["observations"].extend({"label": f"{side} 100 ms RMS", "value": level, "unit": "dBFS"}
                                         for side, level in zip(("Before", "After"), candidate["flank_rms_dbfs"], strict=True))
        elif category == "upper_frequency_drop":
            event["title"] = "Temporary upper-frequency drop"
            before, inside, after = bandwidth_detail["measurement"]["relative_band_db"]
            for index, frequency in enumerate(("4.1–5.5 kHz", "5.5–7.4 kHz"), 1):
                event["observations"].append({"label": f"{frequency} relative drop compared with both flanks",
                                               "value": round(min(before[index], after[index]) - inside[index], 6), "unit": "dB"})
        elif category == "spectral_transition":
            event["title"] = "Possible sound-content transition"
            event["observations"].extend([
                {"label": "Within-side frequency variation", "value": candidate["within_side_instability"], "unit": "distance (0–1)"},
                {"label": "Absolute RMS change between comparison windows", "value": candidate["energy_change_db"], "unit": "dB"},
            ])
        document["events"].append(event)
    return document


def _png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def energy_chart(bins: list[dict], samples: int) -> bytes:
    image = Image.new("RGB", (960, 280), (250, 249, 245))
    draw = ImageDraw.Draw(image)
    draw.text((58, 12), "Mixed-track energy (dBFS; descriptive context)", fill=(30, 50, 40))
    draw.rectangle((58, 40, 938, 232), outline=(135, 150, 135))
    for db in (-120, -60, 0):
        y = 232 - (db+120)/120*192
        draw.text((10, y-5), str(db), fill=(50, 70, 60))
        draw.line((58, y, 938, y), fill=(215, 219, 210))
    for row in bins:
        y = 232 - (row["rms_dbfs"]+120)/120*192
        draw.line((58+row["start_sample"]/samples*880, y, 58+row["end_sample"]/samples*880, y), fill=(45, 100, 78), width=3)
    for fraction in (0, .25, .5, .75, 1):
        draw.text((46+fraction*880, 242), f"{fraction*samples/SAMPLE_RATE:.2f}s", fill=(50, 70, 60))
    return _png(image)


def spectrogram_values(pcm: np.ndarray, start: int, end: int) -> np.ndarray:
    """Full-coverage centered STFT, streamed into at most 880 time columns.

    Every 256-sample hop is retained. Long clips use maximum magnitude over
    consecutive frame groups, so a short sound between overview pixels is not
    silently skipped. No interpolation invents additional frequency slices.
    """
    # Include the final padded center: otherwise a last sample exactly on a
    # Hann endpoint would have zero weight in every retained frame.
    frame_count = (end-start+255)//256 + 1
    column_count = min(880, frame_count)
    output = np.zeros((257, column_count))
    window = np.hanning(512)
    # Batch computation bounds working memory independently of recording length.
    for first in range(0, frame_count, 512):
        count = min(512, frame_count-first)
        values = np.zeros((count, 512))
        for offset in range(count):
            center = start + (first+offset)*256
            a, b = max(start, center-256), min(end, center+256)
            left = a-(center-256)
            values[offset, left:left+b-a] = pcm[a:b]
        amplitudes = np.abs(np.fft.rfft(values*window, axis=1)) / (window.sum()/2)
        for offset, row in enumerate(amplitudes):
            column = (first+offset)*column_count//frame_count
            output[:, column] = np.maximum(output[:, column], row)
    return output


def spectrogram(pcm: np.ndarray, start: int, end: int, selected: tuple[int, int] | None = None) -> bytes:
    """Fixed-scale overview of complete STFT coverage, with peak time pooling."""
    image = Image.new("RGB", (960, 320), (250, 249, 245))
    draw = ImageDraw.Draw(image)
    amplitudes = spectrogram_values(pcm, start, end)
    values = np.clip((20*np.log10(np.maximum(amplitudes, 1e-6))+100)/100, 0, 1)[::-1]
    colors = np.stack((235-205*values, 240-140*values, 230-152*values), axis=2).astype(np.uint8)
    image.paste(Image.fromarray(colors).resize((880, 232), Image.Resampling.NEAREST), (58, 40))
    draw.rectangle((58, 40, 938, 272), outline=(135, 150, 135))
    draw.text((58, 12), "Mixed-track spectrogram (all STFT frames, peak time pooling; -100 to 0 dBFS)", fill=(30, 50, 40))
    for hz in (0, 4000, 8000):
        draw.text((8, 266-hz/8000*232), f"{hz/1000:g}kHz", fill=(50, 70, 60))
    for fraction in (0, .25, .5, .75, 1):
        draw.text((46+fraction*880, 286), f"{(start+fraction*(end-start))/SAMPLE_RATE:.2f}s", fill=(50, 70, 60))
    if selected:
        for sample in selected:
            x = 58+(sample-start)/(end-start)*880
            draw.line((x, 40, x, 272), fill=(188, 106, 33), width=2)
    return _png(image)


def build_content(pcm: np.ndarray, mapping: dict, source: dict, voice_packets: list[dict],
                  make_asset: Callable, waveform: Callable, *, version: str = VERSION) -> dict:
    document = _base(pcm, mapping, source, voice_packets, version=version)
    document["overview"] = {"energy": make_asset(energy_chart(document["bins"], len(pcm)), "image/png", width=960, height=280),
                            "spectrogram": make_asset(spectrogram(pcm, 0, len(pcm)), "image/png", width=960, height=320)}
    for event in document["events"]:
        a, b = (event["interval"][key] for key in ("start_sample", "end_sample"))
        ca, cb = (event["context_interval"][key] for key in ("start_sample", "end_sample"))
        event["artifacts"] = {
            "selected_clip": make_asset(wav_bytes(pcm[a:b]), "audio/wav", samples=b-a, sample_rate=SAMPLE_RATE, channels=1),
            "context_clip": make_asset(wav_bytes(pcm[ca:cb]), "audio/wav", samples=cb-ca, sample_rate=SAMPLE_RATE, channels=1),
            "waveform": make_asset(waveform(pcm, [], start=ca, end=cb, selected=(a,b), scores=False), "image/png", width=960, height=280),
            "spectrogram": make_asset(spectrogram(pcm, ca, cb, (a,b)), "image/png", width=960, height=320),
        }
    return document


def verify_source_content(context: dict, *, pcm: np.ndarray, mapping: dict, source: dict,
                          voice_packets: list[dict], read_asset: Callable, waveform: Callable) -> None:
    """Recompute every descriptive value and pixel/clip hash from frozen PCM."""
    def expected_asset(data, media_type, **metadata):
        return {"sha256": hashlib.sha256(data).hexdigest(), "media_type": media_type, **metadata}
    expected = build_content(pcm, mapping, source, voice_packets, expected_asset, waveform,
                             version=context.get("method", {}).get("version"))
    def without_urls(value):
        if isinstance(value, dict):
            return {k: without_urls(v) for k,v in value.items() if k != "url"}
        return [without_urls(v) for v in value] if isinstance(value, list) else value
    if canonical_hash(without_urls(context)) != canonical_hash(expected):
        raise ValueError("Acoustic context differs from its frozen PCM measurements or derived assets")
    for asset in [*context["overview"].values(), *(asset for event in context["events"] for asset in event["artifacts"].values())]:
        if hashlib.sha256(read_asset(asset["sha256"])).hexdigest() != asset["sha256"]:
            raise ValueError("Acoustic context asset no longer matches its recorded hash")
