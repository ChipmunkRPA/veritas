"""Codec lane as a v2 plugin (V-6; VF §7, §9.6; ARCHITECTURE §3.1 `detect`). The method is in `video/codec.py`.

The detect stage passes `CodecInputs` in `outputs.codec`: the original file, the persisted frame table, the ffprobe
facts and the detected shot cuts.

- **Score.** A capped explanatory probability (0.55 for one irregularity kind, +0.05 per further kind, at most 0.70) with
  outcome weight 0.3, or no aggregate when nothing is irregular; the facts are published either way.
- **Exhibits.** A codec facts table; the GOP strip data (GOP length at each key frame, irregular key frames, timestamp
  gaps) as a series; a key-frame table (at most 500 rows); the encoder identification strings when there are any. When
  there is an aggregate, they cite the signal `<detector_id>:global` the video fuser builds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.detectors.video import codec
from app.detectors.video.adapters.common import CPU_ONLY, FULL_DEEP, H3, video_spec
from app.detectors.video.decode import FrameTable
from app.engine.detector import Applicability, Detector, DetectorResult, InputUnit, RunContext, UnitScore

MAX_KEY_ROWS = 500
REFERENCE = ("Explanatory codec facts: camera and phone encoders usually keep a fixed GOP and aligned streams. An extra "
             "key frame away from a shot cut, several encoder configurations in one stream, discarded leading packets, "
             "timestamp gaps or an audio start or duration difference above 0.5 s are consistent with splicing, trimming, "
             "re-muxing or dubbing, and equally with ordinary editing and re-sharing. Hand-set score within [0.55, 0.70], "
             "weight 0.3, uncalibrated")


@dataclass(frozen=True)
class CodecInputs:
    path: Path
    table: FrameTable
    info: dict[str, Any]  # probe stage facts
    cuts: tuple[int, ...]  # frame-table indices where detected shots start (after the first)
    shots_source: str  # detected | not detected


def _rate(value: Any) -> str:
    num, _, den = str(value or "0/1").partition("/")
    try:
        return f"{float(num) / float(den or 1):.3f}".rstrip("0").rstrip(".")
    except (ValueError, ZeroDivisionError):
        return "unknown"


def _seconds(value: Any) -> str:
    v = codec._float(value)
    return "unknown" if v is None else f"{v:.3f} s"


def _edits(edits: tuple[codec.EditList, ...]) -> str:
    if not edits:
        return "none (no edit list, or not an MP4 / MOV container)"
    parts = []
    for e in edits:
        entries = [f"empty {d:.3f} s" if t is None else f"{d:.3f} s from {t:.3f} s" for d, t in e.entries]
        parts.append(f"{e.handler}: " + ", ".join(entries))
    return "; ".join(parts)


def fact_rows(report: codec.CodecReport, table: FrameTable) -> list[list[str | float | None]]:
    s, gop, t, p = report.streams, report.gop, report.timing, report.packets
    video, audio, fmt, tags = s.video, s.audio, s.format, s.tags()
    if gop.dominant is None:
        gop_text = f"not established ({max(len(gop.lengths) - 1, 0)} complete GOP(s))"
    else:
        gop_text = f"{gop.dominant} frames" + (", regular" if gop.regular else
                                               f", {len(gop.irregular)} irregular key frame(s), {len(gop.at_cuts)} at shot cuts")
    rows: list[list[str | float | None]] = [
        ["container", str(fmt.get("format_name") or "unknown"), None],
        ["video codec", " ".join(str(video.get(k)) for k in ("codec_name", "profile", "pix_fmt") if video.get(k)), None],
        ["resolution", f"{table.width} × {table.height}" + (f", display rotation {table.rotation}°" if table.rotation else ""), None],
        ["frame rate, nominal / average", f"{_rate(video.get('r_frame_rate'))} / {_rate(video.get('avg_frame_rate'))} fps",
         "a difference suggests variable frame rate or frame-rate conversion"],
        ["frames", float(len(table)), None],
        ["key frames", float(len(gop.key_frames)), None],
        ["GOP", gop_text, "camera and phone encoders usually keep a fixed GOP (not measured here)"],
        ["first key frame", "none" if not gop.key_frames else f"frame {gop.key_frames[0]}" + (" (decoding starts mid-GOP)" if gop.starts_mid_gop else ""), None],
        ["B-frames", float(gop.b_frames), None],
        ["frame timing", "variable frame rate" if t.vfr else "constant frame rate",
         "phones and screen recorders often write variable frame rate"],
        ["timestamp gaps / duplicates / non-monotonic", f"{len(t.gaps)} / {t.duplicates} / {t.non_monotonic}", None],
        ["video packets discarded before display / non-monotonic dts", f"{p.discarded} / {p.non_monotonic_dts}" + (" (packet list capped)" if p.capped else ""),
         "a stream-copy trim leaves discarded leading packets"],
        ["edit lists", _edits(report.edits), "an empty edit delays a track; a media time well past the first frames hides them"],
        ["video start / duration", f"{_seconds(video.get('start_time'))} / {_seconds(video.get('duration'))}", None],
        ["audio", "none" if audio is None else " ".join(
            x for x in (str(audio.get("codec_name") or ""), f"{audio.get('sample_rate')} Hz" if audio.get("sample_rate") else "",
                        f"{audio.get('channels')} ch" if audio.get("channels") else "",
                        f"start {_seconds(audio.get('start_time'))}", f"duration {_seconds(audio.get('duration'))}") if x), None],
        ["audio − video start", None if report.av_start_delta_s is None else f"{report.av_start_delta_s:+.3f} s",
         "above 0.5 s: consistent with a remux or a replaced sound track"],
        ["audio − video duration", None if report.av_duration_delta_s is None else f"{report.av_duration_delta_s:+.3f} s", None],
        ["encoder tag", tags.get("encoder") or tags.get("handler_name") or "none", "Lavf / Lavc means an ffmpeg remux or re-encode"],
        ["major brand", tags.get("major_brand") or "none", None],
        ["capture-device tags", ", ".join(f"{k}={v}" for k, v in sorted(tags.items()) if k.startswith(("com.apple.", "com.android."))) or "none",
         "forgeable; a re-encode drops them"],
        ["encoder identification strings", f"{len(report.sei)} (first {report.sei_scanned_bytes / 1048576:.1f} MB scanned"
         + (", scan capped)" if report.sei_capped else ")"), "several in one file are consistent with joined encodes"],
    ]
    return rows


class CodecDetector(Detector):
    spec = video_spec("vid.codec.ffprobe", "Codec lane: GOP structure, timestamps, A/V alignment, edit lists and encoder strings (ffprobe)",
                      InputUnit.FILE, (H3,), ("table", "series"), family="codec", needs=("probe",), lane="cpu", devices=CPU_ONLY,
                      timeout_s=120.0, profiles=FULL_DEEP, version="0.4.1")  # 0.4.1 (N-19p, R10-03): the strings of its tables keep their words apart (tab and newline as a space)

    def applicable(self, ctx: RunContext) -> Applicability:
        inputs = ctx.outputs.codec
        if not isinstance(inputs, CodecInputs) or len(inputs.table) == 0:
            return Applicability(False, "no decodable video frames")
        return Applicability(True)

    def run(self, ctx: RunContext) -> DetectorResult:
        inputs: CodecInputs = ctx.outputs.codec
        table = inputs.table
        report = codec.analyse(inputs.path, table, inputs.cuts, shots_detected=inputs.shots_source == "detected")
        probability = report.probability
        cite = [f"{self.spec.id}:global"] if probability is not None else []
        gop, stamps = report.gop, report.timing
        exhibit_ids = [ctx.exhibits.table(["fact", "value", "reference"], fact_rows(report, table), title="Codec facts",
                                          media_ref="original", source_signal_ids=cite)]
        series = [{"name": "gop_frames", "points": [(round(table.seconds(k), 6), float(n)) for k, n in zip(gop.key_frames, gop.lengths, strict=True)][:5000]}]
        if gop.irregular or gop.at_cuts:
            lengths = dict(zip(gop.key_frames, gop.lengths, strict=True))
            previous = {b: lengths[a] for a, b in zip(gop.key_frames, gop.key_frames[1:], strict=False)}
            series.append({"name": "irregular_key_frames", "points": [(round(table.seconds(k), 6), float(previous[k])) for k in gop.irregular]})
            series.append({"name": "key_frames_at_shot_cuts", "points": [(round(table.seconds(k), 6), float(previous[k])) for k in gop.at_cuts]})
        if stamps.gaps:
            series.append({"name": "timestamp_gap_ms", "points": [(round(table.seconds(i), 6), round(g * 1000, 3)) for i, g in stamps.gaps][:5000]})
        exhibit_ids.append(ctx.exhibits.series(
            series, x_unit="s", title="GOP strip: GOP length at each key frame", media_ref="frame",
            thresholds=[float(gop.dominant)] if gop.dominant else [],
            caption=("Frames from each key frame to the next (the last GOP ends at the end of the file); irregular key frames are "
                     "plotted with the length of the GOP they end, and timestamp gaps in milliseconds. The threshold line is the "
                     "dominant GOP length."), source_signal_ids=cite))
        counts = {}
        for k, n in zip(gop.key_frames, gop.lengths, strict=True):
            types = table.pict_types[k:k + n]
            counts[k] = f"{types.count('I')} / {types.count('P')} / {types.count('B')}"
        notes = {k: "irregular" for k in gop.irregular} | {k: "irregular, at a shot cut" for k in gop.at_cuts}
        rows = [[float(k), round(table.seconds(k), 6), float(table.pts[k]), float(n), counts[k], notes.get(k, "")]
                for k, n in list(zip(gop.key_frames, gop.lengths, strict=True))[:MAX_KEY_ROWS]]
        exhibit_ids.append(ctx.exhibits.table(
            ["key frame", "time (s)", "pts", "GOP frames", "I / P / B", "note"], rows, media_ref="frame", source_signal_ids=cite,
            title="Key frames" + (f" (first {MAX_KEY_ROWS} of {len(gop.key_frames)})" if len(gop.key_frames) > MAX_KEY_ROWS else "")))
        if report.sei:
            exhibit_ids.append(ctx.exhibits.table(["encoder identification string"], [[s[:600]] for s in report.sei],
                                                  title="Encoder identification strings (SEI)", media_ref="original", source_signal_ids=cite))

        gop_text = ("GOP not established" if gop.dominant is None else
                    f"GOP {gop.dominant} frames" + (", regular" if gop.regular else ""))
        observed = "; ".join([f"{gop_text} over {len(gop.key_frames)} key frame(s)", *(x.observed for x in report.irregularities)]) \
            if report.irregularities else (f"{gop_text} over {len(gop.key_frames)} key frame(s); no codec irregularity")
        reasoning = (" ".join(x.reasoning for x in report.irregularities) if report.irregularities else
                     "No codec irregularity was found. Codec facts describe how the file was encoded and muxed; they do not show "
                     "whether the content was manipulated, and a re-encode removes most of them.")
        aggregate = (UnitScore("global", probability, "probability", quality={"outcome_weight": codec.OUTCOME_WEIGHT,
                                                                              "irregularity_kinds": float(len(report.irregularities))})
                     if probability is not None else None)
        return DetectorResult(
            status="ok", aggregate=aggregate, unit_scores=[aggregate] if aggregate else [], observed=observed, reference=REFERENCE,
            reasoning=reasoning, exhibit_ids=exhibit_ids,
            metrics={"frames": float(len(table)), "key_frames": float(len(gop.key_frames)), "gop_dominant_frames": float(gop.dominant or 0),
                     "gop_regular": float(gop.regular), "irregular_key_frames": float(len(gop.irregular)),
                     "key_frames_at_shot_cuts": float(len(gop.at_cuts)), "starts_mid_gop": float(gop.starts_mid_gop),
                     "b_frames": float(gop.b_frames), "timestamp_gaps": float(len(stamps.gaps)), "duplicate_pts": float(stamps.duplicates),
                     "non_monotonic_pts": float(stamps.non_monotonic), "vfr": float(stamps.vfr),
                     "packets_discarded": float(report.packets.discarded), "non_monotonic_dts": float(report.packets.non_monotonic_dts),
                     "edit_list_tracks": float(len(report.edits)), "encoder_strings": float(len(report.sei)),
                     "irregularity_kinds": float(len(report.irregularities)),
                     **({"av_start_delta_s": report.av_start_delta_s} if report.av_start_delta_s is not None else {}),
                     **({"av_duration_delta_s": report.av_duration_delta_s} if report.av_duration_delta_s is not None else {})})
