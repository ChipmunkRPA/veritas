"""Supplemental PDF for a comparison of two source recordings."""

from __future__ import annotations

import pymupdf

from app.evidence.audio_reference import (
    AUDIO_RATE,
    MAX_SECONDS,
    SCHEMA_WITH_LEVEL_ADJUSTMENT,
    SCHEMA_WITH_LOCAL_TIMING,
    SCHEMA_WITH_STEREO,
    SCHEMA_WITH_TRIM_ALIGNMENT,
    SCHEMA_WITH_TRIM_AND_LEVEL,
)
from app.evidence.audio_reference import SCHEMA as AUDIO_SCHEMA
from app.reports.image_reference_pdf import INK, MARGIN, MUTED, PAGE_HEIGHT, PAGE_WIDTH, RIGHT, RULE, _text

TITLE = "Earlier video comparison"
LABELS = {"earlier": "Earlier clip", "submitted": "Analyzed clip"}
PICTURE_STATUS = {
    "comparison_limit": "At least one clip exceeds the picture comparison limits.",
    "unverified_picture_clock": "The comparison could not verify every picture timestamp.",
    "picture_geometry_differs": "The clips have different display dimensions.",
    "picture_timelines_differ": "The clips have different relative picture timestamps.",
}
AUDIO_STATUS = {
    "no_stream": "Neither clip has a soundtrack.",
    "audio_unassessed": "Both soundtracks need a decodable continuous sample clock.",
    "audio_timelines_differ": "The soundtracks have different start times or sample counts.",
}
AUDIO_ONLY_STATUS = {
    "comparison_limit": f"At least one recording exceeds {MAX_SECONDS} seconds of decoded audio.",
    "no_stream": "The files have no readable audio track.",
    "audio_unassessed": "Both recordings need a readable, continuous audio sample clock.",
    "audio_timelines_differ": "The recordings have different start times or decoded sample counts.",
}


def _page(document, title):
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    _text(page, MARGIN, 51, "CHIPMUNK VERIFY", size=10, font="hebo")
    page.draw_line((MARGIN, 65), (RIGHT, 65), color=RULE)
    _text(page, MARGIN, 101, title, size=24, font="hebo")
    page.draw_line((MARGIN, 744), (RIGHT, 744), color=RULE)
    _text(page, MARGIN, 763, "Supplemental source comparison", size=9, color=MUTED)
    _text(page, RIGHT - 36, 763, f"Page {len(document)}", size=9, color=MUTED)
    return page


def _range(span):
    if span["end_s"] is None:
        return f"{span['start_s']:.3f} s to final picture at {span['last_s']:.3f} s"
    return f"{span['start_s']:.3f} to {span['end_s']:.3f} s"


def _audio_file_range(span):
    return f"{span['start_sample'] / AUDIO_RATE:.6f} to {span['end_sample'] / AUDIO_RATE:.6f} s"


def _summary(document, record):
    page = _page(document, TITLE)
    page.insert_textbox((MARGIN, 125, RIGHT, 184), record["interpretation"], fontsize=11, fontname="helv", color=INK)
    for y, (side, label) in zip((203, 265), LABELS.items(), strict=True):
        clip = record[side]
        _text(page, MARGIN, y, label, font="hebo")
        _text(page, MARGIN, y + 17, f"{clip['frames']} pictures, {clip['width']} x {clip['height']} px. Time base {clip['time_base']} s per tick.", color=MUTED)
        _text(page, MARGIN, y + 33, f"SHA-256 {clip['sha256']}", size=9, font="cour", color=MUTED)
    if record["status"] != "measured":
        _text(page, MARGIN, 348, "Comparison unassessed", size=16, font="hebo")
        _text(page, MARGIN, 369, PICTURE_STATUS[record["status"]])
        return
    picture, audio = record["picture"], record["audio"]
    _text(page, MARGIN, 347, "Picture comparison", size=16, font="hebo")
    _text(page, MARGIN, 371, f"{picture['matching_frames']} exact matches and {picture['changed_frames']} changed decoded pictures.")
    if not picture["matching_frames"]:
        _text(page, MARGIN, 390, "Every picture differs. Re-encoding can produce this pattern.", color=MUTED)
    _text(page, MARGIN, 434, "Soundtrack comparison", size=16, font="hebo")
    if audio["status"] == "measured":
        count = len(audio["ranges"])
        note = f"{audio['changed_samples']:,} decoded mono samples differ across {count} {'range' if count == 1 else 'ranges'}."
    else:
        note = AUDIO_STATUS[audio["status"]]
    _text(page, MARGIN, 458, note)
    _text(page, MARGIN, 507, "Times are relative to the first picture in each clip.", color=MUTED)
    _text(page, MARGIN, 526, "Soundtrack ranges group 20 ms bins containing different samples.", color=MUTED)
    local = picture.get("local_comparison")
    if local:
        _text(page, MARGIN, 566, "Local picture comparison", size=16, font="hebo")
        summary = f"Local picture difference at {_range(local['range'])}." if "range" in local else local["summary"]
        page.insert_textbox((MARGIN, 583, RIGHT, 670), summary + "\n" + local["scope"],
                            fontsize=10, fontname="helv", color=INK)


def _audio_summary(document, record):
    page = _page(document, "Earlier audio comparison")
    page.insert_textbox((MARGIN, 125, RIGHT, 198), record["interpretation"], fontsize=11, fontname="helv", color=INK)
    audio = record["audio"]
    for y, (side, label) in zip((221, 281), LABELS.items(), strict=True):
        _text(page, MARGIN, y, label, font="hebo")
        _text(page, MARGIN, y + 20, f"SHA-256 {record[side]['sha256']}", size=9, font="cour", color=MUTED)
    if record["status"] != "measured":
        _text(page, MARGIN, 346, "Comparison unassessed", size=16, font="hebo")
        _text(page, MARGIN, 370, AUDIO_ONLY_STATUS[record["status"]])
        return
    _text(page, MARGIN, 346, "Decoded sample comparison", size=16, font="hebo")
    alignment = audio.get("trim_alignment")
    total = alignment["compared_samples"] if alignment else audio["submitted"]["samples"]
    changed = audio["changed_samples"]
    count = len(audio["ranges"])
    _text(page, MARGIN, 370, f"{changed:,} of {total:,} mono samples differ across {count} {'range' if count == 1 else 'ranges'}.")
    if alignment:
        _text(page, MARGIN, 408, "Shared audio after sample alignment", size=13, font="hebo")
        for y, (side, label) in zip((431, 453), LABELS.items(), strict=True):
            span = alignment["overlap"][side]
            _text(page, MARGIN, y, f"{label}  {_audio_file_range(span)}  [{span['start_sample']}, {span['end_sample']})")
        if changed == 0:
            _text(page, MARGIN, 476, "The decoded mono samples match exactly inside the shared span.", color=MUTED)
        _text(page, MARGIN, 507, "Times use each file's first decoded sample. Source clocks use 16 kHz ticks.", color=MUTED)
        _text(page, MARGIN, 527, "Ranges group adjacent 20 ms bins containing at least one differing sample.", color=MUTED)
        _text(page, MARGIN, 547, "The JSON includes exact alignment samples, their checksums and playable excerpts.", color=MUTED)
        return
    if count == 1 and audio["ranges"][0]["start_sample"] == 0 and audio["ranges"][0]["end_sample"] == total:
        _text(page, MARGIN, 391, "Differences span the whole recording. Encoding or gain changes can cause this.", color=MUTED)
    elif changed == 0:
        _text(page, MARGIN, 391, "The decoded mono samples match exactly.", color=MUTED)
    _text(page, MARGIN, 442, "Times start at the first decoded sample. The source clocks use 16 kHz ticks.", color=MUTED)
    _text(page, MARGIN, 462, "Ranges group adjacent 20 ms bins containing at least one differing sample.", color=MUTED)
    _text(page, MARGIN, 482, "The JSON export includes aligned playable excerpts and their checksums.", color=MUTED)


def _level_adjusted_summary(document, result):
    page = _page(document, "Differences after a volume adjustment")
    db = result["gain_db"]
    direction = "quieter" if db < 0 else "louder"
    precision = 6 if abs(db) < .0005 else 3
    aligned = "file_ranges" in result["support"][0]
    basis = "in two supporting sections of the shared span" if aligned else "in the matching sections at both ends"
    _text(page, MARGIN, 131, f"The analyzed recording is {abs(db):.{precision}f} dB {direction} {basis}.")
    count = result["changed_samples"]
    if count == 0:
        note = "The same volume adjustment explains the shared span." if aligned else "The same volume adjustment explains the entire recording."
    else:
        note = f"{count:,} samples remain different after accounting for the volume adjustment."
    _text(page, MARGIN, 158, note, font="hebo")
    page.insert_textbox((MARGIN, 179, RIGHT, 280), result["interpretation"], fontsize=11, fontname="helv", color=INK)
    _text(page, MARGIN, 303, "Times start at each file's first decoded sample. Excerpts use the original levels." if aligned else
          "Times start at the first decoded sample. Waveform excerpts use the original levels.", color=MUTED)
    _text(page, MARGIN, 324, "The JSON export includes the supporting sample ranges and playable excerpts.", color=MUTED)


def _picture_page(document, span, assets, source):
    local = "tile" in span
    title = "Local picture difference" if local else "Picture differences"
    page = _page(document, f"{span['id']}  {title}")
    _text(page, MARGIN, 124, _range(span), color=MUTED)
    if local:
        box = span["tile"]
        label = "Measured comparison area at"
        _text(page, MARGIN, 160, "Each pair includes the source picture and its native crop.", color=MUTED)
    else:
        row = span["strongest_frame"]
        box = row["difference_bounds"]
        label = f"At the strongest change, {row['changed_pixels']:,} pixels differ within"
    _text(page, MARGIN, 143, f"{label} x {box['x']}, y {box['y']}, {box['w']} x {box['h']} px.")
    y = 178
    for position in ("before", "during", "after"):
        sample = span["previews"].get(position)
        if sample is None:
            continue
        _text(page, MARGIN, y, f"{position.title()}  {sample['time_s']:.3f} s", font="hebo")
        for x, side in zip((MARGIN, 316), LABELS, strict=True):
            _text(page, x, y + 15, f"{LABELS[side]}  PTS {sample[f'{side}_pts']}", size=9, color=MUTED)
            available = 146 if local else 250
            scale = min(available / source["width"], 126 / source["height"])
            width, height = source["width"] * scale, source["height"] * scale
            left, top = x + (available - width) / 2, y + 22 + (126 - height) / 2
            page.insert_image(pymupdf.Rect(left, top, left + width, top + height),
                              stream=assets[sample["assets"][side]["file"]], keep_proportion=False)
            if position == "during" or local:
                page.draw_rect(pymupdf.Rect(left + box["x"] * scale, top + box["y"] * scale,
                                           left + (box["x"] + box["w"]) * scale, top + (box["y"] + box["h"]) * scale),
                               color=(.96, .67, .15), width=1.4)
            if local:
                crop_scale = min(96 / box["w"], 112 / box["h"])
                crop_w, crop_h = box["w"] * crop_scale, box["h"] * crop_scale
                crop_x, crop_y = x + 152 + (96 - crop_w) / 2, y + 22 + (126 - crop_h) / 2
                page.insert_image(pymupdf.Rect(crop_x, crop_y, crop_x + crop_w, crop_y + crop_h),
                                  stream=assets[sample["crop_assets"][side]["file"]], keep_proportion=False)
        y += 168


def _draw_waveform(page, wave, y, maximum, duration):
    page.draw_line((MARGIN, y), (RIGHT, y), color=RULE)
    shape = page.new_shape()
    width = RIGHT - MARGIN
    half_bin = wave["bin_s"] / 2 / duration * width
    for tick, low, high in zip(wave["t"], wave["low"], wave["high"], strict=True):
        x = MARGIN + min(1, tick / duration) * width
        high_y, low_y = y - high / maximum * 44, y - low / maximum * 44
        if low == high:
            shape.draw_line((max(MARGIN, x - half_bin), high_y), (min(RIGHT, x + half_bin), low_y))
        else:
            shape.draw_line((x, high_y), (x, low_y))
    shape.finish(color=(.18, .39, .31), width=.7)
    shape.commit()


def _timing_page(document, timing, assets):
    page = _page(document, "Marked mouth and speech timing")
    page.insert_textbox((MARGIN, 124, RIGHT, 167), timing["summary"], fontsize=12, fontname="hebo", color=INK)
    short_excerpt = timing.get("waveform_basis") == "short_marked_excerpt"
    scope_bottom, interval_y = (226, 236) if short_excerpt else (213, 221)
    page.insert_textbox((MARGIN, 176, RIGHT, scope_bottom), timing["scope"], fontsize=10, fontname="helv", color=MUTED)
    if timing["status"] == "unassessed":
        return
    low, high = timing["event_range_s"]
    _text(page, MARGIN, interval_y, f"Compared interval in analyzed clip  {low:.3f} to {high:.3f} s", color=MUTED)
    for x, (side, label) in zip((MARGIN, 316), LABELS.items(), strict=True):
        row = timing[side]
        _text(page, x, 253, label, font="hebo")
        _text(page, x, 272, f"Native picture {row['frame_index']} at {row['picture_time_s']:.3f} s", color=MUTED)
        _text(page, x, 290, f"PTS {row['picture_pts']} at {row['picture_time_base']} s per tick", size=9, color=MUTED)
        page.insert_image(pymupdf.Rect(x, 304, x + 232, 416), stream=assets[row["crop"]["file"]])
    maximum = max(.01, *(abs(level) for side in LABELS for field in ("low", "high")
                        for level in timing[side]["waveform"][field]))
    for y, (side, label) in zip((505, 638), LABELS.items(), strict=True):
        row = timing[side]
        start, stop = row["pcm_samples"]
        _text(page, MARGIN, y - 64, f"{label} speech  {row['pcm_start_s']:.3f} to {row['pcm_end_s']:.3f} s", font="hebo")
        _draw_waveform(page, row["waveform"], y, maximum, (stop - start) / AUDIO_RATE)
        _text(page, MARGIN, y + 53, f"Decoded samples [{start}, {stop}) at 16 kHz.", size=9, color=MUTED)
    _text(page, MARGIN, 721, "Waveforms share one scale. The JSON includes both playable speech excerpts.", size=9, color=MUTED)


def _soundtrack_timing_page(document, timing, assets):
    page = _page(document, "Local soundtrack timing")
    page.insert_textbox((MARGIN, 124, RIGHT, 165), timing["summary"], fontsize=12, fontname="hebo", color=INK)
    page.insert_textbox((MARGIN, 169, RIGHT, 217), timing["scope"], fontsize=10, fontname="helv", color=MUTED)
    if timing["status"] != "local_shift":
        return
    span = timing["range"]
    picture = span["picture"]
    _text(page, MARGIN, 235, f"Analyzed clip picture {picture['frame_index']} at {picture['time_s']:.3f} s", font="hebo")
    _text(page, MARGIN, 252, f"PTS {picture['pts']} at {picture['time_base']} s per tick", size=9, color=MUTED)
    page.insert_image(pymupdf.Rect(MARGIN, 263, MARGIN + 260, 428), stream=assets[picture["asset"]["file"]])
    maximum = max(.01, *(abs(level) for side in LABELS for field in ("low", "high")
                        for level in span[side]["waveform"][field]))
    for y, (side, label) in zip((503, 635), LABELS.items(), strict=True):
        row = span[side]
        first, end = row["samples"]
        _text(page, MARGIN, y - 61, f"{label}  {row['start_s']:.3f} to {row['end_s']:.3f} s", font="hebo")
        _draw_waveform(page, row["waveform"], y, maximum, (end - first) / timing["sample_rate"])
        _text(page, MARGIN, y + 51, f"Decoded samples [{first}, {end}) at 16 kHz.", size=9, color=MUTED)
    _text(page, MARGIN, 720, "Waveforms share one scale. The JSON includes both playable excerpts.", size=9, color=MUTED)


def _audio_local_timing_page(document, timing):
    page = _page(document, "Local audio timing")
    page.insert_textbox((MARGIN, 124, RIGHT, 167), timing["summary"], fontsize=12, fontname="hebo", color=INK)
    page.insert_textbox((MARGIN, 176, RIGHT, 231), timing["scope"], fontsize=11, fontname="helv", color=MUTED)
    if timing["status"] != "local_shift":
        return
    span = timing["range"]
    maximum = max(.01, *(abs(level) for side in LABELS for field in ("low", "high")
                        for level in span[side]["waveform"][field]))
    rows = (("earlier", "Earlier recording"), ("submitted", "Analyzed recording"))
    for y, (side, label) in zip((343, 512), rows, strict=True):
        row = span[side]
        first, end = row["samples"]
        _text(page, MARGIN, y - 65, f"{label}  {row['start_s']:.3f} to {row['end_s']:.3f} s", font="hebo")
        _draw_waveform(page, row["waveform"], y, maximum, (end - first) / timing["sample_rate"])
        _text(page, MARGIN, y + 55, f"Decoded samples [{first}, {end}) at 16 kHz.", size=9, color=MUTED)
    _text(page, MARGIN, 628, f"{span['supporting_windows']} matching windows support the time difference.", color=MUTED)
    _text(page, MARGIN, 650, "Waveforms share one scale. The JSON includes both playable excerpts.", size=9, color=MUTED)


def _sound_page(document, span, *, level_adjusted=False, track_title=None):
    difference = span.get("difference")
    if level_adjusted:
        title = "Volume-adjusted difference"
        count_note = "samples cannot be explained by the measured volume adjustment."
    else:
        title = track_title or ("Audio differences" if difference is not None else "Soundtrack differences")
        count_note = "differing samples in the selected range."
    page = _page(document, f"{span['id']}  {title}")
    aligned = span.get("file_ranges")
    _text(page, MARGIN, 126, "Analyzed clip  " + _audio_file_range(aligned["submitted"]) if aligned else _range(span), color=MUTED)
    _text(page, MARGIN, 149, f"{span['changed_samples']:,} {count_note}")
    preview = span["preview"]
    if aligned:
        _text(page, MARGIN, 169, f"Earlier clip  {_audio_file_range(aligned['earlier'])}", color=MUTED)
    else:
        _text(page, MARGIN, 169, f"Waveforms cover {span['start_s']:.3f} to {preview['end_s']:.3f} s.")
    if "first_differing_sample" in span:
        _text(page, MARGIN, 188, f"First and last differing samples  {span['first_differing_sample']:,} and {span['last_differing_sample']:,}.",
              size=9, color=MUTED)
    waveforms = preview["waveforms"]
    residual = preview.get("difference_waveform")
    maximum = max(.01, *(abs(level) for values in waveforms.values() for field in ("low", "high") for level in values[field]))
    count = preview["end_sample"] - preview["start_sample"]
    duration = count / AUDIO_RATE
    for y, (side, label) in zip((268, 422), LABELS.items(), strict=True):
        _text(page, MARGIN, y - 59, label, font="hebo")
        _draw_waveform(page, waveforms[side], y, maximum, duration)
        pts = span["source_ranges"][side]["start_pts"]
        _text(page, MARGIN, y + 58, f"Source ticks [{pts}, {pts + count}) at 16 kHz.", size=9, color=MUTED)
        if aligned:
            start = aligned[side]["start_sample"]
            times = _audio_file_range({"start_sample": start, "end_sample": start + count})
            _text(page, MARGIN, y + 71, f"Waveform covers {times} in this file.", size=9, color=MUTED)
    if residual is not None:
        label = "Volume-adjusted difference waveform" if level_adjusted else "Difference waveform"
        _text(page, MARGIN, 517, label, font="hebo")
        residual_scale = max(1 / 32768, *(abs(level) for field in ("low", "high") for level in residual[field]))
        _draw_waveform(page, residual, 576, residual_scale, duration)
        earlier = "volume-adjusted earlier samples" if level_adjusted else "earlier samples"
        _text(page, MARGIN, 632, f"Analyzed minus {earlier}. This plot uses a separate scale of +/-{residual_scale:.6f}.",
              size=9, color=MUTED)
        note = "Adjusted difference" if level_adjusted else "Full range difference"
        _text(page, MARGIN, 650, f"{note}: peak {difference['peak']:.6f}, RMS {difference['rms']:.6f} of full scale.", size=9, color=MUTED)


def _remaining_ranges(document, remaining, title="Additional measured ranges"):
    for start in range(0, len(remaining), 30):
        page = _page(document, title)
        for i, (label, value) in enumerate(remaining[start:start + 30]):
            _text(page, MARGIN, 137 + i * 18, f"{label}  {value}")


def _level_adjusted_pages(document, result):
    _level_adjusted_summary(document, result)
    remaining = []
    for span in result["ranges"]:
        if span.get("preview"):
            _sound_page(document, span, level_adjusted=True)
        else:
            remaining.append((span["id"], f"{_sound_range_label(span)} - {span['changed_samples']:,} differing samples"))
    _remaining_ranges(document, remaining, "Additional volume-adjusted ranges")


def _sound_range_label(span):
    ranges = span.get("file_ranges")
    if ranges:
        return f"{_audio_file_range(ranges['earlier'])} earlier, {_audio_file_range(ranges['submitted'])} analyzed"
    return _range(span)


def _additional_soundtrack_pages(document, track):
    page = _page(document, "Second soundtrack comparison")
    page.insert_textbox((MARGIN, 125, RIGHT, 202), track["scope"], fontsize=11, fontname="helv", color=INK)
    for y, (side, label) in zip((220, 285), LABELS.items(), strict=True):
        index = track["stream_indices"][side]
        _text(page, MARGIN, y, label, font="hebo")
        _text(page, MARGIN, y + 19, "No second soundtrack." if index is None else f"Second soundtrack is source stream {index}.")
        count = len(track["inventory"][side]["unassessed_stream_indices"])
        if count:
            label = "track" if count == 1 else "tracks"
            _text(page, MARGIN, y + 37, f"Unassessed beyond the second soundtrack: {count} {label}.", color=MUTED)
    if track["status"] != "measured":
        extra_status = {"missing_counterpart": "The second track has no counterpart.",
                        "comparison_limit": "At least one second soundtrack exceeds 12 seconds."}
        note = extra_status.get(track["status"]) or AUDIO_STATUS.get(track["status"], "The second tracks could not be compared.")
        _text(page, MARGIN, 365, note)
        return
    _text(page, MARGIN, 365, f"{track['changed_samples']:,} decoded mono samples differ on the second soundtrack.")
    _text(page, MARGIN, 385, "Times are relative to the first picture in each clip.", color=MUTED)
    remaining = []
    for span in track["ranges"]:
        if span.get("preview"):
            _sound_page(document, span, track_title="Second soundtrack differences")
        else:
            remaining.append((span["id"], f"{_range(span)} - {span['changed_samples']:,} differing samples"))
    _remaining_ranges(document, remaining, "Additional second soundtrack ranges")


def _stereo_pages(document, comparison):
    page = _page(document, "Left and right channel comparison")
    page.insert_textbox((MARGIN, 125, RIGHT, 200), comparison["scope"], fontsize=11, fontname="helv", color=INK)
    if comparison["status"] != "measured":
        notes = {**AUDIO_ONLY_STATUS, "stereo_pair_required": "Both copies need a recorded left and right channel.",
                 "audio_unassessed": "Both stereo streams need verifiable, continuous sample clocks."}
        _text(page, MARGIN, 225, notes[comparison["status"]])
        return
    for y, (side, label) in zip((222, 283), LABELS.items(), strict=True):
        source = comparison["sources"][side]
        _text(page, MARGIN, y, f"{label} · source stream {source['stream_index']} at {source['sample_rate']:,} Hz", font="hebo")
        _text(page, MARGIN, y + 19, "Decoded stereo SHA-256 " + source["stereo_pcm_sha256"], size=9, font="cour", color=MUTED)
    for y, (label, result) in zip((365, 392), comparison["channels"].items(), strict=True):
        _text(page, MARGIN, y, f"{label.capitalize()} channel · {result['changed_samples']:,} decoded samples differ.")
    for label, result in comparison["channels"].items():
        remaining = []
        for span in result["ranges"]:
            if span.get("preview"):
                _sound_page(document, span, track_title=f"{label.capitalize()} channel differences")
            else:
                remaining.append((span["id"], f"{_range(span)} - {span['changed_samples']:,} differing samples"))
        _remaining_ranges(document, remaining, f"Additional {label} channel ranges")


def render(record, assets):
    with pymupdf.open() as document:
        is_audio = record["schema"] in (AUDIO_SCHEMA, SCHEMA_WITH_LEVEL_ADJUSTMENT, SCHEMA_WITH_TRIM_ALIGNMENT,
                                        SCHEMA_WITH_TRIM_AND_LEVEL, SCHEMA_WITH_STEREO, SCHEMA_WITH_LOCAL_TIMING)
        document.set_metadata({"title": "Earlier audio comparison" if is_audio else TITLE, "author": "Chipmunk Verify"})
        adjusted = record.get("level_adjusted")
        trimmed = record["schema"] == SCHEMA_WITH_TRIM_AND_LEVEL or (
            record["schema"] == SCHEMA_WITH_LOCAL_TIMING and "trim_alignment" in record.get("audio", {}))
        if adjusted and not trimmed:
            _level_adjusted_pages(document, adjusted)
        (_audio_summary if is_audio else _summary)(document, record)
        if adjusted and trimmed:
            _level_adjusted_pages(document, adjusted)
        remaining = []
        local = record.get("picture", {}).get("local_comparison", {})
        if local.get("range"):
            _picture_page(document, local["range"], assets, record["submitted"])
        for span in record.get("picture", {}).get("ranges", []):
            if span["previews"]:
                _picture_page(document, span, assets, record["submitted"])
            else:
                remaining.append((span["id"], _range(span)))
        for span in record.get("audio", {}).get("ranges", []):
            if span.get("preview"):
                _sound_page(document, span)
            else:
                note = f" - {span['changed_samples']:,} differing samples" if is_audio else ""
                remaining.append((span["id"], _sound_range_label(span) + note))
        _remaining_ranges(document, remaining)
        if "timing" in record:
            _timing_page(document, record["timing"], assets)
        if "soundtrack_alignment" in record:
            _soundtrack_timing_page(document, record["soundtrack_alignment"], assets)
        if "additional_soundtrack" in record:
            _additional_soundtrack_pages(document, record["additional_soundtrack"])
        if "stereo_comparison" in record:
            _stereo_pages(document, record["stereo_comparison"])
        if "local_timing" in record:
            _audio_local_timing_page(document, record["local_timing"])
        return document.tobytes(deflate=True, garbage=4)
