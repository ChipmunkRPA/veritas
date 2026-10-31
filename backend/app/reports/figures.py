"""Deterministic figures for the report templates (ARCHITECTURE §6.2 `figures.py`, §6.3 rule 6; UX §6.2, §7.4, §8.4).

- `scale_svg`: the five-step verdict scale with the plausible range and the estimate marker, as the UI's `ScaleInterval` draws it.
- `icon_array_svg`: a 10×10 natural-frequency icon array: ink dots filled, `border-strong` rings open, rows of 10 left to right.
- `thumbnails`: preview artefacts from the store, re-encoded by PIL with fixed parameters and no metadata, as data URIs.

Colours are read from the generated `tokens.css`; every coordinate is written with fixed precision, so the same inputs give the
same bytes.
"""
from __future__ import annotations

import base64
import hashlib
import io
import math
import re
from functools import cache
from pathlib import Path

from markupsafe import Markup, escape
from PIL import Image, ImageOps

from app.reports.scalar_heatmap import scalar_map_plate as scalar_map_plate
from app.reports.scalar_heatmap import scalar_overlay_plate as scalar_overlay_plate
from app.storage.objects import InvalidObjectId, ObjectNotFound, ObjectStore, default_store

TOKENS_CSS = Path(__file__).resolve().parent / "templates" / "tokens.css"
THUMBNAIL_BOX = (354, 260)  # 30 × 22 mm at 300 dpi (UX §6.3 R3)
_DECL = re.compile(r"--([A-Za-z0-9-]+):\s*([^;]+);")


@cache
def tokens() -> dict[str, str]:
    return dict(_DECL.findall(TOKENS_CSS.read_text(encoding="utf-8")))


def _n(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def scale_svg(step: int, estimate: float | None, low: float | None, high: float | None, *, band: bool = False,
              label: str | None = None) -> Markup:
    """Estimates, range ends and the 15/35/65/85 band boundaries share one 0–100 axis. Step 0 (insufficient) draws an empty track without a marker; a capped decision without an
    estimate (D-49) draws the same track, labelled "No estimate". `band` (D-68, the viewer's `ScaleInterval band`) is the form of
    a result whose number is not a calibrated point estimate: no marker; the plausible range where one is given, else the
    verdict's segment filled; `label` says the number as the report prints it, never "Estimate n of 100" for a score."""
    t = tokens()
    width, gap, track_y, track_h = 1000.0, 6.0, 12.0, 12.0
    boundaries = (0, 15, 35, 65, 85, 100)  # app.core.evidence.band_for and the viewer ScaleInterval
    banded = band and step > 0
    insufficient = not banded and (step == 0 or estimate is None)
    ranged = not insufficient and low is not None and high is not None

    def x(v: float) -> float:
        return max(0.0, min(100.0, v)) / 100.0 * width

    if label is None:
        label = (("Insufficient evidence: no estimate" if step == 0 else "No estimate") if insufficient or banded
                 else f"Estimate {round(estimate)} of 100")  # type: ignore[arg-type]
    parts = [f'<svg class="scale" viewBox="0 0 1000 36" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(label)}">']
    for n in range(1, 6):
        start, end = x(boundaries[n - 1]), x(boundaries[n])
        segment = end - start - (gap if n < 5 else 0)
        filled = banded and not ranged and n == step
        fill, stroke = ((t["color-surface-2"], t["color-border-strong"]) if insufficient
                        else (t[f"color-verdict-{n}" if filled else f"color-verdict-{n}-tint"], t[f"color-verdict-{n}"]))
        parts.append(f'<rect x="{_n(start)}" y="{_n(track_y)}" width="{_n(segment)}" height="{_n(track_h)}" rx="2" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    if ranged:
        a, b = x(min(low, high)), x(max(low, high))  # type: ignore[type-var]
        parts.append(f'<rect x="{_n(a)}" y="{_n(track_y + 4)}" width="{_n(b - a)}" height="4" fill="{t["color-graphic"]}"/>')
        for edge in (a, b):
            parts.append(f'<rect x="{_n(edge - 1)}" y="{_n(track_y - 5)}" width="2" height="22" fill="{t["color-graphic"]}"/>')
    if not insufficient and not banded:
        parts.append(f'<rect class="scale-marker" x="{_n(x(estimate) - 2.5)}" y="2" width="5" height="32" fill="{t["color-text"]}"/>')  # type: ignore[arg-type]
    parts.append("</svg>")
    return Markup("".join(parts))


def icon_array_svg(filled: int) -> Markup:
    t = tokens()
    k = max(0, min(100, int(filled)))
    dots = []
    for i in range(100):
        cx, cy = 5 + 10 * (i % 10), 5 + 10 * (i // 10)
        dots.append(f'<circle cx="{cx}" cy="{cy}" r="3.6" fill="{t["color-text"]}"/>' if i < k
                    else f'<circle cx="{cx}" cy="{cy}" r="3.2" fill="none" stroke="{t["color-border-strong"]}" stroke-width="1"/>')
    return Markup(f'<svg class="icon-array" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" role="img" '
                  f'aria-label="{k} of 100 filled">{"".join(dots)}</svg>')


def reliability_svg(bins: list[tuple[int, float, float]]) -> Markup:
    """The 15 recorded weighted bin means on fixed 0-1 axes; no interpolation or invented bin intervals."""
    if ([b[0] for b in bins] != list(range(15))
            or any(isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1 for _, x, y in bins for v in (x, y))):
        raise ValueError("invalid reliability points")
    t = tokens()
    parts = ['<svg class="reliability-diagram" viewBox="0 0 310 292" xmlns="http://www.w3.org/2000/svg" '
             'role="img" aria-label="Historical calibration: 15 class-balanced AI/manipulated bin means against the equal-score diagonal">',
             '<title>Historical unclamped benchmark scores; weighted AI/manipulated fraction by recorded bin</title>']
    for value in (0, 0.25, 0.5, 0.75, 1):
        x, y = 54 + 220 * value, 242 - 220 * value
        label = _n(value)
        parts.extend([
            f'<line x1="54" y1="{_n(y)}" x2="274" y2="{_n(y)}" stroke="{t["color-border"]}" stroke-width="0.6"/>',
            f'<text x="{_n(x)}" y="260" text-anchor="middle" font-size="11" fill="{t["color-text-muted"]}">{label}</text>',
            f'<text x="45" y="{_n(y + 4)}" text-anchor="end" font-size="11" fill="{t["color-text-muted"]}">{label}</text>',
        ])
    parts.extend([
        f'<path d="M54 22 V242 H274" fill="none" stroke="{t["color-graphic"]}" stroke-width="1"/>',
        f'<line x1="54" y1="242" x2="274" y2="22" stroke="{t["color-graphic"]}" stroke-width="1.2" stroke-dasharray="5 4"/>',
        f'<text x="164" y="281" text-anchor="middle" font-size="11" fill="{t["color-text"]}">Mean raw fuser score</text>',
        f'<text transform="translate(14 132) rotate(-90)" text-anchor="middle" font-size="11" '
        f'fill="{t["color-text"]}">Weighted AI/manipulated fraction</text>',
    ])
    for index, mean_s, mean_y in bins:
        parts.append(f'<rect data-bin="{index}" x="{_n(54 + 220 * mean_s - 2.5)}" '
                     f'y="{_n(242 - 220 * mean_y - 2.5)}" width="5" height="5" fill="{t["color-text"]}">'
                     f'<title>Bin {index + 1}: mean score {mean_s:.6f}; weighted AI/manipulated fraction {mean_y:.6f}</title></rect>')
    parts.append('</svg>')
    return Markup(''.join(parts))


def calibration_support_svg(counts: list[int]) -> Markup:
    """Unweighted sample counts in the 15 retained ordinal groups, not score-density bins."""
    if len(counts) != 15 or any(type(n) is not int or n <= 0 for n in counts):
        raise ValueError("invalid calibration group counts")
    t = tokens()
    maximum = max(counts)
    parts = ['<svg class="calibration-support" viewBox="0 0 310 100" xmlns="http://www.w3.org/2000/svg" '
             'role="img" aria-label="Recorded samples by score group, unweighted">',
             '<title>Recorded samples by score group (unweighted)</title>',
             '<desc>Equal-width bars identify the 15 retained groups in ascending score order. '
             'Height shows recorded sample count, not score density or uncertainty. Exact counts follow in the table.</desc>',
             f'<text x="54" y="13" font-size="11" fill="{t["color-text"]}">Samples (unweighted)</text>',
             f'<path d="M54 26 V58 H274" fill="none" stroke="{t["color-graphic"]}"/>',
             f'<text x="45" y="30" text-anchor="end" font-size="11" fill="{t["color-text-muted"]}">{maximum}</text>',
             f'<text x="45" y="62" text-anchor="end" font-size="11" fill="{t["color-text-muted"]}">0</text>']
    for index, count in enumerate(counts):
        height = 32 * count / maximum
        x = 54 + 220 * (index + 0.5) / 15
        parts.append(f'<rect data-group="{index}" data-count="{count}" x="{_n(x - 5)}" '
                     f'y="{_n(58 - height)}" width="10" height="{_n(height)}" fill="{t["color-text"]}">'
                     f'<title>Group {index + 1}: {count} recorded samples</title></rect>')
        if index in (0, 4, 9, 14):
            parts.append(f'<text x="{_n(x)}" y="75" text-anchor="middle" font-size="11" '
                         f'fill="{t["color-text-muted"]}">{index + 1}</text>')
    parts.append(f'<text x="164" y="95" text-anchor="middle" font-size="11" '
                 f'fill="{t["color-text"]}">Score group (ordered)</text></svg>')
    return Markup(''.join(parts))


def seal_svg(*, outline: bool = False, muted: bool = False, label: str = "documentary") -> Markup:
    """The ◆ documentary seal, drawn (the glyph is outside the bundled font subsets). N-19n (D3-07): the two diamonds are one
    drawing at one size: filled for a record that was validated, in `outline` for a record that anyone can write (a
    self-declared marker), and in the muted ink (`muted`) where the decision did not apply the record. `label`: the
    accessible name, the strength word of the finding."""
    ink = tokens()["color-text-muted" if muted else "color-text"]
    paint = f'fill="none" stroke="{ink}" stroke-width="1"' if outline else f'fill="{ink}"'
    # the outline is drawn inside the filled diamond's edge, so both have one size
    path = "M5 1.3 L8.7 5 L5 8.7 L1.3 5 Z" if outline else "M5 0.6 L9.4 5 L5 9.4 L0.6 5 Z"
    return Markup(f'<svg class="seal" viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{label}">'
                  f'<path d="{path}" {paint}/></svg>')


def marks() -> dict[str, Markup]:
    """The mark that stands in a meter's place for a finding that is a record (`view.finding_mark`)."""
    return {"documentary": seal_svg(), "documentary_muted": seal_svg(muted=True),
            "self_declared": seal_svg(outline=True, label="self-declared"),
            "not_applied": seal_svg(outline=True, muted=True, label="not applied"),
            "invalid": seal_svg(outline=True, muted=True, label="invalid")}


def thumbnail(sha256: str, store: ObjectStore | None = None) -> str | None:
    """A stored preview as a PNG data URI (nearest-neighbour downscale, no metadata), or None when missing or altered."""
    return plate(sha256, THUMBNAIL_BOX, store=store)


def plate(sha256: str, box: tuple[int, int], *, photo: bool = False, store: ObjectStore | None = None) -> str | None:
    """A stored image as a data URI within `box` px: maps and previews as PNG with nearest-neighbour scaling (a map's cells stay
    visible), a source photograph as JPEG (quality 88, 4:4:4, baseline) with Lanczos scaling. No metadata is copied. None when
    the object is missing, altered or not a readable image."""
    store = store or default_store()
    try:
        data = store.read_bytes(sha256)
    except (ObjectNotFound, InvalidObjectId):
        return None
    if hashlib.sha256(data).hexdigest() != sha256:
        return None
    try:
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    image.thumbnail(box, Image.Resampling.LANCZOS if photo else Image.Resampling.NEAREST)
    out = io.BytesIO()
    if photo:
        image.save(out, "JPEG", quality=88, subsampling=0, optimize=False, progressive=False)
        return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")
    image.save(out, "PNG", optimize=False, compress_level=6)
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


# --- forensic figures (UX §8.4; R-3). Row figures are one SVG per table row, so labels stay HTML text and rows align. -----------

def _x(v: float, domain: float) -> float:
    return (max(-domain, min(domain, v)) + domain) / (2 * domain) * 1000.0


def waterfall_row_svg(kind: str, start: float, end: float, domain: float, slight: bool = False) -> Markup:
    """One waterfall row on the log-odds axis ±`domain`: a bar from the running total `start` to `end` (toward indications in
    `dir-ai`, toward no indications in `dir-authentic`, adjustments outlined and dashed), a connector at the start, the 50 line,
    and an ink marker for the starting point and the estimate. `slight` (N-19n, D3-10; the page since N-19k): a check that
    had little effect is a thin bar in the neutral ink; it was a full bar in the colour of its side beside "had little effect"."""
    t = tokens()
    parts = ['<svg class="wf-row" viewBox="0 0 1000 20" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" role="img">',
             f'<rect x="499.5" y="0" width="1" height="20" fill="{t["color-graphic"]}"/>']
    a, b = _x(start, domain), _x(end, domain)
    if kind in ("start", "estimate"):
        parts.append(f'<rect x="{_n(b - 2)}" y="3" width="4" height="14" fill="{t["color-text"]}"/>')
    elif kind == "blank":  # N-17 (the number rule of D-68): the last row of a result that shows no marker
        pass
    elif kind == "flat":
        # N-19k (D2-03): a row of a result outside what was validated. Every bar starts at the one origin (the 0 line) and is
        # in one ink: no bar stands on a running total, so no estimate is read off where the last one ends
        lo, hi = min(a, b), max(a, b)
        y, height = (8, 4) if slight else (5, 10)
        parts.append(f'<rect x="{_n(lo)}" y="{y}" width="{_n(max(hi - lo, 2.0))}" height="{height}" fill="{t["color-dir-neutral"]}"/>')
    else:
        lo, hi = min(a, b), max(a, b)
        width = max(hi - lo, 2.0)
        parts.append(f'<rect x="{_n(a - 0.5)}" y="0" width="1" height="20" fill="{t["color-graphic"]}"/>')
        if kind == "adjustment":
            parts.append(f'<rect x="{_n(lo)}" y="5" width="{_n(width)}" height="10" fill="none" stroke="{t["color-graphic"]}" '
                         f'stroke-width="1.5" stroke-dasharray="4 3"/>')
        elif slight:
            parts.append(f'<rect x="{_n(lo)}" y="8" width="{_n(width)}" height="4" fill="{t["color-dir-neutral"]}"/>')
        else:
            fill = t["color-dir-ai"] if end >= start else t["color-dir-authentic"]
            parts.append(f'<rect x="{_n(lo)}" y="5" width="{_n(width)}" height="10" fill="{fill}"/>')
    parts.append("</svg>")
    return Markup("".join(parts))


def axis_svg(ticks: list[float], domain: float) -> Markup:
    """Tick marks of the waterfall axis at the given log-odds values."""
    t = tokens()
    marks = "".join(f'<rect x="{_n(_x(v, domain) - 0.5)}" y="4" width="1" height="6" fill="{t["color-graphic"]}"/>' for v in ticks)
    return Markup(f'<svg class="wf-axis" viewBox="0 0 1000 10" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">'
                  f'<rect x="0" y="9" width="1000" height="1" fill="{t["color-graphic"]}"/>{marks}</svg>')


def agreement_svg(dots: list[tuple[float, str, int]], estimate: float | None) -> Markup:
    """Agreement strip: (standalone p on 0–1, direction ai | authentic | neutral, label row 0–2) per scored check; toward = filled
    `dir-ai`, against = ring `dir-authentic`, little effect = small muted dot; the estimate as a 4×20 ink marker."""
    t = tokens()
    # Eight units clear the widest marker (ring radius 5.2 + 1 stroke). Matching CSS keeps the 0–100 axis in place.
    parts = ['<svg class="agreement" viewBox="-8 0 1016 64" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" role="img">',
             f'<rect x="0" y="52" width="1000" height="1" fill="{t["color-graphic"]}"/>']
    for v in (0, 500, 999):
        parts.append(f'<rect x="{v}" y="48" width="1" height="9" fill="{t["color-graphic"]}"/>')
    for p, direction, row in dots:
        cx, cy = _n(p * 1000), _n(40 - 12 * row)
        if direction == "ai":
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="{t["color-dir-ai"]}"/>')
        elif direction == "authentic":
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="5.2" fill="none" stroke="{t["color-dir-authentic"]}" stroke-width="2"/>')
        else:
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="2.5" fill="{t["color-text-muted"]}"/>')
    if estimate is not None:
        parts.append(f'<rect x="{_n(estimate * 1000 - 2)}" y="44" width="4" height="20" fill="{t["color-text"]}"/>')
    parts.append("</svg>")
    return Markup("".join(parts))


def lane_svg(segments: list[tuple[float, float, int]], duration: float) -> Markup:
    """One timeline lane: (start s, end s, verdict step 1–5) per segment over 0–`duration`; steps 1–3 tint, 4–5 solid with a
    `lane-flag` top rule; a minimum drawn width; the lane frame at the bottom."""
    t = tokens()
    span = duration if duration > 0 else 1.0
    parts = ['<svg class="lane" viewBox="0 0 1000 28" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" role="img">',
             f'<rect x="0" y="0" width="1000" height="27" fill="{t["color-lane"]}"/>',
             f'<rect x="0" y="27" width="1000" height="1" fill="{t["color-lane-frame"]}"/>']
    for start, end, step in segments:
        a = max(0.0, min(1000.0, start / span * 1000))
        width = max(4.0, min(1000.0, end / span * 1000) - a)
        s = max(1, min(5, step))
        fill = t[f"color-verdict-{s}-tint"] if s <= 3 else t[f"color-verdict-{s}"]
        parts.append(f'<rect x="{_n(a)}" y="4" width="{_n(width)}" height="23" fill="{fill}"/>')
        if s >= 4:
            parts.append(f'<rect x="{_n(a)}" y="1" width="{_n(width)}" height="3" fill="{t["color-lane-flag"]}"/>')
    parts.append("</svg>")
    return Markup("".join(parts))


def series_svg(lines: list[list[tuple[float, float]]], x_range: tuple[float, float], y_range: tuple[float, float],
               thresholds: list[float]) -> Markup:
    """Inline series as 1.5-unit polylines (ink, then `series-face`, `series-general`), thresholds dashed, in fixed ranges."""
    t = tokens()
    (x0, x1), (y0, y1) = x_range, y_range
    dx, dy = (x1 - x0) or 1.0, (y1 - y0) or 1.0

    def point(x: float, y: float) -> str:
        return f"{_n((x - x0) / dx * 1000)},{_n(116 - (min(max(y, y0), y1) - y0) / dy * 112)}"

    colours = (t["color-text"], t["color-series-face"], t["color-series-general"])
    parts = ['<svg class="series" viewBox="0 0 1000 120" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" role="img">',
             f'<rect x="0" y="116" width="1000" height="1" fill="{t["color-graphic"]}"/>']
    # L-4: gridlines at the labelled y ticks (top, middle; the axis is the bottom) and at the x quarter ticks
    for gy in (4, 60):
        parts.append(f'<rect x="0" y="{gy}" width="1000" height="0.6" fill="{t["color-border"]}"/>')
    for gx in (0, 250, 500, 750, 999):
        parts.append(f'<rect x="{gx}" y="4" width="1" height="112" fill="{t["color-border"]}"/>')
    for level in thresholds:
        y = _n(116 - (min(max(level, y0), y1) - y0) / dy * 112)
        parts.append(f'<line x1="0" y1="{y}" x2="1000" y2="{y}" stroke="{t["color-series-threshold"]}" stroke-width="1" stroke-dasharray="6 4"/>')
    for i, pts in enumerate(lines):
        if len(pts) >= 2:
            parts.append(f'<polyline points="{" ".join(point(x, y) for x, y in pts)}" fill="none" stroke="{colours[i % 3]}" stroke-width="1.5"/>')
        elif pts:
            x, y = point(*pts[0]).split(",")
            parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{colours[i % 3]}"/>')
    parts.append("</svg>")
    return Markup("".join(parts))


def waveform_svg(bins: list[tuple[float, float, float, float]], x_range: tuple[float, float], peak: float) -> Markup:
    """A waveform envelope (L-7) in the frame of `series_svg`: per column the span from the lowest to the highest sample in the
    graphic ink and ± the RMS level in the text ink, around a centre line; y runs from -`peak` to +`peak`. `bins` are
    (time, lowest, highest, RMS); bins that share one of the 1,000 columns keep their extremes."""
    t = tokens()
    x0, x1 = x_range
    dx, top, mid, half = (x1 - x0) or 1.0, 4.0, 60.0, 56.0
    columns: dict[int, list[float]] = {}
    for time, low, high, rms in bins:
        column = min(999, max(0, int((time - x0) / dx * 1000)))
        held = columns.setdefault(column, [low, high, rms])
        held[0], held[1], held[2] = min(held[0], low), max(held[1], high), max(held[2], rms)

    def y(value: float) -> float:
        return mid - min(max(value / (peak or 1.0), -1.0), 1.0) * half

    parts = ['<svg class="series" viewBox="0 0 1000 120" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" role="img">',
             f'<rect x="0" y="116" width="1000" height="1" fill="{t["color-graphic"]}"/>',
             f'<rect x="0" y="{_n(top)}" width="1000" height="0.6" fill="{t["color-border"]}"/>']
    for gx in (0, 250, 500, 750, 999):
        parts.append(f'<rect x="{gx}" y="{_n(top)}" width="1" height="112" fill="{t["color-border"]}"/>')
    width = max(1.0, 1000.0 / max(1, len(columns))) if columns and len(columns) < 1000 else 1.0
    for fill, band in ((t["color-graphic"], lambda c: (c[1], c[0])), (t["color-text"], lambda c: (c[2], -c[2]))):
        for column, held in sorted(columns.items()):
            upper, lower = band(held)
            if upper == lower == 0.0 and fill == t["color-text"]:
                continue
            parts.append(f'<rect x="{column}" y="{_n(y(upper))}" width="{_n(width)}" height="{_n(max(0.8, y(lower) - y(upper)))}" fill="{fill}"/>')
    parts.append(f'<rect x="0" y="{_n(mid)}" width="1000" height="0.6" fill="{t["color-graphic"]}"/>')
    parts.append("</svg>")
    return Markup("".join(parts))


HATCH_PITCH_PX, HATCH_WIDTH_PX = 9, 2


def reliability_plate(mask_sha256: str, base_sha256: str | None, box: tuple[int, int], *, photo: bool = True,
                      store: ObjectStore | None = None) -> str | None:
    """The reliability mask as a plate a reader can place (DR P1-8): the source (or the map preview) with a 45° hatch in `lane-hatch`
    over the areas the mask marks unreliable (mask ≤ 127). PNG, nearest-neighbour mask scaling, no metadata. None when the mask is
    missing, altered or unreadable; without a readable base the hatch is drawn on `surface-2`."""
    import numpy as np

    store = store or default_store()
    mask = _load_rgb(mask_sha256, store)
    if mask is None:
        return None
    base = _load_rgb(base_sha256, store) if base_sha256 else None
    t = tokens()
    if base is None:
        base = Image.new("RGB", mask.size, _rgb(t["color-surface-2"]))
    base.thumbnail(box, Image.Resampling.LANCZOS if photo else Image.Resampling.NEAREST)
    pixels = np.asarray(base, dtype=np.float64).copy()
    unreliable = np.asarray(mask.convert("L").resize(base.size, Image.Resampling.NEAREST)) <= 127
    yy, xx = np.indices(unreliable.shape)
    stripes = ((xx + yy) % HATCH_PITCH_PX) < HATCH_WIDTH_PX
    pixels[unreliable & stripes] = _rgb(t["color-lane-hatch"])
    pixels[unreliable & ~stripes] = pixels[unreliable & ~stripes] * 0.55 + 255 * 0.45  # a veil, so the hatch reads on dark areas
    out = io.BytesIO()
    Image.fromarray(np.rint(pixels).astype(np.uint8)).save(out, "PNG", optimize=False, compress_level=6)
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def _load_rgb(sha256: str, store: ObjectStore) -> Image.Image | None:
    """A stored image as RGB; None when the object is missing, altered or not a readable image."""
    try:
        data = store.read_bytes(sha256)
    except (ObjectNotFound, InvalidObjectId):
        return None
    if hashlib.sha256(data).hexdigest() != sha256:
        return None
    try:
        with Image.open(io.BytesIO(data)) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


OVERLAY_ALPHA = 0.7


def overlay_plate(map_sha256: str, source_sha256: str, box: tuple[int, int], *, store: ObjectStore | None = None) -> str | None:
    """Fig. 4 of a localisation exhibit (UX §6.3 R5; DR P1-8): the map preview over the source at 70 %, a PNG within `box` (source
    Lanczos-scaled, map nearest-neighbour to the same size, no metadata). None when either object is missing, altered or unreadable."""
    store = store or default_store()
    source, heat = _load_rgb(source_sha256, store), _load_rgb(map_sha256, store)
    if source is None or heat is None:
        return None
    source.thumbnail(box, Image.Resampling.LANCZOS)
    blended = Image.blend(source, heat.resize(source.size, Image.Resampling.NEAREST), OVERLAY_ALPHA)
    out = io.BytesIO()
    blended.save(out, "PNG", optimize=False, compress_level=6)
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def marks_svg(width: int, height: int, marks: list[tuple[str, dict, str]]) -> Markup:
    """Region and annotation boxes over a plate in original pixels (viewBox = the image, so `meet` letterboxes as the plate's
    `object-fit: contain` does): a `lens-ink` outline on a dark halo, solid for regions and dashed for notes, and the label on a
    `surface` tab inside the frame, at the top-left corner of a region and the bottom-left corner of a note."""
    t = tokens()
    u = max(width, height) / 250.0
    names = ", ".join(label for label, _, _ in marks)
    parts = [f'<svg class="marks" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Marked: {escape(names)}">']
    for label, b, kind in marks:
        x, y, w, h = (float(b[k]) for k in ("x", "y", "w", "h"))
        dash = f' stroke-dasharray="{_n(4 * u)} {_n(3 * u)}"' if kind == "note" else ""
        parts.append(f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}" fill="none" stroke="#000000" stroke-opacity="0.55" '
                     f'stroke-width="{_n(3 * u)}"/>')
        parts.append(f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}" fill="none" stroke="{t["color-lens-ink"]}" '
                     f'stroke-width="{_n(1.5 * u)}"{dash}/>')
        tab_w, tab_h = (len(label) * 4.6 + 4) * u, 9 * u
        tx = min(max(0.0, x), max(0.0, width - tab_w))
        if kind == "note":  # below the bottom-left corner, so a note on a region's box keeps both labels readable
            ty = y + h if y + h + tab_h <= height else y + h - tab_h
        else:
            ty = y - tab_h if y >= tab_h else y
        parts.append(f'<rect x="{_n(tx)}" y="{_n(ty)}" width="{_n(tab_w)}" height="{_n(tab_h)}" fill="{t["color-surface"]}"/>')
        parts.append(f'<text x="{_n(tx + 2 * u)}" y="{_n(ty + 6.8 * u)}" font-family="Inter Variable" font-size="{_n(7 * u)}" '
                     f'font-weight="600" fill="{t["color-text"]}">{escape(label)}</text>')
    parts.append("</svg>")
    return Markup("".join(parts))


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.strip().lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def heat_legend_svg() -> Markup:
    """The heatmap colour bar as 11 discrete swatches (`heat-00` … `heat-10`), low to high."""
    t = tokens()
    swatches = "".join(f'<rect x="{i * 20}" y="0" width="20" height="8" fill="{t[f"heat-{i:02d}"]}"/>' for i in range(11))
    return Markup(f'<svg class="heat-legend" viewBox="0 0 220 8" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" '
                  f'role="img" aria-label="colour bar from low to high">{swatches}</svg>')


def thumbnails(images: dict[str, str], store: ObjectStore | None = None) -> dict[str, str]:
    """Finding id → data URI for the snapshot's thumbnail artefacts that are present and intact."""
    out = {}
    for finding_id, sha256 in sorted(images.items()):
        uri = thumbnail(sha256, store)
        if uri is not None:
            out[finding_id] = uri
    return out
