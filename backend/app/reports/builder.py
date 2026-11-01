"""PDF report generation (ReportLab). The report is evidence-first: every conclusion is traceable to a
Signal row (observed value, reference, direction, reasoning) and to the artifacts rendered alongside.
"""
from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.core.config import settings
from app.core.rounding import half_up
from app.reports import formatting

# Folio design tokens (frontend/design/direction-b-stripe-notion.md): warm paper, ink, hairlines, colour-blind-safe scale
INK = colors.HexColor("#1B1A17")
MUTED = colors.HexColor("#625D54")
FAINT = colors.HexColor("#857F73")
RULE = colors.HexColor("#E4E0D7")
RULE_STRONG = colors.HexColor("#BDB6A9")
PAPER_2 = colors.HexColor("#F4F2EC")
ACCENT = colors.HexColor("#1F4D8F")
BAND_COLOR = {
    "very_likely_authentic": colors.HexColor("#0B6B54"), "likely_authentic": colors.HexColor("#2A7460"), "uncertain": colors.HexColor("#6E685D"),
    "likely_ai": colors.HexColor("#AE4617"), "very_likely_ai": colors.HexColor("#9B2915"),
}
BAND_TINT = {
    "very_likely_authentic": colors.HexColor("#E2F0EA"), "likely_authentic": colors.HexColor("#E8F2EE"), "uncertain": colors.HexColor("#EEEBE4"),
    "likely_ai": colors.HexColor("#F9E4DA"), "very_likely_ai": colors.HexColor("#F7DCD7"),
}
BAND_STEP = {"very_likely_authentic": 1, "likely_authentic": 2, "uncertain": 3, "likely_ai": 4, "very_likely_ai": 5}
HL_AI, HL_MIXED = "#F8DED3", "#EFE7D3"
DIR_LABEL = {"supports_ai": "→ AI", "supports_authentic": "→ authentic", "neutral": "neutral", "inconclusive": "n/a"}

_ss = getSampleStyleSheet()
SERIF, SANS, MONO = "Times-Roman", "Helvetica", "Courier"
S_TITLE = ParagraphStyle("t", parent=_ss["Title"], fontName=SERIF, fontSize=26, leading=30, spaceAfter=2, alignment=TA_LEFT, textColor=INK)
S_EYEBROW = ParagraphStyle("e", parent=_ss["BodyText"], fontName=SANS, fontSize=7.2, leading=10, textColor=MUTED, spaceBefore=14, spaceAfter=4)
S_H = ParagraphStyle("h", parent=_ss["Heading2"], fontName=SANS, fontSize=11, leading=14, spaceBefore=2, spaceAfter=6, textColor=INK)
S_B = ParagraphStyle("b", parent=_ss["BodyText"], fontName=SANS, fontSize=9, leading=13, textColor=INK)
S_SERIF = ParagraphStyle("sb", parent=S_B, fontName=SERIF, fontSize=10.5, leading=15)
S_SM = ParagraphStyle("sm", parent=S_B, fontSize=7.8, leading=10.5, textColor=MUTED)
S_CELL = ParagraphStyle("c", parent=S_B, fontSize=7.6, leading=9.6)
S_CELL_MUTED = ParagraphStyle("cm", parent=S_CELL, textColor=MUTED)
S_MONO = ParagraphStyle("m", parent=S_B, fontName=MONO, fontSize=7.4, leading=9.4)
S_NUMERAL = ParagraphStyle("n", parent=S_B, fontName=SERIF, fontSize=44, leading=44, textColor=INK)
S_BANDLABEL = ParagraphStyle("bl", parent=S_B, fontName=SERIF, fontSize=17, leading=21)


def _eyebrow(text: str) -> Paragraph:
    return Paragraph(html.escape(text).upper().replace(" ", "&nbsp;&nbsp;"), S_EYEBROW)


def _p(text: str, style=S_B) -> Paragraph:
    return Paragraph(html.escape(str(text)).replace("\n", "<br/>"), style)


def _table(rows: list[list], widths: list[float], header: bool = True, zebra: bool = True) -> Table:
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE)]  # hairline row separators, no vertical rules, no fills
    if header:
        style += [("LINEBELOW", (0, 0), (-1, 0), 0.8, RULE_STRONG), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED)]
    t.setStyle(TableStyle(style))
    return t


def _img(path: Path, max_w: float, max_h: float) -> Image | None:
    try:
        from PIL import Image as PILImage

        with PILImage.open(path) as im:
            w, h = im.size
        s = min(max_w / w, max_h / h, 1.0)
        return Image(str(path), width=w * s, height=h * s)
    except Exception:
        return None


def build_report(summary: dict, result: dict, art_dir: Path, out_path: Path, *, review_line: str | None = None,
                 number: str | None = None) -> Path:
    """`review_line` is the review status line of the revision (N-3; SEC-09), printed under the title. `number` is what a v2
    result shows in place of "<n> % p(AI)", as its page and its v2 reports say it (D-68; N-10b, code review 3): the score
    of an uncalibrated fallback result, an estimate that is body text, or that no estimate is shown. None prints the
    verdict's own number: a v1 result, or a calibrated v2 one."""
    kind = summary["kind"]
    verdict = result["verdict"]
    signals = result["signals"]
    tmp_path = out_path.with_suffix(".pdf.tmp")
    doc = SimpleDocTemplate(str(tmp_path), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f"Chipmunk Verify report — {summary['filename']}", author=f"{settings.app_name} {settings.version}")
    W = A4[0] - 32 * mm
    el: list = []

    # --- header ---
    el.append(_eyebrow(f"{kind} · analysis {summary['id'][:8]} · {settings.app_name} {settings.version}"))
    el.append(Paragraph(html.escape(summary["filename"]), S_TITLE))
    subject = {"image": "Deepfake", "video": "Deepfake", "audio": "Synthetic-voice"}.get(kind, "AI-written text")
    el.append(_p(f"{subject} analysis report · generated {formatting.utc(datetime.now(UTC))}", S_SM))
    if review_line:
        el.append(_p(f"Review: {review_line}", S_SM))
    el.append(Spacer(1, 8))
    meta_rows = [["File", summary["filename"]], ["Type", f"{kind} ({summary['content_type']}, {summary['size_bytes']:,} bytes)"], ["SHA-256", summary["sha256"]],
                 ["Submitted", summary.get("created_at", "")], ["Completed", summary.get("finished_at", "")]]
    if kind == "image" and result.get("image"):
        im = result["image"]
        meta_rows.append(["Dimensions", f"{im['width']}×{im['height']} px, {im.get('format')}"])
    if kind == "video" and result.get("video"):
        v = result["video"]
        meta_rows.append(["Stream", f"{v.get('codec')} {v.get('width')}×{v.get('height')} @ {v.get('fps')} fps, {v.get('duration',0):.1f}s, audio={'yes' if v.get('has_audio') else 'no'}"])
    if kind == "audio" and result.get("audio"):
        au = result["audio"]
        meta_rows.append(["Stream", f"{au.get('codec')} @ {au.get('sample_rate')} Hz, {au.get('channels')} channel(s), {au.get('duration_s') or 0:.1f}s"])
    if kind == "document" and result.get("stats"):
        st = result["stats"]
        meta_rows.append(["Text", f"{st['words']:,} words, {st['sentences']} sentences" + (f", {result['document'].get('pages')} pages" if result.get('document', {}).get('pages') else "")])
    el.append(_table([[_p(k, S_CELL_MUTED), Paragraph(html.escape(str(v)), S_MONO if k == "SHA-256" else S_CELL)] for k, v in meta_rows], [30 * mm, W - 30 * mm], header=False, zebra=False))

    # --- verdict ---
    el.append(_eyebrow("Verdict"))
    col = BAND_COLOR.get(verdict["band"], colors.grey)
    if number is not None:
        numeral = Paragraph(html.escape(number), S_BANDLABEL)
    elif verdict.get("probability_ai") is None:  # D-49: capped or not assessable, no estimate published
        numeral = Paragraph("No estimate", S_BANDLABEL)
    else:
        pct = half_up(verdict["probability_ai"] * 100)
        numeral = Paragraph(f'{pct}<font size="14" color="{MUTED.hexval()}"> % p(AI)</font>', S_NUMERAL)
    label = Paragraph(f'<font color="{col.hexval()}">{html.escape(verdict["label"])}</font>', S_BANDLABEL)
    facts = [[_p("Confidence", S_CELL_MUTED), _p(verdict["confidence"], S_CELL)],
             [_p("Signals", S_CELL_MUTED), _p(f"{verdict['signals_for_ai']} → AI · {verdict['signals_for_authentic']} → authentic · of {verdict['signals_total']} informative", S_CELL)]]
    decisive_names = [s_["name"] for s_ in signals if s_.get("decisive")]
    if decisive_names:
        facts.append([_p("Decisive", S_CELL_MUTED), _p("; ".join(decisive_names), S_CELL)])
    if kind == "document" and verdict.get("probability_ai") is not None:
        # Retained sentence diagnostics cannot stand in for an unpublished verdict.
        st = result["stats"]
        facts.append([_p("AI-like share", S_CELL_MUTED), _p(f"{st['ai_share']:.0%} of text · {st['ai_sentences']} sentences (mixed {st['mixed_share']:.0%} · {st['mixed_sentences']}; weighted {st['weighted_ai_share']:.0%})", S_CELL)])
    ft = Table(facts, colWidths=[30 * mm, 55 * mm])
    ft.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 2)]))
    # five-segment scale with marker
    seg_w = (W - 85 * mm - 8 * mm) / 5
    scale_cells = []
    step = BAND_STEP.get(verdict["band"], 3)
    for i in range(1, 6):
        band_i = [k for k, v in BAND_STEP.items() if v == i][0]
        scale_cells.append("")
    scale = Table([scale_cells], colWidths=[seg_w] * 5, rowHeights=[4])
    sc_style = [("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]
    for i, band_i in enumerate(sorted(BAND_STEP, key=BAND_STEP.get)):
        sc_style.append(("BACKGROUND", (i, 0), (i, 0), BAND_COLOR[band_i] if i + 1 <= step else BAND_TINT[band_i]))
    scale.setStyle(TableStyle(sc_style))
    left = [numeral, Spacer(1, 4), label, Spacer(1, 8), scale, _p("authentic · inconclusive · AI / manipulated", S_SM)]
    vt = Table([[left, ft]], colWidths=[W - 85 * mm, 85 * mm])
    vt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEABOVE", (0, 0), (-1, 0), 0.8, RULE_STRONG), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    el.append(vt)
    el.append(Spacer(1, 6))
    el.append(Paragraph(html.escape(verdict["summary"]), S_SERIF))

    # --- reasoning ---
    el.append(_eyebrow("Reasoning"))
    el.append(_p("Each step references an evidence row below; strongest evidence first.", S_SM))
    for i, r in enumerate(verdict["reasoning"], 1):
        el.append(Paragraph(f'<font name="{MONO}" color="{FAINT.hexval()}" size="7.5">{i:>2}</font>&nbsp;&nbsp;{html.escape(r)}', S_SERIF))
    if verdict.get("caveats"):
        el.append(_eyebrow("Caveats"))
        for c in verdict["caveats"]:
            el.append(Paragraph(f'<font color="{FAINT.hexval()}">·</font>&nbsp;&nbsp;{html.escape(c)}', S_SM))

    # --- evidence table ---
    el.append(_eyebrow("Evidence"))
    if kind == "document":
        el.append(_p("Evidence scores are uncalibrated diagnostic values, not probabilities of authorship; w is the fusion weight.", S_SM))
    rows = [[_p("#", S_CELL), _p("Signal", S_CELL), _p("Observed", S_CELL), _p("Reference / expected", S_CELL), _p("Direction", S_CELL), _p("Score · w" if kind == "document" else "p(AI) · w", S_CELL), _p("Why this matters", S_CELL)]]
    for n, s in enumerate(sorted(signals, key=lambda x: (x["ai_probability"] is None, -abs((x["ai_probability"] or 0.5) - 0.5) * x["weight"])), 1):
        pa = f"{s['ai_probability']:.2f} · {s['weight']:.2g}" if s["ai_probability"] is not None else "—"
        dcol = BAND_COLOR["very_likely_ai"] if s["direction"] == "supports_ai" else BAND_COLOR["very_likely_authentic"] if s["direction"] == "supports_authentic" else MUTED
        direction = f'<font color="{dcol.hexval()}">{html.escape(DIR_LABEL.get(s["direction"], s["direction"]))}</font> <font color="{MUTED.hexval()}">{s["strength"]}</font>' + (f'<br/><font color="{dcol.hexval()}" size="6.5">DECISIVE</font>' if s.get("decisive") else "")
        name = html.escape(s["name"]) + f'<br/><font color="{FAINT.hexval()}" size="6.5">{html.escape(s["category"]).upper()}</font>'
        rows.append([Paragraph(str(n), S_MONO), Paragraph(name, S_CELL), Paragraph(html.escape(s["observed"]), S_MONO), _p(s["reference"], S_CELL_MUTED), Paragraph(direction, S_CELL), Paragraph(pa, S_MONO), _p(s["reasoning"], S_CELL)])
    el.append(_table(rows, [6 * mm, 30 * mm, 32 * mm, 30 * mm, 18 * mm, 15 * mm, W - 131 * mm]))

    # --- kind-specific sections ---
    sec = 4
    if kind == "document":
        el.append(_eyebrow("Manuscript"))
        sec += 1
        el.append(_p("Tinted = AI-like (score ≥ 0.65); lighter tint = mixed/uncertain (0.40–0.65); plain = human-like. Sentence scores are uncalibrated diagnostic values, not probabilities of authorship. Bracketed numbers refer to the notes that follow.", S_SM))
        sents = result["sentences"]
        flagged = sorted([x for x in sents if x.get("p_ai") is not None and x["band"] == "ai"], key=lambda x: -x["p_ai"])[:12]
        note_ids = {x["idx"]: i + 1 for i, x in enumerate(flagged)}
        by_par: dict[int, list] = {}
        for x in sents:
            by_par.setdefault(x["paragraph"], []).append(x)
        shown = 0
        for pi in sorted(by_par):
            parts = []
            for x in by_par[pi]:
                t = html.escape(x["text"])
                if x.get("band") == "ai":
                    t = f'<font backColor="{HL_AI}">{t}</font>'
                elif x.get("band") == "mixed":
                    t = f'<font backColor="{HL_MIXED}">{t}</font>'
                if x["idx"] in note_ids:
                    t += f'<super><font color="{BAND_COLOR["very_likely_ai"].hexval()}">[{note_ids[x["idx"]]}]</font></super>'
                parts.append(t)
            el.append(Paragraph(" ".join(parts), S_SERIF))
            el.append(Spacer(1, 4))
            shown += sum(len(x["text"]) for x in by_par[pi])
            if shown > 60000:
                el.append(_p("… (text truncated in the PDF; the full sentence-level analysis is available in the JSON export)", S_SM))
                break
        if flagged:
            el.append(_eyebrow("Notes on flagged sentences"))
            nrows = [[_p("#", S_CELL), _p("Sentence", S_CELL), _p("Score", S_CELL), _p("Evidence", S_CELL)]]
            for x in flagged:
                nrows.append([_p(str(note_ids[x["idx"]]), S_CELL), _p(x["text"][:300], S_CELL), _p(f"{x['p_ai']:.2f}", S_CELL), _p("; ".join(x["reasons"]), S_CELL)])
            el.append(_table(nrows, [8 * mm, 70 * mm, 12 * mm, W - 90 * mm]))
    if kind == "image" and result.get("faces"):
        el.append(_eyebrow("Faces"))
        sec += 1
        frows = [[_p("#", S_CELL), _p("Box (x, y, w, h)", S_CELL), _p("Detector scores p(fake)", S_CELL)]]
        for i, f in enumerate(result["faces"], 1):
            b = f["face"]
            frows.append([_p(str(i), S_CELL), _p(f"{b['x']}, {b['y']}, {b['w']}, {b['h']}", S_CELL), _p(", ".join(f"{k}={v:.2f}" for k, v in f.get("scores", {}).items()) or "—", S_CELL)])
        el.append(_table(frows, [8 * mm, 40 * mm, W - 48 * mm]))
    if kind == "video" and result.get("frames"):
        el.append(_eyebrow("Frame timeline"))
        sec += 1
        fr = result["frames"]
        top = sorted(fr, key=lambda x: -x["p_frame"])[:10]
        trows = [[_p("t (s)", S_CELL), _p("frame", S_CELL), _p("faces", S_CELL), _p("p(frame)", S_CELL), _p("p(face-swap)", S_CELL), _p("per-model", S_CELL)]]
        for x in sorted(top, key=lambda x: x["time"]):
            trows.append([_p(f"{x['time']:.2f}", S_CELL), _p(str(x["index"]), S_CELL), _p(str(x["faces"]), S_CELL), _p(f"{x['p_frame']:.2f}", S_CELL), _p(f"{x['p_face']:.2f}" if x["p_face"] is not None else "—", S_CELL), _p(", ".join(f"{k}={v:.2f}" for k, v in x["scores"].items()), S_CELL)])
        el.append(_p(f"{len(fr)} frames sampled; the 10 highest-scoring frames are listed.", S_SM))
        el.append(_table(trows, [12 * mm, 12 * mm, 12 * mm, 16 * mm, 20 * mm, W - 72 * mm]))

    # --- artifacts ---
    arts = [a for a in result.get("artifacts", []) if (art_dir / a).exists()]
    if arts:
        el.append(PageBreak())
        el.append(_eyebrow("Figures"))
        sec += 1
        captions = {
            "faces.jpg": "Detected faces (YuNet). Face-swap detectors were applied to each box.",
            "spectrum.png": "Frequency analysis: 2-D log power spectrum and its radial profile. Generators leave periodic peaks / abnormal high-frequency tails.",
            "ela.jpg": "Error Level Analysis: brighter = larger change on JPEG re-save. A region that differs from its surroundings was compressed differently (edit/paste).",
            "noise_map.jpg": "Local noise-residual map. Sensor noise is spatially consistent; synthesised or blended regions break that consistency.",
            "timeline.png": "Per-frame detector scores across the clip.",
            "key_frames.jpg": "Highest-scoring frames with detected faces.",
            "sentence_scores.png": "Per-sentence uncalibrated scores in document order.",
        }
        for n, a in enumerate(arts, 1):
            cap = captions.get(a) or (f"Grad-CAM: regions that drove the '{a.replace('gradcam_','').replace('.jpg','')}' classifier toward the synthetic class." if a.startswith("gradcam") else a)
            im = _img(art_dir / a, W, 110 * mm)
            if im:
                el.append(KeepTogether([im, Paragraph(f"<b>Fig. {n}</b> — {html.escape(cap)}", S_SM), Spacer(1, 10)]))

    # --- methodology ---
    el.append(_eyebrow("Methodology, models and limitations"))
    method = {
        "image": "Provenance (C2PA Content Credentials, EXIF/XMP, generator metadata) → face detection (OpenCV YuNet) → learned detectors: UnivFD CLIP ViT-L/14 linear probe (Ojha et al., CVPR 2023), DFDC-winner EfficientNet-B7 face-swap classifier, and fine-tuned Swin/SigLIP/ViT AI-image classifiers → classical forensics (frequency spectrum, error-level analysis, noise-residual consistency) → weighted log-odds fusion with decisive-evidence override.",
        "video": "ffprobe metadata → uniform frame sampling → batched frame scoring with the image detectors (face-swap detectors on the tracked face; whole-frame synthesis detectors on each frame) → temporal consistency (landmark jitter, face/ring flicker, score volatility) → aggregation (top-30% mean for face-swap, mean for whole-frame) → fusion.",
        "document": "Text extraction (PyMuPDF / python-docx) → sentence segmentation (pysbd) → DeBERTa-v3-large AI-text classifier on 3-sentence windows and document chunks → Binoculars zero-shot detector (observer/performer perplexity ratio, per-token, aggregated per sentence) → stylometric tells → per-sentence log-odds fusion and document-level fusion.",
        "audio": "ffprobe stream facts → 16 kHz mono PCM (first 600 s) → speech, music, noise and silence per second (heuristic; recordings with less than 1 s of speech or more music than speech are not assessable) → wav2vec2 anti-spoofing classifiers on ≤ 4 s speech windows at a 2 s hop → voice-lane segments → weighted log-odds fusion (uncalibrated fallback until an audio fuser is calibrated).",
    }[kind]
    el.append(_p(method))
    el.append(Spacer(1, 4))
    el.append(_p("Limitations: all probabilities are statistical estimates from open-source detectors evaluated on public benchmarks; they are evidence subject to uncertainty and error, not a finding of fact. Heavy compression, low resolution, cropping, screenshots, paraphrasing, translation, and very short inputs reduce reliability. Memorised or highly formulaic human text can be flagged; novel generators may evade learned detectors. Decisive provenance evidence (C2PA, generator metadata) can be stripped, so its absence is never treated as evidence of authenticity. A qualified human reviewer should confirm any consequential conclusion.", S_SM))
    doc.build(el)
    import os

    os.replace(tmp_path, out_path)  # atomic publish: readers never see a partial PDF
    return out_path
