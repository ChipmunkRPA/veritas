/** Canvas drawing of the lane stack (UX §8.4 "Timeline lanes"; ARCHITECTURE §7.5). One canvas for every lane, redrawn only when the
 *  model, domain, width, theme or selection change; the playhead is a separate element, so scrubbing never repaints the lanes. */
import { layoutSegments, MIN_SEGMENT_PX } from '../../components/ui'
import { frameStep } from '../../lib/api'
import type { LaneModel, VideoSegment } from './model'

export const LANE_H = 28
export const LANE_H_EXPANDED = 80
export const LANE_H_CURVE = 64
export const LANE_H_WAVE = 64
export const LANE_H_SPECTROGRAM = 96
export const LANE_GAP = 4
export const SYNC_RANGE_MS = 200
/** Display band of the lip-sync lane (UX §5.10 mock "tolerance ±80"): a reading aid, not a detector threshold. */
export const SYNC_TOLERANCE_MS = 80
const FLAG_RULE = 3
const RUG_H = 6
const LABEL_MIN_PX = 80
const SYNC_STRIP_H = 4
const MONO = '11px "JetBrains Mono Variable", ui-monospace, SFMono-Regular, Menlo, monospace'

export interface Domain { start: number; end: number }
export interface LaneRow { lane: LaneModel; y: number; h: number }

export function laneHeight(lane: LaneModel, expanded: boolean): number {
  if (lane.kind === 'estimate' && expanded && lane.state === 'ready' && !lane.inspectionOnly) return LANE_H_EXPANDED
  if (lane.kind === 'lip_sync' && lane.state === 'ready' && lane.curve?.length) return LANE_H_CURVE
  if (lane.kind === 'waveform' && lane.state === 'ready') return LANE_H_WAVE
  if (lane.kind === 'spectrogram' && lane.state === 'ready') return LANE_H_SPECTROGRAM
  return LANE_H
}

/** Value axes share the plot's vertical geometry but live in the label gutter. */
export function laneAxisLabels(lane: LaneModel, h: number): { text: string; y: number }[] {
  if (lane.state !== 'ready') return []
  if (lane.kind === 'lip_sync' && lane.curve?.length) return [
    { text: `+${SYNC_RANGE_MS} ms`, y: 7 }, { text: `−${SYNC_RANGE_MS} ms`, y: h - SYNC_STRIP_H - 7 },
  ]
  if (lane.kind === 'estimate' && h > LANE_H && !lane.inspectionOnly) return [{ text: '1', y: 7 }, { text: '0', y: h - 14 }]
  if (lane.kind === 'waveform') return [{ text: '+1', y: 7 }, { text: '−1', y: h - 7 }]
  if (lane.kind === 'spectrogram') return [{ text: '8 kHz', y: 7 }, { text: '0', y: h - 7 }]
  return []
}

/** The part of a lane's detail the label gutter has room for: whole " · " parts, never a cut word (L-7, design P2: "synthetic
 *  speech · p uncalibrat…"). The lane key and the lane's title carry the full detail. */
export const GUTTER_DETAIL_CHARS = 30
export function gutterDetail(detail: string, limit = GUTTER_DETAIL_CHARS): string {
  const parts = detail.split(' · ')
  let out = ''
  for (const part of parts) {
    const next = out ? `${out} · ${part}` : part
    if (next.length > limit) break
    out = next
  }
  return out
}

export function layoutRows(lanes: readonly LaneModel[], expanded: boolean): { rows: LaneRow[]; height: number } {
  const rows: LaneRow[] = []
  let y = 0
  for (const lane of lanes) {
    const h = laneHeight(lane, expanded)
    rows.push({ lane, y, h })
    y += h + LANE_GAP
  }
  return { rows, height: Math.max(0, y - LANE_GAP) }
}

export interface Palette {
  lane: string; frame: string; hatch: string; flag: string; tolerance: string; text: string; muted: string; graphic: string
  selection: string; annotation: string; tint: string[]; solid: string[]
}

export function readPalette(el: Element = document.documentElement): Palette {
  const css = getComputedStyle(el)
  const v = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback
  return {
    lane: v('--color-lane', '#F3F1EC'), frame: v('--color-lane-frame', '#8F887B'), hatch: v('--color-lane-hatch', '#A39C8F'),
    flag: v('--color-lane-flag', '#1B1A17'), tolerance: v('--color-lane-tolerance', '#E9E6DF'), text: v('--color-text', '#1B1A17'),
    muted: v('--color-text-muted', '#6B665C'), graphic: v('--color-graphic', '#8F887B'), selection: v('--color-selection', '#2F5BB7'),
    annotation: v('--color-annotation', '#2F5BB7'),
    tint: [1, 2, 3, 4, 5].map((n) => v(`--color-verdict-${n}-tint`, '#EEE')), solid: [1, 2, 3, 4, 5].map((n) => v(`--color-verdict-${n}`, '#888')),
  }
}

export const xScale = (domain: Domain, width: number) => (t: number): number =>
  domain.end > domain.start ? ((t - domain.start) / (domain.end - domain.start)) * width : 0
export const tScale = (domain: Domain, width: number) => (x: number): number =>
  width > 0 ? domain.start + (x / width) * (domain.end - domain.start) : domain.start

/** Axis label step: at most ten labels across the domain. */
export function tickStep(span: number): number {
  for (const s of [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200]) if (span / s <= 10) return s
  return 3600
}

const patterns = new WeakMap<CanvasRenderingContext2D, Map<string, CanvasPattern | null>>()
/** 45°, 6 px pitch, 1 px lines (UX §8.3), cached per context and colour. */
function hatch(ctx: CanvasRenderingContext2D, colour: string): CanvasPattern | string {
  let byColour = patterns.get(ctx)
  if (!byColour) patterns.set(ctx, (byColour = new Map()))
  if (!byColour.has(colour)) {
    const tile = document.createElement('canvas')
    tile.width = 6
    tile.height = 6
    const t = tile.getContext('2d')
    if (t) {
      t.strokeStyle = colour
      t.lineWidth = 1
      t.beginPath()
      for (const o of [-6, 0, 6]) { t.moveTo(o, 6); t.lineTo(o + 6, 0) }
      t.stroke()
    }
    byColour.set(colour, t ? ctx.createPattern(tile, 'repeat') : null)
  }
  return byColour.get(colour) ?? colour
}

const BADGE_INSET = 8
const BADGE_BAND = 16  // the height of the label's text with its outline
/** Where the cleared ground of a lane's right-hand label begins: the label's width, its inset and as much again before it. */
export const codecBadgeLeft = (width: number, textWidth: number): number => Math.max(0, width - BADGE_INSET - textWidth - BADGE_INSET)

function label(ctx: CanvasRenderingContext2D, p: Palette, text: string, x: number, y: number, align: CanvasTextAlign = 'left', colour = p.muted) {
  ctx.font = MONO
  ctx.textAlign = align
  ctx.textBaseline = 'middle'
  ctx.lineWidth = 4
  ctx.strokeStyle = p.lane
  ctx.strokeText(text, x, y)
  ctx.fillStyle = colour
  ctx.fillText(text, x, y)
}

function drawSpans(ctx: CanvasRenderingContext2D, p: Palette, lane: LaneModel, domain: Domain, width: number, y: number, h: number) {
  const span = domain.end - domain.start
  const visible = lane.spans.filter((s) => s.end > domain.start && s.start < domain.end)
    .map((s) => ({ start: s.start - domain.start, end: s.end - domain.start, step: s.step }))
  for (const r of layoutSegments(visible, span, width)) {
    ctx.fillStyle = r.step >= 4 && !lane.inspectionOnly ? p.solid[r.step - 1] : p.tint[r.step - 1]
    ctx.fillRect(r.x, y, r.w, h)
    if (r.step >= 4 && !lane.inspectionOnly) {
      ctx.fillStyle = p.flag
      ctx.fillRect(r.x, y, r.w, FLAG_RULE)
    }
  }
}

interface GapLabel { text: string; x: number; width: number; y: number; h: number }

function drawGaps(ctx: CanvasRenderingContext2D, p: Palette, lane: LaneModel, x: (t: number) => number, width: number, y: number, h: number, labels: GapLabel[]) {
  for (const g of lane.gaps) {
    const x0 = Math.max(0, x(g.start))
    const end = Math.min(width, x(g.end))
    if (end <= x0) continue
    const w = Math.min(width - x0, Math.max(MIN_SEGMENT_PX, end - x0))
    ctx.fillStyle = hatch(ctx, p.hatch)
    ctx.fillRect(x0, y, w, h)
    if (w >= LABEL_MIN_PX) labels.push({ text: g.label, x: x0, width: w, y, h })
  }
}

/** Paint gap text after the selection outline, on an opaque chip contained by the gap. */
function drawGapLabel(ctx: CanvasRenderingContext2D, p: Palette, gap: GapLabel) {
  ctx.font = MONO
  let text = gap.text
  let textWidth = ctx.measureText(text).width
  // Keep the non-execution status visible even when its full reason belongs in the lane key/table.
  if (textWidth + 12 > gap.width && text.startsWith('Not run ·')) {
    text = 'Not run'
    textWidth = ctx.measureText(text).width
  }
  if (textWidth + 12 > gap.width) return
  const x = gap.x + gap.width / 2
  const y = gap.y + gap.h / 2
  ctx.fillStyle = p.lane
  ctx.fillRect(x - textWidth / 2 - 6, y - 8, textWidth + 12, 16)
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillStyle = p.muted
  ctx.fillText(text, x, y)
}

function drawCurve(ctx: CanvasRenderingContext2D, p: Palette, lane: LaneModel, x: (t: number) => number, y: number, h: number) {
  const curve = lane.curve ?? []
  const strip = SYNC_STRIP_H
  const plot = h - strip - 2
  const yOf = (ms: number) => y + plot / 2 - (Math.max(-SYNC_RANGE_MS, Math.min(SYNC_RANGE_MS, ms)) / SYNC_RANGE_MS) * (plot / 2 - 2)
  ctx.fillStyle = p.tolerance
  ctx.fillRect(0, yOf(SYNC_TOLERANCE_MS), x(Number.MAX_SAFE_INTEGER) > 0 ? ctx.canvas.width : 0, yOf(-SYNC_TOLERANCE_MS) - yOf(SYNC_TOLERANCE_MS))
  ctx.strokeStyle = p.graphic
  ctx.lineWidth = 1
  ctx.setLineDash([2, 3])
  ctx.beginPath()
  ctx.moveTo(0, Math.round(yOf(0)) + 0.5)
  ctx.lineTo(ctx.canvas.width, Math.round(yOf(0)) + 0.5)
  ctx.stroke()
  ctx.setLineDash([])
  ctx.strokeStyle = p.text
  // The tolerance band is a reading aid; line weight does not imply a detector threshold.
  ctx.lineWidth = 1.5
  for (let i = 1; i < curve.length; i++) {
    const a = curve[i - 1]
    const b = curve[i]
    ctx.beginPath()
    ctx.moveTo(x(a.t), yOf(a.ms))
    ctx.lineTo(x(b.t), yOf(b.ms))
    ctx.stroke()
  }
  for (const c of curve) {
    if (c.confidence == null) continue
    ctx.globalAlpha = Math.max(0, Math.min(1, c.confidence))
    ctx.fillStyle = p.text
    ctx.fillRect(x(c.start), y + h - strip, Math.max(1, x(c.end) - x(c.start)), strip)
    ctx.globalAlpha = 1
  }
}

function drawEstimateLine(ctx: CanvasRenderingContext2D, p: Palette, lane: LaneModel, x: (t: number) => number, y: number, h: number) {
  const band = 8
  const plot = h - band - 4
  const yOf = (v: number) => y + 2 + (1 - v) * plot
  ctx.strokeStyle = p.graphic
  ctx.lineWidth = 1
  ctx.setLineDash([2, 3])
  ctx.beginPath()
  ctx.moveTo(0, Math.round(yOf(0.5)) + 0.5)
  ctx.lineTo(ctx.canvas.width, Math.round(yOf(0.5)) + 0.5)
  ctx.stroke()
  ctx.setLineDash([])
  ctx.strokeStyle = p.text
  ctx.lineWidth = 1.5
  ctx.beginPath()
  ;(lane.points ?? []).forEach(([t, v], i) => (i ? ctx.lineTo(x(t), yOf(v)) : ctx.moveTo(x(t), yOf(v))))
  ctx.stroke()
}

/** Waveform envelope: the span between the lowest and highest sample of each bin in the muted ink, ± RMS in the text ink, around a
 *  centre line; then the content runs as dividers with their labels where they fit. A bin narrower than a pixel keeps its extremes. */
function drawWave(ctx: CanvasRenderingContext2D, p: Palette, lane: LaneModel, x: (t: number) => number, domain: Domain, width: number, y: number, h: number) {
  const mid = y + h / 2
  const half = h / 2 - 3
  const yOf = (v: number) => mid - Math.max(-1, Math.min(1, v)) * half
  const bins = lane.wave ?? []
  const half_bin = (lane.binS ?? 0) / 2
  const columns = new Map<number, { low: number; high: number; rms: number; x1: number }>()
  for (const b of bins) {
    if (b.t + half_bin < domain.start || b.t - half_bin > domain.end) continue
    const x0 = Math.floor(x(b.t - half_bin))
    const x1 = Math.max(x0 + 1, Math.ceil(x(b.t + half_bin)))
    const c = columns.get(x0)
    if (c) { c.low = Math.min(c.low, b.low); c.high = Math.max(c.high, b.high); c.rms = Math.max(c.rms, b.rms); c.x1 = Math.max(c.x1, x1) } else columns.set(x0, { low: b.low, high: b.high, rms: b.rms, x1 })
  }
  ctx.fillStyle = p.graphic
  for (const [x0, c] of columns) ctx.fillRect(x0, yOf(c.high), c.x1 - x0, Math.max(1, yOf(c.low) - yOf(c.high)))
  ctx.fillStyle = p.text
  for (const [x0, c] of columns) if (c.rms > 0) ctx.fillRect(x0, yOf(c.rms), c.x1 - x0, Math.max(1, yOf(-c.rms) - yOf(c.rms)))
  ctx.fillStyle = p.frame
  ctx.fillRect(0, Math.round(mid), width, 1)
  for (const run of lane.runs ?? []) {
    if (run.end < domain.start || run.start > domain.end) continue
    const x0 = Math.max(0, x(run.start))
    const w = Math.min(width, x(run.end)) - x0
    if (run.start > domain.start) { ctx.fillStyle = p.frame; ctx.fillRect(Math.round(x0), y, 1, h) }
    ctx.font = MONO
    if (ctx.measureText(run.label).width + 10 <= w) label(ctx, p, run.label, x0 + 5, y + 8)
  }
}

/** Spectrogram picture stretched over the seconds it spans; outside them the lane stays empty. */
function drawPicture(ctx: CanvasRenderingContext2D, lane: LaneModel, image: CanvasImageSource & { width: number; height: number }, x: (t: number) => number, domain: Domain, y: number, h: number) {
  const seconds = lane.imageSeconds ?? 0
  if (seconds <= 0 || domain.end <= 0 || domain.start >= seconds) return
  const from = Math.max(0, domain.start)
  const to = Math.min(seconds, domain.end)
  const sx = (from / seconds) * image.width
  const sw = Math.max(1, ((to - from) / seconds) * image.width)
  ctx.imageSmoothingEnabled = sw > x(to) - x(from)  // smooth when shrinking, keep cells crisp when zoomed in
  ctx.drawImage(image, sx, 0, sw, image.height, x(from), y, x(to) - x(from), h - 1)
}

export interface DrawInput {
  rows: readonly LaneRow[]
  width: number
  height: number
  domain: Domain
  palette: Palette
  selected: VideoSegment | null
  draft: Domain | null
  /** Loaded lane pictures by lane key (the spectrogram); a lane whose picture is not loaded yet stays empty. */
  images?: ReadonlyMap<string, CanvasImageSource & { width: number; height: number }>
}

export function drawLanes(ctx: CanvasRenderingContext2D, { rows, width, height, domain, palette: p, selected, draft, images }: DrawInput): void {
  ctx.clearRect(0, 0, width, height)
  const x = xScale(domain, width)
  const gapLabels: GapLabel[] = []
  for (const { lane, y, h } of rows) {
    ctx.save()
    ctx.beginPath()
    ctx.rect(0, y, width, h)
    ctx.clip()
    ctx.fillStyle = p.lane
    ctx.fillRect(0, y, width, h)
    if (lane.state === 'not_run') {
      ctx.fillStyle = hatch(ctx, p.hatch)
      ctx.fillRect(0, y, width, h)
      gapLabels.push({ text: `Not run · ${lane.reason ?? 'no data'}`, x: 0, width, y, h })
    } else if (lane.kind === 'waveform') {
      drawWave(ctx, p, lane, x, domain, width, y, h)
    } else if (lane.kind === 'spectrogram') {
      const image = images?.get(lane.key)
      if (image) drawPicture(ctx, lane, image, x, domain, y, h)
    } else if (lane.kind === 'estimate' && h > LANE_H && !lane.inspectionOnly) {
      drawSpans(ctx, p, lane, domain, width, y + h - 8, 8)
      drawEstimateLine(ctx, p, lane, x, y, h)
    } else if (lane.kind === 'lip_sync' && lane.curve?.length) {
      drawCurve(ctx, p, lane, x, y, h)
      drawGaps(ctx, p, lane, x, width, y, h, gapLabels)
    } else if (lane.kind === 'codec' || lane.kind === 'shots') {
      const ticks = (lane.ticks ?? []).filter((t) => t.t >= domain.start && t.t <= domain.end)
      for (const t of ticks) {
        ctx.fillStyle = t.kind === 'cut' ? p.text : t.kind === 'irregular' ? p.solid[3] : p.graphic
        ctx.fillRect(Math.round(x(t.t)), y + (t.kind === 'key' ? 6 : 0), t.kind === 'key' ? 1 : 2, t.kind === 'key' ? h - 6 : h)
        if (t.kind === 'irregular') label(ctx, p, '◆', x(t.t) + 6, y + 8, 'left', p.text)
      }
      // N-19d (design review P2): the lane's label ("10 key frames · GOP 24 frames") was drawn over the frame counts and the
      // ticks at the right end of the lane. It stands on a cleared band of the height of its text, so that a tick behind it
      // still shows above and below, and a frame count that would reach it is left out.
      ctx.font = MONO
      const badgeLeft = lane.badge ? codecBadgeLeft(width, ctx.measureText(lane.badge).width) : width
      if (lane.kind === 'codec') {
        for (let i = 1; i < ticks.length; i++) {
          const w = x(ticks[i].t) - x(ticks[i - 1].t)
          const text = String(ticks[i].frame - ticks[i - 1].frame)
          const centre = x(ticks[i - 1].t) + w / 2
          if (w >= 40 && centre + ctx.measureText(text).width / 2 + 4 <= badgeLeft) label(ctx, p, text, centre, y + h / 2 + 3, 'center')
        }
      }
      if (lane.badge) {
        const band = Math.min(h, BADGE_BAND)
        ctx.fillStyle = p.lane
        ctx.fillRect(badgeLeft, y + (h - band) / 2, width - badgeLeft, band)
        label(ctx, p, lane.badge, width - BADGE_INSET, y + h / 2, 'right', p.text)
      }
    } else if (lane.kind === 'provenance') {
      label(ctx, p, lane.badge ?? '', 8, y + h / 2, 'left', p.text)
    } else if (lane.kind === 'notes') {
      for (const n of lane.notes ?? []) {
        const x0 = x(n.start)
        const w = Math.max(MIN_SEGMENT_PX, x(n.end) - x0)
        ctx.globalAlpha = 0.18
        ctx.fillStyle = p.annotation
        ctx.fillRect(x0, y + 2, w, h - 4)
        ctx.globalAlpha = 1
        ctx.strokeStyle = p.annotation
        ctx.lineWidth = 1.5
        ctx.strokeRect(x0 + 0.75, y + 2.75, Math.max(0, w - 1.5), h - 5.5)
        label(ctx, p, n.label, x0 + 5, y + h / 2, 'left', p.annotation)
      }
      if (!lane.notes?.length) label(ctx, p, 'No interval notes · shift-drag a lane to add one', 8, y + h / 2)
    } else if (lane.kind === 'sampled') {
      ctx.fillStyle = p.muted
      for (const t of lane.ticks ?? []) if (t.t >= domain.start && t.t <= domain.end) ctx.fillRect(Math.round(x(t.t)), y + h - RUG_H, 1, RUG_H)
    } else {
      drawSpans(ctx, p, lane, domain, width, y, h)
      drawGaps(ctx, p, lane, x, width, y, h, gapLabels)
    }
    // A point marks one observation. Its fixed pixel width never represents an interval of scored media.
    for (const [time, value] of lane.samples ?? []) {
      if (time < domain.start || time > domain.end) continue
      const step = frameStep(value)
      ctx.fillStyle = step >= 4 && !lane.inspectionOnly ? p.solid[step - 1] : p.tint[step - 1]
      const markHeight = lane.kind === 'estimate' && h > LANE_H ? 8 : h - 2
      ctx.fillRect(Math.min(width - 2, Math.max(0, x(time) - 1)), y + h - markHeight - 1, 2, markHeight)
    }
    ctx.fillStyle = p.frame
    ctx.fillRect(0, y + h - 1, width, 1)
    ctx.restore()
  }
  if (draft) {
    const x0 = x(Math.min(draft.start, draft.end))
    const w = Math.max(MIN_SEGMENT_PX, x(Math.max(draft.start, draft.end)) - x0)
    ctx.globalAlpha = 0.12
    ctx.fillStyle = p.annotation
    ctx.fillRect(x0, 0, w, height)
    ctx.globalAlpha = 1
    ctx.strokeStyle = p.annotation
    ctx.lineWidth = 1.5
    ctx.setLineDash([4, 3])
    ctx.strokeRect(x0 + 0.75, 0.75, Math.max(0, w - 1.5), height - 1.5)
    ctx.setLineDash([])
  }
  if (selected && selected.end >= domain.start && selected.start <= domain.end) {
    const x0 = x(selected.start)
    const w = Math.max(MIN_SEGMENT_PX, x(selected.end) - x0)
    ctx.strokeStyle = p.selection
    ctx.lineWidth = 1.5
    ctx.strokeRect(x0 + 0.75, 0.75, Math.max(0, w - 1.5), height - 1.5)
  }
  for (const gap of gapLabels) drawGapLabel(ctx, p, gap)
}
