import { useEffect, useMemo, useRef, useState, type PointerEvent } from 'react'
import { Minus, Plus } from 'lucide-react'
import { cx } from '../../lib/cx'
import { timecode } from '../../lib/pts'
import { describeLane, videoInspectionCaption, type VideoModel, type VideoSegment } from './model'
import { drawLanes, gutterDetail, laneAxisLabels, layoutRows, readPalette, tickStep, tScale, xScale, type Domain, type LaneRow, type Palette } from './lanesCanvas'

const AXIS_H = 40
const MIN_SPAN_S = 0.2
const DRAG_PX = 3

function usePalette(): Palette | null {
  const [palette, setPalette] = useState<Palette | null>(null)
  useEffect(() => {
    const read = () => setPalette(readPalette())
    read()
    const mo = new MutationObserver(read)
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => mo.disconnect()
  }, [])
  return palette
}

const tool = 'h-7 min-w-7 px-1.5 inline-flex items-center justify-center rounded-sm text-small text-text-muted whitespace-nowrap hover:text-text hover:bg-surface-3 transition-colors duration-fast'

/**
 * Timeline lanes (UX §5.10, §8.4; ARCHITECTURE §7.5): Estimate, Face A…, Lip-sync, Audio, Synthesis, Codec (GOP strip), Shots,
 * Provenance, Notes and the Sampled rug on one canvas under a time axis with segment callouts, zoom and brush.
 * - Click seeks; shift-click selects the segment under the pointer; shift-drag drafts an interval note (`onInterval`).
 * - Ctrl or ⌘ + wheel zooms at the pointer, a horizontal wheel pans, dragging on the axis brushes a range, double-click fits.
 * - "View as table" renders the same lanes as a table (screen readers and export).
 */
export default function TimelineLanes({ model, time, selected, focus, isolatedTrack = null, onSeek, onSelectSegment, onInterval, compact = false, medium = 'video', className }: {
  model: VideoModel
  /** The audio exhibit reuses the lanes for a recording (A-1): the copy names the recording and its voice fuser. */
  medium?: 'video' | 'recording'
  time: number | null
  selected: VideoSegment | null
  focus?: { key: string; start: number; end: number } | null
  isolatedTrack?: string | null
  onSeek: (t: number) => void
  onSelectSegment?: (segment: VideoSegment | null) => void
  onInterval?: (start: number, end: number) => void
  compact?: boolean
  className?: string
}) {
  const full: Domain = useMemo(() => ({ start: 0, end: Math.max(model.duration, 1e-3) }), [model.duration])
  const [domain, setDomain] = useState<Domain>(full)
  useEffect(() => setDomain(full), [full])
  const [expanded, setExpanded] = useState(false)
  const [table, setTable] = useState(false)
  const [draft, setDraft] = useState<Domain | null>(null)
  const [brush, setBrush] = useState<Domain | null>(null)
  const palette = usePalette()

  const wrap = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const el = wrap.current
    if (!el) return
    const ro = new ResizeObserver((entries) => setWidth(Math.floor(entries[0].contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [table])

  const lanes = useMemo(() => model.lanes.filter((l) => !(model.inspectionReason && l.kind === 'estimate'))
    .filter((l) => !isolatedTrack || (l.kind !== 'face' && l.kind !== 'lip_sync') || l.trackId === isolatedTrack || l.trackId == null), [model.lanes, model.inspectionReason, isolatedTrack])
  const { rows, height } = useMemo(() => layoutRows(lanes, expanded), [lanes, expanded])

  // Lane pictures (the spectrogram of a recording) load once per URL; the canvas redraws when one arrives.
  const pictureKey = lanes.filter((l) => l.imageUrl).map((l) => `${l.key}\n${l.imageUrl}`).join('\n\n')
  const [images, setImages] = useState<ReadonlyMap<string, HTMLImageElement>>(new Map())
  useEffect(() => {
    let live = true
    setImages(new Map())
    for (const entry of pictureKey ? pictureKey.split('\n\n') : []) {
      const [key, url] = entry.split('\n')
      const img = new Image()
      img.onload = () => { if (live) setImages((prev) => new Map(prev).set(key, img)) }
      img.src = url
    }
    return () => { live = false }
  }, [pictureKey])

  useEffect(() => {
    if (!focus) return
    const span = Math.min(full.end, Math.max(1, (focus.end - focus.start) * 3))
    const centre = (focus.start + focus.end) / 2
    const start = Math.max(0, Math.min(full.end - span, centre - span / 2))
    setDomain({ start, end: start + span })
    // the focus key identifies a request to zoom
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus?.key])

  useEffect(() => {
    const c = canvas.current
    const ctx = c?.getContext('2d')
    if (!c || !ctx || !palette || width <= 0 || height <= 0) return
    const dpr = window.devicePixelRatio || 1
    c.width = Math.round(width * dpr)
    c.height = Math.round(height * dpr)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    drawLanes(ctx, { rows, width, height, domain, palette, selected, draft, images })
  // Switching back from the table creates a new canvas even when its measured size is unchanged.
  }, [rows, width, height, domain, palette, selected, draft, table, images])

  const x = xScale(domain, width)
  const tOf = tScale(domain, width)
  const clampDomain = (start: number, end: number): Domain => {
    const span = Math.min(full.end, Math.max(MIN_SPAN_S, end - start))
    const s = Math.max(0, Math.min(full.end - span, start))
    return { start: s, end: s + span }
  }
  const zoom = (factor: number, at?: number) => {
    const pivot = at ?? (time != null && time >= domain.start && time <= domain.end ? time : (domain.start + domain.end) / 2)
    const u = (pivot - domain.start) / (domain.end - domain.start || 1)
    const span = (domain.end - domain.start) * factor
    setDomain(clampDomain(pivot - u * span, pivot - u * span + span))
  }

  // Ctrl/⌘ + wheel zooms at the pointer; a horizontal wheel pans.
  useEffect(() => {
    const el = wrap.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      const r = el.getBoundingClientRect()
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault()
        zoom(Math.exp(e.deltaY * 0.01), tOf(e.clientX - r.left))
      } else if (Math.abs(e.deltaX) > Math.abs(e.deltaY) && domain.end - domain.start < full.end) {
        e.preventDefault()
        const dt = (e.deltaX / Math.max(1, width)) * (domain.end - domain.start)
        setDomain(clampDomain(domain.start + dt, domain.end + dt))
      }
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  })

  const drag = useRef<{ x: number; t: number; shift: boolean; moved: boolean } | null>(null)
  const localX = (e: PointerEvent<HTMLElement>) => e.clientX - e.currentTarget.getBoundingClientRect().left
  const rowAt = (y: number): LaneRow | undefined => rows.find((r) => y >= r.y && y <= r.y + r.h)
  const onDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return
    e.currentTarget.setPointerCapture(e.pointerId)
    const px = localX(e)
    drag.current = { x: px, t: tOf(px), shift: e.shiftKey && !!onInterval, moved: false }
  }
  const onMove = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current
    if (!d) return
    const px = localX(e)
    if (!d.moved && Math.abs(px - d.x) > DRAG_PX) d.moved = true
    if (d.moved && d.shift) setDraft({ start: d.t, end: tOf(px) })
  }
  const onUp = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current
    drag.current = null
    if (!d) return
    const px = localX(e)
    const t = Math.max(0, Math.min(full.end, tOf(px)))
    setDraft(null)
    if (d.shift && d.moved) {
      onInterval?.(Math.max(0, Math.min(d.t, t)), Math.min(full.end, Math.max(d.t, t)))
    } else if (e.shiftKey && onSelectSegment) {
      const y = e.clientY - e.currentTarget.getBoundingClientRect().top
      const inRow = rowAt(y)?.lane.segments.find((s) => s.start <= t && t <= s.end)
      onSelectSegment(inRow ?? model.segments.find((s) => s.start <= t && t <= s.end) ?? null)
    } else if (!d.moved) {
      onSeek(t)
    }
  }

  const brushStart = useRef<number | null>(null)
  const onAxisDown = (e: PointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) return
    e.currentTarget.setPointerCapture(e.pointerId)
    brushStart.current = e.clientX - e.currentTarget.getBoundingClientRect().left
  }
  const onAxisMove = (e: PointerEvent<SVGSVGElement>) => {
    if (brushStart.current == null) return
    const px = e.clientX - e.currentTarget.getBoundingClientRect().left
    if (Math.abs(px - brushStart.current) > DRAG_PX) setBrush({ start: tOf(brushStart.current), end: tOf(px) })
  }
  const onAxisUp = (e: PointerEvent<SVGSVGElement>) => {
    const from = brushStart.current
    brushStart.current = null
    const px = e.clientX - e.currentTarget.getBoundingClientRect().left
    if (brush) setDomain(clampDomain(Math.min(brush.start, brush.end), Math.max(brush.start, brush.end)))
    else if (from != null) onSeek(Math.max(0, Math.min(full.end, tOf(px))))
    setBrush(null)
  }

  const span = domain.end - domain.start
  const step = tickStep(span)
  const ticks: number[] = []
  for (let t = Math.ceil(domain.start / step - 1e-9) * step; t <= domain.end + 1e-9; t += step) ticks.push(Math.round(t * 1000) / 1000)
  const callouts = model.segments.filter((s) => s.start >= domain.start && s.start <= domain.end)
  const playX = time != null && time >= domain.start && time <= domain.end ? x(time) : null
  const zoomed = span < full.end - 1e-6
  const summary = `Timeline lanes. ${lanes.map(describeLane).join('. ')}.`

  return (
    <div data-testid="timeline" className={cx('min-w-0', className)}>
      <div className="flex flex-wrap items-center gap-x-2 min-h-8">
        <h2 className="m-0 eyebrow">Timeline</h2>
        <span data-testid="timeline-domain" className="ml-2 font-mono text-micro text-text-muted num">{timecode(domain.start)} – {timecode(domain.end)}</span>
        <div role="toolbar" aria-label="Timeline view" className="ml-auto flex items-center gap-0.5">
          {!table && (
            <>
              <button type="button" className={tool} onClick={() => zoom(2)} disabled={!zoomed} aria-label="Zoom the timeline out" title="Zoom out (Ctrl or ⌘ + wheel)"><Minus size={14} strokeWidth={1.5} aria-hidden /></button>
              <button type="button" className={tool} onClick={() => zoom(0.5)} aria-label="Zoom the timeline in" title="Zoom in (Ctrl or ⌘ + wheel, or drag on the axis)"><Plus size={14} strokeWidth={1.5} aria-hidden /></button>
              <button type="button" className={tool} onClick={() => setDomain(full)} disabled={!zoomed} title={`Show the whole ${medium} (double-click the axis)`}>Fit</button>
              <span aria-hidden className="mx-1 h-4 w-px bg-border" />
            </>
          )}
          <button type="button" className={cx(tool, table && 'bg-surface-3 text-text')} aria-pressed={table} onClick={() => setTable((v) => !v)}>View as table</button>
        </div>
      </div>

      {model.inspectionReason && <p data-testid="timeline-inspection" className="mt-2 mb-2 text-small text-text-muted">{videoInspectionCaption(model.inspectionReason)}</p>}

      {table ? (
        <table data-testid="lanes-table" className="mt-2 w-full border-collapse text-small">
          <caption className="sr-only">Timeline lanes as a table: segments, not-analysed spans, lanes that did not run and notes</caption>
          <thead>
            <tr className="text-left text-text-muted border-b border-border">
              <th scope="col" className="py-1.5 pr-4 font-medium">Lane</th><th scope="col" className="py-1.5 pr-4 font-medium">Start</th>
              <th scope="col" className="py-1.5 pr-4 font-medium">End</th><th scope="col" className="py-1.5 pr-4 font-medium">Value</th>
              <th scope="col" className="py-1.5 font-medium">Status</th>
            </tr>
          </thead>
          <tbody className="font-mono text-mono num">
            {lanes.flatMap((l) => [
              ...l.segments.map((s) => <tr key={s.key} className="border-b border-border"><td className="py-1 pr-4 font-sans text-text">{l.label}{l.trackId && l.kind !== 'face' ? ` · ${s.trackName}` : ''}</td><td className="py-1 pr-4">{timecode(s.start)}</td><td className="py-1 pr-4">{timecode(s.end)}</td><td className="py-1 pr-4">{l.inspectionOnly ? '—' : s.peak.toFixed(2)}</td><td className="py-1 font-sans">segment {s.letter}, {l.inspectionOnly ? 'for inspection only' : 'flagged'}</td></tr>),
              ...(l.samples ?? []).map(([t, p], i) => <tr key={`${l.key}-sample-${i}`} className="border-b border-border"><td className="py-1 pr-4 font-sans text-text">{l.label}</td><td className="py-1 pr-4">{timecode(t)}</td><td className="py-1 pr-4">—</td><td className="py-1 pr-4">{l.inspectionOnly ? '—' : p.toFixed(2)}</td><td className="py-1 font-sans">sampled value{l.inspectionOnly ? ', for inspection only' : ''}; no interval implied</td></tr>),
              ...(!l.segments.length ? l.spans.map((s, i) => <tr key={`${l.key}-window-${i}`} className="border-b border-border"><td className="py-1 pr-4 font-sans text-text">{l.label}</td><td className="py-1 pr-4">{timecode(s.start)}</td><td className="py-1 pr-4">{timecode(s.end)}</td><td className="py-1 pr-4">{l.inspectionOnly ? '—' : s.p.toFixed(2)}</td><td className="py-1 font-sans">scored window{l.inspectionOnly ? ', for inspection only' : ''}</td></tr>) : []),
              ...l.gaps.map((g, i) => <tr key={`${l.key}-gap-${i}`} className="border-b border-border"><td className="py-1 pr-4 font-sans text-text">{l.label}</td><td className="py-1 pr-4">{timecode(g.start)}</td><td className="py-1 pr-4">{timecode(g.end)}</td><td className="py-1 pr-4">—</td><td className="py-1 font-sans">{g.label}</td></tr>),
              ...(l.notes ?? []).map((n) => <tr key={n.id} className="border-b border-border"><td className="py-1 pr-4 font-sans text-text">Notes</td><td className="py-1 pr-4">{timecode(n.start)}</td><td className="py-1 pr-4">{timecode(n.end)}</td><td className="py-1 pr-4 font-sans">{n.label}</td><td className="py-1 font-sans">{n.body}</td></tr>),
              ...(l.state === 'not_run' ? [<tr key={`${l.key}-not-run`} className="border-b border-border"><td className="py-1 pr-4 font-sans text-text">{l.label}</td><td className="py-1 pr-4">—</td><td className="py-1 pr-4">—</td><td className="py-1 pr-4">—</td><td className="py-1 font-sans">not run: {l.reason}</td></tr>] : []),
              ...(l.state === 'ready' && !l.segments.length && !l.gaps.length && !l.notes?.length && !l.samples?.length && !l.spans.length ? [<tr key={`${l.key}-summary`} className="border-b border-border"><td className="py-1 pr-4 font-sans text-text">{l.label}</td><td className="py-1 pr-4">{timecode(0)}</td><td className="py-1 pr-4">{timecode(model.duration)}</td><td className="py-1 pr-4">{l.spans.length && !l.inspectionOnly ? Math.max(...l.spans.map((s) => s.p)).toFixed(2) : '—'}</td><td className="py-1 font-sans">{l.inspectionOnly ? 'values shown for inspection only' : l.badge ?? (l.spans.length ? 'no segment' : l.detail)}</td></tr>] : []),
            ])}
          </tbody>
        </table>
      ) : (
        <div className="mt-1 grid grid-cols-[184px_minmax(0,1fr)] max-[639px]:grid-cols-[96px_minmax(0,1fr)]">
          <span aria-hidden />
          <svg width={Math.max(0, width)} height={AXIS_H} className="block overflow-visible cursor-col-resize touch-none select-none" aria-hidden
            onPointerDown={onAxisDown} onPointerMove={onAxisMove} onPointerUp={onAxisUp} onDoubleClick={() => setDomain(full)}>
            {callouts.reduce<{ x: number; segments: VideoSegment[] }[]>((groups, s) => {
              // Segments starting within 12 px of each other share one callout ("a b"), so letters never overprint.
              const last = groups[groups.length - 1]
              if (last && Math.abs(x(s.start) - last.x) < 12) last.segments.push(s)
              else groups.push({ x: x(s.start), segments: [s] })
              return groups
            }, []).map((g) => (
              <text key={g.segments[0].key} x={g.x} y={10} textAnchor="middle" className="font-mono text-micro">
                {g.segments.map((s, i) => (
                  <tspan key={s.key} data-callout={s.letter} style={{ fill: selected?.key === s.key ? 'var(--color-selection)' : 'var(--color-text)' }}>{i ? ` ${s.letter}` : s.letter}</tspan>
                ))}
              </text>
            ))}
            {ticks.map((t) => (
              <g key={t}>
                <text x={x(t)} y={25} textAnchor={x(t) < 12 ? 'start' : x(t) > width - 16 ? 'end' : 'middle'} className="font-mono text-micro" style={{ fill: 'var(--color-text-muted)' }}>{step < 1 ? t.toFixed(1) : t}s</text>
                <rect x={Math.min(width - 1, x(t))} y={29} width={1} height={4} style={{ fill: 'var(--color-graphic)' }} />
              </g>
            ))}
            <rect y={33} width={Math.max(0, width)} height={1} style={{ fill: 'var(--color-graphic)' }} />
            {brush && <rect x={x(Math.min(brush.start, brush.end))} y={14} width={Math.abs(x(brush.end) - x(brush.start))} height={20} style={{ fill: 'var(--color-selection)', opacity: 0.18 }} />}
            {playX != null && <path data-mark="playhead" d={`M${playX - 3},34 L${playX + 3},34 L${playX},40 Z`} style={{ fill: 'var(--color-text)' }} />}
          </svg>
          <div className="relative" style={{ height }}>
            {rows.map(({ lane, y, h }) => (
              <div key={lane.key} data-lane={lane.kind} data-lane-key={lane.key} data-state={lane.state} data-picture={lane.imageUrl ? (images.has(lane.key) ? 'loaded' : 'pending') : undefined} className={cx('absolute left-0 right-0 flex items-center min-w-0', laneAxisLabels(lane, h).length ? 'pr-16 max-[639px]:pr-3' : 'pr-3')} style={{ top: y, height: h }} title={`${lane.label}: ${lane.detail}`}>
                {laneAxisLabels(lane, h).map((axis) => <span key={axis.text} data-testid="lane-value-axis" aria-hidden
                  className="absolute right-2 -translate-y-1/2 font-mono text-micro text-text-muted max-[639px]:hidden" style={{ top: axis.y }}>{axis.text}</span>)}
                {lane.kind === 'estimate' && lane.state === 'ready' && !lane.inspectionOnly && !compact ? (
                  <button type="button" aria-expanded={expanded} onClick={() => setExpanded((v) => !v)} className="min-w-0 text-left" title="Expand the estimate into a line chart">
                    <span className="block text-small leading-4 text-text truncate">{lane.label} <span aria-hidden className="text-text-muted">{expanded ? '▾' : '▸'}</span></span>
                    <span data-testid="lane-detail" className="block text-micro leading-3 text-text-muted whitespace-nowrap max-[639px]:hidden">{gutterDetail(lane.detail)}</span>
                  </button>
                ) : (
                  <span className="min-w-0">
                    <span className="block text-small leading-4 text-text truncate">{lane.label}</span>
                    {h >= 28 && <span data-testid="lane-detail" className="block text-micro leading-3 text-text-muted whitespace-nowrap max-[639px]:hidden">{gutterDetail(lane.detail)}</span>}
                  </span>
                )}
              </div>
            ))}
          </div>
          <div ref={wrap} className={cx('relative min-w-0 touch-none', onInterval ? 'cursor-pointer' : 'cursor-default')} style={{ height }}
            onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={() => { drag.current = null; setDraft(null) }}>
            <canvas ref={canvas} data-testid="lanes-canvas" role="img" aria-label={summary} className="block" style={{ width, height }} />
            {playX != null && <span aria-hidden data-testid="lanes-playhead" className="absolute top-0 bottom-0 w-px bg-text pointer-events-none" style={{ left: Math.round(playX) }} />}
          </div>
        </div>
      )}
      {!table && <details data-testid="lane-key" className="mt-3 text-small text-text-muted">
        <summary className="cursor-pointer w-fit hover:text-text">Lane key</summary>
        <dl className="mt-2 mb-0 grid grid-cols-[minmax(72px,auto)_minmax(0,1fr)] gap-x-4 gap-y-1 text-caption">
          {lanes.map((lane) => <div key={lane.key} className="contents">
            <dt className="font-medium text-text">{lane.label}</dt>
            <dd className="m-0">{lane.detail}{lane.explain ? ` · ${lane.explain}` : ''}{lane.kind === 'lip_sync' && lane.curve?.length ? ` · scale −200 to +200 ms; shaded reference ±80 ms${lane.curve.some((point) => point.confidence != null) ? '; bottom strip shows confidence' : ''}` : ''}{lane.state === 'not_run' && lane.reason ? ` · not run: ${lane.reason}` : ''}</dd>
          </div>)}
        </dl>
      </details>}
      {!table && !compact && <p className="mt-2 mb-0 text-micro text-text-muted">Click to seek · shift-click selects a segment{onInterval ? ' · shift-drag adds an interval note' : ''}{!model.inspectionReason && <> · lane scores and segment thresholds remain uncalibrated; headline calibration does not apply to them.</>}</p>}
    </div>
  )
}
