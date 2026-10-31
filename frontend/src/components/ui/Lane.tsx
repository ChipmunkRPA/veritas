import { useEffect, useRef, useState } from 'react'
import { cx } from '../../lib/cx'
import { HatchPattern, useSvgId } from './Hatch'

/*
 * Static SVG timeline lane (UX §8.4 "Timeline lanes"): segment spans by verdict step, a 3 px flag rule at step >= 4,
 * hatched "not analysed" spans with their label, a sampled rug, playhead and selected-segment outline.
 * The interactive canvas lane stack (E-5) reuses the same tokens and `layoutSegments`.
 */

export type LaneStep = 1 | 2 | 3 | 4 | 5
export type LaneSegment = { start: number; end: number; step: LaneStep; id?: string }
export type LaneGap = { start: number; end: number; label: string }
export type LaneState = 'ready' | 'loading' | 'not_run' | 'no_data'
export type SegmentRect = { x: number; w: number; step: LaneStep; ids: string[] }

export const MIN_SEGMENT_PX = 2
const TINT = ['--color-verdict-1-tint', '--color-verdict-2-tint', '--color-verdict-3-tint', '--color-verdict-4-tint', '--color-verdict-5-tint'] as const
const SOLID = ['--color-verdict-1', '--color-verdict-2', '--color-verdict-3', '--color-verdict-4', '--color-verdict-5'] as const

/** Segment spans in px: minimum drawn width 2 px; runs that collide after widening merge by their maximum step. */
export function layoutSegments(segments: readonly LaneSegment[], duration: number, width: number): SegmentRect[] {
  if (!(duration > 0) || !(width > 0)) return []
  const px = (t: number) => (Math.min(duration, Math.max(0, t)) / duration) * width
  const rects: SegmentRect[] = []
  for (const s of [...segments].sort((a, b) => a.start - b.start)) {
    const x1 = px(s.end)
    const w = Math.max(MIN_SEGMENT_PX, x1 - px(s.start))
    const x0 = Math.min(px(s.start), width - w)
    if (s.end < s.start) continue
    const prev = rects[rects.length - 1]
    if (prev && x0 < prev.x + prev.w) {
      prev.w = Math.max(prev.x + prev.w, x0 + w) - prev.x
      prev.step = Math.max(prev.step, s.step) as LaneStep
      if (s.id) prev.ids.push(s.id)
    } else {
      rects.push({ x: x0, w, step: s.step, ids: s.id ? [s.id] : [] })
    }
  }
  return rects
}

/** Axis label step so that at most `maxLabels` labels fit. */
export function niceStep(duration: number, maxLabels = 10): number {
  for (const s of [1, 2, 5, 10, 15, 30, 60, 120, 300, 600]) if (duration / s <= maxLabels) return s
  return 1200
}

function useWidth(fallback: number) {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(fallback)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(1, Math.floor(entry.contentRect.width))))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, width] as const
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`

export function Lane({ label, duration, segments = [], gaps = [], rug, playhead, selected, state = 'ready', reason, height = 28, className }: {
  label: string
  duration: number
  segments?: readonly LaneSegment[]
  gaps?: readonly LaneGap[]
  rug?: readonly number[]
  playhead?: number | null
  selected?: string | null
  state?: LaneState
  reason?: string
  height?: number
  className?: string
}) {
  const [ref, width] = useWidth(640)
  const hatchId = useSvgId('lane-hatch')
  const rects = state === 'ready' ? layoutSegments(segments, duration, width) : []
  const px = (t: number) => (duration > 0 ? (Math.min(duration, Math.max(0, t)) / duration) * width : 0)
  const flagged = rects.filter((r) => r.step >= 4).length
  const summary = state === 'loading' ? `${label}: loading`
    : state === 'not_run' ? `${label}: not run${reason ? `, ${reason}` : ''}`
      : state === 'no_data' ? `${label}: no data`
        : [plural(rects.length, 'segment'), flagged ? `${flagged} flagged` : null, gaps.length ? `not analysed: ${gaps.map((g) => g.label).join(', ')}` : null]
          .filter(Boolean).join('; ')
  const textStyle = { fill: 'var(--color-text-muted)', paintOrder: 'stroke' as const, stroke: 'var(--color-lane)', strokeWidth: 4 }

  return (
    <div role="group" aria-label={label} data-state={state} className={cx('grid grid-cols-[var(--lane-gutter)_minmax(0,1fr)] items-center', className)}>
      <span className="text-small text-text truncate pr-3">{label}</span>
      <div ref={ref} className="min-w-0">
        <svg width={width} height={height} role="img" aria-label={summary} className="block">
          <defs><HatchPattern id={hatchId} /></defs>
          <rect width={width} height={height} style={{ fill: 'var(--color-lane)' }} className={state === 'loading' ? 'animate-pulse-skeleton' : undefined} />
          {rects.map((r) => (
            <g key={`s-${r.x}`} data-step={r.step}>
              <rect x={r.x} width={r.w} height={height} style={{ fill: `var(${r.step >= 4 ? SOLID[r.step - 1] : TINT[r.step - 1]})` }} />
              {r.step >= 4 && <rect x={r.x} width={r.w} height={3} style={{ fill: 'var(--color-lane-flag)' }} />}
              {selected && r.ids.includes(selected) && (
                <rect x={r.x + 0.75} y={0.75} width={Math.max(0, r.w - 1.5)} height={height - 1.5} fill="none" strokeWidth={1.5} style={{ stroke: 'var(--color-selection)' }} />
              )}
            </g>
          ))}
          {state === 'ready' && gaps.map((g) => {
            const x = px(g.start)
            const w = Math.max(MIN_SEGMENT_PX, px(g.end) - x)
            return (
              <g key={`g-${g.start}`} data-gap={g.label}>
                <title>{g.label}</title>
                <rect x={x} width={w} height={height} fill={`url(#${hatchId})`} />
                {w >= 80 && <text x={x + w / 2} y={height / 2} dominantBaseline="central" textAnchor="middle" className="text-micro" style={textStyle}>{g.label}</text>}
              </g>
            )
          })}
          {state === 'not_run' && (
            <>
              <rect width={width} height={height} fill={`url(#${hatchId})`} />
              <text x={8} y={height / 2} dominantBaseline="central" className="text-micro" style={textStyle}>{reason ? `Not run · ${reason}` : 'Not run'}</text>
            </>
          )}
          {state === 'no_data' && <text x={8} y={height / 2} dominantBaseline="central" className="text-micro" style={textStyle}>No data</text>}
          {state === 'ready' && rug?.map((t) => <rect key={`r-${t}`} x={px(t)} y={height - 6} width={1} height={6} style={{ fill: 'var(--color-text-muted)' }} />)}
          {state === 'ready' && playhead != null && <rect data-mark="playhead" x={px(playhead)} width={1} height={height} style={{ fill: 'var(--color-text)' }} />}
          <rect y={height - 1} width={width} height={1} style={{ fill: 'var(--color-lane-frame)' }} />
        </svg>
      </div>
    </div>
  )
}

/** Time axis for a lane stack: labels at a nice step in 11 px mono, playhead triangle, segment callout letters. */
export function LaneAxis({ duration, playhead, callouts = [], className }: {
  duration: number
  playhead?: number | null
  callouts?: readonly { t: number; id: string }[]
  className?: string
}) {
  const [ref, width] = useWidth(640)
  const step = niceStep(duration)
  const ticks = duration > 0 ? Array.from({ length: Math.floor(duration / step) + 1 }, (_, i) => i * step) : []
  const px = (t: number) => (duration > 0 ? (Math.min(duration, Math.max(0, t)) / duration) * width : 0)
  return (
    <div aria-hidden className={cx('grid grid-cols-[var(--lane-gutter)_minmax(0,1fr)]', className)}>
      <span />
      <div ref={ref} className="min-w-0">
        {/* Rows: callout letters (y 0–12), time labels (14–26), ticks and axis rule (29–34), playhead triangle (34–40). */}
        <svg width={width} height={40} className="block overflow-visible text-text-muted">
          {callouts.map((c) => <text key={c.id} x={px(c.t)} y={10} textAnchor="middle" className="font-mono text-micro" style={{ fill: 'var(--color-text)' }}>{c.id}</text>)}
          {ticks.map((t) => (
            <g key={t}>
              <text x={px(t)} y={25} textAnchor={t === 0 ? 'start' : px(t) > width - 16 ? 'end' : 'middle'} className="font-mono text-micro" fill="currentColor">{t}s</text>
              <rect x={Math.min(width - 1, px(t))} y={29} width={1} height={4} style={{ fill: 'var(--color-graphic)' }} />
            </g>
          ))}
          <rect y={33} width={width} height={1} style={{ fill: 'var(--color-graphic)' }} />
          {playhead != null && <path d={`M${px(playhead) - 3},34 L${px(playhead) + 3},34 L${px(playhead)},40 Z`} style={{ fill: 'var(--color-text)' }} />}
        </svg>
      </div>
    </div>
  )
}
