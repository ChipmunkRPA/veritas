/**
 * Viewer deep-link state (ARCHITECTURE §7.2 "Viewer deep links", UX §5.9 "Deep links").
 *
 * `?exhibit=<id>&layers=heatmap:<id>@70,reliability,regions:<id>,faces:<id>,clones:<id>,annotations&thr=0.5&z=2.5&cx=1024&cy=640`
 * `&region=R1&finding=F2&compare=swipe&swipe=50`
 *
 * - The view lives in the URL so notes and reports can link to an exact view; hover, drag and the loupe stay in component state.
 * - `layers` absent means "the preset of the analysis" (UX §5.5: the lens the top finding needs); `layers=` (empty) is "no lenses".
 * - At most one map lens (UX §5.9 "Combinations"); marks stack. Parsing is tolerant: malformed parts are dropped, numbers clamped.
 */
import { MAX_ZOOM, MIN_ZOOM, type Box } from '../../lib/coords'
import { SHOW_THRESHOLD_MAX, SHOW_THRESHOLD_OVERLAY } from '../../lib/colormap'

export type MarkKind = 'reliability' | 'regions' | 'faces' | 'clones' | 'annotations'
export type CompareMode = 'overlay' | 'swipe' | 'side' | 'blink'

export interface MarkLayer { kind: MarkKind; id: string | null }
export interface MapLayer { id: string; opacity: number }
export interface Layers { map: MapLayer | null; marks: MarkLayer[] }

export interface ViewerState {
  exhibit: string | null
  layers: Layers | null
  thr: number
  z: number | null
  cx: number | null
  cy: number | null
  region: string | null
  finding: string | null
  compare: CompareMode
  swipe: number
  /** Exact source-pixel position opened from an image evidence card. */
  focus?: Box | null
}

export const DEFAULT_OPACITY = 70
export const DEFAULT_SWIPE = 50
const MARKS: readonly MarkKind[] = ['reliability', 'regions', 'faces', 'clones', 'annotations']
const MARKS_WITH_ID: ReadonlySet<MarkKind> = new Set(['regions', 'faces', 'clones'])
const COMPARE: readonly CompareMode[] = ['overlay', 'swipe', 'side', 'blink']
const ID = /^[A-Za-z0-9_.-]{1,160}$/
const LABEL = /^[^\s,]{1,32}$/

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
const num = (s: string | null): number | null => {
  if (s === null || s.trim() === '') return null
  const v = Number(s)
  return Number.isFinite(v) ? v : null
}

/** `Show ≥` runs 0–0.9 in steps of 0.05 (UX §8.3). */
export const snapThreshold = (t: number): number => clamp(Math.round(t * 20) / 20, 0, SHOW_THRESHOLD_MAX)

export function parseLayers(raw: string): Layers {
  let map: MapLayer | null = null
  const marks: MarkLayer[] = []
  for (const part of raw.split(',')) {
    const [head, opacityText] = part.split('@')
    const [kind, id = null] = head.split(':') as [string, string | undefined]
    if (kind === 'heatmap') {
      if (map || id === null || !ID.test(id)) continue
      const op = num(opacityText ?? null)
      map = { id, opacity: op === null ? DEFAULT_OPACITY : clamp(Math.round(op), 0, 100) }
    } else if ((MARKS as readonly string[]).includes(kind)) {
      const k = kind as MarkKind
      const mark = { kind: k, id: MARKS_WITH_ID.has(k) && id !== null && ID.test(id) ? id : null }
      if (MARKS_WITH_ID.has(k) && mark.id === null) continue
      if (!marks.some((m) => m.kind === mark.kind && m.id === mark.id)) marks.push(mark)
    }
  }
  return { map, marks }
}

export function serialiseLayers(layers: Layers): string {
  const parts: string[] = []
  if (layers.map) parts.push(`heatmap:${layers.map.id}@${Math.round(layers.map.opacity)}`)
  for (const m of layers.marks) parts.push(m.id ? `${m.kind}:${m.id}` : m.kind)
  return parts.join(',')
}

export function parseViewerState(params: URLSearchParams): ViewerState {
  const z = num(params.get('z'))
  const cx = num(params.get('cx'))
  const cy = num(params.get('cy'))
  const thr = num(params.get('thr'))
  const swipe = num(params.get('swipe'))
  const compare = params.get('compare')
  const exhibit = params.get('exhibit')
  const region = params.get('region')
  const finding = params.get('finding')
  const focusValues = params.get('focus')?.split(',').map(Number)
  const focus = focusValues?.length === 4 && focusValues.every(Number.isFinite)
    && focusValues[0] >= 0 && focusValues[1] >= 0 && focusValues[2] > 0 && focusValues[3] > 0
    ? { x: focusValues[0], y: focusValues[1], w: focusValues[2], h: focusValues[3] } : null
  return {
    exhibit: exhibit && ID.test(exhibit) ? exhibit : null,
    layers: params.has('layers') ? parseLayers(params.get('layers') ?? '') : null,
    thr: thr === null ? SHOW_THRESHOLD_OVERLAY : snapThreshold(thr),
    z: z === null || z <= 0 ? null : clamp(z, MIN_ZOOM, MAX_ZOOM),
    cx: cx === null || cx < 0 ? null : cx,
    cy: cy === null || cy < 0 ? null : cy,
    region: region && LABEL.test(region) ? region : null,
    finding: finding && LABEL.test(finding) ? finding : null,
    compare: (COMPARE as readonly string[]).includes(compare ?? '') ? (compare as CompareMode) : 'overlay',
    swipe: swipe === null ? DEFAULT_SWIPE : clamp(swipe, 0, 100),
    focus,
  }
}

const trim = (v: number, dp: number) => String(Math.round(v * 10 ** dp) / 10 ** dp)

/** Stable parameter order, defaults omitted (`thr` 0.5, `compare` overlay, `swipe` 50). */
export function serialiseViewerState(s: ViewerState): URLSearchParams {
  const p = new URLSearchParams()
  if (s.exhibit) p.set('exhibit', s.exhibit)
  if (s.layers) p.set('layers', serialiseLayers(s.layers))
  if (s.thr !== SHOW_THRESHOLD_OVERLAY) p.set('thr', trim(s.thr, 2))
  if (s.z !== null) p.set('z', trim(s.z, 3))
  if (s.cx !== null) p.set('cx', trim(s.cx, 1))
  if (s.cy !== null) p.set('cy', trim(s.cy, 1))
  if (s.region) p.set('region', s.region)
  if (s.finding) p.set('finding', s.finding)
  if (s.focus) p.set('focus', [s.focus.x, s.focus.y, s.focus.w, s.focus.h].map((v) => trim(v, 3)).join(','))
  if (s.compare !== 'overlay') p.set('compare', s.compare)
  if (s.compare === 'swipe' && s.swipe !== DEFAULT_SWIPE) p.set('swipe', trim(s.swipe, 0))
  return p
}
