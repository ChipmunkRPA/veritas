import { useCallback, useEffect, useMemo, useReducer, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { Link, Navigate, useParams, useSearchParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Minus, MoreHorizontal, PenLine, Plus, Search } from 'lucide-react'
import { api, hasApiKey } from '../../lib/api'
import { apiClient, unwrap } from '../../lib/api/client'
import { qk, useAnalysis, useAnnotations, useExhibitPayloads, useExhibits, useFindings } from '../../lib/api/queries'
import {
  centredView, containsPoint, fitView, frameBox, gridCell, normaliseBox, panBy, pixelAt, stepZoom, toOriginal, viewCentre, zoomAt,
  type Box, type Point, type Size, type View,
} from '../../lib/coords'
import { cx } from '../../lib/cx'
import { useMediaQuery } from '../../lib/useMediaQuery'
import { prefersReducedMotion } from '../../lib/theme'
import { ErrorState, Menu, Skeleton } from '../../components/ui'
import MediaAccessNotice from '../../components/MediaAccessNotice'
import AnnotationDialog, { type AnnotationDraft } from './AnnotationDialog'
import CompareSlider from './CompareSlider'
import { probeMap } from './heatmap'
import { Inspector, LensRail, type Readout } from './LayerPanel'
import { LENSES, activeLenses, allMaps, annotationItems, catalogue, describeRegions, heatmapLens, lensByLetter, mapById, presetLayers, regionItems, toggleLayers, type LensKey, type RegionItem } from './lenses'
import Loupe, { LOUPE_FACTORS } from './Loupe'
import RegionsOverlay from './RegionsOverlay'
import type { Scene } from './scene'
import { useBase } from './useBase'
import { useHeatmaps } from './useHeatmaps'
import ViewportCanvas from './ViewportCanvas'
import { DEFAULT_OPACITY, parseViewerState, serialiseViewerState, snapThreshold, type CompareMode, type Layers, type ViewerState } from './viewerState'
import VideoWorkspace from '../video/VideoWorkspace'

const COMPARE: { id: CompareMode; label: string }[] = [
  { id: 'overlay', label: 'Overlay' }, { id: 'swipe', label: 'Swipe' }, { id: 'side', label: 'Side by side' }, { id: 'blink', label: 'Blink' },
]
const PAN_STEP = 50
const PAN_STEP_LARGE = 250
const CROSSHAIR_STEP = 10
const CROSSHAIR_STEP_LARGE = 50
/** The readout updates at ≤ 30 Hz (UX §5.9). */
const PROBE_INTERVAL_MS = 33
const ZOOM_TO_REGION_MS = 320

const signatureOf = (layers: Layers, heatReady: boolean) =>
  `${layers.map?.id ?? '-'}:${layers.map && heatReady ? 1 : 0}|${layers.marks.map((m) => `${m.kind}${m.id ?? ''}`).sort().join(',')}`

const tool = (on = false) => cx('h-8 px-2 inline-flex items-center gap-1.5 rounded-sm text-small transition-colors duration-fast disabled:opacity-40 disabled:cursor-not-allowed',
  on ? 'bg-surface-3 text-text' : 'text-text-muted enabled:hover:text-text enabled:hover:bg-surface-3')

/**
 * Evidence viewer, image workspace (UX §5.9; ARCHITECTURE §7.2, §7.4). Full viewport: toolbar, lens rail, the dark stage and the
 * inspector.
 *
 * - **Rendering.** A canvas (source, heat layer, reliability hatch, pixel grid) and an SVG overlay (regions, clone pairs, faces,
 *   annotations) share one transform in original px. Sources above 4096 px load as tiles.
 * - **Interaction.** Wheel zoom at the cursor (0.1–16×), drag and arrow pan, pinch, the loupe, swipe / side-by-side / blink compare,
 *   the pixel probe, single-key lenses while the viewer has focus, and box annotations posted in original px.
 * - **State.** Lenses, threshold, zoom, centre, selected region and compare mode live in the URL (§7.2 deep links); hover, drag and
 *   the loupe stay local.
 * - **Timing.** `performance` marks `viewer:first-paint` (source complete at its level) and measures `viewer:layer-toggle` (from a
 *   toggle to the first frame drawn with the new layers), mirrored in `data-first-paint-ms` and `data-toggle-ms` on the stage.
 */
// N-19d: the keys of the status line are those of the lens rail, in its order (Regions has a key since N-19d)
const LENS_KEYS = LENSES.map((l) => l.letter).filter(Boolean).join(' ')

export default function EvidenceViewer() {
  const { id = '' } = useParams()
  const q = useAnalysis(id)
  // A recording has no viewer of its own: its exhibit (playback, lanes, segments) is a section of the result page (A-1).
  if (q.data?.kind === 'audio') return <Navigate to={`/analyses/${id}/exhibit`} replace />
  return q.data?.kind === 'video' ? <VideoWorkspace /> : <ImageEvidenceViewer />
}

function ImageEvidenceViewer() {
  const { id = '' } = useParams()
  const [params, setParams] = useSearchParams()
  const state = useMemo(() => parseViewerState(params), [params])
  const qc = useQueryClient()
  const q = useAnalysis(id)
  const a = q.data
  const done = a?.status === 'done'
  const image = a?.kind === 'image'
  const exhibits = useExhibits(id, done && image)
  const findings = useFindings(id, done && image)
  const annotations = useAnnotations(id, done && image)
  const tok = useQuery({ queryKey: qk.downloadToken(id), queryFn: () => api.downloadToken(id), enabled: done && image, staleTime: 5 * 60_000 })
  const token = tok.data ?? ''
  const tokenReady = !hasApiKey || token !== ''
  const evidenceIds = useMemo(() => (exhibits.data ?? []).filter((x) => x.kind === 'heatmap' || x.kind === 'regions').map((x) => x.id), [exhibits.data])
  const payloads = useExhibitPayloads(id, evidenceIds, done && image)
  const payloadKey = payloads.data.map((p) => p.id).join(',')
  // payloads.data is a fresh array on every render; its ids identify it
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const cat = useMemo(() => catalogue(payloads.data), [payloadKey])
  const evidenceReady = exhibits.isSuccess && !payloads.isPending && findings.isFetched

  const update = useCallback((patch: Partial<ViewerState>) => {
    setParams((prev) => serialiseViewerState({ ...parseViewerState(prev), ...patch }), { replace: true })
  }, [setParams])

  const layers = useMemo(() => state.layers ?? presetLayers(cat, findings.data ?? [], state.finding), [state.layers, cat, findings.data, state.finding])
  const activeMap = mapById(cat, layers.map?.id)
  const mapLens = activeMap ? heatmapLens(activeMap) : null
  const active = useMemo(() => activeLenses(cat, layers), [cat, layers])

  // --- media -----------------------------------------------------------------------------------------------
  const [, bump] = useReducer((x: number) => x + 1, 0)
  const bumpFrame = useRef(0)
  const onTile = useCallback(() => {
    if (!bumpFrame.current) bumpFrame.current = requestAnimationFrame(() => { bumpFrame.current = 0; bump() })
  }, [])
  useEffect(() => () => cancelAnimationFrame(bumpFrame.current), [])
  const { base, source, error: baseError } = useBase({ enabled: done && image && tokenReady && evidenceReady, analysisId: id, tiles: cat.tiles, token, onUpdate: onTile })
  const maps = useMemo(() => allMaps(cat), [cat])
  const heat = useHeatmaps(maps, token, done && image && tokenReady, state.thr)
  const decodedActive = activeMap ? heat.decoded[activeMap.id] ?? null : null
  const heatBitmap = heat.layer(activeMap?.id)

  const focusBox = state.focus && source && state.focus.x + state.focus.w <= source.width && state.focus.y + state.focus.h <= source.height ? state.focus : null
  const focusKey = focusBox ? [focusBox.x, focusBox.y, focusBox.w, focusBox.h].join(',') : null
  const items = useMemo(() => {
    const recorded = regionItems(cat, source)
    if (!focusBox) return recorded
    const focused: RegionItem = { key: 'source-focus', label: 'Area', exhibitId: 'source-focus', lens: 'regions', box: focusBox,
      polygon: null, score: null, probability: null, share: null, pair: null, text: 'Source coordinates opened for inspection. This outline is a navigation aid, not a new detection.' }
    return [...recorded, focused]
    // The key contains all geometry, so zoom URL updates do not rebuild the source selection.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cat, source, focusKey])
  const visibleItems = useMemo(() => items.filter((r) => r.key === 'source-focus' || layers.marks.some((m) => m.kind === r.lens && m.id === r.exhibitId)), [items, layers.marks])
  const notes = useMemo(() => annotationItems(annotations.data ?? []), [annotations.data])
  const visibleNotes = active.has('annotations') ? notes : []

  // --- view ------------------------------------------------------------------------------------------------
  const [stageEl, setStageEl] = useState<HTMLDivElement | null>(null)
  const [viewport, setViewport] = useState<Size>({ width: 0, height: 0 })
  const compact = useMediaQuery('(max-width: 639px)')  // the page, not the stage: the toolbar spans the page
  useEffect(() => {
    if (!stageEl) return
    const ro = new ResizeObserver((entries) => {
      const r = entries[0].contentRect
      setViewport({ width: Math.floor(r.width), height: Math.floor(r.height) })
    })
    ro.observe(stageEl)
    return () => ro.disconnect()
  }, [stageEl])
  const pane: Size = state.compare === 'side' ? { width: Math.max(0, Math.floor((viewport.width - 1) / 2)), height: viewport.height } : viewport
  const paneRef = useRef(pane)
  paneRef.current = pane
  const [view, setView] = useState<View | null>(null)
  const viewRef = useRef(view)
  viewRef.current = view
  const fitMode = useRef(true)
  const lastPane = useRef<Size | null>(null)

  useEffect(() => {
    if (!source || pane.width <= 0 || pane.height <= 0) return
    const prev = lastPane.current
    lastPane.current = pane
    setView((v) => {
      if (!v) {
        const target = state.region ? [...items, ...notes].find((r) => r.label === state.region && r.box) : null
        if (state.z !== null) {
          fitMode.current = false
          return centredView(state.z, { x: state.cx ?? source.width / 2, y: state.cy ?? source.height / 2 }, pane)
        }
        if (target?.box) {
          fitMode.current = false
          return frameBox(target.box, pane)
        }
        if (focusBox) {
          fitMode.current = false
          return frameBox(focusBox, pane)
        }
        fitMode.current = true
        return fitView(source, pane)
      }
      if (fitMode.current) return fitView(source, pane)
      if (prev && (prev.width !== pane.width || prev.height !== pane.height)) return centredView(v.scale, viewCentre(v, prev), pane)
      return v
    })
    // runs when the source or the pane size changes; the deep link is read once, on the first view
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source, pane.width, pane.height, focusKey])

  // Mirror zoom and centre into the URL once a gesture settles (fit stays implicit).
  useEffect(() => {
    if (!view || fitMode.current || paneRef.current.width <= 0) return
    const t = window.setTimeout(() => {
      const c = viewCentre(view, paneRef.current)
      update({ z: Math.round(view.scale * 1000) / 1000, cx: Math.max(0, Math.round(c.x * 10) / 10), cy: Math.max(0, Math.round(c.y * 10) / 10) })
    }, 250)
    return () => window.clearTimeout(t)
  }, [view, update])

  const anim = useRef(0)
  useEffect(() => () => cancelAnimationFrame(anim.current), [])
  const animateTo = useCallback((target: View) => {
    cancelAnimationFrame(anim.current)
    fitMode.current = false
    const from = viewRef.current
    if (!from || prefersReducedMotion()) {
      setView(target)
      return
    }
    const p = paneRef.current
    const c0 = viewCentre(from, p)
    const c1 = viewCentre(target, p)
    const t0 = performance.now()
    const step = (now: number) => {
      const k = Math.min(1, (now - t0) / ZOOM_TO_REGION_MS)
      const e = 1 - (1 - k) ** 3
      setView(centredView(from.scale * (target.scale / from.scale) ** e, { x: c0.x + (c1.x - c0.x) * e, y: c0.y + (c1.y - c0.y) * e }, p))
      if (k < 1) anim.current = requestAnimationFrame(step)
    }
    anim.current = requestAnimationFrame(step)
  }, [])

  const zoomBy = (direction: 1 | -1) => {
    fitMode.current = false
    setView((v) => (v ? zoomAt(v, stepZoom(v.scale, direction), { x: pane.width / 2, y: pane.height / 2 }) : v))
  }
  const oneToOne = () => {
    fitMode.current = false
    setView((v) => (v ? zoomAt(v, 1, { x: pane.width / 2, y: pane.height / 2 }) : v))
  }
  const fit = () => {
    if (!source) return
    fitMode.current = true
    setView(fitView(source, pane))
    update({ z: null, cx: null, cy: null })
  }

  // --- timing ----------------------------------------------------------------------------------------------
  const signature = signatureOf(layers, heatBitmap != null)
  const signatureRef = useRef(signature)
  signatureRef.current = signature
  const pendingToggle = useRef<{ start: number; signature: string } | null>(null)
  const firstPaintRef = useRef<number | null>(null)
  const [firstPaint, setFirstPaint] = useState<number | null>(null)
  const [toggleMs, setToggleMs] = useState<number | null>(null)
  useEffect(() => { performance.mark('viewer:mount') }, [])
  const onDrawn = useCallback(({ baseComplete }: { baseComplete: boolean }) => {
    const now = performance.now()
    if (baseComplete && firstPaintRef.current === null) {
      firstPaintRef.current = now
      performance.mark('viewer:first-paint')
      setFirstPaint(Math.round(now))
    }
    const p = pendingToggle.current
    if (p && p.signature === signatureRef.current) {
      pendingToggle.current = null
      performance.measure('viewer:layer-toggle', { start: p.start, end: now })
      setToggleMs(Math.round((now - p.start) * 10) / 10)
    }
  }, [])

  // --- lenses ----------------------------------------------------------------------------------------------
  const setLayers = (next: Layers) => {
    pendingToggle.current = { start: performance.now(), signature: signatureOf(next, true) }
    update({ layers: next })
  }
  const toggle = (key: LensKey) => {
    const next = toggleLayers(cat, layers, key)
    if (next !== layers) setLayers(next)
  }
  const [mapOnly, setMapOnly] = useState(false)
  const [dim, setDim] = useState(false)
  const [blinkLens, setBlinkLens] = useState(true)
  const [loupe, setLoupe] = useState(false)
  const [loupeFactor, setLoupeFactor] = useState<number>(LOUPE_FACTORS[0])
  const [crosshair, setCrosshair] = useState<Point | null>(null)

  /** Select a region or note: its mark turns on when it is off, and the view frames it. */
  const select = (label: string, frame = true) => {
    const target = [...items, ...notes].find((r) => r.label === label && r.box)
    const lens = target && 'lens' in target ? target.lens : 'annotations'
    const marked = layers.marks.some((m) => m.kind === lens)
    update({ region: label, ...(target && !marked ? { layers: toggleLayers(cat, layers, lens) } : {}) })
    if (frame && target?.box) animateTo(frameBox(target.box, pane))
  }

  // --- annotations -----------------------------------------------------------------------------------------
  const locked = a?.locked_at != null
  const [annotating, setAnnotating] = useState(false)
  const [draft, setDraft] = useState<Box | null>(null)
  const [dialog, setDialog] = useState<{ box: Box | null } | null>(null)
  const create = useMutation({
    mutationFn: (d: AnnotationDraft) => unwrap(apiClient.POST('/api/v2/analyses/{aid}/annotations', {
      params: { path: { aid: id } },
      body: { target_type: 'region', shape: 'box', geometry: { x: d.box.x, y: d.box.y, w: d.box.w, h: d.box.h }, body: d.body, visibility: d.visibility },
    })),
    onSuccess: () => {
      setDialog(null)
      setDraft(null)
      setAnnotating(false)
      void qc.invalidateQueries({ queryKey: qk.annotations(id) })
      if (!active.has('annotations')) update({ layers: { ...layers, marks: [...layers.marks, { kind: 'annotations', id: null }] } })
    },
  })
  const closeDialog = () => {
    setDialog(null)
    setDraft(null)
    create.reset()
  }

  // --- pointer ---------------------------------------------------------------------------------------------
  const [pointer, setPointer] = useState<Point | null>(null)
  const lastProbe = useRef(0)
  const drag = useRef<{ start: Point; last: Point; moved: boolean; mode: 'pan' | 'draw'; origin: Point } | null>(null)
  const pinch = useRef<{ dist: number; view: View; mid: Point } | null>(null)
  const pointers = useRef(new Map<number, Point>())
  const latestPoint = useRef<Point | null>(null)
  const probeTimer = useRef<number | null>(null)
  useEffect(() => () => { if (probeTimer.current !== null) window.clearTimeout(probeTimer.current) }, [])
  /** The readout follows the pointer at ≤ 30 Hz with a trailing update, so it always ends on the last position. */
  const probe = (at: Point | null) => {
    latestPoint.current = at
    if (at === null) {
      if (probeTimer.current !== null) window.clearTimeout(probeTimer.current)
      probeTimer.current = null
      setPointer(null)
      return
    }
    const wait = PROBE_INTERVAL_MS - (performance.now() - lastProbe.current)
    if (wait <= 0) {
      lastProbe.current = performance.now()
      setPointer(at)
    } else if (probeTimer.current === null) {
      probeTimer.current = window.setTimeout(() => {
        probeTimer.current = null
        lastProbe.current = performance.now()
        setPointer(latestPoint.current)
      }, wait)
    }
  }
  const local = (e: PointerEvent<HTMLElement>): Point => {
    const r = e.currentTarget.getBoundingClientRect()
    return { x: e.clientX - r.left, y: e.clientY - r.top }
  }

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 || !view || !source) return
    const at = local(e)
    e.currentTarget.setPointerCapture(e.pointerId)
    pointers.current.set(e.pointerId, at)
    stageEl?.focus({ preventScroll: true })
    if (pointers.current.size === 2) {
      const [p, r] = [...pointers.current.values()]
      pinch.current = { dist: Math.hypot(p.x - r.x, p.y - r.y) || 1, view, mid: { x: (p.x + r.x) / 2, y: (p.y + r.y) / 2 } }
      drag.current = null
      return
    }
    drag.current = { start: at, last: at, moved: false, mode: annotating && !locked ? 'draw' : 'pan', origin: toOriginal(view, at) }
  }

  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const at = local(e)
    probe(at)
    if (pointers.current.has(e.pointerId)) pointers.current.set(e.pointerId, at)
    const pz = pinch.current
    if (pz && pointers.current.size === 2) {
      const [p, r] = [...pointers.current.values()]
      fitMode.current = false
      setView(zoomAt(pz.view, pz.view.scale * (Math.hypot(p.x - r.x, p.y - r.y) / pz.dist), pz.mid))
      return
    }
    const d = drag.current
    if (!d || !view || !source) return
    if (!d.moved && Math.hypot(at.x - d.start.x, at.y - d.start.y) > 3) d.moved = true
    if (d.moved && d.mode === 'pan') {
      fitMode.current = false
      const dx = at.x - d.last.x
      const dy = at.y - d.last.y
      setView((v) => (v ? panBy(v, dx, dy) : v))
    }
    if (d.moved && d.mode === 'draw') setDraft(normaliseBox(d.origin, toOriginal(view, at), source))
    d.last = at
  }

  const onPointerUp = (e: PointerEvent<HTMLDivElement>) => {
    pointers.current.delete(e.pointerId)
    if (pointers.current.size < 2) pinch.current = null
    const d = drag.current
    drag.current = null
    if (!d || !view || !source) return
    const at = local(e)
    if (d.mode === 'draw') {
      const box = d.moved ? normaliseBox(d.origin, toOriginal(view, at), source) : null
      if (box && box.w >= 1 && box.h >= 1) {
        setDraft(box)
        setDialog({ box })
      } else {
        setDraft(null)
      }
      return
    }
    if (!d.moved) {
      const o = toOriginal(view, at)
      const hit = [...visibleItems].reverse().find((r) => containsPoint(r.box, o))?.label
        ?? visibleNotes.find((n) => n.box && n.box.w > 0 && containsPoint(n.box, o))?.label
      if (hit) select(hit, false)
    }
  }

  useEffect(() => {
    if (!stageEl) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const paneEl = (e.target as HTMLElement).closest<HTMLElement>('[data-pane]') ?? stageEl
      const r = paneEl.getBoundingClientRect()
      const at = { x: e.clientX - r.left, y: e.clientY - r.top }
      const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015))
      fitMode.current = false
      setView((v) => (v ? zoomAt(v, v.scale * factor, at) : v))
    }
    stageEl.addEventListener('wheel', onWheel, { passive: false })
    return () => stageEl.removeEventListener('wheel', onWheel)
  }, [stageEl])

  // --- keyboard (active while the viewer has focus, UX §8.7 2.1.4) -------------------------------------------
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const t = e.target as HTMLElement
    if (dialog || e.metaKey || e.ctrlKey || e.altKey || t.closest('input, textarea, select, [role="menu"], [role="slider"], dialog')) return
    const k = e.key
    const centre = { x: pane.width / 2, y: pane.height / 2 }
    const lens = k.length === 1 ? lensByLetter(k) : undefined
    if (lens) toggle(lens.key)
    else if (k === '+' || k === '=') zoomBy(1)
    else if (k === '-' || k === '_') zoomBy(-1)
    else if (k === '0') fit()
    else if (k === '1') oneToOne()
    else if (k === 'l' || k === 'L') {
      setLoupe((x) => !x)
      setCrosshair((c) => c ?? centre)
    } else if (k === '[' || k === ']') {
      setLoupeFactor((f) => LOUPE_FACTORS[Math.min(LOUPE_FACTORS.length - 1, Math.max(0, LOUPE_FACTORS.indexOf(f as 4 | 8 | 16) + (k === ']' ? 1 : -1)))])
    } else if ((k === 'b' || k === 'B') && state.compare === 'blink') {
      setBlinkLens((x) => !x)
    } else if (k.startsWith('Arrow')) {
      const dir = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[k] ?? [0, 0]
      if (loupe) {
        const s = e.shiftKey ? CROSSHAIR_STEP_LARGE : CROSSHAIR_STEP
        setPointer(null)
        setCrosshair((c) => {
          const p = c ?? centre
          return { x: Math.min(pane.width, Math.max(0, p.x + dir[0] * s)), y: Math.min(pane.height, Math.max(0, p.y + dir[1] * s)) }
        })
      } else {
        const s = e.shiftKey ? PAN_STEP_LARGE : PAN_STEP
        fitMode.current = false
        setView((v) => (v ? panBy(v, -dir[0] * s, -dir[1] * s) : v))
      }
    } else if (k === 'Escape') {
      if (annotating) {
        setAnnotating(false)
        setDraft(null)
      } else if (state.region) update({ region: null })
      else return
    } else return
    e.preventDefault()
  }

  // --- derived ---------------------------------------------------------------------------------------------
  const probePoint = pointer ?? (loupe ? crosshair : null)
  const readout: Readout | null = useMemo(() => {
    if (!probePoint || !view || !source) return null
    const px = pixelAt(view, probePoint, source)
    if (!px) return null
    const map = decodedActive && activeMap ? probeMap(decodedActive, gridCell(px, source, [decodedActive.width, decodedActive.height]), activeMap.value_range) : null
    const region = [...visibleItems].reverse().find((r) => containsPoint(r.box, px))?.label ?? null
    return { x: px.x, y: px.y, map, region }
  }, [probePoint, view, source, decodedActive, activeMap, visibleItems])

  if (q.isPending) {
    return <div className="p-12" aria-busy="true"><Skeleton w={320} h={20} /><div className="mt-4"><Skeleton w={640} h={360} /></div></div>
  }
  if (q.error || !a) {
    return (
      <div className="p-12">
        <ErrorState onRetry={() => void q.refetch()}>{q.error?.message ?? 'Not found'} <span className="text-text-muted mx-1">·</span> <Link to="/analyses" className="text-accent underline underline-offset-[3px]">Back to analyses</Link></ErrorState>
      </div>
    )
  }
  if (!done || !image) {
    return (
      <div className="p-12 max-w-[640px]">
        <p className="m-0 text-small text-text">
          {!done ? `The analysis is ${a.status}; the evidence viewer opens once it has finished.` : `The evidence viewer shows image analyses; this is a ${a.kind} analysis.`}
        </p>
        <p className="mt-3 mb-0"><Link to={`/analyses/${id}`} className="text-small text-accent underline underline-offset-[3px]">‹ Back to the result</Link></p>
      </div>
    )
  }

  const scene: Scene = {
    source: source ?? { width: 1, height: 1 },
    base,
    layers: { heat: heatBitmap, opacity: (layers.map?.opacity ?? DEFAULT_OPACITY) / 100, unreliable: decodedActive?.unreliable ?? null, hatch: active.has('reliability'), dim, mapOnly },
  }
  const lensPane = state.compare === 'side' ? 1 : 0
  const panes = state.compare === 'side' ? [0, 1] : [0]
  const loupeAt = pointer ?? crosshair
  const zoomText = view ? `${Math.round(view.scale * 100)} %` : '—'
  const stageMessage = !tokenReady ? 'The preview will appear when secure media links are ready.' : baseError ? `The image could not be loaded: ${baseError}` : !source ? 'Loading the image…' : null
  const readOnly = viewport.width > 0 && viewport.width < 640
  const toggleLoupe = () => { setLoupe((x) => !x); setCrosshair((c) => c ?? { x: pane.width / 2, y: pane.height / 2 }) }

  return (
    <div data-testid="evidence-viewer" className="h-full min-h-0 flex flex-col bg-surface" onKeyDown={onKeyDown}>
      {/* N-19d (UX, small screens; design review P2): below 960 px the viewer says what it was designed for */}
      <p data-testid="viewer-small-screen" className="m-0 px-3 py-1.5 border-b border-border bg-surface-2 text-caption text-text-muted min-[960px]:hidden">The evidence viewer is designed for screens of 960 px and wider.</p>
      {/* N-19b (browser QA "Layout"): below 640 px the title has a row of its own under the back link and the tools, so that it is
          not cut to the room they leave (112 to 181 px at 375 px); it takes two lines at most, with the whole name as its hover text */}
      <header className="h-12 shrink-0 flex items-center gap-2 px-4 border-b border-border bg-surface min-w-0 max-[639px]:h-auto max-[639px]:flex-wrap max-[639px]:gap-y-0 max-[639px]:pb-1.5">
        <Link to={`/analyses/${id}`} className="shrink-0 text-small text-text-muted hover:text-text inline-flex items-center min-h-8">‹ Result</Link>
        <h1 data-testid="workspace-title" className="m-0 ml-2 min-w-0 truncate text-small font-normal text-text max-[639px]:order-last max-[639px]:basis-full max-[639px]:ml-0 max-[639px]:whitespace-normal max-[639px]:line-clamp-2 max-[639px]:[overflow-wrap:anywhere]" title={a.filename}>
          {a.filename}<span className="text-text-muted"> · Evidence viewer</span>
        </h1>
        <div role="toolbar" aria-label="View" data-testid="viewer-toolbar" className="ml-auto flex items-center gap-1 shrink-0">
          {/* below 640 px the bar keeps the zoom steps; fit, actual pixels, compare and the loupe move into the "…" menu (L-7) */}
          {!compact && <button type="button" className={tool()} onClick={fit} title="Fit (0)">Fit</button>}
          {!compact && <button type="button" className={tool()} onClick={oneToOne} title="Actual pixels (1)">1:1</button>}
          <button type="button" className={tool()} onClick={() => zoomBy(-1)} aria-label="Zoom out" title="Zoom out (−)"><Minus size={14} strokeWidth={1.5} aria-hidden /></button>
          <span data-testid="zoom" className="w-14 text-center font-mono text-mono text-text num">{zoomText}</span>
          <button type="button" className={tool()} onClick={() => zoomBy(1)} aria-label="Zoom in" title="Zoom in (+)"><Plus size={14} strokeWidth={1.5} aria-hidden /></button>
          {!compact && (
            <>
              <span aria-hidden className="mx-1 h-5 w-px bg-border" />
              <Menu ariaLabel="Compare mode" items={COMPARE} onSelect={(m) => update({ compare: m })} triggerClassName="h-8 px-2"
                label={<span data-testid="compare-mode">Compare: {COMPARE.find((c) => c.id === state.compare)?.label}</span>} />
              {state.compare === 'blink' && <button type="button" className={tool(blinkLens)} aria-pressed={blinkLens} onClick={() => setBlinkLens((x) => !x)} title="Show or hide the lenses (B)">Blink</button>}
              <button type="button" className={tool(loupe)} aria-pressed={loupe} onClick={toggleLoupe} title="Loupe (L; [ and ] change the magnification)">
                <Search size={14} strokeWidth={1.5} aria-hidden /><span className="max-[1199px]:sr-only">Loupe</span>
              </button>
              {!readOnly && (
                <button type="button" className={tool(annotating)} aria-pressed={annotating} disabled={locked} onClick={() => { setAnnotating((x) => !x); setDraft(null) }}
                  title={locked ? 'Revision locked' : 'Draw a box and add a note (Esc to stop)'}>
                  <PenLine size={14} strokeWidth={1.5} aria-hidden /><span className="max-[1199px]:sr-only">Annotate</span>
                </button>
              )}
            </>
          )}
          <Menu align="right" ariaLabel="More viewer actions" trigger={<MoreHorizontal size={16} strokeWidth={1.5} aria-hidden />} triggerClassName="h-8 w-8 justify-center"
            items={[
              ...(compact ? [
                { id: 'fit', label: 'Fit to the screen' }, { id: 'actual', label: 'Actual pixels (1:1)' },
                ...COMPARE.map((c) => ({ id: `compare:${c.id}`, label: `Compare: ${c.label}${c.id === state.compare ? ' (on)' : ''}` })),
                ...(state.compare === 'blink' ? [{ id: 'blink', label: blinkLens ? 'Hide the lenses' : 'Show the lenses' }] : []),
                { id: 'loupe', label: loupe ? 'Hide the loupe' : 'Show the loupe' },
              ] : []),
              { id: 'copy', label: 'Copy link to this view' },
              { id: 'coordinates', label: 'Add region by coordinates', disabled: locked || readOnly || !source },
            ]}
            onSelect={(k) => {
              if (k === 'copy') navigator.clipboard?.writeText(window.location.href).catch(() => undefined)
              if (k === 'coordinates') setDialog({ box: null })
              if (k === 'fit') fit()
              if (k === 'actual') oneToOne()
              if (k === 'blink') setBlinkLens((x) => !x)
              if (k === 'loupe') toggleLoupe()
              if (k.startsWith('compare:')) update({ compare: k.slice(8) as typeof state.compare })
            }} />
        </div>
      </header>
      <MediaAccessNotice ready={tokenReady} failed={!!tok.error} pending={tok.isFetching} onRetry={() => void tok.refetch()} className="px-4" />

      <div className="flex-1 min-h-0 flex flex-col min-[960px]:grid min-[960px]:grid-cols-[112px_minmax(0,1fr)_320px] min-[960px]:grid-rows-[minmax(0,1fr)]">
        <LensRail cat={cat} activeMap={activeMap} active={active} errors={heat.errors} onToggle={toggle} />

        <div ref={setStageEl} role="region" tabIndex={0} data-testid="viewer-stage" aria-label="Evidence stage" aria-describedby="viewer-stage-description"
          data-view={view ? `${view.scale} ${view.tx} ${view.ty}` : ''} data-source={source ? `${source.width} ${source.height}` : ''}
          data-first-paint-ms={firstPaint ?? undefined} data-toggle-ms={toggleMs ?? undefined} data-active-map={activeMap?.id ?? ''}
          data-base={base?.kind ?? ''} data-compare={state.compare}
          className="viewer-stage stage relative flex-1 min-h-[320px] min-[960px]:min-h-0 bg-stage text-stage-text overflow-hidden outline-none">
          <p id="viewer-stage-description" className="sr-only">{source ? describeRegions(activeMap?.title ?? null, visibleItems, source) : ''}</p>
          {stageMessage && <p role={baseError ? 'alert' : undefined} className="absolute left-6 top-6 m-0 text-small text-stage-muted">{stageMessage}</p>}
          {source && panes.map((i) => (
            <div key={i} data-pane={i} data-testid={i === lensPane ? 'lens-pane' : 'source-pane'}
              className={cx('absolute top-0 bottom-0 overflow-hidden touch-none', annotating ? 'cursor-crosshair' : 'cursor-grab active:cursor-grabbing')}
              style={{ left: i === 0 ? 0 : pane.width + 1, width: pane.width }}
              onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp} onPointerLeave={() => probe(null)}>
              <ViewportCanvas testId={i === lensPane ? 'viewer-canvas' : 'viewer-source-canvas'} scene={scene} view={view} viewport={pane}
                lens={state.compare === 'blink' ? blinkLens : state.compare === 'side' ? i === 1 : true}
                lensFromX={state.compare === 'swipe' ? (pane.width * state.swipe) / 100 : null}
                onDrawn={i === lensPane ? onDrawn : undefined} />
              {view && i === lensPane && <RegionsOverlay view={view} viewport={pane} items={visibleItems} notes={visibleNotes} selected={state.region} draft={draft} />}
              {state.compare === 'side' && <span className="absolute left-3 top-3 px-1.5 rounded-xs bg-stage-material font-mono text-micro text-stage-text">{i === 0 ? 'Source' : 'Lens'}</span>}
              {i === lensPane && state.compare === 'swipe' && <CompareSlider value={state.swipe} onChange={(v) => update({ swipe: v })} />}
              {i === lensPane && loupe && view && loupeAt && <Loupe scene={scene} view={view} at={loupeAt} factor={loupeFactor} lens={state.compare !== 'blink' || blinkLens} />}
            </div>
          ))}
          {source && view && (
            <p aria-hidden className="absolute left-3 bottom-2 m-0 px-1 py-0.5 rounded-xs bg-stage-2 font-mono text-micro text-stage-muted num pointer-events-none max-[959px]:hidden">
              {source.width} × {source.height} · zoom {zoomText}{readout ? ` · x ${readout.x} y ${readout.y}` : ''} · keys {LENS_KEYS} · L loupe · 0 fit · 1 1:1
            </p>
          )}
        </div>

        <Inspector cat={cat} source={source} activeMap={activeMap} mapLens={mapLens}
          opacity={layers.map?.opacity ?? DEFAULT_OPACITY}
          onOpacity={(v) => layers.map && update({ layers: { ...layers, map: { ...layers.map, opacity: v } } })}
          thr={state.thr} onThreshold={(v) => update({ thr: snapThreshold(v) })}
          mapOnly={mapOnly} onMapOnly={setMapOnly} dim={dim} onDim={setDim}
          onSelectMap={(mapId) => layers.map && setLayers({ ...layers, map: { ...layers.map, id: mapId } })}
          readout={readout} items={items} notes={notes} selected={state.region} onSelect={(label) => select(label)}
          onAddNote={() => { setAnnotating(true); stageEl?.focus({ preventScroll: true }) }} onAddByCoordinates={() => setDialog({ box: null })}
          locked={locked} readOnly={readOnly} />
      </div>

      {source && (
        <AnnotationDialog open={dialog !== null} box={dialog?.box ?? null} source={source} pending={create.isPending}
          error={create.error ? create.error.message : null} onCancel={closeDialog} onSave={(d) => create.mutate(d)} />
      )}
    </div>
  )
}
