import { useCallback, useEffect, useMemo, useReducer, useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { hasApiKey } from '../../lib/api'
import { useExhibitPayloads, type AnalysisDetail, type ExhibitMeta, type Finding } from '../../lib/api/queries'
import { SHOW_THRESHOLD_OVERLAY } from '../../lib/colormap'
import { fitView, type Size } from '../../lib/coords'
import { cx } from '../../lib/cx'
import { anonymizeModelNames } from '../analysis/checkNames'
import { activeLenses, allMaps, available, catalogue, heatmapLens, lensDef, mapById, presetLayers, regionItems, toggleLayers, type LensKey } from './lenses'
import RegionsOverlay from './RegionsOverlay'
import type { Scene } from './scene'
import { useBase } from './useBase'
import { useHeatmaps } from './useHeatmaps'
import ViewportCanvas from './ViewportCanvas'
import { DEFAULT_OPACITY, parseViewerState, serialiseViewerState, type Layers } from './viewerState'

/** UX §3 image slot: max-h 560 including the stage padding (24 + 40). */
const STAGE_MAX = 560
const PAD = 24
const TOGGLES: readonly LensKey[] = ['faces', 'heat', 'reliability', 'ela', 'noise']
const NAMES: Partial<Record<LensKey, string>> = { faces: 'Faces', heat: 'Heat', reliability: 'Reliability', ela: 'ELA', noise: 'Noise' }

const toggleCls = (on: boolean) => cx(
  'h-8 px-2.5 inline-flex items-center gap-1.5 text-small font-medium rounded-sm transition-colors duration-fast disabled:opacity-40 disabled:cursor-not-allowed',
  on ? 'bg-surface-3 text-text' : 'text-text-muted enabled:hover:text-text enabled:hover:bg-surface-3',
)

/** Exhibit preview on the result page (UX §5.5, §4.2 L3): the source with the lens preset the top finding needs, one control row
 *  (Faces · Heat · Reliability · ELA · Noise, opacity) and "Open evidence viewer ▸", which carries the chosen lenses into the
 *  viewer's deep link. Face and provenance panels live in the findings, ledger and provenance sections. */
export default function ExhibitPreview({ analysis, exhibits, findings, token, notice }: {
  analysis: AnalysisDetail
  exhibits: readonly ExhibitMeta[]
  findings: readonly Finding[]
  token: string
  notice?: ReactNode
}) {
  const ids = useMemo(() => exhibits.filter((x) => x.kind === 'heatmap' || x.kind === 'regions').map((x) => x.id), [exhibits])
  const payloads = useExhibitPayloads(analysis.id, ids)
  const payloadKey = payloads.data.map((p) => p.id).join(',')
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const cat = useMemo(() => catalogue(payloads.data), [payloadKey])
  const preset = useMemo(() => presetLayers(cat, findings), [cat, findings])
  const [chosen, setChosen] = useState<Layers | null>(null)
  const layers = chosen ?? preset
  const active = activeLenses(cat, layers)
  const tokenReady = !hasApiKey || token !== ''

  const [, bump] = useReducer((x: number) => x + 1, 0)
  const onTile = useCallback(() => bump(), [])
  const { base, source, error } = useBase({ enabled: tokenReady && !payloads.isPending, analysisId: analysis.id, tiles: cat.tiles, token, onUpdate: onTile })
  const maps = useMemo(() => allMaps(cat), [cat])
  const heat = useHeatmaps(maps, token, tokenReady, SHOW_THRESHOLD_OVERLAY)
  const activeMap = mapById(cat, layers.map?.id)
  const decoded = activeMap ? heat.decoded[activeMap.id] ?? null : null
  const items = useMemo(() => regionItems(cat, source), [cat, source])
  const visible = items.filter((r) => layers.marks.some((m) => m.kind === r.lens && m.id === r.exhibitId))

  const box = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const el = box.current
    if (!el) return
    const ro = new ResizeObserver((entries) => setWidth(Math.floor(entries[0].contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  const inner = Math.max(0, width - 2 * PAD)
  const viewport: Size | null = source && inner > 0
    ? { width: Math.min(inner, Math.round(((STAGE_MAX - 64) * source.width) / source.height)), height: Math.min(STAGE_MAX - 64, Math.round((inner * source.height) / source.width)) }
    : null
  const view = source && viewport ? fitView(source, viewport, 0) : null
  const scene: Scene = {
    source: source ?? { width: 1, height: 1 },
    base,
    layers: { heat: heat.layer(activeMap?.id), opacity: (layers.map?.opacity ?? DEFAULT_OPACITY) / 100, unreliable: decoded?.unreliable ?? null, hatch: active.has('reliability'), dim: false, mapOnly: false },
  }
  const shown = TOGGLES.filter((k) => (k === 'reliability' ? maps.some((m) => m.reliability_mask_artefact_id != null) : available(cat, k, activeMap)))
  const href = `/analyses/${analysis.id}/evidence?${serialiseViewerState({ ...parseViewerState(new URLSearchParams()), layers }).toString()}`
  const lensName = activeMap ? lensDef(heatmapLens(activeMap)).name.toLowerCase() : null
  const caption = `Exhibit — Source image${visible.some((r) => r.lens === 'faces') ? ' with detected faces' : ''}${activeMap ? `; ${lensName} lens (${anonymizeModelNames(activeMap.title)}) at ${layers.map?.opacity ?? DEFAULT_OPACITY} %` : ''}.`

  return (
    <section id="section-exhibit" data-testid="exhibit-preview" aria-labelledby="exhibit-heading" className="scroll-mt-14">
      <div className="flex items-end justify-between gap-6 flex-wrap">
        <h2 id="exhibit-heading" className="m-0 eyebrow">Exhibit</h2>
        {shown.length > 0 && (
          <div role="group" aria-label="Lenses" className="flex items-center gap-1 flex-wrap">
            {shown.map((k) => {
              const ok = available(cat, k, activeMap)
              const on = ok && active.has(k)
              return (
                <button key={k} type="button" data-lens={k} aria-pressed={on} disabled={!ok} className={toggleCls(on)}
                  onClick={() => setChosen(toggleLayers(cat, layers, k))} title={`${lensDef(k).name}: ${lensDef(k).meaning}`}><span aria-hidden className={cx('w-1.5 h-1.5 rounded-full bg-text', !on && 'invisible')} />{NAMES[k]}</button>
              )
            })}
            {layers.map && (
              <label className="flex items-center gap-2 pl-3 pr-1 text-micro font-mono text-text-muted">
                <span className="sr-only">Lens opacity</span>
                <input type="range" min={0} max={100} step={5} value={layers.map.opacity} aria-label="Lens opacity" className="w-24 accent-text"
                  onChange={(e) => layers.map && setChosen({ ...layers, map: { ...layers.map, opacity: Number(e.target.value) } })} />
                <span className="num w-8 text-right">{layers.map.opacity}%</span>
              </label>
            )}
          </div>
        )}
      </div>
      {notice}

      <div ref={box} className="stage mt-3 relative rounded-md bg-stage text-stage-text border border-stage-border flex justify-center" style={{ padding: `${PAD}px ${PAD}px ${PAD + 16}px` }}>
        {!tokenReady || (!source && !error) ? (
          <p className="m-0 h-[240px] flex items-center text-small text-stage-muted">{tokenReady ? 'Loading the image…' : 'Preparing secure media links…'}</p>
        ) : error ? (
          <p role="alert" className="m-0 h-[120px] flex items-center text-small text-stage-muted">The image could not be loaded: {error}</p>
        ) : view && viewport && (
          <div className="relative" style={{ width: viewport.width, height: viewport.height }}>
            <ViewportCanvas testId="preview-canvas" scene={scene} view={view} viewport={viewport} lens grid={false} />
            <RegionsOverlay view={view} viewport={viewport} items={visible} notes={[]} selected={null} draft={null} />
          </div>
        )}
        {source && <p className="absolute bottom-3 right-4 m-0 font-mono text-caption text-stage-muted num">{source.width} × {source.height}</p>}
      </div>
      <div className="mt-2 flex items-baseline justify-between gap-x-6 gap-y-1 flex-wrap">
        <p className="m-0 text-caption text-text-muted">{caption}</p>
        <Link to={href} data-testid="open-viewer" className="text-small font-medium text-accent underline underline-offset-[3px] decoration-1 hover:text-accent-hover hover:decoration-2">Open evidence viewer ▸</Link>
      </div>
    </section>
  )
}
