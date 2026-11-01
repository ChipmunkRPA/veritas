import type { ReactNode } from 'react'
import { SHOW_THRESHOLD_MAX } from '../../lib/colormap'
import type { Size } from '../../lib/coords'
import { cx } from '../../lib/cx'
import { Button, HeatLegend, Select } from '../../components/ui'
import { anonymizeModelNames } from '../analysis/checkNames'
import type { ProbeValue } from './heatmap'
import { available, FAMILIES, LENSES, lensDef, shareText, type AnnotationItem, type Catalogue, type HeatmapPayload, type LensKey, type MapLensKey, type RegionItem } from './lenses'

export interface Readout { x: number; y: number; map: ProbeValue | null; region: string | null }

function unavailable(key: LensKey): string {
  if (key === 'reliability') return 'Reliability: the active map has no reliability mask'
  return `${lensDef(key).name}: nothing of this kind in this analysis`
}

/** Lens rail (UX §5.9): families with the letter of each lens as its visible label and shortcut; `●` marks active lenses and
 *  unavailable lenses are disabled with the reason. A 112 px column from 960 px, a horizontal toolbar below (UX §8.8). */
export function LensRail({ cat, activeMap, active, errors, onToggle }: {
  cat: Catalogue
  activeMap: HeatmapPayload | null
  active: ReadonlySet<LensKey>
  errors: Readonly<Record<string, string>>
  onToggle: (key: LensKey) => void
}) {
  return (
    <nav aria-label="Lenses" data-testid="lens-rail"
      className="shrink-0 flex items-center gap-1 h-12 px-2 overflow-x-auto border-b border-border bg-surface-2 min-[960px]:h-auto min-[960px]:flex-col min-[960px]:gap-0 min-[960px]:px-0 min-[960px]:py-3 min-[960px]:overflow-x-visible min-[960px]:overflow-y-auto min-[960px]:border-b-0 min-[960px]:border-r">
      {FAMILIES.map((family, i) => (
        <div key={family} role="group" aria-label={family}
          className={cx('flex items-center gap-1 shrink-0 min-[960px]:flex-col', i > 0 && 'pl-2 border-l border-border min-[960px]:pl-0 min-[960px]:pt-2.5 min-[960px]:mt-2.5 min-[960px]:border-l-0 min-[960px]:border-t min-[960px]:w-full')}>
          <span aria-hidden className="eyebrow text-[10px] leading-4 tracking-[0.06em] max-[959px]:hidden">{family}</span>
          {LENSES.filter((l) => l.family === family).map((l) => {
            const failure = l.kind === 'map' ? cat.maps[l.key as MapLensKey].map((m) => errors[m.id]).find(Boolean) : undefined
            const ok = available(cat, l.key, activeMap) && !failure
            const on = ok && active.has(l.key)
            const title = failure ? `${l.name}: the map could not be loaded (${failure})` : ok ? `${l.name}${l.letter ? ` (${l.letter})` : ''}: ${l.meaning}` : unavailable(l.key)
            return (
              <button key={l.key} type="button" data-lens={l.key} aria-pressed={on} disabled={!ok} title={title} onClick={() => onToggle(l.key)}
                aria-label={`${l.name}${l.letter ? ` (${l.letter})` : ''}`}
                className={cx('relative h-8 min-w-10 px-2 shrink-0 inline-flex items-center justify-center rounded-sm font-mono text-mono transition-colors duration-fast disabled:cursor-not-allowed',
                  // N-19d (design review P2): a lens that is not available was drawn at 40 % opacity, which could not be read; it keeps
                  // the muted text colour and is struck through
                  'disabled:line-through disabled:decoration-border-strong',
                  on ? 'bg-surface-3 text-text' : 'text-text-muted enabled:hover:text-text enabled:hover:bg-surface-3')}>
                {l.letter ?? l.name}
                {on && <span aria-hidden className="absolute right-0.5 top-0.5 w-1.5 h-1.5 rounded-full bg-text" />}
              </button>
            )
          })}
        </div>
      ))}
    </nav>
  )
}

function Section({ title, action, children, testId }: { title: string; action?: ReactNode; children: ReactNode; testId?: string }) {
  return (
    <section data-testid={testId} className="px-4 py-4 border-t border-border first:border-t-0">
      <div className="flex items-center justify-between gap-3 min-h-6">
        <h2 className="m-0 eyebrow">{title}</h2>
        {action}
      </div>
      <div className="mt-2">{children}</div>
    </section>
  )
}

function Slider({ label, value, min, max, step, display, onChange, testId }: {
  label: string; value: number; min: number; max: number; step: number; display: string; onChange: (v: number) => void; testId: string
}) {
  return (
    <label className="mt-3 grid grid-cols-[72px_1fr_44px] items-center gap-3 text-small text-text-muted">
      {label}
      <input data-testid={testId} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-text" />
      <span className="font-mono text-mono text-text text-right num">{display}</span>
    </label>
  )
}

function Row({ label, detail, on, onClick, testId }: { label: string; detail: ReactNode; on: boolean; onClick: () => void; testId: string }) {
  return (
    <li>
      <button type="button" data-testid={testId} aria-current={on || undefined} onClick={onClick}
        className={cx('w-full -mx-2 px-2 min-h-8 grid grid-cols-[48px_1fr_auto] items-center gap-2 rounded-sm text-left text-small transition-colors duration-fast hover:bg-surface-3', on && 'bg-surface-3')}>
        <span className="font-mono text-mono text-text">{label}</span>
        <span className="text-text-muted truncate num">{detail}</span>
        <span aria-hidden className="text-text-muted">▸</span>
      </button>
    </li>
  )
}

/** Inspector (UX §5.9): the active lens with opacity, `Show ≥`, legend and meaning; the pixel readout; regions, clone pairs and
 *  faces; and the analyst's notes. Chrome follows the page theme. */
export function Inspector({ cat, source, activeMap, mapLens, opacity, onOpacity, thr, onThreshold, mapOnly, onMapOnly, dim, onDim, onSelectMap, readout, items, notes, selected, onSelect, onAddNote, onAddByCoordinates, locked, readOnly }: {
  cat: Catalogue
  source: Size | null
  activeMap: HeatmapPayload | null
  mapLens: MapLensKey | null
  opacity: number
  onOpacity: (v: number) => void
  thr: number
  onThreshold: (v: number) => void
  mapOnly: boolean
  onMapOnly: (v: boolean) => void
  dim: boolean
  onDim: (v: boolean) => void
  onSelectMap: (id: string) => void
  readout: Readout | null
  items: readonly RegionItem[]
  notes: readonly AnnotationItem[]
  selected: string | null
  onSelect: (label: string) => void
  onAddNote: () => void
  onAddByCoordinates: () => void
  locked: boolean
  readOnly: boolean
}) {
  const regions = items.filter((r) => r.lens === 'regions')
  const clones = items.filter((r) => r.lens === 'clones' && !r.label.endsWith('′'))
  const faces = items.filter((r) => r.lens === 'faces')
  const hasMask = activeMap?.reliability_mask_artefact_id != null
  const noteAction = locked ? <span className="text-caption text-text-muted" title="Revision locked">Revision locked</span>
    : readOnly ? null : <Button variant="text" onClick={onAddNote}>+ Add</Button>
  return (
    <aside aria-label="Inspector" data-testid="inspector"
      className="min-h-0 overflow-y-auto bg-surface border-t border-border max-[959px]:max-h-[40vh] max-[959px]:shrink-0 min-[960px]:border-t-0 min-[960px]:border-l">
      <Section title="Lens" testId="lens-section">
        {activeMap && mapLens ? (
          <>
            <p className="m-0 text-small text-text">{lensDef(mapLens).name} <span className="text-text-muted">· {anonymizeModelNames(activeMap.title)}</span></p>
            {cat.maps[mapLens].length > 1 && (
              <Select compact aria-label="Map" value={activeMap.id} onChange={(e) => onSelectMap(e.target.value)} className="mt-2 w-full">
                {cat.maps[mapLens].map((m) => <option key={m.id} value={m.id}>{anonymizeModelNames(m.title)}</option>)}
              </Select>
            )}
            <p className="mt-1 mb-0 text-caption text-text-muted">{lensDef(mapLens).meaning}</p>
            <Slider label="Opacity" testId="opacity" value={opacity} min={0} max={100} step={5} display={`${opacity} %`} onChange={onOpacity} />
            <Slider label="Show ≥" testId="threshold" value={thr} min={0} max={SHOW_THRESHOLD_MAX} step={0.05} display={thr.toFixed(2)} onChange={onThreshold} />
            <HeatLegend className="mt-4" title={`${anonymizeModelNames(activeMap.title)}${activeMap.calibrated ? '' : ' · uncalibrated'}`} attention={mapLens === 'attention'} showUnreliable={hasMask} />
            <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-small text-text">
              <label className="inline-flex items-center gap-2 min-h-8"><input type="checkbox" checked={mapOnly} onChange={(e) => onMapOnly(e.target.checked)} className="accent-text" /> Map only</label>
              {hasMask && <label className="inline-flex items-center gap-2 min-h-8"><input type="checkbox" checked={dim} onChange={(e) => onDim(e.target.checked)} className="accent-text" /> Dim unreliable</label>}
            </div>
          </>
        ) : (
          <p className="m-0 text-small text-text-muted">No map lens is on. Choose a letter in the rail; marks stack over the source.</p>
        )}
      </Section>

      <Section title="Readout" testId="readout">
        {readout ? (
          <dl className="m-0 grid grid-cols-[72px_1fr] gap-x-3 gap-y-0.5 text-small">
            <dt className="text-text-muted">pixel</dt>
            <dd data-testid="readout-xy" className="m-0 font-mono text-mono text-text num">x {readout.x} y {readout.y}</dd>
            <dt className="text-text-muted">value</dt>
            <dd data-testid="readout-value" className="m-0 font-mono text-mono text-text num">
              {readout.map ? readout.map.value.toFixed(2) : '—'}
              {readout.map?.reliable != null && <span data-testid="readout-reliability" className="text-text-muted"> · {readout.map.reliable ? 'reliable' : 'unreliable'}</span>}
            </dd>
            <dt className="text-text-muted">region</dt>
            <dd data-testid="readout-region" className="m-0 font-mono text-mono text-text">{readout.region ?? '—'}</dd>
          </dl>
        ) : (
          <p className="m-0 text-small text-text-muted">Point at the image to read the pixel, the map value and its reliability.</p>
        )}
      </Section>

      <Section title="Regions" testId="regions-section">
        {regions.length ? (
          <ul className="m-0 p-0 list-none">
            {regions.map((r) => (
              <Row key={r.key} testId="region-row" label={r.label} on={selected === r.label} onClick={() => onSelect(r.label)}
                detail={<>{source ? shareText(r, source) : ''}{(r.probability ?? r.score) != null ? ` · ${(r.probability ?? r.score)?.toFixed(2)}` : ''}</>} />
            ))}
          </ul>
        ) : <p className="m-0 text-small text-text-muted">No regions marked.</p>}
      </Section>

      {clones.length > 0 && (
        <Section title="Clone pairs">
          <ul className="m-0 p-0 list-none">
            {clones.map((r) => <Row key={r.key} testId="clone-row" label={r.label} on={selected === r.label} onClick={() => onSelect(r.label)} detail={`pair ${r.label} / ${r.pair}`} />)}
          </ul>
        </Section>
      )}

      {faces.length > 0 && (
        <Section title="Faces">
          <ul className="m-0 p-0 list-none">
            {faces.map((r) => (
              <Row key={r.key} testId="face-row" label={r.label} on={selected === r.label} onClick={() => onSelect(r.label)}
                detail={`${Math.round(r.box.w)}×${Math.round(r.box.h)} px${r.score != null ? ` · detection ${r.score.toFixed(2)}` : ''}`} />
            ))}
          </ul>
        </Section>
      )}

      <Section title={`Notes (${notes.length})`} action={noteAction} testId="notes-section">
        {notes.length ? (
          <ul className="m-0 p-0 list-none space-y-2">
            {notes.map((n) => (
              <li key={n.annotation.id}>
                <button type="button" data-testid="note-row" aria-current={selected === n.label || undefined} onClick={() => onSelect(n.label)}
                  className={cx('w-full -mx-2 px-2 py-1 rounded-sm text-left transition-colors duration-fast hover:bg-surface-3', selected === n.label && 'bg-surface-3')}>
                  <span className="flex items-baseline gap-2 text-small">
                    <span className="font-mono text-mono text-text">{n.label}</span>
                    <span className="text-text-muted">{n.annotation.visibility === 'in_report' ? 'in report' : 'internal'}</span>
                  </span>
                  <span className="block text-small text-text line-clamp-3 whitespace-pre-line">{n.annotation.body}</span>
                </button>
              </li>
            ))}
          </ul>
        ) : <p className="m-0 text-small text-text-muted">No notes yet.</p>}
        {!locked && !readOnly && <p className="mt-3 mb-0"><Button variant="text" onClick={onAddByCoordinates}>Add region by coordinates</Button></p>}
      </Section>
    </aside>
  )
}
