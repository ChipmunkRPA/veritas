import { Fragment, useCallback, useEffect, useMemo, useState } from 'react'
import { ChevronRight } from 'lucide-react'
import type { Signal } from '../../lib/api'
import { fmtWeight, prob } from '../../lib/format'
import { cx } from '../../lib/cx'
import { Button, Eyebrow, Menu, ConfidenceMeter } from '../../components/ui'
import { Plate, captionFor, titleFor, useLightbox, type PlateItem } from './Plates'
import { useDemoAction, type StepAction } from '../../demo/types'

type Sort = 'weight' | 'category'

const DIR: Record<string, { word: string; arrow: string | null; arrowCls: string; wordCls: string }> = {
  supports_ai: { word: 'AI', arrow: '→', arrowCls: 'text-verdict-5', wordCls: 'text-text' },
  supports_authentic: { word: 'authentic', arrow: '→', arrowCls: 'text-verdict-1', wordCls: 'text-text' },
  neutral: { word: 'neutral', arrow: null, arrowCls: '', wordCls: 'text-text-muted' },
  inconclusive: { word: 'n/a', arrow: null, arrowCls: '', wordCls: 'text-text-muted' },
}
const strengthLevel = (s: string): 0 | 1 | 2 | 3 => (/strong/i.test(s) ? 3 : /moder/i.test(s) ? 2 : /weak/i.test(s) ? 1 : 0)
const score = (s: Signal) => (s.ai_probability == null ? -1 : Math.abs(s.ai_probability - 0.5) * s.weight)
const COLS = 8

/**
 * Evidence ledger (§4.5). Column plan at 1440: # 28 · SIGNAL (rest, ≥300) · OBSERVED 340 · DIRECTION 112 · STRENGTH 56 ·
 * WEIGHT 56 · P(AI) 64 · ⌄ 28. Fixed table layout so OBSERVED and SIGNAL truncate to one line; the full text lives in
 * `title` and in the expanded row together with REFERENCE and the strength word.
 */
export default function EvidenceTable({ signals, artifactUrl, figIndex = {}, mediaReady = true }: { mediaReady?: boolean; signals: Signal[]; artifactUrl: (n: string) => string; figIndex?: Record<string, number> }) {
  const [open, setOpen] = useState<Record<string, boolean>>({})
  const [sort, setSort] = useState<Sort>('weight')
  const byWeight = useMemo(() => [...signals].sort((a, b) => score(b) - score(a)), [signals])
  const groups = useMemo(() => {
    if (sort === 'weight') return [{ category: null as string | null, rows: byWeight }]
    const m = new Map<string, Signal[]>()
    byWeight.forEach((s) => { const k = s.category || 'Other'; if (!m.has(k)) m.set(k, []); m.get(k)!.push(s) })
    return Array.from(m.entries()).map(([category, rows]) => ({ category, rows }))
  }, [byWeight, sort])
  const allOpen = signals.length > 0 && signals.every((s) => open[s.id])
  const toggleAll = () => setOpen(allOpen ? {} : Object.fromEntries(signals.map((s) => [s.id, true])))

  // `e` toggles expand all when focus is not in a form field.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (e.key === 'e' && !e.metaKey && !e.ctrlKey && !e.altKey && !['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName) && !t.isContentEditable) toggleAll()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  })

  // Evidence ids in the key findings link here (`#signal-<id>`): the linked row opens.
  useEffect(() => {
    const openFromHash = () => {
      const m = /^#signal-(.+)$/.exec(window.location.hash)
      const key = m ? decodeURIComponent(m[1]) : null
      if (key && signals.some((s) => s.id === key)) setOpen((o) => ({ ...o, [key]: true }))
    }
    openFromHash()
    window.addEventListener('hashchange', openFromHash)
    return () => window.removeEventListener('hashchange', openFromHash)
  }, [signals])

  // Guided demo: open the first (heaviest) row, or the first row whose id/category matches.
  useDemoAction(useCallback((a: StepAction) => {
    if (a.type === 'expand-first-row' && byWeight[0]) setOpen((o) => ({ ...o, [byWeight[0].id]: true }))
    if (a.type === 'expand-row') {
      const m = a.match.toLowerCase()
      const hit = byWeight.find((s) => s.id.toLowerCase().includes(m) || s.category.toLowerCase().includes(m))
      if (hit) setOpen((o) => ({ ...o, [hit.id]: true }))
    }
  }, [byWeight]))

  const allArtifacts: PlateItem[] = useMemo(() => {
    if (!mediaReady) return []
    const names = Array.from(new Set(signals.flatMap((s) => s.artifacts)))
    return names.map((n) => ({ name: n, src: artifactUrl(n), title: titleFor(n), caption: captionFor(n), n: figIndex[n] }))
  }, [signals, artifactUrl, figIndex, mediaReady])
  const lb = useLightbox(allArtifacts)
  const openArtifact = (name: string) => { const i = allArtifacts.findIndex((a) => a.name === name); if (i >= 0) lb.open(i) }

  let line = 0
  return (
    <section aria-label="Evidence" data-tour="evidence">
      <div className="flex items-end justify-between gap-6">
        <Eyebrow as="h2">Evidence</Eyebrow>
        <div className="flex items-center gap-5">
          <Menu label={sort === 'weight' ? 'Sort: by weight' : 'Sort: by category'} onSelect={setSort} align="right"
            items={[{ id: 'weight', label: 'By weight' }, { id: 'category', label: 'By category' }]} />
          <Button variant="text" onClick={toggleAll} aria-keyshortcuts="e">{allOpen ? 'Collapse all' : 'Expand all'}</Button>
        </div>
      </div>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full border-collapse table-fixed">
          <colgroup>
            <col className="w-7" />
            <col />
            <col className="w-[340px] max-[959px]:w-[220px]" />
            <col className="w-28" />
            <col className="w-[76px] max-[959px]:w-0" />
            <col className="w-[64px]" />
            <col className="w-16" />
            <col className="w-7" />
          </colgroup>
          <thead>
            <tr className="border-b border-border-strong">
              <th scope="col" className="eyebrow text-right font-medium py-2 pr-3"><span className="sr-only">Line</span>#</th>
              <th scope="col" className="eyebrow text-left font-medium py-2 pr-4">Signal</th>
              <th scope="col" className="eyebrow text-left font-medium py-2 pr-4">Observed</th>
              <th scope="col" className="eyebrow text-left font-medium py-2 pr-4">Direction</th>
              <th scope="col" className="eyebrow text-left font-medium py-2 pr-4 max-[959px]:invisible max-[959px]:p-0 max-[959px]:overflow-hidden">Strength</th>
              <th scope="col" className="eyebrow text-right font-medium py-2 pr-4">Weight</th>
              <th scope="col" className="eyebrow text-right font-medium py-2 pr-2">P(AI)</th>
              <th scope="col"><span className="sr-only">Expand</span></th>
            </tr>
          </thead>
          <tbody>
            {groups.map((g) => (
              <Fragment key={g.category ?? '_all'}>
                {g.category && (
                  <tr className="border-t border-border"><td colSpan={COLS} className="pt-5 pb-1"><Eyebrow as="span">{g.category}</Eyebrow></td></tr>
                )}
                {g.rows.map((s) => {
                  line += 1
                  const d = DIR[s.direction] ?? DIR.neutral
                  const o = !!open[s.id]
                  const p = s.ai_probability
                  const edge = p == null ? null : p >= 0.95 ? 'bg-verdict-5' : p <= 0.05 ? 'bg-verdict-1' : null
                  const decUnderline = s.direction === 'supports_authentic' ? 'decoration-verdict-1' : 'decoration-verdict-5'
                  const figs = s.artifacts.map((a) => figIndex[a]).filter((n): n is number => !!n)
                  const detailId = `ev-${s.id}`
                  const level = strengthLevel(s.strength)
                  return (
                    <Fragment key={s.id}>
                      <tr id={`signal-${s.id}`} data-tour-signal={s.id} data-tour-category={s.category} className={cx('scroll-mt-14 border-b border-border cursor-pointer transition-colors duration-fast hover:bg-surface-3', o && 'bg-surface-2 border-b-0')} onClick={() => setOpen({ ...open, [s.id]: !o })}>
                        <td className={cx('py-3 pr-3 text-right font-mono text-mono num align-top', s.decisive ? 'text-text font-semibold' : 'text-text-muted')}>{line}</td>
                        <td className="py-3 pr-4 align-top min-w-0">
                          <button type="button" aria-expanded={o} aria-controls={detailId} onClick={(e) => { e.stopPropagation(); setOpen({ ...open, [s.id]: !o }) }} className="block w-full min-w-0 text-left rounded-sm">
                            <span className="flex items-baseline gap-2 min-w-0">
                              <span className="text-body font-medium text-text truncate" title={s.name}>{s.name}</span>
                              {s.decisive && <span className={cx('eyebrow shrink-0 underline underline-offset-[3px] decoration-1', decUnderline)}>decisive</span>}
                            </span>
                            <span className="block eyebrow text-micro mt-0.5 truncate">{s.category}{figs.length ? <span className="normal-case tracking-normal text-text-muted"> · see {figs.map((n) => `Fig. ${n}`).join(', ')}</span> : null}</span>
                          </button>
                        </td>
                        <td className="py-3 pr-4 align-top font-mono text-mono text-text whitespace-nowrap overflow-hidden text-ellipsis" title={s.observed}>{s.observed}</td>
                        <td className="py-3 pr-4 align-top text-small whitespace-nowrap">
                          {d.arrow && <span aria-hidden className={cx('mr-1', d.arrowCls)}>{d.arrow}</span>}<span className={d.wordCls}>{d.word}</span>
                        </td>
                        <td className="py-3 pr-4 align-top max-[959px]:invisible max-[959px]:p-0 max-[959px]:overflow-hidden" title={s.strength}>
                          <ConfidenceMeter level={level} className="align-middle" /><span className="sr-only">{s.strength}</span>
                        </td>
                        <td className="py-3 pr-4 align-top text-right font-mono text-mono num text-text">{fmtWeight(s.weight)}</td>
                        <td className="py-3 pr-2 align-top text-right font-mono text-mono num text-text whitespace-nowrap">
                          {edge && <span aria-hidden className={cx('inline-block w-[3px] h-5 mr-2 align-middle', edge)} style={{ borderRadius: 1 }} />}{prob(p)}
                        </td>
                        <td className="py-3 align-top text-text-muted">
                          <ChevronRight size={14} strokeWidth={1.5} aria-hidden className={cx('transition-transform duration-base ease-standard', o && 'rotate-90')} />
                        </td>
                      </tr>
                      <tr className={cx('border-b border-border', !o && 'border-b-0')}>
                        <td colSpan={COLS} className="p-0">
                          <div id={detailId} className="disclosure bg-surface-2" data-open={o}>
                            <div>
                              <div className="px-6 pt-4 pb-5 pl-14">
                                <p className="text-body font-medium text-text">{s.name}</p>
                                <Eyebrow as="h4" className="mt-3">Why this matters</Eyebrow>
                                <p className="mt-1 display text-reader text-text max-w-(--container-reader)">{s.reasoning}</p>
                                <dl className="grid grid-cols-[100px_1fr] gap-x-4 gap-y-1 mt-4 text-small max-w-(--container-reader)">
                                  <dt className="text-text-muted">Observed</dt><dd className="m-0 font-mono text-mono text-text break-words">{s.observed}</dd>
                                  <dt className="text-text-muted">Reference</dt><dd className="m-0 text-text">{s.reference || '—'}</dd>
                                  <dt className="text-text-muted">Strength</dt><dd className="m-0 text-text"><ConfidenceMeter level={level} className="mr-2 align-middle" />{s.strength}</dd>
                                </dl>
                                {mediaReady && s.artifacts.length > 0 && (
                                  <div className="mt-5 flex flex-wrap gap-x-8 gap-y-6">
                                    {s.artifacts.map((a) => (
                                      <Plate key={a} height={160} item={{ name: a, src: artifactUrl(a), title: titleFor(a), caption: captionFor(a), n: figIndex[a] }} onOpen={() => openArtifact(a)} className="max-w-[360px]" />
                                    ))}
                                  </div>
                                )}
                                {Object.keys(s.details ?? {}).length > 0 && <RawDetails details={s.details} />}
                              </div>
                            </div>
                          </div>
                        </td>
                      </tr>
                    </Fragment>
                  )
                })}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
      {lb.node}
    </section>
  )
}

function RawDetails({ details }: { details: Record<string, unknown> }) {
  const [o, setO] = useState(false)
  return (
    <div className="mt-4">
      <button type="button" aria-expanded={o} onClick={() => setO(!o)} className="inline-flex items-center gap-1 text-small text-text-muted hover:text-text rounded-sm min-h-8">
        <ChevronRight size={14} strokeWidth={1.5} aria-hidden className={cx('transition-transform duration-base', o && 'rotate-90')} /> Raw details
      </button>
      <div className="disclosure" data-open={o}>
        <div>
          <pre className="mt-2 max-h-60 overflow-auto bg-surface border border-border rounded-sm p-3 font-mono text-caption leading-[18px] text-text">{JSON.stringify(details, null, 1)}</pre>
        </div>
      </div>
    </div>
  )
}
