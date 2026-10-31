import { Fragment, useCallback, useEffect, useMemo, useState, type KeyboardEvent, type ReactNode } from 'react'
import { ChevronRight } from 'lucide-react'
import { hasApiKey } from '../../lib/api'
import { useModelCards, type AnalysisDetail, type DetectorRun, type ExhibitMeta, type Finding, type SignalV2 } from '../../lib/api/queries'
import { measuredRows } from '../models/catalogue'
import { prob, shortHash } from '../../lib/format'
import { cx } from '../../lib/cx'
import { Button, CodeChip, DirectionMark, Disclosure, EmptyState, Eyebrow, HatchSwatch, Menu, StrengthMeter } from '../../components/ui'
import { useDemoAction, type StepAction } from '../../demo/types'
import { ofTheServer, ON_THIS_SERVER, type Decision } from '../analysis/answer'
import { MENTION_DIRECTION, MENTION_WHY, NOT_COUNTED, isMention, mentionFields, mentionWords, modelUnused, rowNotCounted, signed } from '../analysis/contributions'
import { NOT_IN_ESTIMATE, entered, littleOf, measuredFromNeutral } from '../analysis/vocabulary'
import ModelCardSheet from './ModelCardSheet'
import {
  anchorOf, contributionBar, effect, lensFamily, notRunRows, signalDirection, signalStrength, standaloneP, statusWord, testingLines,
} from './ledger'

type Sort = 'effect' | 'id'
const COLS = 8
const HIDE_1279 = 'max-[1279px]:invisible max-[1279px]:p-0 max-[1279px]:overflow-hidden'
const HIDE_959 = 'max-[959px]:invisible max-[959px]:p-0 max-[959px]:overflow-hidden'
/** Below 640 px a ledger row is a block of its own (N-19b): the id, then two columns for the marks, then the chevron. */
const STACK_ROW = 'max-[639px]:grid max-[639px]:grid-cols-[3rem_auto_minmax(0,1fr)_1.25rem] max-[639px]:gap-x-3 max-[639px]:items-baseline'

/** Centred-zero contribution bar (±1.5 log-odds full scale, `›` on overflow) and the signed value. */
function ContributionCell({ logit, uncounted = false }: { logit: number | null | undefined; uncounted?: boolean }) {
  const b = contributionBar(logit)
  if (logit == null) return <span className="font-mono text-mono text-text-muted">—</span>
  // N-19p (D6-05, R10-05): a row that the decision does not count has no bar and no number (it read "−0.00")
  if (uncounted) return <span className="text-small text-text-muted whitespace-nowrap">{NOT_COUNTED}</span>
  return (
    <span className="inline-flex items-center gap-2 justify-end w-full">
      <span aria-hidden className="relative block w-24 h-3">
        <span className="absolute left-1/2 top-0 h-full w-px bg-graphic" />
        {b.side && <span className={cx('absolute top-0.5 h-2', b.side === 'right' ? 'left-1/2 bg-dir-ai' : 'right-1/2 bg-dir-authentic')} style={{ width: `${b.half * 50}%`, borderRadius: 1 }} />}
        {b.overflow && <span className={cx('absolute -top-1 font-mono text-micro text-text', b.side === 'right' ? '-right-2' : '-left-2')}>{b.side === 'right' ? '›' : '‹'}</span>}
      </span>
      <span className="font-mono text-mono num text-text w-12 text-right">{signed(logit)}</span>
    </span>
  )
}

function Figures({ s, exhibits, token }: { s: SignalV2; exhibits: readonly ExhibitMeta[]; token: string }) {
  const mine = exhibits.filter((x) => ((s.exhibit_ids ?? []).includes(x.id) || (x.source_signal_ids ?? []).includes(s.id))
    && ((x.binaries ?? []).includes('preview') || (x.kind === 'image' && (x.binaries ?? []).includes('payload'))))
  if (mine.length === 0) return null
  const ready = !hasApiKey || token !== ''
  return (
    <div className="flex flex-wrap gap-x-6 gap-y-4">
      {mine.map((x) => {
        const part = (x.binaries ?? []).includes('preview') ? 'preview' : 'payload'
        return (
          <figure key={x.id} className="m-0 max-w-[240px]">
            <span className="inline-block align-top h-[120px] max-w-full rounded-md border border-border bg-surface overflow-hidden">
              {ready && <img src={`/api/v2/exhibits/${x.id}/${part}${token ? `?token=${encodeURIComponent(token)}` : ''}`} alt={x.title} loading="lazy"
                className={cx('block h-full w-auto', x.kind === 'heatmap' && '[image-rendering:pixelated]')} />}
            </span>
            <figcaption className="mt-1 text-caption text-text-muted">{x.display_id ? `${x.display_id} — ` : ''}{x.title}</figcaption>
          </figure>
        )
      })}
    </div>
  )
}

function Expanded({ s, analysis, exhibits, token, totalWeight, largest, onCard }: {
  s: SignalV2; analysis: AnalysisDetail; exhibits: readonly ExhibitMeta[]; token: string; totalWeight: number; largest: number | null
  onCard: (id: string, trigger: HTMLElement) => void
}) {
  const { p, calibrated } = { p: standaloneP(s), calibrated: s.calibrated_probability != null }
  // the cards the workspace has loaded (no request of its own): the count of benchmark results for the "In testing" line
  const card = useModelCards(false).data?.find((c) => c.id === s.detector_id)
  const ref = (analysis.manifest?.detectors ?? []).find((x) => x.id === s.detector_id)
  const artefact = analysis.decision?.fusion_mode === 'artefact'
  const calibratedBy = artefact ? 'calibrated by the fusion model' : 'calibrated'
  const weight = s.effective_weight != null ? `weight ${s.effective_weight.toFixed(2)} of ${totalWeight.toFixed(2)}` : null
  // N-19d: the line of a result without a fusion model no longer ends in "(uncalibrated fallback)"
  const contribution = s.contribution_logit != null
    ? [`${signed(s.contribution_logit)} log-odds`, weight, largest != null && Math.abs(s.contribution_logit) === largest ? 'largest single contribution' : null].filter(Boolean).join(' · ')
    : artefact ? 'none recorded for this check' : [`none recorded: no fusion model ${modelUnused(analysis.condition_profile) ? 'was used for this analysis' : 'is loaded'}`, weight ? `hand-set ${weight}` : null].filter(Boolean).join(' · ')
  const reliability = s.reliability?.factor != null && s.reliability.factor < 1
    ? `${s.reliability.factor.toFixed(2)} · ${(s.reliability.reasons ?? []).join('; ') || 'reduced by the reliability rules'}`
    : 'full weight: no reliability rule reduced it'
  const raw = { metrics: s.metrics, details: s.details, reliability_inputs: s.reliability?.inputs, model_revisions: s.model_revisions, calibration_ref: s.calibration_ref, raw_score: s.raw_score, raw_kind: s.raw_kind }
  const rows: [string, ReactNode][] = [
    // N-19p (browser QA Q6-01): a mention in the finding's words and its field, as the row says (it read the check's label,
    // "IPTC DigitalSourceType = trainedAlgorithmicMedia")
    ['Observed', <span className="font-mono text-mono break-words">{isMention(s)
      ? `${mentionWords(s.observed)}${mentionFields(s.details) ? ` (in ${mentionFields(s.details)})` : ''}` : s.observed || '—'}</span>],
    ['Reference', s.reference || 'No reference distribution on record.'],
    // N-19d: "calibrated" stood beside a model card that says "Calibration: none": under a fusion model it is that model which
    // calibrates the score, not the check
    ['P(AI)', isMention(s) ? 'none: a mention sets no number in this result' : p == null ? 'not established' : calibrated ? `${prob(p)} · ${calibratedBy}${s.calibration_ref ? ` (${s.calibration_ref})` : ''}` : `${prob(p)} · raw detector score, not a calibrated probability`],
    ['Contribution', contribution],
    ['Reliability', reliability],
    ['In testing', <>{testingLines(s, analysis.calibration ?? [], analysis.kind, card ? measuredRows(card).length : 0).map((l) => <span key={l} className="block">{l}</span>)}</>],
    ['Known false alarms', (s.fp_modes ?? []).length > 0 ? (s.fp_modes ?? []).join(' · ') : 'None recorded on the signal; see the model card.'],
  ]
  if (s.provenance_override) rows.push(['Provenance override', `${s.provenance_override.replace(/_/g, ' ')} (provenance precedence, applied before fusion)`])
  return (
    <div className="px-6 pt-4 pb-5 pl-14 max-[639px]:pl-6">
      <p className="m-0 text-body font-medium text-text">{s.name} <span className="ml-2 eyebrow">{lensFamily(s, exhibits)}</span></p>
      {(isMention(s) || s.reasoning) && (<>
        <Eyebrow as="h4" className="mt-3">Why this matters</Eyebrow>
        <p className="mt-1 m-0 display text-reader text-text max-w-(--container-reader)">{isMention(s) ? MENTION_WHY : s.reasoning}</p>
      </>)}
      <dl className="mt-4 m-0 grid grid-cols-[136px_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-small max-w-[80ch] max-[639px]:grid-cols-1">
        {rows.map(([k, v]) => (
          <div key={k} className="contents"><dt className="eyebrow pt-[3px]">{k}</dt><dd className="m-0 text-text min-w-0">{v}</dd></div>
        ))}
        <div className="contents">
          <dt className="eyebrow pt-[3px]">Model</dt>
          <dd className="m-0 text-text min-w-0 flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="font-mono text-mono">{s.detector_id} {s.detector_version}</span>
            {ref?.weights_sha256 && <CodeChip copy={ref.weights_sha256} title="Copy weights sha256">{shortHash(ref.weights_sha256)}</CodeChip>}
            {!ref?.weights_sha256 && ref?.revision && <span className="font-mono text-mono text-text-muted">rev {shortHash(ref.revision)}</span>}
            {ref?.licence && <span className="text-text-muted">· {ref.licence}</span>}
            <Button variant="text" data-testid="model-card-open" onClick={(e) => onCard(s.detector_id, e.currentTarget)}>Model card ▸</Button>
          </dd>
        </div>
      </dl>
      <div className="mt-4"><Figures s={s} exhibits={exhibits} token={token} /></div>
      <Disclosure className="mt-3" label="Raw details">
        <pre className="m-0 max-h-60 overflow-auto bg-surface border border-border rounded-sm p-3 font-mono text-caption leading-[18px] text-text">{JSON.stringify(raw, null, 1)}</pre>
      </Disclosure>
    </div>
  )
}

/**
 * Evidence ledger v2 (UX §5.7; ARCHITECTURE §7.1 `signals/SignalsLedger`, evolving the v1 `EvidenceTable`). Column plan at 1440:
 * # 56 · SIGNAL rest · OBSERVED 220 · DIRECTION 104 · STRENGTH 64 · CONTRIBUTION 160 · P(AI) 56 · ⌄ 28. Observed collapses below
 * 1280 px and Strength below 960 px (collapsed cells, never display:none, under table-fixed). Checks that did not run are grouped
 * at the bottom with their reason.
 */
export default function SignalsLedger({ analysis, signals, runs, exhibits, findings = [], token }: {
  analysis: AnalysisDetail; signals: readonly SignalV2[]; runs: readonly DetectorRun[]; exhibits: readonly ExhibitMeta[]
  findings?: readonly Finding[]; token: string
}) {
  const [open, setOpen] = useState<Record<string, boolean>>({})
  const [sort, setSort] = useState<Sort>('effect')
  const [sheet, setSheet] = useState<{ id: string; trigger: HTMLElement } | null>(null)
  const active = useMemo(() => signals.filter((s) => s.status === 'ok'), [signals])
  // D-63 (`vocabulary.ts`): ln 2 for "little effect" and strength from the contribution in a result measured from neutral points
  const little = littleOf(analysis.decision)
  const measured = measuredFromNeutral(analysis.decision)
  const rows = useMemo(() => [...active].sort(sort === 'id'
    ? (a, b) => (a.display_id ?? a.id).localeCompare(b.display_id ?? b.id)
    : (a, b) => effect(b) - effect(a) || (a.display_id ?? a.id).localeCompare(b.display_id ?? b.id)), [active, sort])
  const notRun = useMemo(() => notRunRows(signals, runs), [signals, runs])
  const totalWeight = active.reduce((w, s) => w + (s.effective_weight ?? 0), 0)
  const contributions = active.map((s) => s.contribution_logit).filter((c): c is number => c != null)
  const largest = contributions.length ? Math.max(...contributions.map(Math.abs)) : null
  const anyRaw = active.some((s) => s.calibrated_probability == null && standaloneP(s) != null)
  const allOpen = rows.length > 0 && rows.every((s) => open[s.id])
  const toggleAll = () => setOpen(allOpen ? {} : Object.fromEntries(rows.map((s) => [s.id, true])))

  // Evidence ids in the key findings link here (`#signal-<id>`): the linked row opens.
  useEffect(() => {
    const openFromHash = () => {
      const m = /^#(signal-.+)$/.exec(window.location.hash)
      const key = m ? decodeURIComponent(m[1]) : null
      const hit = key ? signals.find((s) => anchorOf(s) === key) : undefined
      if (hit) setOpen((o) => ({ ...o, [hit.id]: true }))
    }
    openFromHash()
    window.addEventListener('hashchange', openFromHash)
    return () => window.removeEventListener('hashchange', openFromHash)
  }, [signals])

  // N-19b: where no check has a contribution (a fallback result), that column holds dashes and the Observed column takes its room
  const anyContribution = rows.some((s) => s.contribution_logit != null)

  // Guided demo (re-targeted in E-8): open the first row, or the first row whose id or category contains the match.
  useDemoAction(useCallback((a: StepAction) => {
    if (a.type === 'expand-first-row' && rows[0]) setOpen((o) => ({ ...o, [rows[0].id]: true }))
    if (a.type === 'expand-row') {
      const m = a.match.toLowerCase()
      const hit = rows.find((s) => s.id.toLowerCase().includes(m) || s.category.toLowerCase().includes(m))
      if (hit) setOpen((o) => ({ ...o, [hit.id]: true }))
    }
  }, [rows]))

  // `e` toggles expand all while focus is inside the ledger and not in a form field (WCAG 2.1.4).
  const onKeyDown = (e: KeyboardEvent<HTMLElement>) => {
    const t = e.target as HTMLElement
    if (e.key === 'e' && !e.metaKey && !e.ctrlKey && !e.altKey && !['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName) && !t.isContentEditable && !sheet) toggleAll()
  }

  return (
    <section aria-labelledby="ledger-heading" data-testid="ledger" data-tour="evidence" onKeyDown={onKeyDown}>
      <div className="flex items-end justify-between gap-6 flex-wrap">
        <h3 id="ledger-heading" className="m-0 eyebrow">Evidence ledger</h3>
        <div className="flex items-center gap-5">
          <Menu label={sort === 'effect' ? 'Sort: by effect' : 'Sort: by id'} onSelect={setSort} align="right"
            items={[{ id: 'effect', label: 'By effect' }, { id: 'id', label: 'By id' }]} />
          <Button variant="text" onClick={toggleAll} aria-keyshortcuts="e">{allOpen ? 'Collapse all' : 'Expand all'}</Button>
        </div>
      </div>
      {rows.length === 0 && notRun.length === 0 && <EmptyState className="mt-3">The analysis recorded no signals.</EmptyState>}
      {(rows.length > 0 || notRun.length > 0) && (
        <div className="mt-3 overflow-x-auto">
          {/* N-19b (design review DS-13). Below 640 px each check is one stacked block (UX §5.17): its id and name, the
              direction and the strength, the contribution and the score; the Signal column was 0 px wide there and two
              headers overprinted. At 1440 px the Strength header fits its column, and where no check has a contribution the
              Observed column takes the room of that column. */}
          <table role="table" className="stack-sm w-full border-collapse table-fixed" data-testid="ledger-table" data-contributions={anyContribution}>
            <colgroup className="max-[639px]:hidden">
              <col className="w-14" />
              <col />
              <col className={cx(anyContribution ? 'w-[220px]' : 'w-[260px]', 'max-[1279px]:w-0')} />
              <col className="w-[104px]" />
              <col className="w-[92px] max-[959px]:w-0" />
              <col className={anyContribution ? 'w-40' : 'w-[120px]'} />
              <col className="w-14" />
              <col className="w-7" />
            </colgroup>
            <thead role="rowgroup">
              <tr role="row" className="border-b border-border-strong">
                <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 pr-3">#</th>
                <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 pr-4">Signal</th>
                <th role="columnheader" scope="col" className={cx('eyebrow text-left font-medium py-2 pr-4', HIDE_1279)}>Observed</th>
                <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 pr-4">Direction</th>
                <th role="columnheader" scope="col" className={cx('eyebrow text-left font-medium py-2 pr-4', HIDE_959)}>Strength</th>
                <th role="columnheader" scope="col" className="eyebrow text-right font-medium py-2 pr-4">Contribution</th>
                <th role="columnheader" scope="col" className="eyebrow text-right font-medium py-2 pr-2">P(AI)</th>
                <th role="columnheader" scope="col"><span className="sr-only">Expand</span></th>
              </tr>
            </thead>
            <tbody role="rowgroup">
              {rows.map((s) => {
                const o = !!open[s.id]
                const direction = signalDirection(s, little)
                const uncounted = rowNotCounted(s.id, analysis.decision as Decision | null)
                const strength = uncounted ? { level: 'none' as const, muted: true, word: NOT_COUNTED, outside: false } : signalStrength(s, findings, measured, analysis.decision as Decision | null)
                const p = standaloneP(s)
                const detailId = `ledger-${s.display_id ?? s.id}`
                const toggle = () => setOpen((x) => ({ ...x, [s.id]: !o }))
                return (
                  <Fragment key={s.id}>
                    <tr role="row" id={anchorOf(s)} data-testid="ledger-row" data-signal={s.id} data-tour-signal={s.id} data-tour-category={s.category}
                      className={cx('scroll-mt-14 border-b border-border cursor-pointer transition-colors duration-fast hover:bg-surface-3', STACK_ROW, o && 'bg-surface-2 border-b-0')} onClick={toggle}>
                      <td role="cell" className="py-3 pr-3 align-top font-mono text-mono num text-text max-[639px]:col-start-1 max-[639px]:row-start-1">{s.display_id ?? '—'}</td>
                      <td role="cell" className="py-3 pr-4 align-top min-w-0 max-[639px]:col-start-2 max-[639px]:col-span-2 max-[639px]:row-start-1">
                        <button type="button" aria-expanded={o} aria-controls={detailId} onClick={(e) => { e.stopPropagation(); toggle() }} className="block w-full min-w-0 text-left rounded-sm">
                          <span data-testid="ledger-name" className="block text-body font-medium text-text truncate max-[639px]:whitespace-normal max-[639px]:[overflow-wrap:anywhere]" title={s.name}>{s.name}</span>
                          <span className="block eyebrow text-micro mt-0.5 truncate">{[s.category, lensFamily(s, exhibits)].filter((word, i, words) => words.findIndex((w) => w.trim().toLowerCase() === word.trim().toLowerCase()) === i).join(' · ')}</span>
                        </button>
                      </td>
                      {/* two lines of the observation, with the whole of it as the hover text */}
                      <td role="cell" data-testid="ledger-observed" className={cx('py-3 pr-4 align-top font-mono text-mono text-text max-[639px]:hidden', HIDE_1279)} title={isMention(s) ? mentionWords(s.observed) : s.observed}>
                        <span className="line-clamp-2 [overflow-wrap:anywhere]">{isMention(s) ? mentionWords(s.observed) : s.observed}</span>
                      </td>
                      <td role="cell" className="py-3 pr-4 align-top whitespace-nowrap max-[639px]:col-start-2 max-[639px]:row-start-2">
                        <DirectionMark direction={uncounted ? 'neutral' : direction} word={isMention(s) ? MENTION_DIRECTION : uncounted || (measured && direction === 'neutral' && !entered(s.contribution_logit)) ? 'neutral' : undefined} />
                      </td>
                      <td role="cell" className={cx('py-3 pr-4 align-top max-[639px]:!visible max-[639px]:col-start-3 max-[639px]:row-start-2', HIDE_959)}>
                        <StrengthMeter level={strength.level} muted={strength.muted} label={strength.word} direction={direction === 'ai' ? 'for' : direction === 'authentic' ? 'against' : undefined} />
                        {strength.outside && <span data-testid="not-in-estimate" className="block mt-0.5 text-caption text-text-muted whitespace-nowrap">{NOT_IN_ESTIMATE}</span>}
                      </td>
                      <td role="cell" data-th="contribution" className="py-3 pr-4 align-top text-right max-[639px]:text-left max-[639px]:col-start-2 max-[639px]:col-span-2 max-[639px]:row-start-3"><ContributionCell logit={s.contribution_logit} uncounted={uncounted} /></td>
                      <td role="cell" data-th={s.calibrated_probability != null ? 'P(AI)' : 'raw score'} className={cx('py-3 pr-2 align-top text-right font-mono text-mono num whitespace-nowrap max-[639px]:text-left max-[639px]:col-start-2 max-[639px]:col-span-2 max-[639px]:row-start-4', s.calibrated_probability != null ? 'text-text' : 'text-text-muted')}
                        title={s.calibrated_probability != null ? (analysis.decision?.fusion_mode === 'artefact' ? 'Probability calibrated by the fusion model' : 'Calibrated probability') : 'Raw detector score, not a calibrated probability'}>{prob(p)}</td>
                      <td role="cell" className="py-3 align-top text-text-muted max-[639px]:col-start-4 max-[639px]:row-start-1">
                        <ChevronRight size={14} strokeWidth={1.5} aria-hidden className={cx('transition-transform duration-base ease-standard', o && 'rotate-90')} />
                      </td>
                    </tr>
                    <tr role="row" className={cx('border-b border-border max-[639px]:!p-0', !o && 'border-b-0')}>
                      <td role="cell" colSpan={COLS} className="p-0 max-[639px]:!p-0">
                        <div id={detailId} className="disclosure bg-surface-2" data-open={o}>
                          <div>{o && <Expanded s={s} analysis={analysis} exhibits={exhibits} token={token} totalWeight={totalWeight} largest={largest} onCard={(id, trigger) => setSheet({ id, trigger })} />}</div>
                        </div>
                      </td>
                    </tr>
                  </Fragment>
                )
              })}
              {notRun.length > 0 && (
                <tr role="row"><td role="cell" colSpan={COLS} className="pt-6 pb-2"><Eyebrow as="span" data-testid="not-run-heading">Not run ({notRun.length})</Eyebrow></td></tr>
              )}
              {notRun.map((r) => (
                <tr role="row" key={r.key} data-testid="not-run-row" data-status={r.status} id={r.signalId ? `signal-${r.signalId}` : undefined} className="border-b border-border max-[639px]:flex max-[639px]:gap-x-3">
                  <td role="cell" className="py-2.5 pr-3 align-top font-mono text-mono num text-text-muted max-[639px]:w-12 max-[639px]:shrink-0">{r.displayId ?? ''}</td>
                  <td role="cell" colSpan={COLS - 1} className="py-2.5 align-top max-[639px]:min-w-0">
                    <span className="flex flex-wrap items-center gap-x-3 gap-y-1 text-small">
                      <HatchSwatch label={<span className="text-text">{r.name}</span>} />
                      <span className="text-text-muted">{ofTheServer(r.status, r.reason) ? ON_THIS_SERVER : statusWord(r.status)} · {r.reason}</span>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {anyRaw && <p className="mt-3 m-0 text-caption text-text-muted">P(AI) in muted type is a raw detector score, not a calibrated probability.</p>}
      <ModelCardSheet detectorId={sheet?.id ?? null} returnFocus={sheet?.trigger ?? null} onClose={() => setSheet(null)} />
    </section>
  )
}
