import { Link } from 'react-router-dom'
import type { AnalysisDetail, ModelCard } from '../../lib/api/queries'
import { Dl, ErrorState, SkeletonRows } from '../../components/ui'
import CalibrationDiagram from './CalibrationDiagram'
import { withCi } from '../models/catalogue'
import { detectorValidation, envelopeStatement, fusedErrorRates, methodRows, reliabilityFactors, sliceRows, type ErrorRate, type Row } from './methods'

const THRESHOLD_WORD: Record<string, string> = { 'conformal_0.05': '5 % conformal', 'conformal_0.01': '1 % conformal' }
const threshold = (r: ErrorRate) => THRESHOLD_WORD[r.threshold_kind] ?? `fixed at ${r.threshold}`
const value = (r: Row) => <span className={r.established ? undefined : 'text-text-muted'}>{r.value}</span>
const th = (h: string) => <th key={h} role="columnheader" scope="col" className="eyebrow text-left font-medium py-1.5 pr-4 whitespace-nowrap">{h}</th>
/** N-19b (design review DS-13): below 640 px a table of this section is a list of stacked blocks, each value under its
 *  column's name, so nothing is held off-canvas in a sideways scroller. */
const TABLE = 'stack-sm w-full border-collapse text-small'

/** Methods & validation (PRD UX-01, RP-04; ARCHITECTURE §6.4 section 4; E-7): whether the input sits inside the validated envelope,
 *  the dossier slice metrics with intervals, the fused error rates in testing, each detector's validation for this slice, the
 *  method in plain terms and the reliability-factor table. Absent records read "not established"; no cell is blank. */
export default function MethodsPanel({ analysis, cards, cardsPending = false, cardsError = null, onRetryCards }: {
  analysis: AnalysisDetail; cards: readonly ModelCard[] | undefined; cardsPending?: boolean; cardsError?: Error | null; onRetryCards?: () => void
}) {
  const envelope = envelopeStatement(analysis.condition_profile)
  const rates = fusedErrorRates(analysis.calibration)
  const detectors = detectorValidation(analysis, cards)
  return (
    <section id="section-methods" data-testid="methods" aria-labelledby="methods-heading" className="scroll-mt-14">
      <h2 id="methods-heading" className="m-0 eyebrow">Methods &amp; validation</h2>
      <p data-testid="methods-envelope" data-within={String(envelope.within)} className="mt-3 m-0 text-body text-text max-w-[80ch]">{envelope.text}</p>

      <div className="mt-6 grid grid-cols-2 gap-x-12 gap-y-8 max-[959px]:grid-cols-1">
        <div data-testid="methods-slice">
          <h3 className="m-0 eyebrow">Validation slice</h3>
          <Dl className="mt-2" keyWidth={128} rows={sliceRows(analysis).map((r) => ({ key: r.key, value: value(r) }))} />
        </div>
        <div data-testid="methods-method">
          <h3 className="m-0 eyebrow">Method</h3>
          <Dl className="mt-2" keyWidth={112} rows={methodRows(analysis).map((r) => ({ key: r.key, value: value(r) }))} />
        </div>
      </div>

      <h3 className="mt-8 m-0 eyebrow">Error rates in testing</h3>
      {rates.length > 0 ? (
        <div className="mt-2 overflow-x-auto">
          <table role="table" data-testid="methods-error-rates" className={TABLE}>
            <thead role="rowgroup"><tr role="row" className="border-b border-border-strong">{['Context', 'Threshold', 'TPR [95 % CI]', 'FPR [95 % CI]', 'n AI / genuine', 'Benchmark'].map(th)}</tr></thead>
            <tbody role="rowgroup">
              {rates.map((r, i) => (
                <tr role="row" key={i} className="border-b border-border">
                  <td role="cell" className="py-1.5 pr-4 text-text max-[639px]:font-medium">{r.context}</td><td role="cell" data-th="threshold" className="py-1.5 pr-4 text-text">{threshold(r)}</td>
                  <td role="cell" data-th="TPR [95 % CI]" className="py-1.5 pr-4 font-mono text-mono num text-text">{withCi(r.tpr, r.tpr_ci95)}</td><td role="cell" data-th="FPR [95 % CI]" className="py-1.5 pr-4 font-mono text-mono num text-text">{withCi(r.fpr, r.fpr_ci95)}</td>
                  <td role="cell" data-th="n AI / genuine" className="py-1.5 pr-4 font-mono text-mono num text-text">{r.n_pos} / {r.n_neg}</td><td role="cell" data-th="benchmark" className="py-1.5 pr-4 text-text">{r.bench_version}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <p data-testid="methods-error-rates-empty" className="mt-2 m-0 text-small text-text-muted max-w-[80ch]">Not established: no measured error rate is recorded for the fusion that produced this result.</p>}

      <CalibrationDiagram analysis={analysis} />

      <h3 className="mt-8 m-0 eyebrow">Detectors</h3>
      {cardsError && <ErrorState className="mt-2" onRetry={onRetryCards}>Model cards could not be loaded: {cardsError.message}</ErrorState>}
      {cardsPending ? <SkeletonRows className="mt-2" /> : detectors.length > 0 ? (
        <div className="mt-2 overflow-x-auto">
          <table role="table" data-testid="methods-detectors" className={TABLE}>
            <thead role="rowgroup"><tr role="row" className="border-b border-border-strong">{['Detector', 'Measured for this slice', 'Admission'].map(th)}</tr></thead>
            <tbody role="rowgroup">
              {detectors.map((d) => (
                <tr role="row" key={d.id} className="border-b border-border">
                  <td role="cell" className="py-1.5 pr-4 align-top">
                    {/* N-19d: a check is named by its model card; its id is the reference under the name */}
                    {d.card && d.card.name && d.card.name !== d.id
                      ? <><Link to={`/models/${encodeURIComponent(d.id)}`} className="text-accent underline underline-offset-[3px]">{d.card.name}</Link>
                          <span className="block font-mono text-caption text-text-muted break-all">{d.id} · {d.version}</span></>
                      : <>{d.card
                          ? <Link to={`/models/${encodeURIComponent(d.id)}`} className="font-mono text-mono text-accent underline underline-offset-[3px] break-all">{d.id}</Link>
                          : <span className="font-mono text-mono text-text break-all">{d.id}</span>}
                          <span className="block font-mono text-caption text-text-muted">{d.version}</span></>}
                  </td>
                  <td role="cell" data-th="measured for this slice" className={d.established ? 'py-1.5 pr-4 align-top text-text' : 'py-1.5 pr-4 align-top text-text-muted'}>{d.measured}</td>
                  <td role="cell" data-th="admission" className="py-1.5 pr-4 align-top text-text-muted">{d.admission}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <p className="mt-2 m-0 text-small text-text-muted">The reproducibility manifest lists no detectors.</p>}

      <h3 className="mt-8 m-0 eyebrow">Reliability factors</h3>
      {cardsPending ? <SkeletonRows className="mt-2" /> : (
        <div className="mt-2 overflow-x-auto">
          <table role="table" data-testid="reliability-factors" className={TABLE}>
            <thead role="rowgroup"><tr role="row" className="border-b border-border-strong">{['Factor', 'On record'].map(th)}</tr></thead>
            <tbody role="rowgroup">
              {reliabilityFactors(analysis, cards).map((r) => (
                <tr role="row" key={r.key} data-testid="reliability-factor" data-established={r.established} className="border-b border-border">
                  <th role="rowheader" scope="row" className="py-1.5 pr-4 align-top text-left font-normal text-text w-64 max-[959px]:w-40 max-[639px]:block max-[639px]:w-auto max-[639px]:p-0 max-[639px]:font-medium">{r.key}</th>
                  <td role="cell" className="py-1.5 pr-4 align-top max-w-[80ch]">{value(r)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
