import StandardScreeningNotice from './StandardScreeningNotice'
import SourceLabels from './SourceLabels'
import { useId, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import type { AnalysisDetail, Finding } from '../../lib/api/queries'
import { cx } from '../../lib/cx'
import { Button, ConfidenceMeter, Eyebrow, ScaleInterval } from '../../components/ui'
import {
  agreement, basisOf, checksCounted, errorRateLine, flaggedOutsideSentence, guidance, headline, insufficientReasons, invalidSentence, lowAgreement, nextStep, noEstimateLine, notRunCount, notRunOfServer,
  uncalibratedNote,
  numeral, stepOf, type Basis, type FlagSignal,
} from './answer'
import { numberPhrase, resultNumber, scaleNote, statisticalAlone } from './resultNumber'
import { CAPTURE_BASIS, aloneOtherWay, markerBasisWord, markerOf, outsideValidated, recordOf, statisticalWithheld } from './record'
import { failedSteps, findingsFailed } from './vocabulary'
import { presentedCustomerAssessment } from './customerAssessment'

const STEP_TEXT = ['text-text', 'text-verdict-1', 'text-verdict-2', 'text-verdict-3', 'text-verdict-4', 'text-verdict-5'] as const

function basisWord(basis: Basis, coverage: AnalysisDetail['coverage'], d: NonNullable<AnalysisDetail['decision']>, outside = false): string {
  const checks = checksCounted(coverage, d, outside)
  const ok = coverage.by_status.ok ?? 0
  switch (basis) {
    case 'documentary_declared': return 'Documentary · Content Credentials'
    case 'documentary_capture': return CAPTURE_BASIS  // N-19m (D3-01): no count of checks; the result is the record's
    case 'documentary_metadata': return `${markerBasisWord(markerOf(d))} · ${checks}`  // N-19h: the marker's kind
    // N-19d: never "0 of 0 checks ran"
    case 'insufficient': return coverage.detectors_total === 0 ? 'Insufficient · no check was applied'
      : `Insufficient · ${ok} of ${coverage.detectors_total} ${coverage.detectors_total === 1 ? 'check' : 'checks'} ran`
    default: return `Inferential · ${checks}`
  }
}

const METHOD_WORD: Record<string, string> = { platt: 'Platt scaling', isotonic: 'Isotonic regression', 'legacy-heuristic': 'Hand-set weights' }

/** "Calibrated" fact (UX §5.3): the fuser's method, benchmark and test-set size, or the uncalibrated fallback said plainly. */
function calibratedWord(a: AnalysisDetail, d: NonNullable<AnalysisDetail['decision']>, plain = false): string {
  if (d.fusion_mode === 'fallback') return plain ? 'No · the checks are combined with hand-set weights' : 'No · uncalibrated fallback with hand-set weights'
  const fuser = (a.calibration ?? []).find((c) => c.subject.startsWith('fuser-'))
  if (!fuser) return 'not established'
  const row = fuser.error_rates?.[0]
  // N-19d (browser QA "Copy": "Platt scaling · vb-1.1" in Plain view): no name of a method or of a test set there
  if (plain) return row ? `Yes · against a test set of ${(row.n_pos + row.n_neg).toLocaleString('en-US')} files` : 'Yes · against a test set'
  return [METHOD_WORD[fuser.method] ?? fuser.method, fuser.bench_version ?? 'benchmark not recorded', row ? `n = ${row.n_pos + row.n_neg}` : null]
    .filter(Boolean).join(' · ')
}

/** N-19k (D2-03): the fact "Calibrated" of a result outside what was validated. */
export const CALIBRATION_OUTSIDE = 'Not for this kind of input · the calibration does not cover it'
/** N-19m (D3-01): the fact "Calibrated" belongs to the number. Under a record that decides and did not set it, the number
 *  shown is the statistical checks' own calibrated estimate, and the fact says whose it is before its own words. */
export const CALIBRATED_CHECKS = 'the statistical checks’ estimate'

const sentence = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)
/** N-19n (D4-02): the statistical checks' number beside a record is set in the muted ink, and in the ink of the text where
 *  they alone decide the other answer than the record's ("98 out of 100, they point the other way"). */
const againstInk = (d: NonNullable<AnalysisDetail['decision']>, profile: AnalysisDetail['condition_profile']) =>
  aloneOtherWay(d) && !statisticalWithheld(d, profile) ? 'text-text' : 'text-text-muted'

/** L1 answer (UX §4.3, §5.3; D-16): headline by decision code, estimate and plausible range on the scale, the facts, "what this
 *  means / does not mean" with the natural-frequency line, and the next step. Only rendered once fusion has finished.
 *  The number follows `resultNumber` (D-68): the large numeral and the scale marker are for a decisive calibrated result in which
 *  two checks scored. `onExpert` is set in the plain view, where the not-validated list is one sentence and a link. */
export default function AnswerModule({ analysis, findings, signals = [], extra, onExpert, opinion }: {
  analysis: AnalysisDetail; findings: readonly Finding[]; signals?: readonly FlagSignal[]; extra?: ReactNode; onExpert?: () => void
  /** N-19 (D-69): the examiner's opinion of a signed record, directly under the machine headline */
  opinion?: ReactNode
}) {
  const headingId = useId()
  const d = analysis.decision
  if (!d) return null
  const basis = basisOf(d, findings)
  const retiredMarker = recordOf(d)?.kind === 'self_declared_ai'
  const customer = presentedCustomerAssessment({ ...analysis, findings, signals })
  const step = customer.applies ? customer.direction === 'ai' ? 4 : customer.direction === 'original' ? 2
    : customer.direction === 'unavailable' ? 0 : 3 : stepOf(d, basis)
  const n = resultNumber(d, findings, analysis.condition_profile)
  const ci = n.kind === 'calibrated' || n.kind === 'text' ? n.range : null
  const phrase = numberPhrase(n)
  const g = guidance(analysis.kind, d, basis, errorRateLine(analysis.kind, d, analysis.condition_profile, analysis.calibration ?? []), analysis.coverage, findings, signals,
    analysis.condition_profile, analysis.caveats)
  const customerLimits = [customer.limitation, flaggedOutsideSentence(analysis.kind,
    customer.direction === 'original' ? { ...d, code: 'no_ai_indicators' } : d, findings, signals), invalidSentence(findings)]
    .filter((text): text is string => !!text)
  const displayGuidance = customer.applies ? { means: [customer.meaning], doesNotMean: customerLimits } : g
  const displayedInterval = customer.applies ? customer.interval_pct : ci
  // N-19n (Q4-01): the step that writes the findings failed: no agreement is worked out, and the next step is to run again
  const noFindings = findingsFailed(analysis.caveats)
  const agree = agreement(d, analysis.coverage, findings, noFindings)
  const notRun = notRunCount(analysis.coverage)
  const notValidated = analysis.not_validated ?? []
  const stepsFailed = failedSteps(analysis.caveats)
  const documentary = basis === 'documentary_declared' || basis === 'documentary_capture'
  // the number shown is the statistical checks' own calibrated estimate under a record that decides (CAPLOW's numeral,
  // CAPUNS's range, a marker over an estimate above its floor)
  const checksOwn = recordOf(d) !== null && d.fusion_mode === 'artefact' && (n.kind === 'calibrated' || n.kind === 'text')

  return (
    <section id="section-summary" data-testid="answer" data-code={d.code} data-basis={basis} data-tour="verdict" aria-labelledby={headingId}
      className="scroll-mt-14 border-b border-border pb-8">
      <Eyebrow as="h2" id={headingId}>Overall assessment</Eyebrow>
      <p data-testid="headline" className={cx('mt-3 m-0 display text-h1 font-normal max-w-[36ch]', STEP_TEXT[step])}>
        {customer.applies ? customer.headline : headline(d, basis)}{!customer.applies && lowAgreement(d, findings, analysis.condition_profile) && ' — the checks disagree'}
      </p>
      {customer.applies && customer.scope_note && <div data-testid="assessment-scope" className="mt-4 border-l-2 border-accent bg-surface-2 px-4 py-3 text-body text-text max-w-[76ch]">
        <p className="m-0 font-semibold">{customer.scope_label}</p>
        <p className="mt-1 mb-0">{customer.scope_note}</p>
      </div>}
      {customer.soundtrack && <section aria-label="Speech in this soundtrack" className="mt-4 border border-border bg-surface px-4 py-3 max-w-[76ch]">
        <p className="m-0 text-small font-semibold">Speech in this soundtrack</p>
        <p className="mt-1 mb-0 text-body text-text">{customer.soundtrack.headline}</p>
        {customer.soundtrack.confidence_pct !== null && <p className="mt-1 mb-0 text-small num">Reference-calibrated estimate <strong>{customer.soundtrack.confidence_pct}%</strong></p>}
        <p className="mt-2 mb-0 text-small text-text-muted">{customer.soundtrack.limitation}</p>
        <Link to={`/analyses/${encodeURIComponent(analysis.id)}/media-evidence?view=plain#media-evidence-voice`}
          className="mt-2 inline-block text-small text-accent underline underline-offset-2">Inspect voice evidence</Link>
      </section>}
      <StandardScreeningNotice />
      <SourceLabels value={analysis.manifest?.sampling?.source_labels} source={analysis.input?.sha256} />
      {opinion}

      <div className="mt-6 grid gap-10 min-[960px]:grid-cols-[7fr_5fr]">
        <div className="min-w-0">
          {customer.applies ? (
            <div data-testid="customer-assessment" data-direction={customer.direction}>
              {customer.confidence_pct !== null ? <>
                <p className="m-0 display text-h2 text-text num">{customer.confidence_label}: <strong className="font-medium">{customer.confidence_pct}%</strong></p>
                <p className="mt-2 mb-0 text-small text-text-muted">{customer.direction === 'ai' ? 'In favour of AI generation or AI alteration.'
                  : customer.scope === 'visual_content' ? 'In favour of original visual content.'
                    : customer.scope === 'assessed_components' ? 'In favour of original content in the assessed components.' : 'In favour of an original file.'} This is an estimate, not a verified accuracy rate.</p>
              </> : <p className="m-0 text-body text-text-muted">{customer.probability_ai_pct !== null
                ? `Estimated AI likelihood: ${customer.probability_ai_pct}%. No reliable overall tendency is established.`
                : 'A confidence percentage is not available for this result.'}</p>}
              <p className="mt-3 mb-0 text-small text-text-muted">The overall estimate uses only the applicable, calibrated checks. It is not an average of finding scores; raw responses and uncertainty interval endpoints are not confidence.</p>
            </div>
          ) : basis === 'documentary_declared' ? (
            <div data-testid="documentary-seal">
              <p className="m-0 display text-h2 text-text"><span aria-hidden>◆ </span>Documentary</p>
              <p className={cx('mt-2 m-0 text-small num', againstInk(d, analysis.condition_profile))}>Statistical checks alone: {statisticalAlone(d, analysis.condition_profile)}</p>
            </div>
          ) : n.kind === 'uncalibrated' ? (
            <div data-testid="estimate" data-number={n.kind} data-calibrated="false">
              {/* L-3: an uncalibrated score is not a likelihood, so it is never the hero numeral */}
              <p className="m-0 text-h3 font-normal text-text num">{sentence(n.text)}</p>
              <p className="mt-1 m-0 text-small text-text-muted max-w-[56ch]">{uncalibratedNote(d)}</p>
            </div>
          ) : n.kind === 'record' ? (
            // N-19h (E-20, 1): a number that the record set is said in words, with the statistical checks' own beside it
            <div data-testid="estimate" data-number={n.kind}>
              <p className="m-0 text-h3 font-normal text-text num max-w-[36ch]">{sentence(n.text)}</p>
              <p data-testid="statistical-alone" className={cx('mt-1 m-0 text-small num', againstInk(d, analysis.condition_profile))}>Statistical checks alone: {statisticalAlone(d, analysis.condition_profile)}</p>
            </div>
          ) : n.kind === 'text' ? (
            <p data-testid="estimate" data-number={n.kind} className="m-0 text-body text-text num">{n.text}</p>
          ) : n.kind === 'calibrated' ? (
            <div data-testid="estimate" data-number={n.kind} className="flex items-start gap-2">
              <span aria-hidden className="display text-numeral font-light num text-text">{numeral(n.estimate)}</span>
              <span aria-hidden className="flex flex-col pt-2">
                <span className="display text-h1 font-normal text-text leading-none">%</span>
                {/* N-19n (D4-03): under a record that did not set the number the numeral is the statistical checks' own */}
                <Eyebrow as="span" className="mt-2 normal-case tracking-normal">estimated<br />likelihood{recordOf(d) && <>,<br />statistical checks<br />alone</>}</Eyebrow>
              </span>
              <span className="sr-only">Estimated likelihood{recordOf(d) ? ', statistical checks alone,' : ''} {numeral(n.estimate)} of 100{ci ? `, plausible range ${ci[0]} to ${ci[1]}` : ''}.</span>
            </div>
          ) : (
            <p data-testid="no-estimate" data-number={n.kind} className="m-0 text-body text-text-muted max-w-[56ch]">
              {n.kind === 'capped' ? 'No estimate is shown: this input is outside what was validated.'
                : basis === 'insufficient' || notRunOfServer(d, analysis.coverage) ? noEstimateLine(analysis.coverage, d) : 'No estimate: a check that this analysis depends on did not run.'}
            </p>
          )}
          {!customer.applies && basis !== 'documentary_declared' && (
            <div className="mt-6" data-tour="scale">
              {n.kind === 'calibrated'
                ? <ScaleInterval step={step} estimate={n.estimate * 100} low={ci?.[0]} high={ci?.[1]} note={scaleNote(d, n.kind)} />
                : n.kind === 'none'
                  ? <ScaleInterval step={step} estimate={null} />
                  : <ScaleInterval step={step} band low={ci?.[0]} high={ci?.[1]} label={sentence(phrase ?? 'No estimate')} note={scaleNote(d, n.kind)} />}
            </div>
          )}
          {extra}
        </div>

        <dl data-testid="answer-facts" className="m-0 self-start grid grid-cols-[112px_minmax(0,1fr)] gap-x-4 gap-y-3">
          <dt className="eyebrow pt-1">Basis</dt>
          <dd className="m-0 text-body text-text num">{retiredMarker ? 'Unverified origin marker · historical calculation' : basisWord(basis, analysis.coverage, d, outsideValidated(d, analysis.condition_profile))}</dd>
          {!documentary && !retiredMarker && (
            // N-19m (D3-01): no fact "Agreement" under a documentary basis (a trusted declaration, a trusted capture), as the
            // reports print none there: the result is the record's ("Agreement: High" stood under a declaration)
            <>
              <dt className="eyebrow pt-1">Agreement</dt>
              <dd className="m-0 text-body text-text flex items-center gap-2.5 flex-wrap">
                <span>{agree.word}</span>{agree.level > 0 && <ConfidenceMeter level={agree.level as 1 | 2 | 3} />}
                {agree.why && agree.level === 0 && <span className="text-small text-text-muted">{agree.why}</span>}
                {agree.capped && <span className="text-small text-text-muted">{agree.why}</span>}
              </dd>
            </>
          )}
          {(!recordOf(d) || checksOwn || n.kind === 'uncalibrated') && (
            // N-19n (D4-05; check 31): the fact belongs to the number, as "Calibrated" does: it stands where the number shown is
            // the statistical checks' own, and not where a record set it ("Plausible: not established" read as something the
            // record failed to establish) or decides alone
            <>
              <dt className="eyebrow pt-1">Estimate range</dt>
              <dd data-testid="fact-plausible" className="m-0 text-body text-text num">
                {displayedInterval ? <>
                  <span className="font-medium">{displayedInterval[0]}–{displayedInterval[1]}%</span> <span className="text-small text-text-muted">({customer.applies ? customer.interval_label : '90% interval'})</span>
                  <p className="mt-1 mb-0 text-small text-text-muted">{customer.applies ? customer.interval_note : 'Uncertainty in the automated estimate. It does not cover every detection error or unfamiliar type of content.'}</p>
                </> : <span className="text-small text-text-muted">{n.kind === 'uncalibrated'
                  ? 'Unavailable — this analysis is not calibrated.'
                  : n.kind === 'capped' ? 'Unavailable — this input is outside the validated scope.'
                    : 'Unavailable — no reliable interval was established for this result.'}</span>}
              </dd>
            </>
          )}
          {n.kind !== 'record' && !(n.kind === 'none' && d.fusion_mode === 'artefact') && (
            // N-19h (E-20, 1): no fact "Calibrated" beside a number that a record set. N-19k (Q2-14; D2-03): none beside a
            // result of a fusion model that shows no estimate, which read "Platt scaling · vb-1.1" as if it had one (the
            // fallback's "No · …" claims nothing and stays); outside what was validated the fact says that the
            // calibration does not reach the input
            <>
              <dt className="eyebrow pt-1">{customer.applies ? 'Validation of this result' : 'Calibrated'}</dt>
              <dd data-testid="fact-calibrated" className="m-0 text-body text-text">
                {customer.applies ? 'Comparable error rates are not established for this result. Individual findings can be inspection-only; the combined estimate does not clear unassessed parts of the file.'
                  : n.kind === 'capped' && d.fusion_mode === 'artefact' ? CALIBRATION_OUTSIDE
                  : `${checksOwn ? `${CALIBRATED_CHECKS} · ` : ''}${calibratedWord(analysis, d, onExpert != null)}`}
              </dd>
            </>
          )}
          {notRun > 0 && (
            <>
              <dt className="eyebrow pt-1">Not run</dt>
              <dd className="m-0 text-body text-text num">{notRun} of {analysis.coverage.detectors_total} checks · listed in the analysis record</dd>
            </>
          )}
        </dl>
      </div>

      <div className="mt-8 grid gap-10 min-[960px]:grid-cols-2">
        <div data-testid="means" className="min-w-0">
          <h3 className="m-0 text-h3 font-semibold text-text">What this means</h3>
          {displayGuidance.means.map((p) => <p key={p} className="mt-2 m-0 text-body text-text max-w-[64ch]">{p}</p>)}
          {basis === 'insufficient' && (
            <ul className="mt-2 m-0 p-0 list-none space-y-1">
              {insufficientReasons(d, analysis.condition_profile, analysis.coverage).map((r) => (
                <li key={r} className="grid grid-cols-[16px_minmax(0,1fr)] text-body text-text"><span aria-hidden className="text-text-muted">·</span><span>{r}</span></li>
              ))}
            </ul>
          )}
        </div>
        <div data-testid="does-not-mean" className="min-w-0">
          <h3 className="m-0 text-h3 font-semibold text-text">What this does not mean</h3>
          {displayGuidance.doesNotMean.map((p) => <p key={p} className="mt-2 m-0 text-body text-text max-w-[64ch]">{p}</p>)}
        </div>
      </div>

      {notValidated.length > 0 && (
        <div data-testid="not-validated" className="mt-8 grid grid-cols-[112px_minmax(0,1fr)] gap-x-4 max-[639px]:grid-cols-1">
          <h3 className="m-0 eyebrow pt-1">Not validated</h3>
          {onExpert ? (
            <div className="max-w-[72ch]">
              <p data-testid="not-validated-sentence" className="m-0 text-small text-text">
                {notValidated.length === 1 ? notValidated[0].text : `Not validated for this result: ${notValidated.map((x) => x.label).join('; ')}.`}
              </p>
              <Button variant="text" className="mt-1" onClick={onExpert}>View validation details</Button>
            </div>
          ) : (
            <ul className="m-0 p-0 list-none space-y-1.5 max-w-[72ch]">
              {notValidated.map((x) => <li key={x.id} data-id={x.id} className="text-small text-text">{x.text}</li>)}
            </ul>
          )}
        </div>
      )}

      {stepsFailed.length > 0 && (
        // N-18b: a step of the run failed and the result lacks what it gives; said here, in both views, in the result's words
        <div data-testid="step-failed" className="mt-8 grid grid-cols-[112px_minmax(0,1fr)] gap-x-4 max-[639px]:grid-cols-1">
          <h3 className="m-0 eyebrow pt-1">Step failed</h3>
          <ul className="m-0 p-0 list-none space-y-1.5 max-w-[72ch]">
            {stepsFailed.map((c) => <li key={c} className="text-small text-text">{c}</li>)}
          </ul>
        </div>
      )}

      <div data-testid="next-step" className="mt-8 grid grid-cols-[112px_minmax(0,1fr)] gap-x-4 max-[639px]:grid-cols-1">
        <h3 className="m-0 eyebrow pt-1">Next step</h3>
        <p className="m-0 text-body text-text max-w-[64ch]">{nextStep(d, basis, analysis.coverage, noFindings)}</p>
      </div>
    </section>
  )
}
