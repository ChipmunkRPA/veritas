import type { ReactNode } from 'react'
import { api, hasApiKey } from '../../lib/api'
import { useStages, type AnalysisDetail, type DetectorRun, type ModelCard } from '../../lib/api/queries'
import { fmtBytes, fmtSeconds, shortHash } from '../../lib/format'
import { calibrationMethod, calibrationSubject, deviceWords, fusionWords, policyWords, samplingWords, versionWords } from '../../lib/recordWords'
import { conditionWord, sliceWords } from '../../lib/sliceWords'
import { Button, ButtonLink, CodeChip, Disclosure, Dl, type DlRow } from '../../components/ui'
import { integrity, signed } from './contributions'
import { testViewWords } from './methods'
import { fuserReview } from './vocabulary'
import { errorRateLine, type Decision } from './answer'
import { outsideValidated } from './record'
import { appendixStatistical } from './resultNumber'
import { anonymizeModelNames } from './checkNames'

const or = (v: ReactNode | null | undefined, empty = 'not recorded') => (v == null || v === '' ? <span className="text-text-muted">{empty}</span> : v)
const hash = (h: string | null | undefined) => (h ? <CodeChip copy={h} title="Copy full hash">{shortHash(h, 10, 6)}</CodeChip> : or(null))
const mono = (s: string) => <span className="font-mono text-mono break-all">{s}</span>

function Table({ head, rows, testId }: { head: string[]; rows: ReactNode[][]; testId?: string }) {
  return (
    <div className="overflow-x-auto">
      <table data-testid={testId} className="w-full border-collapse text-small">
        <thead><tr className="border-b border-border-strong">{head.map((h) => <th key={h} scope="col" className="eyebrow text-left font-medium py-1.5 pr-4 whitespace-nowrap">{h}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i} className="border-b border-border">{r.map((c, j) => <td key={j} className="py-1.5 pr-4 align-top text-text">{c}</td>)}</tr>)}</tbody>
      </table>
    </div>
  )
}

/** L4 technical appendix (UX §4.1; ARCHITECTURE §6.6 reproducibility manifest, CC-03): engine and model hashes, configuration,
 *  calibration, condition profile, the contribution-integrity record, caveats, timings, the v1 flat metadata and the raw JSON. */
export default function TechnicalAppendix({ analysis, runs, token, legacyMetadata, engine }: {
  analysis: AnalysisDetail; runs: readonly DetectorRun[]; token: string; legacyMetadata?: Record<string, unknown> | null; engine: string | null
  /** The model cards, for the name of a check beside its id (N-19d); without them the id stands alone. */
  cards?: readonly ModelCard[]
}) {
  const stages = useStages(analysis.id)
  const m = analysis.manifest
  const d = analysis.decision as Decision | null | undefined
  const input = analysis.input
  const cp = analysis.condition_profile
  const runOf = (id: string) => runs.find((r) => r.detector_id === id)
  const check = d ? integrity(d) : null

  const manifestRows: DlRow[] = m ? [
    { key: 'Chipmunk Verify', value: mono(m.veritas_version) },
    { key: 'Pipeline', value: mono(m.pipeline_version) },
    { key: 'Profile', value: `${m.profile}${m.effective_profile && m.effective_profile !== m.profile ? ` → ${m.effective_profile}` : ''}` },
    { key: 'Licence posture', value: or(policyWords(m.licence_posture)) },
    { key: 'Git commit', value: hash(m.git_commit) },
    { key: 'Config sha256', value: hash(m.config_sha256) },
    { key: 'Device', value: or(deviceWords(m.device, m.precision)) },
    { key: 'Template set', value: or(versionWords(m.template_set)) },
    { key: 'Fusion', value: or(fusionWords(d?.fusion_mode)) },
    { key: 'Calibration id', value: hash(m.calibration_id ?? d?.fuser_sha256) },
    { key: 'Policy thresholds', value: or(m.policy_thresholds_id) },
    { key: 'Libraries', value: Object.keys(m.library_versions ?? {}).length ? mono(Object.entries(m.library_versions ?? {}).map(([k, v]) => `${k} ${v}`).join(' · ')) : or(null) },
    { key: 'Sampling', value: or(samplingWords(m.sampling), 'not applicable') },
    ...((m.disclosures ?? []).length ? [{ key: 'Disclosures', value: anonymizeModelNames((m.disclosures ?? []).join(' ')) }] : []),
  ] : []

  // N-19l (R6-02): outside what was validated no line gives the statistical estimate or its logit. That the rows add up is
  // still checked, and the outcome is said in words without the two numbers
  const outside = outsideValidated(d, cp)
  const integrityText = !check ? 'No decision recorded.'
    : outside && check.ok
      ? 'The contributions add up to the estimate of the statistical checks, within the tolerance. Neither number is shown: this input is outside what was validated.'
    : outside && !check.ok && check.reason === 'mismatch'
      ? 'Mismatch: the contributions do not add up to the estimate of the statistical checks within the tolerance. Neither number is shown: this input is outside what was validated. The waterfall is not drawn.'
    : check.ok
      ? `Contributions add up to the estimate: Σ ${signed(check.sum, 6)} vs logit(estimate) ${signed(check.target, 6)} (tolerance ${check.tolerance.toExponential(1)}${check.clamped ? '; estimate at the 2–98 clamp' : ''}).`
      : check.reason === 'mismatch'
        ? `Mismatch: Σ contributions ${signed(check.sum ?? 0, 6)} vs logit(estimate) ${signed(check.target ?? 0, 6)} exceeds the tolerance ${(check.tolerance ?? 0).toExponential(1)}. The waterfall is not drawn.`
        : check.reason === 'record'  // N-19h (E-19): a record decides alone
          ? 'Not applicable: a provenance record decides this result alone, so there is no statistical estimate to decompose.'
        : check.reason === 'fallback'
          ? 'Not applicable: the uncalibrated fallback records no additive contributions.'
          : check.reason === 'no_rows' ? 'The fusion model recorded no contribution rows.' : 'No estimate to decompose.'

  const stageRows = (stages.data ?? []).map((s) => {
    const secs = s.started_at && s.finished_at ? (Date.parse(s.finished_at) - Date.parse(s.started_at)) / 1000 : null
    return [mono(s.stage), s.status, secs != null ? mono(fmtSeconds(secs, 2)) : '—']
  })

  return (
    <section id="section-appendix" data-testid="appendix" aria-labelledby="appendix-heading" className="scroll-mt-14">
      <h2 id="appendix-heading" className="m-0 eyebrow">Technical appendix</h2>
      <div className="mt-3 flex flex-col gap-1">
        <Disclosure label="Reproducibility manifest" summary={m ? m.pipeline_version : 'not recorded'} testId="appendix-manifest">
          {m ? <Dl rows={manifestRows} keyWidth={144} /> : <p className="m-0 text-small text-text-muted">This analysis carries no reproducibility manifest (v0.3 pipeline).</p>}
        </Disclosure>
        <Disclosure label="Detectors and weights" summary={`${(m?.detectors ?? []).length}`} testId="appendix-detectors">
          <Table testId="appendix-detector-table" head={['Check', 'Version', 'Weights / revision', 'Licence', 'Device', 'Status', 'Duration']}
            rows={(m?.detectors ?? []).map((x, i) => {
              const r = runOf(x.id)
              return [`Check ${i + 1}`, mono(x.version), x.weights_sha256 ? hash(x.weights_sha256) : x.revision ? mono(shortHash(x.revision)) : or(null, 'model-free'), or(x.licence),
                or(deviceWords(x.device ?? r?.device, x.precision ?? r?.dtype)), r?.status ?? '—', r?.duration_ms != null ? mono(fmtSeconds(r.duration_ms / 1000, 2)) : '—']
            })} />
        </Disclosure>
        <Disclosure label="Input and result" summary={input ? shortHash(input.sha256) : shortHash(analysis.sha256)} testId="appendix-input">
          <Dl keyWidth={144} rows={[
            { key: 'sha256', value: hash(input?.sha256 ?? analysis.sha256) },
            { key: 'md5', value: input?.md5 ? mono(input.md5) : or(null) },
            { key: 'Size', value: fmtBytes(input?.size_bytes ?? analysis.size_bytes) },
            { key: 'Sniffed type', value: or(input?.mime_sniffed ?? analysis.content_type) },
            { key: 'Intake event', value: input?.intake_event_id ? mono(`custody event ${input.intake_event_id}`) : or(null, 'not recorded (quick upload)') },
            { key: 'Result hash', value: hash(analysis.result_hash) },
            { key: 'Revision', value: `${analysis.revision}${analysis.supersedes_id ? ` · supersedes ${analysis.supersedes_id.slice(0, 8)}` : ''}` },
          ]} />
        </Disclosure>
        <Disclosure label="Calibration and fusion" summary={fusionWords(d?.fusion_mode) ?? '—'} testId="appendix-calibration">
          <Table head={['Subject', 'Method', 'Benchmark', 'ECE', 'Brier', 'Error-rate rows', 'Review']}
            rows={(analysis.calibration ?? []).map((c) => [anonymizeModelNames(calibrationSubject(c.subject)), calibrationMethod(c.method), or(c.bench_version), c.ece != null ? <>{mono(c.ece.toFixed(3))} <span className="text-text-muted">{testViewWords(c.test_view)}</span></> : '—', c.brier != null ? mono(c.brier.toFixed(3)) : '—', String((c.error_rates ?? []).length), fuserReview(c.reviewed_by, c.review_date)])} />
          <p data-testid="appendix-integrity" className="mt-3 m-0 text-small text-text max-w-[80ch]">{integrityText}</p>
          {/* N-19h (E-20, 8): outside what was validated the number is printed beside no record */}
          {d && appendixStatistical(d, cp) != null && <p data-testid="appendix-statistical" className="mt-1 m-0 text-small text-text">{appendixStatistical(d, cp)}</p>}
        </Disclosure>
        <Disclosure label="Condition profile" summary={sliceWords(cp?.slice_id)} testId="appendix-conditions">
          {cp ? <Dl keyWidth={144} rows={[
            { key: 'Validation slice', value: sliceWords(cp.slice_id) },
            // L-3: "inside" never stands alone beside error rates that are not established
            { key: 'Validated envelope', value: !cp.within_validated_envelope ? 'outside'
              : errorRateLine(analysis.kind, d ?? ({} as Decision), cp, analysis.calibration ?? []).startsWith('Known error rates are not established')
                ? 'inside a measured slice; error rates for inputs like this are not established' : 'inside' },
            { key: 'Limiting conditions', value: (cp.limiting_conditions ?? []).join('; ') || 'none' },
            ...Object.entries(cp).filter(([k, v]) => !['slice_id', 'within_validated_envelope', 'limiting_conditions'].includes(k) && v != null).map(([k, v]) => ({ key: conditionWord(k), value: k === 'validation_measured_at' ? sliceWords(String(v)) : mono(String(v)) })),
          ]} /> : <p className="m-0 text-small text-text-muted">No condition profile recorded.</p>}
        </Disclosure>
        <Disclosure label="Caveats" summary={`${(analysis.caveats ?? []).length}`} testId="appendix-caveats">
          <ul className="m-0 pl-4 list-disc text-small text-text space-y-1 max-w-[80ch]">{(analysis.caveats ?? []).map((c) => <li key={c}>{anonymizeModelNames(c)}</li>)}</ul>
        </Disclosure>
        <Disclosure label="Timings" summary={analysis.started_at && analysis.finished_at ? fmtSeconds((Date.parse(analysis.finished_at) - Date.parse(analysis.started_at)) / 1000, 1) : '—'} testId="appendix-timings">
          <Table head={['Stage', 'Status', 'Duration']} rows={stageRows} />
        </Disclosure>
        {legacyMetadata && Object.keys(legacyMetadata).length > 0 && (
          <Disclosure label="Metadata fields as recorded" testId="appendix-flat-metadata">
            <pre className="m-0 max-h-72 overflow-auto bg-surface-2 border border-border rounded-sm p-3 font-mono text-caption leading-[18px] text-text max-w-[80ch]">{JSON.stringify(legacyMetadata, null, 1)}</pre>
          </Disclosure>
        )}
      </div>
      <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
        {!hasApiKey || token ? <ButtonLink variant="text" href={api.exportUrl(analysis.id, token)} target="_blank" rel="noreferrer">Raw result JSON ▸</ButtonLink>
          : <Button variant="ghost" disabled title="Secure media links are unavailable">Raw result JSON ▸</Button>}
        <span className="text-small text-text-muted">{engine ? `This server: ${engine}` : ''}</span>
      </div>
    </section>
  )
}
