import { useId } from 'react'
import { Disclosure } from '../../components/ui'
import type { AnalysisDetail } from '../../lib/api/queries'

type Calibration = NonNullable<AnalysisDetail['calibration']>[number]
export type Diagram = NonNullable<Calibration['reliability_diagram']>
const percent = (value: number) => `${(value * 100).toFixed(1)} %`
const finiteUnit = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n) && n >= 0 && n <= 1
const digest = (value: unknown) => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value)
const positiveInteger = (value: unknown): value is number => typeof value === 'number' && Number.isSafeInteger(value) && value > 0

/** A historical diagram is shown only for its producing fuser, with internally consistent recorded bins. */
export function recordedDiagram(analysis: AnalysisDetail): Diagram | null {
  const decision = analysis.decision
  if (decision?.fusion_mode !== 'artefact' || !decision.fuser_sha256) return null
  const matching = (analysis.calibration ?? []).filter((c) => typeof c?.subject === 'string' && c.subject.startsWith(`fuser-${analysis.kind}-`)
    && c.artefact_sha256 === decision.fuser_sha256 && c.reliability_diagram != null)
  if (matching.length !== 1) return null
  const record = matching[0]
  const diagram = record.reliability_diagram!
  if (diagram.version !== 'veritas.reliability-diagram/1'
    || diagram.published_artefact_sha256 !== record.artefact_sha256
    || diagram.population !== `${analysis.kind}/test`
    || diagram.bench_version !== record.bench_version || diagram.manifest_sha256 !== record.bench_manifest_sha256
    || diagram.weighting !== 'class_balanced' || diagram.binning !== 'equal_mass'
    || diagram.score_semantics !== 'unclamped_fuser' || !finiteUnit(diagram.ece)
    || diagram.publication_binding !== 'documented_metrics_publication'
    || ![diagram.report_sha256, diagram.manifest_sha256, diagram.scope_sha256, diagram.evaluated_artefact_sha256,
      diagram.published_artefact_sha256, diagram.publication_record_sha256].every(digest)
    || typeof diagram.report_generated_at !== 'string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/.test(diagram.report_generated_at)
    || !Number.isFinite(Date.parse(diagram.report_generated_at))
    || typeof diagram.bench_version !== 'string' || diagram.bench_version.length === 0
    || typeof diagram.limitation !== 'string' || diagram.limitation.length === 0
    || (analysis.kind === 'image' ? diagram.view !== 'size_matched' || diagram.variant !== 'norm'
      : diagram.view !== 'full' || diagram.variant !== 'native')
    || (!finiteUnit(record.ece) || Math.abs(record.ece - diagram.ece) > 1e-12)
    || ((record.test_view === 'size_matched') !== (diagram.view === 'size_matched'))
    || diagram.ece_interval_method !== 'grouped_bootstrap_percentile'
    || !Array.isArray(diagram.ece_ci95) || diagram.ece_ci95.length !== 2 || !diagram.ece_ci95.every(finiteUnit)
    || diagram.ece_ci95[0] > diagram.ece_ci95[1]
    || !positiveInteger(diagram.ece_bootstrap_replicates) || !positiveInteger(diagram.ece_bootstrap_valid)
    || diagram.ece_bootstrap_valid > diagram.ece_bootstrap_replicates
    || !Array.isArray(diagram.bins) || diagram.bins.length !== 15) return null
  const bins = diagram.bins
  if (bins.some((b, i) => b == null || typeof b !== 'object' || b.bin !== i || !positiveInteger(b.n)
    || ![b.mean_s, b.mean_y, b.s_min, b.s_max, b.weight].every(finiteUnit)
    || b.weight <= 0 || b.s_min > b.mean_s || b.mean_s > b.s_max
    || (i > 0 && bins[i - 1].s_max > b.s_min))) return null
  // Evidence's canonical serializer rounds each float to 6dp. Bound only that accumulated error;
  // authoritative source extraction additionally checks its exact byte pin and 1e-12 aggregates.
  const halfUnit = 0.5e-6
  const weightError = bins.length * halfUnit
  const eceError = weightError + 2 * halfUnit * (1 + weightError) + halfUnit
  if (!positiveInteger(diagram.n_real) || !positiveInteger(diagram.n_fake)
    || bins.reduce((n, b) => n + b.n, 0) !== diagram.n_real + diagram.n_fake
    || Math.abs(bins.reduce((w, b) => w + b.weight, 0) - 1) > weightError + 1e-12
    || Math.abs(bins.reduce((e, b) => e + b.weight * Math.abs(b.mean_s - b.mean_y), 0) - diagram.ece) > eceError + 1e-12) return null
  return diagram
}

const x = (value: number) => 64 + 260 * value
const y = (value: number) => 16 + 260 * (1 - value)
const ticks = [0, 0.25, 0.5, 0.75, 1]

export default function CalibrationDiagram({ analysis }: { analysis: AnalysisDetail }) {
  const id = useId()
  const diagram = recordedDiagram(analysis)
  if (!diagram) return null
  const ci = diagram.ece_ci95
  const maxSamples = Math.max(...diagram.bins.map((bin) => bin.n))
  const groupStep = 260 / diagram.bins.length
  return <section data-testid="methods-calibration" aria-labelledby={`${id}-heading`} className="mt-8 border-t border-border pt-6">
    <h3 id={`${id}-heading`} className="m-0 eyebrow">Historical benchmark calibration</h3>
    <p className="mt-2 mb-0 text-small text-text-muted max-w-[85ch]">
      This chart compares recorded model scores with observed outcomes in the benchmark. It describes scores before decision caps; it does not establish whether this file is synthetic or how later generators perform.
    </p>
    <div className="mt-4 grid grid-cols-[minmax(0,380px)_minmax(0,1fr)] items-center gap-x-8 gap-y-4 max-[699px]:grid-cols-1">
      <div className="min-w-0">
        <figure className="m-0">
          <svg data-testid="calibration-chart" viewBox="0 0 360 330" role="img" aria-labelledby={`${id}-title ${id}-description`} className="block w-full max-w-[380px] text-text">
            <title id={`${id}-title`}>Historical calibration: predicted score versus observed AI or manipulated share</title>
            <desc id={`${id}-description`}>{diagram.bins.length} recorded, class-balanced score groups. Squares on the dashed diagonal would be perfectly calibrated in this benchmark. Recorded values and sample counts are available in Recorded bins and source.</desc>
            {ticks.map((t) => <g key={t}>
              <line x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} stroke="var(--color-border)" />
              <text x={x(0) - 10} y={y(t) + 4} textAnchor="end" fill="var(--color-text-muted)" fontSize={12}>{t * 100} %</text>
              <line x1={x(t)} x2={x(t)} y1={y(0)} y2={y(0) + 5} stroke="var(--color-graphic)" />
              <text x={x(t)} y={y(0) + 20} textAnchor="middle" fill="var(--color-text-muted)" fontSize={12}>{t * 100} %</text>
            </g>)}
            <path d={`M ${x(0)} ${y(1)} V ${y(0)} H ${x(1)}`} fill="none" stroke="var(--color-graphic)" />
            <line x1={x(0)} x2={x(1)} y1={y(0)} y2={y(1)} stroke="var(--color-graphic)" strokeDasharray="5 4" />
            {diagram.bins.map((bin) => <rect key={bin.bin} data-bin={bin.bin} x={x(bin.mean_s) - 3} y={y(bin.mean_y) - 3} width={6} height={6} fill="currentColor">
              <title>Group {bin.bin + 1}: mean score {percent(bin.mean_s)}, observed share {percent(bin.mean_y)}, {bin.n} samples</title>
            </rect>)}
            <text x={194} y={320} textAnchor="middle" fill="currentColor" fontSize={12}>Mean predicted score</text>
            <text x={17} y={146} textAnchor="middle" transform="rotate(-90 17 146)" fill="currentColor" fontSize={12}>AI/manipulated share (weighted)</text>
          </svg>
          <figcaption className="mt-1 text-caption text-text-muted">Squares: recorded groups. Dashed line: perfect calibration. Per-group confidence intervals were not recorded.</figcaption>
        </figure>
        <figure className="mt-4 mb-0 mx-0">
          <figcaption className="mb-1 text-small font-medium">Recorded samples by score group</figcaption>
          <svg data-testid="calibration-support" viewBox="0 0 360 128" role="img" aria-labelledby={`${id}-support-title`} aria-describedby={`${id}-support-description`} className="block w-full max-w-[380px] text-text">
            <title id={`${id}-support-title`}>Recorded samples by score group</title>
            <desc id={`${id}-support-description`}>Unweighted sample counts for 15 ordinal score groups. Exact counts are available in the Samples column of Recorded bins and source.</desc>
            <text x={54} y={24} textAnchor="end" fill="var(--color-text-muted)" fontSize={11}>{maxSamples}</text>
            <text x={54} y={80} textAnchor="end" fill="var(--color-text-muted)" fontSize={11}>0</text>
            <text x={17} y={48} textAnchor="middle" transform="rotate(-90 17 48)" fill="currentColor" fontSize={11}>Samples</text>
            <path d="M64 20 V76 H324" fill="none" stroke="var(--color-graphic)" />
            {diagram.bins.map((bin) => {
              const height = 56 * bin.n / maxSamples
              const center = 64 + (bin.bin + 0.5) * groupStep
              return <g key={bin.bin}>
                <rect data-group={bin.bin + 1} data-count={bin.n} x={center - groupStep * 0.35} y={76 - height} width={groupStep * 0.7} height={height} fill="currentColor">
                  <title>Group {bin.bin + 1}: {bin.n} samples</title>
                </rect>
                <text x={center} y={94} textAnchor="middle" fill="var(--color-text-muted)" fontSize={11}>{bin.bin + 1}</text>
              </g>
            })}
            <text x={194} y={116} textAnchor="middle" fill="currentColor" fontSize={12}>Score group (ordered)</text>
          </svg>
          <p className="mt-1 mb-0 text-caption text-text-muted">Bars show unweighted counts. Horizontal spacing represents group order, not score intervals.</p>
        </figure>
      </div>
      <div className="min-w-0 text-small">
        <p className="m-0 eyebrow text-text-muted">Expected calibration error</p>
        <p className="my-1 font-mono text-h1 num">{diagram.ece.toFixed(3)}</p>
        <p className="m-0 text-caption text-text-muted">95 % grouped-bootstrap interval: {ci[0].toFixed(3)}–{ci[1].toFixed(3)}. This interval describes the overall error, not individual groups.</p>
        <dl className="mt-4 mb-0 grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-2">
          <dt className="text-text-muted">Benchmark</dt><dd className="m-0 break-words">{diagram.bench_version} · {diagram.population}</dd>
          <dt className="text-text-muted">Recorded</dt><dd className="m-0">{diagram.report_generated_at.slice(0, 10)}</dd>
          <dt className="text-text-muted">Population</dt><dd className="m-0">{diagram.n_real.toLocaleString('en-US')} genuine · {diagram.n_fake.toLocaleString('en-US')} generated or manipulated</dd>
          <dt className="text-text-muted">View</dt><dd className="m-0">{diagram.view === 'size_matched' ? 'Size-matched test set' : 'Full test set'}</dd>
          <dt className="text-text-muted">Grouping</dt><dd className="m-0">{diagram.bins.length} groups with approximately equal weighted mass; the two classes have equal total weight.</dd>
        </dl>
        <p className="mt-4 mb-0 text-caption text-text-muted">Sample counts are unweighted. The observed share and calibration error use class-balanced weights. These historical results do not establish independent validation for this input.</p>
      </div>
    </div>
    <Disclosure className="mt-4" testId="calibration-record" label="Recorded bins and source">
      <div className="overflow-x-auto">
        <table data-testid="calibration-bins" className="w-full border-collapse text-small">
          <caption className="pb-2 text-left text-caption text-text-muted">Recorded score groups, in ascending score order. Shares and weights are percentages rounded to one decimal place.</caption>
          <thead><tr className="border-b border-border-strong">{['Group', 'Mean score', 'Observed share', 'Samples', 'Weight'].map((label) => <th key={label} scope="col" className="py-1.5 pr-4 text-left font-medium whitespace-nowrap">{label}</th>)}</tr></thead>
          <tbody>{diagram.bins.map((bin) => <tr key={bin.bin} className="border-b border-border">
            <th scope="row" className="py-1.5 pr-4 text-left font-normal">{bin.bin + 1}</th>
            <td className="py-1.5 pr-4 font-mono num whitespace-nowrap">{percent(bin.mean_s)}</td>
            <td className="py-1.5 pr-4 font-mono num whitespace-nowrap">{percent(bin.mean_y)}</td>
            <td className="py-1.5 pr-4 font-mono num">{bin.n}</td>
            <td className="py-1.5 pr-4 font-mono num whitespace-nowrap">{percent(bin.weight)}</td>
          </tr>)}</tbody>
        </table>
      </div>
      <dl className="mt-4 mb-0 text-caption text-text-muted space-y-2">
        <div><dt>Preprocessing</dt><dd className="m-0">{diagram.variant === 'norm' ? 'Normalized image inputs' : 'Native audio or video inputs'}</dd></div>
        <div><dt>ECE interval resamples</dt><dd className="m-0">{diagram.ece_bootstrap_valid.toLocaleString('en-US')} valid of {diagram.ece_bootstrap_replicates.toLocaleString('en-US')} requested grouped-bootstrap replicates</dd></div>
        <div><dt>Report SHA-256</dt><dd className="m-0 font-mono break-all">{diagram.report_sha256}</dd></div>
        <div><dt>Benchmark manifest SHA-256</dt><dd className="m-0 font-mono break-all">{diagram.manifest_sha256}</dd></div>
        <div><dt>Evaluation scope SHA-256</dt><dd className="m-0 font-mono break-all">{diagram.scope_sha256}</dd></div>
        <div><dt>Publication record SHA-256</dt><dd className="m-0 font-mono break-all">{diagram.publication_record_sha256}</dd></div>
        <div><dt>Evaluated model SHA-256</dt><dd className="m-0 font-mono break-all">{diagram.evaluated_artefact_sha256}</dd></div>
        <div><dt>Published model SHA-256</dt><dd className="m-0 font-mono break-all">{diagram.published_artefact_sha256}</dd></div>
      </dl>
      <p className="mt-3 mb-0 text-caption text-text-muted">{diagram.limitation}</p>
    </Disclosure>
  </section>
}
