import { Fragment } from 'react'
import { Link } from 'react-router-dom'
import type { AnalysisDetail } from '../../lib/api/queries'
import { MODE_WORD, deltaField, deltaUnchanged, deltaValue, deltaVerdict, useDeltas, type DeltaRow } from '../../lib/api/rerun'
import { Disclosure, Dl, ErrorState, SkeletonRows } from '../../components/ui'

const th = 'py-1.5 pr-4 font-normal'
const td = 'py-2 pr-4'

/** Long signal and manifest ids wrap after `.`, `:` and `_`, not in the middle of a word. */
const breakable = (id: string) => id.split(/(?<=[.:_])/).map((part, i) => <Fragment key={i}>{i > 0 && <wbr />}{part}</Fragment>)

/** The value table of a set of rows; it scrolls sideways inside its own box on a narrow screen (L-7). */
function DeltaTable({ rows, testId }: { rows: DeltaRow[]; testId: string }) {
  return (
    <div className="overflow-x-auto">
      <table data-testid={testId} className="w-full min-w-[560px] border-collapse text-small">
        <caption className="sr-only">Values of this revision against the revision it supersedes</caption>
        <thead>
          <tr className="border-b border-border text-left text-text-muted">
            <th scope="col" className={th}>Item</th>
            <th scope="col" className={th}>Field</th>
            <th scope="col" className={th}>Before</th>
            <th scope="col" className={th}>After</th>
            <th scope="col" className={`${th} text-right`}>Change</th>
            <th scope="col" className="py-1.5 font-normal whitespace-nowrap">Within tolerance</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.item}.${r.field}`} data-testid="delta-row" data-ok={String(r.ok)} className="border-b border-border align-baseline">
              <td className={`${td} font-mono text-mono text-text`}>{breakable(r.item)}</td>
              <td className={`${td} text-text`}>{deltaField(r.field)}</td>
              <td className={`${td} num text-text-muted break-words`}>{deltaValue(r.before, r.field)}</td>
              <td className={`${td} num text-text break-words`}>{deltaValue(r.after, r.field)}</td>
              <td className={`${td} num text-right text-text-muted`}>{r.delta == null ? '' : r.delta.toFixed(6)}</td>
              <td className="py-2 text-text">{r.ok ? 'yes' : <span className="text-danger">no</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** Revision deltas (PRD CC-03; ARCHITECTURE §5.2, §6.5): this revision against the one it supersedes. Per signal the status, raw
 *  score and calibrated probability, then the decision code and probability and, for video, the frame indices, each with the change
 *  and whether it is within the tolerance; then the manifest fields that changed. */
export default function RevisionDeltas({ analysis }: { analysis: AnalysisDetail }) {
  const q = useDeltas(analysis.id, analysis.supersedes_id != null)
  const d = q.data
  const changed = (d?.deltas ?? []).filter((r) => !deltaUnchanged(r))
  const unchanged = (d?.deltas ?? []).filter(deltaUnchanged)
  return (
    <section id="section-revision" data-testid="revision-deltas" data-status={d?.status} data-identical={d?.identical == null ? undefined : String(d.identical)}
      aria-labelledby="revision-heading" className="scroll-mt-14">
      <h2 id="revision-heading" className="m-0 eyebrow">Revision {analysis.revision}</h2>
      {q.isPending && <SkeletonRows className="mt-4" />}
      {q.error && <ErrorState className="mt-4" onRetry={() => void q.refetch()}>{q.error.message}</ErrorState>}
      {d && (
        <>
          <p data-testid="delta-verdict" aria-live="polite" className="mt-3 mb-0 text-body text-text max-w-[70ch]">{deltaVerdict(d)}</p>
          <Dl className="mt-4" keyWidth={140} rows={[
            { key: 'Compared with', value: <Link to={`/analyses/${d.compared_with}`} className="font-mono text-mono text-accent underline underline-offset-[3px]">{d.compared_with.slice(0, 8)}</Link> },
            { key: 'How made', value: d.mode ? MODE_WORD[d.mode] : 'not recorded' },
            { key: 'Tolerance', value: <span className="num">{d.tolerance}{d.cross_device ? ' · the revisions ran on different devices' : ' · same devices'}</span> },
            { key: 'Largest change', value: <span className="num">{d.max_delta == null ? 'none measured' : d.max_delta.toFixed(6)}</span> },
          ]} />
          {d.status === 'ready' && (
            <>
              {changed.length > 0
                ? <div className="mt-6"><DeltaTable rows={changed} testId="delta-table" /></div>
                : <p data-testid="delta-none" className="mt-6 mb-0 text-small text-text">No value changed.</p>}
              {unchanged.length > 0 && (
                <Disclosure className="mt-2" testId="delta-unchanged" label={`${unchanged.length} ${unchanged.length === 1 ? 'value' : 'values'} unchanged`}>
                  <DeltaTable rows={unchanged} testId="delta-table-unchanged" />
                </Disclosure>
              )}
              <h3 className="mt-8 mb-0 text-small font-medium text-text">Manifest changes</h3>
              {d.manifest_changes.length === 0
                ? <p data-testid="manifest-changes" className="mt-2 mb-0 text-small text-text-muted">None of the recorded manifest fields changed: versions, commit, configuration, profile, devices, libraries, calibration, policy thresholds, templates, sampling and detector weights.</p>
                : (
                  <div className="overflow-x-auto"><table data-testid="manifest-changes" className="mt-2 w-full min-w-[480px] border-collapse text-small">
                    <thead>
                      <tr className="border-b border-border text-left text-text-muted">
                        <th scope="col" className={th}>Field</th>
                        <th scope="col" className={th}>Before</th>
                        <th scope="col" className="py-1.5 font-normal">After</th>
                      </tr>
                    </thead>
                    <tbody>
                      {d.manifest_changes.map((m) => (
                        <tr key={m.field} className="border-b border-border align-baseline">
                          <td className={`${td} font-mono text-mono text-text`}>{breakable(m.field)}</td>
                          <td className={`${td} num text-text-muted break-words`}>{deltaValue(m.before)}</td>
                          <td className="py-2 num text-text break-words">{deltaValue(m.after)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table></div>
                )}
            </>
          )}
        </>
      )}
    </section>
  )
}
