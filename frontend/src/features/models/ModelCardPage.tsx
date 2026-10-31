import type { ReactNode } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useMe, useModelCards, useModelRecord, type ModelCard } from '../../lib/api/queries'
import { mayReadEngineeringRecord } from '../../lib/api/roles'
import { shortHash } from '../../lib/format'
import { Markdown } from '../../components/Markdown'
import { CodeChip, Disclosure, Dl, EmptyState, ErrorState, Eyebrow, Skeleton, SkeletonRows, type DlRow } from '../../components/ui'
import { CALIBRATION_NONE, DEVICE_STATE_WORD, LICENCE_CLASS_WORD, MODALITY_WORD, NOT_ESTABLISHED, admissionWord, clientOf, enabledWord, fmtMetric, installedWord, loadedWord, measuredRows, modelMemoryWord, modelParents, pins, sliceWord, sourceLine, sourceLink, withheldFailureNotes, type CardCopy, type ModelEntryState } from './catalogue'

const muted = (s: string) => <span className="text-text-muted">{s}</span>
const list = (items: readonly string[] | null | undefined, empty = NOT_ESTABLISHED): ReactNode => ((items ?? []).length ? (items ?? []).join('; ') : muted(empty))
const text = (v: string | null | undefined, empty = NOT_ESTABLISHED): ReactNode => (v ? v : muted(empty))
const mono = (s: string) => <span className="font-mono text-mono break-all">{s}</span>

function Section({ title, children, testId }: { title: string; children: ReactNode; testId?: string }) {
  return (
    <section data-testid={testId} className="border-t border-border pt-6 mt-10">
      <h2 className="m-0 eyebrow">{title}</h2>
      <div className="mt-3">{children}</div>
    </section>
  )
}

function RecordBody({ detectorId, recordId }: { detectorId: string; recordId: string }) {
  const q = useModelRecord(detectorId, recordId)
  if (q.isPending) return <SkeletonRows />
  if (q.error) return <ErrorState onRetry={() => void q.refetch()}>{q.error.message}</ErrorState>
  return <Markdown source={q.data ?? ''} testId="record-text" className="max-w-[100ch]" />
}

/** Where a recorded string is, for the reader: only an administrator's page has the engineering record (D-71). */
const recordPlace = (admin: boolean): string => (admin ? 'the engineering record' : 'the engineering record, which administrators can open')

function ModelEntry({ m, card, client, admin }: { m: ModelEntryState; card: ModelCard; client: CardCopy; admin: boolean }) {
  const parents = modelParents(card, m.id)
  const fp = m.footprint
  const positive = (v: number | null | undefined): v is number => v != null && v > 0  // a zero footprint is a placeholder, not a measurement
  const footprint = fp ? [positive(fp.params_m) ? `${fp.params_m} M params` : null, positive(fp.fp32_mb) ? `${fp.fp32_mb} MB fp32` : null, positive(fp.fp16_mb) ? `${fp.fp16_mb} MB fp16` : null].filter(Boolean) : []
  const notice = client.licence_notices?.[m.id]
  const rows: DlRow[] = [
    { key: 'Source', value: <span className="break-all">{sourceLine(m)}</span> },
    { key: 'Licence', value: <>{[m.licence.spdx, LICENCE_CLASS_WORD[m.licence.class] ?? m.licence.class.replace(/_/g, ' '), notice].filter(Boolean).join(' · ')}{m.licence.notice && !notice ? <span className="text-text-muted"> · notice in {recordPlace(admin)}</span> : null}</> },
    { key: 'Pinned', value: pins(m).length ? (
      <ul className="m-0 p-0 list-none flex flex-col gap-1">
        {pins(m).map((p) => <li key={p.label} className="flex flex-wrap items-center gap-2"><span className="font-mono text-caption text-text-muted break-all">{p.label}</span><CodeChip copy={p.value} title="Copy full hash">{shortHash(p.value, 10, 6)}</CodeChip></li>)}
      </ul>
    ) : muted('no pinned files') },
    { key: 'Footprint', value: footprint.length ? `${footprint.join(' · ')}${fp?.estimated ? ' (estimated)' : ''}` : muted('not recorded') },
    { key: 'Devices', value: Object.keys(m.devices ?? {}).length ? Object.entries(m.devices ?? {}).map(([k, v]) => `${k}: ${DEVICE_STATE_WORD[v] ?? v.replace(/_/g, ' ')}`).join(' · ') : muted('not recorded') },
    { key: 'On disk', value: m.installed ? 'installed' : muted('not installed') },
    { key: 'In memory', value: m.loaded ? modelMemoryWord(card, m) : muted(modelMemoryWord(card, m)) },
    ...(parents.length ? [{ key: 'Required by', value: mono(parents.join(', ')) }] : []),
    ...((m.depends_on ?? []).length ? [{ key: 'Depends on', value: mono((m.depends_on ?? []).join(', ')) }] : []),
  ]
  return (
    <li data-testid="model-entry" className="py-4 border-b border-border">
      <p className="m-0 font-mono text-mono text-text break-all">{m.id}</p>
      <Dl className="mt-2" keyWidth={112} rows={rows} />
    </li>
  )
}

function Card({ card, client, admin }: { card: ModelCard; client: CardCopy; admin: boolean }) {
  const measured = measuredRows(card)
  const slices = card.validated_slices ?? []
  const records = card.records ?? []
  const notes = client.engineering ?? []
  const a = card.admission
  const withheld = withheldFailureNotes(client)
  const place = recordPlace(admin)
  const inRecord = muted(`recorded in ${place}`)
  /** A one-value field: its client wording, the lead of it when the rest cites the build, or where the recorded value is. */
  const one = (field: 'training_data' | 'coverage_date'): ReactNode => {
    const value = client[field]
    if (!value) return card[field] ? inRecord : muted(NOT_ESTABLISHED)
    return (client.truncated ?? []).includes(field) ? <>{value} <span className="text-text-muted">Continues in {place}.</span></> : value
  }
  return (
    <>
      <Section title="Enablement" testId="card-enablement">
        <Dl keyWidth={144} rows={[
          { key: 'Licence policy', value: client.licence_policy },
          { key: 'Decision', value: card.enabled ? 'enabled' : 'disabled' },
          ...((client.enablement_reasons ?? []).length ? [{ key: 'Reasons', value: <ul className="m-0 pl-4 list-disc space-y-1">{(client.enablement_reasons ?? []).map((r) => <li key={r}>{r}</li>)}</ul> }] : []),
          { key: 'On by default', value: card.default_enabled ? 'yes' : 'no' },
          { key: 'Profiles', value: list(card.profiles, 'none') },
          ...(card.superseded_by ? [{ key: 'Superseded by', value: <Link to={`/models/${encodeURIComponent(card.superseded_by)}`} className="font-mono text-mono text-accent underline underline-offset-[3px] break-all">{card.superseded_by}</Link> }] : []),
          ...((client.disclosures ?? []).length ? [{ key: 'Disclosures', value: (client.disclosures ?? []).join(' ') }] : []),
        ]} />
      </Section>

      <Section title="Card" testId="card-fields">
        <Dl keyWidth={144} rows={[
          { key: 'Family', value: client.family },
          { key: 'Tests whether', value: list(client.hypotheses, 'none') },
          { key: 'Built to detect', value: list(client.threats) },
          { key: 'Lane', value: text(client.lane, 'none') },
          { key: 'Training data', value: one('training_data') },
          { key: 'Coverage date', value: one('coverage_date') },
          { key: 'Intended input', value: list(client.input_constraints, card.input_constraints?.length ? `recorded in ${place}` : NOT_ESTABLISHED) },
          { key: 'Contamination sets', value: list(client.contamination_sets, 'none recorded') },
          { key: 'Calibration', value: card.calibration?.method === 'platt' ? 'Platt scaling' : CALIBRATION_NONE },
          { key: 'Hand-set weight', value: card.legacy_weight != null ? mono(fmtMetric(card.legacy_weight, 2)) : muted('none') },
          // N-19d: the client wording of the source. A download address is a link; a check that is part of Veritas says so, and
          // the files of the repository that hold it are in the engineering record
          { key: 'Source', value: !client.source ? muted(NOT_ESTABLISHED) : sourceLink(client)
            ? <a href={client.source} target="_blank" rel="noreferrer" className="text-accent underline underline-offset-[3px] break-all">{client.source}</a>
            : <span data-testid="card-source" className="break-words">{client.source}</span> },
        ]} />
      </Section>

      <Section title="Measured" testId="card-measured">
        {measured.length > 0 ? (
          <div className="overflow-x-auto">
            {/* N-19b (design review DS-13): below 640 px one stacked block for each measurement */}
            <table role="table" data-testid="card-eval" className="stack-sm w-full border-collapse text-small">
              <thead role="rowgroup"><tr role="row" className="border-b border-border-strong">{['Benchmark', 'Slice', 'AUC', 'TPR at 1 % FPR', 'ECE', 'n', 'Date'].map((h) => <th key={h} role="columnheader" scope="col" className="eyebrow text-left font-medium py-1.5 pr-4 whitespace-nowrap">{h}</th>)}</tr></thead>
              <tbody role="rowgroup">
                {measured.map((r, i) => (
                  <tr role="row" key={i} className="border-b border-border">
                    <td role="cell" className="py-1.5 pr-4 whitespace-nowrap max-[639px]:font-medium max-[639px]:whitespace-normal">{r.bench}</td><td role="cell" data-th="slice" className="py-1.5 pr-4 whitespace-nowrap max-[639px]:whitespace-normal">{sliceWord(r.slice)}</td>
                    <td role="cell" data-th="AUC" className="py-1.5 pr-4 font-mono text-mono num">{fmtMetric(r.auc)}</td><td role="cell" data-th="TPR at 1 % FPR" className="py-1.5 pr-4 font-mono text-mono num">{fmtMetric(r.tpr_at_fpr01)}</td>
                    <td role="cell" data-th="ECE" className="py-1.5 pr-4 font-mono text-mono num">{fmtMetric(r.ece)}</td><td role="cell" data-th="n" className="py-1.5 pr-4 font-mono text-mono num">{r.n ?? '—'}</td><td role="cell" data-th="date" className="py-1.5 pr-4 whitespace-nowrap">{r.date ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p data-testid="card-eval-empty" className="m-0 text-small text-text max-w-[80ch]">
            Benchmark results: {NOT_ESTABLISHED}. The benchmark report writes them to the card, and it has not run for this detector.
          </p>
        )}
        {slices.length > 0 && (
          <>
            <h3 className="mt-6 m-0 eyebrow">Validated slices</h3>
            <ul className="mt-2 m-0 p-0 list-none">{slices.map((s, i) => <li key={i} className="py-1 border-b border-border text-small text-text break-words">{Object.entries(s).map(([k, v]) => `${k.replace(/_/g, ' ')} ${String(v)}`).join(' · ')}</li>)}</ul>
          </>
        )}
      </Section>

      <Section title="Admission" testId="card-admission">
        <Dl keyWidth={144} rows={[
          { key: 'Decision', value: a ? admissionWord(card) : muted('not recorded') },
          { key: 'Evidence', value: a?.evidence ? inRecord : muted('not recorded') },
          { key: 'Record', value: card.admission_record ? (admin ? 'on file, under the engineering record' : 'on file; administrators can open it on this page') : muted('not written: no admission record is on file for this detector yet') },
        ]} />
      </Section>

      <Section title="Weights and licences" testId="card-weights">
        {(card.model_entries ?? []).length > 0
          ? <ul className="m-0 p-0 list-none [&>li:first-child]:pt-0 [&>li:last-child]:border-b-0">{(card.model_entries ?? []).map((m) => <ModelEntry key={m.id} m={m} card={card} client={client} admin={admin} />)}</ul>
          : <p className="m-0 text-small text-text-muted">Model-free: this detector pins no weights.</p>}
      </Section>

      <Section title="Known failure modes" testId="card-failure-modes">
        {(client.failure_modes ?? []).length > 0
          ? <ul className="m-0 pl-4 list-disc text-small text-text space-y-1 max-w-[80ch]">{(client.failure_modes ?? []).map((m) => <li key={m}>{m}</li>)}</ul>
          : <p className="m-0 text-small text-text-muted">{withheld > 0 ? 'None in plain wording.' : 'None recorded.'}</p>}
        {withheld > 0 && (
          <p data-testid="card-failure-withheld" className="mt-3 m-0 text-small text-text-muted max-w-[80ch]">
            {withheld} further {withheld === 1 ? 'note is' : 'notes are'} written in the shorthand of the build and cited to its own records; {withheld === 1 ? 'it is' : 'they are'} in {admin ? 'the engineering record below, word for word' : place}.
          </p>
        )}
      </Section>

      {admin && (
        <Section title="Engineering record" testId="card-engineering">
          <p className="m-0 text-small text-text-muted max-w-[80ch]">
            What the build recorded about this detector, word for word, and its admission record. These use the ticket numbers, file paths and shorthand of the build, and are shown to administrators only.
          </p>
          <div className="mt-2 flex flex-col gap-1">
            {notes.length > 0 && (
              <Disclosure testId="card-notes" label="Build notes" summary={`${notes.length} ${notes.length === 1 ? 'entry' : 'entries'}`}>
                <ul className="m-0 p-0 list-none max-w-[100ch]">
                  {notes.map((n, i) => (
                    <li key={i} className="py-2 border-b border-border last:border-b-0">
                      <p className="m-0 eyebrow">{n.field}</p>
                      <p className="mt-1 m-0 font-mono text-caption leading-[18px] text-text break-words">{n.text}</p>
                    </li>
                  ))}
                </ul>
              </Disclosure>
            )}
            {records.map((r) => (
              <Disclosure key={r.id} testId="card-record" label={client.record_titles?.[r.id] ?? `Admission record: ${card.name}`}>
                <p data-testid="record-path" className="m-0 mb-3 font-mono text-caption text-text-muted break-all">{r.path}</p>
                <RecordBody detectorId={card.id} recordId={r.id} />
              </Disclosure>
            ))}
          </div>
          {notes.length === 0 && records.length === 0 && <p className="mt-2 m-0 text-small text-text-muted">Nothing further is recorded for this detector.</p>}
        </Section>
      )}
    </>
  )
}

/** Model card page (ARCHITECTURE §7.2 `/models/:detectorId`, VG-01, E-7): card fields, the licence class and pins of every model,
 *  installed and loaded state, measured benchmark rows and, for an administrator, the engineering record: the build's notes and
 *  the admission record, shown as a document (D-71). */
export default function ModelCardPage() {
  const { detectorId = '' } = useParams()
  const cards = useModelCards()
  const admin = mayReadEngineeringRecord(useMe().data)
  const card = cards.data?.find((c) => c.id === detectorId)
  return (
    <article>
      <Link to="/models" className="text-small text-text-muted hover:text-text inline-flex items-center min-h-8">‹ Models</Link>
      {cards.isPending && <div aria-busy="true" className="mt-6"><Skeleton w={160} h={12} /><div className="mt-3"><Skeleton w={420} h={32} /></div><SkeletonRows className="mt-8" /></div>}
      {cards.error && <ErrorState className="mt-6" onRetry={() => void cards.refetch()}>{cards.error.message}</ErrorState>}
      {cards.isSuccess && !card && (
        <EmptyState className="mt-6">No detector with the id <span className="font-mono text-mono">{detectorId}</span> is in the models manifest.</EmptyState>
      )}
      {card && (
        <>
          <Eyebrow className="mt-6">{MODALITY_WORD[card.modality] ?? card.modality} · {clientOf(card).family} · Model card</Eyebrow>
          <h1 className="display text-display font-normal text-text mt-1 break-words">{card.name}</h1>
          <p className="mt-1 m-0 font-mono text-mono text-text-muted break-all">{card.id} {card.version}</p>
          <div data-testid="card-state" className="mt-4">
            <Dl keyWidth={96} rows={[
              { key: 'Policy', value: enabledWord(card) },
              { key: 'Weights', value: installedWord(card) },
              { key: 'Memory', value: loadedWord(card) },
              { key: 'Admission', value: admissionWord(card).split(' · ')[0] },
            ]} />
          </div>
          {clientOf(card).citation && <p data-testid="card-citation" className="mt-4 m-0 text-body text-text max-w-[80ch] break-words">{clientOf(card).citation}</p>}
          <Card card={card} client={clientOf(card)} admin={admin} />
        </>
      )}
    </article>
  )
}
