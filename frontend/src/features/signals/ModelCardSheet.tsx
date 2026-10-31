import { useEffect, useRef, type KeyboardEvent } from 'react'
import { Link } from 'react-router-dom'
import { X } from 'lucide-react'
import { useModelCards } from '../../lib/api/queries'
import { shortHash } from '../../lib/format'
import { Button, Dl, EmptyState, ErrorState, Eyebrow, SkeletonRows } from '../../components/ui'
import { CALIBRATION_NONE, LICENCE_CLASS_WORD, admissionWord, clientOf, modelParents, sliceWord, sourceLink, withheldFailureNotes } from '../models/catalogue'

const FOCUSABLE = 'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])'

/** Model card side sheet (UX §5.7): 480 px, hairline left, popover shadow. Focus moves into the sheet, Tab stays inside it, Esc
 *  closes it and focus returns to the trigger. The card is read from the models catalogue; a detector absent from the manifest
 *  says so instead of showing an empty card. */
export default function ModelCardSheet({ detectorId, onClose, returnFocus }: { detectorId: string | null; onClose: () => void; returnFocus: HTMLElement | null }) {
  const cards = useModelCards(detectorId != null)
  const sheet = useRef<HTMLDivElement>(null)
  const closeBtn = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (detectorId == null) return
    closeBtn.current?.focus()
    return () => { returnFocus?.focus() }
  }, [detectorId, returnFocus])

  if (detectorId == null) return null
  const card = cards.data?.find((c) => c.id === detectorId)
  const client = card ? clientOf(card) : null
  const withheld = client ? withheldFailureNotes(client) : 0

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Escape') { e.stopPropagation(); onClose(); return }
    if (e.key !== 'Tab' || !sheet.current) return
    const items = Array.from(sheet.current.querySelectorAll<HTMLElement>(FOCUSABLE))
    if (items.length === 0) return
    const first = items[0]
    const last = items[items.length - 1]
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
  }

  return (
    <div ref={sheet} role="dialog" aria-modal="true" aria-labelledby="model-card-title" data-testid="model-card-sheet" onKeyDown={onKeyDown}
      className="fixed inset-y-0 right-0 z-40 w-[480px] max-w-full overflow-y-auto bg-surface border-l border-border shadow-popover animate-sheet-in">
      <div className="sticky top-0 bg-surface border-b border-border px-6 h-14 flex items-center justify-between gap-4">
        <Eyebrow as="span">Model card</Eyebrow>
        <Button ref={closeBtn} variant="icon" aria-label="Close model card" onClick={onClose}><X size={16} strokeWidth={1.5} aria-hidden /></Button>
      </div>
      <div className="px-6 py-6">
        <h2 id="model-card-title" className="m-0 display text-h2 font-normal text-text break-words">{card?.name ?? detectorId}</h2>
        <p className="mt-1 m-0 font-mono text-caption text-text-muted break-all">{detectorId}{card ? ` ${card.version}` : ''}</p>
        {cards.isPending && <SkeletonRows className="mt-6" />}
        {cards.error && <ErrorState className="mt-6" onRetry={() => void cards.refetch()}>{cards.error.message}</ErrorState>}
        {cards.isSuccess && !card && <EmptyState className="mt-6">No model card on record for this detector: it is not in the models manifest.</EmptyState>}
        {card && client && (
          <>
            {client.citation && <p className="mt-4 m-0 text-body text-text break-words">{client.citation}</p>}
            <Dl className="mt-5" keyWidth={128} rows={[
              { key: 'Family', value: client.family },
              { key: 'Tests whether', value: (client.hypotheses ?? []).join('; ') || '—' },
              { key: 'Intended input', value: (client.input_constraints ?? []).join('; ') || '—' },
              { key: 'Training data', value: client.training_data ?? (card.training_data ? 'on the full model card' : 'not established') },
              { key: 'Coverage date', value: client.coverage_date ?? (card.coverage_date ? 'on the full model card' : 'not established') },
              { key: 'Calibration', value: card.calibration?.method === 'platt' ? 'Platt scaling' : CALIBRATION_NONE },
              { key: 'Enabled', value: card.enabled ? `yes · ${client.licence_policy}` : `no · ${(client.enablement_reasons ?? []).join('; ') || client.licence_policy}` },
              { key: 'Admission', value: admissionWord(card) },
            ]} />

            <h3 className="mt-8 m-0 eyebrow">Evaluation</h3>
            {(card.eval ?? []).length > 0 ? (
              <table data-testid="model-card-eval" className="mt-2 w-full border-collapse text-small">
                <thead>
                  <tr className="border-b border-border-strong">
                    {['Benchmark', 'Slice', 'AUC', 'ECE', 'n', 'Date'].map((h) => <th key={h} scope="col" className="eyebrow text-left font-medium py-1.5 pr-3">{h}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {(card.eval ?? []).map((r, i) => (
                    <tr key={i} className="border-b border-border">
                      <td className="py-1.5 pr-3">{r.bench}</td><td className="py-1.5 pr-3">{sliceWord(r.slice)}</td>
                      <td className="py-1.5 pr-3 font-mono text-mono num">{r.auc?.toFixed(3) ?? '—'}</td><td className="py-1.5 pr-3 font-mono text-mono num">{r.ece?.toFixed(3) ?? '—'}</td>
                      <td className="py-1.5 pr-3 font-mono text-mono num">{r.n ?? '—'}</td><td className="py-1.5 pr-3">{r.date ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <p className="mt-2 m-0 text-small text-text-muted">No benchmark results are recorded on this card yet.</p>}

            <h3 className="mt-8 m-0 eyebrow">Known failure modes</h3>
            {(client.failure_modes ?? []).length > 0
              ? <ul className="mt-2 m-0 pl-4 list-disc text-small text-text space-y-1">{(client.failure_modes ?? []).map((m) => <li key={m}>{m}</li>)}</ul>
              : <p className="mt-2 m-0 text-small text-text-muted">{withheld > 0 ? 'None in plain wording.' : 'None recorded.'}</p>}
            {withheld > 0 && <p className="mt-2 m-0 text-small text-text-muted">{withheld} further {withheld === 1 ? 'note is' : 'notes are'} in the engineering record on the full model card.</p>}

            <h3 className="mt-8 m-0 eyebrow">Weights and licences</h3>
            {(card.model_entries ?? []).length > 0 ? (
              <ul className="mt-2 m-0 p-0 list-none border-t border-border">
                {(card.model_entries ?? []).map((m) => {
                  const parents = modelParents(card, m.id)
                  const pin = m.source.revision ?? m.source.ref ?? m.source.files?.find((f) => f.sha256)?.sha256 ?? null
                  return (
                    <li key={m.id} className="py-2 border-b border-border text-small">
                      <p className="m-0 font-mono text-mono text-text break-all">{m.id}</p>
                      <p className="m-0 text-text-muted break-words">{m.licence.spdx} · {LICENCE_CLASS_WORD[m.licence.class] ?? m.licence.class.replace(/_/g, ' ')}{pin ? ` · pinned ${shortHash(pin)}` : ''}{m.source.repo ? ` · ${m.source.repo}` : ''}</p>
                      {parents.length > 0 && <p className="mt-1 m-0 text-text-muted break-all">Required by {parents.join(', ')}</p>}
                      {client.licence_notices?.[m.id] && <p className="mt-1 m-0 text-text break-words">{client.licence_notices[m.id]}</p>}
                    </li>
                  )
                })}
              </ul>
            ) : <p className="mt-2 m-0 text-small text-text-muted">Model-free: no weights.</p>}
            {(client.disclosures ?? []).length > 0 && <p className="mt-4 m-0 text-small text-text">{(client.disclosures ?? []).join(' ')}</p>}
            <p className="mt-6 m-0 flex flex-wrap gap-x-6 text-small">
              <Link to={`/models/${encodeURIComponent(card.id)}`} data-testid="model-card-full" className="text-accent underline underline-offset-[3px]">Full model card ▸</Link>
              {sourceLink(client) && <a href={sourceLink(client)!} target="_blank" rel="noreferrer" className="text-accent underline underline-offset-[3px]">Source</a>}
            </p>
          </>
        )}
      </div>
    </div>
  )
}
