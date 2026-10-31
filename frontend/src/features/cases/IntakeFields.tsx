import { useId } from 'react'
import { DateTimeField } from '../../components/ui'
import { cx } from '../../lib/cx'
import { inputClass } from './NewCaseDialog'
import { ACQUISITION_OPTIONS, draftProblems, localToUtc, type CustodyDraft } from './intakeDraft'

/** Acquisition fields of an intake (UX §5.1; PRD CC-01): all optional, never blocking submission. Shown in the report's custody
 *  section and recorded in the intake custody event. */
export default function IntakeFields({ draft, onChange }: { draft: CustodyDraft; onChange: (d: CustodyDraft) => void }) {
  const uid = useId()
  const set = <K extends keyof CustodyDraft>(k: K, v: CustodyDraft[K]) => onChange({ ...draft, [k]: v })
  const label = 'text-small text-text-muted pt-2 max-[639px]:pt-0'
  // N-19a: what the form refuses is marked on its field. N-19k (design re-review D2-09): each field has one message of its
  // own under it, all fields at once (one message stood at the foot of the block, the first problem only). A date that
  // cannot be read has the field's own sentence; the message here is for a time that is later than now
  const problems = draftProblems(draft)
  const badSource = problems.source !== undefined
  const badHash = problems.statedHash !== undefined
  const lateReceived = problems.received !== undefined && localToUtc(draft.received) !== undefined
  const problemClass = 'mt-1 mb-0 text-caption text-danger max-w-[60ch]'
  return (
    <div className="grid grid-cols-[140px_minmax(0,1fr)] gap-x-6 gap-y-3 items-start max-[639px]:grid-cols-1 max-[639px]:gap-y-1" data-testid="intake-fields">
      <span id={`${uid}-acq`} className={label}>Acquired by</span>
      <div>
        <div role="radiogroup" aria-labelledby={`${uid}-acq`} className="grid grid-cols-3 gap-x-5 max-w-[560px] text-small text-text max-[959px]:grid-cols-2 max-[639px]:grid-cols-1">
          {ACQUISITION_OPTIONS.map((o) => (
            <label key={o.id} className="inline-flex items-center gap-2 min-h-8">
              <input type="radio" name={`${uid}-method`} checked={draft.method === o.id} onChange={() => set('method', o.id)} className="accent-text" /> {o.label}
            </label>
          ))}
        </div>
        {draft.method === 'other' && (
          <input aria-label="How the file was acquired" value={draft.other} onChange={(e) => set('other', e.target.value)} maxLength={200} className={`${inputClass} mt-1 max-w-[360px]`} />
        )}
        {draft.method === 'screen_capture' && (
          <p className="mt-1 mb-0 text-caption text-text" role="note">Screenshots remove metadata and re-encode the image; several checks will be less reliable.</p>
        )}
      </div>

      <label htmlFor={`${uid}-custodian`} className={label}>Custodian</label>
      <input id={`${uid}-custodian`} value={draft.custodian} onChange={(e) => set('custodian', e.target.value)} maxLength={255} placeholder="Who supplied the file" className={`${inputClass} max-w-[420px]`} />

      <label htmlFor={`${uid}-source`} className={label}>Source URL</label>
      <div>
        <input id={`${uid}-source`} value={draft.source} onChange={(e) => set('source', e.target.value)} maxLength={2048} placeholder="https://…, optional" inputMode="url" autoCapitalize="none" spellCheck={false}
          aria-invalid={badSource || undefined} aria-describedby={badSource ? `${uid}-source-problem` : undefined} className={cx(inputClass, 'max-w-[520px]', badSource && 'border-danger')} />
        {badSource && <p id={`${uid}-source-problem`} role="alert" data-testid="problem-source" className={problemClass}>{problems.source}</p>}
      </div>

      <label htmlFor={`${uid}-received`} className={label}>Received</label>
      <div>
        <DateTimeField id={`${uid}-received`} value={draft.received} onChange={(v) => set('received', v)} />
        {lateReceived && <p role="alert" data-testid="problem-received" className={problemClass}>{problems.received}</p>}
      </div>

      <label htmlFor={`${uid}-hash`} className={label}>Custodian’s hash</label>
      <div>
        <input id={`${uid}-hash`} value={draft.statedHash} onChange={(e) => set('statedHash', e.target.value)} maxLength={80} placeholder="SHA-256 or MD5, optional" spellCheck={false}
          aria-invalid={badHash || undefined} aria-describedby={badHash ? `${uid}-hash-problem` : undefined} className={cx(inputClass, 'max-w-[560px] font-mono text-mono', badHash && 'border-danger')} />
        {badHash
          ? <p id={`${uid}-hash-problem`} role="alert" data-testid="problem-hash" className={problemClass}>{problems.statedHash}</p>
          : <p className="mt-1 mb-0 text-caption text-text-muted">Compared with the hashes computed on intake: agrees, differs or not provided.</p>}
      </div>

      <label htmlFor={`${uid}-notes`} className={label}>Notes</label>
      <div>
        <textarea id={`${uid}-notes`} value={draft.notes} onChange={(e) => set('notes', e.target.value)} rows={2} maxLength={3800}
          className="w-full max-w-[560px] px-2.5 py-1.5 rounded-sm border border-graphic bg-surface text-small leading-6 text-text resize-y focus-visible:border-accent" />
        <p className="mt-1 mb-0 text-caption text-text-muted">Shown in the report’s custody section.</p>
      </div>
    </div>
  )
}
