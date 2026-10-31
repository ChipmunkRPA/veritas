import { useId, useState, type FormEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import { apiClient, unwrap } from '../../lib/api/client'
import type { Annotation, Finding } from '../../lib/api/queries'
import { Button, EmptyState, IdChip, Menu } from '../../components/ui'
import { findingsNotWritten } from '../analysis/vocabulary'
import { EXAMINER_STATUSES, NOTE_REQUIRED, STATUS_WORD, dispositions, type ExaminerStatus, type Visibility } from './review'

export const textareaClass = 'w-full px-2.5 py-1.5 rounded-sm border border-graphic bg-surface text-small leading-6 text-text resize-y focus-visible:border-accent'

/** Whether a note is printed in reports or kept to the workspace: the same choice for a status note as for any other note (L-7). */
export function VisibilityChoice({ name, value, onChange }: { name: string; value: Visibility; onChange: (v: Visibility) => void }) {
  return (
    <fieldset className="mt-2 m-0 p-0 border-0">
      <legend className="sr-only">Visibility</legend>
      <div className="flex flex-wrap items-center gap-x-5 text-small text-text">
        <label className="inline-flex items-center gap-2 min-h-8"><input type="radio" name={name} checked={value === 'internal'} onChange={() => onChange('internal')} className="accent-text" /> Internal</label>
        <label className="inline-flex items-center gap-2 min-h-8"><input type="radio" name={name} checked={value === 'in_report'} onChange={() => onChange('in_report')} className="accent-text" /> Include in report</label>
      </div>
    </fieldset>
  )
}

/** Examiner status of each finding (PRD RW-01; UX §5.13): accepted, disputed, needs check, not relevant. A disputed or not-relevant
 *  status opens a note first, which the examiner keeps internal or includes in the report. Each change is a new annotation; the
 *  most recent status is current (`app/cases/examiner.py`). */
export default function Dispositions({ analysisId, findings, annotations, writable, onChanged, notWritten }: {
  analysisId: string
  findings: readonly Finding[]
  annotations: readonly Annotation[]
  writable: boolean
  onChanged: () => void
  /** N-19n (Q4-01): the step that writes the findings failed: the list says so, not that there is nothing to review, and
   *  where the checks and their scores stand: the evidence ledger, or the expert view from the Plain view */
  notWritten?: 'ledger' | 'expert'
}) {
  const current = dispositions(annotations)
  const [draft, setDraft] = useState<{ findingId: string; status: ExaminerStatus } | null>(null)
  const [note, setNote] = useState('')
  const [visibility, setVisibility] = useState<Visibility>('internal')
  const [invalid, setInvalid] = useState<string | null>(null)
  const uid = useId()
  const post = useMutation({
    mutationFn: ({ findingId, status, body, visibility: shown }: { findingId: string; status: ExaminerStatus; body: string | null; visibility: Visibility }) =>
      unwrap(apiClient.POST('/api/v2/analyses/{aid}/annotations', {
        params: { path: { aid: analysisId } },
        body: { target_type: 'finding', target_id: findingId, shape: 'none', examiner_status: status, body: body ?? undefined, visibility: shown },
      })),
    onSuccess: () => { setDraft(null); setNote(''); setVisibility('internal'); onChanged() },
  })

  const choose = (findingId: string, status: ExaminerStatus) => {
    post.reset()
    setInvalid(null)
    if (NOTE_REQUIRED.has(status)) {
      setDraft({ findingId, status })
      setNote('')
      setVisibility('internal')
      return
    }
    setDraft(null)
    post.mutate({ findingId, status, body: null, visibility: 'internal' })
  }
  const save = (e: FormEvent) => {
    e.preventDefault()
    if (!draft) return
    if (!note.trim()) { setInvalid(`A ${STATUS_WORD[draft.status].toLowerCase()} status needs a note.`); return }
    post.mutate({ ...draft, body: note.trim(), visibility })
  }

  return (
    <div className="mt-8" data-testid="dispositions">
      <h3 className="m-0 text-body font-medium text-text">Findings</h3>
      <p className="mt-1 mb-0 text-caption text-text-muted">Your status for each machine finding. Disputed and not relevant need a note.</p>
      {findings.length === 0 ? <EmptyState className="mt-3">{notWritten ? findingsNotWritten(notWritten) : 'This analysis has no findings to review.'}</EmptyState> : (
        <ul className="mt-3 m-0 p-0 list-none border-t border-border">
          {findings.map((f) => {
            const d = current.get(f.id)
            const editing = draft?.findingId === f.id ? draft : null
            return (
              <li key={f.id} data-testid="disposition" data-finding={f.id} data-status={d?.status ?? 'none'} className="border-b border-border py-2">
                <div className="flex items-baseline gap-3 flex-wrap min-h-8">
                  <IdChip>{f.id}</IdChip>
                  <span className="text-small text-text min-w-0 flex-1 break-words">{f.title}</span>
                  <span className="text-small whitespace-nowrap">
                    {d ? <span className="text-annotation font-medium">{STATUS_WORD[d.status]}</span> : <span className="text-text-muted">No status</span>}
                  </span>
                  {writable && (
                    <Menu align="right" label="Set status" ariaLabel={`Set the examiner status of ${f.id}`} onSelect={(s) => choose(f.id, s)}
                      items={EXAMINER_STATUSES.map((s) => ({ id: s, label: STATUS_WORD[s] }))} />
                  )}
                </div>
                {d?.note && (
                  <p className="mt-0.5 mb-1 pl-11 text-small text-text whitespace-pre-line max-w-[72ch] max-[639px]:pl-0">
                    {d.note} <span data-testid="disposition-visibility" className="text-text-muted whitespace-nowrap">· {d.annotation.visibility === 'in_report' ? 'in report' : 'internal'}</span>
                  </p>
                )}
                {editing && (
                  <form onSubmit={save} className="mt-2 mb-2 pl-11 max-w-[680px] max-[639px]:pl-0" noValidate>
                    <label className="flex flex-col gap-1 text-small text-text-muted">
                      Note for {STATUS_WORD[editing.status].toLowerCase()}
                      {/* eslint-disable-next-line jsx-a11y/no-autofocus */}
                      <textarea value={note} onChange={(e) => setNote(e.target.value)} rows={3} maxLength={10000} autoFocus className={textareaClass} />
                    </label>
                    <VisibilityChoice name={`${uid}-visibility`} value={visibility} onChange={setVisibility} />
                    {(invalid ?? post.error?.message) && <p role="alert" className="mt-2 mb-0 text-small text-danger">{invalid ?? post.error?.message}</p>}
                    <div className="mt-2 flex items-center gap-4">
                      <Button variant="primary" type="submit" disabled={post.isPending}>{post.isPending ? 'Saving…' : `Mark ${STATUS_WORD[editing.status].toLowerCase()}`}</Button>
                      <Button variant="text" onClick={() => { setDraft(null); setInvalid(null) }}>Cancel</Button>
                    </div>
                  </form>
                )}
              </li>
            )
          })}
        </ul>
      )}
      {!draft && post.error && <p role="alert" className="mt-2 mb-0 text-small text-danger">{post.error.message}</p>}
    </div>
  )
}
