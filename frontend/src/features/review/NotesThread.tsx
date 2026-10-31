import { useId, useState, type FormEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import { apiClient, unwrap } from '../../lib/api/client'
import type { Annotation } from '../../lib/api/queries'
import { fmtDateLong } from '../../lib/format'
import { Button, Disclosure, EmptyState, IdChip } from '../../components/ui'
import { VisibilityChoice, textareaClass } from './Dispositions'
import { STATUS_WORD, anchorOf, noteNumbers, type Visibility } from './review'

/** Notes thread (UX §5.13; PRD RW-02, RW-03): every note and drawn mark of the analysis, numbered N1… in creation order, with author,
 *  time, anchor, visibility and plain-text body. An author edits their own notes before sign-off; the history stays readable. */
export default function NotesThread({ analysisId, annotations, names, meId, writable, onChanged }: {
  analysisId: string
  annotations: readonly Annotation[]
  names: ReadonlyMap<string, string>
  meId: string | null
  writable: boolean
  onChanged: () => void
}) {
  const uid = useId()
  const numbers = noteNumbers(annotations)
  const notes = [...numbers.keys()].map((id) => annotations.find((a) => a.id === id)!).filter(Boolean)
  const [body, setBody] = useState('')
  const [visibility, setVisibility] = useState<Visibility>('internal')
  const [invalid, setInvalid] = useState<string | null>(null)
  const [editing, setEditing] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [draftVisibility, setDraftVisibility] = useState<Visibility>('internal')

  const add = useMutation({
    mutationFn: () => unwrap(apiClient.POST('/api/v2/analyses/{aid}/annotations', {
      params: { path: { aid: analysisId } },
      body: { target_type: 'item', shape: 'none', body: body.trim(), visibility },
    })),
    onSuccess: () => { setBody(''); setVisibility('internal'); onChanged() },
  })
  const edit = useMutation({
    mutationFn: ({ id, text, visibility: shown }: { id: string; text: string; visibility: Visibility }) =>
      unwrap(apiClient.PATCH('/api/v2/annotations/{annotation_id}', { params: { path: { annotation_id: id } }, body: { body: text, visibility: shown } })),
    onSuccess: () => { setEditing(null); onChanged() },
  })

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!body.trim()) { setInvalid('A note needs text.'); return }
    setInvalid(null)
    add.mutate()
  }

  return (
    <div className="mt-10" data-testid="notes">
      <h3 className="m-0 text-body font-medium text-text">Notes</h3>
      {notes.length === 0 && <EmptyState className="mt-3">No notes yet.</EmptyState>}
      {notes.length > 0 && (
        <ol className="mt-3 m-0 p-0 list-none border-t border-border">
          {notes.map((a) => {
            const n = numbers.get(a.id)!
            const canEdit = writable && a.author_id === meId && (a.shape === 'none' || a.shape === 'sentence') && a.body != null  // a note on a sentence too (N-19a)
            const previous = a.history.slice(0, -1).filter((h) => h.body !== a.body || h.examiner_status !== a.examiner_status || h.visibility !== a.visibility)
            return (
              <li key={a.id} data-testid="note" data-number={n} className="border-b border-border py-3">
                <div className="flex items-baseline gap-x-2.5 gap-y-1 flex-wrap text-small">
                  <IdChip analyst>{n}</IdChip>
                  <span className="text-text font-medium">{names.get(a.author_id) ?? a.author_id}</span>
                  <span className="text-text-muted num">{fmtDateLong(a.created_at)}</span>
                  <IdChip title="Anchor">{anchorOf(a)}</IdChip>
                  {a.examiner_status && <span className="text-annotation">{STATUS_WORD[a.examiner_status]}</span>}
                  <span className="text-text-muted">{a.visibility === 'in_report' ? 'in report' : 'internal'}</span>
                  {canEdit && editing !== a.id && <Button variant="text" className="ml-auto" onClick={() => { setEditing(a.id); setDraft(a.body ?? ''); setDraftVisibility(a.visibility); edit.reset() }}>Edit</Button>}
                </div>
                {editing === a.id ? (
                  <form className="mt-2 max-w-[680px]" noValidate onSubmit={(e) => { e.preventDefault(); if (draft.trim()) edit.mutate({ id: a.id, text: draft.trim(), visibility: draftVisibility }) }}>
                    <label className="sr-only" htmlFor={`${uid}-edit-${a.id}`}>Edit {n}</label>
                    <textarea id={`${uid}-edit-${a.id}`} value={draft} onChange={(e) => setDraft(e.target.value)} rows={3} maxLength={10000} className={textareaClass} />
                    <VisibilityChoice name={`${uid}-edit-visibility-${a.id}`} value={draftVisibility} onChange={setDraftVisibility} />
                    {edit.error && <p role="alert" className="mt-2 mb-0 text-small text-danger">{edit.error.message}</p>}
                    <div className="mt-2 flex items-center gap-4">
                      <Button variant="primary" type="submit" disabled={edit.isPending || !draft.trim()}>{edit.isPending ? 'Saving…' : 'Save edit'}</Button>
                      <Button variant="text" onClick={() => setEditing(null)}>Cancel</Button>
                    </div>
                  </form>
                ) : (
                  <p className="mt-1.5 mb-0 text-body text-text whitespace-pre-line break-words max-w-[72ch]">{a.body ?? <span className="text-text-muted">Drawn {a.shape} without a note</span>}</p>
                )}
                {a.edited_at && previous.length > 0 && (
                  <Disclosure className="mt-1" label={`edited ${fmtDateLong(a.edited_at)}`}>
                    <ol className="m-0 p-0 list-none flex flex-col gap-2">
                      {previous.map((h) => (
                        <li key={h.revision} className="text-small text-text-muted">
                          <span className="num">Revision {h.revision} · {fmtDateLong(h.edited_at)}</span>
                          <span className="block text-text whitespace-pre-line">{h.body ?? '—'}</span>
                        </li>
                      ))}
                    </ol>
                  </Disclosure>
                )}
              </li>
            )
          })}
        </ol>
      )}

      {writable ? (
        <form onSubmit={submit} className="mt-4 max-w-[680px]" noValidate data-testid="add-note">
          <label className="flex flex-col gap-1 text-small text-text-muted">
            Add a note
            <textarea value={body} onChange={(e) => setBody(e.target.value)} rows={3} maxLength={10000} className={textareaClass} />
          </label>
          <VisibilityChoice name={`${uid}-visibility`} value={visibility} onChange={setVisibility} />
          {(invalid ?? add.error?.message) && <p role="alert" className="mt-2 mb-0 text-small text-danger">{invalid ?? add.error?.message}</p>}
          <Button variant="ghost" type="submit" className="mt-3" disabled={add.isPending}>{add.isPending ? 'Adding…' : 'Add note'}</Button>
        </form>
      ) : notes.length > 0 && <p className="mt-3 mb-0 text-caption text-text-muted">Notes are read-only on this revision.</p>}
    </div>
  )
}
