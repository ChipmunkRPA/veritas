import { useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient, unwrap } from '../../lib/api/client'
import { qk, type Case } from '../../lib/api/queries'
import type { components } from '../../lib/api/schema'
import { Button } from '../../components/ui'
import { textareaClass } from '../review/Dispositions'
import { CLASSIFICATIONS, CLASSIFICATION_WORD } from './labels'
import { inputClass } from './NewCaseDialog'

type Patch = components['schemas']['CasePatch']

/** `2031-03-01` of a stored retention date (kept in UTC), or '' for none. */
export const retentionDay = (iso: string | null | undefined): string => (iso ? new Date(iso).toISOString().slice(0, 10) : '')

/** A typed day as the instant the API stores (the start of that day in UTC), null for an empty field, undefined when it is no date. */
export function retentionInstant(day: string): string | null | undefined {
  const text = day.trim()
  if (text === '') return null
  if (!/^\d{4}-\d{2}-\d{2}$/.test(text)) return undefined
  const t = new Date(`${text}T00:00:00Z`)
  return Number.isNaN(t.getTime()) || t.toISOString().slice(0, 10) !== text ? undefined : `${text}T00:00:00Z`
}

/** What the form changed, as the body of `PATCH /cases/{id}`: only the fields that differ from the case, so the custody event
 *  names what was changed and nothing else. */
export function casePatch(c: Case, f: { title: string; matter: string; description: string; classification: Case['classification']
  closed: boolean; second: boolean; hold: boolean; retention: string | null }, isAdmin: boolean): Patch {
  const patch: Patch = {}
  if (f.title.trim() !== c.title) patch.title = f.title.trim()
  if ((f.matter.trim() || null) !== (c.matter_reference ?? null)) patch.matter_reference = f.matter.trim() || null
  if ((f.description.trim() || null) !== (c.description ?? null)) patch.description = f.description.trim() || null
  if (f.classification !== c.classification) patch.classification = f.classification
  if (f.closed !== (c.status === 'closed')) patch.status = f.closed ? 'closed' : 'open'
  if (isAdmin) {
    if (f.second !== c.requires_second_review) patch.requires_second_review = f.second
    if (f.hold !== c.legal_hold) patch.legal_hold = f.hold
    if ((f.retention ? Date.parse(f.retention) : null) !== (c.retention_until ? Date.parse(c.retention_until) : null)) patch.retention_until = f.retention
  }
  return patch
}

/** Edit case (D-70; design review DS-14; `PATCH /cases/{id}`): the owner changes the title, the client or matter reference, the
 *  description and the classification, and closes or reopens the case; an administrator also sets the second-review rule, the
 *  legal hold and the retention date (the API gives these three to administrators only). Each change is one `case_updated`
 *  event of the custody log. The case number is not edited: issued reports carry it. */
export default function EditCaseDialog({ open, case: c, isAdmin, onClose }: { open: boolean; case: Case; isAdmin: boolean; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null)
  const uid = useId()
  const qc = useQueryClient()
  const [title, setTitle] = useState(c.title)
  const [matter, setMatter] = useState(c.matter_reference ?? '')
  const [description, setDescription] = useState(c.description ?? '')
  const [classification, setClassification] = useState(c.classification)
  const [closed, setClosed] = useState(c.status === 'closed')
  const [second, setSecond] = useState(c.requires_second_review)
  const [hold, setHold] = useState(c.legal_hold)
  const [retention, setRetention] = useState(retentionDay(c.retention_until))
  const [invalid, setInvalid] = useState<string | null>(null)

  const save = useMutation({
    mutationFn: (body: Patch) => unwrap(apiClient.PATCH('/api/v2/cases/{case_id}', { params: { path: { case_id: c.id } }, body })),
    onSuccess: (updated) => {
      qc.setQueryData(qk.case(c.id), updated)
      void qc.invalidateQueries({ queryKey: qk.cases() })
      void qc.invalidateQueries({ queryKey: qk.custody(c.id) })
      // the second-review rule changes what every signed revision of the case reads
      void qc.invalidateQueries({ queryKey: qk.caseAnalyses(c.id) })
      void qc.invalidateQueries({ queryKey: ['analysis'] })
      onClose()
    },
  })

  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) {
      setTitle(c.title); setMatter(c.matter_reference ?? ''); setDescription(c.description ?? ''); setClassification(c.classification)
      setClosed(c.status === 'closed'); setSecond(c.requires_second_review); setHold(c.legal_hold); setRetention(retentionDay(c.retention_until))
      setInvalid(null)
      save.reset()
      d.showModal()
    } else if (!open && d.open) {
      d.close()
    }
    // The form is filled from the case only when the dialog opens.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!title.trim()) { setInvalid('A case needs a title.'); return }
    const until = retentionInstant(retention)
    if (isAdmin && until === undefined) { setInvalid('Write the retention date as 2031-03-01, or leave it empty.'); return }
    setInvalid(null)
    const body = casePatch(c, { title, matter, description, classification, closed, second, hold, retention: until ?? null }, isAdmin)
    if (Object.keys(body).length === 0) { onClose(); return }
    save.mutate(body)
  }

  const busy = save.isPending
  const message = invalid ?? (save.error ? save.error.message : null)
  return (
    <dialog ref={ref} aria-labelledby={`${uid}-title`} data-testid="edit-case-dialog" onCancel={(e) => { e.preventDefault(); if (!busy) onClose() }}
      className="m-auto w-[560px] max-w-[calc(100vw-32px)] p-0 rounded-md border border-border bg-surface text-text" style={{ boxShadow: 'var(--shadow-popover)' }}>
      <form onSubmit={submit} noValidate>
        <h2 id={`${uid}-title`} className="m-0 px-6 pt-6 pb-4 text-[17px] leading-6 font-medium border-b border-border">
          Edit case <span className="font-mono text-mono text-text-muted">{c.reference}</span>
        </h2>
        <div className="px-6 py-5">
          <p className="m-0 text-small text-text-muted">Each change is recorded in the custody log of the case. The case number stays as it is: issued reports carry it.</p>
          <label className="mt-4 flex flex-col gap-1 text-small text-text-muted">
            Title
            <input value={title} onChange={(e) => setTitle(e.target.value)} maxLength={255} required aria-required="true" className={inputClass} />
          </label>
          <label className="mt-3 flex flex-col gap-1 text-small text-text-muted">
            Client / matter reference
            <input value={matter} onChange={(e) => setMatter(e.target.value)} maxLength={255} className={inputClass} />
          </label>
          <label className="mt-3 flex flex-col gap-1 text-small text-text-muted">
            Description
            <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={3} maxLength={10000} className={textareaClass} />
          </label>
          <fieldset className="mt-4 m-0 p-0 border-0">
            <legend className="p-0 text-small text-text-muted">Classification</legend>
            <div className="mt-1 flex flex-wrap items-center gap-x-5 text-small text-text">
              {CLASSIFICATIONS.map((k) => (
                <label key={k} className="inline-flex items-center gap-2 min-h-8">
                  <input type="radio" name={`${uid}-classification`} checked={classification === k} onChange={() => setClassification(k)} className="accent-text" /> {CLASSIFICATION_WORD[k]}
                </label>
              ))}
            </div>
          </fieldset>
          <fieldset className="mt-3 m-0 p-0 border-0">
            <legend className="p-0 text-small text-text-muted">Status</legend>
            <div className="mt-1 flex flex-wrap items-center gap-x-5 text-small text-text">
              <label className="inline-flex items-center gap-2 min-h-8">
                <input type="radio" name={`${uid}-status`} checked={!closed} onChange={() => setClosed(false)} className="accent-text" /> Open
              </label>
              <label className="inline-flex items-center gap-2 min-h-8">
                <input type="radio" name={`${uid}-status`} checked={closed} onChange={() => setClosed(true)} className="accent-text" /> Closed
              </label>
            </div>
            <p className="m-0 text-caption text-text-muted">A closed case takes no new exhibits. It can be reopened.</p>
          </fieldset>
          {isAdmin ? (
            <div role="group" aria-labelledby={`${uid}-rules`} className="mt-5 pt-4 border-t border-border" data-testid="case-rules">
              <p id={`${uid}-rules`} className="eyebrow m-0 mb-2">Rules · administrator</p>
              <label className="flex items-center gap-2 min-h-8 text-small text-text">
                <input type="checkbox" checked={second} onChange={(e) => setSecond(e.target.checked)} className="accent-text" /> Sign-off needs a second reviewer
              </label>
              <label className="flex items-center gap-2 min-h-8 text-small text-text">
                <input type="checkbox" checked={hold} onChange={(e) => setHold(e.target.checked)} className="accent-text" /> Legal hold
                <span className="text-caption text-text-muted">nothing of the case is purged while it is on</span>
              </label>
              <label className="mt-2 flex flex-col gap-1 text-small text-text-muted">
                Retain until
                <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  <input value={retention} onChange={(e) => setRetention(e.target.value)} placeholder="YYYY-MM-DD" inputMode="numeric" autoComplete="off" spellCheck={false}
                    maxLength={10} className={`${inputClass} !w-36 font-mono text-mono num`} />
                  <span className="text-caption text-text-muted">a day in UTC; empty for the retention period of the instance</span>
                </span>
              </label>
            </div>
          ) : (c.requires_second_review || c.legal_hold || c.retention_until) && (
            <p className="mt-4 mb-0 text-caption text-text-muted">The second-review rule, the legal hold and the retention date are set by an administrator.</p>
          )}
        </div>
        <div className="dialog-actions px-6 py-4 border-t border-border">
          {message && <p role="alert" className="m-0 mb-3 text-small text-danger">{message}</p>}
          <div className="flex items-center justify-end gap-4">
            <Button variant="text" onClick={onClose} disabled={busy}>Cancel</Button>
            <Button variant="primary" type="submit" disabled={busy}>{busy ? 'Saving…' : 'Save changes'}</Button>
          </div>
        </div>
      </form>
    </dialog>
  )
}
