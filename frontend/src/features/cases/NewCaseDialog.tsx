import { useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient, unwrap } from '../../lib/api/client'
import { qk, useMe, type Case } from '../../lib/api/queries'
import { Button } from '../../components/ui'
import { CLASSIFICATIONS, CLASSIFICATION_WORD } from './labels'

/** A text field without a width (for fixed-width fields); `inputClass` fills its container. */
export const inputBase = 'h-9 min-w-0 px-2.5 rounded-sm border border-graphic bg-surface text-small text-text placeholder:text-text-muted focus-visible:border-accent'
export const inputClass = `${inputBase} w-full`

/** New case (PRD CM-01; UX §5.12 "New case", §5.1 "+ New case"): an optional case number (the next VX-<year>-<nnn> otherwise), a
 *  title, the client or matter reference, the classification and whether sign-off needs a second review (RW-04). An
 *  administrator can open the case under a legal hold (D-70; the API gives the hold to administrators). A native modal dialog:
 *  focus is trapped and Esc cancels. */
export default function NewCaseDialog({ open, onClose, onCreated }: { open: boolean; onClose: () => void; onCreated: (c: Case) => void }) {
  const ref = useRef<HTMLDialogElement>(null)
  const uid = useId()
  const qc = useQueryClient()
  const [reference, setReference] = useState('')
  const [title, setTitle] = useState('')
  const [matter, setMatter] = useState('')
  const [classification, setClassification] = useState<(typeof CLASSIFICATIONS)[number]>('internal')
  const [second, setSecond] = useState(false)
  const [hold, setHold] = useState(false)
  const isAdmin = useMe().data?.roles.includes('administrator') ?? false
  const [invalid, setInvalid] = useState<string | null>(null)

  const create = useMutation({
    mutationFn: () => unwrap(apiClient.POST('/api/v2/cases', {
      body: { title: title.trim(), reference: reference.trim() || undefined, matter_reference: matter.trim() || undefined, classification, requires_second_review: second, legal_hold: isAdmin && hold },
    })),
    onSuccess: (c) => {
      qc.setQueryData(qk.case(c.id), c)
      void qc.invalidateQueries({ queryKey: qk.cases() })
      onCreated(c)
    },
  })

  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) {
      setReference(''); setTitle(''); setMatter(''); setClassification('internal'); setSecond(false); setHold(false); setInvalid(null)
      create.reset()
      d.showModal()
    } else if (!open && d.open) {
      d.close()
    }
    // `create` is stable per mount; resetting it only when the dialog opens.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!title.trim()) { setInvalid('A case needs a title.'); return }
    if (reference.trim() && !/^[A-Za-z0-9][A-Za-z0-9._/-]{0,63}$/.test(reference.trim())) {
      setInvalid('A case number uses letters, digits and . _ / - (up to 64 characters).')
      return
    }
    setInvalid(null)
    create.mutate()
  }

  const message = invalid ?? (create.error ? create.error.message : null)
  return (
    <dialog ref={ref} aria-labelledby={`${uid}-title`} data-testid="new-case-dialog" onCancel={(e) => { e.preventDefault(); onClose() }}
      className="m-auto w-[480px] max-w-[calc(100vw-32px)] p-0 rounded-md border border-border bg-surface text-text" style={{ boxShadow: 'var(--shadow-popover)' }}>
      <form onSubmit={submit} className="p-6" noValidate>
        <h2 id={`${uid}-title`} className="m-0 text-[17px] leading-6 font-medium">New case</h2>
        <p className="mt-1 mb-0 text-small text-text-muted">You become the case owner. Opening the case is recorded in its custody log.</p>
        <div className="mt-5 grid grid-cols-[1fr_1.4fr] gap-3 max-[479px]:grid-cols-1">
          <label className="flex flex-col gap-1 text-small text-text-muted">
            Case number
            <input value={reference} onChange={(e) => setReference(e.target.value)} placeholder="Assigned when empty" maxLength={64} className={`${inputClass} font-mono text-mono`} />
          </label>
          <label className="flex flex-col gap-1 text-small text-text-muted">
            Client / matter <span className="sr-only">(optional)</span>
            <input value={matter} onChange={(e) => setMatter(e.target.value)} maxLength={255} className={inputClass} />
          </label>
        </div>
        <label className="mt-3 flex flex-col gap-1 text-small text-text-muted">
          Title
          <input value={title} onChange={(e) => setTitle(e.target.value)} maxLength={255} required aria-required="true" className={inputClass} />
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
        <label className="mt-2 flex items-center gap-2 min-h-8 text-small text-text">
          <input type="checkbox" checked={second} onChange={(e) => setSecond(e.target.checked)} className="accent-text" /> Sign-off needs a second reviewer
        </label>
        {isAdmin && (
          <label className="flex items-center gap-2 min-h-8 text-small text-text">
            <input type="checkbox" checked={hold} onChange={(e) => setHold(e.target.checked)} className="accent-text" /> Legal hold
            <span className="text-caption text-text-muted">nothing of the case is purged while it is on</span>
          </label>
        )}
        {message && <p role="alert" className="mt-4 mb-0 text-small text-danger">{message}</p>}
        <div className="dialog-actions dialog-actions-inset mt-2 flex items-center justify-end gap-4">
          <Button variant="text" onClick={onClose}>Cancel</Button>
          <Button variant="primary" type="submit" disabled={create.isPending}>{create.isPending ? 'Opening…' : 'Open case'}</Button>
        </div>
      </form>
    </dialog>
  )
}
