import { useId, useLayoutEffect, useRef, useState, type FormEvent } from 'react'
import type { Box, Size } from '../../lib/coords'
import { Button } from '../../components/ui'

export type Visibility = 'internal' | 'in_report'
export interface AnnotationDraft { box: Box; body: string; visibility: Visibility }

const FIELDS = [['x', 'x'], ['y', 'y'], ['w', 'width'], ['h', 'height']] as const

const fieldsOf = (box: Box | null) => (box ? { x: String(box.x), y: String(box.y), w: String(box.w), h: String(box.h) } : { x: '', y: '', w: '', h: '' })

/** Note on a region (UX §5.9 "Annotate", §5.13): every annotation needs a note and a visibility; coordinates are source pixels.
 *  Opened after drawing a box (prefilled) or from "Add region by coordinates", the keyboard alternative (WCAG 2.5.7). A native
 *  modal dialog: focus is trapped and Esc cancels.
 *  Each opening is a new session: the fields are reset during render, so the committed DOM already holds the drawn box (or empty
 *  fields) when `showModal` runs in the layout effect. Resetting in an effect instead showed the previous note's values for one
 *  commit. */
export default function AnnotationDialog({ open, box, source, pending, error, onCancel, onSave }: {
  open: boolean
  box: Box | null
  source: Size
  pending: boolean
  error: string | null
  onCancel: () => void
  onSave: (draft: AnnotationDraft) => void
}) {
  const ref = useRef<HTMLDialogElement>(null)
  const uid = useId()
  const [session, setSession] = useState<{ open: boolean; box: Box | null }>({ open: false, box: null })
  const [fields, setFields] = useState(() => fieldsOf(box))
  const [body, setBody] = useState('')
  const [visibility, setVisibility] = useState<Visibility>('internal')
  const [invalid, setInvalid] = useState<string | null>(null)

  if (session.open !== open || (open && session.box !== box)) {
    setSession({ open, box })
    if (open) {
      setFields(fieldsOf(box))
      setBody('')
      setVisibility('internal')
      setInvalid(null)
    }
  }

  useLayoutEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) d.showModal()
    else if (!open && d.open) d.close()
  }, [open])

  const submit = (e: FormEvent) => {
    e.preventDefault()
    const n = { x: Number(fields.x), y: Number(fields.y), w: Number(fields.w), h: Number(fields.h) }
    if (Object.values(fields).some((v) => v.trim() === '') || Object.values(n).some((v) => !Number.isFinite(v))) {
      setInvalid('Enter x, y, width and height in source pixels.')
      return
    }
    if (n.x < 0 || n.y < 0 || n.w <= 0 || n.h <= 0 || n.x + n.w > source.width || n.y + n.h > source.height) {
      setInvalid(`The region must lie inside the ${source.width} × ${source.height} px image.`)
      return
    }
    if (!body.trim()) {
      setInvalid('Every annotation needs a note.')
      return
    }
    setInvalid(null)
    onSave({ box: n, body: body.trim(), visibility })
  }

  const message = invalid ?? error
  return (
    <dialog ref={ref} aria-labelledby={`${uid}-title`} data-testid="annotation-dialog" onCancel={(e) => { e.preventDefault(); onCancel() }}
      className="m-auto w-[440px] max-w-[calc(100vw-32px)] p-0 rounded-md border border-border bg-surface text-text" style={{ boxShadow: 'var(--shadow-popover)' }}>
      <form onSubmit={submit} className="p-6" noValidate>
        <h2 id={`${uid}-title`} className="m-0 text-[17px] leading-6 font-medium">{box ? 'Note on the drawn region' : 'Add region by coordinates'}</h2>
        <p className="mt-1 mb-0 text-small text-text-muted">Source pixels of the {source.width} × {source.height} px image. The note joins the examiner layer and the custody log.</p>
        <fieldset className="mt-5 m-0 p-0 border-0 grid grid-cols-4 gap-3">
          <legend className="sr-only">Region in source pixels</legend>
          {FIELDS.map(([k, label]) => (
            <label key={k} className="flex flex-col gap-1 text-small text-text-muted">
              {label}
              <input type="number" inputMode="decimal" step="any" min={0} value={fields[k]} onChange={(e) => setFields((f) => ({ ...f, [k]: e.target.value }))}
                className="h-9 w-full min-w-0 px-2 rounded-sm border border-graphic bg-surface text-text font-mono text-mono num" />
            </label>
          ))}
        </fieldset>
        <label className="mt-4 flex flex-col gap-1 text-small text-text-muted">
          Note
          <textarea value={body} onChange={(e) => setBody(e.target.value)} rows={4} maxLength={10000}
            className="px-2 py-1.5 rounded-sm border border-graphic bg-surface text-text text-small leading-6 resize-y" />
        </label>
        <fieldset className="mt-4 m-0 p-0 border-0">
          <legend className="p-0 text-small text-text-muted">Visibility</legend>
          <div className="mt-1 flex items-center gap-5 text-small text-text">
            <label className="inline-flex items-center gap-2 min-h-8"><input type="radio" name={`${uid}-visibility`} checked={visibility === 'internal'} onChange={() => setVisibility('internal')} className="accent-text" /> Internal</label>
            <label className="inline-flex items-center gap-2 min-h-8"><input type="radio" name={`${uid}-visibility`} checked={visibility === 'in_report'} onChange={() => setVisibility('in_report')} className="accent-text" /> Include in report</label>
          </div>
        </fieldset>
        {message && <p role="alert" className="mt-4 mb-0 text-small text-danger">{message}</p>}
        <div className="dialog-actions dialog-actions-inset mt-2 flex items-center justify-end gap-4">
          <Button variant="text" onClick={onCancel}>Cancel</Button>
          <Button variant="primary" type="submit" disabled={pending}>{pending ? 'Saving…' : 'Save note'}</Button>
        </div>
      </form>
    </dialog>
  )
}
