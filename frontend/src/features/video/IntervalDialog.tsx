import { useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { Button } from '../../components/ui'
import { timecode, type PtsIndex } from '../../lib/pts'
import type { Visibility } from '../viewer/AnnotationDialog'

export interface IntervalDraft { startFrame: number; endFrame: number; body: string; visibility: Visibility }

/** Note on a time interval (UX §5.10 notes, §5.13; ARCHITECTURE §7.5 shift-drag). Opened from a shift-drag on the lanes (prefilled) or
 *  "Add note" (the selected segment or the current frame): the frames are pts-table indices and the note is required. */
export default function IntervalDialog({ open, startFrame, endFrame, index, pending, error, onCancel, onSave }: {
  open: boolean
  startFrame: number
  endFrame: number
  index: PtsIndex
  pending: boolean
  error: string | null
  onCancel: () => void
  onSave: (draft: IntervalDraft) => void
}) {
  const ref = useRef<HTMLDialogElement>(null)
  const uid = useId()
  const [fields, setFields] = useState({ start: '', end: '' })
  const [body, setBody] = useState('')
  const [visibility, setVisibility] = useState<Visibility>('internal')
  const [invalid, setInvalid] = useState<string | null>(null)

  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) {
      setFields({ start: String(startFrame), end: String(endFrame) })
      setBody('')
      setVisibility('internal')
      setInvalid(null)
      d.showModal()
    } else if (!open && d.open) {
      d.close()
    }
  }, [open, startFrame, endFrame])

  const last = index.count - 1
  const s = Number(fields.start)
  const e = Number(fields.end)
  const valid = Number.isInteger(s) && Number.isInteger(e) && s >= 0 && e <= last && e >= s
  const submit = (ev: FormEvent) => {
    ev.preventDefault()
    if (!valid) {
      setInvalid(`Enter whole frame numbers from 0 to ${last}, the end at or after the start.`)
      return
    }
    if (!body.trim()) {
      setInvalid('Every annotation needs a note.')
      return
    }
    setInvalid(null)
    onSave({ startFrame: s, endFrame: e, body: body.trim(), visibility })
  }
  const message = invalid ?? error

  return (
    <dialog ref={ref} aria-labelledby={`${uid}-title`} data-testid="interval-dialog" onCancel={(ev) => { ev.preventDefault(); onCancel() }}
      className="m-auto w-[440px] max-w-[calc(100vw-32px)] p-0 rounded-md border border-border bg-surface text-text" style={{ boxShadow: 'var(--shadow-popover)' }}>
      <form onSubmit={submit} className="p-6" noValidate>
        <h2 id={`${uid}-title`} className="m-0 text-[17px] leading-6 font-medium">Note on an interval</h2>
        <p className="mt-1 mb-0 text-small text-text-muted">Frames of the pts table (0 to {last}). The note joins the examiner layer and the custody log.</p>
        <fieldset className="mt-5 m-0 p-0 border-0 grid grid-cols-2 gap-3">
          <legend className="sr-only">Interval in frames</legend>
          {([['start', 'First frame'], ['end', 'Last frame']] as const).map(([k, label]) => {
            const v = Number(fields[k])
            return (
              <label key={k} className="flex flex-col gap-1 text-small text-text-muted">
                {label}
                <input type="number" inputMode="numeric" step={1} min={0} max={last} value={fields[k]} onChange={(ev) => setFields((f) => ({ ...f, [k]: ev.target.value }))}
                  className="h-9 w-full min-w-0 px-2 rounded-sm border border-graphic bg-surface text-text font-mono text-mono num" />
                <span className="font-mono text-micro num">{Number.isInteger(v) && v >= 0 && v <= last ? timecode(index.times[v]) : '—'}</span>
              </label>
            )
          })}
        </fieldset>
        <label className="mt-4 flex flex-col gap-1 text-small text-text-muted">
          Note
          <textarea value={body} onChange={(ev) => setBody(ev.target.value)} rows={4} maxLength={10000}
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
