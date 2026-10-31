import { useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import { apiClient, unwrap } from '../../lib/api/client'
import { Button } from '../../components/ui'
import { inputClass } from '../cases/NewCaseDialog'

export const MIN_PASSWORD_CHARS = 12  // app.api.auth.MIN_PASSWORD_CHARS

/** What stops a change before it is sent, in the order a person fills the form; null when it can be sent. */
export function passwordProblem(current: string, next: string, repeat: string): string | null {
  if (!current) return 'Enter your current password.'
  if (next.length < MIN_PASSWORD_CHARS) return `The new password needs at least ${MIN_PASSWORD_CHARS} characters.`
  if (next === current) return 'The new password is the current password.'
  if (next !== repeat) return 'The two entries of the new password differ.'
  return null
}

/** A signed-in person replaces their own password (BUILD_SPEC N-6a): `POST /me/password` checks the current one again, ends the
 *  account's other sessions and keeps this one. Until then whoever created the account knows its password. Mounted per opening,
 *  so no entry outlives the dialog. A native modal dialog: focus is trapped and Esc cancels. */
export default function ChangePasswordDialog({ username, onClose }: { username: string; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null)
  const uid = useId()
  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const [repeat, setRepeat] = useState('')
  const [invalid, setInvalid] = useState<string | null>(null)

  const change = useMutation({
    mutationFn: () => unwrap(apiClient.POST('/api/v2/me/password', { body: { current_password: current, new_password: next } })),
    onSuccess: () => { setCurrent(''); setNext(''); setRepeat('') },
  })

  useEffect(() => {
    const d = ref.current
    if (d && !d.open) d.showModal()
  }, [])

  const submit = (e: FormEvent) => {
    e.preventDefault()
    const problem = passwordProblem(current, next, repeat)
    setInvalid(problem)
    if (problem === null) change.mutate()
  }

  const message = invalid ?? (change.error ? change.error.message : null)
  return (
    <dialog ref={ref} aria-labelledby={`${uid}-title`} data-testid="change-password-dialog" onCancel={(e) => { e.preventDefault(); onClose() }}
      className="m-auto w-[400px] max-w-[calc(100vw-32px)] p-0 rounded-md border border-border bg-surface text-text" style={{ boxShadow: 'var(--shadow-popover)' }}>
      {change.isSuccess ? (
        <div className="p-6">
          <h2 id={`${uid}-title`} className="m-0 text-[17px] leading-6 font-medium">Password changed</h2>
          <p role="status" className="mt-2 mb-0 text-small text-text-muted">
            Use the new password from your next sign-in. Any other session of this account was signed out; this one stays open.
          </p>
          <div className="mt-6 flex justify-end">
            <Button variant="primary" onClick={onClose}>Close</Button>
          </div>
        </div>
      ) : (
        <form onSubmit={submit} className="p-6" noValidate>
          <h2 id={`${uid}-title`} className="m-0 text-[17px] leading-6 font-medium">Change password</h2>
          <p className="mt-1 mb-0 text-small text-text-muted">
            Other sessions of this account are signed out. The audit log records the change, never a password.
          </p>
          {/* for password managers: which account the three fields belong to */}
          <input type="text" name="username" value={username} autoComplete="username" readOnly hidden />
          <label className="mt-5 flex flex-col gap-1 text-small text-text-muted">
            Current password
            <input type="password" value={current} onChange={(e) => setCurrent(e.target.value)} autoComplete="current-password"
              maxLength={1024} required aria-required="true" className={inputClass} />
          </label>
          <label className="mt-3 flex flex-col gap-1 text-small text-text-muted">
            New password
            <input type="password" value={next} onChange={(e) => setNext(e.target.value)} autoComplete="new-password"
              maxLength={1024} required aria-required="true" aria-describedby={`${uid}-rule`} className={inputClass} />
          </label>
          <p id={`${uid}-rule`} className="mt-1 mb-0 text-caption text-text-muted">At least {MIN_PASSWORD_CHARS} characters.</p>
          <label className="mt-3 flex flex-col gap-1 text-small text-text-muted">
            Repeat new password
            <input type="password" value={repeat} onChange={(e) => setRepeat(e.target.value)} autoComplete="new-password"
              maxLength={1024} required aria-required="true" className={inputClass} />
          </label>
          {message && <p role="alert" className="mt-4 mb-0 text-small text-danger">{message}</p>}
          <div className="dialog-actions dialog-actions-inset mt-2 flex items-center justify-end gap-4">
            <Button variant="text" onClick={onClose}>Cancel</Button>
            <Button variant="primary" type="submit" disabled={change.isPending}>{change.isPending ? 'Changing…' : 'Change password'}</Button>
          </div>
        </form>
      )}
    </dialog>
  )
}
