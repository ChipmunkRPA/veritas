import { useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient, unwrap } from '../../lib/api/client'
import { qk, type Me } from '../../lib/api/queries'
import { clearSessionEnded } from '../../lib/api/sessionEnd'
import { Button } from '../../components/ui'
import { inputClass } from '../cases/NewCaseDialog'
import { announceSession, noteSignIn } from './session'

const KEY_STORAGE = 'veritas_api_key'

/** The sign-in form in place (N-19a; browser QA P1-11): the session ended while the application was open. The page stays as it
 *  is behind the dialog, so a note, a conclusion or an intake form keeps what was typed; after the sign-in the page reads its
 *  data again and the person sends the form. A different account than before loads the application again: nothing of the
 *  previous principal stays on screen. The dialog cannot be dismissed: every request behind it would be refused. */
export default function SessionEndedDialog({ me }: { me: Me }) {
  const ref = useRef<HTMLDialogElement>(null)
  const secret = useRef<HTMLInputElement>(null)
  const uid = useId()
  const qc = useQueryClient()
  const [username, setUsername] = useState(me.username ?? '')
  const [password, setPassword] = useState('')
  const [invalid, setInvalid] = useState<string | null>(null)

  const login = useMutation({
    mutationFn: async () => {
      await unwrap(apiClient.POST('/api/v2/auth/login', { body: { username: username.trim(), password } }), { signIn: true })
      return unwrap(apiClient.GET('/api/v2/me'))
    },
    onSuccess: (now) => {
      noteSignIn()
      announceSession('signed-in')
      if (localStorage.getItem(KEY_STORAGE) !== null || now.id !== me.id) {
        // a refused stored key is replaced by the session, and another account starts clean
        localStorage.removeItem(KEY_STORAGE)
        window.location.reload()
        return
      }
      qc.setQueryData(qk.me(), now)
      clearSessionEnded()
      void qc.invalidateQueries()  // what failed behind the dialog is read again; the forms keep their text
    },
  })

  useEffect(() => {
    const d = ref.current
    if (d && !d.open) {
      d.showModal()
      if (me.username) secret.current?.focus()  // the name is filled in: the password is what is typed
    }
    return () => { if (d?.open) d.close() }
    // Opened once, when the gate shows it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!username.trim() || !password) { setInvalid('Enter your username and password.'); return }
    setInvalid(null)
    login.mutate()
  }

  const message = invalid ?? (login.error ? login.error.message : null)
  return (
    <dialog ref={ref} aria-labelledby={`${uid}-title`} data-testid="session-ended" onCancel={(e) => e.preventDefault()}
      className="m-auto w-[400px] max-w-[calc(100vw-32px)] p-0 rounded-md border border-border bg-surface text-text" style={{ boxShadow: 'var(--shadow-popover)' }}>
      <form onSubmit={submit} noValidate className="p-6">
        <h2 id={`${uid}-title`} className="m-0 text-[17px] leading-6 font-medium">Your session has ended</h2>
        <p className="mt-1 mb-0 text-small text-text-muted">Sign in again to continue. This page stays as it is: what you have typed is kept.</p>
        <label className="mt-5 flex flex-col gap-1 text-small text-text-muted">
          Username
          <input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" autoCapitalize="none" spellCheck={false}
            maxLength={64} required aria-required="true" className={inputClass} />
        </label>
        <label className="mt-3 flex flex-col gap-1 text-small text-text-muted">
          Password
          <input ref={secret} type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password"
            maxLength={1024} required aria-required="true" className={inputClass} />
        </label>
        {message && <p role="alert" className="mt-4 mb-0 text-small text-danger">{message}</p>}
        <div className="dialog-actions dialog-actions-inset mt-2 flex items-center justify-end">
          <Button variant="primary" type="submit" disabled={login.isPending}>{login.isPending ? 'Signing in…' : 'Sign in'}</Button>
        </div>
      </form>
    </dialog>
  )
}
