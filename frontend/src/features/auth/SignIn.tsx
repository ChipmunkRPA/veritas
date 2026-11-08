import { useId, useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import BrandMark from '../../components/BrandMark'
import { apiClient, unwrap } from '../../lib/api/client'
import { useBilling } from '../../lib/api/billing'
import { clearSessionEnded } from '../../lib/api/sessionEnd'
import { Button } from '../../components/ui'
import { inputClass } from '../cases/NewCaseDialog'
import { announceSession, noteSignIn, sessionNotKeptMessage } from './session'
import { STANDARD_SERVICE_POLICY } from '../analysis/standardServicePolicy'

const KEY_STORAGE = 'veritas_api_key'

/** Browser sign-in (PRD CM-02, PL-04; BUILD_SPEC L-6): shown instead of the workspace when the instance needs credentials and the
 *  browser has none that work. `POST /auth/login` sets the HttpOnly session cookie and the readable CSRF cookie, which the API
 *  client sends back on every write. A stored API key that the server refused is removed, because a key header takes precedence
 *  over the session. `sessionNotKept`: the server accepted a sign-in a moment ago and `GET /me` is still refused, which is what
 *  plain HTTP from another machine gives (the cookies are `Secure`).
 *  `named` (N-9b): the instance let the browser in without an account (single-user mode, a stored key), and the person came here
 *  because a conclusion or a sign-off needs one; `named.back` is where they came from. */
export default function SignIn({ keyRefused, sessionNotKept = false, named }: {
  keyRefused: boolean; sessionNotKept?: boolean; named?: { back: string; singleUser: boolean }
}) {
  const uid = useId()
  const qc = useQueryClient()
  const billing = useBilling()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [invalid, setInvalid] = useState<string | null>(null)

  const login = useMutation({
    mutationFn: () => unwrap(apiClient.POST('/api/v2/auth/login', { body: { username: username.trim(), password } }), { signIn: true }),
    onSuccess: () => {
      noteSignIn()
      announceSession('signed-in')
      if (localStorage.getItem(KEY_STORAGE) !== null) {
        // Media and download helpers read the key once per page load: start clean in session mode.
        localStorage.removeItem(KEY_STORAGE)
        window.location.reload()
        return
      }
      clearSessionEnded()  // the refusals that led to this form are answered (N-19a)
      void qc.resetQueries()
    },
  })

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!username.trim() || !password) { setInvalid('Enter your username and password.'); return }
    setInvalid(null)
    login.mutate()
  }

  const message = invalid ?? (login.error ? login.error.message : null)
  return (
    <main id="main-content" className="min-h-screen bg-surface text-text px-10 pt-16 pb-24 max-[639px]:px-4 max-[639px]:pt-8" data-testid="sign-in">
      {/* N-19d (design review, P2 "screens"): the wordmark and the form are one column in the middle of the page (they stood at
          its left edge, with the rest of a wide window empty) */}
      <p className="mx-auto my-0 max-w-[360px] font-semibold flex items-center gap-2.5">
        <BrandMark className="w-9 h-9 shrink-0" /><span className="leading-none"><span className="block text-[21px] tracking-[-0.045em]">chipmunk</span><span className="block mt-1 text-[10px] uppercase tracking-[0.2em]">Verify</span></span>
      </p>
      <form onSubmit={submit} noValidate aria-labelledby={`${uid}-title`} className="mx-auto mt-16 max-w-[360px] max-[639px]:mt-10">
        <h1 id={`${uid}-title`} className="display text-display font-normal m-0">Sign in</h1>
        <p className="mt-3 mb-0 text-small text-text-muted" data-testid={named ? 'sign-in-named' : undefined}>Sign in to access your private analyses and reports. Use the username and password issued for your account.</p>
        {billing.data?.enabled && <div className="mt-5 p-4 rounded-md border border-border bg-surface-2" aria-label="Analysis price">
          <p className="m-0 text-body font-medium">{billing.data.price_display} USD per analysis</p>
          <p className="mt-2 mb-0 text-small text-text-muted">One payment through Stripe. No subscription. An automated result may be inconclusive.</p>
          {billing.data.mode === 'test' && <p className="mt-2 mb-0 text-caption text-accent">Test mode — no real payment is collected.</p>}
        </div>}
        <p className="mt-5 mb-0 text-small text-text-muted" aria-label="Screening limitations">{STANDARD_SERVICE_POLICY.confidence_notice}</p>
        {keyRefused && <p className="mt-2 mb-0 text-small text-text-muted">The API key stored in this browser was not accepted; signing in replaces it.</p>}
        {sessionNotKept && !login.isPending && (
          <p role="alert" data-testid="session-not-kept" className="mt-4 mb-0 text-small text-danger">
            {sessionNotKeptMessage(window.location.protocol, window.location.hostname)}
          </p>
        )}
        <label className="mt-6 flex flex-col gap-1 text-small text-text-muted">
          Username
          <input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" autoCapitalize="none" spellCheck={false}
            maxLength={64} required aria-required="true" className={inputClass} />
        </label>
        <label className="mt-3 flex flex-col gap-1 text-small text-text-muted">
          Password
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password"
            maxLength={1024} required aria-required="true" className={inputClass} />
        </label>
        {message && <p role="alert" className="mt-4 mb-0 text-small text-danger">{message}</p>}
        <div className="mt-6 flex items-center gap-4 flex-wrap">
          <Button variant="primary" type="submit" disabled={login.isPending}>{login.isPending ? 'Signing in…' : 'Sign in'}</Button>
          {named && <Link to={named.back} className="text-small text-accent underline underline-offset-[3px] decoration-1 hover:decoration-2">Back without signing in</Link>}
        </div>
        <p className="mt-8 mb-0 text-small text-text-muted">Need an account or help signing in? Contact <a href="mailto:sales@chipmunkrpa.com" className="text-accent underline underline-offset-4">sales@chipmunkrpa.com</a>.</p>
      </form>
    </main>
  )
}
