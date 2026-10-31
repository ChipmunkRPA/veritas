/** A sign-in the server accepted, remembered for this tab (it must survive the reload that follows a replaced API key). When
 *  `GET /me` still answers 401 afterwards, the browser did not keep the session cookie (N-6; code review 17). */
const SIGNED_IN_AT = 'veritas_signed_in_at'
const FRESH_MS = 60_000

export function noteSignIn(now = Date.now()): void {
  try { sessionStorage.setItem(SIGNED_IN_AT, String(now)) } catch { /* storage is off: the form comes back without the note */ }
}

export function clearSignIn(): void {
  try { sessionStorage.removeItem(SIGNED_IN_AT) } catch { /* nothing to clear */ }
}

export function signedInJustNow(now = Date.now()): boolean {
  try {
    const at = Number(sessionStorage.getItem(SIGNED_IN_AT))
    return at > 0 && now - at >= 0 && now - at < FRESH_MS
  } catch { return false }
}

const LOOPBACK = new Set(['localhost', '127.0.0.1', '[::1]', '::1'])

/** Why the session was not kept. The session cookie is `Secure` on a server: a browser keeps it over HTTPS, and over plain HTTP
 *  only on the machine itself. */
export function sessionNotKeptMessage(protocol: string, hostname: string): string {
  const accepted = 'Your username and password were accepted, but this browser did not keep the session.'
  if (protocol === 'http:' && !LOOPBACK.has(hostname.toLowerCase())) {
    return `${accepted} Browser sign-in needs HTTPS: over plain HTTP a browser keeps the session only on the server itself. `
      + 'Open this address with https://, or ask your administrator to set up TLS.'
  }
  return `${accepted} Allow cookies for this site, then sign in again.`
}

/** The sign-in page for a principal that is not a named account, returning to `next` (a path of this application). */
export const signInPath = (next: string): string => `/sign-in?next=${encodeURIComponent(next)}`

/** Where the sign-in page returns to: a path of this application, never another origin (`//host`, `/\\host`, a scheme).
 *
 *  The value is judged as a browser resolves it, not as it is spelled (N-10b; code re-verification R2-05): a URL parser drops
 *  a tab or a newline, so `/<tab>/host` is `//host`, another origin, although its second character is not a slash. A value
 *  with a control character or a backslash is refused; the rest is resolved against this origin, must stay on it, and the
 *  page goes to its path, query and fragment. */
export function safeNext(next: string | null | undefined, origin: string = window.location.origin): string {
  if (!next || !next.startsWith('/') || /[\u0000-\u001f\u007f\\]/.test(next)) return '/'
  try {
    const url = new URL(next, origin)
    // A path such as "/.//host/x" or "/x/..//host/x" stays on this origin and resolves to the pathname "//host/x", which a
    // browser handed that string reads as another origin: what is returned is navigated to, so it is judged as well.
    if (url.origin !== new URL(origin).origin || url.pathname.startsWith('//')) return '/'
    return `${url.pathname}${url.search}${url.hash}`
  } catch {
    return '/'
  }
}

const SESSION_EVENT = 'veritas_session_event'
export type SessionChange = 'signed-in' | 'signed-out'

/** Tells the other tabs of this browser that the session changed (N-9b; code review 17). A `storage` event reaches every other
 *  tab of the origin and never the one that wrote; the value differs each time, so each change is delivered. */
export function announceSession(change: SessionChange, now = Date.now()): void {
  try { localStorage.setItem(SESSION_EVENT, `${change}:${now}`) } catch { /* storage is off: the other tabs find out at their next request */ }
}

/** Calls `listener` when another tab signed in or out. Returns the unsubscribe. */
export function onSessionChange(listener: (change: SessionChange) => void): () => void {
  const handler = (e: StorageEvent) => {
    if (e.key !== SESSION_EVENT || !e.newValue) return
    const change = e.newValue.split(':')[0]
    if (change === 'signed-in' || change === 'signed-out') listener(change)
  }
  window.addEventListener('storage', handler)
  return () => window.removeEventListener('storage', handler)
}

/** What a tab does when another tab signed in or out: it loads again, so nothing of the previous principal stays in memory
 *  and every request, form and signer line belongs to the session the cookie now names. */
export function followSessionChanges(): () => void {
  return onSessionChange(() => window.location.reload())
}
