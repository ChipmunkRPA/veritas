/** A session that ended while the application is open (N-19a; browser QA P1-11). A request the server answers with 401 notes
 *  it here; the sign-in gate then shows the sign-in form in place, over the page as it is, so what a person has typed is kept.
 *  The first load without credentials is not this: the gate has no principal then and shows the form by itself. */
import { useSyncExternalStore } from 'react'

const KEY_STORAGE = 'veritas_api_key'

/** What a signed-in person reads where a request was refused for the session: not the sentence about an API key. */
export const SESSION_ENDED = 'Your session has ended. Sign in again to continue.'

let ended = false
const listeners = new Set<() => void>()
const tell = () => { for (const listener of listeners) listener() }

export function noteSessionEnded(): void {
  if (!ended) { ended = true; tell() }
}

export function clearSessionEnded(): void {
  if (ended) { ended = false; tell() }
}

export const sessionEnded = (): boolean => ended

const subscribe = (listener: () => void) => {
  listeners.add(listener)
  return () => { listeners.delete(listener) }
}

export function useSessionEnded(): boolean {
  return useSyncExternalStore(subscribe, sessionEnded, () => false)
}

const keyStored = (): boolean => typeof localStorage !== 'undefined' && localStorage.getItem(KEY_STORAGE) !== null

/** A 401 of any request but the sign-in itself: it is noted, and in session mode (no stored key) its sentence is
 *  `SESSION_ENDED`. With a stored key the server's own sentence stays: the key is what was refused. */
export function refusedSession(detail: string): string {
  noteSessionEnded()
  return keyStored() ? detail : SESSION_ENDED
}
