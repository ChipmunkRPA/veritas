import { useEffect, type ReactNode } from 'react'
import { ApiError } from '../../lib/api/client'
import { useMe } from '../../lib/api/queries'
import { useSessionEnded } from '../../lib/api/sessionEnd'
import SessionEndedDialog from './SessionEndedDialog'
import SignIn from './SignIn'
import { clearSignIn, followSessionChanges, signedInJustNow } from './session'

/** The workspace renders once `GET /me` names a principal (a stored key, a session or single-user mode). A 401 shows the sign-in
 *  form instead of every section failing with "Invalid or missing API key" (L-6). Other failures (an engine that is offline) fall
 *  through to the workspace, whose offline strip explains them. A 401 right after a sign-in the server accepted means the browser
 *  did not keep the session: the form says why instead of coming back empty (N-6). A sign-in or sign-out in another tab loads
 *  this one again (N-9b): the workspace of the principal who signed out does not stay on screen, and the form does not stay
 *  where someone has signed in.
 *
 *  A session that ends while the application is open (N-19a; browser QA P1-11): the gate knows a principal, and a request, of
 *  `GET /me` or of anything else, is answered with 401. The workspace then stays where it is and the sign-in form is shown
 *  in place, over it, so a draft is kept. */
export default function AuthGate({ children }: { children: ReactNode }) {
  const me = useMe()
  const named = me.data !== undefined
  const ended = useSessionEnded()
  const refused = me.error instanceof ApiError && me.error.status === 401
  useEffect(() => { if (named) clearSignIn() }, [named])
  useEffect(() => followSessionChanges(), [])
  if (refused && !named) {
    return <SignIn keyRefused={typeof localStorage !== 'undefined' && localStorage.getItem('veritas_api_key') !== null}
      sessionNotKept={signedInJustNow()} />
  }
  if (me.isPending) return <div className="min-h-screen" aria-busy="true" />
  return <>{children}{me.data && (ended || refused) && <SessionEndedDialog me={me.data} />}</>
}
