import { useEffect } from 'react'
import { Navigate, useSearchParams } from 'react-router-dom'
import { useMe } from '../../lib/api/queries'
import SignIn from './SignIn'
import { followSessionChanges, safeNext, signedInJustNow } from './session'

/** `/sign-in?next=<path>` (N-9b; code review 16): the sign-in form for a browser the instance already lets in without an
 *  account. A single-user instance answers `GET /me` as the local examiner and a stored key as its principal, so the gate never
 *  shows the form; but a conclusion and a sign-off need a named account, and the review panel links here. Once the session
 *  names a principal the page returns to `next`. Where `GET /me` is refused the form is the gate's own. */
export default function SignInPage() {
  const me = useMe()
  const [params] = useSearchParams()
  const back = safeNext(params.get('next'))
  useEffect(() => followSessionChanges(), [])
  if (me.data?.auth === 'session') return <Navigate to={back} replace />
  if (me.isPending) return <div className="min-h-screen" aria-busy="true" />
  return <SignIn keyRefused={!me.isSuccess && typeof localStorage !== 'undefined' && localStorage.getItem('veritas_api_key') !== null}
    sessionNotKept={signedInJustNow()}
    named={me.isSuccess ? { back, singleUser: me.data.unauthenticated_instance } : undefined} />
}
