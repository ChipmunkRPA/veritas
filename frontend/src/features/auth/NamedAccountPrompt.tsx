import type { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { signInPath } from './session'

/** Shown where the API would refuse the caller for not being a named account (N-9b; code review 16): the sentence says what
 *  needs one, and "Sign in" opens the sign-in page, which returns here. No form the server refuses is offered beside it. */
export default function NamedAccountPrompt({ children, className = 'mt-3' }: { children: ReactNode; className?: string }) {
  const loc = useLocation()
  return (
    <p data-testid="named-account-prompt" className={`${className} mb-0 text-small text-text max-w-[72ch]`}>
      {children}{' '}
      <Link to={signInPath(`${loc.pathname}${loc.search}`)} className="text-accent underline underline-offset-[3px] decoration-1 hover:decoration-2">Sign in</Link>
    </p>
  )
}
