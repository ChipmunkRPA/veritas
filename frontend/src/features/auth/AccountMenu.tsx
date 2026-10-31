import { ChevronsUpDown, CircleUserRound } from 'lucide-react'
import { ApiError, apiClient, unwrap } from '../../lib/api/client'
import type { Me } from '../../lib/api/queries'
import { Menu, type MenuItem } from '../../components/ui'
import { announceSession } from './session'

type Action = 'password' | 'sign-out'
const ITEMS: MenuItem<Action>[] = [{ id: 'password', label: 'Change password…' }, { id: 'sign-out', label: 'Sign out' }]

/** L-6: ends the browser session, tells the other tabs, then reloads so no evidence of the signed-out principal stays in memory.
 *  A sign-out the server did not complete rejects with what to tell the person, and nothing is reloaded: the session is still
 *  open, and a page that showed the sign-in form would say otherwise (N-9b; code review 17). */
export async function signOut(): Promise<void> {
  try {
    await unwrap(apiClient.POST('/api/v2/auth/logout'))
  } catch (e) {
    throw new Error(e instanceof ApiError ? e.message : 'The server did not answer.')
  }
  announceSession('signed-out')
  window.location.assign('/')
}

/** The account menu of a browser session (L-6, N-6a): "Change password…" and "Sign out".
 *  - `name`: the display name is the trigger (the open sidebar);
 *  - `icon`: an icon trigger, for the 48 px rail (`side="top"`) and for the top bar below 640 px (`side="bottom"`).
 *  The shell owns the dialog and the sign-out (and what it shows when one fails): one of the three triggers is visible at a
 *  width, and a resize must not close or hide either. */
export default function AccountMenu({ session, variant, onChangePassword, onSignOut, side = 'top', align = 'left' }: {
  session: Me; variant: 'name' | 'icon'; onChangePassword: () => void; onSignOut: () => void; side?: 'top' | 'bottom'; align?: 'left' | 'right'
}) {
  const label = `Account: ${session.display_name}`
  return (
    <Menu<Action> items={ITEMS} side={side} align={align} ariaLabel={label}
      onSelect={(id) => { if (id === 'password') onChangePassword(); else onSignOut() }}
      className={variant === 'name' ? 'block' : undefined}
      triggerClassName={variant === 'name' ? '-mx-1 w-[calc(100%+0.5rem)] justify-between' : 'h-8 w-8 justify-center hover:bg-surface-3'}
      trigger={variant === 'name'
        ? <>
            <span className="min-w-0 truncate text-caption text-text" title={`Signed in as ${session.username}`}>{session.display_name}</span>
            <ChevronsUpDown size={14} strokeWidth={1.5} aria-hidden className="shrink-0" />
          </>
        : <span title={label} className="inline-flex"><CircleUserRound size={16} strokeWidth={1.5} aria-hidden /></span>} />
  )
}
