import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Boxes, FolderOpen, LayoutList, FilePlus2, Sun, Moon, Play, ScanLine } from 'lucide-react'
import { api } from './lib/api'
import { useMe, useModelCards } from './lib/api/queries'
import { availableChecks } from './features/models/availability'
import { engineCapacityNotice, engineNotice, engineReady } from './features/engine/engineState'
import { maySeeEngineFacts, maySeeServerCapacity, mayStartWork } from './lib/api/roles'
import { deviceWords } from './lib/recordWords'
import { useTheme } from './lib/theme'
import { cx } from './lib/cx'
import { Dot, Eyebrow, Button } from './components/ui'
import { useDemo } from './demo/DemoProvider'
import AccountMenu, { signOut } from './features/auth/AccountMenu'
import ChangePasswordDialog from './features/auth/ChangePasswordDialog'

const HEALTH_INTERVAL = 15000

/** The page identity for the fade-in: the analysis workspace's section routes stay one page. */
const pageKey = (pathname: string) => pathname.replace(/^(\/analyses\/[^/]+)\/[^/]+$/, '$1')
/** The evidence viewer is a full-viewport route: the sidebar collapses to its 48 px rail (UX §5.9). */
const isViewer = (pathname: string) => /^\/analyses\/[^/]+\/evidence$/.test(pathname)

function NavItem({ to, end, icon, label, rail }: { to: string; end?: boolean; icon: React.ReactNode; label: string; rail: boolean }) {
  return (
    <NavLink to={to} end={end} title={label}
      className={({ isActive }) => cx(
        'relative flex items-center gap-2.5 h-11 px-3 text-small font-medium transition-colors duration-base max-[639px]:flex-1 max-[639px]:flex-col max-[639px]:gap-1 max-[639px]:justify-center max-[639px]:h-14',
        rail ? 'justify-center px-0' : 'max-[959px]:justify-center max-[959px]:px-0',
        isActive ? 'text-text' : 'text-text-muted hover:text-text',
      )}>
      {({ isActive }) => (
        <>
          {isActive && <span aria-hidden className="absolute left-0 top-0 bottom-0 w-0.5 bg-accent" />}
          <span className="shrink-0 inline-flex" aria-hidden>{icon}</span>
          <span className={rail ? 'sr-only max-[639px]:not-sr-only max-[639px]:text-caption max-[639px]:whitespace-nowrap' : 'max-[959px]:sr-only max-[639px]:not-sr-only max-[639px]:text-caption max-[639px]:whitespace-nowrap'}>{label}</span>
        </>
      )}
    </NavLink>
  )
}

export default function App() {
  const health = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: HEALTH_INTERVAL, retry: 0 })
  const catalogue = useModelCards()
  const me = useMe()
  const session = me.data?.auth === 'session' ? me.data : null
  const { theme, toggle } = useTheme()
  const demo = useDemo()
  const loc = useLocation()
  const rail = isViewer(loc.pathname)
  const [retryIn, setRetryIn] = useState(HEALTH_INTERVAL / 1000)
  const [changingPassword, setChangingPassword] = useState(false)
  const changePassword = () => setChangingPassword(true)
  // N-9b: a sign-out the server did not complete is said; the person is still signed in
  const [signOutFailed, setSignOutFailed] = useState<string | null>(null)
  const [signingOut, setSigningOut] = useState(false)
  const leave = () => {
    if (signingOut) return
    setSigningOut(true)
    setSignOutFailed(null)
    signOut().catch((e: Error) => { setSignOutFailed(e.message); setSigningOut(false) })
  }
  const offline = health.isError

  // Countdown for the offline strip, reset on each fetch.
  useEffect(() => {
    if (!offline) return
    setRetryIn(HEALTH_INTERVAL / 1000)
    const t = window.setInterval(() => setRetryIn((s) => (s > 0 ? s - 1 : 0)), 1000)
    return () => window.clearInterval(t)
  }, [offline, health.dataUpdatedAt, health.errorUpdatedAt])

  const h = health.data
  const starts = mayStartWork(me.data)
  const availableCount = catalogue.isError || !catalogue.data ? null : availableChecks(catalogue.data).length
  const checksLabel = catalogue.isError ? 'unavailable' : catalogue.isPending ? 'loading…' : availableCount == null ? 'unavailable' : `${availableCount} available`
  const checksTitle = catalogue.isPending ? 'Loading available checks…' : availableCount == null ? 'Available checks could not be confirmed.' : `${availableCount} checks available: enabled with required files installed. The profile and source determine which checks run.`
  // N-13: the engine answers and says whether it can take work. Not ready is said, with the first reason a person can act on
  const engineStatus = health.isPending ? 'loading' : offline ? 'offline' : engineReady(h) ? 'ok' : 'failed'
  const notice = offline ? null : engineNotice(h)
  // N-14a: checks the server's memory budget can never run, for the person who can change the server
  const capacityNotice = offline || !maySeeServerCapacity(me.data) ? null : engineCapacityNotice(h)
  // N-19a: the device, the count of checks and the queue describe the server and are shown to administrators; everyone else
  // sees whether the engine is ready
  const facts = maySeeEngineFacts(me.data)
  const engineWord = health.isPending ? 'Engine: checking' : offline ? 'Engine offline' : notice ? 'Engine not ready' : 'Engine ready'
  const engineTitle = !facts ? engineWord
    : offline || !h ? 'offline' : `${deviceWords(h.device) ?? 'device not shown'} · checks ${checksLabel} · queue ${h.queue_depth ?? '—'}${notice ? ' · not ready' : ''}${capacityNotice ? ` · ${capacityNotice}` : ''}`

  return (
    <div className="min-h-screen flex max-[639px]:flex-col">
      <a href="#main-content" className="skip-link">Skip to content</a>
      <aside className={cx(
        'workspace-sidebar shrink-0 bg-surface-2 border-border flex flex-col',
        rail ? 'w-12' : 'w-(--container-sidebar)', 'border-r min-h-screen sticky top-0 max-h-screen',
        'max-[959px]:w-12',
        'max-[639px]:w-full max-[639px]:min-h-0 max-[639px]:max-h-none max-[639px]:items-center max-[639px]:border-r-0 max-[639px]:border-b max-[639px]:static',
      )}>
        <div className={cx('workspace-brand px-5 pt-7 pb-6 max-[959px]:px-0 max-[959px]:text-center max-[639px]:py-0 max-[639px]:px-4 max-[639px]:text-left', rail && 'px-0 text-center')}>
          <NavLink to="/" className="display text-[25px] leading-7 font-medium text-text flex items-center gap-2.5 max-[959px]:justify-center max-[639px]:justify-start" aria-label="Veritas home">
            <ScanLine size={24} strokeWidth={1.5} aria-hidden className={cx('text-accent max-[959px]:hidden max-[639px]:block', rail && 'hidden')} />
            <span className={rail ? 'hidden max-[639px]:inline' : 'max-[959px]:hidden max-[639px]:inline'}>Veritas</span>
            <span className={rail ? 'inline max-[639px]:hidden' : 'hidden max-[959px]:inline max-[639px]:hidden'} aria-hidden>V</span>
          </NavLink>
          <p className={cx('text-caption text-text-muted mt-0.5 max-[959px]:hidden', rail && 'hidden')}>Media authenticity lab</p>
        </div>
        <nav aria-label="Primary" className="workspace-shell-nav flex flex-col gap-1 mx-2 mt-1 max-[959px]:mx-1 max-[639px]:flex-row max-[639px]:mt-0">
          <NavItem rail={rail} to="/" end icon={<FolderOpen size={16} strokeWidth={1.5} />} label="Cases" />
          <NavItem rail={rail} to="/analyses" icon={<LayoutList size={16} strokeWidth={1.5} />} label="Analyses" />
          {starts && <NavItem rail={rail} to="/upload" icon={<FilePlus2 size={16} strokeWidth={1.5} />} label="New analysis" />}
          <NavItem rail={rail} to="/models" icon={<Boxes size={16} strokeWidth={1.5} />} label="Models" />
        </nav>
        {/* N-19a ("Actions by role"): a viewer or a reviewer is not offered what starts work */}
        {starts && <div className={cx('workspace-demo px-4 mt-5 max-[959px]:px-0 max-[959px]:flex max-[959px]:justify-center', rail && 'px-0 flex justify-center')}>
          <Button variant="ghost" data-testid="demo-now" title={demo.active ? 'Demo running' : 'Demo now — a guided walkthrough of five samples'} aria-pressed={demo.active} onClick={demo.start}
            className={cx('w-full h-8 px-3 gap-2 justify-start max-[959px]:w-8 max-[959px]:px-0 max-[959px]:justify-center max-[959px]:border-0', rail && 'w-8 px-0 justify-center border-0')}>
            <Play size={14} strokeWidth={1.5} aria-hidden className="shrink-0" />
            <span className={rail ? 'sr-only' : 'max-[959px]:sr-only'}>{demo.active ? 'Demo running' : 'Demo now'}</span>
          </Button>
        </div>}
        <div className={cx('workspace-engine mt-auto px-4 pb-5 pt-4 border-t border-border max-[959px]:px-0 max-[959px]:flex max-[959px]:flex-col max-[959px]:items-center max-[959px]:gap-3 max-[639px]:ml-auto max-[639px]:border-t-0 max-[639px]:p-0 max-[639px]:pr-3 max-[639px]:mt-0 max-[639px]:flex-row max-[639px]:items-center max-[639px]:gap-2', rail && 'px-0 flex flex-col items-center gap-3')}>
          <div className={rail ? 'hidden max-[639px]:block' : 'max-[959px]:hidden max-[639px]:block'} title={facts ? checksTitle : engineWord} data-testid="engine-block" data-facts={facts}>
            {/* ENGINE line carries the theme toggle, right-aligned to the text column (icon glyph flush with the 16 px inset). */}
            <div className="flex items-center justify-between h-8">
              <div className="flex items-center gap-2">
                <Dot status={engineStatus} />
                <Eyebrow as="span">Engine</Eyebrow>
              </div>
              <div className="flex items-center -mr-2 max-[639px]:mr-0">
                <Button variant="icon" onClick={toggle} aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'} title="Theme">
                  {theme === 'dark' ? <Sun size={16} strokeWidth={1.5} aria-hidden /> : <Moon size={16} strokeWidth={1.5} aria-hidden />}
                </Button>
                {session && <span className="hidden max-[639px]:inline-flex"><AccountMenu session={session} variant="icon" side="bottom" align="right" onChangePassword={changePassword} onSignOut={leave} /></span>}
              </div>
            </div>
            {facts && (
              <dl className="grid grid-cols-[56px_1fr] gap-x-3 gap-y-0.5 mt-1 text-caption num max-[639px]:hidden">
                <dt className="text-text-muted">device</dt><dd className="m-0 text-text" data-testid="engine-device">{h ? deviceWords(h.device) ?? '—' : offline ? 'offline' : '—'}</dd>
                <dt className="text-text-muted">checks</dt><dd className="m-0 text-text">{checksLabel}</dd>
                <dt className="text-text-muted">queue</dt><dd className="m-0 text-text" data-testid="engine-queue">{h ? h.queue_depth ?? '—' : '—'}</dd>
              </dl>
            )}
            {capacityNotice && (
              <p role="status" data-testid="engine-capacity" className="mt-2 mb-0 text-caption text-text num max-[639px]:hidden">{capacityNotice}</p>
            )}
            {session && (
              <div className="mt-3 pt-2 border-t border-border max-[639px]:hidden" data-testid="signed-in">
                <AccountMenu session={session} variant="name" onChangePassword={changePassword} onSignOut={leave} />
              </div>
            )}
          </div>
          <div className={cx('flex-col items-center gap-3', rail ? 'flex max-[639px]:hidden' : 'hidden max-[959px]:flex max-[639px]:hidden')}>
            <span title={engineTitle}><Dot status={engineStatus} /></span>
            <Button variant="icon" onClick={toggle} aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'} title="Theme">
              {theme === 'dark' ? <Sun size={16} strokeWidth={1.5} aria-hidden /> : <Moon size={16} strokeWidth={1.5} aria-hidden />}
            </Button>
            {session && <AccountMenu session={session} variant="icon" onChangePassword={changePassword} onSignOut={leave} />}
          </div>
        </div>
      </aside>
      <div className={cx('flex-1 min-w-0 flex flex-col', rail && 'h-dvh')}>
        {offline && (
          <div role="status" className="min-h-8 py-2 shrink-0 flex flex-wrap items-center gap-2 px-12 max-[639px]:px-4 bg-surface-2 border-b border-border text-small text-text">
            Engine offline — analyses cannot be submitted. Retrying… <span className="font-mono text-mono text-text-muted num">{retryIn}s</span>
          </div>
        )}
        {notice && (
          <div role="status" data-testid="engine-not-ready" className="min-h-8 py-2 shrink-0 flex flex-wrap items-center gap-2 px-12 max-[639px]:px-4 bg-surface-2 border-b border-border text-small text-text">
            {notice}
          </div>
        )}
        {session && signOutFailed && (
          <div role="alert" data-testid="sign-out-failed" className="min-h-8 py-2 shrink-0 flex flex-wrap items-center gap-x-4 gap-y-1 px-12 max-[639px]:px-4 bg-surface-2 border-b border-border text-small text-text">
            <span><span className="font-medium text-danger">Sign-out failed.</span> You are still signed in as {session.display_name}. {signOutFailed}</span>
            <span className="inline-flex items-center gap-4">
              <Button variant="text" onClick={leave} disabled={signingOut}>Try again</Button>
              <Button variant="text" onClick={() => setSignOutFailed(null)}>Dismiss</Button>
            </span>
          </div>
        )}
        <main id="main-content" tabIndex={-1} key={pageKey(loc.pathname)} className={rail ? 'flex-1 min-h-0 w-full' : 'w-full mx-auto max-w-(--container-content) px-10 pt-10 pb-24 animate-fade-in max-[959px]:px-6 max-[639px]:px-4 max-[639px]:pt-6'}>
          <Outlet />
        </main>
      </div>
      {session && changingPassword && <ChangePasswordDialog username={session.username} onClose={() => setChangingPassword(false)} />}
    </div>
  )
}
