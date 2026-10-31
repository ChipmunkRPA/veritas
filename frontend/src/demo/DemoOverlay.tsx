import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { cx } from '../lib/cx'
import { prefersReducedMotion } from '../lib/theme'
import { Button, Dot, Eyebrow, type Status } from '../components/ui'
import { storeView } from '../features/analysis/view'
import { useDemo } from './DemoProvider'
import { cutOut, place, type Box, type Placement } from './place'
import { DEFAULT_STEP_MS, dispatchDemoAction, isDone, isFailed, type TourStep } from './types'

const CARD_W = 340
const ANCHOR_TIMEOUT_MS = 8000

const statusOf = (s: string | undefined): Status => (s === 'done' ? 'done' : s === 'failed' ? 'failed' : s === 'running' ? 'running' : 'queued')

export default function DemoOverlay() {
  const { state, steps, active } = useDemo()
  // The tour walks L3–L4 anchors (exhibit, how reached, provenance, methods), which only the expert result view renders (UX-01).
  useEffect(() => { if (active) storeView('expert') }, [active])
  if (!active) return null
  const step = state.phase === 'touring' ? steps[state.step] : undefined
  const waitingOn = step?.item != null && !isDone(state.items[step.item]?.status) ? state.items[step.item] : null
  const allDone = state.items.every((it) => isDone(it.status) || isFailed(it.status))
  const showStatus = state.phase === 'preparing' || !!waitingOn || !allDone
  return (
    <>
      {showStatus && <StatusCard waitingOn={waitingOn?.title ?? null} />}
      {step && !waitingOn && <TourCard step={step} />}
      {state.phase === 'touring' && step && waitingOn && <StepGate step={step} />}
    </>
  )
}

/** While a step waits for its analysis, a failed sample is skipped rather than blocking the tour. */
function StepGate({ step }: { step: TourStep }) {
  const { state, goTo } = useDemo()
  useEffect(() => {
    const it = step.item != null ? state.items[step.item] : null
    if (it && isFailed(it.status)) goTo(state.step + 1)
  }, [step, state.items, state.step, goTo])
  return null
}

function StatusCard({ waitingOn }: { waitingOn: string | null }) {
  const { state, finish, start } = useDemo()
  const done = state.items.filter((it) => isDone(it.status)).length
  const n = state.items.length
  return (
    <aside role="status" aria-live="polite" data-testid="demo-status"
      className="fixed bottom-6 right-6 z-40 w-[300px] max-w-[calc(100vw-48px)] bg-surface border border-border rounded-md shadow-popover p-4 animate-fade-in">
      <div className="flex items-center justify-between">
        <Eyebrow as="span">Demo</Eyebrow>
        <Button variant="text" className="text-text-muted hover:text-text" onClick={finish}>Cancel</Button>
      </div>
      {state.error ? (
        <p className="mt-2 text-small text-danger">{state.error} <Button variant="text" onClick={start}>Retry</Button></p>
      ) : (
        <>
          <p className="mt-1 text-small text-text num">{state.phase === 'preparing' ? 'Preparing demo' : 'Running demo'} · {n} samples · {done}/{n} analysed</p>
          <ul className="mt-3 m-0 p-0 list-none space-y-1.5">
            {state.items.map((it) => (
              <li key={it.id} className={cx('flex items-center gap-2.5 text-small', isFailed(it.status) ? 'text-danger' : isDone(it.status) ? 'text-text' : 'text-text-muted')}>
                <Dot status={statusOf(it.status)} />
                <span className="truncate">{it.title}</span>
                {!isDone(it.status) && <span className="ml-auto font-mono text-micro text-text-muted">{it.status}</span>}
              </li>
            ))}
          </ul>
          {waitingOn && <p className="mt-3 text-caption text-text-muted">Waiting for “{waitingOn}” to finish before the next step.</p>}
          {state.phase === 'preparing' && <p className="mt-3 text-caption text-text-muted">The walkthrough begins as soon as the first sample is ready. Models load on first use.</p>}
        </>
      )}
    </aside>
  )
}

/** Coach mark: scrim with a cut-out around the anchor, and the step card anchored beside it. */
function TourCard({ step }: { step: TourStep }) {
  const { state, steps, next, back, finish, skip, resumed } = useDemo()
  const nav = useNavigate()
  const loc = useLocation()
  const [target, setTarget] = useState<Element | null>(null)
  const [box, setBox] = useState<Box | null>(null)
  const [pos, setPos] = useState<Placement | null>(null)
  const [paused, setPaused] = useState(false)
  const cardRef = useRef<HTMLDivElement>(null)
  const barRef = useRef<HTMLSpanElement>(null)
  const elapsed = useRef(0)
  const focusedFor = useRef<string | null>(null)
  const reduced = prefersReducedMotion()
  const idx = state.step
  const total = steps.length
  const item = step.item != null ? state.items[step.item] : null
  const ms = step.ms ?? DEFAULT_STEP_MS

  // 1. Go to the step's route when the step begins. A visitor who then opens another address (a deep link, the sidebar, the back
  //    button) stays there: the tour waits and offers to resume, it does not pull the page back (L-7, QA P2). A tour restored
  //    from storage on a page load does not navigate either, so a deep link opened during a tour shows the page it names.
  const onRoute = loc.pathname === step.route
  const sentFor = useRef<string | null>(resumed ? step.key : null)
  const [arrivedFor, setArrivedFor] = useState<string | null>(null)
  useEffect(() => {
    if (onRoute) { setArrivedFor(step.key); sentFor.current = step.key; return }
    if (sentFor.current !== step.key) { sentFor.current = step.key; nav(step.route) }
  }, [onRoute, nav, step.key, step.route])
  const away = !onRoute && (arrivedFor === step.key || (resumed && sentFor.current === step.key))

  // 2. Wait for the anchor (the page may still be loading), then run the step's action and scroll it into view.
  // Keyed on `step.key`, not the step object: steps are rebuilt whenever a later sample's status changes and the
  // current step must not restart because of that.
  const latest = useRef({ step, skip })
  latest.current = { step, skip }
  useEffect(() => {
    setTarget(null); setBox(null); setPos(null); elapsed.current = 0
    if (!onRoute) return
    let cancelled = false
    const started = performance.now()
    const look = () => {
      if (cancelled) return
      const { step: st, skip: leave } = latest.current
      const el = document.querySelector(st.selector) ?? (st.fallback ? document.querySelector(st.fallback) : null)
      if (el) {
        setTarget(el)
        if (st.action) dispatchDemoAction(st.action)
        const r = el.getBoundingClientRect()
        el.scrollIntoView({ block: r.height > window.innerHeight * 0.7 ? 'start' : 'center', behavior: reduced ? 'auto' : 'smooth' })
        return
      }
      if (performance.now() - started > ANCHOR_TIMEOUT_MS) { leave(); return }
      timer = window.setTimeout(look, 150)
    }
    let timer = window.setTimeout(look, 50)
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [step.key, onRoute, reduced])

  // 3. Track the anchor's rect on scroll / resize / layout changes (expanding rows, loading images).
  useEffect(() => {
    if (!target) return
    let raf = 0
    const measure = () => {
      raf = 0
      const r = target.getBoundingClientRect()
      setBox((b) => (b && Math.abs(b.top - r.top) < 0.5 && Math.abs(b.left - r.left) < 0.5 && Math.abs(b.width - r.width) < 0.5 && Math.abs(b.height - r.height) < 0.5 ? b : { top: r.top, left: r.left, width: r.width, height: r.height }))
    }
    const schedule = () => { if (!raf) raf = requestAnimationFrame(measure) }
    measure()
    window.addEventListener('scroll', schedule, true)
    window.addEventListener('resize', schedule)
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(schedule) : null
    ro?.observe(target); ro?.observe(document.body)
    const iv = window.setInterval(schedule, 250)
    return () => { window.removeEventListener('scroll', schedule, true); window.removeEventListener('resize', schedule); ro?.disconnect(); window.clearInterval(iv); if (raf) cancelAnimationFrame(raf) }
  }, [target])

  // 4. Place the card once its size is known.
  useLayoutEffect(() => {
    if (!box || !cardRef.current) return
    const h = cardRef.current.offsetHeight
    setPos(place(box, CARD_W, h, window.innerWidth, window.innerHeight))
  }, [box])

  // 5. Auto-advance with a filling rule; pause on hover / keyboard focus inside the card.
  useEffect(() => {
    if (!box || paused) return
    const bar = barRef.current
    let raf = 0
    const start = performance.now() - elapsed.current
    const tick = (now: number) => {
      const e = now - start
      elapsed.current = e
      if (bar) bar.style.width = `${Math.min(100, (e / ms) * 100)}%`
      if (e >= ms) { if (step.final) finish(); else next(); return }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [box, paused, ms, step.final, next, finish, step.key])

  // Esc exits; focus the dialog for assistive tech without stealing the scroll position.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); finish() } }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [finish])
  useEffect(() => {
    if (pos && focusedFor.current !== step.key) { focusedFor.current = step.key; cardRef.current?.focus({ preventScroll: true }) }
  }, [pos, step.key])

  const onFocusIn = useCallback((e: React.FocusEvent) => { if ((e.target as HTMLElement).matches('button, a, [tabindex="0"]')) setPaused(true) }, [])
  const onFocusOut = useCallback((e: React.FocusEvent) => { if (!cardRef.current?.contains(e.relatedTarget as Node)) setPaused(false) }, [])

  if (away) {
    return (
      <aside role="status" data-testid="demo-paused"
        className="fixed bottom-6 right-6 z-40 w-[300px] max-w-[calc(100vw-48px)] bg-surface border border-border rounded-md shadow-popover p-4">
        <Eyebrow as="span">Demo</Eyebrow>
        <p className="mt-1 mb-0 text-small text-text num">The tour is paused at step {idx + 1} of {total} while you look at this page.</p>
        <div className="mt-3 flex items-center gap-3">
          <Button variant="ghost" className="h-8 px-3" onClick={() => nav(step.route)}>Resume tour</Button>
          <Button variant="text" className="text-text-muted hover:text-text" onClick={finish}>End tour</Button>
        </div>
      </aside>
    )
  }
  if (!box) return null
  const vw = window.innerWidth, vh = window.innerHeight
  // N-19d: the lit area ends above a card that found no room beside its target, so the card never covers what it highlights
  const cut = cutOut(box, pos, vw, vh)
  const motion = reduced ? '' : 'transition-[top,left,width,height] duration-base ease-standard'
  const scrim = 'fixed bg-scrim z-40'
  return (
    <>
      {/* Four scrim panels leave the anchor itself interactive; everything else is dimmed and inert. */}
      <div aria-hidden className={cx(scrim, motion)} style={{ top: 0, left: 0, right: 0, height: cut.top }} />
      <div aria-hidden className={cx(scrim, motion)} style={{ top: cut.bottom, left: 0, right: 0, bottom: 0 }} />
      <div aria-hidden className={cx(scrim, motion)} style={{ top: cut.top, left: 0, width: cut.left, height: cut.bottom - cut.top }} />
      <div aria-hidden className={cx(scrim, motion)} style={{ top: cut.top, left: cut.right, right: 0, height: cut.bottom - cut.top }} />
      <div aria-hidden data-testid="demo-cutout" className={cx('fixed z-40 pointer-events-none rounded-md border border-surface/70', motion)} style={{ top: cut.top, left: cut.left, width: cut.right - cut.left, height: cut.bottom - cut.top }} />

      <div ref={cardRef} role="dialog" aria-modal="false" aria-label="Demo walkthrough" aria-describedby="demo-tour-body" data-testid="demo-tour" data-step={idx + 1} tabIndex={-1}
        onMouseEnter={() => setPaused(true)} onMouseLeave={() => setPaused(false)} onFocus={onFocusIn} onBlur={onFocusOut}
        className={cx('fixed z-50 bg-surface border border-border rounded-md shadow-popover overflow-hidden outline-none', !reduced && 'animate-fade-in')}
        style={{ width: CARD_W, top: pos?.top ?? -9999, left: pos?.left ?? -9999, visibility: pos ? 'visible' : 'hidden' }}>
        <div aria-hidden className="h-0.5 bg-border"><span ref={barRef} className="block h-full bg-text" style={{ width: 0 }} /></div>
        <div className="p-4">
          {item && (
            <Eyebrow as="p">{item.title}{step.intro ? <span className="normal-case tracking-normal text-text-muted"> · {item.kind}</span> : null}</Eyebrow>
          )}
          <h2 className={cx('display text-h3 font-medium text-text', item && 'mt-1')}>{step.title}</h2>
          <p id="demo-tour-body" className="mt-1.5 text-small text-text-muted">{step.body}</p>
          {step.intro && item?.blurb && <p className="mt-2 text-caption text-text-muted border-l-2 border-border pl-2.5">{item.blurb}</p>}
          <div className="mt-4 flex items-center gap-3">
            {/* N-19d (browser QA "Demo tour"): one line from step 10 on too ("Step 10 of 21" took two) */}
            <span data-testid="demo-step" className="shrink-0 whitespace-nowrap font-mono text-caption text-text-muted num">Step {idx + 1} of {total}</span>
            <div className="ml-auto flex items-center gap-3">
              {step.final ? (
                <>
                  <Link to="/upload" onClick={finish} className="text-accent underline underline-offset-[3px] decoration-1 hover:text-accent-hover hover:decoration-2 text-small font-medium">Try your own file</Link>
                  <Button variant="primary" className="h-8 px-3" onClick={finish} data-testid="demo-next">Finish</Button>
                </>
              ) : (
                <>
                  <Button variant="text" className="text-text-muted hover:text-text" onClick={finish} data-testid="demo-skip">Skip tour</Button>
                  <Button variant="ghost" className="h-8 px-3" onClick={back} disabled={idx === 0} data-testid="demo-back">Back</Button>
                  <Button variant="primary" className="h-8 px-3" onClick={next} data-testid="demo-next">Next</Button>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
