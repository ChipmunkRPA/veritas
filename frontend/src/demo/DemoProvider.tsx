import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import { api } from '../lib/api'
import { buildSteps } from './steps'
import { DEMO_STORAGE_KEY, isDone, isFailed, isSettled, type DemoAnalysis, type DemoState, type TourStep } from './types'

const IDLE: DemoState = { phase: 'idle', items: [], step: 0, error: null }
const POLL_MS = 1500

interface DemoContextValue {
  state: DemoState
  steps: TourStep[]
  /** Start (or resume) the walkthrough. */
  start: () => void
  next: () => void
  back: () => void
  finish: () => void
  /** Jump to a step index. */
  goTo: (i: number) => void
  /** Leave a step whose anchor never appeared, in the direction of the last move: Back from the step after it goes on to the
   *  step before it, and does not bounce forward again (N-17; browser QA P1-5). */
  skip: () => void
  active: boolean
  /** The tour was restored from this tab's storage and no step has been taken since: the page the visitor opened stays (L-7). */
  resumed: boolean
}

const DemoContext = createContext<DemoContextValue | null>(null)

function readStored(): DemoState {
  try {
    const raw = sessionStorage.getItem(DEMO_STORAGE_KEY)
    if (!raw) return IDLE
    const s = JSON.parse(raw) as Partial<DemoState>
    if (s.phase !== 'preparing' && s.phase !== 'touring') return IDLE
    if (!Array.isArray(s.items) || !s.items.length) return IDLE
    return { phase: s.phase, items: s.items as DemoAnalysis[], step: typeof s.step === 'number' ? s.step : 0, error: null }
  } catch { return IDLE }
}

function persist(s: DemoState) {
  try {
    if (s.phase === 'idle') sessionStorage.removeItem(DEMO_STORAGE_KEY)
    else sessionStorage.setItem(DEMO_STORAGE_KEY, JSON.stringify(s))
  } catch { /* storage unavailable: the demo still runs for this page load */ }
}

/**
 * Holds the guided-demo state (persisted in sessionStorage so a reload mid-tour resumes), polls the sample analyses
 * until they settle, and exposes navigation. The coach-mark UI lives in `DemoOverlay`.
 */
export function DemoProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<DemoState>(readStored)
  const [resumed, setResumed] = useState(() => readStored().phase === 'touring')
  const nav = useNavigate()
  const loc = useLocation()
  const qc = useQueryClient()
  const starting = useRef(false)

  useEffect(() => { persist(state) }, [state])

  const steps = useMemo(() => (state.items.length ? buildSteps(state.items) : []), [state.items])

  const start = useCallback(() => {
    if (starting.current) return
    setResumed(false)
    setState((s) => (s.phase === 'idle' ? { ...s, error: null } : s))
    if (state.phase !== 'idle' && !state.error) return // already running: the overlay resumes where it is
    starting.current = true
    api.demoStart()
      .then((r) => {
        const items = r.items.filter((it): it is DemoAnalysis => typeof it.id === 'string' && !!it.id).map((it) => ({ ...it, status: it.status ?? 'queued' }))
        if (!items.length) throw new Error('The demo returned no samples')
        setState({ phase: 'preparing', items, step: 0, error: null })
        qc.invalidateQueries({ queryKey: ['analyses'] })
      })
      .catch((e: Error) => setState({ ...IDLE, error: e.message || 'Could not start the demo' }))
      .finally(() => { starting.current = false })
  }, [state.phase, state.error, qc])

  // Deep links: `/demo` (redirected to `/?demo=1` by the router) or `/?demo=1`. Cleared from the URL once consumed.
  useEffect(() => {
    const params = new URLSearchParams(loc.search)
    if (params.get('demo') === '1') {
      params.delete('demo')
      nav({ pathname: loc.pathname, search: params.toString() ? `?${params}` : '' }, { replace: true })
      start()
    }
  }, [loc.search, loc.pathname, nav, start])

  // Poll unsettled analyses while the demo is live.
  useEffect(() => {
    if (state.phase === 'idle') return
    // a done sample without its decision code is fetched once more (L-2: the tour copy follows the decision)
    const pending = state.items.filter((it) => !isSettled(it.status) || (isDone(it.status) && it.decision_code === undefined))
    if (!pending.length) return
    let cancelled = false
    type Update = { id: string; status: string; decision_code: string | null | undefined }
    const poll = async (it: DemoAnalysis): Promise<Update> => {
      const status = isSettled(it.status) ? it.status : (await api.get(it.id)).status
      const decision_code = isDone(status) ? await api.decisionCode(it.id).catch(() => null) : undefined
      return { id: it.id, status, decision_code }
    }
    const tick = async () => {
      const updates = await Promise.all(pending.map((it) => poll(it).catch(() => null)))
      if (cancelled) return
      const changed = updates.filter((u): u is Update => !!u)
      if (!changed.length) return
      setState((s) => {
        const items = s.items.map((it) => {
          const u = changed.find((c) => c.id === it.id)
          return u && (u.status !== it.status || u.decision_code !== it.decision_code) ? { ...it, status: u.status, decision_code: u.decision_code } : it
        })
        return items.some((it, i) => it !== s.items[i]) ? { ...s, items } : s
      })
      if (changed.some((c) => isSettled(c.status))) qc.invalidateQueries({ queryKey: ['analyses'] })
    }
    const t = window.setInterval(tick, POLL_MS)
    void tick()
    return () => { cancelled = true; window.clearInterval(t) }
  }, [state.phase, state.items, qc])

  // Preparing → touring as soon as the first step's sample is done (the overlay waits per step after that).
  useEffect(() => {
    if (state.phase !== 'preparing' || !steps.length) return
    const first = steps[0]
    const item = first.item != null ? state.items[first.item] : null
    if (!item || isDone(item.status)) setState((s) => ({ ...s, phase: 'touring', step: 0 }))
    else if (state.items.every((it) => isFailed(it.status))) setState({ ...IDLE, error: 'All demo analyses failed' })
  }, [state.phase, state.items, steps])

  const finish = useCallback(() => setState(IDLE), [])
  const direction = useRef<1 | -1>(1)
  const goTo = useCallback((i: number) => {
    setResumed(false)
    setState((s) => {
      if (i !== s.step) direction.current = i < s.step ? -1 : 1
      return i >= steps.length ? IDLE : { ...s, step: Math.max(0, i) }
    })
  }, [steps.length])
  const next = useCallback(() => goTo(state.step + 1), [goTo, state.step])
  const back = useCallback(() => goTo(state.step - 1), [goTo, state.step])
  // the first step has nothing before it: from there a missing anchor is skipped forwards
  const skip = useCallback(() => goTo(state.step + (state.step === 0 ? 1 : direction.current)), [goTo, state.step])

  const value = useMemo<DemoContextValue>(() => ({ state, steps, start, next, back, finish, goTo, skip, active: state.phase !== 'idle', resumed }),
    [state, steps, start, next, back, finish, goTo, skip, resumed])
  return <DemoContext.Provider value={value}>{children}</DemoContext.Provider>
}

export function useDemo(): DemoContextValue {
  const v = useContext(DemoContext)
  if (!v) throw new Error('useDemo must be used inside DemoProvider')
  return v
}
