import { useEffect } from 'react'
import type { DemoItem } from '../lib/api'

/** A demo sample once `/api/demo/start` has assigned it an analysis. */
export interface DemoAnalysis extends DemoItem { id: string; status: string; decision_code?: string | null }

export type DemoPhase = 'idle' | 'preparing' | 'touring'

export interface DemoState {
  phase: DemoPhase
  items: DemoAnalysis[]
  step: number
  error: string | null
}

/** Page-side behaviour a step asks for once its anchor is on screen (delivered as a window event, see below). */
export type StepAction =
  | { type: 'expand-first-row' }
  | { type: 'expand-row'; match: string }
  | { type: 'select-first-ai-sentence' }

export interface TourStep {
  key: string
  /** Route the step lives on. */
  route: string
  /** CSS selector of the anchor (`[data-tour="…"]`); `fallback` is tried when the primary is absent. */
  selector: string
  fallback?: string
  title: string
  body: string
  /** Auto-advance after this many ms (default 8000). */
  ms?: number
  /** Skip silently when the anchor never appears (e.g. a signal the engine did not emit). */
  optional?: boolean
  action?: StepAction
  /** Index into `DemoState.items` whose analysis must be `done` before the step can show. */
  item?: number
  /** First step of a sample: the card shows the sample's blurb. */
  intro?: boolean
  final?: boolean
}

export const DEMO_STORAGE_KEY = 'veritas_demo'
export const DEMO_ACTION_EVENT = 'veritas:demo-action'
export const DEFAULT_STEP_MS = 8000

export function dispatchDemoAction(action: StepAction) {
  window.dispatchEvent(new CustomEvent<StepAction>(DEMO_ACTION_EVENT, { detail: action }))
}

/** Components that can be driven by the walkthrough subscribe here (SignalsLedger, ManuscriptView; the v1 EvidenceTable too). */
export function useDemoAction(handler: (action: StepAction) => void) {
  useEffect(() => {
    const on = (e: Event) => handler((e as CustomEvent<StepAction>).detail)
    window.addEventListener(DEMO_ACTION_EVENT, on)
    return () => window.removeEventListener(DEMO_ACTION_EVENT, on)
  }, [handler])
}

export const isDone = (s: string | undefined) => s === 'done'
export const isFailed = (s: string | undefined) => s === 'failed'
export const isSettled = (s: string | undefined) => isDone(s) || isFailed(s)
