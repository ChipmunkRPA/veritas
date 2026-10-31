/** Analysis progress (ARCHITECTURE §3.3, §7.3): Server-Sent Events with a polling fallback.
 *  - `stage_*` events patch `qk.stages(id)`, `progress` patches the analysis detail, `detector_finished` invalidates
 *    `qk.detectorRuns(id)`, and a terminal event invalidates the analysis tree.
 *  - The stream opens with an `events:<id>` resource token, because EventSource cannot send headers.
 *  - After two failed reconnects (or when the server refuses the stream) it polls `/progress?after=seq` every 2 s.
 *  - Nothing here carries a probability: the UI shows no provisional verdict before `fuse`. */
import { useEffect, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { apiClient, unwrap } from './client'
import { qk, isTerminal, type AnalysisDetail, type Stage } from './queries'
import { tokenCache } from './tokens'

export interface ProgressEvent {
  /** null for frames the server synthesises without a stored event (v1 analyses, deleted analyses). */
  seq: number | null
  ts: string | null
  type: string
  payload: Record<string, unknown>
}

export type Transport = 'idle' | 'connecting' | 'sse' | 'polling' | 'closed'

export const EVENT_TYPES = ['stage_started', 'stage_finished', 'detector_finished', 'progress', 'warning', 'partial_signal',
  'done', 'failed', 'cancelled'] as const
export const TERMINAL_EVENTS: ReadonlySet<string> = new Set(['done', 'failed', 'cancelled'])
export const POLL_MS = 2000
export const MAX_FAILED_RECONNECTS = 2
const CLOSED = 2 // EventSource.CLOSED

const str = (v: unknown): string | null => (typeof v === 'string' ? v : null)
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

export function parseFrame(data: string): ProgressEvent | null {
  try {
    const d = JSON.parse(data) as Record<string, unknown> | null
    if (!d || typeof d.type !== 'string') return null
    const payload = d.payload && typeof d.payload === 'object' && !Array.isArray(d.payload) ? d.payload as Record<string, unknown> : {}
    return { seq: num(d.seq), ts: str(d.ts), type: d.type, payload }
  } catch {
    return null
  }
}

/** Applies one `stage_started` / `stage_finished` event to the stage rows. Pure; other events return the rows unchanged. */
export function patchStages(rows: Stage[] | undefined, e: ProgressEvent): Stage[] | undefined {
  if (e.type !== 'stage_started' && e.type !== 'stage_finished') return rows
  const stage = str(e.payload.stage)
  if (!stage) return rows
  const next = rows ? [...rows] : []
  const i = next.findIndex((r) => r.stage === stage)
  const prev: Stage = i >= 0 ? next[i] : { stage, status: 'running', attempt: 1, started_at: e.ts, finished_at: null, error: null }
  const row: Stage = e.type === 'stage_started'
    ? { ...prev, status: 'running', started_at: prev.started_at ?? e.ts, finished_at: null }
    : { ...prev, status: str(e.payload.status) ?? 'done', finished_at: e.ts, error: str(e.payload.reason) ?? prev.error }
  if (i >= 0) next[i] = row
  else next.push(row)
  return next
}

export interface EventSourceLike {
  readyState: number
  onopen: ((ev: Event) => unknown) | null
  onerror: ((ev: Event) => unknown) | null
  addEventListener(type: string, listener: (ev: MessageEvent) => void): void
  close(): void
}

export interface StreamDeps {
  token: (resource: string) => Promise<string>
  open: (url: string) => EventSourceLike
  poll: (analysisId: string, after: number) => Promise<{ status: string; events: ProgressEvent[] }>
  setTimeout: (fn: () => void, ms: number) => unknown
  clearTimeout: (handle: unknown) => void
}

export class ProgressStream {
  lastSeq = 0
  private es: EventSourceLike | null = null
  private failures = 0
  private timer: unknown = null
  private stopped = false

  constructor(
    private readonly analysisId: string,
    private readonly deps: StreamDeps,
    private readonly onEvent: (e: ProgressEvent) => void,
    private readonly onTransport: (t: Transport) => void = () => undefined,
  ) {}

  async start(): Promise<void> {
    this.onTransport('connecting')
    let token: string
    try {
      token = await this.deps.token(`events:${this.analysisId}`)
    } catch {
      if (!this.stopped) this.poll()
      return
    }
    if (this.stopped) return
    const url = `/api/v2/analyses/${encodeURIComponent(this.analysisId)}/events?after=${this.lastSeq}&token=${encodeURIComponent(token)}`
    const es = this.deps.open(url)
    this.es = es
    es.onopen = () => {
      this.failures = 0
      this.onTransport('sse')
    }
    es.onerror = () => {
      if (this.stopped || this.es !== es) return
      this.failures += 1
      if (es.readyState === CLOSED || this.failures > MAX_FAILED_RECONNECTS) {
        es.close()
        this.es = null
        this.poll()
      } else {
        this.onTransport('connecting')
      }
    }
    for (const type of EVENT_TYPES) es.addEventListener(type, (m) => this.deliver(parseFrame(String(m.data))))
  }

  stop(): void {
    this.stopped = true
    this.es?.close()
    this.es = null
    if (this.timer !== null) this.deps.clearTimeout(this.timer)
    this.timer = null
  }

  private deliver(e: ProgressEvent | null): void {
    if (!e || this.stopped) return
    if (e.seq !== null) {
      if (e.seq <= this.lastSeq) return
      this.lastSeq = e.seq
    }
    this.onEvent(e)
    if (TERMINAL_EVENTS.has(e.type)) {
      this.stop()
      this.onTransport('closed')
    }
  }

  private poll(): void {
    this.onTransport('polling')
    const tick = async () => {
      this.timer = null
      if (this.stopped) return
      try {
        const body = await this.deps.poll(this.analysisId, this.lastSeq)
        for (const e of body.events) this.deliver(e)
        if (!this.stopped && body.events.length === 0 && isTerminal(body.status)) {
          this.deliver({ seq: null, ts: null, type: body.status, payload: { reason: 'No progress events were recorded' } })
        }
      } catch {
        // transient: the next tick retries
      }
      if (!this.stopped) this.timer = this.deps.setTimeout(() => void tick(), POLL_MS)
    }
    void tick()
  }
}

export const browserStreamDeps: StreamDeps = {
  token: async (resource) => (await tokenCache.get(resource)).token,
  open: (url) => new EventSource(url),
  poll: async (analysisId, after) => {
    const body = await unwrap(apiClient.GET('/api/v2/analyses/{aid}/progress', { params: { path: { aid: analysisId }, query: { after } } }))
    return { status: body.status, events: body.events.map((e) => ({ seq: e.seq, ts: e.ts, type: e.type, payload: e.payload })) }
  },
  setTimeout: (fn, ms) => window.setTimeout(fn, ms),
  clearTimeout: (handle) => window.clearTimeout(handle as number),
}

export interface EventsState {
  transport: Transport
  /** Sequence number of the last event applied. */
  lastSeq: number
  /** The terminal event type once one arrived. */
  terminal: string | null
}

/** Follows an analysis while `enabled` (normally: it is not terminal yet). The cache patches are the source of truth for
 *  views; the returned state describes the transport. */
export function useAnalysisEvents(analysisId: string, enabled: boolean, deps: StreamDeps = browserStreamDeps): EventsState {
  const qc = useQueryClient()
  const [state, setState] = useState<EventsState>({ transport: 'idle', lastSeq: 0, terminal: null })

  useEffect(() => {
    if (!enabled || !analysisId) return
    const stream = new ProgressStream(analysisId, deps, (e) => {
      if (e.type === 'stage_started' || e.type === 'stage_finished') {
        qc.setQueryData<Stage[]>(qk.stages(analysisId), (rows) => patchStages(rows, e))
      } else if (e.type === 'progress') {
        qc.setQueryData<AnalysisDetail>(qk.analysis(analysisId), (a) => a && {
          ...a, progress: num(e.payload.fraction) ?? a.progress, current_stage: str(e.payload.current_stage) ?? a.current_stage,
        })
      } else if (e.type === 'detector_finished') {
        void qc.invalidateQueries({ queryKey: qk.detectorRuns(analysisId) })
      }
      const terminal = TERMINAL_EVENTS.has(e.type)
      if (terminal) {
        // N-9b: a revision that finishes, fails or is cancelled changes the review state of the one it was queued to replace
        const base = qc.getQueryData<AnalysisDetail>(qk.analysis(analysisId))?.supersedes_id
        void qc.invalidateQueries({ queryKey: qk.analysis(analysisId) })
        if (base) void qc.invalidateQueries({ queryKey: qk.analysis(base), exact: true })
      }
      setState((s) => ({ ...s, lastSeq: stream.lastSeq, terminal: terminal ? e.type : s.terminal }))
    }, (transport) => setState((s) => ({ ...s, transport })))
    void stream.start()
    return () => stream.stop()
  }, [analysisId, enabled, deps, qc])

  return state
}
