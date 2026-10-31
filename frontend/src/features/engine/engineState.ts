/** What `/api/ready` tells this caller (the readiness probe): without credentials the status alone; with them the checks
 *  behind it and what the worker that serves the queue runs on. A 503 is an answer (`not_ready`), not a failure:
 *  only no answer at all is "offline". */
export interface EngineCheck { ok: boolean; detail?: string }
/** N-14a: the checks of the serving worker's default profile that its model memory budget cannot run (`capacity.summary`). */
export interface EngineCapacity {
  profile: string
  device?: string | null
  budget_mb: number
  detectors?: number
  cannot_run: { detector_id: string; need_mb: number | null; reason: string | null }[]
  largest_need_mb: number | null
  smallest_budget_mb?: number
}
export interface EngineState {
  status: string
  checks?: { database?: EngineCheck; storage?: EngineCheck; worker?: EngineCheck }
  version?: string | null
  device?: string | null
  profile?: string | null
  queue_depth?: number | null
  capacity?: EngineCapacity | null
}

/** The body of a readiness answer, or null when the response is not one (a proxy's error page, another status). */
export async function readEngineState(r: Response): Promise<EngineState | null> {
  if (!r.ok && r.status !== 503) return null
  try {
    const body: unknown = await r.json()
    return body !== null && typeof body === 'object' && typeof (body as EngineState).status === 'string' ? (body as EngineState) : null
  } catch {
    return null
  }
}

export const engineReady = (state: EngineState | undefined): boolean => state?.status === 'ready'

/** One sentence for the strip under the header when the engine answers and is not ready; null when it is ready or unknown.
 *  The first failing check is named, in the order a person can act on: storage, the analysis worker, the database. */
export function engineNotice(state: EngineState | undefined): string | null {
  if (!state || state.status === 'ready') return null
  const failed = (name: 'database' | 'storage' | 'worker') => state.checks?.[name]?.ok === false
  if (failed('storage')) return 'Engine not ready. Evidence storage is nearly full; uploads are refused until an administrator frees space.'
  if (failed('worker')) return 'Engine not ready. No analysis worker is running; new analyses wait in the queue.'
  if (failed('database')) return 'Engine not ready. The database is not answering; nothing can be stored for now.'
  return 'Engine not ready. Analyses may wait or be refused; an administrator can see the reason.'
}

const mb = (n: number): string => n.toLocaleString('en-US')

/** One sentence for an administrator when checks of the default profile can never run on this server (N-14a); null when all
 *  can, when the answer names no capacity (no credentials, a worker that has not planned yet) or when it is not well formed.
 *  It counts; it does not name the checks: the server's table does (`models plan`). */
export function engineCapacityNotice(state: EngineState | undefined): string | null {
  const c = state?.capacity
  if (!c || !Array.isArray(c.cannot_run) || c.cannot_run.length === 0) return null
  if (typeof c.profile !== 'string' || typeof c.budget_mb !== 'number') return null
  const n = c.cannot_run.length
  const need = typeof c.largest_need_mb === 'number' ? `, largest need ${mb(c.largest_need_mb)} MB` : ''
  return `${n} ${n === 1 ? 'check' : 'checks'} of the ${c.profile} profile cannot run on this server: memory budget ${mb(c.budget_mb)} MB${need}.`
}
