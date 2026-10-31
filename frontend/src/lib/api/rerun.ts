/** Re-run with the pinned manifest and revision deltas (`POST /api/v2/analyses/{aid}/rerun`, `GET /analyses/{aid}/deltas`; PRD CC-03;
 *  ARCHITECTURE §5.2, §6.5). A re-run queues revision n+1 with this revision's profile, detector versions and weights, fuser artefact
 *  and policy thresholds; the server refuses with 409 and names every unavailable pin. Deltas compare a revision with the one it
 *  supersedes, for re-runs, reanalyses and refusions alike. */
import { useQuery } from '@tanstack/react-query'
import { apiClient, unwrap } from './client'
import { EXAMINER_ROLES } from './roles'
import type { components } from './schema'

export type RerunAccepted = components['schemas']['RerunAccepted']
export type RevisionDeltas = components['schemas']['RevisionDeltas']
export type DeltaRow = components['schemas']['DeltaRow']

export const deltasKey = (aid: string) => ['analysis', aid, 'deltas'] as const

export const MODE_WORD: Record<NonNullable<RevisionDeltas['mode']>, string> = {
  rerun: 'Re-run with the pinned manifest',
  reanalyse: 'Reanalysed with the current detectors and the chosen profile',
  refusion: 'Re-fused from the stored raw scores',
}

/** A revision that was cancelled or failed before it published: it supersedes nothing, and re-running it queues a new revision of
 *  the revision it would have replaced, with the pins it was queued with (K-2). */
export const unpublishedRevision = (status: string | null | undefined, supersedesId: string | null | undefined): boolean =>
  (status === 'cancelled' || status === 'failed') && !!supersedesId

/** Why this revision cannot be re-run from the browser, else null. The server refuses the same states and, beyond them, a pin that
 *  is no longer available. */
export function rerunRefusal(status: string | null | undefined, state: string | null | undefined, role: string | null | undefined,
  supersedesId?: string | null): string | null {
  if (status !== 'done' && !unpublishedRevision(status, supersedesId)) return 'The analysis has not finished; re-run it once it is done.'
  if (state === 'superseded') return 'A newer revision exists; re-run the latest revision.'
  if (role && !EXAMINER_ROLES.has(role)) return 'Re-running needs the examiner role.'
  return null
}

/** Re-running a signed-off revision supersedes it, so the server asks for a reason and for the revision's signer, its
 *  conclusion's author, a case owner or an administrator (the rule of a withdrawal). The dialog's note is that reason. */
export const rerunNeedsReason = (lockedAt: string | null | undefined): boolean => !!lockedAt

export function requestRerun(aid: string, note: string): Promise<RerunAccepted> {
  const text = note.trim()
  return unwrap(apiClient.POST('/api/v2/analyses/{aid}/rerun', { params: { path: { aid } }, body: text ? { note: text } : {} }))
}

/** "Run again" on a run that was cancelled before it published anything and replaces no revision (N-19a; browser QA
 *  "Workflow"): `POST /analyses/{aid}/reanalyse` queues the next revision of the same file in the same case, from the original
 *  the store kept (re-hashed first), with this run's profile. The cancelled run stays in the record. */
export function requestRunAgain(aid: string) {
  return unwrap(apiClient.POST('/api/v2/analyses/{aid}/reanalyse', { params: { path: { aid } }, body: {} }))
}

/** This revision against the one it supersedes; polls while the revision runs. */
export function useDeltas(aid: string, enabled: boolean) {
  return useQuery({
    queryKey: deltasKey(aid),
    enabled,
    queryFn: () => unwrap(apiClient.GET('/api/v2/analyses/{aid}/deltas', { params: { path: { aid } } })),
    refetchInterval: (query) => (query.state.data?.status === 'pending' ? 1500 : false),
  })
}

const FIELD_WORD: Record<string, string> = {
  raw_score: 'raw score', calibrated_probability: 'calibrated probability', code: 'decision code', probability_ai: 'probability',
  sampled: 'sampled frame indices', tracked: 'tracked frame indices',
}

export const deltaField = (field: string): string => FIELD_WORD[field] ?? field.replace(/_/g, ' ')

/** Scores and probabilities print at one precision (the tolerance is 1e-6), so 1 and 0.498145 do not sit in one column. */
const SCORE_FIELDS = new Set(['raw_score', 'calibrated_probability', 'probability_ai'])

export function deltaValue(v: unknown, field?: string): string {
  if (v == null) return 'none'
  if (typeof v === 'number') return Number.isInteger(v) && !(field && SCORE_FIELDS.has(field)) ? String(v) : v.toFixed(6)
  if (Array.isArray(v)) return v.length > 12 ? `${v.length} values: ${v.slice(0, 6).join(', ')} …` : v.join(', ')
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

/** A row whose value is the same before and after: shown behind "n values unchanged". A row outside the tolerance always shows. */
export const deltaUnchanged = (r: DeltaRow): boolean => r.ok && (r.delta == null ? JSON.stringify(r.before) === JSON.stringify(r.after) : r.delta === 0)

/** One sentence for the delta table: the CC-03 identity check, or why there are no deltas yet. */
export function deltaVerdict(d: RevisionDeltas): string {
  if (d.status !== 'ready') return d.detail ?? 'The deltas are computed once the revision has finished.'
  if (d.identical) {
    const frames = d.deltas.some((r) => r.item === 'frames')
    return `Identical: every signal${frames ? ', the decision and the frame indices' : ' and the decision'} agree within ${d.tolerance}.`
  }
  const outside = d.deltas.filter((r) => !r.ok).length
  return `Not identical: ${outside} of ${d.deltas.length} values changed or differ by more than ${d.tolerance}.`
}
