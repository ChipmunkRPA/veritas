import type { CaseAnalysis } from '../../lib/api/queries'

/** A supersede chain (ARCHITECTURE §4.7 `supersedes_id`, §7.2; UX §2.2): the latest revision, then the revisions it
 *  superseded from newest to oldest. `unpublished` are newer revisions of a chain member that were cancelled or failed before
 *  they published; they supersede nothing (K-2), newest first. */
export interface Chain {
  latest: CaseAnalysis
  superseded: CaseAnalysis[]
  unpublished: CaseAnalysis[]
}

/** Statuses of a revision that never published and never will (backend `UNPUBLISHED_TERMINAL`). */
const UNPUBLISHED = new Set(['cancelled', 'failed'])

/** Whether a revision with this status can supersede the one it names: one that was cancelled or failed before it published
 *  replaces nothing (K-2). */
export const supersedes = (status: string | null | undefined): boolean => !UNPUBLISHED.has(status ?? '')

const time = (a: CaseAnalysis) => (a.created_at ? Date.parse(a.created_at) : 0)
const newestFirst = (x: CaseAnalysis, y: CaseAnalysis) => time(y) - time(x) || (x.id < y.id ? 1 : x.id > y.id ? -1 : 0)

/** The supersede chains of a set of analyses, the most recently created latest revision first. A revision is replaced only by a
 *  successor that is queued, running or published; a revision whose predecessor is not in the set starts its chain; the walk
 *  stops on a repeated id. A cancelled or failed successor is listed with the chain of the revision it would have replaced. */
export function supersedeChains(analyses: readonly CaseAnalysis[]): Chain[] {
  const byId = new Map(analyses.map((a) => [a.id, a]))
  const dead = (a: CaseAnalysis) => UNPUBLISHED.has(a.status) && a.supersedes_id != null && byId.has(a.supersedes_id)
  const live = analyses.filter((a) => !dead(a))
  const replaced = new Set(live.map((a) => a.supersedes_id).filter((x): x is string => x != null && byId.has(x)))
  const chains: Chain[] = []
  for (const head of live) {
    if (replaced.has(head.id)) continue
    const superseded: CaseAnalysis[] = []
    const seen = new Set([head.id])
    let prev = head.supersedes_id ? byId.get(head.supersedes_id) : undefined
    while (prev && !seen.has(prev.id) && !dead(prev)) {
      superseded.push(prev)
      seen.add(prev.id)
      prev = prev.supersedes_id ? byId.get(prev.supersedes_id) : undefined
    }
    const members = new Set([head.id, ...superseded.map((a) => a.id)])
    const unpublished = analyses.filter((a) => dead(a) && members.has(a.supersedes_id!)).sort(newestFirst)
    chains.push({ latest: head, superseded, unpublished })
  }
  return chains.sort((x, y) => newestFirst(x.latest, y.latest))
}

/** Supersede chains per media object (an exhibit of the case can carry several independent chains). */
export function chainsByMedia(analyses: readonly CaseAnalysis[]): Map<string, Chain[]> {
  const out = new Map<string, Chain[]>()
  for (const chain of supersedeChains(analyses)) {
    const key = chain.latest.media_id ?? ''
    out.set(key, [...(out.get(key) ?? []), chain])
  }
  return out
}
