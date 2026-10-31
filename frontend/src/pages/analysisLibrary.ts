import { apiClient, unwrap } from '../lib/api/client'
import type { CaseAnalysis } from '../lib/api/queries'
import { CAPTURE_ROW, DECLARATION_ROW, captureRow, declarationRow } from '../features/analysis/record'
import { listNumber, type ListNumber } from '../features/analysis/resultNumber'

export type LibraryAnalysis = CaseAnalysis
export type TypeFilter = 'all' | 'image' | 'video' | 'audio' | 'document'
export type StatusFilter = 'all' | 'done' | 'running' | 'failed' | 'cancelled'
export type VerdictFilter = 'all' | 'no-indications' | 'inconclusive' | 'indications' | 'not-assessable'
export type Sort = 'newest' | 'oldest' | 'indications-desc' | 'indications-asc' | 'name'
type DecisionCode = NonNullable<LibraryAnalysis['decision_code']>
type LibraryDecision = { code: DecisionCode; step: 0 | 1 | 2 | 3 | 4 | 5; label: string; legacy?: boolean }

/** UX §7.1 short labels. A decision is authoritative even when a retained estimate or legacy band disagrees. */
const DECISIONS: Record<DecisionCode, LibraryDecision> = {
  not_assessable: { code: 'not_assessable', step: 0, label: 'Not assessable' },
  no_ai_indicators_strong: { code: 'no_ai_indicators_strong', step: 1, label: 'No indications found' },
  no_ai_indicators: { code: 'no_ai_indicators', step: 2, label: 'No significant indications' },
  inconclusive: { code: 'inconclusive', step: 3, label: 'Inconclusive' },
  ai_supported: { code: 'ai_supported', step: 4, label: 'Indications' },
  ai_supported_strong: { code: 'ai_supported_strong', step: 5, label: 'Strong indications' },
}
const LEGACY_CODES: Record<string, DecisionCode> = {
  very_likely_authentic: 'no_ai_indicators_strong', likely_authentic: 'no_ai_indicators', uncertain: 'inconclusive',
  likely_ai: 'ai_supported', very_likely_ai: 'ai_supported_strong',
}

export function libraryDecision(a: LibraryAnalysis): LibraryDecision | null {
  if (a.status !== 'done') return null
  if (a.decision_code) {
    const decision = DECISIONS[a.decision_code] ?? null
    // N-19m (D3-01): a row that a record decides with a clean code is a capture's, not "No indications found"; its place is
    // the first step, where the record's value stands (the fallback's code alone gave the second)
    if (decision && captureRow(a)) return { ...decision, step: 1, label: CAPTURE_ROW }
    // N-19p (D6-01): a trusted declaration on the fallback is the declaration's row on its step, the fifth, not "Indications"
    return decision && declarationRow(a) ? { ...decision, step: 5, label: DECLARATION_ROW } : decision
  }
  // Only pre-v2 records may fall back to their stored band. Missing v2 decisions are never inferred from a score.
  const legacy = a.result_schema_version < 2 && a.band ? LEGACY_CODES[a.band] : undefined
  return legacy ? { ...DECISIONS[legacy], legacy: true } : null
}

export function libraryProbability(a: LibraryAnalysis): number | null {
  const decision = libraryDecision(a)
  const p = a.probability_ai
  return decision && decision.step > 0 && p != null && Number.isFinite(p) && p >= 0 && p <= 1 ? p : null
}

/** The number a row prints (D-68): `listNumber` on the row's decision. A record from before the decision codes takes the code its
 *  stored band maps to, and its number was never calibrated. Sorting still reads `libraryProbability`. */
export function libraryNumber(a: LibraryAnalysis): ListNumber {
  const d = libraryDecision(a)
  return listNumber({ ...a, decision_code: d?.code ?? null, fusion_mode: d?.legacy ? null : a.fusion_mode })
}

export function matchesLibrary(a: LibraryAnalysis, q: string, t: TypeFilter, s: StatusFilter, v: VerdictFilter): boolean {
  if (t !== 'all' && a.kind !== t) return false
  if (s === 'done' && a.status !== 'done') return false
  if (s === 'failed' && a.status !== 'failed') return false
  if (s === 'cancelled' && a.status !== 'cancelled') return false
  if (s === 'running' && a.status !== 'running' && a.status !== 'queued') return false
  if (v !== 'all') {
    const step = libraryDecision(a)?.step
    if (step == null) return false
    if (v === 'no-indications' && step !== 1 && step !== 2) return false
    if (v === 'inconclusive' && step !== 3) return false
    if (v === 'indications' && step !== 4 && step !== 5) return false
    if (v === 'not-assessable' && step !== 0) return false
  }
  const needle = q.toLowerCase()
  return !needle || a.filename.toLowerCase().includes(needle) || a.sha256.toLowerCase().includes(needle)
}

export function sortLibrary(items: LibraryAnalysis[], sort: Sort): LibraryAnalysis[] {
  const time = (a: LibraryAnalysis) => (a.created_at ? Date.parse(a.created_at) || 0 : 0)
  return [...items].sort((a, b) => {
    if (sort === 'oldest') return time(a) - time(b)
    if (sort === 'name') return a.filename.localeCompare(b.filename)
    if (sort === 'newest') return time(b) - time(a)
    const da = libraryDecision(a), db = libraryDecision(b)
    // Unassessable and unrecorded decisions have no strength position. Done rows precede work without a completed assessment.
    const group = (row: LibraryAnalysis, d: LibraryDecision | null) => d && d.step > 0 ? 0 : row.status === 'done' ? 1 : 2
    const groups = group(a, da) - group(b, db)
    if (groups) return groups
    if (da && db && da.step > 0 && db.step > 0) {
      const direction = sort === 'indications-asc' ? 1 : -1
      if (da.step !== db.step) return direction * (da.step - db.step)
      const pa = libraryProbability(a), pb = libraryProbability(b)
      // Missing estimates sort last only within the same decision, never among unfinished work.
      if (pa == null && pb != null) return 1
      if (pa != null && pb == null) return -1
      if (pa != null && pb != null && pa !== pb) return direction * (pa - pb)
    }
    return time(b) - time(a)
  })
}

export const libraryPollInterval = (items: LibraryAnalysis[] | undefined): number =>
  items?.some((a) => a.status === 'queued' || a.status === 'running') ? 2000 : 15000

const PAGE_SIZE = 200
const MAX_PAGES = 25
/** Read cursor pages with a visible bound, preserving search/sort across loaded pages and avoiding a runaway cursor. */
export async function loadLibrary(): Promise<{ items: LibraryAnalysis[]; hasMore: boolean }> {
  const items: LibraryAnalysis[] = []
  const seen = new Set<string>()
  let cursor: string | undefined
  for (let i = 0; i < MAX_PAGES; i++) {
    const page = await unwrap(apiClient.GET('/api/v2/analyses', { params: { query: { limit: PAGE_SIZE, cursor } } }))
    items.push(...page.items)
    if (!page.next_cursor) return { items, hasMore: false }
    if (seen.has(page.next_cursor)) throw new Error('The analysis list could not be fully loaded. Retry to refresh it.')
    cursor = page.next_cursor
    seen.add(cursor)
  }
  return { items, hasMore: true }
}
