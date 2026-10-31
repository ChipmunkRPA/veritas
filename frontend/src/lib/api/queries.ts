/** Query-key factory and v2 read hooks (ARCHITECTURE §7.3).
 *  - Every per-analysis key starts with `qk.analysis(id)`, so the `done` event invalidates the whole tree with one call.
 *  - Analysis JSON is fresh (`staleTime: 0`) while the analysis runs and immutable (`Infinity`) once it is terminal. The one
 *    part of the detail that changes after that is its review block, which `useAnalysis` keeps fresh by itself (N-9b).
 *  - Per-analysis collections read the first page of up to 200 items; the case views read every page (E-6). */
import { useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { ApiError, apiClient, unwrap } from './client'
import { sentencesInWords } from '../sliceWords'
import { withRecord } from '../../features/analysis/record'

const PAGE = 200
/** Upper bound on the pages a case view reads (25 × 200 items). */
const MAX_PAGES = 25

export const isTerminal = (status: string | null | undefined): boolean =>
  status === 'done' || status === 'failed' || status === 'cancelled'

/** Results are immutable per analysis revision once it is terminal. */
export const staleFor = (status: string | null | undefined): number => (isTerminal(status) ? Infinity : 0)

/** How long the review block of a terminal analysis counts as fresh. The result is immutable, the review block is not: a re-run
 *  supersedes the revision, a cancelled or failed re-run gives it back, a sign-off locks it, and any of these can happen in
 *  another tab or by another person. A cached "superseded" must not outlive the state on the server by more than this
 *  (N-9b; code review 7): a view that mounts after it reads the detail again, and a page that stays open reads it again at
 *  this interval (N-10b; code re-verification R2-08). What this tab does itself invalidates the detail at once, and so does a
 *  write the server refuses with 409 (`queryClient.ts`). */
export const REVIEW_STALE_MS = 30_000

export const qk = {
  /** v1 analyses list (legacy dashboard). */
  analyses: () => ['analyses'] as const,
  /** Root of the per-analysis tree. */
  analysis: (id: string) => ['analysis', id] as const,
  /** When the review block of a terminal analysis was last read (outside the tree: invalidating the tree reads the detail once). */
  reviewFreshness: (id: string) => ['review-freshness', id] as const,
  stages: (id: string) => ['analysis', id, 'stages'] as const,
  detectorRuns: (id: string) => ['analysis', id, 'detector-runs'] as const,
  findings: (id: string) => ['analysis', id, 'findings'] as const,
  signals: (id: string) => ['analysis', id, 'signals'] as const,
  exhibits: (id: string) => ['analysis', id, 'exhibits'] as const,
  /** One exhibit's full payload (tables, regions), under its analysis so the tree invalidates together. */
  exhibit: (id: string, exhibitId: string) => ['analysis', id, 'exhibit', exhibitId] as const,
  /** The pts table, sampling plan, shot cuts and lane index of a video analysis (immutable once the analysis is done). */
  frames: (id: string) => ['analysis', id, 'frames'] as const,
  /** The examiner layer (mutable: never cached as immutable). */
  annotations: (id: string) => ['analysis', id, 'annotations'] as const,
  /** The current examiner conclusion (RW-03) and the sign-off records (RW-04). */
  conclusion: (id: string) => ['analysis', id, 'conclusion'] as const,
  signoffs: (id: string) => ['analysis', id, 'signoffs'] as const,
  /** Model cards of the manifest (not per analysis). */
  models: () => ['models'] as const,
  /** One measurement record of a detector (Markdown text). */
  modelRecord: (detectorId: string, recordId: string) => ['models', detectorId, 'records', recordId] as const,
  /** The v1 view of the same analysis, read by the `legacy/` components. */
  legacy: (id: string) => ['analysis', id, 'legacy'] as const,
  downloadToken: (id: string) => ['download-token', id] as const,
  cases: () => ['cases'] as const,
  case: (id: string) => ['case', id] as const,
  caseItems: (id: string) => ['case', id, 'items'] as const,
  /** Every analysis revision of the case (supersede chains). */
  caseAnalyses: (id: string) => ['case', id, 'analyses'] as const,
  members: (id: string) => ['case', id, 'members'] as const,
  /** Who the owner or an administrator can add to the case (D-67). */
  memberCandidates: (id: string) => ['case', id, 'member-candidates'] as const,
  /** The case's audit events in seq order (the custody log). */
  custody: (id: string) => ['case', id, 'custody'] as const,
  me: () => ['me'] as const,
  token: (resource: string) => ['token', resource] as const,
}

/** Every page of a cursor collection, up to `MAX_PAGES`. */
async function allPages<T>(page: (cursor: string | undefined) => Promise<{ items: T[]; next_cursor?: string | null }>, requireComplete = false): Promise<T[]> {
  const items: T[] = []
  let cursor: string | undefined
  const seen = new Set<string>()
  for (let i = 0; i < MAX_PAGES; i++) {
    const p = await page(cursor)
    items.push(...p.items)
    if (!p.next_cursor) return items
    if (requireComplete && seen.has(p.next_cursor)) throw new Error('The catalogue returned a repeated page.')
    seen.add(p.next_cursor)
    cursor = p.next_cursor
  }
  if (requireComplete) throw new Error('The full catalogue could not be read.')
  return items
}

/** A 4xx other than 408 / 429 will not change on retry. */
const retryServerErrors = (failures: number, error: Error) =>
  failures < 1 && !(error instanceof ApiError && error.status >= 400 && error.status < 500 && error.status !== 408 && error.status !== 429)

// N-19d: a sentence the engine stored may hold the id of a validation slice; the page reads it in words (`sliceWords.ts`).
// N-19h: the decision carries the record that decides, also for a result stored before it named one (`record.ts`)
const getAnalysis = async (id: string) => withRecord(sentencesInWords(await unwrap(apiClient.GET('/api/v2/analyses/{aid}', { params: { path: { aid: id } } }))))
const getStages = async (id: string) =>
  (await unwrap(apiClient.GET('/api/v2/analyses/{aid}/stages', { params: { path: { aid: id } } }))).items
const getDetectorRuns = async (id: string) =>
  (await unwrap(apiClient.GET('/api/v2/analyses/{aid}/detector-runs', { params: { path: { aid: id }, query: { limit: PAGE } } }))).items
const getFindings = async (id: string) =>
  sentencesInWords((await unwrap(apiClient.GET('/api/v2/analyses/{aid}/findings', { params: { path: { aid: id }, query: { limit: PAGE } } }))).items)
const getSignals = async (id: string) =>
  (await unwrap(apiClient.GET('/api/v2/analyses/{aid}/signals', { params: { path: { aid: id }, query: { limit: PAGE } } }))).items
const getExhibits = async (id: string) =>
  (await unwrap(apiClient.GET('/api/v2/analyses/{aid}/exhibits', { params: { path: { aid: id }, query: { limit: PAGE } } }))).items
const getExhibit = (id: string) => unwrap(apiClient.GET('/api/v2/exhibits/{exhibit_id}', { params: { path: { exhibit_id: id } } }))
const getFrames = (id: string) => unwrap(apiClient.GET('/api/v2/analyses/{aid}/frames', { params: { path: { aid: id } } }))
const getAnnotations = async (id: string) =>
  (await unwrap(apiClient.GET('/api/v2/analyses/{aid}/annotations', { params: { path: { aid: id } } }))).items
const getConclusion = async (id: string) => {
  try {
    return await unwrap(apiClient.GET('/api/v2/analyses/{aid}/conclusion', { params: { path: { aid: id } } }))
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) return null  // no conclusion recorded yet
    throw e
  }
}
const getSignoffs = async (id: string) =>
  (await unwrap(apiClient.GET('/api/v2/analyses/{aid}/signoffs', { params: { path: { aid: id } } }))).items
const getModelCards = () => allPages((cursor) => unwrap(apiClient.GET('/api/v2/models', { params: { query: { limit: PAGE, cursor } } })), true)
const getCases = () => allPages((cursor) => unwrap(apiClient.GET('/api/v2/cases', { params: { query: { limit: PAGE, cursor } } })))
const getCase = (id: string) => unwrap(apiClient.GET('/api/v2/cases/{case_id}', { params: { path: { case_id: id } } }))
const getCaseItems = (id: string) =>
  allPages((cursor) => unwrap(apiClient.GET('/api/v2/cases/{case_id}/items', { params: { path: { case_id: id }, query: { limit: PAGE, cursor } } })))
const getCaseAnalyses = (id: string) =>
  allPages((cursor) => unwrap(apiClient.GET('/api/v2/analyses', { params: { query: { case_id: id, limit: PAGE, cursor } } })))
const getMembers = async (id: string) =>
  (await unwrap(apiClient.GET('/api/v2/cases/{case_id}/members', { params: { path: { case_id: id } } }))).items
const getCustody = async (id: string) => {
  const page = (after: number | undefined) => unwrap(apiClient.GET('/api/v2/cases/{case_id}/custody', { params: { path: { case_id: id }, query: { after_seq: after, limit: PAGE } } }))
  const first = await page(undefined)
  const items = [...first.items]
  let next = first.next_after_seq
  for (let i = 1; i < MAX_PAGES && next != null; i++) {
    const p = await page(next)
    items.push(...p.items)
    next = p.next_after_seq
  }
  return items
}
const getMemberCandidates = async (id: string) =>
  (await unwrap(apiClient.GET('/api/v2/cases/{case_id}/members/candidates', { params: { path: { case_id: id } } }))).items
const getMe = () => unwrap(apiClient.GET('/api/v2/me'))

// Response types as openapi-fetch delivers them (its readable view of the schema, where tuples are arrays).
export type AnalysisDetail = Awaited<ReturnType<typeof getAnalysis>>
export type Stage = Awaited<ReturnType<typeof getStages>>[number]
export type DetectorRun = Awaited<ReturnType<typeof getDetectorRuns>>[number]
export type Finding = Awaited<ReturnType<typeof getFindings>>[number]
export type SignalV2 = Awaited<ReturnType<typeof getSignals>>[number]
export type ExhibitMeta = Awaited<ReturnType<typeof getExhibits>>[number]
export type ExhibitPayload = Awaited<ReturnType<typeof getExhibit>>
export type TableExhibit = Extract<ExhibitPayload, { kind: 'table' }>
export type FrameTable = Awaited<ReturnType<typeof getFrames>>
export type ModelCard = Awaited<ReturnType<typeof getModelCards>>[number]
export type Annotation = Awaited<ReturnType<typeof getAnnotations>>[number]
export type Conclusion = NonNullable<Awaited<ReturnType<typeof getConclusion>>>
export type Signoff = Awaited<ReturnType<typeof getSignoffs>>[number]
export type Case = Awaited<ReturnType<typeof getCase>>
export type CaseItem = Awaited<ReturnType<typeof getCaseItems>>[number]
export type CaseAnalysis = Awaited<ReturnType<typeof getCaseAnalyses>>[number]
export type Member = Awaited<ReturnType<typeof getMembers>>[number]
export type AuditEvent = Awaited<ReturnType<typeof getCustody>>[number]
export type Me = Awaited<ReturnType<typeof getMe>>

export function useAnalysis(id: string) {
  const qc = useQueryClient()
  const detail = useQuery({
    queryKey: qk.analysis(id),
    queryFn: async () => {
      const fresh = await getAnalysis(id)
      qc.setQueryData(qk.reviewFreshness(id), fresh.review?.state ?? null)  // read just now: the review block is as fresh
      return fresh
    },
    staleTime: (q) => staleFor(q.state.data?.status),
    enabled: id !== '',
  })
  // N-9b: the detail of a terminal analysis is never stale, but its review block is after `REVIEW_STALE_MS`. This query reads the
  // detail again when a view mounts after that, and writes it into the detail's cache entry, the one the views read. A read
  // that fails changes nothing: the cached result stays on screen.
  // N-10b (code re-verification R2-08): a page that stays open has no view that mounts, so the query also runs at that
  // interval while a view is mounted and the tab is visible (a hidden tab reads when it is shown again and the interval is up).
  useQuery({
    queryKey: qk.reviewFreshness(id),
    queryFn: async () => {
      const fresh = await getAnalysis(id)
      qc.setQueryData(qk.analysis(id), fresh)
      return fresh.review?.state ?? null
    },
    initialData: null,  // fresh from the moment the view first asks: the detail is being read, or was seeded, right now
    staleTime: REVIEW_STALE_MS,
    refetchInterval: REVIEW_STALE_MS,
    enabled: id !== '' && isTerminal(detail.data?.status),
    retry: false,
  })
  return detail
}

/** Analysis details (with their review block) of several analyses, sharing the workspace cache entries. */
export function useAnalysisDetails(ids: readonly string[]) {
  return useQueries({
    queries: ids.map((id) => ({ queryKey: qk.analysis(id), queryFn: () => getAnalysis(id), staleTime: 0 })),
    combine: (results) => new Map(results.flatMap((r) => (r.data ? [[r.data.id, r.data] as const] : []))),
  })
}

/** Stage rows; `useAnalysisEvents` patches them as `stage_*` events arrive. */
export function useStages(id: string, enabled = true) {
  return useQuery({ queryKey: qk.stages(id), queryFn: () => getStages(id), enabled: enabled && id !== '' })
}

export function useDetectorRuns(id: string, enabled = true) {
  return useQuery({ queryKey: qk.detectorRuns(id), queryFn: () => getDetectorRuns(id), enabled: enabled && id !== '' })
}

/** Findings exist once `synthesise` has run, so callers enable this for done analyses only. The examiner layer is joined into
 *  each finding at read time, so review writes invalidate this key. */
export function useFindings(id: string, enabled = true) {
  return useQuery({ queryKey: qk.findings(id), queryFn: () => getFindings(id), enabled: enabled && id !== '', staleTime: Infinity })
}

/** The machine layer, read once the analysis is done (ids S01… per D-26). */
export function useSignals(id: string, enabled = true) {
  return useQuery({ queryKey: qk.signals(id), queryFn: () => getSignals(id), enabled: enabled && id !== '', staleTime: Infinity })
}

export function useExhibits(id: string, enabled = true) {
  return useQuery({ queryKey: qk.exhibits(id), queryFn: () => getExhibits(id), enabled: enabled && id !== '', staleTime: Infinity })
}

/** Full payloads of the given exhibits (the collection route carries metadata only). Results are immutable. */
export function useExhibitPayloads(id: string, exhibitIds: readonly string[], enabled = true) {
  return useQueries({
    queries: exhibitIds.map((x) => ({ queryKey: qk.exhibit(id, x), queryFn: () => getExhibit(x), enabled: enabled && id !== '', staleTime: Infinity })),
    combine: (results) => ({
      data: results.map((r) => r.data).filter((x): x is ExhibitPayload => x != null),
      isPending: results.some((r) => r.isPending),
      error: results.find((r) => r.error)?.error ?? null,
      refetch: () => { results.forEach((r) => { if (r.error) void r.refetch() }) },
    }),
  })
}

/** The pts table, sampled frames, shot cuts and lane index of a done video analysis (E-5). A 404 (no frame table) is final. */
export function useFrames(id: string, enabled = true) {
  return useQuery({ queryKey: qk.frames(id), queryFn: () => getFrames(id), enabled: enabled && id !== '', staleTime: Infinity, retry: retryServerErrors })
}

/** The examiner layer's annotations (C-4), oldest first; invalidated after each write. */
export function useAnnotations(id: string, enabled = true) {
  return useQuery({ queryKey: qk.annotations(id), queryFn: () => getAnnotations(id), enabled: enabled && id !== '' })
}

/** The current conclusion, or null before one is recorded. */
export function useConclusion(id: string, enabled = true) {
  return useQuery({ queryKey: qk.conclusion(id), queryFn: () => getConclusion(id), enabled: enabled && id !== '' })
}

export function useSignoffs(id: string, enabled = true) {
  return useQuery({ queryKey: qk.signoffs(id), queryFn: () => getSignoffs(id), enabled: enabled && id !== '' })
}

/** The complete models catalogue, shared by availability guidance and model cards. A detector without a card is absent, so a sheet
 *  never asks for a card that does not exist (no 404 in the console). */
export function useModelCards(enabled = true) {
  // installed / loaded state changes as models are fetched and warmed: fresh for 30 s, not for the session
  return useQuery({ queryKey: qk.models(), queryFn: getModelCards, enabled, staleTime: 30_000 })
}

const getModelRecord = (detectorId: string, recordId: string) =>
  unwrap(apiClient.GET('/api/v2/models/{detector_id}/records/{record_id}', { params: { path: { detector_id: detectorId, record_id: recordId } }, parseAs: 'text' }))

/** One measurement record of a detector as Markdown text, read when its disclosure first opens. */
export function useModelRecord(detectorId: string, recordId: string) {
  return useQuery({ queryKey: qk.modelRecord(detectorId, recordId), queryFn: () => getModelRecord(detectorId, recordId), staleTime: 5 * 60_000, retry: retryServerErrors })
}

const getAwaitingSecondReview = () =>
  allPages((cursor) => unwrap(apiClient.GET('/api/v2/analyses', { params: { query: { awaiting_second_review: true, limit: PAGE, cursor } } })))

/** N-19a (browser QA P1-12): the revisions an examiner has signed, on cases that require a second review, that no peer has
 *  reviewed yet, among the cases the caller can see: a reviewer's queue. */
export function useAwaitingSecondReview() {
  return useQuery({ queryKey: ['analyses', 'awaiting-second-review'] as const, queryFn: getAwaitingSecondReview })
}

export function useCases() {
  return useQuery({ queryKey: qk.cases(), queryFn: getCases })
}

export function useCase(id: string) {
  return useQuery({ queryKey: qk.case(id), queryFn: () => getCase(id), enabled: id !== '', retry: retryServerErrors })
}

export function useCaseItems(id: string) {
  return useQuery({ queryKey: qk.caseItems(id), queryFn: () => getCaseItems(id), enabled: id !== '' })
}

/** Every revision of the case's analyses; polled every 3 s while one of them is still running. */
export function useCaseAnalyses(id: string) {
  return useQuery({
    queryKey: qk.caseAnalyses(id),
    queryFn: () => getCaseAnalyses(id),
    enabled: id !== '',
    refetchInterval: (q) => (q.state.data?.some((a) => !isTerminal(a.status)) ? 3000 : false),
  })
}

export function useMembers(id: string) {
  return useQuery({ queryKey: qk.members(id), queryFn: () => getMembers(id), enabled: id !== '' })
}

/** The enabled principals that are not members of the case yet (the owner or an administrator; D-67). */
export function useMemberCandidates(id: string, enabled: boolean) {
  return useQuery({ queryKey: qk.memberCandidates(id), queryFn: () => getMemberCandidates(id), enabled: enabled && id !== '', retry: retryServerErrors })
}

/** The custody log of a case (`GET /cases/{id}/custody`: its owner, a reviewer or an administrator): a 403 is a state, not a
 *  failure to retry. */
export function useCustody(id: string, enabled = true) {
  return useQuery({ queryKey: qk.custody(id), queryFn: () => getCustody(id), enabled: enabled && id !== '', retry: retryServerErrors })
}

export function useMe() {
  return useQuery({ queryKey: qk.me(), queryFn: getMe, staleTime: 5 * 60_000, retry: retryServerErrors })  // a 401 is the sign-in state
}
