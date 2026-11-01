/** Reports and evidence packages (`POST /api/v2/analyses/{aid}/reports`, `GET /reports/{id}`, `POST /analyses/{aid}/packages`,
 *  `GET /packages/{id}`; PRD RP-03, RP-05, RP-06; UX §5.16; D-10, D-25, D-27). Each export is a job: request it, poll until it is
 *  done, then download the file with the API credentials (a plain link would carry none). */
import { useQuery } from '@tanstack/react-query'
import { apiClient, ApiError, authHeaders, unwrap } from './client'
import type { components } from './schema'

export type Report = components['schemas']['ReportOut']
export type ReportPackage = components['schemas']['PackageOut']
export type ReportProfile = Report['profile']
export type ExportChoice = ReportProfile | 'bundle'
export type InstanceKey = components['schemas']['PublicKeyOut']

export const reportsKey = (aid: string) => ['analysis', aid, 'reports'] as const
export const instanceKeyKey = ['instance', 'public-key'] as const

export const EXPORT_CHOICES: readonly { id: ExportChoice; label: string; description: string }[] = [
  { id: 'summary', label: 'Executive summary', description: 'Short PDF for decision-makers: the answer, the main reasons and the limits' },
  { id: 'standard', label: 'Standard report', description: 'Layered PDF: summary, findings, evidence, methods, custody and review' },
  { id: 'forensic', label: 'Forensic report', description: 'Full record: findings, exhibits, methods, error rates, custody, declaration and signature' },
  { id: 'newsroom', label: 'Newsroom explainer', description: 'Publishable PDF; the filename, source fields and metadata are hidden' },
  { id: 'bundle', label: 'Evidence ZIP', description: 'Forensic PDF, the same evidence record, source crops and clips, and signed file hashes' },
]

export const PROFILE_WORD: Record<ReportProfile, string> = {
  summary: 'Executive summary', standard: 'Standard report', forensic: 'Forensic report', newsroom: 'Newsroom explainer',
}

/** D-56: an issued export needs the approved state, or the reviewed state on a case that does not require a second review. A
 *  disputed revision (the peer reviewer does not concur) is never issued, and nothing is issued while a review record does not
 *  verify. Mirrors `review.issuable` of the API, which the dialog reads. */
export function issuable(state: string | null | undefined, requiresSecondReview: boolean | null | undefined, signaturesValid = true): boolean {
  return signaturesValid && (state === 'approved' || (state === 'reviewed' && !requiresSecondReview))
}

/** Why nothing can be exported from this revision, else null (the API refuses the same cases with 409). */
export function exportRefusal(status: string | null | undefined, state: string | null | undefined): string | null {
  if (status !== 'done') return 'The analysis has not finished; export once it is done.'
  if (state === 'superseded') return 'A newer revision of this analysis exists; export from the latest revision.'
  if (state === 'withdrawn') return 'This revision was withdrawn and should not be relied on.'
  return null
}

/** The buttons the dialog enables (UX §5.16). "Issue" is offered only for an issuable state (D-56). A PDF's draft marking follows
 *  the issuable state when it is requested, so a PDF is a draft until then and issued after; a bundle can be exported as a draft
 *  at any time and issued once issuable. */
export function exportActions(choice: ExportChoice, state: string | null | undefined, requiresSecondReview: boolean | null | undefined,
  signaturesValid = true): { draft: boolean; issue: boolean } {
  const issue = issuable(state, requiresSecondReview, signaturesValid)
  return { draft: choice === 'bundle' || !issue, issue }
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

/** Poll a job resource until it is done; a failed job or a timeout raises. */
export async function untilDone<T extends { status: string; error: string | null }>(read: () => Promise<T>, onStatus?: (status: string) => void,
  intervalMs = 1000, timeoutMs = 180_000): Promise<T> {
  const end = Date.now() + timeoutMs
  for (;;) {
    const x = await read()
    onStatus?.(x.status)
    if (x.status === 'done') return x
    if (x.status === 'failed') throw new ApiError(500, 'Export failed', x.error ?? 'The export job failed; the server log has the details.')
    if (Date.now() > end) throw new ApiError(504, 'Still running', 'The export is still running; it appears in the list below once it is done.')
    await sleep(intervalMs)
  }
}

/** `reuse` (N-19; browser QA P1-8): the server answers with the newest report of this profile that holds the record as it
 *  stands, when there is one, and makes a report only otherwise. A bundle asks this way, so that it mints no further report. */
export async function generateReport(aid: string, profile: ReportProfile, onStatus?: (status: string) => void, reuse = false): Promise<Report> {
  // `redact` applies to the newsroom profile only, which the server always redacts (D-10); any other profile with it is a 422.
  const created = await unwrap(apiClient.POST('/api/v2/analyses/{aid}/reports', { params: { path: { aid } }, body: { profile, redact: profile === 'newsroom', reuse } }))
  return untilDone(() => unwrap(apiClient.GET('/api/v2/reports/{report_id}', { params: { path: { report_id: created.id } } })), onStatus)
}

export const PUBLIC_PACKAGE_SCHEMA = 'chipmunk.public-evidence/1'

function publicPackage(value: ReportPackage): ReportPackage {
  if (value.schema_id !== PUBLIC_PACKAGE_SCHEMA) {
    throw new ApiError(409, 'Export format unavailable', 'This export is not a supported public evidence ZIP. Generate a new evidence ZIP.')
  }
  return value
}

export async function generatePackage(aid: string, reportId: string, kind: ReportPackage['kind'], includeOriginal = false,
  onStatus?: (status: string) => void): Promise<ReportPackage> {
  const created = publicPackage(await unwrap(apiClient.POST('/api/v2/analyses/{aid}/packages', {
    params: { path: { aid } }, body: { report_id: reportId, kind, include_original: includeOriginal },
  })))
  return untilDone(async () => publicPackage(await unwrap(apiClient.GET('/api/v2/packages/{package_id}', { params: { path: { package_id: created.id } } }))), onStatus)
}

export async function downloadFile(path: string, filename: string): Promise<void> {
  const response = await fetch(path, { headers: authHeaders() })
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { title?: string; detail?: string }
    throw new ApiError(response.status, body.title ?? response.statusText, body.detail ?? (response.statusText || `HTTP ${response.status}`))
  }
  saveBlob(await response.blob(), filename)
}

function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export const reportFilename = (r: Pick<Report, 'display_id'>): string => `chipmunk-verify-${r.display_id}.pdf`
export async function downloadReport(r: Report): Promise<void> {
  if (!r.downloadable) throw new ApiError(409, 'Report unavailable', 'This report is not available for download. Generate a new report.')
  await downloadFile(`/api/v2/reports/${encodeURIComponent(r.id)}/file`, reportFilename(r))
}
export async function downloadPackage(p: ReportPackage): Promise<void> {
  publicPackage(p)
  await downloadFile(`/api/v2/packages/${encodeURIComponent(p.id)}/file`, p.filename ?? `chipmunk-evidence-${p.id.slice(0, 8)}.zip`)
}

/** The key that signs this instance's reports and bundles. Whoever receives a bundle checks it against this key, pinned by
 *  its id or by the key file (`verify --expect-key-id`, `--public-key`): the copy inside a bundle says nothing about who signed
 *  it, so the id and the file leave the instance here, apart from the bundle. */
export function useInstanceKey(enabled: boolean) {
  return useQuery({
    queryKey: instanceKeyKey,
    enabled,
    staleTime: 5 * 60_000,
    queryFn: () => unwrap(apiClient.GET('/api/v2/instance/public-key')),
  })
}

export const instanceKeyFilename = (k: Pick<InstanceKey, 'key_id'>): string => `chipmunk-instance-key-${k.key_id.replace(/^ed25519:/, '')}.pem`
export const saveInstanceKey = (k: InstanceKey): void => saveBlob(new Blob([k.public_key_pem], { type: 'application/x-pem-file' }), instanceKeyFilename(k))

/** The analysis's reports, newest first; polls while one is queued or running. */
export function useReports(aid: string, enabled: boolean) {
  return useQuery({
    queryKey: reportsKey(aid),
    enabled,
    queryFn: async () => (await unwrap(apiClient.GET('/api/v2/analyses/{aid}/reports', { params: { path: { aid } } }))).items,
    refetchInterval: (query) => (query.state.data?.some((r) => r.status === 'queued' || r.status === 'running') ? 1500 : false),
  })
}
