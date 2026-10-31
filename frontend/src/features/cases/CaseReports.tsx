import { useMutation, useQueries } from '@tanstack/react-query'
import { apiClient, unwrap } from '../../lib/api/client'
import { PROFILE_WORD, downloadReport, reportsKey, type Report } from '../../lib/api/reports'
import { Button, EmptyState, ErrorState, SkeletonRows } from '../../components/ui'
import { fmtUtc } from './custody'
import { caseReports, stateAtRequest } from './reportGroups'

const listReports = async (aid: string) => (await unwrap(apiClient.GET('/api/v2/analyses/{aid}/reports', { params: { path: { aid } } }))).items

function Rows({ items, exhibit, group }: { items: readonly Report[]; exhibit: (aid: string) => string | undefined; group: 'issued' | 'drafts' }) {
  const download = useMutation({ mutationFn: (r: Report) => downloadReport(r) })
  return (
    <>
      <ul className="m-0 p-0 list-none" data-testid={`reports-${group}`}>
        {items.map((r) => (
          <li key={r.id} data-testid="case-report" data-report={r.id} className="flex flex-wrap items-baseline gap-x-4 gap-y-1 border-b border-border py-2 text-small">
            <span className="font-mono text-mono text-text num">{r.display_id}</span>
            <span className="text-text">{[exhibit(r.analysis_id) ? `Ex. ${exhibit(r.analysis_id)}` : null, PROFILE_WORD[r.profile] + (r.redacted ? ', redacted' : '')].filter(Boolean).join(' · ')}</span>
            <span className="text-text-muted">{stateAtRequest(r)}</span>
            <span className="font-mono text-mono text-text-muted num">{fmtUtc(r.requested_at)}</span>
            <Button variant="text" className="ml-auto" disabled={download.isPending} onClick={() => download.mutate(r)} aria-label={`Download ${r.display_id}`}>Download</Button>
          </li>
        ))}
      </ul>
      {download.error && <p role="alert" className="mt-2 mb-0 text-small text-danger">{download.error.message}</p>}
    </>
  )
}

/** The reports of a case in two groups, "Issued" and "Drafts" (N-19a; design review DS-02): each row has the R-id that is
 *  printed on the PDF, the exhibit, the profile, the review state the report was requested in, and a download. Read from
 *  the reports of each analysis of the case, which every member can read; the custody log is not needed. */
export default function CaseReports({ analysisIds, exhibit, pending }: {
  analysisIds: readonly string[]
  exhibit: (analysisId: string) => string | undefined
  /** the analyses of the case are still loading */
  pending: boolean
}) {
  const lists = useQueries({
    queries: analysisIds.map((aid) => ({ queryKey: reportsKey(aid), queryFn: () => listReports(aid) })),
    combine: (results) => ({
      loading: results.some((r) => r.isPending), error: results.find((r) => r.error)?.error ?? null,
      reports: results.flatMap((r) => r.data ?? []), refetch: () => { for (const r of results) if (r.error) void r.refetch() },
    }),
  })
  const groups = caseReports(lists.reports)
  const loading = pending || lists.loading

  return (
    <section className="mt-12" aria-labelledby="reports-heading" data-testid="reports">
      <h2 id="reports-heading" className="eyebrow m-0 border-b border-border-strong pb-2">Reports</h2>
      {lists.error && <ErrorState className="mt-4" onRetry={lists.refetch}>{lists.error.message}</ErrorState>}
      <h3 className="mt-4 mb-0 text-small font-medium text-text">Issued</h3>
      {loading ? <SkeletonRows className="mt-2" widths={['60%']} />
        : groups.issued.length === 0 ? <EmptyState className="mt-2">No report has been issued for this case.</EmptyState>
          : <Rows items={groups.issued} exhibit={exhibit} group="issued" />}
      <h3 className="mt-6 mb-0 text-small font-medium text-text">Drafts</h3>
      <p className="mt-1 mb-0 text-caption text-text-muted max-w-[72ch]">A draft was made before the examiner’s sign-off, or while a second review was outstanding. Its id carries DRAFT and it is not an issued report.</p>
      {loading ? <SkeletonRows className="mt-2" widths={['60%']} />
        : groups.drafts.length === 0 ? <EmptyState className="mt-2">No draft report has been made for this case.</EmptyState>
          : <Rows items={groups.drafts} exhibit={exhibit} group="drafts" />}
    </section>
  )
}
