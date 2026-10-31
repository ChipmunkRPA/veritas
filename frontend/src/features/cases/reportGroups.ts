/** The reports of a case (BUILD_SPEC N-19a; design review DS-02; browser QA P1-8), from `GET /analyses/{id}/reports`, which
 *  every member of the case can read. The list was built from the custody log's `report_generated` events: the drafts of
 *  unreviewed assessments stood under "Reports issued" with VR numbers that no PDF carries, and an examiner, who cannot read
 *  that log, saw no list. Two groups: a report requested in an issuable state is issued (its id has no DRAFT); every other
 *  finished report is a draft. A report that is queued, running or failed is no document yet and is not listed. */
import type { Report } from '../../lib/api/reports'

export interface CaseReports { issued: Report[]; drafts: Report[] }

const newestFirst = (x: Report, y: Report) => Date.parse(y.requested_at) - Date.parse(x.requested_at) || (x.id < y.id ? 1 : -1)

export function caseReports(reports: readonly Report[]): CaseReports {
  const done = reports.filter((r) => r.status === 'done').sort(newestFirst)
  return { issued: done.filter((r) => r.issuable), drafts: done.filter((r) => !r.issuable) }
}

/** The review state the report was requested in, in the words of the case table. A signed report that is not issuable is
 *  one whose second review was outstanding, or whose review record did not verify: it never reads as reviewed (D-56). */
export function stateAtRequest(r: Pick<Report, 'review_state' | 'issuable'>): string {
  if (r.issuable) return r.review_state === 'approved' ? 'reviewed · second review' : 'reviewed'
  switch (r.review_state) {
    case 'machine': return 'not reviewed'
    case 'in_review': return 'in review, not signed'
    case 'reviewed': return 'signed, not issuable: second review outstanding'
    case 'approved': return 'signed, not issuable'
    case 'disputed': return 'disputed: the peer reviewer does not concur'
    default: return r.review_state
  }
}
