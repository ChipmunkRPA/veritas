import { Link } from 'react-router-dom'
import type { Case } from '../../lib/api/queries'
import { fmtDate } from '../../lib/format'
import { REVIEW_LABEL, REVIEW_ORDER } from './CaseOverview'
import { CASE_STATUS_WORD, CLASSIFICATION_WORD } from './labels'
import { AWAITING_SECOND_REVIEW } from '../review/review'

const STATES = REVIEW_ORDER

function ReviewCounts({ cases, awaiting }: { cases: Case[]; awaiting: number }) {
  // N-19a (P1-12): of the signed revisions, those that await their second review are counted apart and never as reviewed
  const counts = STATES.flatMap((state) => {
    const n = cases.reduce((sum, c) => sum + (c.review_counts?.[state] ?? 0), 0)
    if (state !== 'reviewed') return [{ key: state as string, n, label: REVIEW_LABEL[state] }]
    const waiting = Math.min(awaiting, n)
    return [{ key: 'awaiting', n: waiting, label: AWAITING_SECOND_REVIEW }, { key: state as string, n: n - waiting, label: REVIEW_LABEL[state] }]
  })
  const total = cases.reduce((sum, c) => sum + c.analysis_count, 0)
  const unknown = Math.max(0, total - counts.reduce((sum, entry) => sum + entry.n, 0))
  if (!total) return <span className="text-text-muted">No analyses</span>
  return <span className="flex flex-col gap-0.5 num">
    {counts.filter(({ n }) => n > 0).map(({ key, n, label }) => <span key={key} data-review-count={key}>{n} {label}</span>)}
    {unknown > 0 && <span className="text-text-muted">{unknown} review not recorded</span>}
  </span>
}

/** Group only the already-filtered visible cases; each underlying case retains its own evidence link. */
export default function CaseLibraryRows({ cases, awaiting }: { cases: Case[]; /** case id → revisions that await their second review */ awaiting?: ReadonlyMap<string, number> }) {
  const unfiled = cases.filter((c) => c.implicit)
  const groups = cases.filter((c) => !c.implicit).map((c) => ({ id: c.id, unfiled: false, cases: [c] }))
  if (unfiled.length) groups.push({ id: 'unfiled', unfiled: true, cases: unfiled })
  const latest = (group: typeof groups[number]) => group.cases.reduce((date, c) => Date.parse(c.updated_at) > Date.parse(date) ? c.updated_at : date, group.cases[0].updated_at)
  groups.sort((a, b) => Date.parse(latest(b)) - Date.parse(latest(a)))
  return <>{groups.map((group) => {
    const c = group.cases[0]
    const statuses = [...new Set(group.cases.map((r) => r.status))]
    const classifications = [...new Set(group.cases.map((r) => r.classification))]
    const exhibits = group.cases.reduce((sum, r) => sum + r.item_count, 0)
    const analyses = group.cases.reduce((sum, r) => sum + r.analysis_count, 0)
    // N-19b (design review DS-04): the status and the classification are a caption under the title, not a column
    const state = `${statuses.length === 1 ? CASE_STATUS_WORD[statuses[0]] : 'Mixed statuses'} · ${classifications.length === 1 ? CLASSIFICATION_WORD[classifications[0]] : 'mixed classifications'}`
    return <tr role="row" key={group.id} className="border-b border-border h-16 max-[639px]:h-auto max-[639px]:px-4 transition-colors duration-fast hover:bg-surface-3">
      {/* the title column takes what the fixed columns leave (`table-fixed`); a long title is clamped to two lines inside it */}
      <td role="cell" className="py-3 pl-4 pr-4 min-w-0">
        {group.unfiled ? <details>
          <summary className="cursor-pointer text-body text-text"><span className="font-medium">Unfiled</span><span className="text-caption text-text-muted ml-2">{group.cases.length} {group.cases.length === 1 ? 'case' : 'cases'}</span></summary>
          <ul className="my-2 pl-5 space-y-2">
            {group.cases.map((source) => <li key={source.id} className="min-w-0"><Link to={`/cases/${source.id}`} title={source.title} className="block text-small text-text hover:underline underline-offset-[3px] line-clamp-2 [overflow-wrap:anywhere]">{source.title}</Link><span className="block font-mono text-caption text-text-muted">{source.reference}</span></li>)}
          </ul>
        </details> : <>
          <Link to={`/cases/${c.id}`} className="flex flex-col gap-0.5 min-w-0 max-w-full text-body text-text hover:underline underline-offset-[3px]">
            <span className="font-mono text-mono text-text-muted">{c.reference}</span>
            <span data-testid="case-title" title={c.title} className="font-medium max-w-full line-clamp-2 [overflow-wrap:anywhere]">{c.title}</span>
          </Link>
          {c.matter_reference && <span title={c.matter_reference} className="block max-w-full text-caption text-text-muted truncate">{c.matter_reference}</span>}
        </>}
        <span data-testid="case-state" className="block mt-0.5 text-caption text-text-muted"><span className="text-text">{state.split(' · ')[0]}</span> · {state.split(' · ')[1]}</span>
      </td>
      <td role="cell" className="pr-4 text-small text-text num whitespace-nowrap">{exhibits} {exhibits === 1 ? 'exhibit' : 'exhibits'} · {analyses} {analyses === 1 ? 'analysis' : 'analyses'}</td>
      <td role="cell" className="pr-4 py-3 text-small text-text"><ReviewCounts cases={group.cases} awaiting={group.cases.reduce((sum, r) => sum + (awaiting?.get(r.id) ?? 0), 0)} /></td>
      <td role="cell" data-th="last activity" data-testid="last-activity" className="pr-4 text-right max-[639px]:text-left text-small text-text-muted num whitespace-nowrap">{fmtDate(latest(group))}</td>
    </tr>
  })}</>
}
