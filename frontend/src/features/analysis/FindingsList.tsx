import type { UseQueryResult } from '@tanstack/react-query'
import type { Finding } from '../../lib/api/queries'
import { SET_ASIDE_NOTE, SET_ASIDE_TITLE, markOf, setAside, type Decision } from './answer'
import { EmptyState, ErrorState, IdChip, SkeletonRows } from '../../components/ui'
import { findingsNotWritten } from './vocabulary'

/** The backend's evidential strength words (ln-LR bands, IF §6.2). */
const STRENGTH_WORD: Record<Finding['strength'], string> = {
  decisive: 'Decisive', strong: 'Strong', moderate: 'Moderate', weak: 'Weak', none: 'No effect',
}
const PRIORITY_WORD: Record<Finding['review_priority'], string> = {
  high: 'Review first', medium: 'Review', low: 'Low priority', info: 'Information',
}

/** N-19n (D3-07): the strength word of a marker's finding is "Self-declared", or "Not applied": no marker is "Strong". */
const recordWord = (f: Finding, decision?: Decision | null): string | null => {
  const mark = markOf(f, decision)
  return mark && mark.level === 'self_declared' ? `${mark.word[0].toUpperCase()}${mark.word.slice(1)}` : null
}

/** Every machine finding (ARCHITECTURE §4.5; ids F1… per D-26) in rank order, with its full statement. */
export function FindingRows({ items, decision }: { items: readonly Finding[]; decision?: Decision | null }) {
  return (
    <ol className="border-t border-border-strong">
      {[...items].sort((x, y) => x.rank - y.rank).map((f) => { const aside = setAside(f, decision); return (
        <li key={f.id} className="grid grid-cols-[48px_minmax(0,1fr)_max-content] gap-x-4 py-4 border-b border-border max-[639px]:grid-cols-[40px_minmax(0,1fr)]">
          <span className="pt-0.5"><IdChip title={`Finding ${f.id}`}>{f.id}</IdChip></span>
          <div className="min-w-0">
            <h3 className="text-body font-medium text-text">{aside ? SET_ASIDE_TITLE : f.title}</h3>
            <p className="mt-1 text-small text-text-muted max-w-[72ch]">{f.statement}</p>
            {aside && <p data-testid="finding-set-aside" className="mt-1 text-small text-text-muted max-w-[72ch]">{SET_ASIDE_NOTE}</p>}
            {f.evidence_ids && f.evidence_ids.length > 0 && (
              <p className="mt-2 font-mono text-caption text-text-muted break-words">{f.evidence_ids.join(' · ')}</p>
            )}
          </div>
          <dl className="m-0 text-right max-[639px]:col-start-2 max-[639px]:mt-2 max-[639px]:text-left">
            <dt className="sr-only">Strength</dt>
            <dd className="m-0 text-small text-text">{aside ? 'Not counted' : recordWord(f, decision) ?? STRENGTH_WORD[f.strength]}</dd>
            <dt className="sr-only">Review priority</dt>
            <dd className="m-0 text-caption text-text-muted">{PRIORITY_WORD[f.review_priority]}</dd>
          </dl>
        </li>
      ) })}
    </ol>
  )
}

/** Findings of an analysis without a v2 decision block: the full list stands in for the key findings module. */
export default function FindingsList({ query, failed = false }: { query: UseQueryResult<Finding[], Error>
  /** N-19n (Q4-01): the step that writes the findings failed (`findingsFailed`): the list says so, not that none was produced */
  failed?: boolean }) {
  const items = query.data ?? []
  return (
    <section id="section-findings" data-testid="findings" aria-labelledby="findings-heading" className="scroll-mt-14">
      <h2 id="findings-heading" className="eyebrow">Findings</h2>
      {query.isPending && <SkeletonRows className="mt-4" />}
      {query.error && <ErrorState className="mt-4" onRetry={() => void query.refetch()}>{query.error.message}</ErrorState>}
      {query.isSuccess && items.length === 0 && <EmptyState className="mt-4">{failed ? findingsNotWritten('ledger') : 'The analysis produced no findings.'}</EmptyState>}
      {items.length > 0 && <div className="mt-3"><FindingRows items={items} /></div>}
    </section>
  )
}
