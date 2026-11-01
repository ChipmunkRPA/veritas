import { Link } from 'react-router-dom'
import { useCaseAnalyses, type AnalysisDetail } from '../../lib/api/queries'
import { supersedes } from '../cases/chains'

/** Preserve revision warnings without introducing a manual examination workflow. */
export default function AutomatedResultStatus({ analysis }: { analysis: AnalysisDetail }) {
  const state = analysis.review?.state
  const revisions = useCaseAnalyses(state === 'superseded' ? analysis.case_id ?? '' : '')
  const successors = revisions.data?.filter((item) => item.supersedes_id === analysis.id
    && item.case_id === analysis.case_id && supersedes(item.status)) ?? []
  const successor = successors.length === 1 ? successors[0] : null

  return (
    <p data-testid="automated-result-status" className="mt-3 mb-0 text-small text-text-muted" role={state === 'withdrawn' ? 'alert' : undefined}>
      {state === 'withdrawn' ? 'This result has been withdrawn. Do not rely on this revision.'
        : state === 'superseded' ? <>This is an earlier analysis.{' '}
          {successor ? <Link to={`/analyses/${successor.id}?view=plain`} className="text-accent underline underline-offset-[3px]">Open the newer revision</Link>
            : 'A newer revision exists; check the analyses list for the current result.'}</>
          : 'Automated analysis complete'}
    </p>
  )
}
