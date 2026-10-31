import { useConclusion, type AnalysisDetail } from '../../lib/api/queries'
import { examinerOpinion, reviewReference } from './review'

const SIGNED = new Set(['reviewed', 'approved', 'disputed'])

/** The examiner's opinion of a signed record, directly under the machine headline and at its size (N-19, D-69; browser QA
 *  P1-6; design review DS-06): both positions stand side by side, with the number of steps and the justification in full.
 *  Nothing before the examiner has signed: a draft conclusion is not an opinion of record. */
export default function ExaminerOpinion({ analysis }: { analysis: AnalysisDetail }) {
  const signed = SIGNED.has(analysis.review?.state ?? 'machine')
  const conclusion = useConclusion(analysis.id, signed)
  if (!signed) return null
  const opinion = examinerOpinion(conclusion.data, analysis.review, reviewReference(analysis, conclusion.data))
  if (!opinion) return null
  return (
    <div data-testid="examiner-opinion" data-departs={opinion.departs ?? undefined} className="mt-2">
      <p className="m-0 display text-h1 font-normal text-text max-w-[36ch]">{opinion.line}</p>
      {opinion.justification && (
        <p data-testid="examiner-justification" className="mt-2 m-0 text-body text-text max-w-[72ch] whitespace-pre-line break-words">
          <span className="text-text-muted">{opinion.label}: </span>{opinion.justification}
        </p>
      )}
      {opinion.also && (
        <p data-testid="examiner-justification-also" className="mt-2 m-0 text-body text-text max-w-[72ch] whitespace-pre-line break-words">
          <span className="text-text-muted">{opinion.also.label}: </span>{opinion.also.text}
        </p>
      )}
    </div>
  )
}
