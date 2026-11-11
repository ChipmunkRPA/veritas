import type { ReactNode } from 'react'
import { eligibilityView, inspectionCheckLabel, inspectionCounts } from './imageEligibility'
import { measurementValue, type CandidateEligibility, type ImageInspectionCheck } from './imagePackets'

export function CandidateEligibilityNote({ gate }: { gate: CandidateEligibility }) {
  const view = eligibilityView(gate)
  return <div className="mt-3 rounded-md border border-border bg-surface-2 p-3 text-small" data-testid="image-candidate-eligibility">
    <p className="m-0 font-medium text-text">{view.title}</p>
    {view.score !== null && <p className="mt-1 mb-0 text-text-muted break-words">
      {view.areaCells ? 'Active screening cells ' : 'Whole-image response '}<span className="font-mono num">{view.score}</span>
      {view.areaCells ? ' · minimum ' : ' · display threshold '}<span className="font-mono num">{view.threshold}</span>.
      {!view.areaCells && ' This response is uncalibrated; it is not a region or pixel probability.'}
    </p>}
    <p className="mt-1 mb-0 text-text-muted">{view.explanation}</p>
  </div>
}

export default function ImageInspectionChecks({ checks, coverageMap }: {
  checks: ImageInspectionCheck[]; coverageMap: (check: ImageInspectionCheck) => ReactNode
}) {
  if (!checks.length) return null
  return <section className="mt-5 rounded-md border border-border p-4" aria-label="Image inspection coverage" data-testid="image-inspection-checks">
    <h3 className="m-0 text-body font-medium text-text">How inspection regions were selected</h3>
    <p className="mt-2 mb-0 text-small text-text-muted">The image gate controls candidate display. The reliability mask separately excludes areas where measurement is less reliable. Neither establishes authenticity.</p>
    <div className="mt-4 space-y-5">{checks.map((check, index) => {
      const counts = inspectionCounts(check)
      const exclusion = check.excluded_area_fraction
      const eligibility = eligibilityView(check.candidate_eligibility)
      const knownGate = eligibility.status === 'eligible' || eligibility.status === 'withheld'
      return <article key={`${check.exhibit_ref}:${index}`} className="border-t border-border pt-3">
        <h4 className="m-0 text-small font-medium text-text">{inspectionCheckLabel(check)}</h4>
        {check.candidate_eligibility ? <CandidateEligibilityNote gate={check.candidate_eligibility} />
          : <p className="mt-2 mb-0 text-small text-text-muted">{eligibilityView().explanation}</p>}
        {counts ? <>
          <dl className="mt-3 mb-0 grid grid-cols-2 gap-x-5 gap-y-3 text-small max-[399px]:grid-cols-1">
            {[
              ['Measured regions before image gate', counts.raw],
              [knownGate ? 'Eligible after image gate' : 'Available for region selection', counts.eligible],
              ['Shown in these evidence cards', counts.selected],
              ['Eligible regions not shown', counts.omitted],
            ].map(([label, value]) => <div key={label} className="min-w-0"><dt className="text-text-muted">{label}</dt><dd className="m-0 mt-1 font-mono num text-text">{value}</dd></div>)}
          </dl>
          {counts.suppressed > 0 && <p className="mt-3 mb-0 text-small text-text-muted">{counts.suppressed} measured {counts.suppressed === 1 ? 'region was' : 'regions were'} withheld by the image gate. The raw map remains in the evidence record; withheld regions are not cleared as authentic.</p>}
          {counts.omitted > 0 && <p className="mt-2 mb-0 text-small text-text-muted">{counts.omitted} eligible {counts.omitted === 1 ? 'region is' : 'regions are'} not shown because of display limits or evidence requirements. These cards are not an exhaustive map of possible edits.</p>}
        </> : <p className="mt-3 mb-0 text-small text-text-muted">Region coverage counts are unavailable. Missing counts do not establish that no regions were measured.</p>}
        {check.map_grid && <p className="mt-3 mb-0 text-caption text-text-muted">Measured grid {check.map_grid.join(' × ')}. Displaying a larger image does not add detection detail.</p>}
        {typeof exclusion === 'number' && Number.isFinite(exclusion) && exclusion >= 0 && exclusion <= 1 && <p className="mt-2 mb-0 text-caption text-text-muted">{measurementValue(exclusion * 100)}% of the measured grid was excluded as less reliable. Those areas were not cleared.</p>}
        {coverageMap(check)}
      </article>
    })}</div>
    <p className="mt-4 mb-0 text-caption text-text-muted">This coverage lists checks with recorded maps. Skipped, failed or unavailable checks remain in the analysis check record; their absence here is not a negative finding.</p>
  </section>
}
