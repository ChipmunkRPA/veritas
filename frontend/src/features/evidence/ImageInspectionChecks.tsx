import type { ReactNode } from 'react'
import { eligibilityView, inspectionCheckLabel, inspectionCounts } from './imageEligibility'
import { measurementValue, type CandidateEligibility, type ImageInspectionCheck } from './imagePackets'

const qualityLabels = {
  clipped: 'Clipped highlights or very dark pixels',
  flat: 'Low local contrast',
  compressed: 'Low-detail JPEG blocks',
  upscaled: 'Smooth patterns consistent with resizing',
} as const

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
      const shares = check.source_quality?.schema === 'image-source-quality/1'
        ? check.source_quality.native_pixel_shares : undefined
      const qualityRows = (Object.keys(qualityLabels) as (keyof typeof qualityLabels)[])
        .flatMap(key => {
          const share = shares?.[key]
          return typeof share === 'number' && share > 0 && share <= 1 ? [[qualityLabels[key], share * 100] as const] : []
        })
      const eligibility = eligibilityView(check.candidate_eligibility)
      const knownGate = eligibility.status === 'eligible' || eligibility.status === 'withheld'
      const flip = check.flip_consistency
      const checkedFlip = flip && counts && flip.schema === 'image-flip-consistency/1'
        && Number.isSafeInteger(flip.inspected_components)
        && flip.inspected_components >= counts.eligible && flip.inspected_components <= Math.min(2, counts.raw)
        && flip.supported_components === counts.eligible
        && Array.isArray(flip.candidates) && flip.candidates.length === flip.inspected_components
        && flip.withheld_components === flip.inspected_components - flip.supported_components ? flip : null
      const gateWithheld = flip && counts ? (eligibility.status === 'withheld' ? counts.raw : 0) : (counts?.suppressed ?? 0)
      const outsideFlipLimit = checkedFlip && counts && eligibility.status !== 'withheld'
        ? counts.raw - checkedFlip.inspected_components : 0
      return <article key={`${check.exhibit_ref}:${index}`} className="border-t border-border pt-3">
        <h4 className="m-0 text-small font-medium text-text">{inspectionCheckLabel(check)}</h4>
        {check.candidate_eligibility ? <CandidateEligibilityNote gate={check.candidate_eligibility} />
          : <p className="mt-2 mb-0 text-small text-text-muted">{eligibilityView().explanation}</p>}
        {counts ? <>
          <dl className="mt-3 mb-0 grid grid-cols-2 gap-x-5 gap-y-3 text-small max-[399px]:grid-cols-1">
            {[
              ['Measured regions before image gate', counts.raw],
              [flip ? 'Supported by the flip check' : knownGate ? 'Eligible after image gate' : 'Available for region selection', counts.eligible],
              ['Shown in these evidence cards', counts.selected],
              [flip ? 'Supported regions not shown' : 'Eligible regions not shown', counts.omitted],
            ].map(([label, value]) => <div key={label} className="min-w-0"><dt className="text-text-muted">{label}</dt><dd className="m-0 mt-1 font-mono num text-text">{value}</dd></div>)}
          </dl>
          {gateWithheld > 0 && <p className="mt-3 mb-0 text-small text-text-muted">{gateWithheld} measured {gateWithheld === 1 ? 'region was' : 'regions were'} withheld by the image gate. The raw map remains in the evidence record; withheld regions are not cleared as authentic.</p>}
          {flip && (checkedFlip ? <>
            <p className="mt-3 mb-0 text-small text-text-muted">The flip check uses the same detector on a horizontal flip. It retains a region when at least 50% of its measured grid cells overlap after mapping back. It contributes no AI authorship weight.</p>
            <p className="mt-2 mb-0 text-small text-text-muted">The flip check withheld {checkedFlip.withheld_components} of {checkedFlip.inspected_components} candidate regions.</p>
            <ul aria-label="Repeat measurements by candidate" className="mt-2 mb-0 list-disc pl-4 text-small text-text-muted">
              {checkedFlip.candidates.map((candidate, index) => {
                const [x, y, width, height] = candidate.bounds
                return <li key={candidate.region_id}>
                  <span>Initial candidate {index + 1}. {candidate.shared_pixels.toLocaleString('en-US')} of {candidate.source_pixels.toLocaleString('en-US')} measured cells overlapped. {candidate.retained ? 'Supported by the repeat.' : 'Withheld after the repeat.'}</span>
                  <span className="block font-mono text-caption">Source x {x}, y {y} · {width} × {height} px</span>
                </li>
              })}
            </ul>
            {outsideFlipLimit > 0 && <p className="mt-2 mb-0 text-small text-text-muted">{outsideFlipLimit} measured {outsideFlipLimit === 1 ? 'region was' : 'regions were'} outside the two-candidate review limit.</p>}
          </> : <p className="mt-2 mb-0 text-small text-text-muted">Flip-check coverage is unavailable.</p>)}
          {counts.omitted > 0 && <p className="mt-2 mb-0 text-small text-text-muted">{counts.omitted} {flip ? 'supported' : 'eligible'} {counts.omitted === 1 ? 'region is' : 'regions are'} not shown because of display limits or evidence requirements. These cards are not an exhaustive map of possible edits.</p>}
        </> : <p className="mt-3 mb-0 text-small text-text-muted">Region coverage counts are unavailable. Missing counts do not establish that no regions were measured.</p>}
        {check.map_grid && <p className="mt-3 mb-0 text-caption text-text-muted">Measured grid {check.map_grid.join(' × ')}. Displaying a larger image does not add detection detail.</p>}
        {typeof exclusion === 'number' && Number.isFinite(exclusion) && exclusion >= 0 && exclusion <= 1 && <p className="mt-2 mb-0 text-caption text-text-muted">{measurementValue(exclusion * 100)}% of the measured grid was excluded as less reliable. Those areas were not cleared.</p>}
        {qualityRows.length > 0 && <>
          <p className="mt-3 mb-0 text-caption text-text-muted">Image quality conditions by share of original pixels. The areas can overlap.</p>
          <dl className="mt-2 mb-0 grid grid-cols-2 gap-2 text-small max-[399px]:grid-cols-1">
            {qualityRows.map(([label, percent]) => <div key={label}>
              <dt className="text-text-muted">{label}</dt><dd className="m-0 mt-1 font-mono num text-text">{measurementValue(percent)}%</dd>
            </div>)}
          </dl>
        </>}
        {coverageMap(check)}
      </article>
    })}</div>
    <p className="mt-4 mb-0 text-caption text-text-muted">This coverage lists checks with recorded maps. Skipped, failed or unavailable checks remain in the analysis check record; their absence here is not a negative finding.</p>
  </section>
}
