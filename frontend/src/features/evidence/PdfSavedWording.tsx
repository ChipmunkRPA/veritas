import { EvidencePicture, evidenceGrid } from './EvidenceMedia'
import type { DocumentEvidenceSet, PdfSavedWording as Review } from './multimediaPackets'

const unavailable: Record<NonNullable<Review['inspection']['reason']>, string> = {
  size_limit: 'The PDF is larger than the saved-version check supports.',
  page_limit: 'A retained save exceeds the page limit for this check.',
  save_limit: 'Earlier saves fall outside the five-save comparison limit.',
  page_text_incomplete: 'Some pages have missing, covered, or reordered text that this check could not compare reliably.',
  unverified_chain: 'The stored PDF version boundaries could not be verified.',
  time_limit: 'The saved-version check reached its time limit.',
  reader_unavailable: 'The saved-version reader was unavailable.',
}

export default function PdfSavedWording({ review, set, token, mediaReady }: {
  review: Review; set: DocumentEvidenceSet; token: string; mediaReady: boolean
}) {
  const { inspection, artifacts } = review
  const reading = inspection.comparisons.at(-1)
  const latest = inspection.saves[0]
  return <><section id="media-evidence-pdf-saved-wording" aria-labelledby="pdf-saved-wording-heading" className="mt-4 rounded-md border border-border p-4 scroll-mt-20">
    <h3 id="pdf-saved-wording-heading" className="m-0 text-body font-medium">Changes across saved PDF versions</h3>
    <p className="mt-2 mb-0 text-small text-text-muted">This check compares native page text across saves retained inside the uploaded PDF. Ordinary editing can change it. The comparison carries no AI confidence weight.</p>
    {inspection.status === 'no_history' && <p className="mt-2 mb-0 text-small">The file retains one verified save.</p>}
    {inspection.status === 'unavailable' && inspection.reason && <p className="mt-2 mb-0 text-small">{unavailable[inspection.reason]}</p>}
    {inspection.status === 'no_change' && <p className="mt-2 mb-0 text-small">The page wording agrees across the {inspection.saves.length} retained saves checked.</p>}
    {inspection.status === 'changed' && reading && <>
      <p className="mt-2 mb-0 text-small">Compared save {reading.earlier_version} with the submitted save {latest.version}. Checked wording on {reading.checked.length} matched page {reading.checked.length === 1 ? 'pair' : 'pairs'}.</p>
      {reading.unassessed.length > 0 && <p className="mt-2 mb-0 text-small text-text-muted">Current pages {reading.unassessed.map(pair => pair[1]).join(', ')} could not be assessed for wording changes.</p>}
      {reading.order_only.length > 0 && <p className="mt-2 mb-0 text-small text-text-muted">The text order changed on current pages {reading.order_only.map(pair => pair[1]).join(', ')}.</p>}
      {reading.earlier_only_pages.length > 0 && <p className="mt-2 mb-0 text-small">Earlier pages without a matching page in the submitted save · {reading.earlier_only_pages.join(', ')}</p>}
      {reading.current_only_pages.length > 0 && <p className="mt-2 mb-0 text-small">Current pages without a matching earlier page · {reading.current_only_pages.join(', ')}</p>}
      <ol className="mt-3 mb-0 list-none space-y-3 p-0">{reading.changes.map((change, index) => {
        const label = { replace: 'Wording replaced', insert: 'Wording added', delete: 'Wording removed' }[change.kind]
        return <li key={index} id={`media-evidence-pdf-save-change-${index + 1}`} className="rounded-md border border-border p-3 scroll-mt-20">
          <h4 className="m-0 text-small font-medium">{label}</h4>
          {change.current && <>
            <blockquote className="mt-2 mb-0 break-words text-small">{change.current.excerpt}{change.current.word_range[1] > change.current.excerpt_word_range[1] && '…'}</blockquote>
            <p className="mt-1 mb-0 text-caption text-text-muted">Current page {change.current.page}, word positions [{change.current.word_range.join(', ')}).</p>
            <div className={`mt-3 ${evidenceGrid}`}>
              {(['crop', 'overlay'] as const).map(role => {
                const asset = artifacts[`change${index + 1}_${role}`]
                return asset && <EvidencePicture key={role} set={set} artifact={asset} token={token} ready={mediaReady} chart
                  alt={role === 'crop' ? `Unmarked PDF crop for change ${index + 1}` : `Highlighted PDF crop for change ${index + 1}`} />
              })}
            </div>
          </>}
          {change.earlier && <p className="mt-2 mb-0 text-caption text-text-muted">Earlier page {change.earlier.page}, word positions [{change.earlier.word_range.join(', ')}).</p>}
        </li>
      })}</ol>
      {reading.changed_ranges > reading.changes.length && <p className="mt-2 mb-0 text-small text-text-muted">Showing {reading.changes.length} of {reading.changed_ranges} wording changes.</p>}
      <p className="mt-2 mb-0 text-caption text-text-muted">Earlier wording is represented by its location in the retained save. Word positions are zero-based and the ending position is exclusive.</p>
    </>}
    {inspection.saves.length > 1 && <details className="mt-3 text-caption text-text-muted"><summary className="cursor-pointer">Saved version records</summary>
      <ul className="pl-4 list-disc">{inspection.saves.map(save => <li key={save.version}>Save {save.version}, first {save.prefix_bytes.toLocaleString('en-US')} file bytes.
        <span className="block break-all font-mono">SHA-256 {save.sha256}</span></li>)}</ul>
    </details>}
  </section>
  {inspection.pictures && <SavedPictures review={review} set={set} token={token} mediaReady={mediaReady} />}
  </>
}

const pictureUnavailable = {
  ...unavailable,
  no_comparable_pictures: 'The check found no comparable pictures in at least one retained save.',
  partial_picture_coverage: 'Some pictures or pages could not be compared. No local changes were found in the areas checked.',
}

function SavedPictures({ review, set, token, mediaReady }: {
  review: Review; set: DocumentEvidenceSet; token: string; mediaReady: boolean
}) {
  const pictures = review.inspection.pictures
  if (!pictures) return null
  const reading = pictures.comparisons.at(-1)
  const earlier = pictures.saves.at(-1)
  return <section id="media-evidence-pdf-saved-pictures" aria-labelledby="pdf-saved-pictures-heading" className="mt-4 rounded-md border border-border p-4 scroll-mt-20">
    <h3 id="pdf-saved-pictures-heading" className="m-0 text-body font-medium">Pictures across saved PDF versions</h3>
    <p className="mt-2 mb-0 text-small text-text-muted">Highlighted pixels mark visible differences in a picture found at the same position in two retained saves. Ordinary editing and AI editing can both produce differences. This check contributes zero weight to the AI assessment.</p>
    {pictures.status === 'unavailable' && pictures.reason && <p className="mt-2 mb-0 text-small">{pictureUnavailable[pictures.reason]}</p>}
    {pictures.status === 'no_change' && <p className="mt-2 mb-0 text-small">No local picture differences were found in the areas checked across {pictures.saves.length} verified saves.</p>}
    {pictures.status === 'changed' && reading && earlier && <>
      <p className="mt-2 mb-0 text-small">Compared save {reading.earlier_version} with submitted save {pictures.saves[0].version}.</p>
      <ol className="mt-3 mb-0 list-none space-y-3 p-0">{reading.changes.map((change, index) =>
        <li key={index} id={`media-evidence-pdf-picture-change-${index + 1}`} className="rounded-md border border-border p-3 scroll-mt-20">
          <h4 className="m-0 text-small font-medium">Picture change {index + 1}</h4>
          <p className="mt-1 mb-0 text-small">Earlier page {change.earlier_page} · submitted page {change.current_page}.</p>
          <p className="mt-1 mb-0 text-caption text-text-muted">Region [{change.region_pixels.join(', ')}] at {change.render_scale} rendered pixels per PDF point, from the unrotated page’s top left. {change.marked_pixels} highlighted pixels.</p>
          <p className="mt-1 mb-0 text-caption text-text-muted">{change.matching_tiles.toLocaleString('en-US')} of {change.tested_tiles.toLocaleString('en-US')} picture tiles matched pixel for pixel between these saves.</p>
          <div className={`mt-3 ${evidenceGrid}`}>
            {(['overlay', 'crop'] as const).map(role => {
              const asset = review.artifacts[`picture${index + 1}_${role}`]
              const label = role === 'overlay' ? 'Submitted picture with the measured difference highlighted' : 'Unmarked crop from the submitted PDF'
              return asset && <figure key={role} className="m-0 min-w-0">
                <EvidencePicture set={set} artifact={asset} token={token} ready={mediaReady} chart alt={label} />
                <figcaption className="mt-1 text-caption text-text-muted">{label}</figcaption>
              </figure>
            })}
          </div>
        </li>)}</ol>
      {reading.changed_regions > reading.changes.length && <p className="mt-2 mb-0 text-small">Showing {reading.changes.length} of {reading.changed_regions} picture regions.</p>}
      <details className="mt-3 text-caption text-text-muted"><summary className="cursor-pointer">Earlier picture source</summary>
        <p className="mb-0">Save {earlier.version}, first {earlier.prefix_bytes.toLocaleString('en-US')} file bytes.
          <span className="block break-all font-mono">SHA-256 {earlier.sha256}</span>
        </p>
      </details>
    </>}
    {pictures.comparisons.map(comparison => {
      const checked = comparison.pages.reduce((sum, page) => sum + page.compared, 0)
      const skipped = comparison.pages.reduce((sum, page) => sum + page.skipped, 0)
      return <p key={comparison.earlier_version} className="mt-2 mb-0 text-caption text-text-muted">
        Save {comparison.earlier_version} comparison · {checked} picture {checked === 1 ? 'placement' : 'placements'} checked.
        {skipped > 0 && ` ${skipped} placements or pages unassessed.`}
        {comparison.unpaired_pages.some(pages => pages.length > 0) && ' Some pages had no matching page in the other save.'}
      </p>
    })}
  </section>
}
