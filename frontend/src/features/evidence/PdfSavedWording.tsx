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
  return <section id="media-evidence-pdf-saved-wording" aria-labelledby="pdf-saved-wording-heading" className="mt-4 rounded-md border border-border p-4 scroll-mt-20">
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
}
