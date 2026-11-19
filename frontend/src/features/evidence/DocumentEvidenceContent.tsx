import { useEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { Button } from '../../components/ui'
import { EvidenceLimits, EvidencePicture, EvidenceReferences, evidenceCard, evidenceGrid } from './EvidenceMedia'
import DocumentInspectionMap, { packageAnchorLabel, passageLocation, RecordedPassage, windowInspectionDirection } from './DocumentInspectionMap'
import DocumentContextComparison from './DocumentContextComparison'
import DocumentPassageCoverage from './DocumentPassageCoverage'
import DocumentRepeatedPassages from './DocumentRepeatedPassages'
import DocumentUnselectedResponses from './DocumentUnselectedResponses'
import PdfSavedWording from './PdfSavedWording'
import SavedLinkDestinations from './SavedLinkDestinations'
import { type DocumentEvidencePacket, type DocumentEvidenceSet, type DocumentEvidenceWindow, type DocumentOriginalTextLocation, type DocumentPassagePosition, type EmailBodyAlternatives, type EmailClassifierInputScope, type MappedTextFinding, type PublishedWordingColumn, type PublishedWordingReview, type PublishedWordingSection, type WritingCheckReason } from './multimediaPackets'

function BibliographyCoverage({ review, doiReviewed }: {
  review: NonNullable<NonNullable<DocumentEvidenceSet['coverage']>['writing_bibliography']>
  doiReviewed: boolean
}) {
  const { first_location: first, last_location: last } = review
  let location: string | undefined
  if (first.kind === 'pdf' && last.kind === 'pdf') {
    location = first.page === last.page ? `PDF page ${first.page}` : `PDF pages ${first.page}–${last.page}`
  } else if (first.kind === 'docx' && last.kind === 'docx') {
    location = `Word body blocks ${first.body_index + 1}–${last.body_index + 1}`
  } else if (first.kind === 'text' && last.kind === 'text') {
    location = `Source characters ${first.start + 1}–${last.end}`
  }
  return <section className="mt-4 rounded-md border border-border p-4 text-small text-text-muted">
    <h3 className="m-0 text-body font-medium text-text">Excluded references</h3>
    <p className="mt-2 mb-0">The writing input excluded a terminal reference list with {review.entry_count} citation entries.
      {location && ` ${location}.`}{' '}
      {doiReviewed ? <a href="#document-doi-review" className="underline">See the DOI reference check.</a> : 'Citation accuracy was not assessed.'}</p>
  </section>
}

const originalLocationLabels: Record<DocumentOriginalTextLocation['coordinate'], string> = {
  decoded_text: 'Original decoded text characters',
  word_paragraph: 'Original Word paragraph characters',
  word_note_paragraph: 'Original Word note paragraph characters',
  odt_paragraph: 'Decoded OpenDocument paragraph characters',
  rtf_paragraph: 'Decoded Rich Text block characters',
  markdown_paragraph: 'Decoded Markdown paragraph characters',
  pptx_block: 'Decoded slide text characters',
  odp_block: 'Decoded OpenDocument slide text characters',
  epub_block: 'Decoded EPUB text characters',
  xlsx_cell: 'Saved Excel cell characters',
  ods_cell: 'Saved OpenDocument spreadsheet cell characters',
  csv_cell: 'Decoded CSV field characters',
  tsv_cell: 'Decoded TSV field characters',
  decoded_email: 'Decoded email part characters',
  decoded_email_html: 'Decoded email HTML source characters',
  decoded_html: 'Original HTML source characters',
}

const invisibleCharacterNames: Record<string, string> = {
  'U+202D': 'Left-to-right override', 'U+202E': 'Right-to-left override', 'U+FEFF': 'Embedded byte order mark',
  'U+00AD': 'Soft hyphen', 'U+200B': 'Zero-width space', 'U+2060': 'Word joiner',
}

const emailInputRoles = [
  ['body', 'Message only'], ['quoted', 'Quote only'], ['signature', 'Signature only'],
  ['mixed', 'Mixed roles'], ['unmapped', 'Role unavailable'],
] as const

const writingCheckLabels = {
  completed: 'Completed',
  not_applicable: 'Outside this check’s scope',
  skipped: 'Skipped',
  failed: 'Failed',
  timed_out: 'Timed out',
}

const writingCheckReasons: Record<WritingCheckReason, string> = {
  memory_budget: 'Available analysis memory was too low to run this check.',
  unavailable: 'This check was unavailable on the analysis server.',
  verified_credentials: 'Verified Content Credentials established AI involvement.',
  unsupported_language: 'The extracted writing was outside English coverage.',
  mixed_language: 'The extracted text contained English and another language.',
  language_unclear: 'The check could not establish the language of the extracted writing.',
  too_little_after_cover: 'Too little text remained after pages with covered text were excluded.',
  unsupported_word_revisions: 'Some Word revisions could change which text is visible.',
  too_little_after_word_revisions: 'Too little text remained after applying the supported Word revisions.',
  no_text: 'There was no extracted text for this check to read.',
}

function EmailInputScope({ scope }: { scope: EmailClassifierInputScope }) {
  return <section className="mt-4 rounded-md border border-border p-4" aria-labelledby="email-input-scope-title">
    <h3 id="email-input-scope-title" className="m-0 text-body font-medium">What the email writing checks read</h3>
    <p className="mt-2 mb-0 text-caption text-text-muted">Selected MIME part {scope.mime_part}. Counts cover retained complete classifier inputs. Each input can contain several sentences.</p>
    {scope.mapping === 'unmapped' && <p className="mt-2 mb-0 text-small">The source mapping could not separate message text, quotes and signatures.</p>}
    {scope.checks.length > 0 ? <div className="mt-2 overflow-x-auto">
      <table className="w-full text-left text-small">
        <thead><tr><th scope="col" className="py-1 pr-3">Check</th>
          {emailInputRoles.map(([role, label]) => <th key={role} scope="col" className="py-1 pr-3">{label}</th>)}
          <th scope="col" className="py-1">Unlisted</th>
        </tr></thead>
        <tbody>{scope.checks.map((check) => <tr className="border-t border-border" key={check.signal_ref.id}>
          <th scope="row" className="py-1 pr-3 font-normal">{check.signal_ref.display_id}</th>
          {emailInputRoles.map(([role]) => <td key={role} className="py-1 pr-3 font-mono">{check.counts[role]}</td>)}
          <td className="py-1 font-mono">{check.unlisted}</td>
        </tr>)}</tbody>
      </table>
    </div> : <p className="mt-2 mb-0 text-small">No complete local classifier inputs were retained for this email.</p>}
    <p className="mt-2 mb-0 text-caption text-text-muted">Roles come from the decoded message structure. Unlisted inputs exceed the display bounds or lack complete token records. These counts describe coverage.</p>
  </section>
}

function PassagePosition({ review }: { review: DocumentPassagePosition }) {
  if ('contexts' in review) {
    const selected = review.selected == null ? null : review.contexts[review.selected]
    const unchecked = review.contexts.filter(row => !row.readings).map(row => row.sentences[1])
    const labels = ['Previous sentence', 'Selected sentence', 'Following sentence']
    return <section id="document-passage-position" aria-labelledby="document-passage-position-title" className="mt-4 rounded-md border border-border p-4">
      <h3 id="document-passage-position-title" className="m-0 text-body font-medium">Longer-passage writing check</h3>
      <p className="mt-2 mb-0 text-small">{review.contexts.length} eligible sentence centers across {review.paragraphs} paragraphs and {review.sentence_count} sentences.</p>
      <p className="mt-2 mb-0 text-small">{review.status === 'flagged'
        ? 'The selected sentence showed a stronger AI-like writing response than its immediate neighbors. Review the wording in context.'
        : review.status === 'unassessed' ? 'Complete responses were unavailable for the eligible centers.'
        : review.status === 'partial' ? `Completed ${review.contexts.length - unchecked.length} of ${review.contexts.length} eligible centers. No sentence was selected because the check was incomplete.`
        : 'No sentence was selected among the eligible centers. Authorship remains undetermined.'}</p>
      {review.status === 'partial' && <p className="mt-2 mb-0 text-small">
        Original character ranges without complete readings{' '}
        <span className="font-mono">{unchecked.map(row => `[${row.source_start}, ${row.source_end})`).join(', ')}</span>.
      </p>}
      {selected && <>
        <blockquote className="my-3 border-l-2 border-border pl-3 break-words text-small">{review.focus_text}</blockquote>
        <ul className="mt-2 mb-0 list-none p-0 text-small">{selected.sentences.map((row, i) =>
          <li key={row.source_start}>{labels[i]} <span className="font-mono">[{row.source_start}, {row.source_end})</span></li>)}</ul>
      </>}
      <p className="mt-2 mb-0 text-caption text-text-muted">Positions count Unicode characters in the original text. This optional check carries zero weight in the overall assessment.</p>
    </section>
  }
  const labels = ['Opening sentence', 'Middle sentence', 'Closing sentence']
  const locator = review.source_locator
  return <section id="document-passage-position" aria-labelledby="document-passage-position-title" className="mt-4 rounded-md border border-border p-4">
    <h3 id="document-passage-position-title" className="m-0 text-body font-medium">Three-sentence writing check</h3>
    <p className="mt-2 mb-0 text-small">{review.status === 'flagged'
      ? 'The middle sentence showed a stronger AI-like writing response than both neighbors. This check marked that sentence for inspection.'
      : review.status === 'unassessed' ? 'The input met the scope for this check. Complete responses were unavailable.'
      : 'The middle sentence did not reach the review level for a stronger AI-like writing response than both neighbors. This result leaves authorship undetermined.'}</p>
    <blockquote className="my-3 border-l-2 border-border pl-3 break-words text-small">{review.focus_text}</blockquote>
    {locator && <p className="mt-2 mb-0 text-caption text-text-muted">Original character positions in {locator.kind === 'docx'
      ? `Word body paragraph ${locator.paragraph_index + 1}` : `decoded email MIME part ${locator.mime_part}`}.</p>}
    <ul className="mt-2 mb-0 list-none p-0 text-small">{review.sentences.map((row, i) =>
      <li key={row.source_start}>{labels[i]} <span className="font-mono">[{row.source_start}, {row.source_end})</span></li>)}</ul>
    <p className="mt-2 mb-0 text-caption text-text-muted">Positions count Unicode characters in the original text. Each response uses its sentence and immediate neighbors. This check carries zero weight in the overall assessment.</p>
  </section>
}

function MappedFindingLocation({ row }: { row: MappedTextFinding }) {
  const source = row.anchor
  const place = source.kind === 'pdf' ? `PDF page ${source.page}, text block ${source.block}` : source.kind === 'docx'
    ? `Word paragraph ${(source.paragraph_index ?? 0) + 1}` : source.kind === 'email' ? `Email body part ${source.mime_part}` : 'Plain text'
  return <>
    <p className="mt-1 mb-0 text-caption text-text-muted">{place} · Extracted characters [{row.start}, {row.end})</p>
    {source.body_path?.includes('control') && <p className="mt-1 mb-0 text-caption text-text-muted">
      Word body path {source.body_path.join(' / ')}
    </p>}
    {'ranges' in row.location && <p className="mt-1 mb-0 text-caption text-text-muted">
      Original text positions {row.location.ranges.map(([start, end]) => `[${start}, ${end})`).join(', ')}
    </p>}
  </>
}

function CalendarConsistency({ review }: { review: NonNullable<DocumentEvidenceSet['calendar_consistency']> }) {
  if (!review.checked_dates) return null
  return <section id="document-calendar-consistency" aria-labelledby="document-calendar-heading" className="mt-4 rounded-md border border-border p-4 scroll-mt-20">
    <h3 className="m-0 text-body font-medium" id="document-calendar-heading">Calendar date check</h3>
    <p className="mt-2 mb-0 text-small">Checked {review.checked_dates} explicit {review.checked_dates === 1 ? 'weekday and date pair' : 'weekday and date pairs'}. Found {review.disagreements} {review.disagreements === 1 ? 'disagreement' : 'disagreements'}.</p>
    {review.locations.map(row =>
      <article key={row.start} className="mt-3 border-t border-border pt-3">
        <blockquote className="m-0 break-words text-small"><mark>{row.text}</mark></blockquote>
        <p className="mt-2 mb-0 text-small">{row.reason === 'weekday_mismatch'
          ? `The stated date fell on ${row.expected_weekday}. The text says ${row.stated_weekday}.`
          : 'The stated day does not exist in that month and year on the Gregorian calendar.'}</p>
        <MappedFindingLocation row={row} />
      </article>)}
    {review.disagreements > review.locations.length && <p className="mt-2 mb-0 text-caption text-text-muted">
      Showing the first {review.locations.length} disagreements.</p>}
    <p className="mt-2 mb-0 text-caption text-text-muted">{review.limitation}</p>
  </section>
}

function ArithmeticConsistency({ review }: { review: NonNullable<DocumentEvidenceSet['arithmetic_consistency']> }) {
  if (!review.checked_equations) return null
  return <section id="document-arithmetic-consistency" aria-labelledby="document-arithmetic-heading" className="mt-4 rounded-md border border-border p-4 scroll-mt-20">
    <h3 id="document-arithmetic-heading" className="m-0 text-body font-medium">Arithmetic check</h3>
    <p className="mt-2 mb-0 text-small">Checked {review.checked_equations} explicit {review.checked_equations === 1 ? 'equation' : 'equations'}. Found {review.disagreements} {review.disagreements === 1 ? 'disagreement' : 'disagreements'}.</p>
    {!!review.length_conversions_checked && <p className="mt-2 mb-0 text-small">Includes {review.length_conversions_checked} {review.length_conversions_checked === 1 ? 'length conversion' : 'length conversions'}. Expected values use standard length units and the precision shown in the document.</p>}
    {review.locations.map(row =>
      <article key={row.start} className="mt-3 border-t border-border pt-3">
        <blockquote className="m-0 break-words text-small"><mark>{row.text}</mark></blockquote>
        <p className="mt-2 mb-0 text-small">The calculation gives {row.calculated}. The document states {row.stated}.</p>
        <MappedFindingLocation row={row} />
      </article>)}
    {review.disagreements > review.locations.length && <p className="mt-2 mb-0 text-caption text-text-muted">
      Showing the first {review.locations.length} disagreements.</p>}
    <p className="mt-2 mb-0 text-caption text-text-muted">{review.limitation}</p>
  </section>
}

function PercentageConsistency({ review }: { review: NonNullable<DocumentEvidenceSet['percentage_consistency']> }) {
  return <section aria-labelledby="document-percentage-consistency" className="mt-4 rounded-md border border-border p-4">
    <h3 className="m-0 text-body font-medium" id="document-percentage-consistency">Repeated percentage statements</h3>
    <p className="mt-2 mb-0 text-small">{review.scope}</p>
    <p className="mt-2 mb-0 text-small">{review.status === 'different_values'
      ? `${review.different_claims} repeated ${review.different_claims === 1 ? 'statement uses' : 'statements use'} different values.`
      : review.status === 'checked' ? 'No differently stated values were found in the eligible repeated sentences.'
        : 'No eligible percentage statement was repeated across separate paragraphs.'}</p>
    {review.groups.map(group => <div key={group[0].start} className="mt-3 rounded-md border border-border p-3">
      {group.map(row => {
        const chars = Array.from(row.text)
        const start = row.quantity_start - row.start
        const end = row.quantity_end - row.start
        const source = row.anchor
        const location = source?.kind === 'pdf' ? `PDF page ${source.page}` : source?.kind === 'docx'
          ? `Word body paragraph ${source.body_paragraph}` : source?.kind === 'email' ? `Email body part ${source.mime_part}` : null
        const native = row.value_location && 'ranges' in row.value_location
          ? row.value_location.ranges.map(([first, last]) => `[${first}, ${last})`).join(', ') : null
        return <div key={row.start} className="mb-2 last:mb-0">
          <blockquote className="m-0 break-words text-small">{chars.slice(0, start).join('')}<mark>{chars.slice(start, end).join('')}</mark>{chars.slice(end).join('')}</blockquote>
          <p className="mt-1 mb-0 font-mono text-caption text-text-muted">{location && `${location} · `}Statement [{row.start}, {row.end}) · Value [{row.quantity_start}, {row.quantity_end})</p>
          {native && <p className="mt-1 mb-0 font-mono text-caption text-text-muted">Original value position {native}</p>}
        </div>
      })}
    </div>)}
    {review.different_claims > review.groups.length && <p className="mt-2 mb-0 text-caption text-text-muted">
      {review.different_claims - review.groups.length} additional statements exceed the display limit.</p>}
    <p className="mt-2 mb-0 text-caption text-text-muted">{review.limitation} Positions use extracted Unicode characters.</p>
  </section>
}

function PublicationIdentifiers({ review }: { review: NonNullable<DocumentEvidenceSet['publication_identifier_consistency']> }) {
  return <section aria-labelledby="document-publication-identifiers" className="mt-4 rounded-md border border-border p-4">
    <h3 id="document-publication-identifiers" className="m-0 text-body font-medium">Publication identifier check</h3>
    <p className="mt-2 mb-0 text-small">{review.scope}</p>
    <p className="mt-2 mb-0 text-small">Checked {review.checked} {review.checked === 1 ? 'identifier' : 'identifiers'}.
      {' '}Found {review.disagreements} {review.disagreements === 1 ? 'disagreement' : 'disagreements'}.</p>
    {review.locations.map(row => {
      const source = row.anchor
      const place = {
        pdf: `PDF page ${source.page}`,
        docx: `Word body block ${Number(source.body_path?.[0] ?? 0) + 1}`,
        email: `Email body part ${source.mime_part}`,
        text: 'Plain text',
        decoded_html: 'Decoded article HTML',
        odt_paragraph: `Decoded OpenDocument paragraph ${(source.paragraph_index ?? 0) + 1}`,
        rtf_paragraph: `Decoded Rich Text block ${(source.paragraph_index ?? 0) + 1}`,
      }[source.kind]
      return <article key={row.start} className="mt-3 border-t border-border pt-3">
        <blockquote className="m-0 break-words text-small"><mark>{row.text}</mark></blockquote>
        <p className="mt-2 mb-0 text-small">The {row.kind} ends in {row.written}. The preceding digits give a check digit of {row.expected}.</p>
        <p className="mt-1 mb-0 text-caption text-text-muted">{place} · Extracted characters [{row.start}, {row.end})</p>
        {'ranges' in row.location && <p className="mt-1 mb-0 text-caption text-text-muted">
          Original text positions {row.location.ranges.map(([start, end]) => `[${start}, ${end})`).join(', ')}
        </p>}
        {'boxes' in row.location && <p className="mt-1 mb-0 text-caption text-text-muted">
          PDF text bounds in page points {row.location.boxes.map(box => `[${box.map(value => value.toFixed(2)).join(', ')}]`).join(', ')}
        </p>}
      </article>
    })}
    {review.disagreements > review.locations.length && <p className="mt-2 mb-0 text-caption text-text-muted">
      Showing the first {review.locations.length} disagreements.</p>}
    <p className="mt-2 mb-0 text-caption text-text-muted">{review.limitation}</p>
  </section>
}

function DoiSourcePosition({ label, start, end, location }: {
  label: string; start: number; end: number; location: MappedTextFinding['location']
}) {
  return <p className="mt-1 mb-0 text-caption text-text-muted">{label} · Extracted characters [{start}, {end})
    {'ranges' in location
      ? ` · Original characters ${location.ranges.map(([first, last]) => `[${first}, ${last})`).join(', ')}`
      : ` · PDF text bounds in page points ${location.boxes.map(box => `[${box.map(value => value.toFixed(2)).join(', ')}]`).join(', ')}`}
  </p>
}

function DoiReferenceCheck({ review }: { review: NonNullable<DocumentEvidenceSet['doi_reference_consistency']> }) {
  return <section id="document-doi-review" aria-labelledby="document-doi-heading" className="mt-4 rounded-md border border-border p-4 scroll-mt-20">
    <h3 id="document-doi-heading" className="m-0 text-body font-medium">DOI reference check</h3>
    <p className="mt-2 mb-0 text-small">{review.scope}</p>
    <p className="mt-2 mb-0 text-small">{review.references} bibliography {review.references === 1 ? 'entry' : 'entries'}.
      {' '}{review.consistent} matched, {review.different_work} pointed to a different work, and {review.unassessed} {review.unassessed === 1 ? 'was' : 'were'} unassessed.</p>
    {review.locations.map(row => {
      const anchor = row.anchor
      const place = anchor.kind === 'docx' ? `Word body path ${anchor.body_path?.join(' / ') ?? ''}`
        : anchor.kind === 'pdf' ? `PDF page ${anchor.page}` : anchor.kind === 'email' ? `Email body part ${anchor.mime_part}`
        : anchor.kind === 'odt_paragraph' ? `Decoded OpenDocument paragraph ${(anchor.paragraph_index ?? 0) + 1}`
        : anchor.kind === 'rtf_paragraph' ? `Decoded Rich Text block ${(anchor.paragraph_index ?? 0) + 1}`
        : anchor.kind === 'decoded_html' ? 'Decoded article HTML' : 'Plain text'
      return <article key={row.start} className="mt-3 space-y-2 border-t border-border pt-3">
        <blockquote className="m-0 break-words text-small"><mark>{row.text}</mark> · Cited title “{row.cited_title}”</blockquote>
        <dl className="m-0 grid gap-3 text-small sm:grid-cols-2">
          <div className="min-w-0 break-words">
            <dt className="font-medium">Work registered under the printed DOI</dt>
            <dd className="m-0">{row.printed.title}</dd>
          </div>
          <div className="min-w-0 break-words">
            <dt className="font-medium">Work matching the title and first author</dt>
            <dd className="m-0">{row.cited.title} · <span className="font-mono">{row.cited.doi}</span></dd>
          </div>
        </dl>
        <div>
          <p className="m-0 text-caption text-text-muted">{place}</p>
          <DoiSourcePosition label="Printed DOI" start={row.start} end={row.end} location={row.location} />
          <DoiSourcePosition label="Cited title" start={row.title_start} end={row.title_end} location={row.title_location} />
        </div>
      </article>
    })}
    {review.different_work > review.locations.length && <p className="mt-2 mb-0 text-caption text-text-muted">
      Showing {review.locations.length} of {review.different_work} different-work observations.</p>}
    <p className="mt-2 mb-0 text-caption text-text-muted">Local {review.snapshot.provider} snapshot dated {review.snapshot.as_of} with {review.snapshot.records} registry records. {review.limitation}</p>
  </section>
}

function ChatFormatMarkers({ review }: { review: NonNullable<NonNullable<DocumentEvidenceSet['coverage']>['chat_format_markers']> }) {
  if (!review.turns) return null
  return <section aria-labelledby="document-chat-format-markers" className="mt-4 rounded-md border border-border p-4">
    <h3 id="document-chat-format-markers" className="m-0 text-body font-medium">Chat formatting markers</h3>
    <p className="mt-2 mb-0 text-small">Found {review.turns} {review.turns === 1 ? 'message' : 'messages'} with a matching opening and closing marker.</p>
    <ol className="mt-3 mb-0 list-none space-y-3 p-0">
      {review.locations.map(({ opener, closer }) =>
        <li key={opener.start} className="space-y-2 rounded-md border border-border p-3">
          {([['Opening', opener], ['Closing', closer]] as const).map(([label, marker]) => {
            const anchor = marker.anchor
            const source = anchor.kind === 'pdf' ? `PDF page ${anchor.page}, text block ${anchor.block}`
              : anchor.kind === 'docx' ? `Word body path ${anchor.body_path.join(' / ')}`
              : anchor.kind === 'email' ? `Decoded email part ${anchor.mime_part}` : 'Decoded source'
            return <div key={marker.start}>
              <p className="m-0 break-all text-small">{label} <code>{marker.literal}</code></p>
              <p className="mt-1 mb-0 text-caption text-text-muted">{source}
                {marker.ranges && ` · characters ${marker.ranges.map(([a, b]) => `[${a}, ${b})`).join(', ')}`}
                {' '}· Extracted characters [{marker.start}, {marker.end})</p>
            </div>
          })}
        </li>)}
    </ol>
    {review.turns > review.locations.length && <p className="mt-2 mb-0 text-caption text-text-muted">
      {review.turns - review.locations.length} further messages exceed the display limit.</p>}
    <p className="mt-2 mb-0 text-caption text-text-muted">{review.limitation} Positions are zero-based and end offsets are exclusive.</p>
  </section>
}

function EmailBodyComparison({ review }: { review?: EmailBodyAlternatives }) {
  if (!review) return null
  return <section id="email-body-alternatives" aria-labelledby="email-body-alternatives-heading" className="mt-4 rounded-md border border-border p-4">
    <h3 id="email-body-alternatives-heading" className="m-0 text-body font-medium">Email body rendition comparison</h3>
    <p className="mt-2 mb-0 text-small">{review.status === 'difference'
      ? 'Some wording differs between the plain-text and HTML bodies. Ordinary email formatting and different versions of a message can produce these differences.'
      : review.status === 'equivalent' ? 'No material difference in the text checked.'
      : 'The email body renditions could not be compared within the supported scope.'}</p>
    <p className="mt-2 mb-0 text-caption text-text-muted">This comparison covers text only and carries zero AI confidence weight.</p>
    {review.differences?.map((change) => <article key={change.id} id={`email-body-alternatives-${change.id}`} className="mt-3 rounded-md border border-border p-3">
      <h4 className="m-0 text-small font-medium">{change.id} · Wording difference</h4>
      <div className="mt-2 grid gap-3 sm:grid-cols-2">{(['plain', 'html'] as const).map((side) => {
        const excerpt = change[side]
        return <div key={side}>
          <p className="m-0 text-small font-medium">{side === 'plain' ? 'Plain-text body' : 'HTML body text'}</p>
          {excerpt ? <><p className="mt-1 mb-0 break-words text-small">{excerpt.text}</p>
            <details className="mt-1 text-caption text-text-muted"><summary className="cursor-pointer">MIME part {review.parts?.[side].id} · decoded character ranges</summary>
              <p className="mb-0 break-words">{excerpt.source_ranges.map(([start, end]) => `[${start}, ${end})`).join(', ')}.{excerpt.excerpt_truncated && ' Excerpt shortened.'}</p>
            </details></> : <p className="mt-1 mb-0 text-caption text-text-muted">No corresponding wording change recorded.</p>}
        </div>
      })}</div>
    </article>)}
    {!!review.omitted_differences && <p className="mt-2 mb-0 text-caption text-text-muted">{review.omitted_differences} further differences exceed the excerpt display limit.</p>}
  </section>
}

function ScoredContext({ window, index }: { window: DocumentEvidenceWindow; index: number }) {
  const ratio = window.score_kind === 'perplexity_ratio'
  const supports = windowInspectionDirection(window) === true
  const matched = typeof window.matched_context_selected === 'boolean'
  const eligible = typeof window.score_exact_hex === 'string'
  return <div className="rounded-md border border-border p-4">
    <p className="m-0 text-small"><span className="font-medium">{window.signal_ref.display_id ?? `Check ${index + 1}`}</span>
      {' '}· <span className="font-mono">{window.score.toFixed(3)}</span> {ratio ? 'raw writing-pattern ratio' : 'uncalibrated response'}
      {' '}· {matched ? (supports ? 'AI-like context selected for review' : eligible ? 'Context did not select this sentence' : 'Outside short-passage review coverage')
        : supports ? 'AI-like response for this input' : 'No AI-like flag for this input'}</p>
    <p className="mt-2 mb-0 text-caption text-text-muted">{matched ? (eligible
      ? 'The check read three sentences together and uses the middle sentence as a review location. The response is an uncalibrated writing-pattern measurement.'
      : 'This response was recorded for the original writing check. Short-passage review covers only eligible middle sentences.')
      : ratio ? 'This ratio compares how predictable the text was with a reference based on token predictions. It carries no passage-level authorship confidence.'
      : 'This raw response measures writing patterns across the input. It carries no passage-level authorship confidence.'}</p>
    {ratio && typeof window.mean_negative_log_likelihood === 'number' && typeof window.mean_cross_entropy === 'number'
      && <p className="mt-2 mb-0 text-caption text-text-muted">
        Mean surprise for the observed tokens <span className="font-mono">{window.mean_negative_log_likelihood.toFixed(4)}</span>
        {' '}÷ mean reference cross-entropy <span className="font-mono">{window.mean_cross_entropy.toFixed(4)}</span>.
        Values are rounded for display.
      </p>}
    {ratio && <p className="mt-3 mb-0 text-caption text-text-muted">Source text covering the actual token contexts is shown below. The recorded token sequences and scored token ranges are preserved in JSON; the excerpt alone does not reproduce the token input.</p>}
    {ratio ? <details className="mt-3 text-small"><summary className="cursor-pointer text-text-muted">Full source text for the recorded token contexts ({Array.from(window.text).length.toLocaleString()} characters)</summary>
      <blockquote className="m-0 mt-3 pl-3 border-l border-border whitespace-pre-wrap break-words">{window.text}</blockquote>
    </details> : <blockquote className="m-0 mt-3 pl-3 border-l border-border text-small whitespace-pre-wrap break-words">{window.text}</blockquote>}
    <details className="mt-3 text-caption text-text-muted"><summary className="cursor-pointer">Input coverage and record</summary>
      <p className="mb-0">{ratio ? 'Source envelope' : 'Sentence ranges'}: {window.spans.map((span) => `[${span.start}, ${span.end})`).join(', ')}.
        {!ratio && ' Sentences are joined by a single space for this check.'}</p>
      {typeof window.token_coverage.tokens === 'number' && <p className="mb-0">{window.token_coverage.tokens} {ratio ? 'selected tokens used for this response' : 'content tokens'}{typeof window.token_coverage.token_windows === 'number' ? ` across ${window.token_coverage.token_windows} token window(s)` : ''}.</p>}
      <p className="font-mono break-all">{ratio ? 'Source envelope' : 'Input'} SHA-256: {window.input_sha256}</p>
      {window.inference_contexts?.map((context) => <div key={context.id} className="mt-3 border-t border-border pt-2">
        <p>Actual token context [{context.source_token_range.join(', ')}); predicted tokens in that forward pass [{context.scored_token_range.join(', ')}).
          {' '}{context.anchor_token_count} non-source anchor token(s); context limit {context.context_limit}, stride {context.stride}.</p>
        <p className="font-mono break-all">Token sequence SHA-256: {context.token_ids_sha256}</p>
      </div>)}
      {window.scored_spans && <p>Response reduction covers source characters {window.scored_spans.map((span) => `[${span.start}, ${span.end})`).join(', ')}
        {window.tokens_scored != null && ` and ${window.tokens_scored} recorded tokens`}. Surrounding input can influence the response.
        {window.neighbor_expansion && ' Neighbouring sentences are included because the focus sentence is short.'}</p>}
    </details>
  </div>
}

function PagePlates({ set, packet, page, token, mediaReady }: { set: DocumentEvidenceSet; packet: DocumentEvidencePacket; page: number; token: string; mediaReady: boolean }) {
  const assets = Object.values(packet.artifacts).filter((asset) => asset.page === page)
  const [highlight, setHighlight] = useState(() => assets.some((asset) => asset.role === 'overlay'))
  const context = assets.find((asset) => asset.role === 'context')
  const crop = assets.find((asset) => asset.role === (highlight ? 'overlay' : 'crop'))
  const ocrInput = assets.find((asset) => asset.role === 'ocr_input')
  const area = packet.inspection_scope?.basis === 'scored_context' ? 'scored context'
    : packet.inspection_scope?.basis === 'ratio_reduction' ? 'measured ratio range' : 'excerpt'
  if (!context || !crop) return null
  return <div className="mt-5">
    <div className="flex flex-wrap justify-between gap-2 items-center mb-2"><h4 className="m-0 text-small font-medium">Page {page}</h4>
      {assets.some((asset) => asset.role === 'overlay') && <Button variant="text" aria-pressed={highlight} onClick={() => setHighlight((value) => !value)}>
        {highlight ? 'Show original page crop' : `Show ${area} highlight`}</Button>}</div>
    <div className={evidenceGrid}>
      <figure className="m-0 min-w-0"><EvidencePicture set={set} artifact={context} token={token} ready={mediaReady} alt={`${packet.id}: page ${page} with the ${area} position marked`} />
        <figcaption className="mt-2 text-caption text-text-muted">Source page with the {area} outlined.</figcaption></figure>
      <figure className="m-0 min-w-0"><EvidencePicture set={set} artifact={crop} token={token} ready={mediaReady} alt={`${packet.id}: page ${page} ${highlight ? `${area} highlight` : 'original crop'}`} />
        <figcaption className="mt-2 text-caption text-text-muted">{highlight ? 'Highlights locate extracted words; they do not identify which words were generated.' : 'Original page crop. Its layout is source context, not an authorship signal.'}</figcaption></figure>
    </div>
    {ocrInput && <details className="mt-3 text-small text-text-muted"><summary className="cursor-pointer">Exact page image used for text recognition</summary>
      <div className="mt-3 max-w-xl"><EvidencePicture set={set} artifact={ocrInput} token={token} ready={mediaReady} alt={`${packet.id}: page ${page} exact frozen text-recognition input`} /></div>
      <p className="mt-2 mb-0 text-caption">This is the saved page image that produced the extracted text. Compare wording with the page before relying on a writing-style response; a transcription error is not evidence of AI authorship.</p>
    </details>}
  </div>
}

function PublishedWording({ review, method }: { review: PublishedWordingReview; method?: string }) {
  const columns = review.schema === 'document-published-wording-review/2'
  const sections = columns ? review.sections : [review]
  const earlierSource = review.schema === 'document-published-wording-review/4'
    ? 'a locally retained copy of the cited federal release' : 'the cited U.S. Government Accountability Office highlight'
  return <section id="document-published-wording" aria-labelledby="published-wording-title" className="mt-4 rounded-md border border-border p-4">
    <h3 id="published-wording-title" className="m-0 text-body font-medium">Earlier published wording</h3>
    <p className="mt-2 mb-0 text-small">This check compares the marked text with {earlierSource}. Wording differences can result from ordinary editing, transcription or AI assistance. The comparison contributes zero weight to the overall AI assessment.</p>
    {columns && <p className="mt-2 mb-0 text-small">Only the cited spans in the listed columns were compared. The check sets aside bullets, matching wrapped words, and standalone headings without an earlier counterpart.</p>}
    {method === 'odt' && <p className="mt-2 mb-0 text-small">Locations count characters within extracted OpenDocument body paragraphs in reading order.</p>}
    {sections.map(section => <WordingSection key={'column' in section ? `${section.page}-${section.column}` : section.reference.id}
      section={section} matches={review.published_style_matches} />)}
  </section>
}

function WordingLocations({ locations, column }: {
  locations: PublishedWordingSection['changes'][number]['locations']; column?: string
}) {
  if (column) return locations.map(place => 'page' in place && <p key={place.start} className="m-0 text-caption">
    Original PDF page {place.page} · Extracted characters <span className="font-mono">[{place.start}, {place.end})</span>. Word boxes are recorded in JSON.
  </p>)
  const pages = [...new Set(locations.flatMap(place => 'page' in place ? [place.page] : []))]
  return pages.length ? <p className="m-0 text-caption">Original PDF pages {pages.join(', ')}. Word boxes are recorded in JSON.</p>
    : locations.map(place => {
      if (!('ranges' in place)) return null
      const label = place.coordinate_space === 'extracted-paragraph-codepoints'
        ? `Extracted OpenDocument paragraph ${place.paragraph} characters`
        : place.paragraph ? `Word paragraph ${place.paragraph} characters` : 'Original text characters'
      return <p key={place.start} className="m-0 text-caption">
        {label} <span className="font-mono">{place.ranges.map(([start, end]) => `[${start}, ${end})`).join(', ')}</span>.
      </p>
    })
}

function WordingSection({ section, matches }: {
  section: PublishedWordingSection | PublishedWordingColumn
  matches: PublishedWordingReview['published_style_matches']
}) {
  const source = section.reference
  const transcript = Array.from(source.text)
  const earlier = ([start, end]: [number, number]) => transcript.slice(start, end).join('')
  const column = 'column' in section ? section.column : undefined
  const sentences = section.changed_sentences ?? []
  const isRelease = 'publisher' in source
  return <div className="mt-3">
    {'column' in section && <h4 className="m-0 text-small font-medium">Original PDF page {section.page} · {section.column === 'both' ? 'Both columns, left to right' : `${section.column === 'left' ? 'Left' : 'Right'} column`}</h4>}
    {isRelease ? <>
      <p className="mt-2 mb-0 text-small"><a className="underline" href={source.source_url}>{source.title}</a> · {source.publisher === 'bea' ? 'U.S. Bureau of Economic Analysis' : 'U.S. Census Bureau'} · Published {source.published_date}</p>
      <p className="mt-2 mb-0 text-caption text-text-muted break-words">Retained publisher page SHA-256 <span className="font-mono">{source.source_record_sha256}</span>. The current publisher page may include later corrections.</p>
    </> : <p className="mt-2 mb-0 text-small"><a className="underline" href={`https://www.gao.gov/products/${encodeURIComponent(source.id)}`}>U.S. GAO report {source.id}</a> · Published {source.published_date}</p>}
    <p className="mt-2 mb-0 text-caption text-text-muted">Compared {column ? 'column' : 'extracted'} characters <span className="font-mono">[{section.submitted_range[0]}, {section.submitted_range[1]})</span>. {isRelease
      ? <a className="underline" href={source.publisher === 'bea' ? 'https://www.bea.gov/help/faq/145' : 'https://www.census.gov/about/policies/citation.html'}>Publisher reuse and citation guidance</a>
      : <><a className="underline" href={source.version_url}>GovReport transcript</a> · <a className="underline" href="https://gov-report-data.github.io/#license">CC BY 4.0</a></>}</p>
    {!!matches.length && <>
      <h4 className="mt-3 mb-0 text-small font-medium">Writing highlights with exact published wording</h4>
      {matches.map(match => <div key={match.start} className="mt-2 text-small">
        <p className="m-0">Extracted characters <span className="font-mono">[{match.start}, {match.end})</span>. Same wording in the dated source, allowing for spacing.</p>
        <blockquote className="my-2 border-l-2 border-border pl-3 whitespace-pre-line break-words">{earlier(match.archive_range)}</blockquote>
      </div>)}
    </>}
    {!!sentences.length && <>
      <h4 className="mt-4 mb-0 text-small font-medium">Sentences with wording differences</h4>
      {sentences.map(row => <div key={row.sentence} className="mt-3 rounded-md border border-border p-3 text-small">
        <h5 className="m-0 text-small font-medium">Sentence {row.sentence}</h5>
        <blockquote className="my-2 border-l-2 border-border pl-3 whitespace-pre-line break-words">{row.text}</blockquote>
        <WordingLocations locations={row.locations} />
        <p className="mt-2 mb-0 flex flex-wrap gap-x-3 text-caption">
          {row.differences.map(number => <a key={number} className="underline" href={`#published-wording-difference-${number}`}>Difference {number}</a>)}
        </p>
      </div>)}
    </>}
    <h4 className="mt-3 mb-0 text-small font-medium">Wording differences</h4>
    {!section.changes.length && <p className="mt-2 mb-0 text-small">No bounded wording differences were recorded.</p>}
    {section.changes.map((change, index) =>
      <div key={`${change.start}-${change.end}-${change.archive_range[0]}`} id={sentences.length ? `published-wording-difference-${index + 1}` : undefined}
        className="mt-3 scroll-mt-20 rounded-md border border-border p-3 text-small">
        {!!sentences.length && <h5 className="m-0 mb-2 text-small font-medium">Difference {index + 1}</h5>}
        {!column && <p className="m-0 text-caption">{change.start === change.end
          ? `Omission at extracted character ${change.start}. Locations show the next retained character.`
          : <>Extracted characters <span className="font-mono">[{change.start}, {change.end})</span>.</>}</p>}
        {column && change.start === change.end && <p className="m-0 text-caption">Omission before the indicated text.</p>}
        <WordingLocations locations={change.locations} column={column} />
        <dl className="mt-2 mb-0 grid gap-2 sm:grid-cols-2">
          <div className="min-w-0"><dt className="font-medium">Earlier wording</dt><dd className="m-0 break-words whitespace-pre-line">{earlier(change.archive_range) || 'No counterpart in this part of the earlier source.'}</dd></div>
          <div className="min-w-0"><dt className="font-medium">Submitted wording</dt><dd className="m-0 break-words whitespace-pre-line">{change.submitted || 'Omitted at this position.'}</dd></div>
        </dl>
      </div>)}
  </div>
}

function WritingPassageIntervals({ review, method, emailHtml, wording }: {
  review: NonNullable<NonNullable<DocumentEvidenceSet['coverage']>['writing_passage_intervals']>
  method?: string
  emailHtml: boolean
  wording?: PublishedWordingReview
}) {
  const summary = {
    'document-writing-passage-intervals/3': 'It also looks for passages that stand out within the document.',
    'document-writing-passage-intervals/4': 'This file matched the writing patterns of the economic-report reference set. Each displayed passage also stood out against that reference.',
  } as const
  const approach = review.schema === 'document-writing-passage-intervals/3' || review.schema === 'document-writing-passage-intervals/4'
    ? summary[review.schema]
    : 'Sustained AI-like patterns can highlight separate passages for review.'
  const paragraphLabel = method === 'odt' ? 'Decoded OpenDocument paragraph' : method === 'rtf' ? 'Decoded Rich Text block' : 'Word paragraph'
  const ranges = review.ranges.filter(range => !wording?.published_style_matches.some(match => match.start === range.start && match.end === range.end))
  return <section id="document-writing-passage-intervals" aria-labelledby="writing-passage-intervals-title" className="mt-4 rounded-md border border-border p-4">
    <h3 id="writing-passage-intervals-title" className="m-0 text-body font-medium">Possible AI-like writing passages</h3>
    <p className="mt-2 mb-0 text-small">This optional check combines readings of individual sentences and their surrounding text. {approach}</p>
    {method === 'email' && <p className="mt-2 mb-0 text-small">This check covers the mapped reply body.</p>}
    {(ranges.length > 0 || !wording?.published_style_matches.length) && <p className="mt-2 mb-0 text-small">{review.status === 'flagged' ? 'Inspect the highlighted passages in context.'
      : review.status === 'no_ranges' ? 'No passage was selected. Authorship remains undetermined by this check.'
      : 'Complete readings or original passage positions were unavailable.'}</p>}
    {!!wording?.published_style_matches.length && <p className="mt-2 mb-0 text-small">Passages that match dated published wording appear in the <a className="underline" href="#document-published-wording">source comparison</a>.</p>}
    {ranges.map(range => {
      const pages = 'locations' in range ? [...new Set(range.locations.flatMap(place => 'page' in place ? [place.page] : []))] : []
      return <div key={range.start} className="mt-3">
        {'source_start' in range
          ? <p className="m-0 text-small">Original text characters <span className="font-mono">[{range.source_start}, {range.source_end})</span>.</p>
          : pages.length > 0
            ? <p className="m-0 text-small">Original PDF pages {pages.join(', ')}. Exact word boxes are recorded in JSON.</p>
            : <ul className="m-0 list-none pl-0 text-small">{range.locations.map((place, index) => 'ranges' in place &&
              <li key={index}>{'paragraph' in place ? `${paragraphLabel} ${place.paragraph}` : `Decoded email ${emailHtml ? 'HTML ' : ''}body part ${place.mime_part}`} characters <span className="font-mono">{place.ranges.map(([start, end]) => `[${start}, ${end})`).join(', ')}</span>.</li>
            )}</ul>}
        {range.basis === 'document_baseline' && <p className="m-0 text-small">Selected by comparing this passage with the writing patterns elsewhere in this file.</p>}
        <blockquote className="my-2 border-l-2 border-border pl-3 whitespace-pre-line break-words text-small">{range.preview}{range.end - range.start > Array.from(range.preview).length ? '…' : ''}</blockquote>
      </div>
    })}
    <p className="mt-2 mb-0 text-caption text-text-muted">Positions are zero-based with the end excluded. Ordinary writing can produce similar patterns. This check carries zero weight in the overall assessment.</p>
  </section>
}

export default function DocumentEvidenceContent({ set, token, mediaReady }: { set: DocumentEvidenceSet; token: string; mediaReady: boolean }) {
  const [selectedId, setSelectedId] = useState<string | null>(set.packets[0]?.id ?? null)
  const { hash, key } = useLocation()
  const articles = useRef(new Map<string, HTMLElement>())
  const landed = useRef<string | null>(null)
  useEffect(() => {
    const navigation = `${key}:${hash}:${set.id}`
    if (landed.current === navigation) return
    const packet = set.packets.find((candidate) => hash === `#media-evidence-${encodeURIComponent(candidate.id)}`)
    if (!packet) return
    const article = articles.current.get(packet.id)
    if (!article) return
    landed.current = navigation
    setSelectedId(packet.id)
    article.focus({ preventScroll: true })
    article.scrollIntoView?.({ block: 'start' })
  }, [hash, key, set.id, set.packets])
  const selectPassage = (id: string) => {
    setSelectedId(id)
    const article = articles.current.get(id)
    article?.focus({ preventScroll: true })
    article?.scrollIntoView?.({ block: 'start' })
  }
  const email = !!set.coverage?.email || set.packets.some((packet) => packet.anchors.some((anchor) => anchor.kind === 'email'))
  const emailCoverage = typeof set.coverage?.email === 'object' ? set.coverage.email : null
  const bodyFormat = emailCoverage?.body_content_type === 'text/plain' ? 'plain-text'
    : emailCoverage?.body_content_type === 'text/html' ? 'HTML' : null
  const attachmentCount = emailCoverage?.attachments?.length ?? 0
  const unmapped = [...new Set([...(set.coverage?.unmapped_pages ?? []), ...(set.coverage?.ocr?.unchecked_pages ?? [])])]
  const partialPages = set.coverage?.partially_checked_pages ?? []
  const textVisibility = set.coverage?.pdf_text_visibility
  const wordVisibility = set.coverage?.word_text_visibility
  const invisible = set.coverage?.invisible_characters
  const characterReview = set.coverage?.writing_character_review
  const lookalikes = invisible?.schema === 'document-invisible-characters/2'
  const ocrPages = set.coverage?.ocr?.pages ?? []
  const matched = set.coverage?.matched_context_review
  const writingAgreement = set.coverage?.writing_second_check
  const writingChecks = set.coverage?.writing_check_coverage
  const languageWitness = writingChecks?.schema === 'document-writing-check-coverage/3' ? writingChecks.language_witness : null
  const scoredWindows = set.coverage?.scored_windows
  const balancedSelection = set.coverage?.balanced_passage_selection
  const tracked = set.coverage?.word_tracked_changes
  const savedFields = set.coverage?.word_saved_fields
  const selectablePages = [...new Set(writingAgreement?.pdf_spans?.map((span) => span.page) ?? [])]
  const usesOcr = ocrPages.some((page) => page.status === 'extracted') || set.packets.some((packet) => packet.anchors.some((anchor) => anchor.extraction === 'ocr'))
  return <div data-testid="document-evidence-content">
    {set.coverage?.writing_passage_intervals && <WritingPassageIntervals review={set.coverage.writing_passage_intervals}
      method={set.source.method} emailHtml={emailCoverage?.body_content_type === 'text/html'} wording={set.coverage.published_wording_review} />}
    {set.coverage?.published_wording_review && <PublishedWording review={set.coverage.published_wording_review} method={set.source.method} />}
    {set.coverage?.pdf_saved_wording && <PdfSavedWording review={set.coverage.pdf_saved_wording} set={set} token={token} mediaReady={mediaReady} />}
    <EmailBodyComparison review={set.coverage?.email_body_alternatives} />
    {set.coverage?.email_classifier_input_scope && <EmailInputScope scope={set.coverage.email_classifier_input_scope} />}
    {savedFields && <section aria-label="Saved Word field values" className="mt-4 rounded-md border border-border p-4">
      <h3 className="m-0 text-body font-medium">Saved Word field values</h3>
      <p className="mt-2 mb-0 text-small">{savedFields.included} of {savedFields.found} saved fields included in the extracted text.</p>
      <p className="mt-2 mb-0 text-small text-text-muted">Includes saved values from supported simple fields in ordinary Word body paragraphs.
        {' '}Field formulas and links are never run. The displayed values can be out of date.</p>
      {savedFields.locations.length > 0 && <ul className="mt-3 mb-0 list-none space-y-3 p-0">
        {savedFields.locations.map(row => <li key={row.start} className="border-t border-border pt-3">
          <blockquote className="m-0 break-words text-small">{row.text}</blockquote>
          <p className="mt-2 mb-0 text-caption text-text-muted">{savedFields.part} · Body block {row.body_index}, field child {row.field_index} (zero-based).
            {' '}Word characters [{row.source_start}, {row.source_end}). Extracted characters [{row.start}, {row.end}).</p>
        </li>)}
      </ul>}
    </section>}
    {tracked && <section aria-label="Word tracked text changes" className="mt-4 rounded-md border border-border p-4">
      <h3 className="m-0 text-body font-medium">Word tracked text changes</h3>
      {tracked.writing_input === 'accepted_inline_changes' && <p className="mt-2 mb-0 text-small">Writing checks use the version with these inline insertions included and deletions excluded.</p>}
      {tracked.writing_input === 'accepted_paragraph_changes' && <p className="mt-2 mb-0 text-small">Writing checks use the accepted wording and paragraph breaks.</p>}
      {tracked.writing_input === 'unsupported_tracked_changes' && <p className="mt-2 mb-0 text-small">Some Word revisions could change which text is visible. Automated writing checks were skipped.</p>}
      <p className="mt-2 mb-0 text-small">Checked {tracked.paragraphs} body paragraphs. Found {tracked.inserted} insertions and {tracked.deleted} deletions.</p>
      {!!tracked.unsupported_marks && <p className="mt-2 mb-0 text-small">{tracked.unsupported_marks} additional revision marks were outside scope.</p>}
      <p className="mt-2 mb-0 text-small text-text-muted">{tracked.scope} {tracked.limitation}</p>
      {!!tracked.rows.length && <ol className="mt-3 mb-0 list-none space-y-3 p-0">
        {tracked.rows.map((row) => <li key={`${row.body_index}:${row.inline_index}`} className="rounded-md border border-border p-3">
          <p className="m-0 text-small font-medium">{row.kind === 'inserted' ? 'Inserted' : 'Deleted'} · Body paragraph {row.paragraph}</p>
          <blockquote className="mt-2 mb-0 whitespace-pre-wrap break-words text-small">{row.text}
            {row.characters > Array.from(row.text).length && `… (${row.characters} source characters)`}</blockquote>
          <p className="mt-2 mb-0 text-caption text-text-muted">{tracked.part} · XML body index {row.body_index}, paragraph child index {row.inline_index} (zero-based).</p>
          {row.writing_input_range && <p className="mt-2 mb-0 text-caption text-text-muted">Accepted paragraph characters [{row.writing_input_range[0]}, {row.writing_input_range[1]}).</p>}
        </li>)}
      </ol>}
      {!!tracked.paragraph_marks?.length && <ol className="mt-3 mb-0 list-none space-y-3 p-0">
        {tracked.paragraph_marks.map((mark) => <li key={mark.body_index} className="rounded-md border border-border p-3">
          <p className="m-0 text-small font-medium">{mark.kind === 'inserted' ? 'Inserted' : 'Deleted'} paragraph break · Body paragraph {mark.paragraph}</p>
          <p className="mt-2 mb-0 text-caption text-text-muted">{tracked.part} · XML body index {mark.body_index}, child path {mark.child_path.join(' / ')} (zero-based).</p>
        </li>)}
      </ol>}
      {!!tracked.omitted_excerpts && <p className="mt-2 mb-0 text-caption text-text-muted">{tracked.omitted_excerpts} further excerpts exceed the display limit.</p>}
      {!!tracked.omitted_paragraph_marks && <p className="mt-2 mb-0 text-caption text-text-muted">{tracked.omitted_paragraph_marks} further paragraph breaks exceed the display limit.</p>}
    </section>}
    <DocumentRepeatedPassages result={set.repeated_passages} />
    {set.percentage_consistency && <PercentageConsistency review={set.percentage_consistency} />}
    {set.calendar_consistency && <CalendarConsistency review={set.calendar_consistency} />}
    {set.arithmetic_consistency && <ArithmeticConsistency review={set.arithmetic_consistency} />}
    {set.publication_identifier_consistency && <PublicationIdentifiers review={set.publication_identifier_consistency} />}
    {set.printed_ai_use_heading?.schema === 'chipmunk.printed-ai-use-heading/1' && set.printed_ai_use_heading.affects_overall_confidence === false &&
      <section id="media-evidence-printed-ai-use-heading" aria-labelledby="printed-ai-use-heading-title" className="mt-4 rounded-md border border-border p-4 scroll-mt-20">
        <h3 id="printed-ai-use-heading-title" className="m-0 text-body font-medium">Printed AI-use heading</h3>
        <blockquote className="my-2 border-l-2 border-border pl-3 text-small">{set.printed_ai_use_heading.text}</blockquote>
        <p className="mt-2 mb-0 text-caption text-text-muted">PDF page {set.printed_ai_use_heading.page} · Extracted characters <span className="font-mono">[{set.printed_ai_use_heading.start}, {set.printed_ai_use_heading.end})</span>.</p>
        <p className="mt-2 mb-0 text-small">A printed heading can be part of a quotation or example. This observation carries zero weight in the AI assessment.</p>
      </section>}
    {set.doi_reference_consistency && <DoiReferenceCheck review={set.doi_reference_consistency} />}
    {set.saved_link_hosts && <SavedLinkDestinations review={set.saved_link_hosts} />}
    {set.coverage?.chat_format_markers && <ChatFormatMarkers review={set.coverage.chat_format_markers} />}
    {invisible && <section aria-labelledby="document-invisible-characters" className="mt-4 rounded-md border border-border p-4">
      <h3 id="document-invisible-characters" className="m-0 text-body font-medium">
        {lookalikes ? 'Invisible and look-alike characters in the source text' : 'Invisible characters in the source text'}
      </h3>
      <p className="mt-2 mb-0 text-small">Checked {invisible.checked_chars} extracted characters across {invisible.checked_blocks} mapped blocks.
        {' '}Found {invisible.matches} marked {invisible.matches === 1 ? 'character' : 'characters'}.</p>
      <p className="mt-2 mb-0 text-small text-text-muted">Direction overrides, embedded byte order marks, and selected invisible breaks
        or joiners inside basic Latin letters or digits are in scope.
        {lookalikes ? ' This also checks listed Greek and Cyrillic look-alikes in mostly basic Latin words. Ordinary multilingual text, technical notation and copy and paste can introduce these characters.'
          : ' Ordinary formatting and copy and paste can introduce them.'}
        {' '}This observation carries zero AI-authorship weight.</p>
      {invisible.locations.length > 0 && <ul className="mt-2 mb-0 pl-5 text-small">
        {invisible.locations.map(row => <li key={row.text_offset}>
          {row.looks_like ? <>Looks like <span className="font-mono">{row.looks_like}</span></> : invisibleCharacterNames[row.codepoint]}
          {' '}<span className="font-mono">{row.codepoint}</span> · {row.body_path
            ? `Word body path ${row.body_path.join(' / ')}, paragraph characters`
            : row.mime_part ? `Email part ${row.mime_part}, ${invisible.coordinate_space === 'decoded_email_html'
              ? 'decoded HTML source characters' : 'decoded source characters'}` : 'Decoded source characters'}
          {' '}[{row.source_start}, {row.source_end}) · Extracted offset {row.text_offset}
        </li>)}
      </ul>}
      {!!invisible.locations.length && <p className="mt-2 mb-0 text-caption text-text-muted">Positions are zero-based. End offsets are exclusive.</p>}
      {invisible.matches > invisible.locations.length && <p className="mt-2 mb-0 text-caption text-text-muted">
        {invisible.matches - invisible.locations.length} additional characters exceed the location display limit.
      </p>}
    </section>}
    {!!characterReview?.ranges.length && <section aria-labelledby="document-character-review" className="mt-4 rounded-md border border-border p-4">
      <h3 id="document-character-review" className="m-0 text-body font-medium">Writing response after character adjustment</h3>
      <p className="mt-2 mb-0 text-small">Two checks reread the context after adjusting selected invisible breaks and look-alike letters.
        {' '}Ordinary text can show the same response shift. The excerpts show the submitted text. This review carries zero AI-authorship weight.</p>
      <ul className="mt-2 mb-0 list-none space-y-3 p-0 text-small">
        {characterReview.ranges.map(({ spans, location, changes, excerpt, state }) => {
          const focus = spans[1]
          const source = location.kind === 'pdf' ? `PDF page ${location.page}` : location.kind === 'docx'
            ? `Word body path ${location.body_path[0]}` : location.kind === 'email' ? `Email part ${location.mime_part}` : 'Decoded text'
          return <li key={focus.start} className="border-t border-border pt-3">
            <p className="m-0 font-medium">{{ changed: 'The writing response changed direction after character adjustment.',
              different: 'The adjusted readings gave different directions.', consistent: 'The response direction held after character adjustment.' }[state]}</p>
            <p className="mt-1 mb-0 font-mono text-caption text-text-muted">Extracted passage [{focus.start}, {focus.end}) · {source}
              {'ranges' in location && location.ranges.map(([start, end]) => ` · Source [${start}, ${end})`)}</p>
            <blockquote className="mt-2 mb-0 border-l border-border pl-3">{excerpt}</blockquote>
            <p className="mt-2 mb-0 text-caption text-text-muted">Characters adjusted in the scored context
              {changes.map(({ codepoint, offset }) => ` · ${codepoint} at ${offset}`)}.</p>
          </li>
        })}
      </ul>
    </section>}
    {wordVisibility && <section aria-labelledby="document-word-text-visibility" className="mt-4 rounded-md border border-border p-4">
      <h3 id="document-word-text-visibility" className="m-0 text-body font-medium">Word text visibility</h3>
      <p className="mt-2 mb-0 text-small">Checked {wordVisibility.checked_paragraphs} extracted paragraphs.
        {' '}Found {wordVisibility.hidden_runs} text {wordVisibility.hidden_runs === 1 ? 'run' : 'runs'} marked hidden.</p>
      <p className="mt-2 mb-0 text-small text-text-muted">This check covers direct hidden formatting in body and table text.
        {' '}Word can display hidden text depending on its settings. This check carries zero AI-authorship weight.</p>
      {wordVisibility.locations.length > 0 && <>
        <ul className="mt-2 mb-0 pl-5 text-small">
          {wordVisibility.locations.map(row => <li key={`${row.body_path.join('/')}:${row.run}`}>
            {row.body_path.includes('control') ? 'Word content control' : row.body_path.length > 1
              ? `Table cell (row ${Number(row.body_path.at(-4)) + 1}, column ${Number(row.body_path.at(-2)) + 1})`
              : `Body paragraph ${row.body_paragraph}`} · {row.run_basis === 'extracted' ? 'Extracted run' : 'Run'} {row.run + 1} · Characters {row.source_start + 1}–{row.source_end}
            <p className="mt-1 mb-2 font-mono text-caption text-text-muted">
              {wordVisibility.part} · Body path {row.body_path.join(' / ')} (zero-based)
            </p>
          </li>)}
        </ul>
        {wordVisibility.hidden_runs > wordVisibility.locations.length && <p className="mt-2 mb-0 text-caption text-text-muted">
          {wordVisibility.hidden_runs - wordVisibility.locations.length} additional hidden text runs exceed the location display limit.
        </p>}
      </>}
    </section>}
    {textVisibility && <section aria-labelledby="document-pdf-text-visibility" className="mt-4 rounded-md border border-border p-4">
      <h3 id="document-pdf-text-visibility" className="m-0 text-body font-medium">PDF text visibility</h3>
      <p className="mt-2 mb-0 text-small">Checked {textVisibility.examined_pages.length} of {textVisibility.total_pages} pages for explicitly invisible text.</p>
      {textVisibility.locations.length > 0 && <>
        <ul className="mt-2 mb-0 pl-5 text-small">
          {textVisibility.locations.map(row => <li key={row.page}>
            Page {row.page} · {row.invisible_spans} invisible text {row.invisible_spans === 1 ? 'span' : 'spans'}
            <p className="mt-1 mb-1 font-mono text-caption text-text-muted">{row.boxes.map(box => `[${box.join(', ')}]`).join(', ')}</p>
          </li>)}
        </ul>
        <p className="mt-2 mb-0 text-small text-text-muted">Locations use unrotated PDF points. Searchable scans and accessibility overlays can contain invisible text. This check carries zero AI-authorship weight.</p>
      </>}
      {textVisibility.replacement_examined_pages && <p className="mt-2 mb-0 text-small">
        Compared copyable text with glyph text on {textVisibility.replacement_examined_pages.length} of {textVisibility.total_pages} pages.
      </p>}
      {!!textVisibility.replacement_differences?.length && <>
        <ul className="mt-2 mb-0 pl-5 text-small">
          {textVisibility.replacement_differences.map(row => <li key={row.page}>
            Page {row.page} · Copyable text differs from glyph text
            {row.boxes.length > 0 && <p className="mt-1 mb-1 font-mono text-caption text-text-muted">
              {row.boxes.map(box => `[${box.join(', ')}]`).join(', ')} (unrotated PDF points)
            </p>}
            {row.locations_incomplete && <p className="mt-1 mb-1 text-caption text-text-muted">Locations are approximate or incomplete.</p>}
          </li>)}
        </ul>
        <p className="mt-2 mb-0 text-small text-text-muted">PDF accessibility and OCR corrections can legitimately change copyable text. This check carries zero AI-authorship weight.</p>
      </>}
      {textVisibility.covered_examined_pages && <p className="mt-2 mb-0 text-small">
        Checked {textVisibility.covered_examined_pages.length} of {textVisibility.total_pages} pages for text behind solid covers.
      </p>}
      {!!textVisibility.covered_locations?.length && <>
        <ul className="mt-2 mb-0 pl-5 text-small">
          {textVisibility.covered_locations.map(row => <li key={row.page}>
            Page {row.page} · {row.covered_spans} covered copyable text {row.covered_spans === 1 ? 'span' : 'spans'}
            <p className="mt-1 mb-1 text-caption text-text-muted">Excluded from writing checks.</p>
            <p className="mt-1 mb-1 font-mono text-caption text-text-muted">
              {row.boxes.map(box => `[${box.join(', ')}]`).join(', ')} (unrotated PDF points)
            </p>
          </li>)}
        </ul>
        <p className="mt-2 mb-0 text-small text-text-muted">This check covers simple opaque rectangles in the page content and carries zero AI-authorship weight. It cannot verify the quality of a redaction.</p>
      </>}
    </section>}
    {writingChecks && <section className="mt-4 rounded-md border border-border p-4" aria-labelledby="writing-check-coverage-title">
      <h3 id="writing-check-coverage-title" className="m-0 text-body font-medium">Writing check coverage</h3>
      <ul className="mt-2 mb-0 list-disc pl-5 text-small">
        {([['passage_classifier', 'Passage classifier'], ['writing_pattern_ratio', 'Writing-pattern ratio']] as const)
          .map(([key, name]) => {
            const state = writingChecks.checks[key]
            const reason = writingChecks.schema !== 'document-writing-check-coverage/1' && writingChecks.reasons[key]
            return state && <li key={key}>{name} · {writingCheckLabels[state]}
              {reason && <span className="ml-1 text-text-muted">{writingCheckReasons[reason]}</span>}
            </li>
          })}
      </ul>
      {languageWitness && <div id="writing-language-location" className="mt-3 text-small">
        <p className="m-0">Two overlapping samples in extracted sentence {languageWitness.sentence} indicated another language.</p>
        <p className="mt-1 mb-0 text-caption text-text-muted">
          Marked characters [{languageWitness.range.start}, {languageWitness.range.end}) and supporting context
          {' '}[{languageWitness.context.start}, {languageWitness.context.end}) use zero-based extracted-text offsets.
        </p>
      </div>}
    </section>}
    <DocumentInspectionMap packets={set.packets} selectedId={selectedId} onSelect={selectPassage} />
    {set.coverage?.writing_bibliography && <BibliographyCoverage review={set.coverage.writing_bibliography}
      doiReviewed={!!set.doi_reference_consistency} />}
    <DocumentPassageCoverage coverage={set.coverage?.local_passage_coverage} />
    <DocumentUnselectedResponses responses={set.coverage?.unselected_passage_responses} />
    {set.coverage?.passage_position_review && <PassagePosition review={set.coverage.passage_position_review} />}
    {writingAgreement && <section className="mt-4 rounded-md border border-border p-4 text-small text-text-muted">
      <h3 className="m-0 text-body font-medium text-text">Second writing check</h3>
      <p className="mt-2 mb-0">The second check revisits the exact text behind up to six AI-like responses. This evidence carries zero weight in the overall assessment.</p>
      {writingAgreement.schema !== 'document-writing-second-check/1' && writingAgreement.schema !== 'document-writing-second-check/6' && <p className="mt-2 mb-0">
        This check covers contexts wholly inside the reply body. Text marked as a quote or signature and contexts that cross those boundaries are outside this check.
      </p>}
      {(writingAgreement.schema === 'document-writing-second-check/4' || writingAgreement.schema === 'document-writing-second-check/5') && <p className="mt-2 mb-0">
        The JSON includes exact positions in the decoded HTML body of the selected email part.
      </p>}
      {selectablePages.length > 0 && <p className="mt-2 mb-0">
        The second check uses selectable text on PDF {selectablePages.length === 1 ? 'page' : 'pages'} {selectablePages.join(', ')}.
        It leaves OCR, partially checked pages, and contexts crossing pages outside this check.
      </p>}
      {writingAgreement.ranges.length === 0 ? <p className="mt-2 mb-0">The first check supplied no AI-like contexts for this review.</p>
        : <ul className="mb-0 pl-4 list-disc">{writingAgreement.ranges.map((row) => {
          const packet = set.packets.find((item) => item.focus.start === row.start && item.focus.end === row.end)
          const page = writingAgreement.pdf_spans?.find((span) => span.start <= row.start && row.end <= span.end)?.page
          const range = `${page == null ? '' : `Page ${page} · `}Passage [${row.start}, ${row.end})`
          return <li key={row.start} className="mt-2">
            {packet ? <a href={`#media-evidence-${packet.id}`} onClick={(event) => { event.preventDefault(); selectPassage(packet.id) }}>{range}</a> : range}
            {' · '}{row.state === 'corroborated' ? 'Both checks gave an AI-like response'
              : row.state === 'different' ? 'The checks gave different responses' : 'Second response unavailable'}
            <div>First-check context {row.spans.map((span) => `[${span.start}, ${span.end})`).join(', ')}.
              {!packet && ' No detailed passage card was selected.'}</div>
          </li>
        })}</ul>}
    </section>}
    <div className="mt-4 rounded-md border border-border p-4 text-small text-text-muted">
      <p className="m-0">{scoredWindows == null ? 'Exact scored-context coverage is unavailable.'
        : scoredWindows === 0 ? 'No local writing responses were recorded.'
        : `Recorded ${scoredWindows} scored ${scoredWindows === 1 ? 'context' : 'contexts'}.`}
        {!!scoredWindows && set.coverage?.candidate_spans === 0
          && ' The writing checks selected no passages for review. AI editing may still be present.'}
        {set.packets.length > 0 && ' The marked text shows where the checks measured a pattern. Ordinary editing and polished writing can produce a similar response.'}</p>
      {balancedSelection && <p className="mt-2 mb-0">
        Showing {balancedSelection.displayed_passages} of {set.coverage?.candidate_spans} candidate passages across{' '}
        {balancedSelection.displayed_blocks} of {balancedSelection.candidate_blocks} source blocks with flags.
        The selection reserves space for disagreements.
      </p>}
      {matched && <p className="mt-2 mb-0">Short-passage review selected {matched.selected_sentences} of {matched.eligible_sentences} eligible sentences. This evidence has zero weight in the overall assessment.</p>}
      {unmapped.length > 0 && <p className="mt-2 mb-0">Pages without mapped text: {unmapped.join(', ')}. Those pages have not been cleared.</p>}
      {partialPages.length > 0 && <>
        <p className="mt-2 mb-0">{partialPages.length === 1 ? 'Page' : 'Pages'} {partialPages.map((page) => page.page).join(', ')} {partialPages.length === 1 ? 'has' : 'have'} mapped text and unchecked image areas.</p>
        <details className="mt-3">
          <summary className="cursor-pointer">Unchecked image areas</summary>
          <ul className="pl-4 list-disc">{partialPages.map((page) => <li key={page.page}>
            <strong>Page {page.page}.</strong> {page.reason}{' '}
            <span>Bounds in unrotated PDF points {page.image_boxes.map((box) => `[${box.join(', ')}]`).join(', ')}.</span>
          </li>)}</ul>
        </details>
      </>}
      {usesOcr && <p className="mt-2 mb-0">Some text was recognized from page images. Recognition errors can change a check’s input; compare the excerpt with the page image. Recognition confidence is not AI-detection confidence.</p>}
      {set.coverage?.checks?.some((check) => check.status !== 'recorded') && <details className="mt-3"><summary className="cursor-pointer">Checks with incomplete local evidence</summary>
        <ul className="pl-4 list-disc">{set.coverage.checks.filter((check) => check.status !== 'recorded').map((check) => <li key={check.signal_ref.id}>
          {check.signal_ref.display_id ?? 'Recorded check'}: {check.scored_windows} local context(s) saved.
          {' '}{check.limitation ?? 'Some input contexts were not retained within the evidence size limit; missing context is not a negative finding.'}
        </li>)}</ul>
      </details>}
      {ocrPages.length > 0 && <details className="mt-3"><summary className="cursor-pointer">Scanned-page reading coverage</summary>
        <ul className="pl-4 list-disc">{ocrPages.map((page) => <li key={page.page}>Page {page.page}{page.coverage_scope === 'image_region' && ' image area'}: {page.status === 'extracted' ? 'text extracted for analysis' : 'not checked'}.
          {page.reason && ` ${page.reason}`}{typeof page.mean_recognition_confidence === 'number' && ` Recognition quality: ${page.mean_recognition_confidence.toFixed(1)} out of 100; this is not an AI score.`}</li>)}</ul>
      </details>}
      {bodyFormat && emailCoverage?.body_part && <p className="mt-2 mb-0">The extracted {bodyFormat} body is MIME part {emailCoverage.body_part}.
        {attachmentCount > 0 && ` ${attachmentCount} ${attachmentCount === 1 ? 'attachment was' : 'attachments were'} outside the analysis.`}</p>}
      {email && <p className="mt-2 mb-0">Email quotations and signatures can belong to someone else. Quote boundaries are heuristic; headers and writing scores do not authenticate a sender.</p>}
    </div>
    <div className="mt-6 space-y-7">{set.packets.map((packet) => {
      const pages = [...new Set(Object.values(packet.artifacts).map((asset) => asset.page).filter((page): page is number => typeof page === 'number'))].sort((a, b) => a - b)
      const roles = new Set(packet.anchors.map((anchor) => anchor.quote_role))
      const includesQuotedText = roles.has('quoted') || roles.has('signature')
      const directions = packet.supporting_windows.map(windowInspectionDirection)
      return <article key={packet.id} id={`media-evidence-${packet.id}`} tabIndex={-1}
        ref={(node) => { if (node) articles.current.set(packet.id, node); else articles.current.delete(packet.id) }}
        aria-label={`${packet.id}: source passage and reasons`} data-selected={selectedId === packet.id ? 'true' : undefined}
        className={`${evidenceCard} focus-visible:outline-2 focus-visible:outline-accent`}>
        <h3 className="m-0 text-body font-medium"><span className="font-mono text-accent mr-3">{packet.id}</span>{packet.title}</h3>
        {set.coverage?.isolated_paragraph_priority?.packet_id === packet.id && <p className="mt-2 mb-0 text-small text-text-muted">
          This passage appears first because the AI-like response persisted when the same check read the whole paragraph. This ordering carries zero weight in the overall assessment.
        </p>}
        <p className="mt-2 mb-0 text-caption font-mono text-accent">{passageLocation(packet)}</p>
        <p className="mt-2 mb-0 text-body">{packet.observation}</p>
        <RecordedPassage packet={packet} />
        <p className="mt-2 mb-0 text-caption font-mono text-text-muted">
          {packet.matched_context_review ? 'Selected review sentence' : packet.inspection_scope?.basis === 'scored_context' ? 'Navigation sentence' : 'Extracted text characters'} [{packet.focus.start}, {packet.focus.end}) · Unicode code points
        </p>
        {includesQuotedText && <p className="mt-2 mb-0 text-small text-text-muted">{roles.has('body')
          ? 'Highlighted text includes message text and an email quote or signature.'
          : 'The highlight covers an email quote or signature. Do not attribute its wording to the email sender.'}</p>}
        {pages.map((page) => <PagePlates key={page} set={set} packet={packet} page={page} token={token} mediaReady={mediaReady} />)}
        <div className="mt-5 space-y-3">
          <h4 className="m-0 text-small font-medium">What each check actually read</h4>
          {directions.includes(true) && directions.includes(false) && <p className="m-0 text-small text-text-muted">
            Disagreement is retained. One response was AI-like and another did not flag its input. Authorship remains undetermined.
          </p>}
          {packet.supporting_windows.map((window, index) => <ScoredContext key={`${window.signal_ref.id}:${index}`} window={window} index={index} />)}
        </div>
        <DocumentContextComparison comparison={packet.context_comparison} focus={packet.focus} />
        <details className="mt-4 text-small text-text-muted"><summary className="cursor-pointer">Source anchors</summary>
          <ul className="pl-4 list-disc">{packet.anchors.map((anchor, index) => {
            const original = set.coverage?.original_text_locations?.packets[packet.id]?.find((row) => row.anchor_index === index)
            return <li key={index}>
              {packageAnchorLabel(anchor) ?? (anchor.page != null ? `Page ${anchor.page}` : anchor.paragraph_index != null
                ? `${original?.coordinate === 'markdown_paragraph' ? 'Decoded Markdown paragraph' : original?.coordinate === 'rtf_paragraph' ? 'Text block' : 'Paragraph'} ${anchor.paragraph_index + 1}`
                : anchor.paragraph != null ? `Paragraph ${anchor.paragraph}` : 'Recorded source passage')}
              {anchor.kind === 'docx' && anchor.body_path && ` · document path ${anchor.body_path.join(' / ')}`}
              {anchor.mime_part != null && ` · email part ${anchor.mime_part}`}{anchor.quote_role && ` · ${anchor.quote_role}`}
              {anchor.extraction === 'ocr' && ' · text recognized from the page image'}
              {anchor.selected_start != null && anchor.selected_end != null && ` · Extracted characters [${anchor.selected_start}, ${anchor.selected_end})`}
              {original && <p className="mt-1 mb-2">{originalLocationLabels[original.coordinate]}{' '}
                <span className="font-mono">{original.ranges.map(([start, end]) => `[${start}, ${end})`).join(', ')}</span></p>}
            </li>
          })}</ul>
        </details>
        <p className="mt-4 mb-0 text-small">{packet.interpretation}</p>
        <EvidenceLimits lines={packet.limitations.filter((line) => line !== 'Each check uses its own recorded operating threshold. Threshold direction is not a validated probability or proof of authorship.')} />
        <EvidenceReferences packet={packet} />
      </article>
    })}</div>
  </div>
}
