import { useEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { Button } from '../../components/ui'
import { EvidenceLimits, EvidencePicture, EvidenceReferences, evidenceCard, evidenceGrid } from './EvidenceMedia'
import DocumentInspectionMap, { passageLocation, RecordedPassage, windowInspectionDirection } from './DocumentInspectionMap'
import DocumentContextComparison from './DocumentContextComparison'
import DocumentPassageCoverage from './DocumentPassageCoverage'
import DocumentRepeatedPassages from './DocumentRepeatedPassages'
import DocumentUnselectedResponses from './DocumentUnselectedResponses'
import { type DocumentEvidencePacket, type DocumentEvidenceSet, type DocumentEvidenceWindow, type DocumentOriginalTextLocation, type DocumentPassagePosition, type EmailBodyAlternatives, type EmailClassifierInputScope } from './multimediaPackets'

const originalLocationLabels: Record<DocumentOriginalTextLocation['coordinate'], string> = {
  decoded_text: 'Original decoded text characters',
  word_paragraph: 'Original Word paragraph characters',
  decoded_email: 'Decoded email part characters',
  decoded_email_html: 'Decoded email HTML source characters',
}

const emailInputRoles = [
  ['body', 'Message only'], ['quoted', 'Quote only'], ['signature', 'Signature only'],
  ['mixed', 'Mixed roles'], ['unmapped', 'Role unavailable'],
] as const

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
  if (review.schema === 'document-passage-position-review/2') {
    const selected = review.selected == null ? null : review.contexts[review.selected]
    const labels = ['Previous sentence', 'Selected sentence', 'Following sentence']
    return <section id="document-passage-position" aria-labelledby="document-passage-position-title" className="mt-4 rounded-md border border-border p-4">
      <h3 id="document-passage-position-title" className="m-0 text-body font-medium">Longer-passage writing check</h3>
      <p className="mt-2 mb-0 text-small">{review.contexts.length} eligible sentence centers across {review.paragraphs} paragraphs and {review.sentence_count} sentences.</p>
      <p className="mt-2 mb-0 text-small">{review.status === 'flagged'
        ? 'The selected sentence showed a stronger AI-like writing response than its immediate neighbors. Review the wording in context.'
        : review.status === 'unassessed' ? 'Complete responses were unavailable for the eligible centers.'
        : 'No sentence was selected among the eligible centers. Authorship remains undetermined.'}</p>
      {selected && <>
        <blockquote className="my-3 border-l-2 border-border pl-3 break-words text-small">{review.focus_text}</blockquote>
        <ul className="mt-2 mb-0 list-none p-0 text-small">{selected.sentences.map((row, i) =>
          <li key={row.source_start}>{labels[i]} <span className="font-mono">[{row.source_start}, {row.source_end})</span></li>)}</ul>
      </>}
      <p className="mt-2 mb-0 text-caption text-text-muted">Positions count Unicode characters in the original text. This optional check carries zero weight in the overall assessment.</p>
    </section>
  }
  const labels = ['Opening sentence', 'Middle sentence', 'Closing sentence']
  return <section id="document-passage-position" aria-labelledby="document-passage-position-title" className="mt-4 rounded-md border border-border p-4">
    <h3 id="document-passage-position-title" className="m-0 text-body font-medium">Three-sentence writing check</h3>
    <p className="mt-2 mb-0 text-small">{review.status === 'flagged'
      ? 'The middle sentence showed a stronger AI-like writing response than both neighbors. This check marked that sentence for inspection.'
      : review.status === 'unassessed' ? 'The input met the scope for this check. Complete responses were unavailable.'
      : 'The middle sentence did not reach the review level for a stronger AI-like writing response than both neighbors. This result leaves authorship undetermined.'}</p>
    <blockquote className="my-3 border-l-2 border-border pl-3 break-words text-small">{review.focus_text}</blockquote>
    <ul className="mt-2 mb-0 list-none p-0 text-small">{review.sentences.map((row, i) =>
      <li key={row.source_start}>{labels[i]} <span className="font-mono">[{row.source_start}, {row.source_end})</span></li>)}</ul>
    <p className="mt-2 mb-0 text-caption text-text-muted">Positions count Unicode characters in the original text. Each response uses its sentence and immediate neighbors. This check carries zero weight in the overall assessment.</p>
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
        return <div key={row.start} className="mb-2 last:mb-0">
          <blockquote className="m-0 break-words text-small">{chars.slice(0, start).join('')}<mark>{chars.slice(start, end).join('')}</mark>{chars.slice(end).join('')}</blockquote>
          <p className="mt-1 mb-0 font-mono text-caption text-text-muted">{location && `${location} · `}Statement [{row.start}, {row.end}) · Value [{row.quantity_start}, {row.quantity_end})</p>
        </div>
      })}
    </div>)}
    {review.different_claims > review.groups.length && <p className="mt-2 mb-0 text-caption text-text-muted">
      {review.different_claims - review.groups.length} additional statements exceed the display limit.</p>}
    <p className="mt-2 mb-0 text-caption text-text-muted">{review.limitation} Positions use extracted Unicode characters.</p>
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
  const ocrPages = set.coverage?.ocr?.pages ?? []
  const matched = set.coverage?.matched_context_review
  const writingAgreement = set.coverage?.writing_second_check
  const tracked = set.coverage?.word_tracked_changes
  const selectablePages = [...new Set(writingAgreement?.pdf_spans?.map((span) => span.page) ?? [])]
  const usesOcr = ocrPages.some((page) => page.status === 'extracted') || set.packets.some((packet) => packet.anchors.some((anchor) => anchor.extraction === 'ocr'))
  return <div data-testid="document-evidence-content">
    <EmailBodyComparison review={set.coverage?.email_body_alternatives} />
    {set.coverage?.email_classifier_input_scope && <EmailInputScope scope={set.coverage.email_classifier_input_scope} />}
    {tracked && <section aria-label="Word tracked text changes" className="mt-4 rounded-md border border-border p-4">
      <h3 className="m-0 text-body font-medium">Word tracked text changes</h3>
      <p className="mt-2 mb-0 text-small">Checked {tracked.paragraphs} body paragraphs. Found {tracked.inserted} insertions and {tracked.deleted} deletions.</p>
      {!!tracked.unsupported_marks && <p className="mt-2 mb-0 text-small">{tracked.unsupported_marks} additional revision marks were outside scope.</p>}
      <p className="mt-2 mb-0 text-small text-text-muted">{tracked.scope} {tracked.limitation}</p>
      {!!tracked.rows.length && <ol className="mt-3 mb-0 list-none space-y-3 p-0">
        {tracked.rows.map((row) => <li key={`${row.body_index}:${row.inline_index}`} className="rounded-md border border-border p-3">
          <p className="m-0 text-small font-medium">{row.kind === 'inserted' ? 'Inserted' : 'Deleted'} · Body paragraph {row.paragraph}</p>
          <blockquote className="mt-2 mb-0 whitespace-pre-wrap break-words text-small">{row.text}
            {row.characters > Array.from(row.text).length && `… (${row.characters} source characters)`}</blockquote>
          <p className="mt-2 mb-0 text-caption text-text-muted">{tracked.part} · XML body index {row.body_index}, paragraph child index {row.inline_index} (zero-based).</p>
        </li>)}
      </ol>}
      {!!tracked.omitted_excerpts && <p className="mt-2 mb-0 text-caption text-text-muted">{tracked.omitted_excerpts} further excerpts exceed the display limit.</p>}
    </section>}
    <DocumentRepeatedPassages result={set.repeated_passages} />
    {set.percentage_consistency && <PercentageConsistency review={set.percentage_consistency} />}
    <DocumentInspectionMap packets={set.packets} selectedId={selectedId} onSelect={selectPassage} />
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
      <p className="m-0">{set.coverage?.scored_windows != null ? `${set.coverage.scored_windows} scored context window(s) were recorded.` : 'Exact scored-context coverage is unavailable.'}
        {' '}The marked ranges show the text behind the recorded measurements. The responses do not determine the origin of individual words.</p>
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
              {anchor.page != null ? `Page ${anchor.page}` : anchor.paragraph_index != null ? `Paragraph ${anchor.paragraph_index + 1}`
                : anchor.paragraph != null ? `Paragraph ${anchor.paragraph}` : 'Recorded source passage'}
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
