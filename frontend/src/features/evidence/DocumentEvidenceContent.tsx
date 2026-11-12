import { useEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { Button } from '../../components/ui'
import { EvidenceLimits, EvidencePicture, EvidenceReferences, evidenceCard, evidenceGrid } from './EvidenceMedia'
import DocumentInspectionMap, { passageLocation, RecordedPassage, windowSupportsInspection } from './DocumentInspectionMap'
import DocumentContextComparison from './DocumentContextComparison'
import DocumentPassageCoverage from './DocumentPassageCoverage'
import DocumentRepeatedPassages from './DocumentRepeatedPassages'
import { type DocumentEvidencePacket, type DocumentEvidenceSet, type DocumentEvidenceWindow } from './multimediaPackets'

function ScoredContext({ window, index }: { window: DocumentEvidenceWindow; index: number }) {
  const ratio = window.score_kind === 'perplexity_ratio'
  const threshold = window.threshold ?? 0.5
  const supports = windowSupportsInspection(window)
  const matched = typeof window.matched_context_selected === 'boolean'
  const eligible = typeof window.score_exact_hex === 'string'
  return <div className="rounded-md border border-border p-4">
    <p className="m-0 text-small"><span className="font-medium">{window.signal_ref.display_id ?? `Check ${index + 1}`}</span>
      {' '}· <span className="font-mono">{window.score.toFixed(3)}</span> {ratio ? 'raw writing-pattern ratio' : 'uncalibrated response'}
      {' '}· {matched ? (supports ? 'AI-like context selected for review' : eligible ? 'Context did not select this sentence' : 'Outside short-passage review coverage')
        : supports ? 'AI-like response for this input' : 'Opposing response for this input'}</p>
    <p className="mt-2 mb-0 text-caption text-text-muted">{matched ? (eligible
      ? 'The check read three sentences together and uses the middle sentence as a review location. The response is an uncalibrated writing-pattern measurement.'
      : 'This response was recorded for the original writing check. Short-passage review covers only eligible middle sentences.')
      : ratio ? `Ratios below ${threshold} support inspection; this is not a percentage or an authorship probability.`
      : `Responses at or above ${threshold} support inspection; the response is not a calibrated authorship probability.`}
      {' '}Displayed values are rounded; the direction uses the recorded full-precision response.
      {!supports && !matched && ' Disagreement is retained.'}</p>
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
  const usesOcr = ocrPages.some((page) => page.status === 'extracted') || set.packets.some((packet) => packet.anchors.some((anchor) => anchor.extraction === 'ocr'))
  return <div data-testid="document-evidence-content">
    <DocumentRepeatedPassages result={set.repeated_passages} />
    <DocumentInspectionMap packets={set.packets} selectedId={selectedId} onSelect={selectPassage} />
    <DocumentPassageCoverage coverage={set.coverage?.local_passage_coverage} />
    {writingAgreement && <section className="mt-4 rounded-md border border-border p-4 text-small text-text-muted">
      <h3 className="m-0 text-body font-medium text-text">Second writing check</h3>
      <p className="mt-2 mb-0">The second check revisits the exact text behind up to six AI-like responses. This evidence carries zero weight in the overall assessment.</p>
      {writingAgreement.ranges.length === 0 ? <p className="mt-2 mb-0">The first check supplied no AI-like contexts for this review.</p>
        : <ul className="mb-0 pl-4 list-disc">{writingAgreement.ranges.map((row) => {
          const packet = set.packets.find((item) => item.focus.start === row.start && item.focus.end === row.end)
          const range = `Passage [${row.start}, ${row.end})`
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
          {packet.supporting_windows.map((window, index) => <ScoredContext key={`${window.signal_ref.id}:${index}`} window={window} index={index} />)}
        </div>
        <DocumentContextComparison comparison={packet.context_comparison} focus={packet.focus} />
        <details className="mt-4 text-small text-text-muted"><summary className="cursor-pointer">Source anchors</summary>
          <ul className="pl-4 list-disc">{packet.anchors.map((anchor, index) => <li key={index}>
            {anchor.page != null ? `Page ${anchor.page}` : anchor.paragraph_index != null ? `Paragraph ${anchor.paragraph_index + 1}`
              : anchor.paragraph != null ? `Paragraph ${anchor.paragraph}` : 'Recorded source passage'}
            {anchor.kind === 'docx' && anchor.body_path && ` · document path ${anchor.body_path.join(' / ')}`}
            {anchor.mime_part != null && ` · email part ${anchor.mime_part}`}{anchor.quote_role && ` · ${anchor.quote_role}`}
            {anchor.extraction === 'ocr' && ' · text recognized from the page image'}
            {anchor.selected_start != null && anchor.selected_end != null && ` · characters [${anchor.selected_start}, ${anchor.selected_end})`}
          </li>)}</ul>
        </details>
        <p className="mt-4 mb-0 text-small">{packet.interpretation}</p><EvidenceLimits lines={packet.limitations} /><EvidenceReferences packet={packet} />
      </article>
    })}</div>
  </div>
}
