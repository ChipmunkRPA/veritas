import { Fragment, useState } from 'react'
import { Button } from '../../components/ui'
import type { DocumentEvidencePacket, DocumentEvidenceWindow } from './multimediaPackets'

export function windowSupportsInspection(window: DocumentEvidenceWindow): boolean {
  if (window.direction) return window.direction === 'supports_ai'
  const boundary = window.threshold ?? 0.5
  const lowerSupports = window.ai_when === 'lt' || (!window.ai_when && window.score_kind === 'perplexity_ratio')
  return lowerSupports ? window.score < boundary : window.score >= boundary
}

export function passageLocation(packet: DocumentEvidencePacket): string {
  const locations = packet.anchors.map((anchor) => {
    const place = anchor.page != null ? `Page ${anchor.page}`
      : anchor.paragraph_index != null ? `Paragraph ${anchor.paragraph_index + 1}`
        : anchor.paragraph != null ? `Paragraph ${anchor.paragraph}` : null
    const email = anchor.mime_part ? `Email part ${anchor.mime_part}` : null
    return [place, email, anchor.quote_role].filter(Boolean).join(' · ')
  }).filter(Boolean)
  const ranges = (packet.inspection_scope?.spans ?? [packet.focus]).map(({ start, end }) => `${start}–${end}`).join(', ')
  return [...new Set(locations)].join('; ') || `Characters ${ranges}`
}

/** Match code-point coordinates and source text, never a guessed substring occurrence. */
export function sourceHighlight(packet: Pick<DocumentEvidencePacket, 'focus' | 'context' | 'inspection_scope'>) {
  const { focus, context } = packet
  if (![focus.start, focus.end, context.start, context.end].every(Number.isSafeInteger)
      || context.start < 0 || focus.start < context.start || focus.end <= focus.start || focus.end > context.end) return null
  const characters = Array.from(context.text)
  const start = focus.start - context.start, end = focus.end - context.start
  if (characters.length !== context.end - context.start || Array.from(focus.text).length !== focus.end - focus.start
      || characters.slice(start, end).join('') !== focus.text) return null
  const ranges = (packet.inspection_scope?.spans ?? [focus]).map((span) => ({
    start: span.start - context.start, end: span.end - context.start,
  }))
  if (!ranges.length || ranges.some((range, i) =>
    !Number.isSafeInteger(range.start) || !Number.isSafeInteger(range.end)
    || range.start < (ranges[i - 1]?.end ?? 0) || range.end <= range.start || range.end > characters.length)) return null
  return { characters, start: ranges[0].start, end: ranges[ranges.length - 1].end, ranges }
}

export function RecordedPassage({ packet }: { packet: DocumentEvidencePacket }) {
  const [expanded, setExpanded] = useState(false)
  const match = sourceHighlight(packet)
  if (!match) return <div className="mt-4">
    <p className="mb-2 text-small font-medium">Exact extracted passage</p>
    <blockquote className="m-0 border-l-2 border-accent pl-4 py-2 text-body whitespace-pre-wrap break-words">{packet.focus.text}</blockquote>
    <p className="mt-2 mb-0 text-caption text-text-muted">The saved passage does not match its context coordinates. It is shown separately; no highlight position is inferred.</p>
    <details className="mt-3 text-small"><summary className="cursor-pointer text-text-muted">Recorded source context</summary>
      <p className="whitespace-pre-wrap break-words">{packet.context.text}</p>
    </details>
  </div>
  const first = expanded ? 0 : Math.max(0, match.start - 180)
  const last = expanded ? match.characters.length : Math.min(match.characters.length, match.end + 180)
  const shortened = first > 0 || last < match.characters.length
  const basis = packet.inspection_scope?.basis
  const label = basis === 'ratio_reduction' ? 'Measured writing-pattern range'
    : basis === 'scored_context' ? 'Scored text in its source context' : 'Selected passage in its source context'
  return <div className="mt-4" data-testid={`source-passage-${packet.id}`}>
    <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
      <p className="m-0 text-small font-medium">{label}</p>
      {(shortened || expanded) && <Button variant="text" onClick={() => setExpanded((value) => !value)} aria-expanded={expanded}>
        {expanded ? 'Show nearby context' : 'Show full saved context'}
      </Button>}
    </div>
    <blockquote className="m-0 rounded-md border border-border bg-surface-2 p-4 text-body leading-relaxed whitespace-pre-wrap break-words">
      {first > 0 && <span aria-label="Earlier saved context not displayed">… </span>}
      {match.ranges.map((range, i) => <Fragment key={range.start}>
        {match.characters.slice(match.ranges[i - 1]?.end ?? first, range.start).join('')}
        <mark className={basis === 'scored_context' ? 'bg-surface text-text border-b-2 border-accent rounded-xs' : 'bg-hl-ai text-text border-b-2 border-hl-ai-rule rounded-xs'}
          data-testid={`passage-highlight-${packet.id}`}>{match.characters.slice(range.start, range.end).join('')}</mark>
      </Fragment>)}
      {match.characters.slice(match.end, last).join('')}
      {last < match.characters.length && <span aria-label="Later saved context not displayed"> …</span>}
    </blockquote>
    <p className="mt-2 mb-0 text-caption text-text-muted">{basis === 'scored_context'
      ? 'The highlight covers the measured context. The navigation sentence gives you a place to start reading. The response does not identify generated words.'
      : basis === 'ratio_reduction'
        ? 'The highlight covers the tokens used for the writing-pattern ratio. Surrounding text can influence that response.'
        : 'The highlight locates the selected passage. The response used a larger context and does not identify individual AI-written words.'}
      {' '}Unhighlighted text is not cleared.</p>
  </div>
}

export default function DocumentInspectionMap({ packets, selectedId, onSelect }: {
  packets: DocumentEvidencePacket[]; selectedId: string | null; onSelect: (id: string) => void
}) {
  if (!packets.length) return null
  return <section aria-label="Passages to inspect" className="mt-4 rounded-md border border-border bg-surface-2 p-4">
    <h3 className="m-0 text-body font-medium">Passages to inspect <span className="font-mono text-text-muted">· {packets.length}</span></h3>
    <p className="mt-2 mb-0 text-small text-text-muted">Select a review location to see the text the check measured, its source and the recorded responses.</p>
    <ol className="m-0 mt-4 p-0 list-none grid gap-2 min-[960px]:grid-cols-2">
      {packets.map((packet) => {
        const supporting = packet.supporting_windows.filter(windowSupportsInspection).length
        const opposing = packet.supporting_windows.length - supporting
        const basis = packet.inspection_scope?.basis
        const reading = supporting && opposing ? 'Responses disagree' : supporting
          ? basis === 'scored_context' ? 'Context response' : basis === 'ratio_reduction' ? 'Measured ratio' : 'Inspection flag'
          : 'Recorded comparison'
        const match = basis === 'scored_context' ? sourceHighlight(packet) : null
        const characters = match ? match.characters.slice(match.start, match.end) : Array.from(packet.focus.text)
        const snippet = characters.slice(0, 160).join('') + (characters.length > 160 ? '…' : '')
        return <li key={packet.id}>
          <button type="button" aria-label={`Inspect passage ${packet.id}: ${passageLocation(packet)}`} aria-pressed={selectedId === packet.id}
            className={`w-full h-full rounded-md border p-3 text-left transition-colors focus-visible:outline-2 focus-visible:outline-accent ${selectedId === packet.id ? 'border-accent bg-surface' : 'border-border hover:border-accent'}`}
            onClick={() => onSelect(packet.id)}>
            <span className="flex flex-wrap justify-between gap-2 text-caption"><span className="font-mono text-accent">{packet.id} · {passageLocation(packet)}</span><span className="text-text-muted">{reading}</span></span>
            <span className="block mt-2 text-small whitespace-pre-wrap break-words">“{snippet}”</span>
            <span className="block mt-2 text-caption text-text-muted">{selectedId === packet.id ? 'Selected · ' : ''}View source and reasons →</span>
          </button>
        </li>
      })}
    </ol>
  </section>
}
