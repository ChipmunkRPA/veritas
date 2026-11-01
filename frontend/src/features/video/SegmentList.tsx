import { Link } from 'react-router-dom'
import type { Finding } from '../../lib/api/queries'
import { cx } from '../../lib/cx'
import { anonymizeFindingText } from '../analysis/checkNames'
import type { VideoSegment } from './model'

/** `mm:ss.s` for list rows (UX §5.10 "00:12.4–00:15.9"); hours are added when needed. */
export function shortTime(s: number): string {
  const tenths = Math.max(0, Math.round(s * 10))
  const h = Math.floor(tenths / 36000)
  const m = Math.floor(tenths / 600) % 60
  const sec = (tenths % 600) / 10
  const body = `${String(m).padStart(2, '0')}:${sec.toFixed(1).padStart(4, '0')}`
  return h ? `${h}:${body}` : body
}

/** Findings that cite the segment's exhibit or scope a time span overlapping it on the same lane. */
export function findingsFor(segment: VideoSegment, findings: readonly Finding[]): Finding[] {
  return findings.filter((f) => {
    if ((f.evidence_ids ?? []).includes(segment.exhibitId)) return true
    const scope = f.scope as { start_s?: number; end_s?: number; lane?: string | null } | null | undefined
    return scope?.start_s != null && scope.end_s != null && scope.start_s <= segment.end && segment.start <= scope.end_s && (scope.lane ?? segment.lane) === segment.lane
  })
}

/** Ranked flagged segments (ARCHITECTURE §7.5 SegmentList): highest lane probability first, then by start. Selecting one seeks the
 *  player to the segment's first frame and zooms the timeline to it. */
export default function SegmentList({ analysisId, segments, selected, findings, onSelect, limit, emptyText, inspectionOnly = false }: {
  analysisId: string
  segments: readonly VideoSegment[]
  selected: string | null
  findings: readonly Finding[]
  onSelect: (segment: VideoSegment) => void
  limit?: number
  emptyText?: string
  inspectionOnly?: boolean
}) {
  const ranked = [...segments].sort((a, b) => b.peak - a.peak || a.start - b.start).slice(0, limit ?? segments.length)
  // Segments come from detector lanes (face tracks, lip-sync, audio, synthesis). The sentence names no other lane: a result of a
  // fusion model has no Estimate lane (N-17).
  if (!segments.length) return <p className="m-0 text-small text-text-muted">{emptyText ?? (inspectionOnly ? 'No detector segments were recorded. Values are available for inspection only.' : 'No detector lane reached the segment threshold (0.60, held above 0.45 for at least 0.5 s).')}</p>
  return (
    <ol className="m-0 p-0 list-none" data-testid="segment-list">
      {ranked.map((s) => {
        const cited = findingsFor(s, findings)
        return (
          <li key={s.key} className="flex items-center gap-2 border-b border-border last:border-b-0">
            <button type="button" data-testid="segment-row" data-segment={s.letter} aria-current={selected === s.key || undefined} onClick={() => onSelect(s)}
              className={cx('flex-1 min-w-0 grid grid-cols-[20px_minmax(0,1fr)_auto] gap-x-2 py-1.5 text-left text-small transition-colors duration-fast', selected === s.key ? 'text-text' : 'text-text-muted hover:text-text')}>
              <span className={cx('row-span-2 font-mono text-mono', selected === s.key ? 'text-selection' : 'text-text')}>{s.letter}</span>
              <span className="font-mono text-mono num">{shortTime(s.start)}–{shortTime(s.end)}</span>
              <span className="font-mono text-mono num text-text">{inspectionOnly ? <span className="sr-only">For inspection only</span> : s.peak.toFixed(2)}</span>
              <span className="col-span-2 truncate">{[s.laneName.toLowerCase(), s.trackName].filter(Boolean).join(' · ')}</span>
            </button>
            {cited.map((f) => <Link key={f.id} to={`/analyses/${analysisId}/findings`} className="font-mono text-micro text-accent underline underline-offset-[3px]" title={anonymizeFindingText(f, f.title)}>{f.id}</Link>)}
          </li>
        )
      })}
    </ol>
  )
}
