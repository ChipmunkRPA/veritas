import { Link, useNavigate } from 'react-router-dom'
import type { AnalysisDetail, Finding } from '../../lib/api/queries'
import { hasApiKey } from '../../lib/api'
import { nearestFrame, timecode } from '../../lib/pts'
import { SkeletonRows, ErrorState } from '../../components/ui'
import SegmentList from './SegmentList'
import TimelineLanes from './TimelineLanes'
import { uncountedNote } from './model'
import { stillUrl, useVideoEvidence } from './VideoWorkspace'

const TOP_SEGMENTS = 3

/** Exhibit preview of a video analysis on the result page (UX §5.5, §5.10): the analysed frame at the strongest segment, the lane stack
 *  and the top segments; every control opens the video workspace at that time or segment. */
export default function VideoExhibitPreview({ analysis, findings, token }: { analysis: AnalysisDetail; findings: readonly Finding[]; token: string }) {
  const nav = useNavigate()
  const { index, frames, model, error, retry } = useVideoEvidence(analysis, true)
  const base = `/analyses/${analysis.id}/evidence`
  const heading = <h2 id="exhibit-heading" className="m-0 eyebrow">Exhibit</h2>
  if (error) return <section id="section-exhibit" className="scroll-mt-14" aria-labelledby="exhibit-heading">{heading}<ErrorState className="mt-4" onRetry={retry}>{error.message}</ErrorState></section>
  if (!model || !index || !frames) return <section id="section-exhibit" className="scroll-mt-14" aria-labelledby="exhibit-heading">{heading}<SkeletonRows className="mt-4" /></section>

  const top = [...model.segments].sort((a, b) => b.peak - a.peak || a.start - b.start)[0] ?? null
  const k = nearestFrame(model.stills.map((s) => s.frame), top?.startFrame ?? 0)
  const still = model.stills.find((s) => s.frame === k) ?? null
  const tokenReady = !hasApiKey || token !== ''

  return (
    <section id="section-exhibit" data-testid="video-preview" aria-labelledby="exhibit-heading" className="scroll-mt-14">
      {heading}
      <div className="stage mt-3 rounded-md bg-stage text-stage-text border border-stage-border p-4 grid gap-4 min-[960px]:grid-cols-[minmax(0,1fr)_300px]">
        <figure className="m-0 min-w-0">
          <Link to={`${base}?t=${index.times[still?.frame ?? 0] ?? 0}${top ? `&seg=${top.letter}` : ''}`} className="block">
            <span className="block h-[300px] rounded-sm overflow-hidden bg-stage-2">
              {still && tokenReady && <img src={stillUrl(analysis.id, token)(still)} alt={`Analysed frame ${still.frame}`} className="w-full h-full object-contain" />}
            </span>
          </Link>
          <figcaption className="mt-2 font-mono text-micro text-stage-muted num">
            {still ? `Analysed frame ${still.frame} · ${timecode(index.times[still.frame] ?? 0)}${top ? ` · nearest to segment ${top.letter}` : ''}` : 'No analysed frame'} · {frames.width} × {frames.height} · {index.count} frames
          </figcaption>
        </figure>
        <div className="min-w-0">
          <h3 className="m-0 eyebrow text-stage-muted">{model.inspectionReason ? 'Detector segments' : 'Strongest segments'}</h3>
          <div className="mt-2"><SegmentList inspectionOnly={!!model.inspectionReason} analysisId={analysis.id} segments={model.segments} selected={null} findings={findings} limit={TOP_SEGMENTS} onSelect={(s) => nav(`${base}?seg=${s.letter}&t=${s.start}`)} /></div>
          {uncountedNote(model.uncounted, model.lanes) && <p data-testid="segments-uncounted" className="mt-2 mb-0 text-small text-stage-muted">{uncountedNote(model.uncounted, model.lanes)}</p>}
        </div>
      </div>
      <TimelineLanes className="mt-6" compact model={model} time={null} selected={null} onSeek={(t) => nav(`${base}?t=${Math.round(t * 1000) / 1000}`)} />
      <div className="mt-2 flex items-baseline justify-between gap-x-6 gap-y-1 flex-wrap">
        <p className="m-0 text-caption text-text-muted">Exhibit — Source video: the analysed frame nearest {model.inspectionReason ? 'a detector segment' : 'the strongest segment'}, the lane stack and the {model.inspectionReason ? 'detector' : 'flagged'} segments.</p>
        <Link to={base} data-testid="open-video-workspace" className="text-small font-medium text-accent underline underline-offset-[3px] decoration-1 hover:text-accent-hover hover:decoration-2">Open video workspace ▸</Link>
      </div>
    </section>
  )
}
