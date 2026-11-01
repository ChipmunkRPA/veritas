import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { PenLine } from 'lucide-react'
import { api, hasApiKey } from '../../lib/api'
import { apiClient, unwrap } from '../../lib/api/client'
import { qk, useAnalysis, useAnnotations, useExhibitPayloads, useExhibits, useFindings, useFrames, useSignals, type AnalysisDetail, type Annotation, type Finding, type FrameTable } from '../../lib/api/queries'
import { buildPtsIndex, frameAt, timecode, type PtsIndex } from '../../lib/pts'
import { cx } from '../../lib/cx'
import { ErrorState, Skeleton } from '../../components/ui'
import MediaAccessNotice from '../../components/MediaAccessNotice'
import FramePlayer, { useFramePlayer } from './FramePlayer'
import IntervalDialog, { type IntervalDraft } from './IntervalDialog'
import SegmentList, { findingsFor, shortTime } from './SegmentList'
import SyncOffsetPlot from './SyncOffsetPlot'
import TimelineLanes from './TimelineLanes'
import TrackGallery from './TrackGallery'
import { buildVideoModel, uncountedNote, type StillItem, type VideoModel, type VideoSegment } from './model'
import { useMultimediaEvidence } from '../evidence/multimediaPackets'

const NO_ANNOTATIONS: readonly Annotation[] = []
const KEY_FRAMES_PER_SEGMENT = 2
const NOTE_DEFAULT_S = 0.5

export const stillUrl = (analysisId: string, token: string) => (s: StillItem): string =>
  `/api/v2/analyses/${analysisId}/stills/${s.ms}.jpg${token ? `?token=${encodeURIComponent(token)}` : ''}`

/** Frame table, lane payloads and annotations of a done video analysis, and the workspace model built from them. */
export function useVideoEvidence(analysis: AnalysisDetail | undefined, enabled: boolean) {
  const id = analysis?.id ?? ''
  const exhibits = useExhibits(id, enabled)
  const frames = useFrames(id, enabled)
  const annotations = useAnnotations(id, enabled)
  const ids = useMemo(() => (exhibits.data ?? []).filter((x) => x.kind === 'series' || x.kind === 'segments' || x.kind === 'face_track').map((x) => x.id), [exhibits.data])
  const payloads = useExhibitPayloads(id, ids, enabled && exhibits.isSuccess)
  const payloadKey = payloads.data.map((p) => p.id).join(',')
  const index = useMemo(() => (frames.data ? buildPtsIndex(frames.data) : null), [frames.data])
  const c2pa = analysis?.assessments?.find((x) => x.hypothesis === 'H3_provenance_integrity')?.c2pa_state ?? null
  const annotationData = annotations.data ?? NO_ANNOTATIONS
  // N-19k (Q2-12): the signals say which checks the estimate does not count; their lanes are drawn for inspection
  const signals = useSignals(id, enabled)
  const model = useMemo(
    () => (index && frames.data && exhibits.data && !payloads.isPending && !signals.isPending
      ? buildVideoModel({ index, frames: frames.data, exhibits: exhibits.data, payloads: payloads.data, annotations: annotationData, c2paState: c2pa, decision: analysis?.decision, signals: signals.data ?? [] })
      : null),
    // payloads.data is a fresh array on every render; payloadKey identifies it
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [index, frames.data, exhibits.data, payloadKey, payloads.isPending, annotationData, c2pa, analysis?.decision, signals.data, signals.isPending],
  )
  const error = exhibits.error ?? frames.error ?? payloads.error ?? null
  const retry = () => { void exhibits.refetch(); void frames.refetch(); payloads.refetch() }
  return { index, frames: frames.data ?? null, model, error, retry }
}

const tool = (on = false) => cx('h-8 px-2 inline-flex items-center gap-1.5 rounded-sm text-small transition-colors duration-fast disabled:opacity-40 disabled:cursor-not-allowed',
  on ? 'bg-surface-3 text-text' : 'text-text-muted enabled:hover:text-text enabled:hover:bg-surface-3')

/**
 * Video workspace (UX §5.10; ARCHITECTURE §7.2 `evidence`, §7.5): the frame player on the dark stage over the lane stack, with the
 * inspector (selected segment, ranked segments, face tracks, sync offset, key frames, interval notes).
 * - **URL state:** `t` (seconds, written when paused), `seg` (segment letter) and `track` (isolated track), so notes and reports can
 *   link to an exact view.
 * - **Keys:** Space play/pause; `,` / `.` one frame, with Shift ten; ← / → one second; `[` / `]` previous / next flagged segment;
 *   Home / End; Esc clears the selection.
 */
export default function VideoWorkspace() {
  const { id = '' } = useParams()
  const q = useAnalysis(id)
  const a = q.data
  const done = a?.status === 'done' && a.kind === 'video'
  const evidence = useVideoEvidence(a, done)
  const sourceEvidence = useMultimediaEvidence(id, 'video', done)
  const sourceClock = sourceEvidence.data?.schema === 'chipmunk.video-evidence/1' && sourceEvidence.data.status === 'ready'
    ? sourceEvidence.data.source.timeline_origin_s ?? null : null
  const findings = useFindings(id, done)
  const tok = useQuery({ queryKey: qk.downloadToken(id), queryFn: () => api.downloadToken(id), enabled: done, staleTime: 5 * 60_000 })
  const token = tok.data ?? ''
  if (q.isPending || (done && !evidence.error && !evidence.model)) {
    return <div className="p-12" aria-busy="true"><Skeleton w={320} h={20} /><div className="mt-4"><Skeleton w={640} h={360} /></div></div>
  }
  if (q.error || !a) {
    return <div className="p-12"><ErrorState onRetry={() => void q.refetch()}>{q.error?.message ?? 'Not found'} <span className="text-text-muted mx-1">·</span> <Link to="/analyses" className="text-accent underline underline-offset-[3px]">Back to analyses</Link></ErrorState></div>
  }
  if (!done) {
    return (
      <div className="p-12 max-w-[640px]">
        <p className="m-0 text-small text-text">The analysis is {a.status}; the video workspace opens once it has finished.</p>
        <p className="mt-3 mb-0"><Link to={`/analyses/${id}`} className="text-small text-accent underline underline-offset-[3px]">‹ Back to the result</Link></p>
      </div>
    )
  }
  if (evidence.error || !evidence.model || !evidence.index || !evidence.frames) {
    return <div className="p-12"><ErrorState onRetry={evidence.retry}>{evidence.error?.message ?? 'The frame table could not be read.'}</ErrorState></div>
  }
  return <Workspace analysis={a} index={evidence.index} frames={evidence.frames} model={evidence.model} findings={findings.data ?? []} token={token}
    timelineOrigin={sourceClock}
    mediaFailed={!!tok.error} mediaPending={tok.isFetching} onMediaRetry={() => void tok.refetch()} />
}

function Workspace({ analysis, index, frames, model, findings, token, mediaFailed, mediaPending, onMediaRetry, timelineOrigin }: {
  analysis: AnalysisDetail
  index: PtsIndex
  frames: FrameTable
  model: VideoModel
  findings: readonly Finding[]
  token: string
  timelineOrigin: number | null
  mediaFailed: boolean
  mediaPending: boolean
  onMediaRetry: () => void
}) {
  const qc = useQueryClient()
  const [params, setParams] = useSearchParams()
  /** Patches the live URL: a delayed write (the `t` mirror) must not restore parameters another action changed meanwhile, which a
   *  functional `setSearchParams` from an older render would do. */
  const update = (patch: Record<string, string | null>) => {
    const next = new URLSearchParams(window.location.search)
    for (const [k, v] of Object.entries(patch)) {
      if (v == null) next.delete(k)
      else next.set(k, v)
    }
    setParams(next, { replace: true })
  }
  const selected = model.segments.find((s) => s.letter === params.get('seg')) ?? null
  const isolated = model.tracks.some((t) => t.trackId === params.get('track')) ? params.get('track') : null
  const initialT = params.get('t') != null ? Number(params.get('t')) : null
  const tokenReady = !hasApiKey || token !== ''
  const src = tokenReady ? api.sourceUrl(analysis.id, token) : null
  const player = useFramePlayer(index, initialT != null && Number.isFinite(initialT) ? frameAt(index, initialT) : selected?.startFrame ?? null, src, timelineOrigin)
  const time = index.times[player.frame] ?? 0
  const still = useMemo(() => tokenReady ? stillUrl(analysis.id, token) : () => null, [analysis.id, token, tokenReady])

  useEffect(() => {
    if (player.playing) return
    const h = window.setTimeout(() => update({ t: String(Math.round(time * 1000) / 1000) }), 250)
    return () => window.clearTimeout(h)
    // update is stable in effect: it only writes the URL
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [time, player.playing])

  const [focus, setFocus] = useState<{ key: string; start: number; end: number } | null>(null)
  const zoomCount = useRef(0)
  const root = useRef<HTMLDivElement>(null)
  const selectSegment = (s: VideoSegment | null, seek = true) => {
    update({ seg: s?.letter ?? null })
    if (s && seek) {
      player.seekFrame(s.startFrame)
      zoomCount.current += 1
      setFocus({ key: `${s.key}:${zoomCount.current}`, start: s.start, end: s.end })
    }
  }
  const byStart = [...model.segments].sort((x, y) => x.startFrame - y.startFrame || x.letter.localeCompare(y.letter))
  const prevFlagged = [...byStart].reverse().find((s) => s.startFrame < player.frame) ?? null
  const nextFlagged = byStart.find((s) => s.startFrame > player.frame) ?? null

  const [dialog, setDialog] = useState<{ start: number; end: number } | null>(null)
  const create = useMutation({
    mutationFn: (d: IntervalDraft) => unwrap(apiClient.POST('/api/v2/analyses/{aid}/annotations', {
      params: { path: { aid: analysis.id } },
      body: { target_type: 'interval', shape: 'interval', geometry: { start_pts: frames.pts[d.startFrame], end_pts: frames.pts[d.endFrame], time_base: frames.time_base }, body: d.body, visibility: d.visibility },
    })),
    onSuccess: () => {
      setDialog(null)
      void qc.invalidateQueries({ queryKey: qk.annotations(analysis.id) })
    },
  })
  const locked = analysis.locked_at != null
  const openNote = (start: number, end: number) => {
    if (locked) return
    create.reset()
    setDialog({ start: frameAt(index, start), end: Math.max(frameAt(index, start), frameAt(index, end)) })
  }

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const t = e.target as HTMLElement
    if (dialog || e.metaKey || e.ctrlKey || e.altKey || t.closest('input, textarea, select, [role="menu"], [role="slider"], dialog')) return
    if ((e.key === ' ' || e.key === 'Enter') && t.closest('button, a')) return
    switch (e.key) {
      case ' ': player.toggle(); break
      case ',': case '<': player.step(e.shiftKey || e.key === '<' ? -10 : -1); break
      case '.': case '>': player.step(e.shiftKey || e.key === '>' ? 10 : 1); break
      case 'ArrowLeft': player.seekFrame(frameAt(index, time - 1)); break
      case 'ArrowRight': player.seekFrame(frameAt(index, time + 1)); break
      case '[': if (prevFlagged) selectSegment(prevFlagged); break
      case ']': if (nextFlagged) selectSegment(nextFlagged); break
      case 'Home': player.seekFrame(0); break
      case 'End': player.seekFrame(index.count - 1); break
      case 'Escape': if (!selected) return; update({ seg: null }); break
      default: return
    }
    e.preventDefault()
  }

  const syncLanes = model.lanes.filter((l) => l.kind === 'lip_sync' && l.curve?.length && (!isolated || l.trackId === isolated))
  const overlapping = selected ? model.segments.filter((s) => s.key !== selected.key && s.start < selected.end && selected.start < s.end) : []
  const keyFrames: { still: StillItem; segment: VideoSegment | null }[] = useMemo(() => {
    const picked = model.segments.flatMap((s) => model.stills.filter((x) => x.frame >= s.startFrame && x.frame <= s.endFrame).slice(0, KEY_FRAMES_PER_SEGMENT).map((x) => ({ still: x, segment: s })))
    const unique = picked.filter((p, i) => picked.findIndex((q) => q.still.frame === p.still.frame) === i)
    return unique.length ? unique : model.stills.slice(0, 4).map((x) => ({ still: x, segment: null }))
  }, [model.segments, model.stills])

  return (
    <div ref={root} tabIndex={-1} data-testid="video-workspace" className="h-full min-h-0 flex flex-col bg-surface outline-none" onKeyDown={onKeyDown}
      onPointerDown={(e) => { if (!(e.target as HTMLElement).closest('button, a, input, textarea, select, [tabindex]:not([tabindex="-1"]), dialog')) root.current?.focus({ preventScroll: true }) }}>
      {/* N-19d (UX, small screens; design review P2): below 960 px the viewer says what it was designed for */}
      <p data-testid="viewer-small-screen" className="m-0 px-3 py-1.5 border-b border-border bg-surface-2 text-caption text-text-muted min-[960px]:hidden">The evidence viewer is designed for screens of 960 px and wider.</p>
      {/* N-19b (browser QA "Layout"): below 640 px the title has a row of its own under the back link and the tools, so that it is
          not cut to the room they leave (112 to 181 px at 375 px); it takes two lines at most, with the whole name as its hover text */}
      <header className="h-12 shrink-0 flex items-center gap-2 px-4 border-b border-border bg-surface min-w-0 max-[639px]:h-auto max-[639px]:flex-wrap max-[639px]:gap-y-0 max-[639px]:pb-1.5">
        <Link to={`/analyses/${analysis.id}`} className="shrink-0 text-small text-text-muted hover:text-text inline-flex items-center min-h-8">‹ Result</Link>
        <h1 data-testid="workspace-title" className="m-0 ml-2 min-w-0 truncate text-small font-normal text-text max-[639px]:order-last max-[639px]:basis-full max-[639px]:ml-0 max-[639px]:whitespace-normal max-[639px]:line-clamp-2 max-[639px]:[overflow-wrap:anywhere]" title={analysis.filename}>{analysis.filename}<span className="text-text-muted"> · Video workspace</span></h1>
        <div role="toolbar" aria-label="Workspace" className="ml-auto flex items-center gap-1 shrink-0">
          <span className="font-mono text-micro text-text-muted num max-[1199px]:hidden">{frames.width} × {frames.height} · {index.count} frames · {model.stills.length} analysed</span>
          <span aria-hidden className="mx-2 h-5 w-px bg-border max-[1199px]:hidden" />
          <button type="button" className={tool()} disabled={locked} title={locked ? 'Revision locked' : 'Add a note on the selected segment or the current second (or shift-drag a lane)'}
            onClick={() => (selected ? openNote(selected.start, selected.end) : openNote(time, Math.min(model.duration, time + NOTE_DEFAULT_S)))}>
            <PenLine size={14} strokeWidth={1.5} aria-hidden /><span>Add note</span>
          </button>
        </div>
      </header>
      <MediaAccessNotice ready={tokenReady} failed={mediaFailed} pending={mediaPending} onRetry={onMediaRetry} className="px-4" />

      <div className="flex-1 min-h-0 overflow-y-auto min-[960px]:overflow-hidden min-[960px]:grid min-[960px]:grid-cols-[minmax(0,1fr)_320px] min-[960px]:grid-rows-[minmax(0,1fr)]">
        <div className="min-w-0 p-4 flex flex-col gap-5 min-[960px]:overflow-y-auto">
          <FramePlayer player={player} index={index} frames={frames} model={model} src={src} stillUrl={still} isolatedTrack={isolated}
            onPrevFlagged={prevFlagged ? () => selectSegment(prevFlagged) : null} onNextFlagged={nextFlagged ? () => selectSegment(nextFlagged) : null} />
          <TimelineLanes model={model} time={time} selected={selected} focus={focus} isolatedTrack={isolated}
            onSeek={(t) => player.seekFrame(frameAt(index, t))} onSelectSegment={(s) => selectSegment(s, false)} onInterval={locked ? undefined : openNote} />
        </div>

        <aside aria-label="Inspector" className="min-w-0 border-t min-[960px]:border-t-0 min-[960px]:border-l border-border p-4 flex flex-col gap-7 min-[960px]:overflow-y-auto">
          <section data-testid="segment-inspector" aria-labelledby="segment-heading">
            <h2 id="segment-heading" className="m-0 eyebrow">{selected ? `Segment ${selected.letter}` : 'Segment'}</h2>
            {selected ? (
              <div className="mt-2 text-small text-text">
                <p className="m-0 font-mono text-mono num">{timecode(selected.start)} – {timecode(selected.end)} · {(selected.end - selected.start).toFixed(2)} s</p>
                <p className="m-0 mt-1 font-mono text-micro text-text-muted num">frames {selected.startFrame}–{selected.endFrame}</p>
                <p className="m-0 mt-2">{[selected.trackName, selected.laneName.toLowerCase()].filter(Boolean).join(' · ')}</p>
                <p className="m-0 mt-1">{model.inspectionReason ? <span className="text-text-muted">Values shown for inspection only.</span> : <>peak <span className="font-mono num">{selected.peak.toFixed(2)}</span> <span className="text-text-muted">· smoothed lane probability, uncalibrated</span></>}</p>
                {overlapping.length > 0 && (
                  <p className="m-0 mt-2 text-text-muted">Overlaps {overlapping.map((s) => <button key={s.key} type="button" onClick={() => selectSegment(s)} className="font-mono text-mono text-text underline underline-offset-[3px] mr-2">{s.letter} {s.laneName.toLowerCase()}{!model.inspectionReason && ` ${s.peak.toFixed(2)}`}</button>)}</p>
                )}
                {findingsFor(selected, findings).length > 0 && (
                  <p className="m-0 mt-2">Findings {findingsFor(selected, findings).map((f) => <Link key={f.id} to={`/analyses/${analysis.id}/findings`} className="font-mono text-mono text-accent underline underline-offset-[3px] mr-2">{f.id}</Link>)}</p>
                )}
              </div>
            ) : (
              <p className="mt-2 mb-0 text-small text-text-muted">Select a segment below, press ] for the next {model.inspectionReason ? 'detector segment' : 'flagged one'}, or shift-click a lane.</p>
            )}
          </section>

          <section aria-labelledby="segments-heading" data-tour="segments">
            <h2 id="segments-heading" className="m-0 eyebrow">Segments</h2>
            <div className="mt-2"><SegmentList inspectionOnly={!!model.inspectionReason} analysisId={analysis.id} segments={model.segments} selected={selected?.key ?? null} findings={findings} onSelect={(s) => selectSegment(s)} /></div>
            {uncountedNote(model.uncounted, model.lanes) && <p data-testid="segments-uncounted" className="mt-2 mb-0 text-small text-text-muted">{uncountedNote(model.uncounted, model.lanes)}</p>}
          </section>

          <section aria-labelledby="tracks-heading">
            <h2 id="tracks-heading" className="m-0 eyebrow">Tracks</h2>
            <div className="mt-2"><TrackGallery inspectionOnly={!!model.inspectionReason} tracks={model.tracks} stills={model.stills} stillUrl={still} mediaWidth={frames.width} isolated={isolated} onIsolate={(tid) => update({ track: tid })} /></div>
          </section>

          {syncLanes.map((lane) => (
            <section key={lane.key} aria-labelledby={`sync-${lane.key}`}>
              <h2 id={`sync-${lane.key}`} className="m-0 eyebrow">Sync offset · {lane.detail.replace(' · offset ms', '')}</h2>
              <div className="mt-2"><SyncOffsetPlot lane={lane} time={time} duration={model.duration} onSeek={(t) => player.seekFrame(frameAt(index, t))} /></div>
            </section>
          ))}

          <section aria-labelledby="keyframes-heading">
            <h2 id="keyframes-heading" className="m-0 eyebrow">Key frames</h2>
            <ul className="mt-2 m-0 p-0 list-none grid grid-cols-2 gap-2">
              {keyFrames.map(({ still: s, segment }) => (
                <li key={s.frame}>
                  <button type="button" data-testid="key-frame" data-frame={s.frame} onClick={() => player.seekFrame(s.frame)} className="block w-full text-left">
                    <span className="viewer-stage relative block h-[90px] rounded-xs bg-stage overflow-hidden">
                      {tokenReady && <img src={still(s) ?? undefined} alt={`Analysed frame ${s.frame}`} loading="lazy" className="w-full h-full object-contain" />}
                      {segment && <span className="absolute left-1 top-1 px-1 rounded-xs bg-stage-material font-mono text-micro text-stage-text" title={`Segment ${segment.letter}`}>{segment.letter}</span>}
                    </span>
                    <span className="block mt-1 font-mono text-micro text-text-muted num whitespace-nowrap">{shortTime(index.times[s.frame] ?? 0)} · frame {s.frame}</span>
                  </button>
                </li>
              ))}
            </ul>
          </section>

          <section aria-labelledby="notes-heading" data-testid="notes-panel">
            <div className="flex items-center justify-between">
              <h2 id="notes-heading" className="m-0 eyebrow">Notes ({model.notes.length})</h2>
              <button type="button" className={tool()} disabled={locked} onClick={() => (selected ? openNote(selected.start, selected.end) : openNote(time, Math.min(model.duration, time + NOTE_DEFAULT_S)))}>+ Add</button>
            </div>
            {model.notes.length ? (
              <ul className="mt-2 m-0 p-0 list-none">
                {model.notes.map((n) => (
                  <li key={n.id} data-testid="interval-note" className="border-b border-border last:border-b-0 py-1.5">
                    <button type="button" onClick={() => player.seekFrame(frameAt(index, n.start))} className="w-full text-left">
                      <span className="flex items-baseline gap-2 text-small"><span className="font-mono text-mono text-annotation">{n.label}</span><span className="font-mono text-mono num text-text">{shortTime(n.start)}–{shortTime(n.end)}</span><span className="ml-auto text-micro text-text-muted">{n.visibility === 'in_report' ? 'in report' : 'internal'}</span></span>
                      <span className="block text-small text-text">{n.body}</span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : <p className="mt-2 mb-0 text-small text-text-muted">No notes on the timeline yet.</p>}
          </section>
        </aside>
      </div>

      <IntervalDialog open={dialog !== null} startFrame={dialog?.start ?? 0} endFrame={dialog?.end ?? 0} index={index} pending={create.isPending}
        error={create.error ? create.error.message : null} onCancel={() => setDialog(null)} onSave={(d) => create.mutate(d)} />
    </div>
  )
}
